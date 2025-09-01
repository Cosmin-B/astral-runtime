# Backend Architecture - Pluggable Provider Design

## Overview

Astral uses a **pluggable backend provider** architecture that allows multiple inference backends (CPU, CUDA, Metal, DirectML) to coexist without breaking the stable C ABI. The backend is selected at model load time and remains opaque to the caller.

## Design Decision

After evaluating three options:

- **Option A**: Pure CPU First (simplest, most portable)
- **Option B**: Pluggable Backend Provider (our choice)
- **Option C**: Engine-Owned Memory Only (orthogonal; implemented alongside Option B)

We chose **Option B + C**:
- **Pluggable backends** for flexibility without ABI changes
- **Engine-owned memory preferred** for maximum control and performance

## Architecture

### Backend Provider Interface

```cpp
// astral/src/backend/backend.hpp (internal, not exposed in C ABI)
namespace astral::backend {

// Zero-copy view of provider logits for sampling.
// Lifetime: valid until the next provider call that mutates session state.
struct BackendLogitsView {
    const float* logits;
    uint32_t vocab_size;
};

// Provider ops table (POD function pointers).
// Rationale:
// - provider-agnostic (llama today, others later)
// - no C++ vtable/ABI coupling (providers can live in separate libs)
// - predictable, minimal dispatch cost (single indirect call)
struct BackendOps {
    // Model lifetime
    void* (*model_load)(const AstralModelDesc* desc, AstralErr* out_err);
    void (*model_unload)(void* model_ctx);

    // Text processing
    AstralErr (*tokenize)(void* model_ctx, AstralSpanU8 text,
                          int32_t* out_tokens, uint32_t max_tokens,
                          bool add_special, bool parse_special,
                          uint32_t* out_count);
    AstralErr (*detokenize)(void* model_ctx, const int32_t* tokens, uint32_t count,
                            AstralMutSpanU8 out_text, uint32_t* out_len);

    // Metadata
    AstralErr (*model_info)(void* model_ctx, uint32_t* out_vocab_size,
                            uint32_t* out_ctx_size);

    // Session primitives (KV cache owned by provider)
    void* (*session_create)(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err);
    void (*session_destroy)(void* session_ctx);
    AstralErr (*session_feed)(void* session_ctx, const int32_t* tokens, uint32_t count);

    // Core-owned sampling:
    // - provider exposes logits view (zero-copy)
    // - core chooses next token
    // - provider advances the KV cache for that token
    AstralErr (*session_logits)(void* session_ctx, BackendLogitsView* out_view);
    AstralErr (*session_accept)(void* session_ctx, int32_t token);
};

struct BackendProvider {
    const char* name;  // "cpu", "cuda", "metal", "directml"
    const BackendOps* ops;  // Required

    // Capabilities
    bool supports_gpu;
    uint32_t min_gpu_layers;  // 0 for pure CPU
};

// Registration + selection is handled by BackendRegistry (stores a fixed-size array).

} // namespace astral::backend
```

### Backend Selection Logic

```cpp
// astral/src/backend/backend.cpp
namespace astral::backend {

// select_backend(gpu_layers):
// - gpu_layers == 0: pick the first CPU provider
// - gpu_layers > 0: pick the first GPU provider that supports the requested layers
// - fallback to CPU if no GPU provider exists

} // namespace astral::backend
```

## CPU Backend Implementation

### Registration

```cpp
// astral/src/backend/cpu/cpu_backend.cpp
namespace astral::backend {

// Implement BackendOps functions for llama.cpp:
// - model_load/model_unload manage llama_model*
// - session_create/session_destroy manage llama_context*
// - session_feed does prompt prefill via llama_decode() on batched tokens
// - session_logits returns llama_get_logits_ith(ctx, -1) as a BackendLogitsView (zero-copy)
// - session_accept advances the KV cache by decoding the chosen token (llama_decode on a 1-token batch)

// Provider
static BackendProvider g_cpu_provider = {
    .name = "cpu",
    .ops = &kCpuBackendOps,
    .supports_gpu = false,
    .min_gpu_layers = 0
};

// Auto-register at static init time
static struct RegisterCpuBackend {
    RegisterCpuBackend() {
        BackendRegistry::instance().register_backend(&g_cpu_provider);
    }
} g_register_cpu_backend;

} // namespace astral::backend
```

## Future GPU Backend (CUDA Example)

```cpp
// astral/src/backend/cuda/cuda_backend.cpp (future implementation)
namespace astral::backend {

// Provider implements BackendOps; model/session contexts are opaque void* owned by the provider.
// The core still owns sampling (provider exposes logits view + accept/advance).

static void* cuda_model_load(const AstralModelDesc* desc, AstralErr* out_err);
static void  cuda_model_unload(void* model_ctx);
static AstralErr cuda_tokenize(void* model_ctx, AstralSpanU8 text, int32_t* out_tokens, uint32_t max_tokens,
                               bool add_special, bool parse_special, uint32_t* out_count);
static AstralErr cuda_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count,
                                 AstralMutSpanU8 out_text, uint32_t* out_len);
static AstralErr cuda_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size);
static void* cuda_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err);
static void  cuda_session_destroy(void* session_ctx);
static AstralErr cuda_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count);
static AstralErr cuda_session_logits(void* session_ctx, BackendLogitsView* out_view);
static AstralErr cuda_session_accept(void* session_ctx, int32_t token);

static const BackendOps kCudaBackendOps = {
    /*model_load=*/cuda_model_load,
    /*model_unload=*/cuda_model_unload,
    /*tokenize=*/cuda_tokenize,
    /*detokenize=*/cuda_detokenize,
    /*model_info=*/cuda_model_info,
    /*session_create=*/cuda_session_create,
    /*session_destroy=*/cuda_session_destroy,
    /*session_feed=*/cuda_session_feed,
    /*session_logits=*/cuda_session_logits,
    /*session_accept=*/cuda_session_accept,
};

static BackendProvider g_cuda_provider = {
    .name = "cuda",
    .ops = &kCudaBackendOps,
    .supports_gpu = true,
    .min_gpu_layers = 1
};

static struct RegisterCudaBackend {
    RegisterCudaBackend() {
        BackendRegistry::instance().register_backend(&g_cuda_provider);
    }
} g_register_cuda_backend;

} // namespace astral::backend
```

## Integration with C ABI

The C ABI remains unchanged. Backend selection happens internally:

```c
// User code (same for CPU or GPU)
AstralModelDesc desc = {
    .model_path = model_path_span,
    .gpu_layers = 32,  // Use GPU if available
    .n_ctx = 2048,
    // ...
};

AstralHandle model;
AstralErr err = astral_model_load(&desc, &model);
// Backend auto-selected: CUDA if available, else CPU
```

### Model Handle Implementation

```cpp
// astral/src/inference/model.cpp
struct Model {
    const backend::BackendProvider* backend;  // selected provider
    void* backend_model_ctx;                  // provider-owned model context
};

AstralErr astral_model_load(const AstralModelDesc* desc, AstralHandle* out_model) {
    // select provider (cpu vs future gpu providers)
    // call provider->ops->model_load() to get backend_model_ctx
}
```

## Memory Strategy: Engine-Owned Memory Preferred

### Allocator Passthrough

Provider implementations may optionally retain an allocator pointer from `astral_init()` and use it for all non-hot-path allocations (model load, session create). In v0.1 this is not fully plumbed through to llama.cpp yet; it is planned work.

### llama.cpp Integration

Pass custom allocator to llama.cpp:

```cpp
// llama.cpp supports custom allocators via callback
static void* llama_alloc_wrapper(void* user, size_t size) {
    AstralAllocator* alloc = static_cast<AstralAllocator*>(user);
    return alloc->alloc(alloc->user, size, 16 /*align*/);
}

static void llama_free_wrapper(void* user, void* ptr) {
    AstralAllocator* alloc = static_cast<AstralAllocator*>(user);
    alloc->free(alloc->user, ptr, 0, 0);
}

// Pseudocode: call from provider model_load().
llama_model_params params = llama_model_default_params();
if (alloc && alloc->alloc && alloc->free) {
    params.alloc_fn = llama_alloc_wrapper;
    params.free_fn = llama_free_wrapper;
    params.alloc_user = alloc;
}
llama_model* model = llama_model_load_from_file(path, params);
```

## Benefits of Pluggable Architecture

1. **Stable ABI**: Adding GPU backends doesn't change `astral_rt.h`
2. **Transparent Selection**: User specifies `gpu_layers` (or an optional backend override); we pick best backend
3. **Future-Proof**: New backends (Metal, DirectML, WebGPU) register at runtime
4. **Testability**: Mock backends for unit tests (deterministic output)
5. **Modularity**: Backend code isolated in `src/backend/{cpu,cuda,metal}/`

## Future Backends

| Backend | Platform | Priority | Notes |
|---------|----------|----------|-------|
| CPU (llama.cpp) | All | **v0.1** | Current implementation |
| CUDA | Linux, Windows | v0.2 | NVIDIA GPUs |
| Metal | macOS, iOS | v0.2 | Apple Silicon |
| DirectML | Windows | v0.3 | AMD/Intel GPUs |
| WebGPU | Browser | v1.0 | WASM + WebGPU |
| Vulkan Compute | Android, Linux | v1.0 | Mobile GPUs |

## Backend Capabilities Query (Future)

```c
// Future API: query backend capabilities
typedef struct AstralBackendInfo {
    const char* name;
    uint8_t supports_gpu;
    uint8_t supports_embeddings;
    uint8_t supports_streaming;
    uint32_t max_ctx_size;
} AstralBackendInfo;

// Query available backends
uint32_t astral_backend_count(void);
const AstralBackendInfo* astral_backend_info(uint32_t index);

// Force specific backend
// v0.1: use AstralModelDesc.backend_name (optional override)
// AstralModelDesc desc = {0};
// desc.backend_name = {(const uint8_t*)"cpu", 3};
// astral_model_load(&desc, &model);
```

## Testing Strategy

### Unit Tests

```cpp
TEST(backend_cpu_init) {
    // Selecting the backend does not create provider contexts.
    // Model/session contexts are created via the ops table.
    const BackendProvider* provider = BackendRegistry::instance().select_backend(/*gpu_layers=*/0);
    ASSERT_TRUE(provider != nullptr);
    ASSERT_TRUE(provider->ops != nullptr);
}

TEST(backend_selection_priority) {
    // Future: register a mock GPU provider and ensure selection prefers it when gpu_layers > 0.
}
```

### Integration Tests

- Load model with `gpu_layers=0` → CPU backend selected
- Load model with `gpu_layers=32` → GPU backend selected (if available)
- Decode 1000 tokens → verify output identical across backends (deterministic seed)
- Embeddings fast path → verify vector output matches reference

## Performance Targets

| Backend | Decode Latency | Throughput | Notes |
|---------|----------------|------------|-------|
| CPU (llama.cpp) | <50ms first token | >20 tok/s | 7B Q4_K_M, Ryzen 7 |
| CUDA (future) | <20ms first token | >100 tok/s | 7B Q4_K_M, RTX 4090 |
| Metal (future) | <30ms first token | >80 tok/s | 7B Q4_K_M, M2 Max |

## Migration Path

1. **v0.1**: CPU backend only (llama.cpp)
2. **v0.2**: Add CUDA backend registration; no ABI change
3. **v0.2**: Add Metal backend registration; no ABI change
4. **v0.3**: Add DirectML backend; no ABI change
5. **v1.0**: Stabilize backend API; expose backend query functions

## References

- Backend interface: `astral/src/backend/backend.hpp`
- Backend registry/selection: `astral/src/backend/backend.cpp`
- CPU implementation: `astral/src/backend/cpu/cpu_backend.cpp`
- Memory architecture: [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md)
- Coding standards: [../rules/CODING_STANDARDS.md](../rules/CODING_STANDARDS.md)

---

**Design principle**: Keep the C ABI stable; add flexibility via internal plugin architecture.
