# Astral: High-Performance LLM Inference for Game Engines

Astral is a C++20 high-performance inference layer on top of LLaMA-class backends, designed specifically for real-time game engines (Unity, Unreal Engine 5). Built with zero-allocation hot paths, lock-free concurrency, and explicit memory control.

## Key Features

- **Zero-Allocation Hot Paths**: No dynamic allocations during token streaming, decoding, or sampling
- **CAS-Free Concurrency Primitives**: Bounded MPMC queue (ticket + per-slot sequence) and SPSC token rings; ARM-friendly WFE/SEV waiting
- **C ABI Surface**: Clean separation between C ABI and C++ implementation; pre-v0.1 so ABI can still evolve
- **Engine-Native Integration**: Direct use of Unity `NativeArray` and Unreal `FMemory` allocators
- **Cross-Platform**: armv7, armv8, arm64, x86-64 on Linux, Windows, macOS, Android, iOS
- **Embeddings**: End-to-end embeddings API (`astral_embed_*`) for embeddings-only models

## Project Structure

```
astral/
├── docs/                       # Documentation
│   ├── MASTER_SPEC.md          # Overall architecture and specifications
│   ├── architecture/           # Detailed architecture documents
│   │   ├── MEMORY_ARCHITECTURE.md
│   │   └── CONCURRENCY_MODEL.md
│   ├── integration/            # Engine integration guides
│   │   ├── UNITY_INTEGRATION.md
│   │   └── UNREAL_INTEGRATION.md
│   ├── rules/                  # Coding standards and rules
│   │   └── CODING_STANDARDS.md
│   └── workstreams/            # Developer sub-agent prompts
│       ├── CORE_RUNTIME_AGENT.md
│       ├── MEMORY_SPECIALIST_AGENT.md
│       └── CONCURRENCY_SPECIALIST_AGENT.md
├── include/                    # Public headers
│   ├── astral_rt.h             # C ABI (stable)
│   └── astral_rt.hpp           # C++ wrapper (optional)
├── src/                        # Core implementation
│   ├── core/                   # Runtime init, shutdown, handles
│   ├── memory/                 # Allocators, arenas, pools
│   ├── concurrency/            # Lock-free MPMC/SPSC, epoch reclaim
│   ├── inference/              # LLaMA backend integration
│   ├── platform/               # OS-specific code (Linux/Windows/macOS)
│   └── utils/                  # UTF-8, logging, intrinsics
├── plugins/                    # Engine plugins
│   ├── unity/                  # Unity C# P/Invoke + NativeArray adapters
│   └── unreal/                 # Unreal UE5 module + FMemory bridge
├── tests/                      # Unit + integration tests
├── benchmarks/                 # Performance benchmarks
└── cmake/                      # CMake modules and presets
```

## Quick Start

### Prerequisites

- C++20 compiler (GCC 11+, Clang 13+, MSVC 2022+)
- CMake 3.20+
- (Optional) Unity 2021.3+ or Unreal Engine 5.1+

### Building

```bash
# Clone repository
git clone https://github.com/your-org/astral.git
cd astral

# Configure
cmake --preset=release

# Build
cmake --build build/release

# Run tests
cd build/release && ctest
```

### CMake Presets

- `dev`: Debug build with tests and benchmarks
- `release`: Release build, optimized
- `release-with-tests`: Release build with tests and benchmarks
- `unity-plugin`: Build Unity plugin (outputs to `plugins/unity/Runtime/Plugins/`)
- `unreal-plugin`: Build Unreal plugin (outputs to `plugins/unreal/Binaries/`)

### Benchmarks

```bash
cmake --preset=release-with-tests
cmake --build --preset release-with-tests -j

# Concurrency microbenches (rdtsc/mach time where available)
./build/release-test/benchmarks/astral_benchmarks

# End-to-end streamed inference (requires a GGUF under tests/models/ or ASTRAL_BENCH_MODEL)
./build/release-test/benchmarks/astral_benchmarks --only infer --infer-tokens 128
```

To keep output quiet (llama.cpp can be verbose), set `ASTRAL_LLAMA_LOG=none` (or `error|warn|info|debug`).

## Usage Example (C API)

```c
#include "astral_rt.h"

int main() {
  // Initialize runtime
  AstralInit cfg = {
    .sys_alloc = {NULL, NULL, NULL}, // Use default allocator
    .log_cb = NULL,
    .log_user = NULL,
    .reserve_bytes = 2ULL << 30, // 2 GB
    .thread_count = 0, // Auto-detect
    .numa_node = 0xFFFFFFFF,
    .enable_hugepages = 0
  };
  astral_init(&cfg);

  // Optional: load a backend provider plugin at startup.
  // (The plugin registers a provider name used by AstralModelDesc.backend_name.)
  //
  // const char* plugin_path = "/abs/path/to/libastral_backend_provider.so";
  // AstralSpanU8 plugin_span = {(const uint8_t*)plugin_path, (uint32_t)strlen(plugin_path)};
  // astral_backend_load_plugin(plugin_span);

  // Load model
  const char* model_path = "models/llama-7b-q4.gguf";
  AstralModelDesc model_desc = {
    .model_path = {(const uint8_t*)model_path, strlen(model_path)},
    .backend_name = {NULL, 0}, // Optional override (e.g., "cpu", "mock")
    .gpu_layers = 0,
    .n_ctx = 2048,
    .n_batch = 512,
    .n_threads = 0,
    .embeddings_only = 0
  };
  AstralHandle model;
  AstralErr err = astral_model_load(&model_desc, &model);
  if (err != ASTRAL_OK) {
    fprintf(stderr, "model_load failed: %s (%s)\n", astral_error_string(err), astral_last_error());
    return 1;
  }

  // Create session
  AstralSessionDesc sess_desc = {
    .model = model,
    .max_tokens = 512,
    .temperature = 0.7f,
    .top_k = 40,
    .top_p = 0.95f,
    .stream_enabled = 1,
    .seed = 0 // 0 = auto
  };
  AstralHandle session;
  err = astral_session_create(&sess_desc, &session);
  if (err != ASTRAL_OK) {
    fprintf(stderr, "session_create failed: %s (%s)\n", astral_error_string(err), astral_last_error());
    astral_model_release(model);
    return 1;
  }

  // Feed prompt
  const char* prompt = "Once upon a time";
  AstralSpanU8 prompt_span = {(const uint8_t*)prompt, strlen(prompt)};
  astral_session_feed(session, prompt_span, 1);

  // Start decode
  astral_session_decode(session);

  // Read tokens
  uint8_t buffer[4096];
  AstralMutSpanU8 out_buf = {buffer, sizeof(buffer)};
  for (;;) {
    int bytes_read = astral_stream_read(session, out_buf, 100 /*ms*/);
    if (bytes_read == ASTRAL_E_TIMEOUT) continue;
    if (bytes_read < 0) break;
    if (bytes_read == 0) break; // end-of-stream
    fwrite(buffer, 1, bytes_read, stdout);
  }

  // Wait + stats
  astral_session_wait(session, 60000);
  AstralStats stats = {0};
  astral_session_stats(session, &stats);

  // Optional: reuse the same session with a new seed / sampling params.
  // (Only valid when the session is idle.)
  sess_desc.seed = 1234;
  astral_session_reset(session, &sess_desc);

  // Cleanup
  astral_session_destroy(session);
  astral_model_release(model);
  astral_shutdown();
}
```

## Unity Integration

See [Unity Integration Guide](docs/integration/UNITY_INTEGRATION.md) for details.

```csharp
using Astral;

// Initialize
var initCfg = new AstralInit {
    sys_alloc = AstralAllocatorAdapter.CreatePersistent(),
    reserve_bytes = 2UL << 30
};
AstralNative.astral_init(ref initCfg);

// Create session
var session = new AstralSession(modelHandle, maxTokens: 512);
session.Feed("Tell me a story", finalize: true);
session.Decode();

// Poll for tokens in Update()
void Update() {
    string token = session.StreamReadString(timeoutMs: 10);
    if (!string.IsNullOrEmpty(token)) {
        Debug.Log(token);
    }
}
```

## Unreal Integration

See [Unreal Integration Guide](docs/integration/UNREAL_INTEGRATION.md) for details.

```cpp
#include "AstralSession.h"

// Create session
UAstralSession* Session = NewObject<UAstralSession>(this);
FAstralSessionDesc Desc;
Desc.MaxTokens = 512;
Session->Create(ModelHandle, Desc);

// Feed prompt
Session->FeedPrompt(TEXT("Tell me a story"), true);
Session->Decode();

// Bind token event
Session->OnTokenReceived.AddDynamic(this, &AMyActor::OnToken);

void AMyActor::OnToken(const FString& Token) {
    UE_LOG(LogTemp, Log, TEXT("%s"), *Token);
}
```

## Architecture Highlights

### Memory Management

- **Virtual Reserve**: Reserve 2GB address space, commit pages on demand
- **Frame Allocators**: Bump allocators per session; reset after each decode frame
- **Object Pools**: Lock-free pools for tokens, callbacks (16-64 byte objects)
- **Zero Hot Path Allocations**: All inference operations use pre-allocated memory

See [Memory Architecture](docs/architecture/MEMORY_ARCHITECTURE.md) for details.

### Concurrency

- **MPMC Queue**: Lock-free work queue for task dispatch (bounded, ticket-based)
- **SPSC Ring**: Token streaming per session (producer: decode thread, consumer: app thread)
- **Epoch-Based Reclamation**: Safe memory reclamation without hazard pointers
- **Explicit Memory Ordering**: All atomics use `acquire`, `release`, `acq_rel`, or `relaxed`

See [Concurrency Model](docs/architecture/CONCURRENCY_MODEL.md) for details.

### C ABI Design

- **Stable Interface**: No breaking changes within major version
- **POD Structs**: All config/output via plain-old-data structs
- **UTF-8 Spans**: Strings as `{const uint8_t* data, uint32_t len}`
- **Error Codes**: All functions return `AstralErr`; out params for results
- **No Exceptions**: Zero exceptions across C ABI boundary

See [Master Specification](docs/MASTER_SPEC.md) for details.

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Init Latency | <100ms | Reserve 2GB, spawn 8 threads |
| Token Latency | <50ms | First token (7B model, Q4_K_M, CPU) |
| Throughput | >10K tok/s | Embeddings, batch=512, Ryzen 7 |
| MPMC Enqueue | <100ns | Per operation, x86-64, low contention |
| SPSC Push | <50ns | Per token, single producer/consumer |
| Memory Overhead | <530MB | Per session (8K ctx, 32 layers) |

## Developer Workstreams

Specialized sub-agent prompts for focused development:

- [Core Runtime Agent](docs/workstreams/CORE_RUNTIME_AGENT.md): Init, shutdown, handle management, C ABI
- [Memory Specialist Agent](docs/workstreams/MEMORY_SPECIALIST_AGENT.md): Allocators, arenas, pools, VM
- [Concurrency Specialist Agent](docs/workstreams/CONCURRENCY_SPECIALIST_AGENT.md): MPMC, SPSC, epoch reclaim
- More workstreams coming: Inference, Platform, Unity, Unreal specialists

## Coding Standards

- **C++20**: No modules, no RTTI across ABI, no STL containers
- **Explicit Memory Ordering**: Always specify `memory_order_*`
- **Zero Allocations**: Profile and validate with Valgrind/ASAN
- **Cache-Line Alignment**: 64 bytes for hot atomics
- **Platform Support**: Test on x86-64 and ARM (strong/weak memory models)

See [Coding Standards](docs/rules/CODING_STANDARDS.md) for full details.

## Testing

```bash
# Debug tests
ctest --preset dev -j8

# Release validation (tests + benchmarks)
ctest --preset release-with-tests -j8
./build/release-test/benchmarks/astral_benchmarks

# Memory validation
valgrind --leak-check=full ./tests/test_core
clang++ -fsanitize=address ./tests/test_memory.cpp

# Thread safety
clang++ -fsanitize=thread ./tests/test_concurrency.cpp
```

## Roadmap

### v0.1 (Current)

- [x] Core C ABI design
- [x] Memory subsystem (virtual reserve, frame allocators, pools)
- [x] Concurrency primitives (MPMC, SPSC, epoch reclaim)
- [x] CPU backend integration (llama.cpp)
- [x] Unity plugin (C# P/Invoke)
- [x] Unreal plugin (UE5 module)

### v0.2

- [ ] Embeddings fast path (SIMD, batching)
- [ ] Streaming backpressure tuning
- [ ] NUMA-aware allocation
- [ ] Mobile optimizations (ARM big.LITTLE)

### v1.0

- [ ] Production-ready (1M+ tokens, zero leaks)
- [ ] Full test coverage (unit, integration, stress)
- [ ] Benchmark suite (vs. llama.cpp, LLMUnity)
- [ ] Documentation (API reference, examples, tutorials)

## Contributing

Contributions welcome! Please read [CODING_STANDARDS.md](docs/rules/CODING_STANDARDS.md) before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Follow coding standards (C++20, no STL containers, explicit memory order)
4. Add tests (unit + integration)
5. Run benchmarks (no regressions)
6. Submit PR with detailed description

## License

[License TBD - to be determined by project lead]

## References

- LLMUnity: https://github.com/undreamai/LLMUnity
- LlamaCppUnity: https://github.com/DefiosLab/LlamaCppUnity
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Dmitry Vyukov's MPMC: https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue

## Support

- GitHub Issues: https://github.com/your-org/astral/issues
- Discord: [Link TBD]
- Email: [Email TBD]

---

Built with performance in mind. Every cycle counts.
