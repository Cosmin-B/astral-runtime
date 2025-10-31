# Astral: Native LLM Runtime for Game Engines

Astral is a C++20 native inference layer on top of LLaMA-class backends, designed for real-time game engines (Unity, Unreal Engine 5). The runtime emphasizes allocation-gated hot paths, explicit concurrency primitives, and predictable memory ownership.

## Key Features

- **Allocation-Gated Hot Paths**: Release gates track steady-state heap behavior for token streaming, decoding, and sampling
- **CAS-Free Concurrency Primitives**: Bounded MPMC queue (ticket + per-slot sequence) and SPSC token rings; ARM-friendly WFE/SEV waiting
- **C ABI Surface**: Clean separation between C ABI and C++ implementation; v0.1 (ABI may still evolve until v1.0)
- **Engine Integration**: Unity `NativeArray` adapters and Unreal wrapper code for native runtime ownership
- **Cross-Platform**: Core supports Linux/Windows/macOS (x86_64/arm64) today; Android/iOS packaging is planned for v0.1.1
- **Embeddings**: End-to-end embeddings API (`astral_embed_*`) for embeddings-only models

## Project Structure

```
astral/
├── docs/                       # Documentation
│   ├── architecture/           # Detailed architecture documents
│   │   ├── BACKEND_ARCHITECTURE.md
│   │   ├── CONCURRENCY_MODEL.md
│   │   ├── MEMORY_ARCHITECTURE.md
│   │   └── WORK_STEALING_SCHEDULER.md
│   ├── api/                    # Public API behavior notes
│   ├── integration/            # Engine integration guides
│   │   ├── UNITY_INTEGRATION.md
│   │   ├── UNREAL_57_QUICKSTART.md
│   │   └── UNREAL_INTEGRATION.md
│   ├── release/                # Release evidence, ABI, and dependency manifests
│   ├── rules/                  # Coding standards and rules
│   ├── embedded/               # Embedded profile notes
│   └── PRODUCTION_READINESS_AUDIT.md
├── backend_plugins/            # Dynamic backend provider examples
├── include/                    # Public headers
│   ├── astral_rt.h             # Public C ABI
│   └── astral_backend.h        # Backend provider ABI
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
├── scripts/                    # Release, validation, and engine-runner tooling
└── cmake/                      # CMake modules and presets
```

## Quick Start

### Prerequisites

- C++20 compiler (GCC 11+, Clang 13+, MSVC 2022+)
- CMake 3.20+
- (Optional) Unity 2021.3+ or Unreal Engine 5.4+; UE 5.7 is the primary validation target

### Building

```bash
# Clone repository
git clone https://github.com/Cosmin-B/astral.git
cd astral

# Required submodules (llama.cpp; Tracy is optional unless you use *-prof presets)
git submodule update --init --recursive

# Configure + build (dev includes tests/benchmarks)
cmake --preset dev
cmake --build --preset dev -j

# Run tests
ctest --preset dev -j8
```

### CMake Presets

- `dev`: Debug build with tests and benchmarks
- `dev-prof`: Debug build with Tracy enabled (requires `external/tracy` submodule)
- `dev-cuda`: Debug build with CUDA backend enabled (requires CUDA toolkit)
- `dev-prof-cuda`: Debug build with Tracy + CUDA backend enabled
- `release`: Release build, optimized
- `release-with-tests`: Release build with tests and benchmarks
- `release-prof`: Release build with Tracy enabled (requires `external/tracy` submodule)
- `release-cuda`: Release build with CUDA backend enabled (requires CUDA toolkit)
- `release-prof-cuda`: Release build with Tracy + CUDA backend enabled
- `unity-plugin`: Build Unity plugin (outputs to `plugins/unity/Runtime/Plugins/`)
- `unity-plugin-prof`: Unity plugin build with Tracy enabled (requires `external/tracy` submodule)
- `unreal-plugin`: Stage the native C ABI header and static library into the Unreal plugin `ThirdParty` layout
- `unreal-plugin-prof`: Unreal plugin build with Tracy enabled (requires `external/tracy` submodule)
- `embedded-*`: Embedded/robotics profiles and cross-compilation presets (see `docs/EMBEDDED_PROFILE.md`)

Release presets build both a static and a shared runtime library on desktop platforms (`ASTRAL_BUILD_STATIC_LIB=ON` and
`ASTRAL_BUILD_SHARED_LIB=ON`).

### Benchmarks

```bash
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j

# Concurrency microbenches (rdtsc/mach time where available)
./build/release-test/benchmarks/astral_benchmarks

# End-to-end streamed inference (requires a GGUF under tests/models/ or ASTRAL_BENCH_MODEL)
./build/release-test/benchmarks/astral_benchmarks --only infer --infer-tokens 128

# Canonical surface benchmark (used for CI artifacts; embeddings/KV/grammar/logprobs)
./build/release-test/benchmarks/astral_benchmarks --only features
```

To keep output quiet (llama.cpp can be verbose), set `ASTRAL_LLAMA_LOG=none` (or `error|warn|info|debug`).

### CUDA (optional)

CUDA is disabled by default. To build with the CUDA backend provider enabled:

```bash
cmake --preset dev-cuda
cmake --build --preset dev-cuda -j
```

#### Models (tests/models)

Astral’s integration tests/benches use GGUF models on disk. Download small defaults via:

```bash
./tests/model_downloader.sh --preset gpt2-q2k
./tests/model_downloader.sh --preset embed-minilm-q2k
```

For embedded/no-filesystem deployments, use `astral_model_load2()` with `AstralModelDesc` and a `MEMORY` or custom `IO` source (single-file GGUF; `use_mmap` is forced OFF for these sources).

#### Threading knobs (benchmarks)

The benchmark runner has two independent thread controls:

- `ASTRAL_BENCH_RUNTIME_THREADS`: Astral runtime worker pool size (default: `1` for benchmarks).
- `ASTRAL_BENCH_MODEL_THREADS`: backend/model thread count (plumbed to `AstralModelDesc.n_threads`; default: `0` = auto).
- `ASTRAL_BENCH_THREADS`: compatibility alias for `ASTRAL_BENCH_MODEL_THREADS` (model threads only).

For single-core profiling runs (e.g. `taskset -c 0`), set both runtime and model threads to `1` to avoid oversubscription.

#### Inference bench mode

- `ASTRAL_BENCH_INFER_STREAM=1` (default): measure streamed decode (includes detokenize + stream overhead).
- `ASTRAL_BENCH_INFER_STREAM=0`: measure decode without streaming (skips detokenize + stream backpressure; useful for CPU kernel profiling).

#### Continuous batching bench mode (v0.2+)

Run continuous batching across multiple slots (conversations) using the same `--only infer` benchmark:

- `ASTRAL_BENCH_SLOTS=N` (default: `1`): when `N > 1`, the benchmark uses `astral_conv_*` + the model executor instead of `astral_session_*`.
- `ASTRAL_BENCH_MAX_BATCH_TOKENS=N` (default: `64`): per-tick token cap for the executor (keep `<= n_batch` for your provider/model).
- `ASTRAL_BENCH_PROMPT_REPEAT=N` (default: `1`): repeat the default prompt to stress prompt ingestion.
- `ASTRAL_BENCH_PROMPT_HEAVY_SLOTS=N` (default: `0`): first N slots use the repeated prompt (mixed workload).

#### Profiling (CPU)

Tracy captures (recommended for coarse “what/when” timelines) are documented in `docs/PROFILING_TRACY.md`.

Hotspot profile (call stacks):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1

taskset -c 0 env ASTRAL_LLAMA_LOG=none \
  ASTRAL_BENCH_RUNTIME_THREADS=1 \
  ASTRAL_BENCH_MODEL_THREADS=1 \
  perf record -F 999 -g -- \
  ./build/release-test/benchmarks/astral_benchmarks --only infer --infer-warmup 16 --infer-tokens 256

perf report
```

Embeddings hotspot profile:

```bash
sudo sysctl -w kernel.perf_event_paranoid=1

taskset -c 0 env ASTRAL_LLAMA_LOG=none \
  ASTRAL_BENCH_RUNTIME_THREADS=1 \
  ASTRAL_BENCH_MODEL_THREADS=1 \
  ASTRAL_BENCH_EMB_BACKEND=cpu \
  ASTRAL_BENCH_EMB_MODEL=tests/models/all-MiniLM-L6-v2-Q2_K.gguf \
  perf record -F 999 -g -- \
  ./build/release-test/benchmarks/astral_benchmarks --only embed --embed-iters 2000

perf report
```

Bound-ness approximation (IPC + miss rates):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1

taskset -c 0 env ASTRAL_LLAMA_LOG=none \
  ASTRAL_BENCH_RUNTIME_THREADS=1 \
  ASTRAL_BENCH_MODEL_THREADS=1 \
  perf stat -r 1 \
    -e cycles,instructions,branches,branch-misses,cache-references,cache-misses \
  -- ./build/release-test/benchmarks/astral_benchmarks --only infer --infer-warmup 16 --infer-tokens 256
```

Notes:
- Start with `perf stat -r 1` (it repeats the full command `N` times).
- `perf stat --topdown` is only available when your kernel/PMU exposes Topdown/TMA metric groups (common on Intel; often missing on AMD).
- Check availability with `perf list metricgroup | rg -i 'topdown|tma'`.

## Usage Example (C API)

The maintained native quickstart is [examples/astral_c_quickstart.c](examples/astral_c_quickstart.c).
It initializes the runtime, sets required ABI struct fields, loads a model,
streams output, reports errors, resets the session, and releases native handles.

```bash
cmake -S . -B build/examples -DASTRAL_BUILD_EXAMPLES=ON -DASTRAL_BUILD_TESTS=OFF -DASTRAL_BUILD_BENCHMARKS=OFF
cmake --build build/examples --target astral_c_quickstart -j
./build/examples/examples/astral_c_quickstart --backend mock --prompt "Once upon a time"
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
#include "AstralLog.h"

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
    UE_LOG(LogAstralRT, Log, TEXT("%s"), *Token);
}
```

## Architecture Highlights

### Memory Management

- **Virtual Reserve**: Reserve 2GB address space, commit pages on demand
- **Frame Allocators**: Bump allocators per session; reset after each decode frame
- **Object Pools**: Lock-free pools for tokens, callbacks (16-64 byte objects)
- **Allocation-Gated Hot Paths**: Maintained tests track steady-state heap calls in decode, sampling, and streaming paths

See [Memory Architecture](docs/architecture/MEMORY_ARCHITECTURE.md) for details.

### Concurrency

- **MPMC Queue**: Lock-free work queue for task dispatch (bounded, ticket-based)
- **SPSC Ring**: Token streaming per session (producer: decode thread, consumer: app thread)
- **Epoch-Based Reclamation**: Safe memory reclamation without hazard pointers
- **Explicit Memory Ordering**: All atomics use `acquire`, `release`, `acq_rel`, or `relaxed`

See [Concurrency Model](docs/architecture/CONCURRENCY_MODEL.md) for details.

### C ABI Design

- **Versioned Surface**: v0.1 ABI changes require release notes and migration guidance
- **POD Structs**: All config/output via plain-old-data structs
- **UTF-8 Spans**: Strings as `{const uint8_t* data, uint32_t len}`
- **Error Codes**: All functions return `AstralErr`; out params for results
- **No Exceptions Across ABI**: C ABI functions return error codes; `ASTRAL_NO_THROW_ABI=ON` catches/translates unexpected C++ exceptions at the ABI boundary

See [ABI Versioning](docs/ABI_VERSIONING.md),
[Error Handling](docs/api/ERROR_HANDLING.md), and the public header in
[include/astral_rt.h](include/astral_rt.h) for the current contract.

## Performance Targets

These are engineering targets, not release guarantees. The current release
evidence and blockers are tracked in `docs/PRODUCTION_READINESS_AUDIT.md`.

| Metric | Target | Notes |
|--------|--------|-------|
| Init Latency | <100ms | Reserve 2GB, spawn 8 threads |
| Token Latency | <50ms | First token (7B model, Q4_K_M, CPU) |
| Throughput | >10K tok/s | Embeddings, batch=512, Ryzen 7 |
| MPMC Enqueue | <100ns | Per operation, x86-64, low contention |
| SPSC Push | <50ns | Per token, single producer/consumer |
| Memory Overhead | <530MB | Per session (8K ctx, 32 layers) |

## Coding Standards

- **C++20**: No modules, no RTTI across ABI, no STL containers
- **Explicit Memory Ordering**: Always specify `memory_order_*`
- **Allocation Gates**: Profile and validate tracked heap behavior with the allocation, ASAN, and Valgrind gates
- **Cache-Line Alignment**: 64 bytes for hot atomics
- **Platform Support**: Test on x86-64 and ARM (strong/weak memory models)

See [Coding Standards](docs/rules/CODING_STANDARDS.md) for full details.

## Testing

```bash
# Fast native presubmit
./scripts/run_fast_presubmit.sh

# Debug tests
cmake --preset dev
cmake --build --preset dev -j
ctest --preset dev -j8

# Release validation
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests -j8

# Benchmark smoke
./build/release-test/benchmarks/astral_benchmarks

# Memory and sanitizer helpers
./scripts/run_asan.sh
./scripts/run_tsan.sh
./scripts/run_valgrind.sh

# Comment and documentation inventory
python3 ./scripts/inventory_comments.py --format summary --fail-orphan-markers
```

## Roadmap

### Current Local Gates

- Core C ABI, mock provider, CPU provider, memory, concurrency, embeddings, and media mock tests pass in `release-with-tests`.
- Release metadata, SBOM, dependency pins, release notes, source scans, ABI layout, shared exports, Unreal header mirror, and package artifact checks are gated locally.
- Unreal and Unity runner scripts fail hard on missing native/runtime evidence instead of silently skipping release lanes.

### Required Before Production Release

- Run UE 5.7 container Automation and the UE 5.4/5.5/5.6/5.7 compatibility matrix with real Epic editor access.
- Run real Unity EditMode tests with the packaged native runtime.
- Promote CUDA, multimodal, HF GGUF matrix, Windows large-page, mobile, and protected signing lanes from local tooling to release-candidate evidence.
- Publish release notes with checksums, signatures, SBOM, dependency manifest, ABI layout, and rollback instructions.

## Contributing

Read [CODING_STANDARDS.md](docs/rules/CODING_STANDARDS.md) and keep changes tied to a issue tracker issue before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Follow coding standards (C++20, no STL containers, explicit memory order)
4. Add tests (unit + integration)
5. Run benchmarks (no regressions)
6. Submit PR with detailed description

## License

This source tree is not licensed for redistribution except under a separate
written agreement with the copyright holders. See [LICENSE](LICENSE), [NOTICE](NOTICE),
and [third-party notices](docs/release/THIRD_PARTY_NOTICES.md).

## References

- LLMUnity: https://github.com/undreamai/LLMUnity
- LlamaCppUnity: https://github.com/DefiosLab/LlamaCppUnity
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Tracy profiler: https://github.com/wolfpld/tracy
- Dmitry Vyukov's MPMC: https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue

## Support

- GitHub Issues: https://github.com/Cosmin-B/astral/issues

---
Keep support claims tied to the validation gates above and the production readiness audit.
