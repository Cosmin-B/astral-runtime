# Astral

Astral is a C++17 native inference runtime for applications and game engines.
It exposes a C ABI over llama.cpp-class backends, with bounded streaming,
continuous batching, embeddings, native retrieval, and Unity and Unreal
wrappers.

The ABI is pre-1.0 and may still change. Capability claims are tracked in the
[feature matrix](docs/FEATURE_MATRIX.md); release evidence requirements are
kept separate from source availability.

## Capabilities

- CPU inference through the built-in llama.cpp provider, plus mock, dynamic,
  and remote provider surfaces.
- Asynchronous sessions and continuous-batching conversations with streamed
  UTF-8 output, cancellation, reset, logprobs, grammar, prompt cache, and LoRA.
- Text and multimodal embedding APIs, with multimodal support opt-in at build
  time and dependent on compatible model assets.
- Exact flat and bounded graph vector search with compact storage lanes,
  batch search, persistence, mapped snapshots, and optional f32 reranking.
- Native chunking, toolsets, agents, model presets, and request-status APIs.
- Unity and Unreal packages that preserve bytes-first ownership at the native
  boundary.
- Fixed-capacity concurrency primitives and allocation gates for maintained
  runtime hot paths.

## Requirements

- CMake 3.20+
- GCC 11+, Clang 13+, or MSVC 2022+
- A C++17 toolchain

Unity 6000.0+ and Unreal Engine 5.4+ are optional. Unity 6000.0 and UE 5.7 are
the primary engine validation targets.

## Build And Test

Clone with submodules, then use the checked-in presets:

```bash
git clone --recurse-submodules https://github.com/Cosmin-B/astral-runtime.git
cd astral

cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests --output-on-failure
```

For a debug build:

```bash
cmake --preset dev
cmake --build --preset dev -j
ctest --preset dev --output-on-failure
```

Sanitizers are separate validation lanes:

```bash
./scripts/run_asan.sh
./scripts/run_tsan.sh
```

See [BUILD.md](BUILD.md) for presets, optional CUDA and profiling builds,
packaging, and platform notes.

## Native Quickstart

Build the maintained C example without a model download by using the mock
provider:

```bash
cmake -S . -B build/examples \
  -DASTRAL_BUILD_EXAMPLES=ON \
  -DASTRAL_BUILD_TESTS=OFF \
  -DASTRAL_BUILD_BENCHMARKS=OFF
cmake --build build/examples --target astral_c_quickstart -j
./build/examples/examples/astral_c_quickstart \
  --backend mock --prompt "Once upon a time"
```

The source is
[examples/astral_c_quickstart.c](https://github.com/Cosmin-B/astral-runtime/blob/main/examples/astral_c_quickstart.c).
It demonstrates initialization, model loading, streaming, error handling,
reset, and handle release through the public C ABI.

## Engine Integration

Unity:

- [Package guide](https://github.com/Cosmin-B/astral-runtime/blob/main/plugins/unity/README.md)
- [Integration reference](docs/integration/UNITY_INTEGRATION.md)
- Maintained samples under `plugins/unity/Samples~`

Unreal Engine:

- [Plugin guide](https://github.com/Cosmin-B/astral-runtime/blob/main/plugins/unreal/AstralRT/README.md)
- [Integration reference](docs/integration/UNREAL_INTEGRATION.md)
- [UE 5.7 quickstart](docs/integration/UNREAL_57_QUICKSTART.md)

The wrappers do not own model execution. They translate engine-facing data and
lifecycle operations into the same native handles and spans used by C callers.

## Runtime Design

- The public boundary is a C ABI with sized POD descriptors, UTF-8 spans,
  explicit handles, and error codes. Exceptions never cross that boundary.
- Sessions and conversations have one decode producer and one stream consumer.
  SPSC queues transfer tokens and metadata without a compare-and-swap loop.
- Continuous-batching slot snapshots use epoch reclamation once per scheduling
  pass rather than per-conversation reference-count traffic.
- Runtime-owned scratch and fixed-capacity structures keep maintained decode,
  sampling, and streaming paths inside their allocation contracts.
- Compact memory-index lanes use runtime-dispatched x86 and ARM kernels where
  available, with scalar paths retained for portability and correctness.

Read [docs/README.md](docs/README.md) for the maintained documentation map.

## Validation

The normal local release check is:

```bash
git diff --check
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests --output-on-failure
```

Additional scripts cover sanitizers, memory and syscall gates, CUDA parity,
model matrices, engine runners, packaging, and release metadata. Those lanes
are documented beside the subsystem they validate rather than summarized as
blanket platform support.

Real Unity Editor, Unreal Editor, CUDA, multimodal, mobile, Windows large-page,
and protected-signing runs remain release-evidence requirements. See
[ROADMAP.md](ROADMAP.md) and the
[release acceptance matrix](docs/release/RELEASE_ACCEPTANCE_MATRIX.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[coding standards](docs/rules/CODING_STANDARDS.md). Keep changes scoped, add
tests for behavior changes, and attach performance evidence for hot-path work.

## License

Astral is licensed under the Apache License 2.0. See
[LICENSE](https://github.com/Cosmin-B/astral-runtime/blob/main/LICENSE),
[NOTICE](https://github.com/Cosmin-B/astral-runtime/blob/main/NOTICE), and the
[third-party notices](docs/release/THIRD_PARTY_NOTICES.md).

## Support

Use [GitHub Issues](https://github.com/Cosmin-B/astral-runtime/issues) for reproducible
bugs and integration problems.
