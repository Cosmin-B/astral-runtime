# Astral

Astral is a C++17 inference control plane for native applications and game
engines. Applications call one C ABI for models, requests, retrieval, and
agents. Astral owns handles, bounded queues, cancellation, streaming, and
scheduling around those calls. Providers own model execution.

That split matters when inference shares a process with a frame loop. A Unity,
Unreal, or C caller can drain a bounded byte stream on its own thread while a
model executor batches active conversations. Switching from a local GGUF model
to the remote provider does not require a second request lifecycle in engine
code.

## What is in the runtime

- A C ABI built from sized POD descriptors, byte spans, explicit handles, and
  error codes. C++ exceptions do not cross it.
- Built-in CPU, CUDA-offload, and remote providers, plus a C provider
  table for dynamic plugins.
- Asynchronous sessions and model-scoped continuous batching with bounded
  streaming, cancellation, grammar, prompt caches, logprobs, and LoRA.
- Text and multimodal embedding surfaces, native chunking, tool-call parsing,
  and fixed-capacity memory search.
- Native agents that assemble system prompts, summaries, retrieved context,
  history, and the current turn before entering the same conversation executor.
- Unity and Unreal wrappers that keep native bytes and handles intact until the
  engine chooses to marshal them.

The [runtime architecture](docs/architecture/RUNTIME_ARCHITECTURE.md) follows
one request across those pieces. The [feature matrix](docs/FEATURE_MATRIX.md)
is the shorter answer for a specific build or platform.

## Ownership across one request

| Boundary | What crosses it | Who owns the work |
| --- | --- | --- |
| Application to C ABI | Sized descriptors, spans, and handles | The caller owns input and output buffers |
| C ABI to Astral core | Validated native state | Astral owns lifetimes, scheduling, cancellation, and streams |
| Astral to provider | Provider operation table and contexts | The provider owns model evaluation |
| Astral back to application | Bounded byte, metadata, embedding, and search output | The caller decides when and where to consume it |

A normal asynchronous session uses a runtime worker as its single decode
producer. A continuous-batching model uses one dedicated executor thread for
all active slots on that model. In both cases, one consumer drains each stream.
The stream is a bounded SPSC ring, so a slow consumer creates backpressure
instead of unbounded memory growth.

Engine wrappers preserve this ownership. Native code writes into caller-owned
buffers. Unity can poll into a `NativeArray<byte>`, and Unreal can move delivery
onto the game thread without asking the decode loop to invoke an engine
callback.

## Retrieval and agents

The memory index stores fixed-dimension vectors and stable record keys in
native memory. Flat search scans every matching row and is exact for the scores
in the selected storage format. The bounded graph index trades that exhaustive
scan for a smaller candidate set. Group filters use the flat scanner because a
sparse filter breaks the locality that the graph depends on.

Storage is independent of traversal. Current lanes include f32, q8, scaled
Float6 formats, E5M2, and compact-plus-f32 rerank variants. Astral selects the
metric and storage kernel before the scoring loop, with x86, ARM, and scalar
implementations behind the same API.

The [retrieval architecture](docs/architecture/RETRIEVAL_ARCHITECTURE.md)
explains the flat and graph paths, snapshot ownership, and E5M2 widening.
Agents build on that retrieval layer: the application may search directly, or
copy selected document chunks into an agent's bounded memory context.

## Build and run the C quickstart

Astral requires CMake 3.20 or later, GCC 11+, Clang 13+, or MSVC 2022+ with
C++17 support. Clone the public repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/Cosmin-B/astral-runtime.git
cd astral-runtime

cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests --output-on-failure
```

Before running the C example, provide a readable GGUF model:

```bash
cmake -S . -B build/examples \
  -DASTRAL_BUILD_EXAMPLES=ON \
  -DASTRAL_BUILD_TESTS=OFF \
  -DASTRAL_BUILD_BENCHMARKS=OFF
cmake --build build/examples --target astral_c_quickstart -j
./build/examples/examples/astral_c_quickstart \
  --backend cpu --model /absolute/path/to/model.gguf \
  --prompt "Once upon a time"
```

The [example source](https://github.com/Cosmin-B/astral-runtime/blob/main/examples/astral_c_quickstart.c)
shows initialization, model loading, streaming, reset, error handling, and
handle release through the public ABI. [BUILD.md](BUILD.md) covers the other
presets, CUDA, shared libraries, packaging, and sanitizers.

## Platforms and integrations

| Surface | Current public code |
| --- | --- |
| Desktop core | Linux, macOS, and Windows platform implementations |
| CPU inference | Built-in llama.cpp provider |
| GPU inference | Optional GGML CUDA offload through the same provider contract |
| Embedded | x86-64, ARM64, and ARMv7 presets with arena-backed core memory |
| Unity | Unity 6000.0+ package, native buffers, jobs, samples, and editor tests |
| Unreal Engine | UE 5.4+ plugin surface with a UE 5.7 sample and automation project |

Embedded presets disable threads, dynamic loading, and other desktop services.
Continuous batching and native agents are therefore desktop features in those
profiles. The [embedded guide](docs/EMBEDDED_PROFILE.md),
[Unity guide](plugins/unity/README.md), and
[Unreal guide](plugins/unreal/AstralRT/README.md) carry the build and packaging
rules.

## Public API and docs

- [C runtime header](https://github.com/Cosmin-B/astral-runtime/blob/main/include/astral_rt.h)
- [Provider ABI](https://github.com/Cosmin-B/astral-runtime/blob/main/include/astral_backend.h)
- [Documentation map](docs/README.md)
- [Continuous batching](docs/api/CONTINUOUS_BATCHING.md)
- [Memory index](docs/api/MEMORY_INDEX.md)
- [Agent runtime](docs/api/AGENT_RUNTIME.md)
- [ABI versioning](docs/ABI_VERSIONING.md)

The ABI is still pre-1.0. Descriptor sizes and ABI version checks make changes
detectable, but they do not promise that every pre-1.0 descriptor will remain
unchanged.

## License and support

Astral is licensed under the Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE),
and the [third-party notices](docs/release/THIRD_PARTY_NOTICES.md).

Use [GitHub Issues](https://github.com/Cosmin-B/astral-runtime/issues) for
reproducible bugs and integration problems.
