# Astral Roadmap

Astral's next platform work is WebAssembly and WebGPU. The goal is to preserve
the same C ABI and explicit ownership model while adding a browser deployment
profile and a portable GPU execution path.

## WebAssembly

The first step is a WebAssembly CPU build that can run the mock provider,
tokenization, embeddings, and compact memory search without dynamic loading or
native virtual-memory assumptions.

The maintained profile will define:

- Emscripten build and test presets.
- Fixed and growable linear-memory configurations.
- Single-threaded operation plus an opt-in worker-thread configuration where
  the browser provides the required isolation features.
- A small JavaScript boundary that keeps model bytes and generated output under
  explicit ownership.
- Browser tests for initialization, streaming, cancellation, reset, embedding,
  and memory-index persistence.

## WebGPU

WebGPU will be added as a backend rather than as a separate public API. Device,
queue, buffer, and pipeline ownership will remain behind Astral's provider
boundary.

Initial work will cover:

- Adapter and device selection with explicit capability reporting.
- Pipeline caching outside dispatch loops.
- Buffer layouts that avoid per-token uploads and unnecessary host copies.
- f32 and compact-storage kernels with CPU reference comparisons.
- Browser and native WebGPU validation using the same correctness fixtures.

## Acceptance

WebAssembly and WebGPU support will be listed as available only after the
checked-in presets build from a clean checkout and the browser tests exercise
the public API. Performance claims will include the browser, device, model,
threading mode, memory configuration, and raw measurement command.
