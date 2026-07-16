# Changelog

Astral is pre-1.0. Entries remain under **Unreleased** until the first tagged
release.

## Unreleased

### Added

- C++17 native runtime with a sized C ABI, static and shared library builds,
  mock and llama.cpp CPU providers, and optional CUDA offload.
- Streaming generation, continuous batching, embeddings, prompt caches, LoRA
  adapters, structured output, model presets, native agents, and vector search.
- Unity 6000.0 and Unreal Engine 5.4+ packages with maintained samples and
  native artifact layouts.
- Release checks for ABI layout, exported symbols, dependency pins, SBOMs,
  checksums, documentation links, engine package layout, and hot-path
  allocation and syscall contracts.

### Changed

- Consolidated initialization under `AstralInit` and `astral_init` before the
  first public ABI release.
- Moved continuous-batching conversation lifetime protection from per-object
  executor reference traffic to epoch reclamation.
- Added runtime AVX2 and F16C feature checks before selecting E5M2 conversion
  kernels.
- Set Unity 6000.0 as the minimum supported Unity editor version.
- Restricted automated repository jobs to trusted Linux self-hosted runners.
- Licensed the project under Apache License 2.0.

### Fixed

- Hardened conversation stream ownership, cancellation, reset, and token spill
  behavior.
- Corrected compact-vector conversion behavior for non-finite E5M2 encodings.
- Added bounds and failure handling to concurrency, virtual-memory, and public
  ABI boundaries.
