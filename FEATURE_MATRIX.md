# Feature Matrix

The maintained capability table is [docs/FEATURE_MATRIX.md](docs/FEATURE_MATRIX.md).
It maps each support claim to a build preset, local test, external release lane,
or an explicit caveat.

## Current Support

- **CPU desktop**: static and shared C ABI builds, llama.cpp inference,
  tokenization, embeddings, continuous batching, memory search, and optional
  multimodal support.
- **CUDA desktop**: CPU feature surface plus CUDA offload, with real-device
  parity and end-to-end validation required for release evidence.
- **Embedded**: static, arena-backed CPU builds for x86-64, ARM64, and ARMv7;
  virtual memory, dynamic providers, and JSON-schema grammar are disabled by
  the embedded presets.
- **Engine integrations**: maintained Unity and Unreal packages with separate
  native build, ABI, packaging, and runtime validation lanes.

## Detailed Matrix

See the [canonical table](docs/FEATURE_MATRIX.md) for provider availability,
build options, test coverage, and platform-specific caveats. Support is not
inferred from source presence alone; entries without current evidence remain
marked as conditional or pending.
