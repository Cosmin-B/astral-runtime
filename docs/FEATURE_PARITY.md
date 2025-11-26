# Feature Parity Notes

This document records product gaps against Unity-focused llama.cpp wrappers. It
is not a release status page. Current support status lives in
[`FEATURE_MATRIX.md`](FEATURE_MATRIX.md), and production blockers live in
[`PRODUCTION_READINESS_AUDIT.md`](PRODUCTION_READINESS_AUDIT.md).

## Competitor Baseline

Unity wrappers such as LLMUnity and LlamaCppUnity commonly provide:

- GGUF model loading and text generation.
- Streaming output callbacks.
- Chat-style helpers and prompt templates.
- Configurable CPU threads and GPU offload through llama.cpp builds.
- Embeddings, grammar constraints, LoRA, and KV-state workflows.
- Unity editor utilities for model management and samples.
- Mobile package workflows for Android and iOS.

LLMUnity also includes application-layer features that Astral should not pull
into the core runtime by default, such as RAG indexing and remote-server
workflows. Those belong behind optional layers or separate packages.

## Astral Maintained Surface

Astral already has maintained native surfaces for:

- C ABI runtime initialization, model handles, sessions, streaming, reset, and
  error reporting.
- CPU llama.cpp provider integration.
- Provider registration and dynamic provider loading.
- Grammar, logprobs, KV-state save/load, slots, LoRA/adapters, and embeddings
  where the selected backend/model supports them.
- Unity P/Invoke wrappers with `NativeArray` streaming.
- Unreal runtime wrappers with bytes-first streaming and native ownership.
- Native release gates for ABI layout, shared exports, dependency pins, source
  prose, release metadata, allocation tracking, RSS, and syscall behavior.

Do not read this list as production sign-off. Real Unity Editor, UnrealEditor,
CUDA, multimodal, Windows large-page, HF matrix, mobile, and protected signing
evidence is still required before release claims.

## Deliberate Gaps

These areas remain product work or release validation work:

| Area | Status |
|---|---|
| Unity editor tooling | Basic validation helpers exist; richer model-management UI is not implemented. |
| Chat templates | Roadmap item; no maintained template engine is part of the core runtime today. |
| Function calling | Policy-layer work; grammar primitives exist, but tool dispatch is not a core runtime feature. |
| RAG / ANN | Out of core scope unless introduced as an optional package. |
| Remote server | Out of core scope; keep native runtime embeddable. |
| CUDA release path | Build/runtime surface exists; real GPU parity/e2e evidence is required. |
| Mobile release path | Artifact lanes exist; device/editor validation remains required. |
| Unreal production path | Plugin and runners exist; UE 5.7 container and UE 5.4+ editor evidence remains required. |

## Strategy

- Keep the core C ABI small and engine-neutral.
- Put Unity and Unreal convenience APIs in their plugins, not in the native
  runtime.
- Treat GPU, media, mobile, and engine editor support as release-evidence
  requirements, not marketing checkboxes.
- Add higher-level features only when they have a clear owner, tests, and a
  boundary that keeps decode/stream hot paths direct.

## References

- [`FEATURE_MATRIX.md`](FEATURE_MATRIX.md): maintained capability map.
- [`CUDA_PARITY.md`](CUDA_PARITY.md): CUDA parity policy and commands.
- [`integration/UNREAL_57_QUICKSTART.md`](integration/UNREAL_57_QUICKSTART.md):
  UE 5.7 runner path and evidence list.
- [`../plugins/unity/README.md`](../plugins/unity/README.md): current Unity
  package usage.
