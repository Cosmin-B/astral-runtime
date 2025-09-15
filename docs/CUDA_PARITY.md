# CUDA Parity (v0.2 workbench)

This document defines the concrete “parity checklist” for Astral’s CUDA backend (CUDA first; Metal + DirectML later).

Today (v0.1), the `cuda` provider is a *scaffold*: it reuses the CPU provider ops table, and CUDA is enabled through llama.cpp/ggml (`GGML_CUDA=ON`). Parity work in v0.2 is about making this behavior reliable, testable, and fast.

## What “Parity” Means

Parity is not just “it runs”. For a given model + prompt + sampler config:

- Correctness: token IDs produced are consistent (or within documented tolerances when floating-point differences exist).
- Feature coverage: every API surface works (streaming, logprobs, grammar, KV save/load, embeddings, slots).
- Performance: stable throughput and latency without regressions vs CPU-only for CPU workloads.

## Phase 1 — Parity Checklist (Must Work)

**Kernel modes / backend variants**
- [ ] Default CUDA kernels (mmq auto / cuBLAS auto) pass all Phase 1 tests.
- [ ] Forced cuBLAS (`GGML_CUDA_FORCE_CUBLAS=ON`) passes all Phase 1 tests.
- [ ] Forced MMQ (`GGML_CUDA_FORCE_MMQ=ON`) passes all Phase 1 tests.
  - Rationale: ggml-cuda selects between custom MMQ kernels and cuBLAS depending on GPU/quant; we treat both as first-class supported modes and test them separately.

**Model load**
- [ ] `astral_model_load()` / `astral_model_load2(PATH)` with `backend_name="cuda"` loads successfully.
- [ ] `gpu_layers > 0` selects CUDA backend when enabled (and falls back to CPU when not built with CUDA).
- [ ] Multi-file GGUF splits (if supported by llama.cpp) load via PATH.
- [ ] Memory/IO sources: document the policy clearly (embedded builds must not require filesystem syscalls).

**Session + streaming**
- [ ] `astral_session_create`, `feed`, `decode`, `stream_read`, `wait` works with CUDA offload.
- [ ] Cancellation (`astral_session_cancel` + `wait`) behaves correctly under load.

**Meta side-channel (token ids/logprobs)**
- [ ] `astral_session_set_logprobs(1)` emits `AstralTokenMeta` events (`token_id` always valid).
- [ ] `top_n` logprobs behaves when enabled (clamped to `ASTRAL_LOGPROBS_MAX`).

**Grammar**
- [ ] GBNF grammar: compile + apply (both session-scoped and slot-scoped variants if supported).
- [ ] JSON schema grammar: same as above when `ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR=ON`.

**KV/state**
- [ ] `astral_session_state_save/load` round-trips and produces identical continuation tokens.

**Embeddings**
- [ ] Embeddings produce expected dimensionality and stable output shape.

## End-to-end validation

The `test_cuda_e2e` suite exercises the following against a real GGUF model:

- Logprobs meta shape + consistency (`AstralTokenMeta.top_n`, ordering, membership)
- GBNF grammar applied in decoding (output constrained to an allowed byte set)
- KV save/load continuation (requires Astral wrapper state; see `session_state_*`)
- Embeddings enqueue/collect

Run it on a CUDA machine:

```bash
ASTRAL_TEST_CUDA_E2E=1 ASTRAL_TEST_CUDA_PARITY_INFER=1 scripts/run_cuda_parity.sh --preset dev-cuda
```

To validate kernel modes (default + cuBLAS + MMQ) in one go:

```bash
ASTRAL_TEST_CUDA_E2E=1 ASTRAL_TEST_CUDA_PARITY_INFER=1 scripts/run_cuda_parity_matrix.sh --arch 120a-real
```

**Slots / executor**
- [ ] Slot selection works; multi-slot scheduling does not deadlock and is callback-safe.

## Phase 2 — Performance Checklist (Must Be Fast)

- [ ] TTFT and tok/s benchmarks for representative model tiers (small, medium).
- [ ] No pathological GPU/CPU synchronization in the decode loop.
- [ ] Optional profiling build (`*-prof` presets) shows sensible Tracy zones for CUDA hot paths.

## Testing Strategy

1) **Always-on smoke tests** (CPU machines, CI-friendly):
- CUDA backend presence/absence surface behavior.

2) **CUDA machine tests** (manual/CI best-effort):
- Run `tests/test_cuda_parity.cpp` with `ASTRAL_TEST_CUDA_PARITY_INFER=1`.
- Strict mode (`ASTRAL_TEST_CUDA_PARITY_STRICT=1`) is opt-in and checks “near parity”:
  - Each backend’s chosen token must be within the other backend’s captured `top_n` (currently 8).
  - Each backend’s chosen token must be within the other backend’s top-8 ranks.
  - Comparison is done for the first generated token only (later tokens condition on different contexts after any drift).
- For debugging exact token-id parity (expected to fail on some machines), also set `ASTRAL_TEST_CUDA_PARITY_EXACT=1`.

## Reference runner

Use `scripts/run_cuda_parity.sh` to build + run the parity test on a CUDA machine.
