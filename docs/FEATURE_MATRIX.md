# Feature Matrix (v0.1)

This document is the “what actually works where” map for Astral v0.1: CPU-only builds, CUDA builds, and embedded/minimal presets.

If you are comparing to external frameworks, see `docs/FEATURE_PARITY.md`. This file is strictly about **Astral’s shipped surfaces** and **their build/preset requirements**.

## Legend

- ✅ available (implemented + validated via tests)
- ⚠️ available but with caveats (documented below)
- 🧪 available but not yet validated on that platform/config
- ❌ not available in that configuration

## Core runtime / build knobs

| Capability | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| C ABI runtime (`astral_rt`) | ✅ | ✅ | ✅ |
| Static lib | ✅ | ✅ | ✅ |
| Shared lib | ✅ (optional) | ✅ (optional) | ✅ (optional) |
| Virtual memory runtime (`astral_init`) | ✅ | ✅ | ❌ (use `astral_init2` arena) |
| Arena init (`astral_init2`, owned/borrowed) | ✅ | ✅ | ✅ |
| Exceptions across ABI | ✅ (caught at ABI) | ✅ (caught at ABI) | ✅ (presets build with `ASTRAL_NO_EXCEPTIONS=ON`) |
| Dynamic backends (`dlopen`/`LoadLibrary`) | ✅ (optional) | ✅ (optional) | ❌ (presets disable) |
| JSON-schema grammar helper | ✅ (optional) | ✅ (optional) | ❌ (presets disable) |
| Multimodal media (`ASTRAL_ENABLE_MTMD`) | ⚠️ (opt-in; real fixtures required for release) | ⚠️ (opt-in; CUDA media routing still needs real fixture evidence) | ❌ |

Notes:
- Embedded presets are intended to be “no VM / no dynamic loader / no JSON-schema grammar” by default (`docs/EMBEDDED_PROFILE.md`).
- “No exceptions” is a build goal; third-party code may still throw unless fully audited.
- Multimodal media requires `ASTRAL_ENABLE_MTMD=ON` and a media projector file initialized via `astral_model_media_init`.

## Providers / backends

| Provider | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| `cpu` (llama.cpp) | ✅ | ✅ | ✅ |
| `cuda` (llama.cpp GGML_CUDA offload) | ❌ | ✅ | 🧪 (generally not targeted) |
| `mock` | ✅ | ✅ | ✅ |

Notes:
- CUDA support is implemented through llama.cpp GGML_CUDA offload. Production
  release sign-off still requires the real CUDA parity/e2e matrix in
  `docs/CUDA_PARITY.md`.
- CUDA builds support multiple ggml-cuda kernel selection modes:
  - Default “auto” selection (`dev-cuda` / `release-cuda`)
  - Forced cuBLAS (`dev-cuda-cublas`)
  - Forced MMQ kernels (`dev-cuda-mmq`)
  - Validate all three modes via `scripts/run_cuda_parity_matrix.sh` (see `docs/CUDA_PARITY.md`).
- CUDA builds expose multi-GPU selection/split fields through `AstralModelDesc`;
  release sign-off still needs real multi-GPU routing evidence.

## Model loading (PATH / MEMORY / IO)

| Model source | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| `PATH` (`astral_model_load`) | ✅ | ✅ | ✅ |
| `PATH` (`astral_model_load2(PATH)`) | ✅ | ✅ | ✅ |
| `MEMORY` (`astral_model_load2(MEMORY)`) | ⚠️ | ⚠️ | 🧪 |
| `IO` (`astral_model_load2(IO)`) | ⚠️ | ⚠️ | 🧪 |

Caveats (MEMORY/IO):
- Current llama.cpp no longer exposes an in-memory loader API that Astral can call directly, so CPU/CUDA providers currently **materialize to a temp file** and then call `llama_model_load_from_file` on desktop.
- This means MEMORY/IO is **not yet a hard guarantee** of “no filesystem/syscalls” in embedded mode; making that guarantee is tracked under the embedded pillars work.

## Inference features (sessions)

| Feature | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| Session lifecycle (`create/feed/decode/wait/destroy`) | ✅ | ✅ | ✅ |
| Streaming UTF-8 output (`astral_stream_read`) | ✅ | ✅ | ✅ |
| Logprobs meta stream (`set_logprobs` + `stream_read_meta`) | ✅ | ✅ | ✅ |
| Stop sequences (tokenized) | ✅ | ✅ | ✅ |
| Slots (`astral_session_set_slot`) | ✅ (provider-dependent) | ✅ (provider-dependent) | ✅ (provider-dependent) |
| Image/audio prompt feed (`astral_session_feed_image/audio`) | ⚠️ (mtmd + media init; real fixtures required for release) | ⚠️ (mtmd + media init; CUDA routing still needs real fixture evidence) | ❌ |
| KV/state save/load (`state_size/save/load`) | ✅ | ✅ | ✅ |
| KV/state deterministic continuation | ✅ | ✅ | ✅ |
| GBNF grammar (`set_grammar_gbnf`) | ✅ | ✅ | ✅ |
| JSON schema grammar (`set_grammar_json_schema`) | ✅ (optional) | ✅ (optional) | ❌ (presets disable) |

Notes:
- KV save/load now includes an Astral header that serializes sampler + RNG state so continuations can be deterministic after load.
- Cross-backend (CPU vs CUDA) **exact token determinism** is not guaranteed in general; see `docs/CUDA_PARITY.md` for current policy/tests.
- Conversation media prompts require provider slot position queries (`session_slot_pos`); mock + CPU backends support this when mtmd is enabled.

## Embeddings

| Feature | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| Embeddings API (`astral_embed_*`) | ✅ | ✅ | ✅ |
| Image/audio embeddings (`astral_embed_enqueue_*`) | ⚠️ (mtmd + model support) | 🧪 | ❌ |

## Test validation map

| Test | What it covers | Where it should run |
|---|---|---|
| `test_embeddings` | embeddings mock + CPU e2e | CPU-only + CUDA build |
| `test_media` | mock media feed + multimodal embeddings | CPU-only + CUDA build |
| `test_cuda_parity` | CUDA surface + (optional) CPU-vs-CUDA parity harness | CUDA build (optional inference via env) |
| `test_cuda_e2e` | end-to-end logprobs/grammar/kv/embeddings on real model | CPU-only always; CUDA when `ASTRAL_TEST_CUDA_E2E=1` |

Useful invocations:

```bash
# CPU-only validation
cmake --preset release-with-tests && cmake --build --preset release-with-tests -j && ctest --preset release-with-tests -j

# CUDA validation (on a CUDA box)
ASTRAL_TEST_CUDA_E2E=1 ASTRAL_TEST_CUDA_PARITY_INFER=1 scripts/run_cuda_parity.sh --preset dev-cuda --arch 120a-real
```

HF GGUF bench matrix (optional):

```bash
# Download a pinned set of HF GGUF repos (large!):
./scripts/hf_gguf_download_manifest.sh --out tests/models/hf

# Run the feature-surface bench across all downloaded GGUFs (CPU + CUDA):
./scripts/run_hf_bench_matrix.sh --models-dir tests/models/hf --arch 120a-real

# Parse a consolidated log to CSV (one row per model/preset run):
./scripts/parse_hf_matrix_log.py --in benchmarks/results/hf-matrix-<...>-cuda-auto.txt --out /tmp/hf-matrix.csv
```
