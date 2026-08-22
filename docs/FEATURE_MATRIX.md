# Feature matrix

This is the maintained map from Astral features to CPU-only, CUDA, and embedded
builds. A check means the implementation is available in that configuration. A
warning points to an optional dependency or provider constraint described below.

## Legend

- ✅ available in the named build or preset
- ⚠️ implemented with the caveat described below the table
- 🧪 available but not yet a supported configuration
- ❌ not available in that configuration

## Core runtime / build knobs

| Capability | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| C ABI runtime (`astral_rt`) | ✅ | ✅ | ✅ |
| Static lib | ✅ | ✅ | ✅ |
| Shared lib | ✅ (optional) | ✅ (optional) | ✅ (optional) |
| Virtual memory runtime (`astral_init`) | ✅ | ✅ | ❌ (use an arena mode) |
| Arena init (`astral_init`, owned/borrowed) | ✅ | ✅ | ✅ |
| Exceptions across ABI | ✅ (caught at ABI) | ✅ (caught at ABI) | ✅ (presets build with `ASTRAL_NO_EXCEPTIONS=ON`) |
| Dynamic backends (`dlopen`/`LoadLibrary`) | ✅ (optional) | ✅ (optional) | ❌ (presets disable) |
| JSON-schema grammar helper | ✅ (optional) | ✅ (optional) | ❌ (presets disable) |
| Multimodal media (`ASTRAL_ENABLE_MTMD`) | ⚠️ (opt-in, compatible projector required) | ⚠️ (opt-in, CUDA routing is model-dependent) | ❌ |

Notes:
- Embedded presets are intended to be “no VM / no dynamic loader / no JSON-schema grammar” by default (`docs/EMBEDDED_PROFILE.md`).
- “No exceptions” is a build constraint. Third-party code may still throw.
- Multimodal media requires `ASTRAL_ENABLE_MTMD=ON` and a media projector file initialized via `astral_model_media_init`.

## Providers / backends

| Provider | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| `cpu` (llama.cpp) | ✅ | ✅ | ✅ |
| `cuda` (llama.cpp GGML_CUDA offload) | ❌ | ✅ | 🧪 (generally not targeted) |
| `remote` (HTTP transport) | ⚠️ | ⚠️ | ❌ (threads disabled) |
| Dynamic provider plugins | ✅ (optional) | ✅ (optional) | ❌ |

Notes:
- CUDA support uses llama.cpp GGML_CUDA offload. The supported selection modes
  and parity commands are in `docs/CUDA_PARITY.md`.
- CUDA builds support multiple ggml-cuda kernel selection modes:
  - Default “auto” selection (`dev-cuda` / `release-cuda`)
  - Forced cuBLAS (`dev-cuda-cublas`)
  - Forced MMQ kernels (`dev-cuda-mmq`)
  - Validate all three modes via `scripts/run_cuda_parity_matrix.sh` (see `docs/CUDA_PARITY.md`).
- CUDA builds expose multi-GPU selection and split fields through
  `AstralModelDesc`. Availability depends on the GGML CUDA build and devices.

## Model loading (PATH / MEMORY / IO)

| Model source | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| `PATH` (`astral_model_load`) | ✅ | ✅ | ✅ |
| `MEMORY` (`astral_model_load`) | ⚠️ | ⚠️ | 🧪 |
| `IO` (`astral_model_load`) | ⚠️ | ⚠️ | 🧪 |

Caveats (MEMORY/IO):
- Current llama.cpp no longer exposes an in-memory loader API that Astral can call directly, so CPU/CUDA providers currently **materialize to a temp file** and then call `llama_model_load_from_file` on desktop.
- MEMORY/IO does not guarantee a filesystem-free embedded load while this materialization path is in use.

## Inference features (sessions)

| Feature | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| Session lifecycle (`create/feed/decode/wait/destroy`) | ✅ | ✅ | ✅ |
| Streaming UTF-8 output (`astral_stream_read`) | ✅ | ✅ | ✅ |
| Logprobs meta stream (`set_logprobs` + `stream_read_meta`) | ✅ | ✅ | ✅ |
| Stop sequences (tokenized) | ✅ | ✅ | ✅ |
| Slots (`astral_session_set_slot`) | ✅ (provider-dependent) | ✅ (provider-dependent) | ✅ (provider-dependent) |
| Image/audio prompt feed (`astral_session_feed_image/audio`) | ⚠️ (MTMD and media init required) | ⚠️ (MTMD, media init, and model support required) | ❌ |
| KV/state save/load (`state_size/save/load`) | ✅ | ✅ | ✅ |
| KV/state deterministic continuation | ✅ | ✅ | ✅ |
| GBNF grammar (`set_grammar_gbnf`) | ✅ | ✅ | ✅ |
| JSON schema grammar (`set_grammar_json_schema`) | ✅ (optional) | ✅ (optional) | ❌ (presets disable) |

Streaming rows describe the complete output stream. Individual reads return
byte chunks and may end within a UTF-8 code point, so text consumers need an
incremental decoder.

Notes:
- KV save/load now includes an Astral header that serializes sampler + RNG state so continuations can be deterministic after load.
- Cross-backend CPU/CUDA token determinism is not guaranteed. See `docs/CUDA_PARITY.md` for the comparison policy.
- Conversation media prompts require provider slot position queries (`session_slot_pos`). The CPU backend supports them when MTMD is enabled.

## Embeddings

| Feature | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| Embeddings API (`astral_embed_*`) | ✅ | ✅ | ✅ |
| Image/audio embeddings (`astral_embed_enqueue_*`) | ⚠️ (mtmd + model support) | 🧪 | ❌ |

## Native product surfaces

| Feature | CPU-only (desktop) | CUDA build (desktop) | Embedded presets |
|---|---:|---:|---:|
| Tokenization count/batch/detokenize sizing | ✅ | ✅ | ✅ |
| Prompt cache handles, stats, save/load, token view | ✅ | ✅ | ✅ |
| LoRA adapter handles and session attachment | ⚠️ (backend/model support) | ⚠️ (backend/model support) | ⚠️ (backend/model support) |
| Toolsets and tool-call parsing | ✅ | ✅ | ✅ |
| Text/token chunk planning | ✅ | ✅ | ✅ |
| Vector memory index, save/load, cursor fetch | ✅ | ✅ | ✅ |
| Continuous-batching conversations | ✅ | ✅ | ❌ (threads disabled) |
| Native agents with system prompt/history/prompt cache stats | ✅ | ✅ | ❌ (threads disabled) |
| Remote runtime transport | ⚠️ | ⚠️ | ❌ (threads disabled) |

Notes:
- Vector memory supports exhaustive flat search and bounded graph search. Flat
  results are exact for scores produced by the selected storage format. Group
  filters use that flat scanner.
- Continuous batching requires a backend with slot/batch operations. The CPU
  backend implements that surface. It also requires
  `ASTRAL_ENABLE_THREADS=ON`.
- Remote runtime support uses `backend_name = "remote"` with an HTTP provider
  for health, tokenization, streaming completion chunks, authentication, and
  embeddings.
  HTTPS is rejected with `ASTRAL_E_UNSUPPORTED` when the HTTP client is built
  without TLS support.

## Test validation map

| Test | What it covers | Where it should run |
|---|---|---|
| `test_tokenization` | tokenization sizing, batch, detokenize | CPU-only + CUDA build |
| `test_prompt_cache` | prompt cache hits, eviction, save/load | CPU-only + CUDA build |
| `test_inference` | sessions, grammar, LoRA, prompt cache, agents, vector memory | CPU-only + CUDA build |
| `test_continuous_batching` | conversation slot fairness and CPU probe | CPU-only + CUDA build |
| `test_cuda_parity` | CUDA surface + (optional) CPU-vs-CUDA parity harness | CUDA build (optional inference via env) |
| `test_cuda_e2e` | end-to-end logprobs/grammar/kv/embeddings on real model | CPU-only always, CUDA when `ASTRAL_TEST_CUDA_E2E=1` |

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

# Parse a consolidated log to CSV and reject failed, skipped-only, or incomplete evidence:
./scripts/parse_hf_matrix_log.py --in benchmarks/results/hf-matrix-<...>-cuda-auto.txt --out /tmp/hf-matrix.csv --require-pass
```
