# Model Sources (GGUF) for Tests/Bench

Astral tests and benchmarks use GGUF models downloaded from Hugging Face. The helper script is `astral/tests/model_downloader.sh`.

## Default (inference smoke)
- Preset: `gpt2-q2k`
- Notes: small generative GGUF (good for CI and quick local runs).

## Embeddings models (small)
These are encoder/embedding GGUFs (typically 384-dim class):
- Preset: `embed-minilm-q2k`
  - Repo: `second-state/All-MiniLM-L6-v2-Embedding-GGUF`
  - File: `all-MiniLM-L6-v2-Q2_K.gguf`
- Preset: `embed-bge-small-q4km`
  - Repo: `CompendiumLabs/bge-small-en-v1.5-gguf`
  - File: `bge-small-en-v1.5-q4_k_m.gguf`
- Preset: `embed-nomic-v1-5-q2k`
  - Repo: `nomic-ai/nomic-embed-text-v1.5-GGUF`
  - File: `nomic-embed-text-v1.5.Q2_K.gguf`

## LiquidAI (generative)
These are “small” relative to multi-billions, but still much larger than GPT-2:
- Preset: `liquid-lfm2-350m-q4km`
  - Repo: `LiquidAI/LFM2-350M-GGUF`
  - File: `LFM2-350M-Q4_K_M.gguf`

## Newer small generative fixtures
These are useful for local and Unreal real-model smoke runs when GPT-2 is too old or too weak:
- Preset: `gemma3-270m-q4km`
  - Repo: `gguf-org/gemma-3-270m-gguf`
  - File: `gemma-3-270m-q4_k_m.gguf`
- Preset: `gemma3-1b-it-q4km`
  - Repo: `ggml-org/gemma-3-1b-it-GGUF`
  - File: `gemma-3-1b-it-Q4_K_M.gguf`
- Preset: `qwen3-0.6b-q8`
  - Repo: `Qwen/Qwen3-0.6B-GGUF`
  - File: `Qwen3-0.6B-Q8_0.gguf`
- Preset: `qwen3-1.7b-q8`
  - Repo: `Qwen/Qwen3-1.7B-GGUF`
  - File: `Qwen3-1.7B-Q8_0.gguf`
- Preset: `smollm3-3b-q4km`
  - Repo: `ggml-org/SmolLM3-3B-GGUF`
  - File: `SmolLM3-Q4_K_M.gguf`

## LiquidAI (multimodal / v2.5)
Used for vision/audio coverage and benches:
- `LFM2.5-1.2B-Base` (text): `LiquidAI/LFM2.5-1.2B-Base-GGUF`
- `LFM2.5-1.2B-Instruct` (instruction): `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`
- `LFM2.5-1.2B-JP` (text, JP): `LiquidAI/LFM2.5-1.2B-JP-GGUF`
- `LFM2.5-VL-1.6B` (vision): `LiquidAI/LFM2.5-VL-1.6B-GGUF`
- `LFM2.5-Audio-1.5B` (audio): `LiquidAI/LFM2.5-Audio-1.5B-GGUF`

Download helpers:
- `scripts/hf_gguf_download_lfm25_text.sh`
- `scripts/hf_gguf_download_lfm25_all.sh`

## Downloader usage

### Presets (recommended)
- `./tests/model_downloader.sh --preset gpt2-q2k`
- `./tests/model_downloader.sh --preset embed-minilm-q2k`
- `./tests/model_downloader.sh --preset gemma3-270m-q4km`
- `./tests/model_downloader.sh --preset gemma3-1b-it-q4km`
- `./tests/model_downloader.sh --preset qwen3-0.6b-q8`
- `./tests/model_downloader.sh --preset qwen3-1.7b-q8`
- `./tests/model_downloader.sh --preset smollm3-3b-q4km`

Preset inspection and local validation:
- `./tests/model_downloader.sh --preset qwen3-0.6b-q8 --dry-run`
- `./tests/model_downloader.sh --preset qwen3-0.6b-q8 --info`
- `./tests/model_downloader.sh --preset qwen3-0.6b-q8 --print-path`
- `./tests/model_downloader.sh --preset qwen3-0.6b-q8 --validate-only`

`--validate-only` does not download. It verifies the existing local file against
the preset size and SHA-256, so truncated or checksum-drifted files fail before
tests or sample runners spend time configuring a build.

`--info` prints machine-readable preset metadata for setup tools that need the
resolved path, URL, checksum, model type, context length, embedding dimension,
and repeatable download command without starting a network request.

### Environment variables
Astral uses a single model for decode/stream gates and (optionally) a different model for embeddings tests:
- `ASTRAL_TEST_DECODE_MODEL`: decode/stream gates + CPU provider harness (expects a generative/decode-capable GGUF)
- `ASTRAL_TEST_EMBED_MODEL`: embeddings tests (can be an encoder-only GGUF like MiniLM/BGE)
- `ASTRAL_TEST_MODEL`: legacy fallback used when the specific variable is not set

Fixture-dependent tests use explicit `SKIP_TEST(...)` reporting when the needed
model or projector is absent. A skipped fixture probe is acceptable in the fast
mock/default lane; release lanes set the required environment variables and fail
before CTest when required fixtures are missing.

Multimodal (optional):
- `ASTRAL_TEST_VISION_MODEL` + `ASTRAL_TEST_VISION_MEDIA` (mmproj/projector GGUF)
- `ASTRAL_TEST_AUDIO_MODEL` + `ASTRAL_TEST_AUDIO_MEDIA` (mmproj/projector GGUF)

### Hugging Face repo + file
- `./tests/model_downloader.sh --hf-repo second-state/All-MiniLM-L6-v2-Embedding-GGUF --hf-file all-MiniLM-L6-v2-Q2_K.gguf`

### Direct URL
- `./tests/model_downloader.sh --url <hf-resolve-url> --file <name.gguf>`

### Private/gated repos
- `export HF_TOKEN=...`
- Then run the same commands; the script sends an `Authorization: Bearer` header when `HF_TOKEN` is set.
