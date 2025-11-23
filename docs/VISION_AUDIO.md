# Vision + Audio Support

Astral supports multimodal GGUF models (vision + audio) via the optional **libmtmd** path in the CPU backend. This enables:

- **Multimodal inference**: images/audio influence text generation.
- **Multimodal embeddings**: image/audio (and text+image/audio) → vectors when supported by the model.

## Build flag

Enable libmtmd integration:

```bash
cmake -B build -DASTRAL_ENABLE_MTMD=ON
cmake --build build
```

## Model initialization (media projector)

Media support must be initialized **before** creating sessions or embedders that use media:

```c
AstralModelMediaDesc media{};
media.size = sizeof(AstralModelMediaDesc);
media.source_kind = ASTRAL_MODEL_SOURCE_PATH;
media.media_path = span_from_cstr("/path/to/mmproj-or-media.gguf");
media.flags = 0; // or ASTRAL_MEDIA_FLAG_USE_GPU / ASTRAL_MEDIA_FLAG_WARMUP
media.gpu_route_flags = 0; // optional: ASTRAL_GPU_ROUTE_DEVICE / DEVICE_MASK / STREAM
media.gpu_device = 0;      // CUDA device index request when DEVICE is set
media.gpu_device_mask = 0; // allowed-device bitset request when DEVICE_MASK is set
media.gpu_stream = nullptr; // backend-specific stream handle (optional)

astral_model_media_init(model, &media);
```

The `media_path` is model-specific (typically a projector/encoder GGUF). If media is already initialized, the call returns `ASTRAL_E_STATE`.

## Input formats

### `AstralImageDesc`
- RGB8 / RGBA8 / RGB_F32 pixel formats
- `row_stride` is bytes; 0 means tightly packed
- GPU routing fields are advisory requests; set `gpu_route_flags` for the fields the caller wants the backend to consume
- caller owns pixel memory for the duration of the feed/enqueue call

### `AstralAudioDesc`
- PCM F32 or I16
- `frame_count` is per-channel frames
- GPU routing fields are advisory requests; set `gpu_route_flags` for the fields the caller wants the backend to consume
- caller owns sample memory for the duration of the feed/enqueue call

## Multi-GPU (CUDA model load)

For CUDA builds, `AstralModelDesc` exposes multi-GPU selection knobs (device indices refer to CUDA devices as enumerated by ggml):
- `gpu_devices` / `gpu_device_count` or `gpu_device_mask` to pick devices
- `gpu_main` and `gpu_split_mode` (none/layer/row)
- `gpu_tensor_split` for explicit split ratios

Set `gpu_flags` to indicate which fields are active. The llama.cpp CUDA backend consumes the supported model-load fields; release sign-off still requires real multi-GPU routing evidence.

## Sessions and conversations

```c
// Sessions
astral_session_feed(session, text_chunk, /*finalize=*/0);
astral_session_feed_image(session, &image, /*finalize=*/1);
astral_session_decode(session);

// Conversations (continuous batching)
astral_conv_feed_audio(conv, &audio, /*finalize=*/1);
astral_conv_decode(conv);
```

Conversations require provider slot-position support (`session_slot_pos`); mock + CPU backends implement this when mtmd is enabled.

## Embeddings

```c
astral_embed_enqueue_image(embedder, &image, &ticket);
astral_embed_collect(embedder, ticket, out_vec);
```

Use `astral_embed_enqueue_multimodal` for text+image/audio when supported by the model.

## Capabilities

Use `astral_model_caps()` and `astral_model_media_info()` to check:
- `ASTRAL_CAP_IMAGE` / `ASTRAL_CAP_AUDIO`
- `ASTRAL_CAP_MM_EMBEDDINGS`
- `AstralMediaInfo.supports_image` / `supports_audio`

## Reference models (Liquid)

These are the Liquid models used for validation and bench coverage:

- `LFM2.5-1.2B-Base` → Hugging Face (https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base)
- `LFM2.5-1.2B-Instruct` → Hugging Face, LEAP, Playground (https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- `LFM2.5-1.2B-JP` → Hugging Face, LEAP (https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP)
- `LFM2.5-VL-1.6B` → Hugging Face, LEAP, Playground, Demo
- `LFM2.5-Audio-1.5B` → Hugging Face, LEAP, Playground

## Tests and benches

Real media init smoke tests:

- `ASTRAL_TEST_VISION_MODEL`, `ASTRAL_TEST_VISION_MEDIA`
- `ASTRAL_TEST_AUDIO_MODEL`, `ASTRAL_TEST_AUDIO_MEDIA`

The default `test_media` run skips real fixtures when they are absent. The release lane makes them required:

```bash
./scripts/run_multimodal_validation.sh \
  --fixture-manifest scripts/mtmd_fixture_manifest_lfm25.json \
  --fixture-dir tests/models/hf-lfm25 \
  --check-fixtures
./scripts/run_multimodal_validation.sh \
  --fixture-manifest scripts/mtmd_fixture_manifest_lfm25.json \
  --fixture-dir tests/models/hf-lfm25 \
  --bench
```

The manifest path is the release-default route because it pins Hugging Face
revisions and filenames. For local experiments with newer small multimodal
fixtures, keep the manifest format and pass explicit `--vision-model`,
`--vision-media`, `--audio-model`, or `--audio-media` overrides only when
comparing candidates.

Feature bench inputs:

- `ASTRAL_BENCH_VISION_MODEL`, `ASTRAL_BENCH_VISION_MEDIA`
- `ASTRAL_BENCH_AUDIO_MODEL`, `ASTRAL_BENCH_AUDIO_MEDIA`
- `ASTRAL_BENCH_MEDIA_IMAGE_W`, `ASTRAL_BENCH_MEDIA_IMAGE_H`
- `ASTRAL_BENCH_MEDIA_AUDIO_RATE`, `ASTRAL_BENCH_MEDIA_AUDIO_CHANNELS`, `ASTRAL_BENCH_MEDIA_AUDIO_FRAMES`
