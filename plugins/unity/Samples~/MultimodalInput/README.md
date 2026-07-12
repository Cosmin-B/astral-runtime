# Multimodal Input

Attach `MultimodalInputExample` and `AstralRuntimeInitializer` to a GameObject.
Assign a readable `RGBA32` texture for image calls and an accessible `AudioClip`
for audio calls. Set `mediaProjectorPath` to the projector or encoder GGUF used
by the selected model; the mock backend accepts the default mock paths.

- `RunImageGeneration` feeds the prompt and caller-owned texture bytes.
- `RunAudioGeneration` copies interleaved clip samples into a temporary
  `NativeArray<float>` and feeds explicit channel and sample-rate metadata.
- `EmbedImage` enqueues a multimodal embedding, creates a generic request
  reference, collects into a caller-owned vector, and reports terminal status.

Every path checks model capabilities before calling the media API. Texture and
audio memory remain valid until the synchronous feed or collect operation
returns.
