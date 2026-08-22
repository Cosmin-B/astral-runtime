# AstralSample

Generated sidecar Unreal sample project for AstralRT.

Build the native plugin package from the Astral repo first:

```bash
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

Open `AstralSample.uproject` in UE 5.7 and run the default map. The sample
GameMode spawns `AAstralSampleActor`, which demonstrates model load, streaming generation,
cancellation, embeddings, image/audio media feed, packaged content bytes,
Saved cache bytes, native memory search, and expected error logging through
`LogAstralSample`.

For real-model local runs, pass command-line overrides instead of editing the
generated project:

```bash
AstralSample.sh -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit \
  -AstralBackend=cpu \
  -AstralMemoryBackend=cpu \
  -AstralMediaBackend=cpu \
  -AstralModel=/absolute/path/to/Qwen3-0.6B-Q8_0.gguf \
  -AstralEmbeddingModel=/absolute/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf \
  -AstralMediaPath=/absolute/path/to/mmproj.gguf \
  -AstralMediaPathRoot=Raw \
  -AstralPrompt="Say hello from Astral."
```

`-AstralMemoryBackend=cpu` loads the packaged Content and Saved byte sources
through the CPU backend.
`-AstralMediaPath` and `-AstralMediaPathRoot` initialize a media projector for
the media feed sample.

## Gameplay components

Add `Astral Stateful Npc` to an actor to keep agent history across level or
actor recreation. Configure its model, prompt state, and history file, then call
`Ask`, `Cancel`, `Save History`, or `Load History` from Blueprint. The component
polls one chat request per frame, reports structured `move_to` tool calls, and
stores history below `ProjectSavedDir/Astral` by default.

Add `Astral Local Knowledge` to an actor to index short lore, quest, or dialogue
documents. Set an embeddings-capable model, edit `Document Text`, then call
`Build Index` and `Search`. `Matched Text` contains the original chunks for UI
or prompt assembly. `Save Index` and `Load Index` persist the native snapshot
below `ProjectSavedDir/Astral`. A model without embeddings support returns an
unsupported operation instead of silently substituting vectors.

Add `Astral Character Variants` to compare prompt and model customization
without duplicating model ownership. `Prepare Prompt Cache` tokenizes and
round-trips the system prompt, stores it under a stable cache key, and writes a
snapshot below `ProjectSavedDir/Astral`. `Run` applies the configured stop
sequence and JSON-schema grammar, then optionally attaches a model-scoped
adapter. Adapter paths are optional. An empty path runs the base model.

Add `Astral Multimodal Input` to feed a CPU-readable texture or interleaved
PCM16 bytes. Set a media projector path or payload before calling the image,
audio, or embedding controls. The component checks model capabilities before
each operation, copies Unreal-owned media into Astral descriptors, exposes the
resulting embedding, and keeps model-dependent auto-run disabled.

Use `scripts/run_unreal_sample_package.sh` on a UE 5.7 runner to build, cook,
package, and validate the generated project.
