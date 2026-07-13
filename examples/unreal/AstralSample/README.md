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
  -AstralMemoryBackend=mock \
  -AstralMediaBackend=mock \
  -AstralModel=/absolute/path/to/Qwen3-0.6B-Q8_0.gguf \
  -AstralEmbeddingModel=/absolute/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf \
  -AstralMediaPath=/absolute/path/to/mmproj.gguf \
  -AstralMediaPathRoot=Raw \
  -AstralPrompt="Say hello from Astral."
```

`-AstralMemoryBackend=mock` keeps the packaged Content/Saved byte demos on the
mock backend while text generation and embeddings use the real CPU backend.
`-AstralMediaPath` and `-AstralMediaPathRoot` initialize a media projector for
the media feed demo; leave them unset with `-AstralMediaBackend=mock` for a
lightweight descriptor/bridge smoke. Real projector validation remains part of
the MTMD release lane.

Real production sign-off still requires packaging this project on the UE 5.7
release runner and recording the Automation/package logs as release evidence.
