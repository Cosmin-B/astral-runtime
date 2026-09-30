---
title: "Documentation"
slug: docs
---

<a id="documentation"></a>

This directory is the maintained documentation source for Astral. The public
header defines the ABI. These documents explain behavior, ownership, and
integration. Release manifests remain separate from reader guides.

## Start here

- [Build guide](/BUILD)
- [Feature matrix](/docs/FEATURE_MATRIX)
- [ABI versioning](/docs/ABI_VERSIONING)
- [Error handling](/docs/api/ERROR_HANDLING)
- [Release acceptance](/docs/release/RELEASE_ACCEPTANCE_MATRIX)

## Native APIs

- [Continuous batching](/docs/api/CONTINUOUS_BATCHING)
- [Generation controls](/docs/api/GENERATION_CONTROLS)
- [Tokenization](/docs/api/TOKENIZATION)
- [Prompt cache](/docs/api/PROMPT_CACHE)
- [Structured output](/docs/api/STRUCTURED_OUTPUT)
- [LoRA adapters](/docs/api/LORA_ADAPTERS)
- [Chunking](/docs/api/CHUNKING)
- [Memory index](/docs/api/MEMORY_INDEX)
- [Agent runtime](/docs/api/AGENT_RUNTIME)
- [Asynchronous delivery](/docs/api/ASYNC_DELIVERY)
- [Remote runtime](/docs/api/REMOTE_RUNTIME)
- [Model paths](/docs/api/MODEL_PATHS) and [model presets](/docs/api/MODEL_PRESETS)

The complete function and descriptor surface is in
[`include/astral_rt.h`](https://github.com/Cosmin-B/astral-runtime/blob/main/include/astral_rt.h).

## Engine integration

- [Unity package guide](/plugins/unity)
- [Unreal plugin guide](/plugins/unreal/AstralRT)
- [Unreal 5.7 quickstart](/docs/integration/UNREAL_57_QUICKSTART)

Package-specific usage and samples live under
[`plugins/unity`](https://github.com/Cosmin-B/astral-runtime/tree/main/plugins/unity) and
[`plugins/unreal/AstralRT`](https://github.com/Cosmin-B/astral-runtime/tree/main/plugins/unreal/AstralRT).

## Runtime architecture

- [Runtime architecture](/docs/architecture/RUNTIME_ARCHITECTURE)
- [Retrieval architecture](/docs/architecture/RETRIEVAL_ARCHITECTURE)
- [Backend architecture](/docs/architecture/BACKEND_ARCHITECTURE)
- [Concurrency model](/docs/architecture/CONCURRENCY_MODEL)
- [Memory architecture](/docs/architecture/MEMORY_ARCHITECTURE)
- [Low-level primitives](/docs/architecture/LOW_LEVEL_PRIMITIVES)
- [Embedded profiles](/docs/EMBEDDED_PROFILE)
- [Vision and audio](/docs/VISION_AUDIO)

Architecture documents describe current implementation constraints. They are
not a roadmap and do not override the feature or release matrices.

## Profiling and measurement

- [Hot-path profiling boundaries](/docs/api/HOT_PATH_PROFILING)
- [Tracy profiling](/docs/PROFILING_TRACY)
- [CUDA parity](/docs/CUDA_PARITY)
- [CUDA kernel strategy](/docs/CUDA_KERNEL_STRATEGY)
- [Release acceptance and manifests](/docs/release/RELEASE_ACCEPTANCE_MATRIX)

Raw benchmark output, profiler files, machine-specific paths, and release
credentials do not belong in repository documentation. Performance pages carry
the command and configuration needed to reproduce a published result.

## Documentation site

The [published documentation](https://astralruntime.dev/docs/) renders
this Markdown and the maintained root guides directly. Build the same strict
artifact with:

```bash
./scripts/build_docs_site.sh
```

The script uses a local MkDocs Material installation when available and falls
back to the pinned container image. Generated HTML is written to
`build/docs-site` and is not committed.

Source: [View the pinned source](https://github.com/Cosmin-B/astral-runtime/blob/f2d13b77c70624ede5bc06823d4a794a4b955e10/docs/README.md) · [Edit this source](https://github.com/Cosmin-B/astral-runtime/edit/main/docs/README.md)
