# Documentation

This directory is the maintained documentation source for Astral. The public
header defines the ABI; these documents explain behavior, integration, and the
evidence required for support claims.

## Start Here

- [Build guide](../BUILD.md)
- [Feature matrix](FEATURE_MATRIX.md)
- [ABI versioning](ABI_VERSIONING.md)
- [Error handling](api/ERROR_HANDLING.md)
- [Release acceptance](release/RELEASE_ACCEPTANCE_MATRIX.md)

## Native APIs

- [Continuous batching](api/CONTINUOUS_BATCHING.md)
- [Generation controls](api/GENERATION_CONTROLS.md)
- [Tokenization](api/TOKENIZATION.md)
- [Prompt cache](api/PROMPT_CACHE.md)
- [Structured output](api/STRUCTURED_OUTPUT.md)
- [LoRA adapters](api/LORA_ADAPTERS.md)
- [Chunking](api/CHUNKING.md)
- [Memory index](api/MEMORY_INDEX.md)
- [Agent runtime](api/AGENT_RUNTIME.md)
- [Remote runtime](api/REMOTE_RUNTIME.md)
- [Model paths](api/MODEL_PATHS.md) and [model presets](api/MODEL_PRESETS.md)

The complete function and descriptor surface is in
[`include/astral_rt.h`](../include/astral_rt.h).

## Engine Integration

- [Unity integration](integration/UNITY_INTEGRATION.md)
- [Unreal integration](integration/UNREAL_INTEGRATION.md)
- [Unreal 5.7 quickstart](integration/UNREAL_57_QUICKSTART.md)
- [Engine ownership patterns](architecture/ENGINE_INTEGRATION_PATTERNS.md)

Package-specific usage and samples live under [`plugins/unity`](../plugins/unity)
and [`plugins/unreal/AstralRT`](../plugins/unreal/AstralRT).

## Runtime Architecture

- [Backend architecture](architecture/BACKEND_ARCHITECTURE.md)
- [Concurrency model](architecture/CONCURRENCY_MODEL.md)
- [Memory architecture](architecture/MEMORY_ARCHITECTURE.md)
- [Low-level primitives](architecture/LOW_LEVEL_PRIMITIVES.md)
- [Embedded profiles](EMBEDDED_PROFILE.md)
- [Vision and audio](VISION_AUDIO.md)

The architecture documents include implementation constraints and measured
design targets. Current product support still follows the feature and release
matrices.

## Profiling And Validation

- [Hot-path profiling boundaries](api/HOT_PATH_PROFILING.md)
- [Tracy profiling](PROFILING_TRACY.md)
- [CUDA parity](CUDA_PARITY.md)
- [CUDA kernel strategy](CUDA_KERNEL_STRATEGY.md)
- [Release manifests](release/)

Raw benchmark output, profiler captures, machine-specific paths, and private
release credentials are not repository documentation. Keep published claims
tied to reproducible commands and the maintained acceptance gates.
