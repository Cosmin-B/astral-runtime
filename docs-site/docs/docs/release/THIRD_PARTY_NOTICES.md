---
title: "Third-Party notices"
slug: docs/release/THIRD_PARTY_NOTICES
---

<a id="third-party-notices"></a>

This file tracks dependencies that may be bundled, linked, copied into engine
packages, or required to build release artifacts. It is intentionally short; the
full license text remains in each dependency's upstream license file.

## Runtime dependencies

| Component | Source | Local path | License | Release note |
|---|---|---|---|---|
| llama.cpp / ggml | <https://github.com/ggml-org/llama.cpp> | `external/llama.cpp` | MIT | CPU/CUDA/MTMD backend integration. Include `external/llama.cpp/LICENSE` in source or binary distributions that contain llama.cpp-derived code. |
| Tracy | <https://github.com/wolfpld/tracy> | `external/tracy` | BSD 3-Clause | Optional profiling dependency for `*-prof` presets. Include `external/tracy/LICENSE` if Tracy code is shipped. |

## Engine package metadata

| Component | Local path | Release note |
|---|---|---|
| Unity package metadata | `plugins/unity/package.json` | Uses Unity package metadata and declares `com.unity.collections`. Unity package releases must carry the root `LICENSE` and this notice file. |
| Unreal plugin metadata | `plugins/unreal/AstralRT/AstralRT.uplugin` | Unreal packages must carry the root `LICENSE`, this notice file, and any bundled native ThirdParty artifacts. |

## Release checklist

- Verify submodule SHAs before cutting an artifact.
- Include this file and root `LICENSE`/`NOTICE` with source and engine plugin packages.
- Regenerate release metadata and checksums after artifact creation.
- Do not claim artifact signing unless a detached signature or signed checksum file is present.

Source: [View the pinned source](https://github.com/Cosmin-B/astral-runtime/blob/f2d13b77c70624ede5bc06823d4a794a4b955e10/docs/release/THIRD_PARTY_NOTICES.md) · [Edit this source](https://github.com/Cosmin-B/astral-runtime/edit/main/docs/release/THIRD_PARTY_NOTICES.md)
