# Third-Party Notices

This file tracks dependencies that may be bundled, linked, copied into engine
packages, or required to build release artifacts. It is intentionally short; the
full license text remains in each dependency's upstream license file.

## Runtime Dependencies

| Component | Source | Local path | License | Release note |
|---|---|---|---|---|
| llama.cpp / ggml | <https://github.com/ggml-org/llama.cpp> | `external/llama.cpp` | MIT | CPU/CUDA/MTMD backend integration. Include `external/llama.cpp/LICENSE` in source or binary distributions that contain llama.cpp-derived code. |
| Tracy | <https://github.com/wolfpld/tracy> | `external/tracy` | BSD 3-Clause | Optional profiling dependency for `*-prof` presets. Include `external/tracy/LICENSE` if Tracy code is shipped. |

## Engine Package Metadata

| Component | Local path | Release note |
|---|---|---|
| Unity package metadata | `plugins/unity/package.json` | Uses Unity package metadata and declares `com.unity.collections`. Unity package releases must carry the root `LICENSE` and this notice file. |
| Unreal plugin metadata | `plugins/unreal/AstralRT/AstralRT.uplugin` | Unreal packages must carry the root `LICENSE`, this notice file, and any bundled native ThirdParty artifacts. |

## Release Checklist

- Verify submodule SHAs before cutting an artifact.
- Include this file and root `LICENSE`/`NOTICE` with source and engine plugin packages.
- Regenerate release metadata and checksums after artifact creation.
- Do not claim artifact signing unless a detached signature or signed checksum file is present.
