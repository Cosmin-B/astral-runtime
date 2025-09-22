# Release Acceptance Matrix

This matrix is the minimum evidence required before calling an Astral build a
release candidate.

## Fast Native Gate

| Area | Command | Required for RC |
|---|---|---|
| Debug native build/tests | `cmake --preset dev && cmake --build --preset dev -j && ctest --preset dev -j --output-on-failure` | Yes |
| Release native build/tests | `cmake --preset release-with-tests && cmake --build --preset release-with-tests -j && ctest --preset release-with-tests -j --output-on-failure` | Yes |
| Release metadata | `./scripts/generate_release_metadata.sh dist` | Yes |

## Unreal Gate

| Area | Command or lane | Required for RC |
|---|---|---|
| UE 5.7 full container | `ghcr.io/epicgames/unreal-engine:dev-5.7.4` | Yes |
| UE 5.7 slim container | `ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4` | Yes |
| Automation tests | `UNREAL_EDITOR=/path/to/UnrealEditor-Cmd ./scripts/run_unreal_ci_tests.sh` | Yes |
| UE 5.4/5.5/5.6 compatibility | Compile plugin and run smoke tests | Yes |

## Backend And Model Gate

| Area | Required evidence | Required for RC |
|---|---|---|
| CPU llama | Native tests plus feature benchmark with a real GGUF | Yes |
| CUDA | Real GPU runner, CPU-vs-CUDA parity, auto/cuBLAS/MMQ modes | Yes |
| Multimodal | `ASTRAL_ENABLE_MTMD=ON` build plus real vision/audio projector fixtures | Yes |
| HF matrix | Pinned GGUF matrix logs and pass/fail summary | Yes |

## Release Artifacts

| Artifact | Required evidence |
|---|---|
| Core runtime package | Zip in `dist/`, `checksums.sha256`, dependency manifest |
| Unity package | Native binaries, package metadata, license/notice files, ABI tests |
| Unreal plugin package | ThirdParty native libraries, license/notice files, UE Automation evidence |
| Signing | Detached signatures or documented waiver |
| Rollback | Release notes identify previous known-good artifact and dependency pins |
