# Unreal 5.7 Quickstart

This path is for the AstralRT Unreal plugin on Linux with Unreal Engine 5.7 as
the production target. It builds the native Astral runtime, stages the
ThirdParty files into the plugin, and runs the same Automation entrypoint used
by CI.

Real release sign-off still requires access to Epic's UE images or installed
editors. The local native build proves the plugin package is current; it does
not replace a real UnrealEditor run.

## Prerequisites

- A checkout with submodules initialized.
- CMake and a C++20 compiler for the native package build.
- Unreal Engine 5.7.4 for the production lane.
- Access to Epic's GitHub Container Registry packages for the container lanes.
- Unreal 5.4, 5.5, 5.6, and 5.7 editor paths for compatibility evidence.

UE 5.7 Linux validation should report clang `20.1.8` from Epic's Linux toolchain
metadata. The container wrapper prints `Build.version`, `Linux_SDK.json`, and
`clang --version` before running Automation.

## Build The ThirdParty Package

From the Astral repo root:

```bash
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

The package target stages:

- `plugins/unreal/AstralRT/Source/ThirdParty/AstralCore/include/astral_rt.h`
- `plugins/unreal/AstralRT/Source/ThirdParty/AstralCore/lib/<Platform>/...`

The build hashes the staged header and native library after copy. A successful
build prints `Unreal ThirdParty provenance OK`, which means the staged plugin
files match the current source header and built `astral_rt` target.

## Install In A Project

For a project-level install, copy the plugin directory:

```bash
mkdir -p /path/to/MyProject/Plugins
cp -a plugins/unreal/AstralRT /path/to/MyProject/Plugins/AstralRT
```

The repo CI path can stage a sidecar project for you. Leave
`ASTRAL_UNREAL_PROJECT` unset to use `build/unreal-ci-project`, or point it at
an existing project that already has `Plugins/AstralRT`.

## Generate The Sample Project

To create a sidecar UE 5.7 sample project without committing generated files:

```bash
./scripts/create_unreal_sample_project.sh --out /tmp/AstralSample
```

The script creates `AstralSample.uproject`, links the local `AstralRT` plugin,
stages a small mock model payload as UFS content, and writes C++ code that
demonstrates model load, streaming, cancellation, embeddings, packaged content
bytes, Saved cache bytes, and expected failure logging through
`LogAstralSample`. Use `--plugin-mode copy` when the project must be packaged on
a machine without access to the Astral checkout. Real sign-off still requires
packaging that generated project in UE 5.7 and keeping the logs as release
evidence.

For release-candidate sample packaging, use the maintained runner:

```bash
UNREAL_RUNUAT=/opt/Unreal-5.7/Engine/Build/BatchFiles/RunUAT.sh \
  ./scripts/run_unreal_sample_package.sh --platform Linux
```

The runner rebuilds the native ThirdParty package, creates the sidecar sample
with a copied `AstralRT` plugin, runs `RunUAT BuildCookRun`, and prints
`[unreal_sample]` evidence lines for the project, archive, platform, package
mode, and result.

For a packaged runtime smoke, launch the archived binary with
`-NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout`.
The sample GameMode spawns `AAstralSampleActor` on the default engine entry map;
the log should include successful packaged-content and Saved-cache memory model
loads before exit.

To package and immediately launch the sample against local real text and
embedding GGUFs:

```bash
UNREAL_RUNUAT=/opt/Unreal-5.7/Engine/Build/BatchFiles/RunUAT.sh \
  ./scripts/run_unreal_sample_package.sh --platform Linux --run-sample \
  --runtime-log build/unreal-sample-package/runtime.log \
  --sample-model "$PWD/tests/models/Qwen3-0.6B-Q8_0.gguf" \
  --sample-embedding-model "$PWD/tests/models/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --sample-memory-backend mock --sample-media-backend mock
```

## Run Local Automation

Use an installed UnrealEditor or UnrealEditor-Cmd:

```bash
UNREAL_EDITOR=/opt/Unreal-5.7/Engine/Binaries/Linux/UnrealEditor-Cmd \
  ./scripts/run_unreal_ci_tests.sh
```

The runner rebuilds the native package unless told otherwise, stages the plugin,
runs `AstralRT` Automation, and validates that the log/report are non-empty
and free of clear Automation failure markers.

## Run UE 5.7 Containers

Slim image:

```bash
docker pull ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6
./scripts/run_unreal_container_ci.sh --variant slim --skip-native-build
```

Full image:

```bash
docker pull ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce
./scripts/run_unreal_container_ci.sh --variant full --install-cmake
```

Use `--skip-native-build` only when the ThirdParty package was already rebuilt
with the UE Linux SDK and the provenance check has passed in the same workspace.
For release-candidate UE 5.7 evidence, run the full container first with
`--install-cmake` so the native package is compiled by Epic's clang/libc++
toolchain, then run the slim container with `--skip-native-build` to reuse the
staged package.

For compatibility probes, pass the UE version so the runner selects the matching
Epic image tag and bundled Linux SDK preflight:

```bash
./scripts/run_unreal_container_ci.sh --ue-version 5.4 --variant slim --skip-native-build
./scripts/run_unreal_container_ci.sh --ue-version 5.5 --variant slim --skip-native-build
./scripts/run_unreal_container_ci.sh --ue-version 5.6 --variant slim --skip-native-build
```

Use `--pull-timeout` or `ASTRAL_UNREAL_PULL_TIMEOUT` when a first-time Epic
image pull needs more than the default bounded wait. If Docker already has the
image, use `--skip-pull` so the run starts from the cached image digest.

The compatibility container commands are smoke evidence. Release sign-off still
requires the editor matrix below with all supported UE versions.

### Run Real-Model Smoke Probes

The real-model Automation tests stay opt-in so ordinary fast lanes do not depend
on local GGUF files. Download the small fixtures first:

```bash
./tests/model_downloader.sh --preset qwen3-0.6b-q8
./tests/model_downloader.sh --preset qwen3-embed-0.6b-q8
```

Then run the slim container against the package built by the full UE SDK lane:

```bash
ASTRAL_UNREAL_TEST_MODEL=/workspace/astral/tests/models/Qwen3-0.6B-Q8_0.gguf \
ASTRAL_UNREAL_REQUIRE_REAL_GENERATION=1 \
ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE=1 \
ASTRAL_UNREAL_TEST_EMBED_MODEL=/workspace/astral/tests/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING=1 \
./scripts/run_unreal_container_ci.sh --ue-version 5.7 --variant slim --skip-pull --skip-native-build --filter AstralRT.Real
```

The Automation JSON report records `[unreal_generation_smoke]` and
`[unreal_session_lifecycle]` entries for the text model, plus
`[unreal_embedding_probe]` entries for the embedding model. Keep the report and
command line with release evidence; generated model files and container logs
stay out of git.

To cycle the packaged sample through the downloaded small Hugging Face text
fixtures, use the matrix wrapper. It auto-selects known small GGUFs already
present under `tests/models` and reuses the Qwen3 embedding fixture when it is
available:

```bash
./scripts/run_unreal_small_model_matrix.sh \
  --runuat "$UNREAL_RUNUAT" \
  --skip-native-build
```

Use `--dry-run` to inspect the generated `run_unreal_sample_package.sh`
commands before starting Unreal, or `--preset qwen3-0.6b-q8` to run one
candidate.

## Run The Compatibility Matrix

```bash
UNREAL_54_EDITOR=/opt/Unreal-5.4/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_55_EDITOR=/opt/Unreal-5.5/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_56_EDITOR=/opt/Unreal-5.6/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_57_EDITOR=/opt/Unreal-5.7/Engine/Binaries/Linux/UnrealEditor-Cmd \
  ./scripts/run_unreal_compatibility_matrix.sh
```

The matrix is required release evidence for UE 5.4+ support. Do not mark Unreal
support production-ready from the native package build alone.

## Evidence To Keep

For release candidates, keep the logs from:

- `./scripts/run_unreal_container_ci.sh --variant full --install-cmake`
- `./scripts/run_unreal_container_ci.sh --variant slim --skip-native-build`
- `./scripts/run_unreal_compatibility_matrix.sh`

Record those files in `release-evidence.json` using
`docs/release/RELEASE_EVIDENCE_TEMPLATE.json`.
