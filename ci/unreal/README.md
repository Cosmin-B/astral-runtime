# Unreal CI project

`ci/unreal/AstralCiUnrealProject/` is a tiny UE project scaffold for the plugin Automation tests under `plugins/unreal/AstralRT/Source/AstralRT/Private/Tests/`.

The production target is Unreal Engine 5.7 with UE 5.4+ compatibility. On Linux, the expected 5.7 toolchain is clang 20.1.8, matching Epic's 5.7 development images such as `ghcr.io/epicgames/unreal-engine:dev-5.7.4` and `ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4` when access is configured.

## Setup

Build/package the native ThirdParty library into the plugin first:

```bash
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

The CI runner stages a sidecar project under `build/unreal-ci-project/` and copies the plugin there, so it does not modify tracked files. To use your own project, copy the plugin into:

```text
<ProjectRoot>/Plugins/AstralRT/
```

From this repo, the plugin root is `plugins/unreal/AstralRT/`.

## Run Automation tests

```bash
UNREAL_EDITOR=/path/to/UnrealEditor-Cmd ./scripts/run_unreal_ci_tests.sh
```

Useful overrides:

- `UNREAL_TEST_FILTER=AstralRT.Mock*` narrows the Automation filter.
- `ASTRAL_UNREAL_PROJECT=/path/to/project` runs against an existing project that already has `Plugins/AstralRT` installed.

Manual Editor path: `Window > Developer Tools > Session Frontend > Automation`, then run `AstralRT.*`.

## UE 5.7 container runner

For the production UE 5.7 lane, run the same staged project through Epic's Linux development images:

```bash
docker pull ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6
./scripts/run_unreal_container_ci.sh --variant slim
```

Full-image validation uses:

```bash
docker pull ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce
./scripts/run_unreal_container_ci.sh --variant full
```

The wrapper rebuilds the native Unreal ThirdParty package, prints the container image reference, Unreal `Build.version`, Linux SDK metadata, and `clang --version`, then delegates to `scripts/run_unreal_ci_tests.sh`. Use `--skip-native-build` only when validating an already-built plugin package.
