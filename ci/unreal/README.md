# Unreal CI project (minimal)

`ci/unreal/AstralCiUnrealProject/` is a tiny UE project scaffold intended to run the plugin’s Automation tests under `plugins/unreal/AstralRT/Source/AstralRT/Private/Tests/`.

## Setup

1) Create a UE project (or use this scaffold as the project directory).
2) Copy the plugin into the project:

```text
<ProjectRoot>/Plugins/AstralRT/
```

From this repo, the plugin root is `plugins/unreal/AstralRT/`.

3) Build/package the native ThirdParty library into the plugin:

```bash
cd astral
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

## Run Automation tests

- In Editor: `Window → Developer Tools → Session Frontend → Automation`
- CLI (example):
  - `Automation RunTests AstralRT.*`

