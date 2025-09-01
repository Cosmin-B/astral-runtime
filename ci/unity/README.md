# Unity CI project (minimal)

`ci/unity/AstralCiUnityProject/` is a tiny Unity project intended to run the package tests located in `plugins/unity/Tests/`.

## Setup

- Open `ci/unity/AstralCiUnityProject/` in Unity (2021.3+).
- Ensure the Astral native plugin binaries are present under `plugins/unity/Runtime/Plugins/<arch>/`.
  - On desktop, you can build/package them from the repo:
    - `cmake --preset unity-plugin`
    - `cmake --build --preset unity-plugin -j`

## Run tests (batch)

This repo includes `scripts/run_unity_ci_tests.sh` which runs EditMode tests in batchmode. You must provide a Unity editor path.

Example:

```bash
cd astral
UNITY_EDITOR="/path/to/Unity" ./scripts/run_unity_ci_tests.sh
```

