# Unity CI project (minimal)

`ci/unity/AstralCiUnityProject/` is a tiny Unity project intended to run the package tests located in `plugins/unity/Tests/`.

## Setup

- Open `ci/unity/AstralCiUnityProject/` in Unity 6000.0 or newer. The checked-in
  CI project currently pins `6000.0.57f1`.
- Ensure the Astral native plugin binaries are present under `plugins/unity/Runtime/Plugins/<arch>/`.
  - On desktop, you can build/package them from the repo:
    - `cmake --preset unity-plugin`
    - `cmake --build --preset unity-plugin -j`

## Run tests (batch)

This repo includes `scripts/run_unity_ci_tests.sh` which runs EditMode tests in batchmode. You must provide a Unity editor path.
The script builds the native Unity plugin first and sets `ASTRAL_UNITY_REQUIRE_NATIVE=1`, so ABI tests fail instead of skipping when `libastral_rt` is missing or missing an entry point.

Example:

```bash
cd astral
UNITY_EDITOR="/path/to/Unity" ./scripts/run_unity_ci_tests.sh
```

## GameCI v4

The manual workflow `.github/workflows/unity-gameci.yml` runs the same EditMode
ABI project through `game-ci/unity-test-runner@v4` on Linux. It requires Unity
license secrets as repository or environment secrets:

- `UNITY_LICENSE`
- `UNITY_EMAIL`
- `UNITY_PASSWORD`

Run it from GitHub Actions with a Unity 6000.0 editor and the native plugin
build enabled. The workflow builds `cmake --preset unity-plugin` before invoking
GameCI so the package tests load the current native library.

For local Linux container validation, use:

```bash
./scripts/run_unity_gameci_tests.sh
```

The script resolves the checked-in Unity version and uses
`unityci/editor:ubuntu-6000.0.57f1-base-3.2.2` unless `--image` or
`ASTRAL_UNITY_GAMECI_IMAGE` overrides it.
