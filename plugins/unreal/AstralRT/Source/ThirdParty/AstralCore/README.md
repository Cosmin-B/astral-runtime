# AstralCore (ThirdParty)

This folder is populated by the CMake preset `unreal-plugin`:

```bash
cd astral-runtime
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

It copies:
- `include/astral_rt.h`
- `lib/<Platform>/*` (static library for the target platform)

The build checks the staged header and platform library against the current
source header and native target by SHA-256 before the package target succeeds.
