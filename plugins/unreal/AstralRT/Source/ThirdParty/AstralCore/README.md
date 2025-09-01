# AstralCore (ThirdParty)

This folder is populated by the CMake preset `unreal-plugin`:

```bash
cd astral
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

It copies:
- `include/astral_rt.h`
- `lib/<Platform>/*` (static library for the target platform)

