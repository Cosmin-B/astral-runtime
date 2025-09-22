# Dependency Manifest

Astral release artifacts must be traceable to exact source and dependency pins.

## Current Pins

| Dependency | Pin source | Current value |
|---|---|---|
| Astral runtime | `CMakeLists.txt`, `include/astral_rt.h` | `0.1.0` |
| llama.cpp | Git submodule `external/llama.cpp` | `b9025`, `eff06702b2a52e1020ea009ebd86cb9f5acabab5` |
| Tracy | Git submodule `external/tracy` | `a602127eddb60825ac91e726986c12955e9a0082` |
| Unity package | `plugins/unity/package.json` | `com.astral.runtime` `0.1.0`, Unity `2021.3` |
| Unreal package | `plugins/unreal/AstralRT/AstralRT.uplugin` | UE 5.4+ compatibility floor; UE 5.7 is the production target |

## Generated Manifest

Generate the release dependency manifest and artifact checksums after packaging:

```bash
./scripts/generate_release_metadata.sh dist
```

The script writes:

- `dist/dependency-manifest.json`
- `dist/checksums.sha256`

The generated manifest records the Astral version, source commit, dirty state,
submodule commits, and engine package versions. `checksums.sha256` covers files
already present in the output directory, excluding the generated manifest files.
