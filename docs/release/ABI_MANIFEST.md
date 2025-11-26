# ABI Manifest

Astral's public ABI is the C surface exported through:

- `include/astral_rt.h`
- `include/astral_backend.h`
- `include/astral_backend_plugin.h`

Current runtime version: `0.1.0`.

## Compatibility Rules

- Public structs with a `size` field must keep that field first.
- Existing enum numeric values must not be reused for different meanings.
- Existing exported C function names must not change without a version bump and
  migration note.
- New fields should be appended to public structs and guarded by size checks.
- C++ types, exceptions, STL containers, and ownership transfer by implication
  must not cross the public C ABI.

## Required Local Checks

Run these before packaging a release candidate:

```bash
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests -j --output-on-failure
```

For shared-library symbol review on Linux:

```bash
ctest --preset release-with-tests -R '^gate_shared_exports$' -V
```

The gate compares `build/release-test/libastral_rt.so` against `include/astral_rt.h`
and fails if the shared library exports C++ implementation symbols, dependency
symbols, or any `astral_*` symbol that is not declared with `ASTRAL_API`.

For engine binding layout checks:

- `scripts/package_release.sh` must emit `dist/abi-layout.json`, and
  `dist/checksums.sha256` must cover it.
- `scripts/generate_abi_layout_report.sh --check` must pass. The report covers
  pointer size, `size_t` size, resolved C calling-convention metadata, public
  struct sizes/alignments, and public enum/constant numeric values.
- `ctest --preset release-with-tests -R '^test_abi_invalid_args$' -V`
- `ctest --preset release-with-tests -R '^gate_unreal_header_mirror$' -V`
- Unity ABI tests must pass in the Unity package lane.
- Unreal Automation tests must pass in the UE 5.7 lane.

## Current Status

`gate_shared_exports` now enforces the public runtime symbol surface on Linux,
and `scripts/generate_abi_layout_report.sh` produces release-candidate ABI
layout and constant-value evidence. Unity/Unreal binding layout results from
real engine runners are still required before release sign-off.
