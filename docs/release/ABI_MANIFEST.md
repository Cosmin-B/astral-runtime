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
nm -D --defined-only build/release-test/libastral_rt.so | awk '{print $3}' | sort
```

For engine binding layout checks:

- Unity ABI tests must pass in the Unity package lane.
- Unreal Automation tests must pass in the UE 5.7 lane.

## Current Status

This is a manifest and policy document, not a full ABI diff tool. A production
release still needs a generated exported-symbol baseline and struct layout report
checked in or archived with the release candidate.
