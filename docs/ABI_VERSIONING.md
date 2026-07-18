# ABI and Versioning Policy (Draft)

Astral exposes a C ABI in `include/astral_rt.h`. The intent is to keep it stable within a major version once `v1.0.0` is tagged.

## Current status

- Astral is **pre-1.0**: the C ABI may change as required.
- Despite pre-1.0 status, changes should still be made conservatively and with clear migration notes.

## Versioning

- The project uses **semantic versioning**: `MAJOR.MINOR.PATCH`.
- `astral_version()` reports the runtime version.

## ABI compatibility rules

### Pre-1.0

- Breaking C ABI changes are allowed, but must:
  - Be explicitly called out in `CHANGELOG.md`
  - Come with a clear upgrade path for Unity/Unreal plugins

### 1.0 and later

- Breaking changes require a MAJOR bump.
- Minor/Patch releases must preserve:
  - Exported symbol names and signatures
  - Struct layout and alignment for all public POD structs
  - Enum numeric values (or provide explicit compatibility strategy)

Layout compatibility is defined per target ABI. In particular, 32-bit ARM
AAPCS aligns 64-bit scalar fields differently from 32-bit x86, so the public
header validates those layouts separately.

## Patterns to keep ABI stable

- Prefer adding new functions over changing existing signatures.
- Prefer adding new fields at the end of structs and gating usage by version checks.
- Avoid embedding pointers to C++ types or STL objects in public structs.
- Never throw exceptions across the C ABI boundary.

## Compatibility testing (recommended)

- Keep Unity/Unreal ABI layout tests updated (e.g., Unity Editor tests and Unreal Automation tests).
- Add a CI job that runs ABI/layout assertions on all Tier-1 platforms when possible.
