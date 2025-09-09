# Contributing to Astral

## Scope and goals

Astral is a performance-first runtime for real-time engines. Contributions should preserve:
- Zero/low-allocation hot paths
- Stable C ABI principles
- Portability across desktop/mobile/embedded profiles

## Build and test

Native dev build:

```bash
cmake --preset dev
cmake --build --preset dev -j
ctest --preset dev -j8
```

Release validation:

```bash
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests -j8
```

## Coding standards

See `docs/rules/CODING_STANDARDS.md`.

Key expectations:
- No STL containers in core hot paths
- No exceptions across the C ABI
- Avoid hidden allocations and syscalls in steady-state inference/streaming

## Pull requests

Please include:
- Motivation and scope (what and why)
- Validation performed (tests/benchmarks/gates)
- Platform(s) tested

Avoid drive-by refactors and unrelated formatting changes; keep PRs focused.

