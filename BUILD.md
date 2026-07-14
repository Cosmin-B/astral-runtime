# Building Astral

## Requirements

- CMake 3.20+
- GCC 11+, Clang 13+, or MSVC 2022+
- A C++17 toolchain
- Git submodules initialized with `git submodule update --init --recursive`

CUDA, Tracy, Unity, and Unreal are optional and only required for their
corresponding build or validation lanes.

## Preset Workflow

Use CMake configure, build, and test presets from the repository root:

```bash
cmake --preset dev
cmake --build --preset dev -j
ctest --preset dev --output-on-failure

cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests --output-on-failure
```

The main presets are:

| Preset | Purpose |
| --- | --- |
| `dev` | Debug runtime, tests, and benchmarks |
| `release` | Optimized static and shared runtime libraries |
| `release-with-tests` | Optimized runtime, tests, benchmarks, and release gates |
| `dev-prof`, `release-prof` | Tracy-enabled native profiling builds |
| `dev-prof-micro`, `release-prof-micro` | Tracy builds with fine-grained micro zones |
| `dev-cuda`, `release-cuda` | CUDA-enabled builds |
| `unity-plugin` | Unity native plugin output |
| `unreal-plugin` | Unreal ThirdParty header and library staging |
| `embedded-*` | Embedded x86-64, ARM64, and ARMv7 profiles |

Profiling variants also exist for the engine plugin presets. Inspect the
current list with `cmake --list-presets`.

Release desktop presets build both `astral_rt` (static) and
`astral_rt_shared` (shared). On Windows the static output is
`astral_rt_static.lib`; the shared output is `astral_rt.dll` with
`astral_rt.lib` as its import library.

## Optional CUDA Build

CUDA is disabled by default:

```bash
cmake --preset dev-cuda
cmake --build --preset dev-cuda -j
```

Relevant cache options include:

- `ASTRAL_CUDA_ARCHITECTURES`
- `ASTRAL_CUDA_FORCE_CUBLAS`
- `ASTRAL_CUDA_FORCE_MMQ`

CUDA source availability is not release sign-off. Run the real-device matrix
described in [docs/CUDA_PARITY.md](docs/CUDA_PARITY.md).

## Sanitizers

Debug presets do not enable sanitizers automatically. Use the dedicated
scripts:

```bash
./scripts/run_asan.sh
./scripts/run_tsan.sh
```

AddressSanitizer and UndefinedBehaviorSanitizer can also be enabled on a custom
debug configuration with `-DASTRAL_ENABLE_ASAN=ON`.

## Profiling

Tracy is required only by `*-prof` presets:

```bash
./scripts/setup_tracy_submodule.sh
cmake --preset release-prof
cmake --build --preset release-prof -j
```

See [docs/PROFILING_TRACY.md](docs/PROFILING_TRACY.md) and
[docs/api/HOT_PATH_PROFILING.md](docs/api/HOT_PATH_PROFILING.md) for the native
and engine profiling boundaries.

## Packaging

Build, test, and package desktop artifacts under `dist/`:

```bash
./scripts/package_release.sh \
  --preset release-with-tests \
  --unity \
  --unreal
```

The packaging script produces the core archive and optional Unity and Unreal
archives. Release metadata, dependency pins, notices, checksums, and signature
requirements are defined under [docs/release](docs/release/).

## Platform Notes

- Linux links the threaded runtime against `pthread` and `dl` where required.
- macOS uses Mach virtual-memory APIs.
- Windows uses `VirtualAlloc` and exports the shared-library ABI when enabled.
- Embedded presets deliberately disable selected desktop facilities; see
  [docs/EMBEDDED_PROFILE.md](docs/EMBEDDED_PROFILE.md).

## Troubleshooting

Reinitialize dependencies after changing branches:

```bash
git submodule update --init --recursive
```

Start from a fresh out-of-tree build if cached compiler or option state is in
doubt:

```bash
rm -rf build/dev
cmake --preset dev
```

For subsystem-specific requirements, use the
[documentation index](docs/README.md) rather than inferring support from a
successful compile alone.
