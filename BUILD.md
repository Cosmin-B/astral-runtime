# Astral Runtime Build System

## Quick Start

Build the project using CMake presets:

```bash
# Development build (Debug, with tests and benchmarks)
cmake --preset dev
cmake --build build/dev

# Release build (optimized)
cmake --preset release
cmake --build build/release

# Release with tests (for validation)
cmake --preset release-with-tests
cmake --build build/release-test
```

## Build Presets

| Preset | Build Type | Tests | Benchmarks | Description |
|--------|-----------|-------|------------|-------------|
| `dev` | Debug | ON | ON | Development with ASAN/UBSAN |
| `dev-prof` | Debug | ON | ON | Development profiling build (Tracy enabled; requires `external/tracy`) |
| `release` | Release | OFF | OFF | Optimized production build |
| `release-with-tests` | Release | ON | ON | Optimized with validation |
| `release-prof` | Release | OFF | OFF | Release profiling build (Tracy enabled; requires `external/tracy`) |
| `unity-plugin` | Release | OFF | OFF | Unity native plugin build |
| `unity-plugin-prof` | Release | OFF | OFF | Unity plugin profiling build (Tracy enabled; requires `external/tracy`) |
| `unreal-plugin` | Release | OFF | OFF | Unreal Engine plugin build |
| `unreal-plugin-prof` | Release | OFF | OFF | Unreal plugin profiling build (Tracy enabled; requires `external/tracy`) |
| `dev-cuda` | Debug | ON | ON | Development build with CUDA backend enabled (requires CUDA toolkit) |
| `dev-prof-cuda` | Debug | ON | ON | Development profiling build with CUDA backend enabled |
| `release-cuda` | Release | OFF | OFF | Release build with CUDA backend enabled (requires CUDA toolkit) |
| `release-prof-cuda` | Release | OFF | OFF | Release profiling build with CUDA backend enabled |
| `embedded-*` | Release | varies | OFF | Embedded/robotics profiles and cross-compilation presets |

## CUDA Builds (optional)

CUDA is off by default. To build the optional CUDA backend provider:

```bash
cmake --preset dev-cuda
cmake --build build/dev-cuda -j
```

Profiling + CUDA:

```bash
cmake --preset dev-prof-cuda
cmake --build build/dev-prof-cuda -j
```

Tuning knobs (CMake cache variables):
- `ASTRAL_CUDA_ARCHITECTURES` (default: `75;80;86;89;90`)
- Blackwell/Hetzner builds should override `ASTRAL_CUDA_ARCHITECTURES=120a-real` to emit `sm_120a`.
- `ASTRAL_CUDA_FORCE_CUBLAS` / `ASTRAL_CUDA_FORCE_MMQ` (default: `OFF`; “auto” prefers cuBLAS when beneficial)

## Build Targets

- `astral_rt` - Core static library (C++ implementation)
- `astral_rt_shared` - Core shared library (built when `ASTRAL_BUILD_SHARED_LIB=ON`; outputs as `astral_rt` on disk)
- `astral_tests` - Unit test suite (if tests enabled)
- `astral_benchmarks` - Performance benchmarks (if benchmarks enabled)

## Compiler Optimizations

### Release Build Flags

Applied automatically in Release mode:

- **Function Alignment**: `-falign-functions=32` (2-5% I-cache miss reduction)
- **Loop Alignment**: `-falign-loops=32` (better branch prediction)
- **LTO**: Link-time optimization enabled (cross-TU inlining)
- **Platform SIMD**:
  - x86-64: `-mavx2 -mfma` (AVX2 baseline, 2013+)
  - ARM64: `-march=armv8-a+fp+simd` (NEON)
  - ARMv7: `-mfpu=neon -mfloat-abi=hard` (NEON if available)

### Debug Build Flags

Applied automatically in Debug mode:

- **AddressSanitizer**: `-fsanitize=address` (buffer overflows, use-after-free)
- **UndefinedBehaviorSanitizer**: `-fsanitize=undefined` (integer overflow, etc.)
- **Stack Protector**: `-fstack-protector-strong` (buffer overflow detection)
- **Debug Symbols**: `-g3` (full debug info)

## Requirements

- **CMake**: 3.20 or higher
- **C++ Compiler**: C++20 support required
  - GCC 10+ or Clang 12+ recommended
  - MSVC 2019 16.11+ on Windows
- **Platform**: Linux, macOS, Windows (cross-platform)

## Directory Structure

```
astral/
├── CMakeLists.txt          # Main build configuration
├── CMakePresets.json       # Build presets (dev, release, etc.)
├── .clang-format           # Code formatting rules
├── cmake/
│   └── CompilerFlags.cmake # Platform-specific optimizations
├── include/
│   └── astral_rt.h         # C ABI header
├── src/                    # C++ implementation
│   ├── core/               # Init, handles
│   ├── memory/             # Allocators
│   ├── concurrency/        # Lock-free primitives
│   ├── platform/           # OS-specific code
│   ├── backend/            # CPU/GPU backends
│   ├── inference/          # Tokenizer, sampling
│   └── utils/              # UTF-8, logging
├── tests/                  # Unit tests
├── benchmarks/             # Performance tests
└── plugins/
    ├── unity/              # Unity C# bindings
    └── unreal/             # Unreal Engine module
```

## Build Output

### Debug Build
- **Size**: ~1 MB (with debug symbols)
- **Optimizations**: None (-O0)
- **Sanitizers**: ASAN + UBSAN enabled
- **Use Case**: Development, testing, debugging

### Release Build
- **Size**: ~33 KB (stripped, optimized)
- **Optimizations**: Full (-O3 + LTO)
- **Sanitizers**: Disabled
- **Use Case**: Production, Unity/Unreal plugins

## Forbidden Flags

These flags are NEVER used (see CODING_STANDARDS.md):

- `-ffast-math` - Breaks IEEE 754 semantics
- `-fomit-frame-pointer` - Makes debugging impossible
- `-mavx512f` - Not universally available (use runtime dispatch)

## Platform Notes

### Linux
- Links against `pthread` and `dl`
- Supports transparent huge pages (THP)

### macOS
- Uses `mach_vm_allocate` for virtual memory
- CoreFoundation linked automatically

### Windows
- Uses `VirtualAlloc` for virtual memory
- Exports symbols for DLL builds (when `ASTRAL_BUILD_SHARED_LIB=ON`)

When building both static + shared on Windows, the static library output is renamed to avoid clashing with the shared
import library:
- Static: `astral_rt_static.lib`
- Shared: `astral_rt.dll` + `astral_rt.lib` (import)

## Testing

Run tests after building:

```bash
# Build and run tests
cmake --preset dev
cmake --build build/dev
cd build/dev
ctest --output-on-failure

# Or use CMake test preset
cmake --preset dev
cmake --build build/dev
ctest --preset dev
```

## Clean Build

Remove build artifacts:

```bash
rm -rf build/
```

## Troubleshooting

### Compilation Errors

If you encounter unused function warnings, ensure you're using C++20:

```bash
# Check compiler version
g++ --version   # GCC 10+
clang++ --version  # Clang 12+
```

### Missing Dependencies

Ensure all submodules are initialized (if using Git):

```bash
git submodule update --init --recursive
```

Tracy is optional and only required for `*-prof` presets. See `docs/PROFILING_TRACY.md`.

## Packaging (desktop artifacts)

To build, test, and produce zip artifacts under `dist/`:

```bash
./scripts/package_release.sh --preset release-with-tests --unity --unreal
```

This produces:
- Core install zip: `dist/astral-<version>-<os>-<arch>.zip` (headers + libs)
- Unity plugin zip: `dist/astral-<version>-unity-plugin-<os>-<arch>.zip`
- Unreal plugin zip: `dist/astral-<version>-unreal-plugin-<os>-<arch>.zip`

### Platform-Specific Issues

- **Linux**: Install `build-essential` package
- **macOS**: Install Xcode Command Line Tools
- **Windows**: Use Visual Studio 2019+ or MinGW-w64

## References

- **Architecture**: [docs/MASTER_SPEC.md](docs/MASTER_SPEC.md)
- **Coding Standards**: [docs/rules/CODING_STANDARDS.md](docs/rules/CODING_STANDARDS.md)
- **Memory Architecture**: [docs/architecture/MEMORY_ARCHITECTURE.md](docs/architecture/MEMORY_ARCHITECTURE.md)
