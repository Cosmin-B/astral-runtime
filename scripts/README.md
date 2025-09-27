# Astral Validation Scripts

This directory contains validation and testing scripts for the Astral runtime.

## Memory Validation Scripts

### Prerequisites

Install Valgrind:
```bash
# Ubuntu/Debian
sudo apt-get install valgrind

# RHEL/CentOS
sudo yum install valgrind

# macOS
brew install valgrind
```

### Usage

#### 1. AddressSanitizer (Recommended First)
**Fastest validation tool** - runs at near-native speed with 2x slowdown.

```bash
./scripts/run_asan.sh
```

**Detects**:
- Buffer overflows
- Use-after-free
- Stack corruption
- Integer overflows
- Undefined behavior

**Output**: CTest output for `gate_allocations` and `gate_rss_cap`.

---

#### 2. Valgrind Memcheck (Comprehensive)
**Most thorough leak detection** - runs 10-20x slower.

```bash
./scripts/run_valgrind.sh
```

**Detects**:
- Memory leaks (definitely lost, possibly lost, still reachable)
- Invalid reads/writes
- Use-after-free
- Uninitialized values
- Double frees

**Output**: `valgrind_memcheck.log`

**View Results**:
```bash
# Quick summary
grep "ERROR SUMMARY\|LEAK SUMMARY" valgrind_memcheck.log

# Detailed leak report
grep -A 20 "LEAK SUMMARY" valgrind_memcheck.log

# Full report
cat valgrind_memcheck.log
```

---

#### 3. Valgrind Massif (Heap Profiler)
**Heap growth analysis** - critical for detecting allocations in hot paths.

```bash
./scripts/run_massif.sh
```

**Detects**:
- Peak memory usage
- Heap growth over time
- Allocation hotspots

**Output**: `massif.out`, `massif_report.txt`

**View Results**:
```bash
# Quick summary
grep "peak" massif_report.txt

# Full timeline
cat massif_report.txt

# Visual graph (if ms_print supports it)
ms_print massif.out
```

**Critical Check**:
The Massif target is `gate_allocations`; growth after steady-state setup needs
review before a release candidate.

---

## Running All Validations

```bash
# Run in order (fastest to slowest):
./scripts/run_asan.sh          # ~2x slowdown, 1 minute
./scripts/run_valgrind.sh      # ~15x slowdown, 5-10 minutes
./scripts/run_massif.sh        # ~20x slowdown, 10-15 minutes
```

---

## Tracy Profiling (Optional)

Tracy is integrated as an optional git submodule and is only required for `*-prof` presets.

```bash
./scripts/setup_tracy_submodule.sh
cmake --preset dev-prof
cmake --build --preset dev-prof -j
ctest --preset dev-prof -j8
```

See `docs/PROFILING_TRACY.md` for capture notes and what zones/plots are instrumented.

### Quick capture helper

Build a profiling preset and run a small workload so you can attach the Tracy UI:

```bash
./scripts/run_tracy_capture.sh --preset dev-prof
```

---

## Release Packaging (Desktop)

Build + test and emit zip artifacts under `dist/`:

```bash
./scripts/package_release.sh --preset release-with-tests --unity --unreal
```

This is what the CI “v0.1 desktop artifacts” job uses.

The packaging script also writes release governance metadata:

- `dist/abi-layout.json`
- `dist/dependency-manifest.json`
- `dist/checksums.sha256`

To regenerate metadata for an existing output directory:

```bash
./scripts/generate_abi_layout_report.sh --out dist/abi-layout.json
./scripts/generate_release_metadata.sh dist
```

Check committed dependency pins against the working tree:

```bash
./scripts/validate_dependency_pins.sh
```

To sign the artifact set, sign the checksum file after metadata generation:

```bash
ASTRAL_RELEASE_SIGN_KEY=release@example.com ./scripts/sign_release_artifacts.sh --out-dir dist
```

`package_release.sh` can run the same signing step:

```bash
./scripts/package_release.sh --preset release-with-tests --unity --unreal --sign --sign-key release@example.com
```

Verification:

```bash
gpg --verify dist/checksums.sha256.asc dist/checksums.sha256
(cd dist && sha256sum -c checksums.sha256)
```

Validate release notes before publishing a release candidate:

```bash
./scripts/validate_release_notes.sh docs/release/RELEASE_NOTES_TEMPLATE.md
```

## Unreal Automation (Optional)

Build the Unreal ThirdParty package, then run the plugin Automation tests through Unreal:

```bash
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
UNREAL_EDITOR=/path/to/UnrealEditor-Cmd ./scripts/run_unreal_ci_tests.sh
```

By default the runner stages a sidecar project under `build/unreal-ci-project/` and writes logs to `build/unreal-ci-results/`. Set `ASTRAL_UNREAL_PROJECT` to run against an existing project with `Plugins/AstralRT` already installed.

## Required Release Gates

`run_release_required_gates.sh` is the hard release-candidate lane. It runs native release tests, CUDA release parity/e2e in auto/cuBLAS/MMQ modes, and the real MTMD media gate.

```bash
ASTRAL_TEST_VISION_MODEL=/models/vision.gguf \
ASTRAL_TEST_VISION_MEDIA=/models/mmproj-vision.gguf \
ASTRAL_TEST_AUDIO_MODEL=/models/audio.gguf \
ASTRAL_TEST_AUDIO_MEDIA=/models/mmproj-audio.gguf \
  ./scripts/run_release_required_gates.sh --cuda-strict --mtmd-bench
```

The CUDA part uses release-with-tests CUDA presets and requires a real CUDA runner:

```bash
ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 \
  ./scripts/run_cuda_parity_matrix.sh --preset-set release --strict
```

The MTMD part fails before CTest if any required model/projector fixture is missing:

```bash
./scripts/run_multimodal_validation.sh --bench
```

## Interpreting Results

### PASS Criteria (per MASTER_SPEC § Performance Targets)

1. **No Memory Leaks**
   - Valgrind: "definitely lost: 0 bytes"
   - ASAN: Exit code 0

2. **Zero Hot Path Allocations**
   - Custom allocator: "Decode hot path: ZERO allocations"
   - Massif: Flat heap during decode iterations

3. **No Memory Errors**
   - Valgrind: "ERROR SUMMARY: 0 errors"
   - ASAN: No error output

4. **No Undefined Behavior**
   - UBSAN: No warnings

### FAIL Indicators

**Immediate Failures**:
- ASAN detects heap-buffer-overflow
- Valgrind reports "definitely lost" bytes
- Test reports non-zero allocations in hot path

**Performance Failures** (violate MASTER_SPEC):
- Massif shows heap growth during decode loop
- More than 10 syscalls in hot path (`strace` check)
- Allocations detected in decode/stream/sample loops

---

## Embedded / Edge Validation

These helpers target the “embedded/robotics” profile and are designed to be runnable on a developer machine (host build) as well as in CI (cross-compile + QEMU smoke).

### run_embedded_smoke.sh

Builds the `embedded-x86_64` preset and runs the embedded CLI sample.

```bash
./scripts/run_embedded_smoke.sh
```

### run_embedded_validation.sh

Runs release validation gates and then runs the embedded smoke:
- `gate_allocations`
- `gate_io_syscalls`
- `gate_rss_cap` (Linux-only; configurable via `ASTRAL_RSS_MAX_MB`)

```bash
./scripts/run_embedded_validation.sh
```

## Troubleshooting

### Valgrind Not Found
```bash
# Install Valgrind
sudo apt-get install valgrind

# Verify installation
valgrind --version
```

### ASAN/UBSAN Errors During Build
ASAN is disabled by default for Valgrind compatibility. The `run_asan.sh` script builds with ASAN flags in a separate directory (`build/asan`).

### Test Binary Not Found
```bash
# Build test first
cmake --preset dev
cmake --build --preset dev --target gate_allocations -j
```

### Slow Execution
This is expected with Valgrind (10-20x slowdown). For faster validation:
1. Run ASAN first (only 2x slowdown)
2. Run Valgrind only if ASAN passes
3. Use release builds for benchmarks (not validation)

---

## Script Details

### run_asan.sh
- Creates separate build directory: `build/asan`
- Compiles with: `-fsanitize=address -fsanitize=undefined`
- Runs `gate_allocations` and `gate_rss_cap` through CTest.
- Exit code: 0 (pass), non-zero (fail)

### run_valgrind.sh
- Uses existing build: `build/dev`
- Runs `gate_allocations`
- Runs with: `--tool=memcheck --leak-check=full --track-origins=yes`
- Log file: `valgrind_memcheck.log`
- Reports: Errors, leaks, invalid accesses

### run_massif.sh
- Uses existing build: `build/dev`
- Runs `gate_allocations`
- Runs with: `--tool=massif --detailed-freq=1`
- Output files: `massif.out`, `massif_report.txt`
- Reports: Peak heap, heap timeline, allocation hotspots

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Memory Validation (Example)

on: [push, pull_request]

jobs:
  asan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: sudo apt-get install -y build-essential cmake
      - name: Run ASAN
        run: ./scripts/run_asan.sh

  valgrind:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: sudo apt-get install -y valgrind cmake
      - name: Run Valgrind
        run: ./scripts/run_valgrind.sh
```

Note: This is an illustrative snippet. The repo’s actual CI is in `.github/workflows/ci.yml`.

---

## Future Enhancements

- [ ] Add `strace` wrapper to count syscalls in hot path
- [ ] Add `perf stat` wrapper for CPU metrics
- [ ] Add automated regression testing
- [ ] Add HTML report generation
- [ ] Add benchmark comparison (before/after)
- [ ] Add thread sanitizer (TSan) for concurrency validation

---

## References

**Astral Documentation**:
- `tests/MEMORY_VALIDATION.md` - Current memory validation gates
- `/home/user/workspace/astral/docs/MASTER_SPEC.md` - Performance targets
- `/home/user/workspace/astral/docs/rules/CODING_STANDARDS.md` - Memory rules

**Valgrind Documentation**:
- https://valgrind.org/docs/manual/manual.html
- https://valgrind.org/docs/manual/mc-manual.html (Memcheck)
- https://valgrind.org/docs/manual/ms-manual.html (Massif)

**Sanitizer Documentation**:
- https://github.com/google/sanitizers/wiki/AddressSanitizer
- https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html

---

**Last Updated**: 2025-10-21
**Maintained By**: Astral Team
