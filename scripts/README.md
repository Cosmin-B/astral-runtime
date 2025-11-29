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

#### 2. ThreadSanitizer
Race detection for the maintained concurrency and memory tests.

```bash
./scripts/run_tsan.sh
```

**Detects**:
- Data races in concurrency primitives
- Races in memory test paths
- Lock-order failures reported by ThreadSanitizer

**Output**: CTest output for tests labeled `tsan`.

---

#### 3. Valgrind Memcheck (Comprehensive)
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

#### 4. Valgrind Massif (Heap Profiler)
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
./scripts/run_fast_presubmit.sh  # native configure/build/CTest presubmit
./scripts/run_asan.sh          # ~2x slowdown, 1 minute
./scripts/run_tsan.sh          # race checks for concurrency/memory tests
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

## Primitive Perf Capture

Capture concurrency benchmark output, perf counters, symbols, disassembly, and
a target-range comparison into an out-of-repo artifact directory:

```bash
./scripts/run_primitive_perf_capture.sh --pin --items 1000000
```

The script writes to `../astral-perf-runs/<timestamp>` by default. Use
`--out <dir>` for a specific artifact path, `--require-perf` when perf counters
must be present, and `--perf-events <csv>` to tune the counter set.

## ARM64 Hardware Validation

On a native ARM64 Linux host, run the primitive, embedded, perf, and
disassembly evidence batch into an out-of-repo directory:

```bash
./scripts/run_arm64_hardware_validation.sh --jobs 4 --items 1000000
```

This script exits on non-ARM64 hosts unless `--allow-non-arm64` is used for a
script smoke. It intentionally does not power off the machine; remote automation
should copy the evidence directory first, then shut down the cloud instance.

---

## Comment Inventory

Generate a review inventory for maintained comments and documentation prose:

```bash
python3 ./scripts/inventory_comments.py --format tsv > build/comment-inventory.tsv
python3 ./scripts/inventory_comments.py --format review-tsv > build/comment-review.tsv
python3 ./scripts/inventory_comments.py --format summary --fail-orphan-markers
```

Use `review-tsv` for the human pass from the moonshot plan. Fill `decision`
with `keep`, `rewrite`, `delete`, or `issue`; use `issue` for comments that
identify real follow-up work and record the issue tracker ID in the `issue` column.
The inventory output is a local review artifact. Keep it out of git unless a
specific excerpt is promoted into maintained documentation.

---

## Release Packaging (Desktop)

Build + test and emit zip artifacts under `dist/`:

```bash
./scripts/package_release.sh --preset release-with-tests --unity --unreal
```

This is what the CI “v0.1 desktop artifacts” job uses.

For a release candidate with external gate evidence already collected, copy and
validate that manifest into the artifact set. Use `pre-sign` before the
protected signing workflow has produced `checksums.sha256.asc`; use `complete`
after signatures exist:

```bash
./scripts/package_release.sh --preset release-with-tests --unity --unreal --evidence path/to/release-evidence.json --evidence-phase pre-sign
```

The packaging script also writes release governance metadata:

- `dist/abi-layout.json`
- `dist/dependency-manifest.json`
- `dist/release-sbom.spdx.json`
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

Protected release signing is handled by the manual `release-sign` GitHub Actions
workflow. It signs checksum manifests from an existing artifact run in the
`release` environment after release-note, dependency-pin, and release-evidence
validation. The downloaded artifact set must contain `release-evidence.json`
next to `checksums.sha256`; the workflow validates it before importing the
release signing key. The workflow also verifies each downloaded checksum
manifest before importing the signing key.

Verification:

```bash
./scripts/validate_release_artifacts.sh --dist dist --expect-unity --expect-unreal --require-signature
(cd dist && sha256sum -c checksums.sha256)
```

When `--require-signature` is set, the artifact verifier cryptographically
checks `checksums.sha256.asc` with `gpg --verify` or
`checksums.sha256.minisig` with `minisign -Vm`.

`package_release.sh` runs the artifact verifier automatically before reporting success.

Release candidates also need an evidence manifest that points at the logs and
artifacts from every required lane. Start from
`docs/release/RELEASE_EVIDENCE_TEMPLATE.json`, write the filled manifest next to
the release artifacts, and validate it:

```bash
python3 ./scripts/validate_release_evidence.py dist/release-evidence.json --base-dir dist
```

The manifest records evidence. It does not replace the Unreal, Unity, CUDA,
multimodal, signing, or Windows runner commands.

Validate the template shape during maintenance:

```bash
./scripts/validate_release_notes.sh --allow-placeholders docs/release/RELEASE_NOTES_TEMPLATE.md
```

Validate filled release notes before publishing a release candidate:

```bash
./scripts/validate_release_notes.sh dist/release-notes.md
```

## Unreal Automation

Build the Unreal ThirdParty package, then run the plugin Automation tests through Unreal. This is a local developer check when one editor is configured, and release-candidate evidence when the UE 5.4+ matrix is run with every supported editor.

Check whether the current machine has enough access for real Unreal validation before starting a long run:

```bash
./scripts/check_unreal_validation_access.sh --check-registry
```

The access check does not pull images or launch Unreal. It reports whether the
pinned UE 5.7 full/slim containers are cached or readable from Epic GHCR, and
whether `UNREAL_54_EDITOR`, `UNREAL_55_EDITOR`, `UNREAL_56_EDITOR`, and
`UNREAL_57_EDITOR` point to executable editors.

For Epic container runs, `run_unreal_container_ci.sh` defaults to UE 5.7. Use
the full image with `--install-cmake` when the ThirdParty package must be
rebuilt inside Epic's clang/libc++ Linux SDK, then use the slim image with
`--skip-native-build` to reuse that staged package. Use `--ue-version` for
compatibility smoke runs so the script selects the matching image tag and Linux
SDK toolchain preflight:

```bash
./scripts/run_unreal_container_ci.sh --variant full --install-cmake
./scripts/run_unreal_container_ci.sh --variant slim --skip-native-build
./scripts/run_unreal_container_ci.sh --ue-version 5.4 --variant slim --skip-native-build
./scripts/run_unreal_container_ci.sh --ue-version 5.5 --variant slim --skip-native-build
./scripts/run_unreal_container_ci.sh --ue-version 5.6 --variant slim --skip-native-build
```

Image manifest and pull operations are bounded by `--pull-timeout` or
`ASTRAL_UNREAL_PULL_TIMEOUT` so a slow Epic GHCR layer cannot hang a validation
session indefinitely. If a large image has already been cached outside the
runner, pass `--skip-pull` to avoid registry access.

```bash
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
UNREAL_EDITOR=/path/to/UnrealEditor-Cmd ./scripts/run_unreal_ci_tests.sh
```

By default the runner stages a sidecar project under `build/unreal-ci-project/` and writes logs to `build/unreal-ci-results/`. Set `ASTRAL_UNREAL_PROJECT` to run against an existing project with `Plugins/AstralRT` already installed.

For the UE 5.4+ compatibility lane, set one editor per version and run the matrix wrapper:

```bash
UNREAL_54_EDITOR=/opt/Unreal-5.4/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_55_EDITOR=/opt/Unreal-5.5/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_56_EDITOR=/opt/Unreal-5.6/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_57_EDITOR=/opt/Unreal-5.7/Engine/Binaries/Linux/UnrealEditor-Cmd \
  ./scripts/run_unreal_compatibility_matrix.sh
```

Each version writes separate reports under `build/unreal-ci-results/ue-<version>/`. Release-candidate runs should provide all four editors; `--allow-missing` is only for local discovery.

`run_unreal_ci_tests.sh` validates the Unreal log and Automation report after
the editor exits. A release-candidate run must leave a non-empty report and no
AstralRT Automation failure markers.

For the UE 5.7 sample package lane, generate and package the sidecar sample
with a copied plugin:

```bash
UNREAL_RUNUAT=/opt/Unreal-5.7/Engine/Build/BatchFiles/RunUAT.sh \
  ./scripts/run_unreal_sample_package.sh --platform Linux
```

## Required Release Gates

`run_release_required_gates.sh` is the hard release-candidate lane. It runs native release tests, ASAN/UBSAN, TSan, CUDA release parity/e2e in auto/cuBLAS/MMQ modes, the real MTMD media gate, UE 5.7 full/slim container Automation, Unreal 5.4+ Automation compatibility, UE 5.7 sample packaging, and Unity EditMode ABI tests.

```bash
ASTRAL_TEST_VISION_MODEL=/models/vision.gguf \
ASTRAL_TEST_VISION_MEDIA=/models/mmproj-vision.gguf \
ASTRAL_TEST_AUDIO_MODEL=/models/audio.gguf \
ASTRAL_TEST_AUDIO_MEDIA=/models/mmproj-audio.gguf \
UNREAL_54_EDITOR=/opt/Unreal-5.4/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_55_EDITOR=/opt/Unreal-5.5/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_56_EDITOR=/opt/Unreal-5.6/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_57_EDITOR=/opt/Unreal-5.7/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_RUNUAT=/opt/Unreal-5.7/Engine/Build/BatchFiles/RunUAT.sh \
UNITY_EDITOR=/opt/Unity/Editor/Unity \
  ./scripts/run_release_required_gates.sh --cuda-arch native --cuda-strict --mtmd-bench
```

The `--skip-sanitizers`, `--skip-engine`, `--skip-unreal`, and `--skip-unity`
flags are for partial local diagnosis only. A release candidate should not use
them. The Unreal lane starts with `check_unreal_validation_access.sh --check-registry`,
then runs the full UE 5.7 container with `--install-cmake`, the slim UE 5.7
container with `--skip-native-build`, then the editor matrix and sample package.

For a fast release-candidate preflight that prints the required lanes and checks
that the release environment variables are present without starting builds or
engine jobs:

```bash
ASTRAL_TEST_VISION_MODEL=/models/vision.gguf \
ASTRAL_TEST_VISION_MEDIA=/models/mmproj-vision.gguf \
ASTRAL_TEST_AUDIO_MODEL=/models/audio.gguf \
ASTRAL_TEST_AUDIO_MEDIA=/models/mmproj-audio.gguf \
UNREAL_54_EDITOR=/opt/Unreal-5.4/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_55_EDITOR=/opt/Unreal-5.5/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_56_EDITOR=/opt/Unreal-5.6/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_57_EDITOR=/opt/Unreal-5.7/Engine/Binaries/Linux/UnrealEditor-Cmd \
UNREAL_RUNUAT=/opt/Unreal-5.7/Engine/Build/BatchFiles/RunUAT.sh \
UNITY_EDITOR=/opt/Unity/Editor/Unity \
  ./scripts/run_release_required_gates.sh --print-plan --cuda-arch native --cuda-strict --mtmd-bench
```

The CUDA part uses release-with-tests CUDA presets and requires a real CUDA runner.
Pass `--cuda-arch` with the deployed architecture list that the release evidence
will claim; direct `run_cuda_parity_matrix.sh --preset-set release` runs follow
the same rule through `--arch`.

```bash
ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 \
  ./scripts/run_cuda_parity_matrix.sh --preset-set release --arch native --strict
```

`run_cuda_parity.sh` and `run_cuda_parity_matrix.sh` require those two
environment variables by default. `--allow-probes` is only for local CUDA build
discovery where the real inference/e2e sections are intentionally disabled.

The MTMD part fails before CTest if any required model/projector fixture is
missing. Use `--check-fixtures` to verify paths and minimum sizes without
starting configure, build, or CTest:

```bash
./scripts/validate_mtmd_fixture_manifest.py scripts/mtmd_fixture_manifest_lfm25.json
./scripts/hf_gguf_download_lfm25_all.sh --out tests/models/hf-lfm25
./scripts/run_multimodal_validation.sh \
  --fixture-manifest scripts/mtmd_fixture_manifest_lfm25.json \
  --fixture-dir tests/models/hf-lfm25 \
  --check-fixtures
./scripts/run_multimodal_validation.sh \
  --fixture-manifest scripts/mtmd_fixture_manifest_lfm25.json \
  --fixture-dir tests/models/hf-lfm25 \
  --bench
```

The fixture manifest pins the Hugging Face repo revisions, license metadata, and
required GGUF model/projector filenames. The validation runner can resolve its
vision/audio paths from that manifest plus the download directory; explicit
`--vision-*` and `--audio-*` paths still override manifest defaults for local
experiments with newer small Hugging Face models. Do not replace release
manifests with floating `main` downloads.

The HF GGUF matrix is fail-hard by default: any `[bench] FAILED` row makes
`run_hf_bench_matrix.sh` exit non-zero. Use `--allow-failures` only for local
investigation, not for release evidence.

`hetzner_watchdog.sh` can keep long-running HF downloads and wait+bench jobs
alive on a remote runner. Use `--dry-run` to inspect the exact commands before
letting it start background jobs.

For Unreal packaged-sample smoke runs across the downloaded small text fixtures,
use:

```bash
./scripts/run_unreal_small_model_matrix.sh --list
./scripts/run_unreal_small_model_matrix.sh --dry-run --preset qwen3-0.6b-q8
./scripts/run_unreal_small_model_matrix.sh --download-missing --preset gemma3-1b-it-q4km --dry-run
./scripts/run_unreal_small_model_matrix.sh --runuat "$UNREAL_RUNUAT" --skip-native-build
```

The runner is intentionally a thin wrapper around
`run_unreal_sample_package.sh`: it chooses the local GGUF paths, creates
per-model output directories, keeps runtime logs out of the repo, and validates
each runtime log after a real sample launch. Use `--skip-runtime-validation`
only for local bring-up when the sample is expected to be incomplete. Use
`--download-missing` to fetch missing known Gemma/Qwen/SmolLM presets through
`tests/model_downloader.sh`; downloaded GGUFs stay under ignored
`tests/models/` paths.

```bash
./scripts/validate_unreal_sample_runtime_log.py \
  --log build/unreal-small-model-matrix/Qwen3-0.6B-Q8_0/runtime.log \
  --expect-model Qwen3-0.6B-Q8_0.gguf \
  --expect-embedding-model Qwen3-Embedding-0.6B-Q8_0.gguf
```

When RunUAT is only available inside a cached Epic image, use the container
wrapper:

```bash
./scripts/run_unreal_small_model_matrix_container.sh --skip-pull --print-command --preset qwen3-0.6b-q8
./scripts/run_unreal_small_model_matrix_container.sh --skip-pull --variant slim --preset qwen3-0.6b-q8 --dry-run
```

The container wrapper mounts the repo at `/workspace/astral`, detects the engine
root in the image, and then invokes the same matrix runner. It reuses the staged
ThirdParty package by default; use `--build-native` for a UE Linux SDK rebuild.

## Windows Large Pages

Run the Windows large-page validation script twice on a Windows host:

```powershell
pwsh -File .\scripts\run_windows_large_page_validation.ps1 -ExpectFallback
```

Then grant the test account `SeLockMemoryPrivilege`, start a fresh elevated
shell so the token contains the new privilege, and run:

```powershell
pwsh -File .\scripts\run_windows_large_page_validation.ps1 -ExpectLargePages
```

The script writes `build/windows-large-pages/windows-large-pages.log` and
`windows-privileges.txt`. Attach both logs to the release evidence manifest
under the `windows_large_pages` lane.

## Unity EditMode Results

`run_unity_ci_tests.sh` validates `editmode-results.xml` after Unity exits. The
XML must be well-formed, report zero failures, and include at least one passing
test case.

## Interpreting Results

### PASS Criteria

1. **No Memory Leaks**
   - Valgrind: "definitely lost: 0 bytes"
   - ASAN: Exit code 0

2. **Allocation-Gated Hot Paths**
   - Custom allocator: "Decode hot path: no tracked allocations"
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

**Performance Failures**:
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
- `gate_embedded_presets`
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

### run_tsan.sh
- Creates separate build directory: `build/tsan`
- Compiles with: `-fsanitize=thread`
- Builds `test_concurrency_tsan` and `test_memory_tsan`.
- Runs tests labeled `tsan` through CTest.
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
- [ ] Add automated regression testing
- [ ] Add HTML report generation
- [ ] Add benchmark comparison (before/after)

---

## References

**Astral Documentation**:
- `tests/MEMORY_VALIDATION.md` - Current memory validation gates
- `/home/user/workspace/astral/docs/release/RELEASE_ACCEPTANCE_MATRIX.md` - Release acceptance evidence
- `/home/user/workspace/astral/docs/FEATURE_MATRIX.md` - Supported feature surface and required evidence
- `/home/user/workspace/astral/docs/rules/CODING_STANDARDS.md` - Memory rules

**Valgrind Documentation**:
- https://valgrind.org/docs/manual/manual.html
- https://valgrind.org/docs/manual/mc-manual.html (Memcheck)
- https://valgrind.org/docs/manual/ms-manual.html (Massif)

**Sanitizer Documentation**:
- https://github.com/google/sanitizers/wiki/AddressSanitizer
- https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual
- https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html

---

**Last Updated**: 2025-10-21
**Maintained By**: Astral Team
