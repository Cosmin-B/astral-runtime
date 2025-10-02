# Astral Tests

This directory contains the native unit tests, integration tests, and hard validation gates for Astral.

## Running

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

## Integration model

`test_integration` runs real end-to-end inference when a GGUF model is available.

Model selection order:
1. `ASTRAL_TEST_MODEL=/abs/path/to/model.gguf`
2. `astral/tests/models/gpt2.Q2_K.gguf` (default downloader output, if present)
3. `astral/tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf` (legacy filename, if present)
4. `astral/tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (legacy filename, if present)

If no model is found, the test prints a `[SKIP]` message and continues. If a model path is found but the load fails, the test fails.

### Downloading a small model

```bash
bash astral/tests/model_downloader.sh
export ASTRAL_TEST_MODEL="$PWD/astral/tests/models/gpt2.Q2_K.gguf"
```

The downloader supports overrides via `--url/--file/--min-bytes` and env vars (`ASTRAL_TEST_MODEL_URL`, `ASTRAL_TEST_MODEL_FILE`, `ASTRAL_MODEL_MIN_BYTES`).

## Media (vision/audio) tests

`test_media` exercises mock vision/audio feeds + multimodal embeddings. Real CPU media init smoke checks run when:

- `ASTRAL_TEST_VISION_MODEL` + `ASTRAL_TEST_VISION_MEDIA` are set
- `ASTRAL_TEST_AUDIO_MODEL` + `ASTRAL_TEST_AUDIO_MEDIA` are set

Media init requires `ASTRAL_ENABLE_MTMD=ON` at build time.

Set `ASTRAL_TEST_REQUIRE_MEDIA=1` to make missing or undersized fixtures fail instead of skipping. The release gate does this through:

```bash
./scripts/run_multimodal_validation.sh --bench
```

## What the gates cover

- `gate_source_scans`: repo-wide source scan that enforces hard rules (tracked cleanup markers, no compare-and-swap ops in sources/docs, no suspicious full-vocab logits copies, no production C/C++ lambdas, and no unreviewed filler prose in comments/docs).
- `gate_shared_exports`: Linux shared-library export scan that allows only `astral_*` symbols declared with `ASTRAL_API` in `include/astral_rt.h`.
- `gate_unreal_header_mirror`: verifies the tracked Unreal ThirdParty `astral_rt.h` mirror matches the root public ABI header exactly.
- `gate_abi_layout_report`: compiles `include/astral_rt.h` and checks that a JSON struct size/alignment report can be generated for release evidence.
- `gate_release_metadata`: checks that release metadata generation keeps packaged zips, `abi-layout.json`, and `dependency-manifest.json` covered by `checksums.sha256`, then runs the release artifact verifier.
- `gate_release_evidence`: smoke-checks the release evidence manifest validator, including a missing-lane failure.
- `gate_release_required_plan`: checks the fast release gate preflight path and required release-candidate environment reporting.
- `gate_release_sign_workflow`: checks that protected release signing validates downloaded evidence before importing signing credentials.
- `gate_windows_large_page_runner`: checks that the Windows large-page validation runner and opt-in test expectations stay wired.
- `gate_release_notes`: checks release notes include artifact, validation, engine, rollback, and known-gap evidence.
- `gate_dependency_pins`: checks the committed release pin manifest against submodule and engine package versions.
- `gate_allocations`: best-effort heap allocation interposition gate for steady-state decode/stream (runs mock always; CPU is opt-in via `ASTRAL_GATE_CPU_ALLOC=1`).

`test_abi_invalid_args` is the fast public C ABI boundary matrix. It exercises
null outputs, invalid handles, and empty plugin paths without requiring model
fixtures.

`test_platform` accepts two Windows-only validation environment hooks:
`ASTRAL_TEST_EXPECT_LARGE_PAGES=1` requires explicit large-page allocation to
succeed, while `ASTRAL_TEST_EXPECT_LARGE_PAGE_FALLBACK=1` requires fallback.
Use `scripts/run_windows_large_page_validation.ps1` rather than setting those
variables by hand.
