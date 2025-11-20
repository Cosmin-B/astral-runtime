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

Fixture-dependent cases call `SKIP_TEST(...)` when no model is found, so CTest logs show which real-model coverage did not run. If a model path is found but the load fails, the test fails.

### Downloading a small model

```bash
bash astral/tests/model_downloader.sh
export ASTRAL_TEST_MODEL="$PWD/astral/tests/models/gpt2.Q2_K.gguf"
```

The downloader supports overrides via `--url/--file/--min-bytes` and env vars (`ASTRAL_TEST_MODEL_URL`, `ASTRAL_TEST_MODEL_FILE`, `ASTRAL_MODEL_MIN_BYTES`).
For additional small smoke fixtures, use the Qwen3 presets:

```bash
./tests/model_downloader.sh --preset qwen3-0.6b-q8
./tests/model_downloader.sh --preset qwen3-embed-0.6b-q8
```

For a newer tiny text-generation fixture, use Gemma 3 270M:

```bash
./tests/model_downloader.sh --preset gemma3-270m-q4km
```

## Media (vision/audio) tests

`test_media` exercises mock vision/audio feeds + multimodal embeddings. Real CPU media init smoke checks run when:

- `ASTRAL_TEST_VISION_MODEL` + `ASTRAL_TEST_VISION_MEDIA` are set
- `ASTRAL_TEST_AUDIO_MODEL` + `ASTRAL_TEST_AUDIO_MEDIA` are set

Media init requires `ASTRAL_ENABLE_MTMD=ON` at build time.

Set `ASTRAL_TEST_REQUIRE_MEDIA=1` to make missing or undersized fixtures fail instead of skipping. The release gate does this through:

```bash
./scripts/validate_mtmd_fixture_manifest.py scripts/mtmd_fixture_manifest_lfm25.json
./scripts/hf_gguf_download_lfm25_all.sh --out tests/models/hf-lfm25
./scripts/run_multimodal_validation.sh --check-fixtures
./scripts/run_multimodal_validation.sh --bench
```

## What the gates cover

- `gate_source_scans`: repo-wide source scan that enforces hard rules (tracked cleanup markers, no compare-and-swap ops in sources/docs, no suspicious full-vocab logits copies, no production C/C++ lambdas, and no unreviewed filler prose in comments/docs).
- `gate_shared_exports`: Linux shared-library export scan that allows only `astral_*` symbols declared with `ASTRAL_API` in `include/astral_rt.h`.
- `gate_unreal_header_mirror`: verifies the tracked Unreal ThirdParty `astral_rt.h` mirror matches the root public ABI header exactly.
- `gate_abi_layout_report`: compiles `include/astral_rt.h` and checks that a JSON struct size/alignment report can be generated for release evidence.
- `gate_release_metadata`: checks that release metadata generation keeps packaged zips, `abi-layout.json`, `dependency-manifest.json`, and optional `release-evidence.json` covered by `checksums.sha256`, then runs the release artifact and release-evidence verifiers.
- `gate_release_artifact_signature`: checks that `--require-signature` rejects an invalid detached checksum signature.
- `gate_release_evidence`: smoke-checks the release evidence manifest validator, including a missing-lane failure.
- `gate_release_required_plan`: checks the fast release gate preflight path and required release-candidate environment reporting.
- `gate_release_sign_workflow`: checks that protected release signing validates downloaded evidence before importing signing credentials.
- `gate_windows_large_page_runner`: checks that the Windows large-page validation runner and opt-in test expectations stay wired.
- `gate_unreal_automation_results`: smoke-checks the Unreal Automation result validator for pass, missing-report, and failing-log cases.
- `gate_unreal_container_runner`: checks that the UE 5.7 full and slim container lanes reject missing Epic GHCR auth before invoking the container engine. `gate_source_scans` pins the container runner's UE 5.7 clang and Linux SDK preflight checks.
- `gate_unreal_validation_access`: checks the fast Unreal access readiness probe for missing access, cached UE 5.7 containers, readable Epic GHCR manifests, and executable UE 5.4-5.7 editor paths.
- `gate_unreal_compatibility_matrix`: checks that the UE 5.4-5.7 matrix runner fails clearly for missing required editors, rejects unsupported versions, and does not accept all-skipped discovery runs as release evidence.
- `gate_unreal_sample_scaffold`: generates a temporary UE sample project and checks that it includes model load, streaming, cancellation, embeddings, and error-reporting sample code.
- `gate_unity_editmode_results`: smoke-checks the Unity EditMode XML validator for pass, failed-result, and malformed-result cases.
- `gate_hf_matrix_log`: checks that HF matrix logs can be parsed and that failed, empty, skipped-only, or incomplete feature logs are rejected when pass evidence is required.
- `gate_mtmd_fixture_manifest`: checks that the MTMD fixture manifest uses pinned revisions, license metadata, required vision/audio model/projector files, and the runner's fast fixture preflight.
- `gate_cuda_parity_runner`: checks that CUDA parity runners require real inference/e2e flags and GPU/toolkit runner visibility unless probe-only mode is explicit, and that the release matrix still covers auto, cuBLAS, and MMQ presets.
- `gate_model_churn_soak`: repeated model/session load, decode, reset, unload, and RSS-drift sampling. It runs a fast mock churn by default; real GGUF churn is opt-in with `ASTRAL_SOAK_MODEL`.
- `gate_release_notes`: checks release notes include artifact, validation, engine, rollback, and known-gap evidence.
- `gate_dependency_pins`: checks the committed release pin manifest against submodule and engine package versions.
- `gate_embedded_presets`: checks embedded presets keep VM, temp-file mmap, dynamic loading, JSON-schema grammar, and Astral worker threads disabled.
- `gate_allocations`: tracked heap allocation gate for steady-state decode/stream. It runs mock coverage by default; CPU model coverage is opt-in with `ASTRAL_GATE_CPU_ALLOC=1`.

`test_embeddings` always runs mock lifecycle and queue-pressure coverage. Set
`ASTRAL_TEST_EMBED_MODEL` to a readable embeddings GGUF to run the CPU fixture
probe; the log prints `[embedding_probe]` with model path, dimension, and vector
sanity metadata.

Unreal real-model embedding evidence uses the same native CPU backend through
`UAstralModel` and `UAstralEmbedder`. Run it inside a UE container with a model
path that exists in the mounted workspace:

```bash
ASTRAL_UNREAL_TEST_MODEL=/workspace/astral/tests/models/Qwen3-0.6B-Q8_0.gguf \
ASTRAL_UNREAL_REQUIRE_REAL_GENERATION=1 \
scripts/run_unreal_container_ci.sh --ue-version 5.7 --variant slim --skip-pull --skip-native-build --filter AstralRT.Real.GenerationSmoke

ASTRAL_UNREAL_TEST_MODEL=/workspace/astral/tests/models/Qwen3-0.6B-Q8_0.gguf \
ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE=1 \
scripts/run_unreal_container_ci.sh --ue-version 5.7 --variant slim --skip-pull --skip-native-build --filter AstralRT.Real.SessionLifecycle

ASTRAL_UNREAL_TEST_EMBED_MODEL=/workspace/astral/tests/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING=1 \
scripts/run_unreal_container_ci.sh --ue-version 5.7 --variant slim --skip-pull --skip-native-build --filter AstralRT.Real.EmbeddingProbe
```

The generation smoke log prints `[unreal_generation_smoke]` with byte-count and
text evidence. The session lifecycle smoke prints `[unreal_session_lifecycle]`
with cancel, reset, and reuse evidence. The embedding probe log prints
`[unreal_embedding_probe]` with backend, model path, dimension, and vector
sanity metadata.

`test_abi_invalid_args` is the fast public C ABI boundary matrix. It exercises
null outputs, invalid handles, and empty plugin paths without requiring model
fixtures.

`test_platform` accepts two Windows-only validation environment hooks:
`ASTRAL_TEST_EXPECT_LARGE_PAGES=1` requires explicit large-page allocation to
succeed, while `ASTRAL_TEST_EXPECT_LARGE_PAGE_FALLBACK=1` requires fallback.
Use `scripts/run_windows_large_page_validation.ps1` rather than setting those
variables by hand.
