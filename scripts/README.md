# Repository scripts

Scripts in this directory are command-line entry points for validation,
benchmarks, packaging, and engine integration. Run a script with `--help` when
it provides one. The subsystem guides carry the configuration and acceptance
rules that make each command meaningful.

## Common validation

```bash
./scripts/run_fast_presubmit.sh
./scripts/run_asan.sh
./scripts/run_tsan.sh
./scripts/run_valgrind.sh
./scripts/run_release_required_gates.sh
```

The preset build and CTest suite remain the normal starting point. These
wrappers add sanitizer, syscall, memory, or release-policy lanes. See
[BUILD.md](../BUILD.md), [the release acceptance matrix](../docs/release/RELEASE_ACCEPTANCE_MATRIX.md),
and [memory validation](../tests/MEMORY_VALIDATION.md).

## Documentation and source checks

```bash
./scripts/build_docs_site.sh
python3 scripts/validate_doc_links.py
python3 scripts/inventory_comments.py --help
```

The docs build runs MkDocs in strict mode and writes generated HTML under
`build/docs-site`. The local-link validator scans maintained Markdown without
requiring a site build.

## Retrieval and performance

- `run_memory_bench_matrix.sh` and `summarize_memory_bench_matrix.py` measure
  comparable memory-index shapes.
- `run_memory_ann_tuning.sh` and `run_memory_search_acceptance.sh` tune graph
  budgets against an exhaustive flat oracle.
- `run_feature_bench_suite.sh`, `run_ci_bench_features.sh`, and
  `run_primitive_perf_capture.sh` measure broader runtime and primitive cases.
- `run_allocator_perf_capture.sh` measures allocator fixtures.

Use the exact storage, metric, dimension, capacity, top-k, and iteration count
from [the memory-index guide](../docs/api/MEMORY_INDEX.md). A result without its
command and machine context is not a portable performance claim.

## Models, CUDA, and multimodal fixtures

- `hf_gguf_download_manifest.sh`, `hf_gguf_sync.py`, and the `run_hf_*` scripts
  manage opt-in GGUF model matrices.
- `run_cuda_parity.sh` and `run_cuda_parity_matrix.sh` exercise CUDA selection
  modes described in [CUDA parity](../docs/CUDA_PARITY.md).
- `resolve_mtmd_fixtures.py` and `run_multimodal_validation.sh` handle the
  opt-in model/projector fixtures described in [vision and audio](../docs/VISION_AUDIO.md).

Model downloads are large and are not part of the default test suite. Use
[tests/MODEL_SOURCES.md](../tests/MODEL_SOURCES.md) for the maintained sources
and pinning rules.

## Embedded and engine integration

- `run_embedded_smoke.sh`, `run_embedded_validation.sh`, and
  `run_arm64_hardware_validation.sh` cover embedded profiles.
- `run_unity_ci_tests.sh` and `run_unity_gameci_tests.sh` drive Unity tests.
- `create_unreal_sample_project.sh`, `run_unreal_ci_tests.sh`,
  `run_unreal_compatibility_matrix.sh`, `run_unreal_container_ci.sh`, and the
  `run_unreal_sample_*` scripts drive Unreal builds and samples.

See the [embedded guide](../docs/EMBEDDED_PROFILE.md),
[Unity package guide](../plugins/unity/README.md), and
[Unreal plugin guide](../plugins/unreal/AstralRT/README.md) before running these
commands. Engine and container scripts may require separately licensed tools or
authenticated registries.

## Packaging and release metadata

- `package_release.sh` builds staged archives.
- `generate_abi_layout_report.sh`, `generate_release_metadata.sh`, and
  `generate_release_sbom.py` produce release metadata.
- `validate_dependency_pins.sh`, `validate_release_artifacts.sh`,
  `validate_release_evidence.py`, and `validate_release_notes.sh` check staged
  release inputs.
- `sign_release_artifacts.sh` signs an already prepared artifact set.

These scripts do not replace the manifests under
[`docs/release`](../docs/release/). Signing and publication are separate,
explicit release operations.
