# Release Acceptance Matrix

This matrix is the minimum evidence required before calling an Astral build a
release candidate.

## Fast Native Gate

| Area | Command | Required for RC |
|---|---|---|
| Debug native build/tests | `cmake --preset dev && cmake --build --preset dev -j && ctest --preset dev -j --output-on-failure` | Yes |
| Release native build/tests | `cmake --preset release-with-tests && cmake --build --preset release-with-tests -j && ctest --preset release-with-tests -j --output-on-failure` | Yes |
| Memory-search recall | `./scripts/run_memory_search_acceptance.sh --preset release-with-tests --iters 10 --dim 384 --capacity 10000 --metric cosine --graph-search 256 --query-search 256 --graph-neighbors 32 --recall-queries 32 --graph-storages f32,q8,q8f32,f6e3m2f32,f8e5m2f32 --skip-flat-baseline --min-recall-pct 99 --recall-storages f32,q8,q8f32,f6e3m2f32,f8e5m2f32` | Yes |
| Sanitizer validation | `./scripts/run_asan.sh && ./scripts/run_tsan.sh` for ASAN/UBSAN and TSan evidence | Yes |
| Comment review | `python3 ./scripts/inventory_comments.py --format review-tsv` plus `python3 ./scripts/inventory_comments.py --format summary --fail-orphan-markers` | Yes |
| Release gate preflight | `ASTRAL_TEST_VISION_MODEL=... ASTRAL_TEST_VISION_MEDIA=... ASTRAL_TEST_AUDIO_MODEL=... ASTRAL_TEST_AUDIO_MEDIA=... UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... UNREAL_RUNUAT=... ./scripts/run_release_required_gates.sh --print-plan --cuda-arch <deployed-arch-list> --cuda-strict --mtmd-bench` | Yes |
| Required release gates | `ASTRAL_TEST_VISION_MODEL=... ASTRAL_TEST_VISION_MEDIA=... ASTRAL_TEST_AUDIO_MODEL=... ASTRAL_TEST_AUDIO_MEDIA=... UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... UNREAL_RUNUAT=... ./scripts/run_release_required_gates.sh --cuda-arch <deployed-arch-list> --cuda-strict --mtmd-bench` | Yes |
| Release metadata | `./scripts/generate_release_metadata.sh dist`, or `./scripts/package_release.sh ... --evidence <release-evidence.json>` for RC packaging | Yes |
| Artifact verifier | `./scripts/validate_release_artifacts.sh --dist dist --expect-unity --expect-unreal --require-signature`; verifies `checksums.sha256.asc` with GPG or `checksums.sha256.minisig` with minisign | Yes |
| Evidence manifest | `python3 ./scripts/validate_release_evidence.py dist/release-evidence.json --base-dir dist` | Yes |
| Release notes | `./scripts/validate_release_notes.sh <release-notes.md>` | Yes |
| Dependency pins | `./scripts/validate_dependency_pins.sh` | Yes |

## Unreal Gate

| Area | Command or lane | Required for RC |
|---|---|---|
| UE 5.7 access preflight | `./scripts/check_unreal_validation_access.sh --check-registry` | Yes |
| UE 5.7 full container | `./scripts/run_unreal_container_ci.sh --variant full --filter AstralRT --install-cmake` using `ghcr.io/epicgames/unreal-engine:dev-5.7.4`; this lane rebuilds the ThirdParty package inside Epic's UE Linux SDK | Yes |
| UE 5.7 slim container | `ASTRAL_UNREAL_TEST_MODEL=... ASTRAL_UNREAL_REQUIRE_REAL_GENERATION=1 ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE=1 ASTRAL_UNREAL_TEST_EMBED_MODEL=... ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING=1 ./scripts/run_unreal_container_ci.sh --variant slim --filter AstralRT --skip-native-build` using `ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4`; this lane reuses the package staged by the full-container build and runs the lightweight real-model probes | Yes |
| Automation tests | `UNREAL_EDITOR=/path/to/UnrealEditor-Cmd ./scripts/run_unreal_ci_tests.sh` | Yes |
| UE 5.4/5.5/5.6/5.7 compatibility | `ASTRAL_UNREAL_TEST_MODEL=... ASTRAL_UNREAL_REQUIRE_REAL_GENERATION=1 UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... ./scripts/run_unreal_compatibility_matrix.sh` | Yes |
| UE 5.7 sample package and runtime smoke | `UNREAL_RUNUAT=... ./scripts/run_unreal_sample_package.sh --platform Linux --run-sample --runtime-log build/unreal-sample-package/runtime.log --sample-model ... --sample-embedding-model ... --sample-memory-backend mock --sample-media-backend mock` | Yes |

The UE 5.7 sample runtime log must include the media demo line with
`texture image`, proving the packaged sample exercised the Unreal `UTexture2D`
descriptor bridge in addition to raw RGBA bytes and PCM16 audio.

UE 5.7 container logs must show the pinned image/digest, manifest access check,
local image digest, clang `20.1.8`, Linux SDK `v26`/`20.1.8`, Unreal ThirdParty
provenance, module lifecycle Automation including EndPIE, and `[unreal-results]
OK`. The slim release lane must also show `AstralRT.Real.GenerationSmoke`,
`AstralRT.Real.SessionLifecycle`, `AstralRT.Real.EmbeddingProbe`, and the
`[unreal_generation_smoke]`, `[unreal_session_lifecycle]`, and
`[unreal_embedding_probe]` evidence markers. Real embedding logs must also show
the `[unreal_embedding_throughput]` marker. Mock embedder Automation must keep
covering queued batching, cancellation, queue backpressure, and positive
throughput through `AstralRT.Mock.EmbedderQueuePressure` and the
`[unreal_embedding_acceptance]` marker.

UE compatibility matrix logs must include a non-skipped `[unreal_matrix] UE ...`
section for every supported editor version from 5.4 through 5.7, with
`AstralRT` Automation output and `[unreal-results] OK` for each version. The UE
5.4 section must also show `AstralRT.Real.GenerationSmoke` and
`[unreal_generation_smoke] backend=cpu`, so the compatibility floor includes a
real CPU model smoke instead of mock Automation alone.

UE sample package logs must show `AstralSample.uproject`, copied plugin mode,
`BuildCookRun`, Linux platform packaging, and `[unreal_sample] OK`. The paired
runtime log must show `AstralSampleAutoQuit`, pak/IoStore mounts, and successful
packaged-content plus Saved-cache memory model loads.

## Backend And Model Gate

| Area | Required evidence | Required for RC |
|---|---|---|
| CPU llama | Native tests plus feature benchmark with a real GGUF | Yes |
| CUDA | `./scripts/run_cuda_parity_matrix.sh --preset-set release --arch <deployed-arch-list> --strict` on a real GPU runner with `ASTRAL_TEST_CUDA_PARITY_INFER=1` and `ASTRAL_TEST_CUDA_E2E=1` | Yes |
| Multimodal | `./scripts/run_multimodal_validation.sh --bench` with real vision and audio model/projector fixtures, preserving `mtmd-features.txt` with `features.media feed_image` and `features.media feed_audio` rows | Yes |
| HF matrix | Pinned GGUF matrix logs and pass/fail summary. The pinned manifest includes narrow Qwen3 0.6B text and Qwen3 Embedding 0.6B includes for fast real-model coverage, alongside the larger release matrix repos. | Yes |
| Windows large pages | `pwsh -File .\scripts\run_windows_large_page_validation.ps1 -ExpectFallback`, then `-ExpectLargePages` from a token with `SeLockMemoryPrivilege` | Yes |

## Release Artifacts

| Artifact | Required evidence |
|---|---|
| Core runtime package | Zip in `dist/`, `checksums.sha256`, dependency manifest, SPDX SBOM, ABI layout report |
| Unity package | Native binaries, package metadata, license/notice files, ABI tests |
| Unreal plugin package | ThirdParty native libraries, license/notice files, UE Automation evidence |
| Signing | Protected `release-sign` workflow validates `release-evidence.json` and checksum manifests before importing signing credentials, then writes and verifies detached signatures for each `checksums.sha256`; artifact verification must also reject invalid checksum signatures |
| Rollback | Release notes identify previous known-good artifact checksum, dependency pins, and waiver expiration dates |

Release notes must name the checksum signature file, identify the public
verification key or signer fingerprint, and include the exact checksum signature
verification command used for the release candidate.

## Release Evidence Manifest

Use `docs/release/RELEASE_EVIDENCE_TEMPLATE.json` as the starting point for
`dist/release-evidence.json`. Every required lane must be marked `pass` and must
point at a non-empty log or artifact under the release evidence directory. When a
SHA-256 is supplied, the validator checks it against the referenced file. The
`sanitizer_validation` lane records the ASAN/UBSAN log from `run_asan.sh` and
the TSan log from `run_tsan.sh`. The `comment_review` lane records the local
`review-tsv` artifact used for human comment classification and the zero orphan
marker summary; the validator checks both artifact contents.
The `release_artifacts`, `release_signing`, and `sanitizer_validation` lanes
must list the named artifacts in the matrix above, not only a generic log.

Validate with `--phase pre-sign` before the protected signing workflow has
produced `checksums.sha256.asc`; validate with the default complete phase after
signatures are present.

The manifest is a release-candidate artifact, not a source file to keep updating
in the repository. It records the runs that happened for a candidate; it does not
stand in for the external Unreal, Unity, CUDA, multimodal, signing, HF matrix, or
Windows validation lanes.
