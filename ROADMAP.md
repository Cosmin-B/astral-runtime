# Astral Roadmap

This roadmap tracks the work needed to turn Astral's validated native runtime
and engine packages into a production release candidate. Current readiness and
external blockers are recorded in `docs/PRODUCTION_READINESS_AUDIT.md`; this
file should not claim support that is not backed by gates or release evidence.

## Current Native Baseline

Validated locally:

- `dev` and `release-with-tests` native CTest lanes.
- Static and shared desktop runtime artifacts from `scripts/package_release.sh`.
- ABI layout, shared-export, checksum, SBOM, dependency-pin, release-note,
  source-prose, allocation, RSS, and syscall gates.
- Mock/CPU native inference, embeddings, model-source, provider, and
  continuous-batching tests.
- Tracy profiling presets and coarse zones for maintained hot paths.
- Embedded presets that disable VM, dynamic backends, and JSON schema grammar
  as described in `docs/EMBEDDED_PROFILE.md`.

Release blockers:

- Real Unreal Editor/container runs.
- Real Unity Editor EditMode run.
- Real CUDA runner parity/e2e matrix.
- Real multimodal fixtures with MTMD enabled.
- Real Windows large-page privilege variants.
- Protected release signing run with evidence attached.

## Unreal-First Engine Target

Primary target:

- UE 5.7 on Linux with Epic's clang 20.1.8 toolchain and the pinned
  `dev-5.7.4` / `dev-slim-5.7.4` images documented in
  `docs/integration/UNREAL_57_QUICKSTART.md`.

Compatibility target:

- UE 5.4, 5.5, 5.6, and 5.7 compile/smoke matrix through
  `scripts/run_unreal_compatibility_matrix.sh`.

Remaining Unreal work:

- Run Automation tests in real UE 5.7 full and slim containers.
- Package a UE 5.7 sample project for Linux.
- Validate load, stream, cancel, reset, media feed, embeddings, and shutdown in
  editor and packaged-game lifecycles.
- Keep native ThirdParty provenance, `LogAstralRT`, and UE profiler scopes
  gated.

## CUDA, Models, And Multimodal

CUDA remains a release blocker until it has real hardware evidence:

- Required CPU-vs-CUDA parity and CUDA e2e flags are documented in
  `docs/CUDA_PARITY.md`.
- Release candidates must run auto, cuBLAS-forced, and MMQ-forced lanes on the
  target GPU runner.
- Multi-GPU routing and deployed architecture coverage need explicit evidence
  before being listed as supported.

Model and multimodal work:

- Keep small text and embedding fixtures for fast native gates.
- Run the HF GGUF matrix before release-candidate sign-off.
- Run `ASTRAL_ENABLE_MTMD=ON` with real vision/audio projectors and fixtures.
- Record fixture, projector, CUDA media-routing, and failure-case evidence in
  the release manifest.

## Mobile And Embedded

Mobile work is not part of the current Unreal-first release sign-off:

- Android and iOS artifacts still need build, packaging, and runtime smoke
  validation.
- Unity mobile support requires a tested Editor/package matrix and native
  binary layout per architecture.
- Unreal mobile support should follow the UE 5.4+ desktop compatibility matrix,
  not introduce a separate version policy.

Embedded work stays scoped to explicit profiles:

- Preserve no-VM/no-dynamic-backend/no-JSON-schema embedded presets.
- Document exactly when MEMORY/IO sources materialize temp files on desktop.
- Add hardware or QEMU evidence before claiming a deployment profile.

## Release Governance

Before a release candidate:

- Run native dev and release-with-tests gates.
- Run sanitizer lanes through `scripts/run_asan.sh` and `scripts/run_tsan.sh`.
- Package artifacts with checksums, SBOM, ABI layout, dependency manifest,
  license notices, Unity package, and Unreal plugin zips.
- Attach release evidence for Unreal, Unity, CUDA, multimodal, HF matrix,
  Windows large pages, artifact verification, signing, dependency review, and
  release notes.
- Complete protected signing with a verified detached checksum signature.
