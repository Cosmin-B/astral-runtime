# Astral Roadmap (Productization)

This is the living checklist to turn Astral into a world-class, product-ready framework for game engines and embedded/robotics targets.

## v0.1 (Desktop core + optional scaffold)

**Release targets**: Windows/macOS/Linux (x86_64 + arm64). GPU support is a build-time/runtime optional scaffold in v0.1 (best-effort); Android/iOS plugins follow in v0.1.1.

- Embedded pillars (done):
  - `astral_init2()` arena modes (`BORROWED` / `OWNED`) + deterministic per-session scratch blocks.
  - Embedded presets disable VM, dynamic backends, and JSON schema grammar (`docs/EMBEDDED_PROFILE.md`).
  - `astral_model_load2()` supports `PATH` / `MEMORY` / `IO` for the built-in CPU provider (single-file GGUF; PATH uses mmap; MEMORY/IO can optionally use a temp-file mmap path on desktop; embedded presets disable this).
- Profiling (done):
  - Tracy integration + profiling presets (+ optional micro-zones) across hot paths (`docs/PROFILING_TRACY.md`).
- CUDA scaffold (done, non-blocking until v0.2):
  - Optional provider registration + presets (`dev-cuda`, `release-cuda`) and a surface test for presence/absence.
- Release hygiene (todo):
  - Session affinity policy + callback-safe streaming/error surfaces (done):
    - Decode work is submitted via `submit_work_affine()` to a fixed `session->worker_id`.
    - `astral_stream_read()` / `astral_stream_read_meta()` fail fast on concurrent consumers (no UB/data races).
  - Ship both static + shared artifacts on desktop (done): `astral_rt` (static) + `astral_rt_shared` (shared; outputs as `astral_rt`).
    - Windows naming: static is `astral_rt_static.lib` when both are built (avoids clash with the shared import lib).
  - CI matrix aligned with v0.1 scope and artifact story (done): release artifacts are built via `scripts/package_release.sh` and uploaded from CI as “v0.1 desktop” zips.
  - Docs pass (done): keep `README.md`, `BUILD.md`, and `docs/*` consistent with the shipped presets/features.

## v0.1.1 (Mobile plugins)

- Android/iOS Unity plugin packaging + validation (arm64; Unity 2022.3 LTS+).
- iOS-specific constraints: page size, entitlements, and minimal-syscall guidance.
- Unreal mobile plugin packaging/validation (Android/iOS) after Unity; ship prebuilt ThirdParty libs for UE 5.1–5.7.

## v0.2 (GPU backend real work)

- CUDA backend parity + perf (then Metal, then DirectML):
  - Define a parity checklist per backend (model load, tokenize/detokenize, streaming, embeddings, grammar, logprobs, batching).
  - Add backend parity tests and reproducible perf benches per target GPU tier.
- Keep optional deps truly optional (CPU-only builds must not pull GPU toolchains).

See: `docs/CUDA_PARITY.md` (living checklist + parity runner notes).

## Embedded / Robotics (Ongoing)

- Minimal-footprint feature matrix with tested preset combinations (threads on/off, VM on/off, dynamic backends on/off, IO on/off).
- Allocator policy hardening: sizing guidance + deterministic failure surfaces for third-party integrators.

## Packaging (Ongoing)

- Unity: deterministic init paths + allocation-minimized logging bridge (done).
- Unreal: module packaging, logging bridge, allocator integration, and stable plugin binary naming/layout.
