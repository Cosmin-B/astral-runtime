# Tracy Profiling

Astral can be built with Tracy to capture coarse profiling zones (decode, streaming, scheduling) with low overhead.

## Setup (submodule)

Tracy is integrated as an optional git submodule:

```bash
./scripts/setup_tracy_submodule.sh
```

Pinned revision:
- `external/tracy` @ `a602127eddb60825ac91e726986c12955e9a0082` (`v0.13.1-55-ga602127e`)

If the submodule is not present yet, add it:

```bash
git submodule add https://github.com/wolfpld/tracy.git external/tracy
git submodule update --init --recursive external/tracy
```

## Build presets

- Dev profiling: `cmake --preset dev-prof && cmake --build --preset dev-prof -j`
- Dev profiling (micro zones): `cmake --preset dev-prof-micro && cmake --build --preset dev-prof-micro -j`
- Release profiling: `cmake --preset release-prof && cmake --build --preset release-prof -j`
- Release profiling (micro zones): `cmake --preset release-prof-micro && cmake --build --preset release-prof-micro -j`
- Unity plugin profiling: `cmake --preset unity-plugin-prof && cmake --build --preset unity-plugin-prof -j`
- Unity plugin profiling (micro zones): `cmake --preset unity-plugin-prof-micro && cmake --build --preset unity-plugin-prof-micro -j`
- Unreal plugin profiling: `cmake --preset unreal-plugin-prof && cmake --build --preset unreal-plugin-prof -j`
- Unreal plugin profiling (micro zones): `cmake --preset unreal-plugin-prof-micro && cmake --build --preset unreal-plugin-prof-micro -j`

Embedded presets keep Tracy disabled by default.

## Runtime

Profiling presets compile the Tracy client into the native library.

Defaults are “product-safe”:
- on-demand capture
- localhost-only
- no broadcast

## What is instrumented (coarse)

This integration intentionally avoids micro/per-token instrumentation for now. You should see zones like:
- `astral.decode_work`, `astral.decode_loop`, `astral.generation_loop`
- `astral.session_feed`, `astral.stream_read`
- `astral.submit_work`

And plots like:
- `astral.work_queue_depth`
- `astral.tokens_total`, `astral.tokens_per_s`, `astral.ttft_ms`

Connect using the Tracy UI to start a capture.

## Quick capture workflow

1) Ensure the Tracy submodule exists:

```bash
./scripts/setup_tracy_submodule.sh
```

2) Build a profiling preset and run a small workload while the Tracy UI is open:

```bash
./scripts/run_tracy_capture.sh --preset dev-prof
```

If you want higher detail (more overhead), use a micro preset:

```bash
./scripts/run_tracy_capture.sh --preset dev-prof-micro
```

## Micro instrumentation (optional)

If you need much finer detail (queue ops, backend call breakdown), enable the micro zones:
- Build presets: use `*-prof-micro` variants (or set `-DASTRAL_ENABLE_TRACY_MICRO=ON`).

Notes:
- This increases overhead and trace volume significantly; prefer coarse zones for product-like runs.
