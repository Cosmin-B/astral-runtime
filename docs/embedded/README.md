# Embedded and robotics profile (Linux edge)

This profile targets small Linux edge devices (Pi-class ARM SBCs) and robotics workloads where predictable steady-state behavior matters more than peak throughput: low memory footprint, no per-token heap work, and backpressure-aware streaming.

## Goals

- Keep steady-state decode/stream free of heap allocations and unnecessary overhead.
- Choose the backend once at model load (`AstralModelDesc.backend_name`) and keep per-token paths provider-agnostic.
- Make it easy to run a minimal bytes-first streaming loop that is compatible with tight control loops.

## Starting points

- Embedded presets use arena initialization because virtual memory and Astral
  worker threads are disabled. Size the arena from the selected model and
  feature set.
- `AstralInit.enable_hugepages`: off by default.
- `AstralModelDesc.n_ctx` / `n_batch`: leave `0` to use backend defaults, then tune based on the model and RAM budget.
- CPU backend uses mmap for model weights by default (startup and page-cache behavior will dominate TTFT on cold cache).

## Build + run (host)

```bash
cd astral-runtime
cmake --preset embedded-x86_64
cmake --build --preset embedded-x86_64 -j

./build/embedded-x86_64/examples/embedded/astral_embedded_cli --model tests/models/gpt2.Q2_K.gguf --prompt "hi" --tokens 128
```

Or run the scripted smoke (downloads the default small GGUF if missing):

```bash
cd astral-runtime
./scripts/run_embedded_smoke.sh
```

Or run the “validation” wrapper that also enforces the allocation gate:

```bash
cd astral-runtime
./scripts/run_embedded_validation.sh
```

This wrapper runs the following gates before the embedded smoke:
- `gate_embedded_presets`: embedded CMake presets keep VM, temp-file mmap, dynamic loading, JSON-schema grammar, and Astral worker threads disabled
- `gate_allocations`: no allocations during steady-state decode/stream. The mock path always runs, and CPU runs when a GGUF is available.
- `gate_io_syscalls`: no I/O-ish syscalls in steady-state (warmup/reset excluded)
- `gate_rss_cap`: RSS cap check (Linux-only, configurable via `ASTRAL_RSS_MAX_MB`)

## Native build (on device)

```bash
cd astral-runtime
cmake --preset embedded-native
cmake --build --preset embedded-native -j
```

## Cross-compile (templates)

Toolchain templates live under `toolchains/`. They assume a Debian-style cross compiler is installed and available in `PATH`.

```bash
cmake --preset embedded-arm64-cross
cmake --build --preset embedded-arm64-cross -j
```

If your toolchain uses different compiler names/paths, edit the toolchain file or set `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER`.

## Runtime notes

- **mmap and TTFT**: cold-cache runs will page fault. Use a warmup decode when comparing TTFT.
- **Backpressure**: a slow consumer applies stream backpressure. Use cancel and wait for bounded shutdown under load.

## CI and QEMU

The GitHub Actions `cross-compile` job cross-compiles the embedded binaries for arm64/armv7 and runs a small QEMU smoke using the mock backend (no GGUF needed). This is a basic correctness gate for ARM execution + wait/wake behavior.
