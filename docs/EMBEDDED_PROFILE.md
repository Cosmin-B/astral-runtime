# Embedded / Robotics Profile

This document describes the build-time feature matrix for Astral when targeting embedded/robotics-style deployments (tight memory, limited syscalls, deterministic latency).

## Goals

- Guarantee that **no C++ exceptions cross the C ABI** (return error codes + set `astral_last_error()` instead).
- Keep dependencies minimal (no dynamic loading, optional helper features off).
- Provide a predictable, testable configuration via CMake presets.

## Recommended presets

- Host (quick local smoke): `cmake --preset embedded-x86_64`
- Cross + QEMU gates: `cmake --preset embedded-arm64-ci` / `embedded-armv7-ci`

See also: `docs/embedded/README.md` for runtime/CI notes and example commands.

## Feature matrix (CMake options)

### `ASTRAL_NO_THROW_ABI` (default: ON)

Catch and translate C++ exceptions at the **C ABI boundary**.

- ON (recommended for embedded/product builds):
  - Wraps all `ASTRAL_API` functions in `try/catch(...)` (when exceptions are enabled) and translates to error returns.
  - Prevents C++ exceptions from unwinding into Unity/Unreal/other C callers.

### `ASTRAL_NO_EXCEPTIONS` (default: OFF)

Compile **Astral sources** with exceptions disabled (`-fno-exceptions` or MSVC `/EHs-c-`).

Notes:
- This is **not** a full “no exceptions in the entire process” guarantee today because third-party code (notably llama.cpp/ggml) may still use exceptions internally.
- If an exception escapes a third-party dependency and unwinds through code compiled with exceptions disabled, the process may terminate. Treat this option as **experimental** unless you also vendor/patch dependencies to be exception-free.

### `ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR` (default: ON)

Enables JSON-schema → GBNF helper plumbing for grammar-constrained decoding.

- OFF in embedded presets (to avoid exception-reliant helper paths and extra dependency surface).

### `ASTRAL_ENABLE_DYNAMIC_BACKENDS` (default: ON)

Enables runtime loading of backend providers via `dlopen`/`LoadLibrary` (`astral_backend_load_plugin()`).

- OFF in embedded presets (removes `dlopen` dependency and dynamic loading surface).
- When OFF, `astral_backend_load_plugin()` returns `ASTRAL_E_UNSUPPORTED`.

### `ASTRAL_ENABLE_THREADS` (default: ON)

Controls **Astral’s internal worker pool** used for async work scheduling.

- OFF:
  - No Astral worker threads are created.
  - Work submitted via `submit_work()` runs synchronously on the caller thread.

Notes:
- Backends may still use threads internally (e.g., llama.cpp). Disabling Astral’s worker pool does not guarantee a fully thread-free process.
- Continuous-batching conversations require Astral thread support and return `ASTRAL_E_UNSUPPORTED` in this profile.

### `ASTRAL_ENABLE_VIRTUAL_MEMORY` (default: ON)

Controls whether Astral is allowed to use platform virtual memory APIs (`vm_reserve/vm_commit`) for its runtime backing store.

- OFF (embedded presets):
  - VM-backed `astral_init()` / `ASTRAL_MEMMODE_VM` is unavailable (returns `ASTRAL_E_UNSUPPORTED`).
  - Use `astral_init2()` with an arena memory mode instead (see below).

## Arena mode (no VM dependency)

Use the `astral_init2()` surface with one of the arena modes:

- `ASTRAL_MEMMODE_ARENA_BORROWED`: you provide `arena.base` + `arena.size`.
- `ASTRAL_MEMMODE_ARENA_OWNED`: Astral allocates the arena (or you provide `arena.base`).

In arena modes, Astral provisions deterministic per-session scratch blocks and a bounded runtime heap from the arena. Astral-owned allocations return `ASTRAL_E_NOMEM` when those regions are exhausted; they do not spill into `sys_alloc`. A backend library can still have a separate allocation contract, so check the selected provider before treating the whole process as arena-only.

### Arena partitioning knobs (optional)

`AstralArenaDesc` includes a small reserved array for additional partitioning/sizing without changing ABI layout:

- `arena._reserved[0]`: worker scratch bytes per worker (default: 256 KiB)
- `arena._reserved[1]`: runtime heap bytes (default: 2 MiB)

These are used to reserve a small front slice of the arena for:

- worker-thread-local transient scratch (bump-only)
- a small size-class heap for deterministic non-hot allocations (model/session objects, plugin strings)

## Model loading source contracts

Astral exposes `astral_model_load2()` with `AstralModelDesc` to select a model source (`PATH` / `MEMORY` / custom `IO`) so embedded targets can avoid filesystem paths.

Status:
- The public ABI and mock provider support `MEMORY` and `IO` sources without requiring a filesystem path.
- The built-in CPU llama backend does not currently provide a true callback-backed no-filesystem real-model path. On desktop builds with `ASTRAL_CPU_MEMORY_SOURCE_MMAP=ON`, it may materialize `MEMORY`/`IO` input to a temporary file so llama.cpp can use file-backed mmap.
- Embedded presets keep both `ASTRAL_ENABLE_VIRTUAL_MEMORY=OFF` and `ASTRAL_CPU_MEMORY_SOURCE_MMAP=OFF`, so they cannot silently take the desktop temp-file mmap path.

Current limitations:
- Real CPU GGUF loading from `MEMORY`/`IO` in embedded presets returns unsupported until the backend has a non-file llama.cpp loader path.
- Split GGUF models are not supported for `IO`/`MEMORY` sources (single-file only).

See `tests/MODEL_SOURCES.md` for test model notes.

## Embedded hardening checklist

- Use an embedded preset (`embedded-*`) to turn off dynamic backends, JSON schema grammar, and Astral worker threads (and keep the C ABI no-throw guard on).
- Decide how strict you need “no exceptions” to be:
  - Product-safe: keep exceptions enabled, rely on `ASTRAL_NO_THROW_ABI=ON` (default).
  - Strict/experimental: `ASTRAL_NO_EXCEPTIONS=ON` only if you control/verify the full dependency graph.
- Run gates that are already designed for embedded profiles:
  - `gate_embedded_presets`
  - `gate_allocations`
  - `gate_io_syscalls`
  - `gate_rss_cap` (Linux)
