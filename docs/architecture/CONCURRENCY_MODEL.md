# Concurrency and Threading Model

This document describes the **current** (v0.1) concurrency primitives and the runtime threading contract. Astral is intentionally conservative: correctness first, then measurable performance.

## Status (v0.1)

Implemented:
- **SPSC ring**: bounded token ring for streaming with backpressure and event signaling.
- **MPSC ring**: bounded fan-in ring with non-blocking `try_push`; full queues map to `ASTRAL_E_BUSY` at public API boundaries.
- **MPMC queue**: bounded ring with **ticket + per-slot sequence**, blocking `enqueue_wait` / `dequeue_wait` (no atomic compare-and-swap usage).
- **ARM-friendly waiting**: `cpu_pause`, `cpu_wait_for_event`, `cpu_signal_event` primitives for low overhead spin/wait.
- **CPU dispatch probe**: private runtime feature detection for x86_64 AVX2 and ARM NEON dispatch tiers.
- **Worker pool**: fixed worker threads + internal work queue (`astral::core::submit_work`)

Not implemented yet (planned):
- Work stealing scheduler.

## Threading contract

- **Model** objects may be shared across threads, but provider ops must document which calls are thread-safe.
- **Session** objects follow a **single-session-thread rule**:
  - Provider session state (KV cache) is mutated by the decode producer thread.
  - Token bytes are transferred to the consumer via a **single-producer/single-consumer ring**.
  - No other concurrent access to a session is supported.

Current implementation note:
- `astral_session_decode()` enqueues decode work onto the runtime worker pool and returns immediately.

## Platform wait primitives

Astral uses lightweight CPU hints instead of OS waits in hot paths:
- `cpu_pause()` is used for short spin phases (x86 PAUSE, ARM YIELD).
- `cpu_wait_for_event()` maps to ARM WFE (x86 fallback: pause).
- `cpu_signal_event()` maps to ARM SEV (x86 fallback: no-op).

See `astral/src/platform/atomics.h` for the exact mappings.

## MPSC ring (fan-in)

File: `astral/src/concurrency/mpsc_ring.hpp`

### Design

- Bounded ring with `Capacity` (power of 2).
- Multiple producers reserve publish positions with one producer-arbitration CAS.
- A single consumer owns dequeue order and does not need consumer-side CAS.
- Each slot contains a `seq` number that controls when a slot may be written or read.
- `try_push(const T&)` returns `false` when the ring is full.
- `pop(T& out)` returns `false` when the ring is empty and avoids pointer validation in the hot path.

### Memory-order contract

- Producers load `slot.seq` with acquire before reusing a slot released by the consumer.
- Producer reservation uses relaxed ordering; the slot sequence is the synchronization object.
- Producers write `slot.data`, then publish with `slot.seq.store(release, pos + 1)`.
- The consumer loads `slot.seq` with acquire before reading `slot.data`.
- The consumer releases the slot with `slot.seq.store(release, pos + Capacity)`, then advances the consumer cursor.

Use this primitive for many-producer, one-owner paths such as tool-call fan-in,
embedding/RAG ingest results, and cold progress events. Prefer SPSC when the
ownership model can be reduced to one producer and one consumer.

## MPMC queue (work dispatch)

File: `astral/src/concurrency/mpmc_queue.hpp`

### Design

- Bounded ring with `Capacity` (power of 2).
- Producers and consumers reserve positions via monotonically increasing tickets:
  - `enqueue_pos.fetch_add(1)`
  - `dequeue_pos.fetch_add(1)`
- Each slot contains a `seq` number that indicates whether the slot is ready for enqueue/dequeue.
- Waiting uses short spinning followed by `cpu_wait_for_event()`; successful operations call `cpu_signal_event()`.

### Memory-order contract

- Reservation tickets use `fetch_add(relaxed)`. Tickets only choose a slot; they do not publish data.
- Producers wait for `slot.seq.load(acquire) == pos`, write `slot.data`, then publish with `slot.seq.store(release, pos + 1)`.
- Consumers wait for `slot.seq.load(acquire) == pos + 1`, read `slot.data`, then release the slot with `slot.seq.store(release, pos + Capacity)`.
- `slot.seq` is the synchronization object. Data becomes visible only through the per-slot acquire/release pair.

This is the reviewed weak-memory contract for ARM and x86. Future edits that move data reads or writes across the acquire/release pair need a new review, a matching test change, and a benchmark note.

### API

- `enqueue_wait(const T&)` blocks until space is available.
- `dequeue_wait(T* out)` blocks until an item is available.

If non-blocking variants are needed in the future (`try_enqueue`/`try_dequeue`), they must still respect the “no compare-and-swap” constraint and should be introduced only with clear use-cases and benchmarks.

## SPSC ring (token streaming)

File: `astral/src/concurrency/spsc_ring.hpp`

### Design

- Single producer (decode thread) writes tokens.
- Single consumer (engine thread) reads tokens.
- Head/tail are separate cache-line aligned atomics to avoid false sharing.
- Producer signals on **empty → non-empty** to wake a waiting consumer.
- Consumer signals on **full → not-full** to wake a waiting producer.

### Memory-order contract

- The producer is the only writer of `head`; the consumer is the only writer of `tail`.
- Producer `push`: load `tail` with acquire to observe consumer progress, load `head` relaxed, write the element, then publish `head.store(release)`.
- Consumer `pop`: load `head` with acquire to observe producer data, load `tail` relaxed, read the element, then publish `tail.store(release)`.
- `size()` and `empty()` are approximate statistics and use relaxed loads only.
- `reset()` is a lifecycle operation. It must not run concurrently with `push()` or `pop()`.

This ring intentionally does not add ownership checks to the hot path. Callers enforce the single-producer/single-consumer rule before using the primitive.

## Validation

- Unit tests: `ctest --preset dev` and `ctest --preset release-with-tests`
- Sanitizers: TSAN tests are configured; on Linux x86_64 they run with ASLR disabled to avoid known TSAN mapping issues.
- Microbenchmarks: `./build/release-test/benchmarks/astral_benchmarks`
- Source contract gate: `gate_source_scans` checks the maintained MPMC/SPSC acquire/release prose against the source headers and this document.

Remaining release evidence:
- Multi-hour queue and streaming stress runs on release hardware.
- ARM hardware runs for the weak-memory contract.
- Cancellation and backpressure stress in the engine integration path.
