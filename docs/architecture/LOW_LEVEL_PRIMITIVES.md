# Low-Level Primitives

This document is an implementation map for Astral maintainers. These C++ types
are private and may change without an ABI revision. The public contract remains
[`include/astral_rt.h`](https://github.com/Cosmin-B/astral-runtime/blob/main/include/astral_rt.h).

## Selection Rules

Choose a primitive from the ownership topology before considering its peak
throughput:

| Topology | Primitive | Shared hot state |
| --- | --- | --- |
| One producer, one consumer | `SpscRing` | Published producer and consumer cursors |
| Fixed producers, one consumer | `SpscFanIn` | One SPSC lane per producer |
| Many producers, one consumer with immediate full feedback | `MpscRing` | Producer reservation cursor and per-slot sequence |
| Many producers, one consumer with accepted backpressure | `MpscTicketRing` | One ticket allocation and per-slot sequence |
| Many producers, many consumers | `MpmcQueue` | Producer/consumer tickets and per-slot sequence |
| One reader interval protecting many objects | `EpochManager` | Participant-local epoch announcement |

Prefer owner-local state over a more general queue. Fixed capacity and explicit
backpressure are part of each type's contract, not implementation details.

## Concurrency Primitives

The maintained implementations live under
[`src/concurrency`](https://github.com/Cosmin-B/astral-runtime/tree/main/src/concurrency):

- `SpscRing` uses cached remote cursors, batch operations, and event signaling.
  The producer and consumer each own their local cursor.
- `SpscFanIn` composes fixed SPSC lanes so producers never contend with one
  another. The consumer polls or batches across lanes.
- `MpscRing` offers non-blocking `try_push`; public API boundaries use it when
  saturation must return `ASTRAL_E_BUSY`.
- `MpscTicketRing` reserves positions with unconditional `fetch_add` and waits
  for slot ownership. It is restricted to internal paths where backpressure is
  acceptable.
- `MpmcQueue` uses tickets plus per-slot sequence values for bounded blocking
  transfer.
- `EventSpinLock` combines a short pause phase with the platform event hint and
  signals on unlock.
- `EpochManager` uses per-participant SPSC retirement queues and cache-line
  isolated epoch announcements. Queue overflow retains ownership with the
  caller; exactly one collector advances safe frontiers.

The exact acquire/release relationships and production uses are documented in
[the concurrency model](CONCURRENCY_MODEL.md).

## Platform Primitives

[`src/platform`](https://github.com/Cosmin-B/astral-runtime/tree/main/src/platform) owns
the OS and architecture boundary:

- `vm.h` provides reserve, commit, decommit, release, and best-effort large-page
  operations;
- `thread.h` provides worker creation, join, and hardware concurrency;
- `atomics.h` provides pause, wait-for-event, signal-event, and compiler fences;
- `cpu_features.*` performs runtime x86 AVX2/F16C detection and identifies the
  ARM NEON tier;
- `file_map.*` owns read-only file mapping;
- `time.h` supplies the monotonic tick clock.

Platform functions return explicit failure values. Callers must not update
allocation or mapping state until the corresponding platform operation has
succeeded.

## Memory Primitives

[`src/memory`](https://github.com/Cosmin-B/astral-runtime/tree/main/src/memory) contains
storage mechanisms rather than global
allocation policy:

- `FrameAllocator`: owner-local aligned bump allocation and reset;
- `LocalObjectPool`: owner-local fixed object reuse;
- `ObjectPool`: shared fixed object reuse under `EventSpinLock`;
- `MemoryStats`: plain accounting data.

The runtime size-class heap and session scratch partitioning are described in
[the memory architecture](MEMORY_ARCHITECTURE.md).

## Text And Diagnostics

[`src/utils`](https://github.com/Cosmin-B/astral-runtime/tree/main/src/utils) contains
strict UTF-8 spans and validation,
stack-backed string builders, logging, and profiling boundaries.

Use `StackStringBuilder` when truncation is acceptable and spill is forbidden.
Use `StringBuilder` only when the owner accepts a cold runtime-allocation spill.
Neither builder is shared across threads. Public ABI text remains pointer-plus-
length UTF-8; NUL termination is not part of span semantics.

## CPU Dispatch

Architecture-specific kernels must have a scalar implementation and a runtime
feature guard. Compile-time availability only determines whether a specialized
function can be built; it does not establish that the executing CPU or OS has
enabled the required state.

On x86, AVX2 conversion paths that use F16C require all of:

- CPUID XSAVE, OSXSAVE, and AVX;
- XCR0 XMM and YMM state enabled by the OS;
- CPUID AVX2;
- CPUID F16C.

On ARM, current dispatch identifies the NEON tier. Any future optional ISA
extension needs its own runtime capability evidence before use.

## Review Checklist

For a new or changed primitive:

1. State producer, consumer, and collector ownership.
2. State capacity and saturation behavior.
3. Identify the atomic operation that publishes data.
4. Identify which thread owns each mutable cursor.
5. Keep blocking or retry behavior out of public non-blocking contracts.
6. Add weak-memory, wraparound, saturation, and invalid-input tests.
7. Measure the actual production topology, not only an uncontended loop.
8. Keep raw disassembly and hardware-counter captures outside the product
   repository.

## Validation

```bash
cmake --build --preset release-with-tests -j --target \
  test_platform test_memory test_concurrency test_utf8 test_inference
ctest --preset release-with-tests --output-on-failure \
  -R '^(test_platform|test_memory|test_concurrency|test_utf8|test_inference)$'
```

Run `scripts/run_tsan.sh` for ownership or ordering changes and retain the raw
sanitizer and performance evidence outside the repository.
