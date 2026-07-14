# Memory Architecture

Astral separates core runtime allocation, per-session scratch storage, caller
buffers, and provider-owned memory. This document describes the implemented
core contract. Provider libraries may allocate independently unless their own
documentation states otherwise.

## Initialization Modes

`AstralInit.memory_mode` selects one of three modes:

| Mode | Backing storage | Exhaustion behavior |
| --- | --- | --- |
| `ASTRAL_MEMMODE_VM` | Astral reserves virtual address space and commits pages as the core heap grows | Allocation fails if reserve or commit fails |
| `ASTRAL_MEMMODE_ARENA_BORROWED` | Caller supplies one arena that Astral never frees | Astral-owned allocation returns `ASTRAL_E_NOMEM`; it does not spill to the host allocator |
| `ASTRAL_MEMMODE_ARENA_OWNED` | Astral obtains one arena through the configured allocator or `malloc` | Same fixed-boundary behavior after initialization |

VM mode is zero and therefore the default for a zero-initialized descriptor.
Every caller must set `AstralInit.size = sizeof(AstralInit)`.

## VM Mode

The default reserve is 2 GiB when `reserve_bytes` is zero. Astral initially
commits up to 2 MiB and grows the committed portion of the core heap on cold
allocation paths. Linux, macOS, and Windows implement the same
reserve/commit/decommit/release interface with platform-native APIs.

`enable_hugepages` is best effort. Linux requests transparent huge pages,
Windows attempts a large-page allocation and falls back to normal pages, and
macOS reports explicit large pages as unsupported. Failure to obtain huge pages
does not fail normal initialization; failure to reserve or commit normal pages
does.

VM growth belongs to allocation paths, not decode, sampling, or stream-copy
loops. The maintained allocation and syscall gates enforce the supported
steady-state paths on platforms that provide the required interception
mechanism.

## Arena Layout

Arena modes partition the supplied region in this order:

1. worker-local scratch storage;
2. the Astral core size-class heap;
3. fixed-size session scratch blocks.

The default worker scratch allowance is 256 KiB per worker. The default core
heap is 2 MiB. The remaining bytes back session blocks, whose size defaults to
2 MiB. `AstralArenaDesc._reserved[0]` and `_reserved[1]` currently configure
the worker scratch and core heap sizes; the public header is the authoritative
layout contract.

Arena mode defaults to one worker when `thread_count` is zero. Configuration
fails if the requested partitions do not fit. Session creation fails with
`ASTRAL_E_NOMEM` when no suitable fixed block remains, and destroying the
session returns its block to the pool.

## Core Heap

Astral-owned objects use a bounded size-class allocator implemented in
`src/core/init.cpp`:

- small and medium blocks are grouped into power-of-two subdivisions;
- each thread keeps owner-local free lists for common allocation and release;
- cold refill and flush operations use one lock per shared bucket;
- the arena cursor grows under a separate lock;
- free operations require the original size and alignment, so blocks need no
  per-allocation header;
- arena-backed allocation never falls through to the configured system
  allocator after exhaustion.

In VM mode, requests outside the core heap's handled range may use the
configured system allocator. In arena modes, the arena is a hard ownership and
capacity boundary for Astral-owned objects.

## Short-Lived Storage

The runtime also provides focused fixed-capacity primitives:

- `FrameAllocator` is an owner-local bump allocator with constant-time reset;
- `LocalObjectPool` is an owner-local intrusive free list with no atomic
  operations;
- `ObjectPool` is a shared fixed-capacity pool whose free list is protected by
  a small event spin lock;
- `StackStringBuilder` keeps bytes inline and truncates on overflow;
- `StringBuilder` keeps bytes inline and may spill through the runtime allocator
  on a cold overflow path.

These types do not acquire backing memory themselves. Their owner supplies or
constructs the storage before entering the hot path.

## Caller And Provider Memory

Input and output spans are borrowed for the duration documented by each ABI
operation. Streaming reads write into caller-owned buffers. Astral handles own
core metadata but do not transfer ownership of caller buffers.

The llama.cpp provider owns model weights, KV state, and provider scratch under
its own allocation contract. Configuring an Astral arena does not imply that a
third-party provider is allocation-free or arena-backed.

## Accounting

`runtime_memory_stats` reports the core region:

- VM mode reports committed and reserved core heap bytes;
- arena modes report the fixed arena size for both values.

Provider memory and mapped model files are outside those counters. Use process
RSS and provider-specific telemetry when measuring total application memory.

## Failure Rules

- Allocation functions return null or an Astral error; they do not silently
  continue after an OS commit failure.
- Arena exhaustion never falls back to host allocation for Astral-owned data.
- A failed session scratch acquisition leaves ownership unchanged.
- Caller-provided arena storage must outlive `astral_shutdown` in borrowed mode.
- All handles must be released before shutdown.

## Validation

```bash
cmake --build --preset release-with-tests -j --target \
  test_arena test_memory gate_allocations gate_io_syscalls gate_rss_cap
ctest --preset release-with-tests --output-on-failure \
  -R '^(test_arena|test_memory|gate_allocations|gate_io_syscalls|gate_rss_cap)$'
```

Platform-specific interception gates are only evidence on platforms where the
build enables their tracking implementation. Release claims also require the
sanitizer and soak lanes in the release acceptance matrix.
