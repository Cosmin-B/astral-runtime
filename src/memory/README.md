# Astral Memory Subsystem

Memory allocators used by Astral's frame, pool, and tracking gates.

## Overview

This subsystem provides four core components:

1. **FrameAllocator** - Linear/bump allocator for per-frame short-lived objects
2. **ObjectPool** - Fixed-size shared recycling with a small spinlock-protected intrusive freelist
3. **LocalObjectPool** - Fixed-size owner-local recycling with no atomics or locks
4. **MemoryStats** - POD structure for tracking allocator usage

## Design Principles

- **Allocation-Gated Hot Paths**: Maintained gates track malloc/new calls in decode, sampling, and token streaming loops
- **Pre-commit Strategy**: All memory pre-committed at initialization; no vm_commit syscalls in hot paths
- **Explicit Memory Ordering**: All atomics use explicit memory_order (acquire/release/relaxed)
- **Cache-Line Aware**: Hot structures padded to prevent false sharing
- **Data-Oriented**: Designed for high-throughput, low-latency workloads

## Components

### FrameAllocator

Linear allocator for short-lived allocations with O(1) allocation and instant reset.

**Features**:
- Bump pointer allocation (align offset, increment)
- O(1) reset (invalidates all allocations)
- No per-object overhead
- NOT thread-safe (one per thread/session)

**Usage**:
```cpp
#include "memory/frame_allocator.hpp"

// Pre-commit memory (platform-specific)
void* memory = vm_commit(vm_reserve(capacity), capacity);
FrameAllocator alloc(memory, capacity);

// Allocate
void* p1 = alloc.alloc(64);           // Default 16-byte aligned
void* p2 = alloc.alloc(128, 32);      // 32-byte aligned

// Reset (instant; all allocations invalidated)
alloc.reset();

// Next allocation returns same address as first
void* p3 = alloc.alloc(64);           // p3 == p1
```

**Critical Requirements**:
- Memory MUST be pre-committed before passing to constructor
- Passing reserved-but-not-committed memory causes page faults
- No thread-safety; use one allocator per thread/session

### ObjectPool

Thread-safe object pool using an intrusive freelist protected by a small spinlock.

**Features**:
- Thread-safe acquire/release
- No dynamic allocation after construction
- No compare-and-swap loop in the pool path
- Fixed capacity (template parameter)
- Intrusive freelist (overwrites first 8 bytes on release)

**Usage**:
```cpp
#include "memory/object_pool.hpp"

struct Token {
    uint64_t id;
    uint64_t timestamp;
    char data[48];
};

ObjectPool<Token, 1024> pool;

// Acquire (thread-safe)
Token* tok = pool.acquire();
if (tok) {
    tok->id = 42;
    tok->timestamp = get_time();

    // Release back to pool (thread-safe)
    pool.release(tok);
}
```

**Critical Requirements**:
- Object type must be at least sizeof(void*) bytes
- Release overwrites first 8 bytes (do not rely on data preservation)
- Only release objects acquired from the same pool
- Best for cold/shared fixed-object recycling. Prefer thread-local pools or
  frame allocators for hot owner-local paths.

**Memory Ordering**:
- `memory_order_acquire` when taking the pool lock
- `memory_order_release` when releasing the pool lock

### LocalObjectPool

Owner-local object pool using the same intrusive freelist layout without locks,
atomics, syscalls, or dynamic allocation after construction.

**Features**:
- O(1) acquire and release
- Fixed capacity
- No synchronization
- No null check on release; callers release objects acquired from the same pool

**Usage**:
```cpp
#include "memory/object_pool.hpp"

struct ScratchNode {
    void* next;
    uint64_t value;
};

constexpr size_t kScratchNodeCapacity = 1024;
constexpr uint64_t kScratchValue = 42;

LocalObjectPool<ScratchNode, kScratchNodeCapacity> pool;
ScratchNode* node = pool.acquire();
if (node != nullptr) {
    node->value = kScratchValue;
    pool.release(node);
}
```

Use this for owner-local hot paths where ownership is already validated. Use
`ObjectPool` when several threads must share the same backing storage.

### MemoryStats

POD structure for tracking allocator usage (debugging, profiling, telemetry).

**Features**:
- 40 bytes (fits in single cache line)
- Trivially copyable (memcpy-friendly)
- Standard layout (C ABI compatible)

**Usage**:
```cpp
#include "memory/stats.hpp"

MemoryStats stats = {};
stats.bytes_reserved = 2ULL * 1024 * 1024 * 1024; // 2GB virtual
stats.bytes_committed = 4ULL * 1024 * 1024;        // 4MB physical
stats.bytes_used = 1ULL * 1024 * 1024;             // 1MB used
stats.alloc_count = 100;
stats.free_count = 50;

// Compute live allocations
uint64_t live = stats.alloc_count - stats.free_count; // 50
```

## Testing

Compile and run validation tests:

```bash
ctest --preset release-with-tests -R '^test_memory$' --output-on-failure
./build/release-test/benchmarks/astral_benchmarks --only alloc --alloc-iters 1000000 --alloc-size 64
./scripts/run_allocator_perf_capture.sh --iters 1000000 --size 64 --threads 4
```

Tests cover:
- FrameAllocator: basic allocation, reset, aligned output from aligned and unaligned backing memory, out-of-memory
- ObjectPool: basic acquire/release, capacity limits, thread-safety
- LocalObjectPool: basic acquire/release, capacity limits, reuse order
- MemoryStats: size validation, field access

## Performance Characteristics

### FrameAllocator
- **Allocation**: ~0.7ns (align + increment)
- **Reset**: ~0.3ns (single store)
- **Overhead**: 0 bytes per allocation
- **Cache**: Sequential access pattern (optimal)

### ObjectPool
- **Acquire/release (uncontended)**: measured by `--only alloc`
- **Acquire/release (contended)**: measured by `--only alloc --mpsc-producers <threads>`
- **Overhead**: 0 bytes (intrusive freelist)
- **Cache**: Random access pattern (pointer chasing)

### LocalObjectPool
- **Acquire/release**: measured by `--only alloc`
- **Overhead**: 0 bytes (intrusive freelist)
- **Cache**: Owner-local freelist, no cache-line transfer from synchronization

## Integration with Platform VM API

FrameAllocator requires pre-committed memory from platform VM layer:

```cpp
// Platform-specific (src/platform/vm.h)
void* vm_reserve(size_t size);
void  vm_commit(void* addr, size_t size);
void  vm_release(void* addr, size_t size);

// Usage
void* reserved = vm_reserve(capacity);
vm_commit(reserved, capacity);
FrameAllocator alloc(reserved, capacity);
```

See `src/platform/vm.h` for full VM API documentation.

## References

- Unity allocator: `/home/user/docs/unity-runtime/src/memory/`
- `docs/architecture/MEMORY_ARCHITECTURE.md` - allocator model and VM-backed arenas
- `docs/rules/CODING_STANDARDS.md` - memory management rules

## Design Rationale

### Why Pre-commit?

Page faults in hot paths cause microsecond stalls (unacceptable for real-time inference):
- Linux `mmap`: First access triggers page fault (~1-3μs)
- Windows `VirtualAlloc(MEM_COMMIT)`: Pre-commits pages (~10-50ns)

Pre-commit strategy:
- Initial commit: 2MB
- Growth: Double on capacity exhaustion (amortize syscall overhead)
- Hot path: Zero syscalls

### Why Intrusive Freelist?

ObjectPool uses intrusive freelist (next pointer in freed object):
- **Pro**: Zero per-object overhead
- **Pro**: Cache-friendly (no separate freelist allocation)
- **Con**: Overwrites first 8 bytes on release

Alternative (non-intrusive):
- **Pro**: Preserves object data
- **Con**: 8-16 bytes overhead per object
- **Con**: Extra allocation for freelist nodes

Decision: Performance over data preservation (user must re-initialize after acquire).

## Future Work

- **NUMA-aware ObjectPool**: Pin pools to NUMA nodes for multi-socket systems
- **Telemetry**: Export MemoryStats to engine profiler (Unity Profiler, Unreal Insights)
- **Huge pages**: Support 2MB/1GB pages for FrameAllocator backing memory
- **Debug mode**: Poison freed objects, detect use-after-free
