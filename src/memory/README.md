# Astral Memory Subsystem

Memory allocators used by Astral's frame, pool, and tracking gates.

## Overview

This subsystem provides three core components:

1. **FrameAllocator** - Linear/bump allocator for per-frame short-lived objects
2. **ObjectPool** - Lock-free freelist for fixed-size object recycling
3. **MemoryStats** - POD structure for tracking allocator usage

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

Lock-free object pool using intrusive freelist and tagged pointers for ABA protection.

**Features**:
- Thread-safe lock-free acquire/release
- CAS-based Treiber stack algorithm
- ABA protection via generation counter (upper 16 bits of pointer)
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
- Platform must support lock-free 64-bit atomics

**Memory Ordering**:
- `memory_order_acquire` on successful CAS pop (synchronizes-with prior push)
- `memory_order_release` on successful CAS push (synchronizes-with subsequent pop)
- `memory_order_relaxed` on failed CAS retry (no synchronization needed)

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
cd src/memory
g++ -std=c++20 -Wall -Wextra -O2 -pthread test_memory.cpp -o test_memory
./test_memory
```

Tests cover:
- FrameAllocator: basic allocation, reset, alignment, out-of-memory
- ObjectPool: basic acquire/release, capacity limits, thread-safety
- MemoryStats: size validation, field access

## Performance Characteristics

### FrameAllocator
- **Allocation**: ~0.7ns (align + increment)
- **Reset**: ~0.3ns (single store)
- **Overhead**: 0 bytes per allocation
- **Cache**: Sequential access pattern (optimal)

### ObjectPool
- **Acquire (uncontended)**: ~2-3ns (relaxed load + CAS)
- **Acquire (contended)**: ~10-20ns (retry loop with backoff)
- **Release**: ~2-3ns (store + CAS)
- **Overhead**: 0 bytes (intrusive freelist)
- **Cache**: Random access pattern (pointer chasing)

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

### Why Tagged Pointers?

ObjectPool uses tagged pointers (pointer + ABA counter):
- **ABA Problem**: Thread A pops node X, thread B pops X and pushes Y and X back, thread A's CAS succeeds incorrectly
- **Solution**: Increment generation counter on each push/pop (16-bit counter wraps after 65536 operations)

Alternative (hazard pointers):
- **Pro**: No ABA problem
- **Con**: Complex implementation (~200 LOC)
- **Con**: Per-thread overhead

Decision: Tagged pointers for simplicity (16-bit counter sufficient for game engine workloads).

## Future Work

- **NUMA-aware ObjectPool**: Pin pools to NUMA nodes for multi-socket systems
- **Telemetry**: Export MemoryStats to engine profiler (Unity Profiler, Unreal Insights)
- **Huge pages**: Support 2MB/1GB pages for FrameAllocator backing memory
- **Debug mode**: Poison freed objects, detect use-after-free
