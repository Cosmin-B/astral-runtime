# Platform Abstraction Layer

Low-level OS abstractions for virtual memory management and CPU utilities.

## Overview

The platform layer provides portable interfaces to OS-specific functionality:

- **Virtual Memory (`vm.h`)**: Reserve/commit/decommit/release address space
- **Atomics (`atomics.h`)**: Cache line detection, CPU pause, compiler fences

All implementations follow the same contract across Linux, Windows, and macOS.

## Virtual Memory API

### Design Philosophy

Virtual memory management follows a two-phase allocation model:

1. **Reserve**: Allocate address space (no physical memory)
2. **Commit**: Allocate physical memory for reserved pages

This enables:
- Large upfront address space reservation (2GB+) without consuming RAM
- Incremental memory commitment as needed
- Decommit to release physical memory without releasing address space

### Functions

#### `void* vm_reserve(size_t size)`

Reserve virtual address space without committing physical memory.

- **Linux**: `mmap(PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE)`
- **Windows**: `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)`
- **macOS**: `mmap(PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE)`

**Returns**: Base address of reserved region, or `nullptr` on failure

**Notes**:
- Pages are inaccessible until committed
- Does not consume physical memory, only address space
- Size should be page-aligned (typically 4KB)

#### `bool vm_commit(void* addr, size_t size)`

Commit physical memory to reserved pages.

- **Linux**: `mprotect(PROT_READ | PROT_WRITE) + madvise(MADV_WILLNEED)`
- **Windows**: `VirtualAlloc(MEM_COMMIT, PAGE_READWRITE)`
- **macOS**: `mprotect(PROT_READ | PROT_WRITE) + madvise(MADV_NORMAL)`

**Parameters**:
- `addr`: Address within reserved region (must be page-aligned)
- `size`: Size to commit (must be multiple of page size)

**Returns**: `true` when the range is accessible, or `false` when the OS rejects the request

**Notes**:
- Safe to call on already-committed pages (no-op)
- May fail if system is out of memory
- **CRITICAL**: Do NOT call in hot paths (see Pre-Commit Strategy below)

#### `void vm_decommit(void* addr, size_t size)`

Decommit pages (release physical memory, keep address space reserved).

- **Linux**: `madvise(MADV_DONTNEED) + mprotect(PROT_NONE)`
- **Windows**: `VirtualFree(MEM_DECOMMIT)`
- **macOS**: `madvise(MADV_FREE_REUSABLE) + mprotect(PROT_NONE)`

**Parameters**:
- `addr`: Address within reserved region (must be page-aligned)
- `size`: Size to decommit (must be multiple of page size)

**Notes**:
- Releases physical memory back to OS
- Address space remains reserved
- Accessing decommitted pages causes page fault

#### `void vm_release(void* addr, size_t size)`

Release entire virtual address space reservation.

- **Linux**: `munmap()`
- **Windows**: `VirtualFree(MEM_RELEASE)` (size must be 0)
- **macOS**: `munmap()`

**Parameters**:
- `addr`: Base address from `vm_reserve()` (must be original address)
- `size`: Size from `vm_reserve()` (must match original size)

**Notes**:
- Releases both physical memory and address space
- Pointer becomes invalid after this call
- On Windows, size is ignored (must pass 0)

#### `bool vm_try_hugepages(void* addr, size_t size)`

Attempt to use huge pages (2MB/1GB) for better TLB performance.

- **Linux**: `madvise(MADV_HUGEPAGE)` (requires THP enabled)
- **Windows**: Retroactive promotion is not supported (returns `false`)
- **macOS**: Not supported (returns `false`)

**Parameters**:
- `addr`: Address of committed region (must be huge-page-aligned, typically 2MB)
- `size`: Size in bytes (must be multiple of huge page size)

**Returns**: `true` if huge pages were enabled, `false` otherwise

**Notes**:
- Linux returns `true` only when the `MADV_HUGEPAGE` request is accepted.
- On Linux, huge-page backing still depends on THP/kernel policy or `vm.nr_hugepages`.
- On Windows, huge pages must be requested at allocation time (not retroactively)
- `false` means the region remains backed by regular pages.

#### `void* vm_reserve_large(size_t size, size_t* out_size)`

Reserve a large-page region for initialization-time memory pools.

- **Linux**: reserves a 2 MiB-aligned region, commits it, and applies `MADV_HUGEPAGE`
- **Windows**: calls `VirtualAlloc(MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES)`
- **macOS**: returns `nullptr`

**Parameters**:
- `size`: Requested byte size
- `out_size`: Optional actual allocation size after platform rounding

**Notes**:
- Windows requires `SeLockMemoryPrivilege`; failure falls back to normal pages.
- A successful Windows large-page allocation is already committed and cannot be
  decommitted in pieces; release it with `vm_release()`.
- This is not a hot-path primitive.

### Pre-Commit Strategy

**CRITICAL**: Never call `vm_commit()` in hot paths (decode/sampling loops).

Page faults cause microsecond stalls that are unacceptable for real-time inference.

**Recommended approach**:
1. Pre-commit 2MB during initialization
2. If allocator grows, double commit size during setup phase
3. Never commit during steady-state operation

Example:
```cpp
// Initialization (NOT hot path)
void* base = vm_reserve(2GB);
if (base == nullptr || !vm_commit(base, 2MB)) {
  return false;
}

// Allocator setup (NOT hot path)
if (used > committed) {
  if (!vm_commit(base + committed, committed)) {
    return false;
  }
  committed *= 2;
}

// Hot path (NO vm_commit calls!)
void* alloc = bump_allocate(allocator, size);  // No syscalls
```

### Platform Differences

| Feature | Linux | Windows | macOS |
|---------|-------|---------|-------|
| Reserve | mmap | VirtualAlloc | mmap |
| Commit | mprotect | VirtualAlloc | mprotect |
| Decommit | madvise | VirtualFree | madvise |
| Release | munmap | VirtualFree | munmap |
| Huge pages | MADV_HUGEPAGE | MEM_LARGE_PAGES | Not supported |
| Page size | 4KB | 4KB | 4KB (Intel), 16KB (ARM) |

### Error Handling

These primitives do not throw:

- `vm_reserve()` returns `nullptr` on failure
- `vm_commit()` returns `false` and leaves pages inaccessible if the OS rejects the request
- `vm_decommit()` may leave pages resident if the OS rejects the decommit request
- `vm_release()` expects the original reservation base and size; invalid inputs are caller bugs
- Accessing uncommitted memory faults at the OS boundary
- Memory exhaustion is treated as a fatal runtime condition by callers that pre-commit pools

For debugging, enable OS-level tracing:
- **Linux**: `strace -e mmap,mprotect,munmap,madvise ./program`
- **Windows**: Process Monitor (filter on VirtualAlloc/VirtualFree)
- **macOS**: `dtruss -n program` or `vmmap <pid>`

## Atomics API

### Functions

#### `size_t cache_line_size()`

Get L1 data cache line size for the current CPU.

**Detection strategy**:
- **x86**: CPUID instruction (leaf 0x80000006, ECX bits 7:0)
- **ARM64**: Read CTR_EL0 register (DminLine field)
- **ARM32**: Read CTR register via CP15
- **Fallback**: 64 bytes (safe default)

**Returns**: Cache line size in bytes (typically 64 or 128)

**Typical values**:
- x86/x64: 64 bytes
- ARM Cortex-A: 64 bytes
- Apple Silicon (M1/M2): 128 bytes

**Usage**: Pad shared atomics to prevent false sharing.

Example:
```cpp
struct alignas(cache_line_size()) MpmcNode {
  std::atomic<uint64_t> head;
  char padding[cache_line_size() - sizeof(uint64_t)];
  std::atomic<uint64_t> tail;
};
```

#### `void cpu_pause()`

Pause instruction for spin-wait loops.

**Platform implementation**:
- **x86**: `PAUSE` instruction (rep; nop)
- **ARM**: `YIELD` instruction
- **Fallback**: compiler fence

**Benefits**:
- Reduces power consumption during busy-waiting
- Improves performance by reducing memory order violations
- Prevents speculative execution overhead

**Usage**:
```cpp
while (!ready.load(std::memory_order_acquire)) {
  cpu_pause();  // Reduce contention and power usage
}
```

#### `void compiler_fence()`

Compiler barrier (prevents instruction reordering by compiler only).

**Platform implementation**:
- **GCC/Clang**: `asm volatile("" ::: "memory")`
- **MSVC**: `_ReadWriteBarrier()`
- **Fallback**: `std::atomic_signal_fence(std::memory_order_acq_rel)`

**Important**: This is NOT a memory barrier. It only prevents compiler reordering.

**Usage**: Rare. Use `std::atomic` with explicit memory ordering for multi-threaded code.

## Testing

Run validation tests:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -g -Isrc \
  src/platform/vm_linux.cpp \
  src/platform/atomics.cpp \
  src/platform/test_platform.cpp \
  -o test_platform

./test_platform
```

Expected output:
```
=== Astral Platform Abstraction Layer Tests ===

Testing vm_reserve/commit/decommit/release...
  [PASS] Basic VM operations

Testing vm_try_hugepages...
  [PASS] Huge pages test (graceful fallback)

Testing cache_line_size...
  [PASS] Cache line size detection

=== All tests passed! ===
```

## Build Integration

CMakeLists.txt (minimal example):

```cmake
# Platform-specific sources
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(PLATFORM_SOURCES src/platform/vm_linux.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(PLATFORM_SOURCES src/platform/vm_win32.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(PLATFORM_SOURCES src/platform/vm_macos.cpp)
else()
  message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# Common sources
add_library(astral_platform
  ${PLATFORM_SOURCES}
  src/platform/atomics.cpp
)

target_include_directories(astral_platform PUBLIC src)
target_compile_features(astral_platform PUBLIC cxx_std_17)
```

## Performance Characteristics

### Virtual Memory Operations

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| `vm_reserve()` | 1-10 μs | Syscall overhead |
| `vm_commit()` | 10-100 μs | Page fault + syscall |
| `vm_decommit()` | 5-50 μs | Syscall + TLB flush |
| `vm_release()` | 1-10 μs | Syscall overhead |
| `vm_try_hugepages()` | 10-100 us | Kernel may reject the promotion request |

**Conclusion**: Never call in hot paths (decode/sampling loops).

### Atomics Utilities

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| `cache_line_size()` | <1 ns | Cached (first call: ~100 ns) |
| `cpu_pause()` | ~10 cycles | x86: 10 cycles, ARM: 1 cycle |
| `compiler_fence()` | 0 cycles | Compile-time only |

**Conclusion**: Safe to use in hot paths.

## References

### Virtual Memory

- **Linux mmap**: [man 2 mmap](https://man7.org/linux/man-pages/man2/mmap.2.html)
- **Linux madvise**: [man 2 madvise](https://man7.org/linux/man-pages/man2/madvise.2.html)
- **Windows VirtualAlloc**: [MSDN VirtualAlloc](https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc)
- **macOS mach_vm**: [XNU source](https://github.com/apple/darwin-xnu)

### Atomics

- **x86 PAUSE**: Intel SDM Vol. 2B, section 4.3
- **ARM YIELD**: ARM Architecture Reference Manual, section C6.2.339
- **Cache line detection**: [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) (Ulrich Drepper, 2007)

### Unity allocator Reference

- `/home/user/docs/unity-runtime/src/memory/`
- `/home/user/docs/unity-runtime/src/platform/`

## License

Part of the Astral project. See main LICENSE file for details.
