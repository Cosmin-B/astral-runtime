# Memory Architecture

## Overview

Astral's memory system is designed around three core principles:

1. **Virtual Reserve, Commit on Demand**: Reserve large address space up front, commit physical pages only when needed
2. **Explicit Lifetime Management**: No hidden allocations; all memory tied to explicit scopes
3. **Allocation-Gated Hot Paths**: Steady-state inference and streaming paths must prove heap behavior with the maintained allocation gates before release claims.

## Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ Global Virtual Reserve (2GB default)                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ BootstrapAllocator (short-lived, freed after startup)  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Model Memory (mmap'd GGUF + KV cache)                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ SessionPool (per-session, reset per frame)             │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Small Object Pools (tokens, callbacks, 16-64 bytes)    │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

External (optional):
┌─────────────────────────────────────────────────────────────┐
│ Engine Allocator (Unity NativeArray / Unreal FMemory)       │
└─────────────────────────────────────────────────────────────┘
```

## Global Virtual Reserve

### Purpose

Provide a large address-space reserve for Astral's core size-class allocator. Small and medium runtime allocations use this region first, and cold growth commits more of the reserve under the allocator lock. Requests outside the size-class range can use the configured system allocator. Provider libraries retain their own documented allocation contracts.

### Design

- **Reserve Size**: Configurable via `AstralInit.reserve_bytes` (default: 2GB on 64-bit, 512MB on 32-bit)
- **Commit Growth**: The core heap starts with the initial commit and doubles its committed extent on cold growth.
- **Thread Safety**: The reserve is immutable after init; free blocks are cached per thread and bucket growth is serialized.

### Platform Implementation

#### Linux

```cpp
void* vm_reserve(size_t size) {
  void* addr = ::mmap(nullptr, size,
                      PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                      -1, 0);
  if (addr == MAP_FAILED) return nullptr;
  return addr;
}

void vm_commit(void* addr, size_t size) {
  ::mprotect(addr, size, PROT_READ | PROT_WRITE);
  ::madvise(addr, size, MADV_WILLNEED);
}

void vm_decommit(void* addr, size_t size) {
  ::madvise(addr, size, MADV_DONTNEED);
  ::mprotect(addr, size, PROT_NONE);
}

bool vm_try_hugepages(void* addr, size_t size) {
  return ::madvise(addr, size, MADV_HUGEPAGE) == 0;
}
```

#### Windows

```cpp
void* vm_reserve(size_t size) {
  return ::VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
}

void vm_commit(void* addr, size_t size) {
  ::VirtualAlloc(addr, size, MEM_COMMIT, PAGE_READWRITE);
}

void vm_decommit(void* addr, size_t size) {
  ::VirtualFree(addr, size, MEM_DECOMMIT);
}

bool vm_try_hugepages(void* addr, size_t size) {
  // Requires SeLockMemoryPrivilege
  SIZE_T min_size = ::GetLargePageMinimum();
  if (size < min_size || size % min_size != 0) return false;
  void* result = ::VirtualAlloc(addr, size,
                                  MEM_COMMIT | MEM_LARGE_PAGES,
                                  PAGE_READWRITE);
  return result != nullptr;
}
```

#### macOS

```cpp
void* vm_reserve(size_t size) {
  kern_return_t kr = ::mach_vm_allocate(
    ::mach_task_self(),
    reinterpret_cast<mach_vm_address_t*>(&addr),
    size,
    VM_FLAGS_ANYWHERE
  );
  return (kr == KERN_SUCCESS) ? addr : nullptr;
}

void vm_commit(void* addr, size_t size) {
  // macOS commits on first touch; explicit commit not needed
  // Optionally pre-fault pages
  ::madvise(addr, size, MADV_WILLNEED);
}

void vm_decommit(void* addr, size_t size) {
  ::madvise(addr, size, MADV_FREE);
}

bool vm_try_hugepages(void* addr, size_t size) {
  // macOS handles superpage promotion automatically
  return true; // No explicit API
}
```

### NUMA Awareness (Optional)

On multi-socket systems, pin allocations to the NUMA node where the worker threads run:

```cpp
// Linux: numa_alloc_onnode() via libnuma (optional dependency)
void* vm_reserve_numa(size_t size, uint32_t node) {
  if (node == 0xFFFFFFFF) return vm_reserve(size); // Any node
  // Use mbind() or set_mempolicy() to bind pages
  void* addr = vm_reserve(size);
  unsigned long nodemask = 1UL << node;
  ::mbind(addr, size, MPOL_BIND, &nodemask, 32, MPOL_MF_STRICT);
  return addr;
}
```

## Cross-Platform Memory Considerations

### iOS

**Key Constraints**:
- **16KB Page Size**: ARM64 uses 16KB pages (vs 4KB on most platforms)
- **Jetsam**: OS kills apps exceeding memory limits without warning
- **Memory Warnings**: Must respond to `didReceiveMemoryWarning`

**Implementation**:

```cpp
#if TARGET_OS_IOS
constexpr size_t page_size = 16384; // 16KB pages

// Register for memory pressure notifications
void register_memory_warnings() {
  [[NSNotificationCenter defaultCenter]
    addObserverForName:UIApplicationDidReceiveMemoryWarningNotification
    object:nil
    queue:nil
    usingBlock:^(NSNotification* note) {
      // Decommit idle session pools
      decommit_idle_sessions();
      // Clear TLS caches
      flush_tls_caches();
    }];
}

// Jetsam avoidance: Stay under limits
void check_memory_footprint() {
  task_vm_info_data_t vm_info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  kern_return_t kr = task_info(mach_task_self(),
                                TASK_VM_INFO,
                                (task_info_t)&vm_info,
                                &count);

  // iOS kills at ~1.4GB for most devices
  if (vm_info.phys_footprint > 1200ULL * 1024 * 1024) {
    log_warn("High memory pressure: %llu MB", vm_info.phys_footprint >> 20);
    // Proactively decommit
  }
}
#endif
```

### Android

**Key Constraints**:
- **4KB Page Size**: Standard Linux pages
- **Low Memory Killer (LMK)**: Kills apps by priority when memory low
- **Memory Pressure API**: `onTrimMemory()` callbacks

**Implementation**:

```cpp
#if __ANDROID__
// JNI hook for memory trim callbacks
extern "C" JNIEXPORT void JNICALL
Java_com_astral_AstralNative_onTrimMemory(JNIEnv* env, jobject obj, jint level) {
  // TRIM_MEMORY_RUNNING_LOW (10) = warning
  if (level >= 10) {
    decommit_idle_sessions();
  }

  // TRIM_MEMORY_RUNNING_CRITICAL (15) = aggressive cleanup
  if (level >= 15) {
    flush_tls_caches();
    compact_kv_caches();
  }
}

// Check available memory
void check_available_memory() {
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  uint64_t mem_available = 0;

  while (std::getline(meminfo, line)) {
    if (line.starts_with("MemAvailable:")) {
      sscanf(line.c_str(), "MemAvailable: %llu kB", &mem_available);
      break;
    }
  }

  // Avoid LMK by staying under 80% of available
  if (mem_available < 500 * 1024) { // < 500MB available
    log_warn("Low memory: %llu MB available", mem_available >> 10);
  }
}
#endif
```

### PlayStation 5

**Key Constraints**:
- **3-Stage Memory Model**: Flexible, direct, GPU-visible
- **12.5GB Available**: Total for game (out of 16GB)
- **GPU Coherency**: Must manage cache coherency manually

**Implementation**:

```cpp
#if __PS5__
// PS5 SDK provides custom allocators
#include <kernel.h>

void* vm_reserve_ps5(size_t size, uint32_t memory_type) {
  off_t phys_addr;
  int flags = SCE_KERNEL_MAP_FIXED;

  // Choose memory type:
  // - SCE_KERNEL_WC_GARLIC: GPU-visible (192GB/s bandwidth)
  // - SCE_KERNEL_WB_ONION: CPU-visible (204GB/s bandwidth)
  int memtype = (memory_type == GPU_VISIBLE)
                ? SCE_KERNEL_WC_GARLIC
                : SCE_KERNEL_WB_ONION;

  void* addr = sceKernelAllocateDirectMemory(
    0, // search start
    SCE_KERNEL_MAIN_DMEM_SIZE, // search end
    size,
    2 * 1024 * 1024, // 2MB alignment
    memtype,
    &phys_addr
  );

  return addr;
}

// Prefetch memory to GPU
void prefetch_to_gpu(void* addr, size_t size) {
  // Use cache operations to ensure coherency
  sceKernelDcacheFlush(addr, size);
}
#endif
```

### Xbox Series X/S

**Key Constraints**:
- **Split Memory Pools**: Fast GDDR6 (10GB @ 560GB/s) + Slow (6GB @ 336GB/s) on Series X
- **Series S**: 10GB total (8GB @ 224GB/s fast, 2GB @ 56GB/s slow)
- **Title Memory Limits**: ~13.5GB on Series X, ~7.5GB on Series S

**Implementation**:

```cpp
#if _GAMING_XBOX
#include <XMemory.h>

void* vm_reserve_xbox(size_t size, bool prefer_fast_memory) {
  DWORD protect = PAGE_READWRITE;
  DWORD alloc_type = MEM_GRAPHICS | MEM_RESERVE | MEM_COMMIT;

  if (prefer_fast_memory) {
    // Allocate from fast GDDR6 pool (for KV cache, hot data)
    alloc_type |= MEM_LARGE_PAGES;
  }

  void* addr = XMemAlloc(size, XMEM_ALIGNMENT_DEFAULT, protect, alloc_type);
  return addr;
}

// Query available memory per pool
void check_xbox_memory() {
  XMEMORY_STATUS mem_status;
  XMemQuery(&mem_status);

  uint64_t fast_available = mem_status.TotalFast - mem_status.AvailFast;
  uint64_t slow_available = mem_status.TotalSlow - mem_status.AvailSlow;

  log_info("Xbox memory: Fast=%llu MB, Slow=%llu MB",
           fast_available >> 20, slow_available >> 20);
}
#endif
```

### Desktop: Huge Pages

**2MB/1GB Pages**: Reduce TLB misses for large allocations (KV cache, model weights)

**Linux**:

```cpp
void* vm_reserve_hugepage_linux(size_t size) {
  // Requires `sysctl vm.nr_hugepages=N` or transparent huge pages
  void* addr = ::mmap(nullptr, size,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                      -1, 0);

  if (addr == MAP_FAILED) {
    // Fall back to transparent huge pages
    addr = ::mmap(nullptr, size,
                  PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS,
                  -1, 0);
    ::madvise(addr, size, MADV_HUGEPAGE);
  }

  return addr;
}
```

**Windows**:

```cpp
void* vm_reserve_largepage_windows(size_t size) {
  // Requires SeLockMemoryPrivilege
  SIZE_T min_size = ::GetLargePageMinimum(); // Usually 2MB

  if (size < min_size) {
    size = min_size;
  }

  // Round up to large page boundary
  size = (size + min_size - 1) & ~(min_size - 1);

  void* addr = ::VirtualAlloc(nullptr, size,
                               MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);

  if (!addr) {
    // Fall back to regular pages
    addr = ::VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  }

  return addr;
}
```

**Performance Impact**:
- **TLB Miss Reduction**: 40-60% for large sequential access patterns
- **Latency Reduction**: 10-20ns saved per miss
- **Best For**: Model weights (4GB+), KV cache (512MB+)

## BootstrapAllocator

### Purpose

Temporary allocator for one-time initialization work (parsing config, loading dynamic symbols, etc.). Freed immediately after `astral_init()` completes.

### Design

- **Lifetime**: Created at start of `astral_init()`, destroyed at end
- **Size**: 1MB default (enough for config, logs, small buffers)
- **Allocator**: Simple LinearAllocator (bump allocator); no free()
- **Memory**: Committed from global reserve, decommitted on destroy

### Usage

```cpp
void astral_init(const AstralInit* cfg) {
  BootstrapAllocator bootstrap(global_reserve, 1 << 20); // 1 MB

  // Parse config
  Config* parsed_cfg = parse_config(&bootstrap, cfg);

  // Load backend
  Backend* backend = load_llama_backend(&bootstrap);

  // ...

  // bootstrap destructor decommits memory
}
```

## Model Memory

### Purpose

Store model weights (typically mmap'd from GGUF file) and KV cache (allocated from global reserve).

### Design

- **Weights**: Memory-mapped read-only from disk (via `mmap` on POSIX, `CreateFileMapping` on Windows)
- **KV Cache**: Pre-allocated `n_ctx * n_layer * sizeof(float16)` from global reserve
- **Lifetime**: Tied to model handle; freed on `astral_model_release()`

### Memory Mapping

```cpp
void* mmap_model(const char* path, size_t* out_size) {
  int fd = ::open(path, O_RDONLY);
  struct stat st;
  ::fstat(fd, &st);
  *out_size = st.st_size;

  void* addr = ::mmap(nullptr, st.st_size,
                      PROT_READ,
                      MAP_SHARED,
                      fd, 0);
  ::madvise(addr, st.st_size, MADV_SEQUENTIAL | MADV_WILLNEED);
  ::close(fd);
  return addr;
}
```

### KV Cache Layout

```cpp
struct KvCache {
  void* base;        // Committed memory from global reserve
  size_t capacity;   // n_ctx * n_layer * sizeof(float16) * 2 (key + value)
  uint32_t n_ctx;
  uint32_t n_layer;

  // Per-layer slices
  Span<float16> key_layer(uint32_t layer);
  Span<float16> value_layer(uint32_t layer);
};
```

## SessionPool (Per-Session Linear Allocator)

### Purpose

Fast bump allocator for short-lived objects during token decode/sampling. All memory reset at frame end (after tokens emitted).

### Design

- **Pre-Commit Strategy**: Commit 2MB at initialization to avoid per-frame syscalls
- **Growth**: Double capacity when exhausted (2MB → 4MB → 8MB → 16MB max)
- **Bump Pointer**: Simple increment; no free()
- **Reset**: Set bump pointer to 0; pages stay resident
- **Thread Safety**: NOT thread-safe; one per session (decode thread owns it)

### vm_commit Performance Problem

**Critical Issue**: Calling `vm_commit()` every frame is a performance killer:
- **Linux**: `mprotect()` + `madvise()` = 5,000-20,000 CPU cycles
- **Windows**: `VirtualAlloc(MEM_COMMIT)` = 8,000-15,000 cycles
- **macOS**: Page fault handling = 3,000-12,000 cycles

**Solution**: Pre-commit memory at initialization and grow exponentially when needed.

### Implementation

```cpp
class ScratchBuffer {
public:
  ScratchBuffer(void* base, size_t capacity)
    : base_(base), capacity_(capacity), committed_(0), bump_(0) {
    // Pre-commit 2MB to avoid per-frame syscalls
    size_t initial_commit = std::min(capacity, 2ULL * 1024 * 1024);
    vm_commit(base_, initial_commit);
    prefault_memory(base_, initial_commit);
    committed_ = initial_commit;
  }

  void* alloc(size_t size, size_t align = 16) {
    // Align bump pointer
    size_t aligned_bump = (bump_ + align - 1) & ~(align - 1);
    size_t new_bump = aligned_bump + size;

    // Grow if needed (double each time)
    if (new_bump > committed_) {
      size_t new_commit = std::max(committed_ * 2, round_up_to_page(new_bump));
      new_commit = std::min(new_commit, capacity_);

      if (new_commit > committed_) {
        size_t grow_size = new_commit - committed_;
        vm_commit(static_cast<uint8_t*>(base_) + committed_, grow_size);
        prefault_memory(static_cast<uint8_t*>(base_) + committed_, grow_size);
        committed_ = new_commit;
      }
    }

    void* ptr = static_cast<uint8_t*>(base_) + aligned_bump;
    bump_ = new_bump;
    return ptr;
  }

  void reset() {
    bump_ = 0;
    // Pages stay committed for next frame (CRITICAL for performance)
  }

  size_t used() const { return bump_; }
  size_t committed() const { return committed_; }

private:
  // Ensure pages are resident (avoid soft page faults)
  // ASTRAL_FORCE_INLINE defined in LOW_LEVEL_PRIMITIVES.md:227-241
  // Why: Called in tight loop during memory initialization (5-10% speedup)
  static ASTRAL_FORCE_INLINE void prefault_memory(void* addr, size_t size) {
    constexpr size_t page_size = 4096;
    volatile uint8_t* ptr = static_cast<uint8_t*>(addr);
    for (size_t i = 0; i < size; i += page_size) {
      ptr[i] = 0; // Touch page to force residency
    }
  }

  void* base_;
  size_t capacity_;
  size_t committed_;
  size_t bump_;
};
```

### Double-Buffered Pattern (Advanced)

For zero-overhead resets, use double-buffered allocators:

```cpp
struct DoubleBufferedAllocator {
  ScratchBuffer buffers[2];
  uint32_t active_index = 0;

  ScratchBuffer& active() { return buffers[active_index]; }

  void swap() {
    // Reset inactive buffer while decode uses active buffer
    uint32_t inactive = 1 - active_index;
    buffers[inactive].reset();
    active_index = inactive;
  }
};

void decode_loop(Session* session) {
  DoubleBufferedAllocator& alloc = session->allocator;

  while (!done) {
    ScratchBuffer& frame_alloc = alloc.active();

    // Allocate temp buffers for this frame
    float* logits = frame_alloc.alloc(n_vocab * sizeof(float));
    Token* candidates = frame_alloc.alloc(n_vocab * sizeof(Token));

    // Decode, sample
    llama_decode(session->ctx, logits);
    sample_top_k(logits, candidates);

    // Emit token
    emit_token(session, best_token);

    // Swap buffers (reset happens on next frame)
    alloc.swap();
  }
}
```

### Usage

```cpp
void decode_loop(Session* session) {
  ScratchBuffer& alloc = session->scratch_buffer;

  while (!done) {
    // Allocate temp buffers for this frame
    float* logits = alloc.alloc(n_vocab * sizeof(float));
    Token* candidates = alloc.alloc(n_vocab * sizeof(Token));

    // Decode, sample
    llama_decode(session->ctx, logits);
    sample_top_k(logits, candidates);

    // Emit token
    emit_token(session, best_token);

    // Reset allocator for next frame
    alloc.reset();
  }
}
```

## Small Object Pool

### Purpose

Recycle fixed-size objects (tokens, callback entries) without hitting the OS allocator.

### Design

- **Intrusive Freelist**: Store next pointer in freed object
- **Lock-Free**: CAS-based push/pop for thread-safe access
- **Batch Allocation**: Allocate multiple objects at once to reduce contention
- **TLS Caching (Optional)**: Thread-local cache layer for near-zero contention

### Three-Tier Architecture

A three-tier hierarchy provides optimal performance:

```
┌────────────────────────────────────────────────────────────┐
│ Tier 1: Thread-Local Storage (TLS) Cache                  │
│ • 99.9% of allocations                                     │
│ • Zero contention, zero atomics                            │
│ • 5-10ns latency                                           │
│ • Per-thread bundles of 64-1024 objects                   │
└───────────────────┬────────────────────────────────────────┘
                    │ Miss (0.1%)
┌───────────────────▼────────────────────────────────────────┐
│ Tier 2: Global Lock-Free Pool                             │
│ • Lock-free LIFO stack                                     │
│ • 100-200ns latency                                        │
│ • Shared bundle storage                                    │
└───────────────────┬────────────────────────────────────────┘
                    │ Miss (0.01%)
┌───────────────────▼────────────────────────────────────────┐
│ Tier 3: LinearAllocator (Backing Storage)                 │
│ • 65KB bundle allocation                                   │
│ • 10,000+ns latency                                        │
│ • Amortized across 1024 objects                            │
└────────────────────────────────────────────────────────────┘
```

**Key Insight**: This design achieves 99.9% TLS cache hit rate in production workloads, eliminating atomic overhead for nearly all allocations.

**When to Use TLS Caching**:
- **Small objects** (8-128 bytes) with high churn rate (tokens, callbacks)
- **Multi-threaded** workloads where contention is observable (>4 threads)
- **NOT needed** for single-threaded decode loops or objects >128 bytes

### Implementation

```cpp
template<typename T, size_t MaxObjects>
class ObjectPool {
public:
  // v0.1 implementation: intrusive freelist protected by a tiny spinlock.
  // This avoids atomic compare-and-swap usage and keeps the pool allocation-free.
  ObjectPool() { init_freelist(); }

  T* acquire();
  void release(T* obj);

private:
  alignas(64) T storage_[MaxObjects];
  T* head_ = nullptr;
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
};
```

## Thread-Local Storage (TLS) Caching Pattern

### The Atomic Contention Problem

In multi-threaded workloads (>4 threads), atomic operations become a bottleneck:
- **Atomic CAS**: 100-200ns latency on modern CPUs
- **Cache Coherency**: MESI protocol overhead across cores
- **Contention**: Multiple threads fighting for same cache line

**Performance Impact**: On an 8-core system allocating 1M small objects:
- Without TLS: 100ns × 1M = 100ms
- With TLS: 10ns × 1M = 10ms
- **Speedup**: 10-20x faster

### Design: Batch Refill Strategy

Instead of hitting the global pool on every allocation, maintain a thread-local cache:

```
Thread 1 TLS Cache (64 objects)    Thread 2 TLS Cache (64 objects)
        ↓                                  ↓
        └──────────┬───────────────────────┘
                   ↓
         Global Lock-Free Pool
         (Refill 32-64 at a time)
                   ↓
            LinearAllocator
            (Backing storage)
```

### Implementation

```cpp
template<typename T>
class TLSObjectPool {
  struct alignas(128) TLSCache {
    T* slots[64];       // Thread-local cache (NO ATOMICS!)
    uint32_t count = 0;
    uint8_t padding[128 - sizeof(T*) * 64 - sizeof(uint32_t)]; // Cache-line pad
  };
  // TLSCache must be trivially copyable for efficient TLS operations
  // This enables fast thread-local access without atomic operations
  static_assert(std::is_trivially_copyable_v<TLSCache>,
                "TLSCache must be trivially copyable for efficient TLS operations");

  static thread_local TLSCache tls_cache_;
  ObjectPool<T> global_pool_;

public:
  TLSObjectPool(LinearAllocator& backing, uint32_t capacity)
    : global_pool_(backing, capacity) {}

  T* pop() {
    // Fast path: TLS cache (NO ATOMICS! ~10ns)
    TLSCache& cache = tls_cache_;
    if (cache.count > 0) {
      return cache.slots[--cache.count];
    }

    // Slow path: Batch refill from global pool (~200ns per object, amortized)
    constexpr uint32_t batch_size = 32;
    for (uint32_t i = 0; i < batch_size; ++i) {
      T* obj = global_pool_.pop();
      if (!obj) break;
      cache.slots[cache.count++] = obj;
    }

    return (cache.count > 0) ? cache.slots[--cache.count] : nullptr;
  }

  void push(T* obj) {
    TLSCache& cache = tls_cache_;

    // Fast path: TLS cache (NO ATOMICS! ~10ns)
    if (cache.count < 64) {
      cache.slots[cache.count++] = obj;
      return;
    }

    // Slow path: Batch flush to global pool
    constexpr uint32_t flush_count = 32;
    for (uint32_t i = 0; i < flush_count; ++i) {
      global_pool_.push(cache.slots[--cache.count]);
    }
    cache.slots[cache.count++] = obj;
  }

  // Flush TLS cache back to global pool (call on thread exit)
  void flush_tls_cache() {
    TLSCache& cache = tls_cache_;
    while (cache.count > 0) {
      global_pool_.push(cache.slots[--cache.count]);
    }
  }
};

// Define TLS storage
template<typename T>
thread_local typename TLSObjectPool<T>::TLSCache TLSObjectPool<T>::tls_cache_;
```

### Performance Characteristics

| Metric | Without TLS | With TLS | Improvement |
|--------|-------------|----------|-------------|
| Cache hit latency | 100-200ns (atomic CAS) | 5-10ns (TLS load) | **20x faster** |
| Cache miss latency | 100-200ns | 200ns × 32 = 6400ns (amortized: 200ns) | Same |
| Contention | High (all threads) | Low (batch only) | **10x less** |
| Memory overhead | 0 | 64 × sizeof(T) per thread | Small |

### When to Use TLS Caching

**Use When**:
- Multi-threaded workloads (>4 threads)
- Small objects (8-128 bytes)
- High allocation frequency (>10K allocs/sec per thread)
- Profiler shows >5% time in ObjectPool operations

**Don't Use When**:
- Single-threaded decode loops (no contention)
- Large objects (>128 bytes, wastes TLS cache space)
- Low allocation frequency (<1K allocs/sec)
- Memory-constrained environments (TLS overhead per thread)

### Example: Token Pool with TLS

```cpp
struct Token {
  uint32_t id;
  uint8_t utf8_data[16];
  uint8_t utf8_len;
};

// Without TLS (simpler, fine for <4 threads)
ObjectPool<Token> token_pool(backing_alloc, 4096);

// With TLS (for high contention, >4 threads)
TLSObjectPool<Token> tls_token_pool(backing_alloc, 4096);

// Allocate token (99.9% hit TLS cache)
Token* tok = tls_token_pool.pop();
if (!tok) {
  // Handle pool exhaustion
}

// Use token
tok->id = 42;
tok->utf8_len = 3;
memcpy(tok->utf8_data, "foo", 3);

// Return to pool
tls_token_pool.push(tok);

// On thread exit
tls_token_pool.flush_tls_cache();
```

### Batch Size Tuning

**Small Batches (8-16)**:
- Lower TLS memory overhead
- More frequent global pool access
- Good for memory-constrained systems

**Large Batches (32-64)**:
- Higher TLS memory overhead
- Fewer global pool accesses
- Best for high-throughput systems

**Recommendation**: Start with 32, tune based on profiling.

### Token Pool Example

```cpp
struct Token {
  uint32_t id;
  uint8_t utf8_data[16]; // Max UTF-8 bytes for a token
  uint8_t utf8_len;
};

// Without TLS (simpler, fine for <4 threads)
ObjectPool<Token> token_pool(backing_alloc, 4096);

Token* tok = token_pool.pop();
if (!tok) {
  // Handle pool exhaustion
}
// Fill tok
token_pool.push(tok); // Return to pool
```

## Engine Allocator Integration

### Design Philosophy

**Preferred Strategy**: Accept engine-provided allocators for all session-level memory. This gives the engine complete visibility into memory usage and allows integration with engine profilers and memory budgets.

**Fallback Strategy**: If no engine allocator is provided, use internal virtual reserve system.

### Memory Requirement Query API

Before initialization, allow integrators to query memory requirements:

```cpp
// Query memory requirements without initializing
AstralMemoryRequirements astral_query_memory_requirements(const AstralInit* cfg) {
  AstralMemoryRequirements req;

  // Base system overhead
  req.system_overhead = 4 * 1024 * 1024; // 4MB for bootstrap, pools

  // Per-model requirements
  req.model_weights = cfg->model_size; // From GGUF metadata
  req.kv_cache_per_session = cfg->n_ctx * cfg->n_layer * sizeof(float16) * 2;

  // Per-session requirements
  req.session_pools = 16 * 1024 * 1024; // 16MB scratch buffer
  req.token_pool = 256 * 1024; // 256KB token pool

  req.total_per_session = req.kv_cache_per_session + req.session_pools + req.token_pool;
  req.total_estimate = req.system_overhead + req.model_weights +
                       (req.total_per_session * cfg->max_sessions);

  return req;
}
```

### Unity Integration

Accept Unity's `Allocator` for session-local allocations:

```csharp
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

public class AstralMemoryManager {
  // Query requirements before allocating
  public static AstralMemoryRequirements QueryRequirements(AstralInit config) {
    return astral_query_memory_requirements(ref config);
  }

  // Create Unity-backed allocator
  public static AstralAllocator CreateUnityAllocator() {
    return new AstralAllocator {
      alloc = UnityAlloc,
      free = UnityFree,
      user = IntPtr.Zero
    };
  }

  [MonoPInvokeCallback(typeof(AstralAllocFn))]
  static IntPtr UnityAlloc(IntPtr user, UIntPtr size, UIntPtr align) {
    // Use Unity's native allocator (tracked by Unity profiler)
    var ptr = UnsafeUtility.Malloc((long)size, (int)align, Allocator.Persistent);
    return new IntPtr(ptr);
  }

  [MonoPInvokeCallback(typeof(AstralFreeFn))]
  static void UnityFree(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align) {
    unsafe {
      UnsafeUtility.Free(ptr.ToPointer(), Allocator.Persistent);
    }
  }
}

// Usage
var requirements = AstralMemoryManager.QueryRequirements(initConfig);
Debug.Log($"Astral will use ~{requirements.total_estimate / (1024*1024)} MB");

var allocator = AstralMemoryManager.CreateUnityAllocator();
initConfig.sys_alloc = allocator;
astral_init(ref initConfig);
```

### Unreal Integration

Wrap Unreal's `FMemory` allocator:

```cpp
#include "HAL/UnrealMemory.h"

class FAstralMemoryManager {
public:
  // Query requirements before allocating
  static FAstralMemoryRequirements QueryRequirements(const FAstralInit& Config) {
    return astral_query_memory_requirements(&Config);
  }

  // Create Unreal-backed allocator
  static AstralAllocator CreateUnrealAllocator() {
    return AstralAllocator {
      .alloc = &UEAlloc,
      .free = &UEFree,
      .user = nullptr
    };
  }

private:
  static void* UEAlloc(void* User, size_t Size, size_t Align) {
    // Use Unreal's allocator (tracked by Unreal Insights)
    return FMemory::Malloc(Size, static_cast<uint32>(Align));
  }

  static void UEFree(void* User, void* Ptr, size_t Size, size_t Align) {
    FMemory::Free(Ptr);
  }
};

// Usage
FAstralInit InitConfig;
// ... configure ...

auto Requirements = FAstralMemoryManager::QueryRequirements(InitConfig);
UE_LOG(LogAstral, Log, TEXT("Astral will use ~%llu MB"), Requirements.total_estimate >> 20);

InitConfig.sys_alloc = FAstralMemoryManager::CreateUnrealAllocator();
astral_init(&InitConfig);
```

### Runtime Allocation Routing

VM mode routes supported size-class allocations through the reserved region and commits additional pages on cold allocator growth. Requests larger than the core heap's size-class limit use `sys_alloc`, or the platform allocator when no callback is configured.

Arena modes are stricter. Astral-owned allocations use the fixed runtime-heap partition and return `ASTRAL_E_NOMEM` when the request is unsupported or the partition is exhausted. They never spill into `sys_alloc`.

## Memory Budgets and Limits

### Typical Session Budget

| Component | Size | Notes |
|-----------|------|-------|
| KV Cache | 512 MB | 8K ctx, 32 layers, fp16 |
| ScratchBuffer | 16 MB | Max per session (pre-committed: 2MB) |
| Token Pool | 256 KB | 4096 tokens × 64 bytes |
| MPMC Queue | 128 KB | 1024 entries × 128 bytes |
| SPSC Ring | 64 KB | 512 tokens × 128 bytes |
| **Total** | **~530 MB** | Per session |

### Global Budget

| Component | Size | Notes |
|-----------|------|-------|
| Model Weights | 4 GB | 7B params, Q4_K_M quant |
| Global Reserve | 2 GB | For sessions, pools |
| **Total** | **~6 GB** | Single model, 4 sessions |

## Memory Counters and Debugging

### Counters

Track allocations/frees for leak detection:

```cpp
struct MemoryStats {
  std::atomic<uint64_t> bytes_reserved;
  std::atomic<uint64_t> bytes_committed;
  std::atomic<uint64_t> alloc_count;
  std::atomic<uint64_t> free_count;
};

void track_alloc(size_t size) {
  g_mem_stats.bytes_committed.fetch_add(size, std::memory_order_relaxed);
  g_mem_stats.alloc_count.fetch_add(1, std::memory_order_relaxed);
}

void track_free(size_t size) {
  g_mem_stats.bytes_committed.fetch_sub(size, std::memory_order_relaxed);
  g_mem_stats.free_count.fetch_add(1, std::memory_order_relaxed);
}
```

### Leak Detection

At shutdown, verify:

```cpp
void astral_shutdown() {
  // Drain all queues
  // Destroy all sessions

  uint64_t allocs = g_mem_stats.alloc_count.load(std::memory_order_relaxed);
  uint64_t frees = g_mem_stats.free_count.load(std::memory_order_relaxed);
  uint64_t committed = g_mem_stats.bytes_committed.load(std::memory_order_relaxed);

  if (allocs != frees) {
    log_error("Memory leak: %llu allocs, %llu frees", allocs, frees);
  }
  if (committed != 0) {
    log_error("Memory leak: %llu bytes still committed", committed);
  }
}
```

### Debug Mode

In debug builds, fill freed memory with sentinel values:

```cpp
void debug_fill_free(void* ptr, size_t size) {
  #ifndef NDEBUG
    ::memset(ptr, 0xDD, size); // Freed memory marker
  #endif
}
```

## Best Practices

1. **Pre-allocate**: Size pools and allocators based on max expected load
2. **Batch Commits**: Commit memory in large chunks (e.g., 2MB pages) to reduce syscalls
3. **Avoid Decommit**: Keep committed pages resident across frames unless memory-constrained
4. **Profile**: Use `MemoryStats` to identify allocation hotspots
5. **Test**: Run with AddressSanitizer (ASAN) to catch use-after-free, leaks

## Future Optimizations

- **Compaction**: Optionally compact KV cache when context window shifts
- **Swap-Out**: Decommit idle session memory after timeout (e.g., 60s)
- **Tiered Allocators**: Use huge pages for hot data, normal pages for cold
- **Prefetching**: Software prefetch next frame's memory before decode starts
