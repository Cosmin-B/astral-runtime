/**
 * init.cpp - Runtime initialization and shutdown
 *
 * Global runtime state management with virtual memory allocation,
 * thread pool initialization, and logging setup.
 *
 * Design:
 * - Singleton global runtime state
 * - Virtual memory reserve/commit strategy for deterministic allocation
 * - Engine allocator preferred, internal fallback
 * - Worker pool + internal work queue
 *
 * Thread Safety: Init/shutdown are NOT thread-safe; call from main thread only.
 */

#include "../../include/astral_rt.h"
#include "../platform/vm.h"
#include "../concurrency/mpmc_queue.hpp"
#include "../memory/frame_allocator.hpp"
#include "../utils/logging.hpp"
#include "../utils/trace.hpp"
#include "../platform/time.h"
#include "../platform/thread.h"
#include "../platform/atomics.h"
#include "abi_guard.hpp"
#include "alloc_utils.hpp"
#include "handles.hpp"
#include "work_queue.hpp"
#include "runtime_state.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#ifndef ASTRAL_ENABLE_VIRTUAL_MEMORY
  #define ASTRAL_ENABLE_VIRTUAL_MEMORY 1
#endif

namespace astral::core {

// ============================================================================
// Global Runtime State
// ============================================================================

struct WorkItem {
    WorkFn fn;
    void* user;
};

constexpr size_t kWorkQueueCapacity = 1024;
using WorkQueue = concurrency::MpmcQueue<WorkItem, kWorkQueueCapacity>;

namespace {

// Simple arena-backed size-class allocator for non-hot allocations.
//
// Properties:
// - Deterministic when backed by a fixed arena (no syscalls).
// - No per-allocation headers (free requires size).
// - Thread-safe via per-bucket spinlocks (rarely contended for typical workloads).
// - Not intended for decode hot paths.
class ArenaHeap {
public:
    ArenaHeap() = default;

    void reset() noexcept {
        epoch_.fetch_add(1, std::memory_order_relaxed);
        base_ = nullptr;
        size_ = 0;
        used_ = 0;
        arena_lock_.clear(std::memory_order_relaxed);
        for (auto& b : buckets_) {
            b.free_list = nullptr;
            b.grow = 0;
            b.lock.clear(std::memory_order_relaxed);
        }
    }

    void init(void* base, size_t size) noexcept {
        reset();
        base_ = static_cast<uint8_t*>(base);
        size_ = size;
        used_ = 0;
        for (uint32_t i = 0; i < kBucketCount; ++i) {
            buckets_[i].free_list = nullptr;
            buckets_[i].grow = 0;
        }
    }

    void* allocate(size_t size, size_t align) noexcept {
        if (base_ == nullptr || size_ == 0) {
            return nullptr;
        }
        if (size == 0) {
            return nullptr;
        }
        if (align > kMaxAlign) {
            return nullptr;
        }

        const uint32_t bucket = bucket_index_for(size);
        if (bucket == kInvalidBucket) {
            return nullptr;
        }

        ThreadCache& tc = tls_cache(epoch_.load(std::memory_order_relaxed));
        void*& local_head = tc.local_lists[bucket];
        uint16_t& local_count = tc.local_counts[bucket];
        if (local_head != nullptr) {
            void* p = local_head;
            local_head = *reinterpret_cast<void**>(p);
            if (local_count > 0) {
                --local_count;
            }
            return p;
        }

        Bucket& b = buckets_[bucket];
        // Refill local cache from the global freelist (fast path is lock-free).
        {
            lock_bucket(b);
            void* node = b.free_list;
            uint32_t moved = 0;
            while (node != nullptr && moved < kLocalRefill) {
                void* next = *reinterpret_cast<void**>(node);
                b.free_list = next;
                *reinterpret_cast<void**>(node) = local_head;
                local_head = node;
                node = next;
                ++moved;
            }
            unlock_bucket(b);
            if (moved > 0) {
                local_count = static_cast<uint16_t>(moved - 1u);
                void* p = local_head;
                local_head = *reinterpret_cast<void**>(p);
                return p;
            }
        }

        // Grow on demand from the backing arena.
        if (!grow_bucket(bucket)) {
            return nullptr;
        }

        // Post-grow: refill again (should succeed unless arena exhausted between).
        lock_bucket(b);
        void* node = b.free_list;
        uint32_t moved = 0;
        while (node != nullptr && moved < kLocalRefill) {
            void* next = *reinterpret_cast<void**>(node);
            b.free_list = next;
            *reinterpret_cast<void**>(node) = local_head;
            local_head = node;
            node = next;
            ++moved;
        }
        unlock_bucket(b);
        if (moved == 0) {
            return nullptr;
        }

        local_count = static_cast<uint16_t>(moved - 1u);
        void* p = local_head;
        local_head = *reinterpret_cast<void**>(p);
        return p;
    }

    void deallocate(void* ptr, size_t size, size_t align) noexcept {
        if (ptr == nullptr || base_ == nullptr) {
            return;
        }
        if (align > kMaxAlign) {
            return;
        }

        const uint32_t bucket = bucket_index_for(size);
        if (bucket == kInvalidBucket) {
            return;
        }

        ThreadCache& tc = tls_cache(epoch_.load(std::memory_order_relaxed));
        void*& local_head = tc.local_lists[bucket];
        uint16_t& local_count = tc.local_counts[bucket];

        *reinterpret_cast<void**>(ptr) = local_head;
        local_head = ptr;
        if (local_count < 0xFFFFu) {
            ++local_count;
        }

        if (local_count < kLocalMax) {
            return;
        }

        Bucket& b = buckets_[bucket];
        lock_bucket(b);
        uint32_t moved = 0;
        while (local_head != nullptr && moved < kLocalFlush) {
            void* node = local_head;
            local_head = *reinterpret_cast<void**>(node);
            if (local_count > 0) {
                --local_count;
            }
            *reinterpret_cast<void**>(node) = b.free_list;
            b.free_list = node;
            ++moved;
        }
        unlock_bucket(b);
    }

    bool enabled() const noexcept { return base_ != nullptr && size_ != 0; }
    size_t capacity_bytes() const noexcept { return size_; }
    bool owns(void* ptr) const noexcept {
        if (ptr == nullptr || base_ == nullptr) {
            return false;
        }
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t b = reinterpret_cast<uintptr_t>(base_);
        return p >= b && p < (b + size_);
    }

private:
    static constexpr uint32_t kInvalidBucket = 0xFFFFFFFFu;
    static constexpr size_t kMaxAlign = 16;
    static constexpr uint32_t kMinPow2 = 32;
    static constexpr uint32_t kMaxPow2 = 1u << 20; // 1 MiB max block size
    static constexpr uint32_t kSubdiv = 8;
    static constexpr uint32_t kLevels = 16; // 32..1 MiB
    static constexpr uint32_t kBucketCount = kLevels * kSubdiv;
    static constexpr uint32_t kLocalRefill = 32;
    static constexpr uint16_t kLocalMax = 64;
    static constexpr uint32_t kLocalFlush = 32;

    struct Bucket {
        std::atomic_flag lock = ATOMIC_FLAG_INIT;
        void* free_list = nullptr;
        uint32_t grow = 0;
    };

    struct ThreadCache {
        uint64_t epoch = 0;
        void* local_lists[kBucketCount]{};
        uint16_t local_counts[kBucketCount]{};
    };

    static ThreadCache& tls_cache(uint64_t epoch) noexcept {
        thread_local ThreadCache c{};
        if (c.epoch != epoch) {
            c.epoch = epoch;
            std::memset(c.local_lists, 0, sizeof(c.local_lists));
            std::memset(c.local_counts, 0, sizeof(c.local_counts));
        }
        return c;
    }

    static inline void* align_up_ptr_local(void* p, size_t align) noexcept {
        const uintptr_t v = reinterpret_cast<uintptr_t>(p);
        const uintptr_t a = static_cast<uintptr_t>(align);
        return reinterpret_cast<void*>((v + (a - 1u)) & ~(a - 1u));
    }

    static inline uint32_t highest_bit_nonzero_u32(uint32_t v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
        return 31u - static_cast<uint32_t>(__builtin_clz(v));
#else
        uint32_t r = 0;
        while (v >>= 1u) {
            ++r;
        }
        return r;
#endif
    }

    static inline size_t bucket_size_for(size_t req) noexcept {
        uint32_t n = static_cast<uint32_t>(req);
        if (n < kMinPow2) n = kMinPow2;
        if (n > kMaxPow2) return 0;

        uint32_t hb = highest_bit_nonzero_u32(n);
        uint32_t base = 1u << hb;
        if (base < kMinPow2) {
            base = kMinPow2;
            hb = highest_bit_nonzero_u32(kMinPow2);
        }

        uint32_t step = base / kSubdiv;
        if (step == 0) step = 1;
        uint32_t rounded = (n + (step - 1u)) & ~(step - 1u);

        if (rounded >= base * 2u) {
            base *= 2u;
            if (base > kMaxPow2) return 0;
            step = base / kSubdiv;
            if (step == 0) step = 1;
            rounded = base;
        }

        return static_cast<size_t>(rounded);
    }

    static inline uint32_t bucket_index_for(size_t req) noexcept {
        const size_t bs = bucket_size_for(req);
        if (bs == 0) return kInvalidBucket;

        uint32_t rounded = static_cast<uint32_t>(bs);
        uint32_t hb = highest_bit_nonzero_u32(rounded);
        uint32_t base = 1u << hb;
        if (base > rounded) {
            base >>= 1u;
            --hb;
        }
        if (base < kMinPow2) {
            base = kMinPow2;
            hb = highest_bit_nonzero_u32(kMinPow2);
        }
        if (base > kMaxPow2) return kInvalidBucket;

        const uint32_t level = hb - highest_bit_nonzero_u32(kMinPow2);
        if (level >= kLevels) return kInvalidBucket;

        const uint32_t step = (base / kSubdiv) != 0 ? (base / kSubdiv) : 1u;
        const uint32_t sub = (rounded - base) / step;
        if (sub >= kSubdiv) return kInvalidBucket;

        return level * kSubdiv + sub;
    }

    void lock_bucket(Bucket& b) noexcept {
        uint32_t spins = 0;
        while (b.lock.test_and_set(std::memory_order_acquire)) {
            if (spins < 64) {
                astral::platform::cpu_pause();
            } else {
                astral::platform::cpu_wait_for_event();
            }
            if (spins < 1024) {
                ++spins;
            }
        }
    }

    void unlock_bucket(Bucket& b) noexcept { b.lock.clear(std::memory_order_release); }

    bool grow_bucket(uint32_t bucket) noexcept {
        if (bucket >= kBucketCount) {
            return false;
        }

        const size_t block_size = bucket_size_for(bucket_req_size(bucket));
        if (block_size == 0) {
            return false;
        }

        Bucket& b = buckets_[bucket];
        // Growth policy: start small, then double (bounded).
        uint32_t count = b.grow == 0 ? 32u : (b.grow < 1024u ? b.grow * 2u : 1024u);
        const size_t bytes = static_cast<size_t>(count) * block_size;

        // Allocate from backing arena with coarse lock.
        void* chunk = nullptr;
        {
            uint32_t spins = 0;
            while (arena_lock_.test_and_set(std::memory_order_acquire)) {
                if (spins < 64) {
                    astral::platform::cpu_pause();
                } else {
                    astral::platform::cpu_wait_for_event();
                }
                if (spins < 1024) {
                    ++spins;
                }
            }

            uint8_t* cursor = base_ + used_;
            cursor = static_cast<uint8_t*>(align_up_ptr_local(cursor, kMaxAlign));
            const size_t off = static_cast<size_t>(cursor - base_);
            if (off + bytes <= size_) {
                chunk = cursor;
                used_ = off + bytes;
            }

            arena_lock_.clear(std::memory_order_release);
        }

        if (chunk == nullptr) {
            return false;
        }

        // Push blocks onto the bucket freelist.
        lock_bucket(b);
        uint8_t* p = static_cast<uint8_t*>(chunk);
        for (uint32_t i = 0; i < count; ++i) {
            void* node = p + static_cast<size_t>(i) * block_size;
            *reinterpret_cast<void**>(node) = b.free_list;
            b.free_list = node;
        }
        b.grow = count;
        unlock_bucket(b);
        return true;
    }

    static inline size_t bucket_req_size(uint32_t bucket) noexcept {
        const uint32_t level = bucket / kSubdiv;
        const uint32_t sub = bucket % kSubdiv;
        const uint32_t base = kMinPow2 << level;
        const uint32_t step = (base / kSubdiv) != 0 ? (base / kSubdiv) : 1u;
        return static_cast<size_t>(base + sub * step);
    }

    uint8_t* base_{nullptr};
    size_t size_{0};
    size_t used_{0};
    std::atomic<uint64_t> epoch_{1};
    std::atomic_flag arena_lock_ = ATOMIC_FLAG_INIT;
    Bucket buckets_[kBucketCount]{};
};

} // namespace

/**
 * Global runtime state (singleton).
 *
 * Design:
 * - Initialized once at startup via astral_init()
 * - Released at shutdown via astral_shutdown()
 * - NOT thread-safe for init/shutdown
 * - Individual subsystems may be thread-safe
 *
 * Memory layout:
 * - vm_base: Base address of reserved virtual memory
 * - vm_size: Total size of reserved virtual memory
 * - vm_committed: Current committed size (grows as needed)
 * - sys_alloc: Engine allocator (or null for internal allocator)
 * - log_cb: User logging callback
 * - thread_count: Number of worker threads
 */
struct AstralRuntime {
    // Initialization state
    std::atomic<bool> initialized;

    // Memory management
    void* vm_base;
    size_t vm_size;
    size_t vm_committed;
    AstralAllocator sys_alloc;

    // Memory mode (C ABI value)
    uint32_t memory_mode;

    // Arena modes (borrowed/owned).
    void* arena_base;
    size_t arena_size;
    AstralAllocator arena_alloc; // Used only for OWNED mode; otherwise ignored.
    bool arena_owned;

    // Arena heap for non-hot dynamic allocations (arena modes only).
    ArenaHeap arena_heap;

    // Per-worker scratch region (allocated from arena modes).
    uint8_t* worker_scratch_base;
    uint32_t worker_scratch_bytes;

    // Fixed-size per-session scratch blocks (allocated from arena modes).
    uint32_t session_block_size;
    uint32_t session_block_count;
    uint8_t* session_blocks_base;
    uint32_t* session_block_next; // lives inside arena memory
    uint32_t session_block_free_head;
    std::atomic_flag session_block_lock = ATOMIC_FLAG_INIT;

    // Logging
    AstralLogFn log_cb;
    void* log_user;

    // Threading
    uint32_t thread_count;
    astral::platform::Thread* workers;
    uint32_t worker_alloc_count;

    // Work queues (one per worker, MPMC, bounded, CAS-free hot path).
    WorkQueue* work_queues;
    std::atomic<uint32_t> work_queue_rr;
#if ASTRAL_ENABLE_TRACY
    std::atomic<uint32_t> work_queue_depth;
#endif

    // NUMA configuration
    uint32_t numa_node;

    // Memory policy
    bool enable_hugepages;

    AstralRuntime()
        : initialized(false)
        , vm_base(nullptr)
        , vm_size(0)
        , vm_committed(0)
        , sys_alloc{nullptr, nullptr, nullptr}
        , memory_mode(0)
        , arena_base(nullptr)
        , arena_size(0)
        , arena_alloc{nullptr, nullptr, nullptr}
        , arena_owned(false)
        , arena_heap()
        , worker_scratch_base(nullptr)
        , worker_scratch_bytes(0)
        , session_block_size(0)
        , session_block_count(0)
        , session_blocks_base(nullptr)
        , session_block_next(nullptr)
        , session_block_free_head(0xFFFFFFFFu)
        , log_cb(nullptr)
        , log_user(nullptr)
        , thread_count(0)
        , workers(nullptr)
        , worker_alloc_count(0)
        , work_queues(nullptr)
        , work_queue_rr(0)
#if ASTRAL_ENABLE_TRACY
        , work_queue_depth(0)
#endif
        , numa_node(0xFFFFFFFF)
        , enable_hugepages(false)
    {}
};

// Global runtime singleton
static AstralRuntime g_runtime;

// ============================================================================
// Helper: Allocator Wrapper
// ============================================================================

/**
 * Internal allocator wrapper.
 * Uses engine allocator if provided, otherwise falls back to malloc.
 *
 *  This is NOT used in hot paths.
 * Hot paths use pre-committed linear allocators.
 */
void* internal_alloc(size_t size, size_t align) {
    return astral::core::alloc_raw(g_runtime.sys_alloc, size, align);
}

void internal_free(void* ptr, size_t size, size_t align) {
    astral::core::free_raw(g_runtime.sys_alloc, ptr, size, align);
}

namespace {

constexpr uint32_t kInvalidIndex = 0xFFFFFFFFu;

static inline uintptr_t align_up_uintptr(uintptr_t p, size_t align) {
    const uintptr_t a = static_cast<uintptr_t>(align);
    return (p + (a - 1u)) & ~(a - 1u);
}

static inline void* align_up_ptr(void* p, size_t align) {
    return reinterpret_cast<void*>(align_up_uintptr(reinterpret_cast<uintptr_t>(p), align));
}

static inline size_t align_up_size(size_t n, size_t align) {
    const size_t a = align;
    return (n + (a - 1u)) & ~(a - 1u);
}

static void session_pool_reset() {
    g_runtime.session_block_size = 0;
    g_runtime.session_block_count = 0;
    g_runtime.session_blocks_base = nullptr;
    g_runtime.session_block_next = nullptr;
    g_runtime.session_block_free_head = kInvalidIndex;
}

static bool session_pool_init_from_arena(void* arena_base, size_t arena_size, uint32_t block_size, uint32_t block_count) {
    session_pool_reset();

    if (arena_base == nullptr || arena_size == 0 || block_size == 0) {
        return false;
    }

    // Keep alignment modest; arena may come from static buffers.
    const size_t kMetaAlign = alignof(uint32_t);
    const size_t kBlockAlign = 64;

    uint8_t* cursor = static_cast<uint8_t*>(align_up_ptr(arena_base, kMetaAlign));
    const uint8_t* end = static_cast<const uint8_t*>(arena_base) + arena_size;
    if (cursor >= end) {
        return false;
    }

    // Auto block count if not specified: keep it bounded.
    if (block_count == 0) {
        const size_t avail = static_cast<size_t>(end - cursor);
        const size_t max_blocks_by_size = avail / static_cast<size_t>(block_size);
        const size_t capped = max_blocks_by_size > 16 ? 16 : max_blocks_by_size;
        block_count = static_cast<uint32_t>(capped);
        if (block_count == 0) {
            block_count = 1;
        }
    }

    // Place the freelist metadata inside the arena first.
    const size_t meta_bytes = static_cast<size_t>(block_count) * sizeof(uint32_t);
    if (static_cast<size_t>(end - cursor) < meta_bytes) {
        return false;
    }

    uint32_t* next = reinterpret_cast<uint32_t*>(cursor);
    cursor += meta_bytes;

    cursor = static_cast<uint8_t*>(align_up_ptr(cursor, kBlockAlign));
    if (cursor >= end) {
        return false;
    }

    const size_t avail_for_blocks = static_cast<size_t>(end - cursor);
    const size_t max_blocks = avail_for_blocks / static_cast<size_t>(block_size);
    if (max_blocks == 0) {
        return false;
    }

    if (block_count > static_cast<uint32_t>(max_blocks)) {
        block_count = static_cast<uint32_t>(max_blocks);
    }

    // Initialize the freelist.
    for (uint32_t i = 0; i < block_count; ++i) {
        next[i] = (i + 1u < block_count) ? (i + 1u) : kInvalidIndex;
    }

    g_runtime.session_block_size = block_size;
    g_runtime.session_block_count = block_count;
    g_runtime.session_blocks_base = cursor;
    g_runtime.session_block_next = next;
    g_runtime.session_block_free_head = 0u;
    return true;
}

static inline void session_pool_lock() {
    uint32_t spins = 0;
    while (g_runtime.session_block_lock.test_and_set(std::memory_order_acquire)) {
        if (spins < 64) {
            astral::platform::cpu_pause();
        } else {
            astral::platform::cpu_wait_for_event();
        }
        if (spins < 1024) {
            ++spins;
        }
    }
}

static inline void session_pool_unlock() {
    g_runtime.session_block_lock.clear(std::memory_order_release);
}

static void* session_pool_acquire() {
    session_pool_lock();
    const uint32_t head = g_runtime.session_block_free_head;
    if (head == kInvalidIndex) {
        session_pool_unlock();
        return nullptr;
    }

    g_runtime.session_block_free_head = g_runtime.session_block_next[head];
    session_pool_unlock();
    return g_runtime.session_blocks_base + static_cast<size_t>(head) * static_cast<size_t>(g_runtime.session_block_size);
}

static void session_pool_release(void* ptr) {
    if (ptr == nullptr || g_runtime.session_blocks_base == nullptr || g_runtime.session_block_size == 0) {
        return;
    }

    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t base = reinterpret_cast<uintptr_t>(g_runtime.session_blocks_base);
    const uintptr_t span = static_cast<uintptr_t>(g_runtime.session_block_count) * static_cast<uintptr_t>(g_runtime.session_block_size);
    if (p < base || p >= base + span) {
        return;
    }

    const uintptr_t off = p - base;
    const uint32_t idx = static_cast<uint32_t>(off / static_cast<uintptr_t>(g_runtime.session_block_size));
    if (idx >= g_runtime.session_block_count) {
        return;
    }

    session_pool_lock();
    g_runtime.session_block_next[idx] = g_runtime.session_block_free_head;
    g_runtime.session_block_free_head = idx;
    session_pool_unlock();
}

} // namespace

// ============================================================================
// Helper: Logging Callback Adapter
// ============================================================================

/**
 * Adapter from internal logging system to C ABI callback.
 */
void logging_callback_adapter(void* user, int level, const uint8_t* msg, uint32_t len) {
    // Cast to C ABI types
    AstralLogFn cb = g_runtime.log_cb;
    if (cb) {
        AstralSpanU8 span;
        span.data = msg;
        span.len = len;
#if defined(__cpp_exceptions)
        try {
            cb(user, level, span);
        } catch (...) {
            // Never allow user exceptions to unwind through Astral.
        }
#else
        cb(user, level, span);
#endif
    }
}

namespace {

thread_local uint32_t g_tls_worker_id = 0xFFFFFFFFu;
thread_local uint8_t* g_tls_worker_scratch_base = nullptr;
thread_local size_t g_tls_worker_scratch_size = 0;
thread_local size_t g_tls_worker_scratch_used = 0;

static inline void bind_worker_tls(uint32_t idx) {
    g_tls_worker_id = idx;
    g_tls_worker_scratch_base = nullptr;
    g_tls_worker_scratch_size = 0;
    g_tls_worker_scratch_used = 0;

    if (g_runtime.worker_scratch_base != nullptr && g_runtime.worker_scratch_bytes != 0) {
        const size_t stride = static_cast<size_t>(g_runtime.worker_scratch_bytes);
        g_tls_worker_scratch_base = g_runtime.worker_scratch_base + static_cast<size_t>(idx) * stride;
        g_tls_worker_scratch_size = stride;
    }
}

#if ASTRAL_ENABLE_THREADS
	void worker_loop(uint32_t worker_idx) {
	    ASTRAL_ZONE_N("astral.worker_loop");
	    for (;;) {
	        ASTRAL_ZONE_MICRO_N("astral.worker.dequeue_wait");
	        WorkItem item{};
	        g_runtime.work_queues[worker_idx].dequeue_wait(&item);
	        if (item.fn == nullptr) {
	            return;
	        }
#if ASTRAL_ENABLE_TRACY
        // Best-effort queue depth tracking for profiling/visualization.
        const uint32_t depth = g_runtime.work_queue_depth.fetch_sub(1, std::memory_order_relaxed) - 1;
        ASTRAL_PLOT("astral.work_queue_depth", static_cast<double>(depth));
#endif
	        ASTRAL_ZONE_MICRO_N("astral.worker.work_item");
	        item.fn(item.user);
	    }
	}

void worker_loop_thunk(void* user) {
    const uint32_t idx = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(user));
#if ASTRAL_ENABLE_TRACY
    char name[32];
    std::snprintf(name, sizeof(name), "astral_worker_%u", idx);
    ASTRAL_THREAD_NAME(name);
#endif
    bind_worker_tls(idx);
    worker_loop(idx);
}
#endif

} // namespace

AstralErr submit_work(WorkFn fn, void* user) {
    ASTRAL_ZONE_N("astral.submit_work");
    if (fn == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (!g_runtime.initialized.load(std::memory_order_acquire)) {
        return ASTRAL_E_STATE;
    }

#if ASTRAL_ENABLE_THREADS
#if ASTRAL_ENABLE_TRACY
    {
        const uint32_t depth = g_runtime.work_queue_depth.fetch_add(1, std::memory_order_relaxed) + 1;
        ASTRAL_PLOT("astral.work_queue_depth", static_cast<double>(depth));
    }
#endif
    const uint32_t n = g_runtime.thread_count;
    const uint32_t idx = n > 0 ? (g_runtime.work_queue_rr.fetch_add(1, std::memory_order_relaxed) % n) : 0;
    g_runtime.work_queues[idx].enqueue_wait(WorkItem{fn, user});
#else
    // Embedded/minimal footprint mode: execute synchronously on the caller thread.
    fn(user);
#endif
    return ASTRAL_OK;
}

AstralErr submit_work_affine(uint32_t worker_id, WorkFn fn, void* user) {
    ASTRAL_ZONE_N("astral.submit_work_affine");
    if (fn == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (!g_runtime.initialized.load(std::memory_order_acquire)) {
        return ASTRAL_E_STATE;
    }

#if ASTRAL_ENABLE_THREADS
#if ASTRAL_ENABLE_TRACY
    {
        const uint32_t depth = g_runtime.work_queue_depth.fetch_add(1, std::memory_order_relaxed) + 1;
        ASTRAL_PLOT("astral.work_queue_depth", static_cast<double>(depth));
    }
#endif
    const uint32_t n = g_runtime.thread_count;
    const uint32_t idx = n > 0 ? (worker_id % n) : 0;
    g_runtime.work_queues[idx].enqueue_wait(WorkItem{fn, user});
#else
    (void)worker_id;
    fn(user);
#endif
    return ASTRAL_OK;
}

bool runtime_initialized() {
    return g_runtime.initialized.load(std::memory_order_acquire);
}

uint32_t runtime_thread_count() {
    return g_runtime.thread_count;
}

uint32_t runtime_assign_worker_id() {
    if (!g_runtime.initialized.load(std::memory_order_acquire)) {
        return 0;
    }

#if ASTRAL_ENABLE_THREADS
    const uint32_t n = g_runtime.thread_count;
    return n > 0 ? (g_runtime.work_queue_rr.fetch_add(1, std::memory_order_relaxed) % n) : 0;
#else
    return 0;
#endif
}

void* runtime_alloc(size_t size, size_t align) {
    if (size == 0) {
        return nullptr;
    }

    if (!g_runtime.initialized.load(std::memory_order_acquire)) {
        return internal_alloc(size, align);
    }

    if (runtime_uses_arena() && g_runtime.arena_heap.enabled()) {
        void* p = g_runtime.arena_heap.allocate(size, align);
        if (p != nullptr) {
            return p;
        }
    }

    return internal_alloc(size, align);
}

void runtime_free(void* ptr, size_t size, size_t align) {
    if (ptr == nullptr) {
        return;
    }

    if (runtime_uses_arena() && g_runtime.arena_heap.enabled() && g_runtime.arena_heap.owns(ptr)) {
        g_runtime.arena_heap.deallocate(ptr, size, align);
        return;
    }

    internal_free(ptr, size, align);
}

bool runtime_on_worker_thread() {
    return g_tls_worker_id != 0xFFFFFFFFu;
}

uint32_t runtime_worker_id() {
    return runtime_on_worker_thread() ? g_tls_worker_id : 0u;
}

void* runtime_worker_scratch_alloc(size_t size, size_t align) {
    if (!runtime_on_worker_thread()) {
        return nullptr;
    }
    if (g_tls_worker_scratch_base == nullptr || g_tls_worker_scratch_size == 0) {
        return nullptr;
    }
    if (size == 0) {
        return nullptr;
    }

    const size_t a = align == 0 ? 1 : align;
    const size_t mask = a - 1u;
    const size_t off = (g_tls_worker_scratch_used + mask) & ~mask;
    if (off + size > g_tls_worker_scratch_size) {
        return nullptr;
    }

    void* p = g_tls_worker_scratch_base + off;
    g_tls_worker_scratch_used = off + size;
    return p;
}

void runtime_worker_scratch_reset() {
    if (!runtime_on_worker_thread()) {
        return;
    }
    g_tls_worker_scratch_used = 0;
}

void runtime_memory_stats(uint64_t* out_committed, uint64_t* out_reserved) {
    if (out_committed) {
        if (g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_BORROWED || g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_OWNED) {
            *out_committed = static_cast<uint64_t>(g_runtime.arena_size);
        } else {
            *out_committed = static_cast<uint64_t>(g_runtime.vm_committed);
        }
    }
    if (out_reserved) {
        if (g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_BORROWED || g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_OWNED) {
            *out_reserved = static_cast<uint64_t>(g_runtime.arena_size);
        } else {
            *out_reserved = static_cast<uint64_t>(g_runtime.vm_size);
        }
    }
}

bool runtime_hugepages_enabled() {
    if (g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_BORROWED || g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_OWNED) {
        return false;
    }
    return g_runtime.enable_hugepages;
}

bool runtime_uses_arena() {
    return g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_BORROWED || g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_OWNED;
}

void* runtime_session_scratch_acquire(size_t min_size, size_t alignment, size_t* out_size) {
    if (out_size) {
        *out_size = 0;
    }

    if (!g_runtime.initialized.load(std::memory_order_acquire)) {
        return nullptr;
    }

    if (runtime_uses_arena()) {
        if (g_runtime.session_block_size < min_size) {
            return nullptr;
        }
        void* p = session_pool_acquire();
        if (out_size) {
            *out_size = static_cast<size_t>(g_runtime.session_block_size);
        }
        (void)alignment;
        return p;
    }

#if ASTRAL_ENABLE_VIRTUAL_MEMORY
    using namespace astral::platform;
    void* mem = nullptr;
    bool huge_pages = false;
    size_t alloc_size = min_size;
    if (g_runtime.enable_hugepages) {
        mem = vm_reserve_large(min_size, &alloc_size);
        huge_pages = (mem != nullptr);
    }
    if (mem == nullptr) {
        alloc_size = min_size;
        mem = vm_reserve(min_size);
    }
    if (mem == nullptr) {
        return nullptr;
    }

    if (!huge_pages) {
        vm_commit(mem, min_size);
    }

    if (out_size) {
        *out_size = alloc_size;
    }
    (void)alignment;
    return mem;
#else
    (void)min_size;
    (void)alignment;
    return nullptr;
#endif
}

void runtime_session_scratch_release(void* ptr, size_t size) {
    if (ptr == nullptr) {
        return;
    }

    if (runtime_uses_arena()) {
        (void)size;
        session_pool_release(ptr);
        return;
    }

#if ASTRAL_ENABLE_VIRTUAL_MEMORY
    astral::platform::vm_release(ptr, size);
#else
    (void)size;
#endif
}

} // namespace astral::core

// ============================================================================
// C ABI Implementation
// ============================================================================

extern "C" {

ASTRAL_API AstralErr ASTRAL_CALL astral_init(const AstralInit* cfg) {
    ASTRAL_ABI_TRY_BEGIN
    if (cfg == nullptr) {
        return ASTRAL_E_INVALID;
    }

    AstralInit2 cfg2{};
    cfg2.base = *cfg;
    cfg2.memory_mode = ASTRAL_MEMMODE_VM;
    cfg2.flags = 0;
    cfg2.arena = {};

    return astral_init2(&cfg2);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_init2(const AstralInit2* cfg2) {
    ASTRAL_ABI_TRY_BEGIN
    using namespace astral::core;
    using namespace astral::logging;

    if (cfg2 == nullptr) {
        return ASTRAL_E_INVALID;
    }

    const AstralInit* cfg = &cfg2->base;

    // Check if already initialized
    if (g_runtime.initialized.exchange(true, std::memory_order_acq_rel)) {
        return ASTRAL_E_STATE;
    }

    // Initialize global runtime state (common fields)
    g_runtime.sys_alloc = cfg->sys_alloc;
    g_runtime.log_cb = cfg->log_cb;
    g_runtime.log_user = cfg->log_user;
    g_runtime.numa_node = cfg->numa_node;
    g_runtime.memory_mode = cfg2->memory_mode;

    g_runtime.vm_base = nullptr;
    g_runtime.vm_size = 0;
    g_runtime.vm_committed = 0;
    g_runtime.enable_hugepages = false;

    g_runtime.arena_base = nullptr;
    g_runtime.arena_size = 0;
    g_runtime.arena_alloc = {nullptr, nullptr, nullptr};
    g_runtime.arena_owned = false;
    session_pool_reset();

    auto fail_init = [&](AstralErr err) -> AstralErr {
#if ASTRAL_ENABLE_THREADS
        const uint32_t n_alloc = g_runtime.worker_alloc_count;

        if (g_runtime.workers != nullptr) {
            for (uint32_t i = 0; i < g_runtime.thread_count; ++i) {
                if (g_runtime.work_queues) {
                    g_runtime.work_queues[i].enqueue_wait(WorkItem{nullptr, nullptr});
                }
            }
            for (uint32_t i = 0; i < g_runtime.thread_count; ++i) {
                astral::platform::thread_join(&g_runtime.workers[i]);
            }
            astral::core::free_array(g_runtime.sys_alloc, g_runtime.workers, n_alloc);
            g_runtime.workers = nullptr;
        }
        if (g_runtime.work_queues != nullptr) {
            astral::core::free_array(g_runtime.sys_alloc, g_runtime.work_queues, n_alloc);
            g_runtime.work_queues = nullptr;
        }
        g_runtime.worker_alloc_count = 0;
#endif

#if ASTRAL_ENABLE_VIRTUAL_MEMORY
        if (g_runtime.memory_mode == ASTRAL_MEMMODE_VM && g_runtime.vm_base != nullptr) {
            astral::platform::vm_release(g_runtime.vm_base, g_runtime.vm_size);
        }
#endif

        if ((g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_OWNED) && g_runtime.arena_owned && g_runtime.arena_base != nullptr) {
            AstralAllocator a = g_runtime.arena_alloc;
            if (a.free) {
                a.free(a.user, g_runtime.arena_base, g_runtime.arena_size, 16);
            } else {
                ::free(g_runtime.arena_base);
            }
        }

        g_runtime.vm_base = nullptr;
        g_runtime.vm_size = 0;
        g_runtime.vm_committed = 0;
        g_runtime.arena_base = nullptr;
        g_runtime.arena_size = 0;
        g_runtime.arena_owned = false;
        g_runtime.arena_heap.reset();
        g_runtime.worker_scratch_base = nullptr;
        g_runtime.worker_scratch_bytes = 0;
        session_pool_reset();

        g_runtime.initialized.store(false, std::memory_order_release);
        return err;
    };

    // Determine thread count early (used by arena partitioning).
    // In arena modes, default to a single worker for determinism unless the user explicitly requests otherwise.
    g_runtime.thread_count = cfg->thread_count;
    if (g_runtime.thread_count == 0) {
        if (cfg2->memory_mode == ASTRAL_MEMMODE_ARENA_BORROWED || cfg2->memory_mode == ASTRAL_MEMMODE_ARENA_OWNED) {
            g_runtime.thread_count = 1;
        } else {
            const uint32_t hw = astral::platform::hardware_concurrency();
            g_runtime.thread_count = hw > 0 ? hw : 4;
        }
    }

    // Memory mode setup
    if (cfg2->memory_mode == ASTRAL_MEMMODE_VM) {
#if !ASTRAL_ENABLE_VIRTUAL_MEMORY
        return fail_init(ASTRAL_E_UNSUPPORTED);
#else
        using namespace astral::platform;

        // Determine reserve size
        size_t reserve_size = cfg->reserve_bytes;
        if (reserve_size == 0) {
            reserve_size = 2ULL << 30;
        }

        if (cfg->enable_hugepages) {
            constexpr size_t kHugeAlign = 2 * 1024 * 1024;
            reserve_size = align_up_size(reserve_size, kHugeAlign);
        }

        bool huge_pages_enabled = false;
        size_t allocated_size = reserve_size;
        void* vm_base = nullptr;
        if (cfg->enable_hugepages) {
            vm_base = vm_reserve_large(reserve_size, &allocated_size);
            huge_pages_enabled = (vm_base != nullptr);
        }
        if (!vm_base) {
            allocated_size = reserve_size;
            vm_base = cfg->enable_hugepages ? vm_reserve_aligned(reserve_size, 2 * 1024 * 1024) : vm_reserve(reserve_size);
        }
        if (!vm_base) {
            return fail_init(ASTRAL_E_NOMEM);
        }

        size_t initial_commit = 2 * 1024 * 1024;
        if (initial_commit > allocated_size) {
            initial_commit = allocated_size;
        }
        if (huge_pages_enabled) {
            initial_commit = allocated_size;
        } else {
            vm_commit(vm_base, initial_commit);
        }

        if (cfg->enable_hugepages && !huge_pages_enabled) {
            huge_pages_enabled = vm_commit_large(vm_base, initial_commit);
        }

        g_runtime.vm_base = vm_base;
        g_runtime.vm_size = allocated_size;
        g_runtime.vm_committed = initial_commit;
        g_runtime.enable_hugepages = huge_pages_enabled;

#endif
    } else if (cfg2->memory_mode == ASTRAL_MEMMODE_ARENA_BORROWED || cfg2->memory_mode == ASTRAL_MEMMODE_ARENA_OWNED) {
        const uint64_t arena_size_u64 = cfg2->arena.size;
        if (arena_size_u64 == 0) {
            return fail_init(ASTRAL_E_INVALID);
        }

        void* arena_base = cfg2->arena.base;
        AstralAllocator arena_alloc = cfg2->arena.alloc;
        if (arena_alloc.alloc == nullptr) {
            arena_alloc = cfg->sys_alloc;
        }

        if (cfg2->memory_mode == ASTRAL_MEMMODE_ARENA_OWNED) {
            if (arena_base == nullptr) {
                if (arena_alloc.alloc) {
                    arena_base = arena_alloc.alloc(arena_alloc.user, static_cast<size_t>(arena_size_u64), 16);
                } else {
                    arena_base = ::malloc(static_cast<size_t>(arena_size_u64));
                }
                if (arena_base == nullptr) {
                    return fail_init(ASTRAL_E_NOMEM);
                }
            }
            g_runtime.arena_owned = true;
            g_runtime.arena_alloc = arena_alloc;
        } else {
            if (arena_base == nullptr) {
                return fail_init(ASTRAL_E_INVALID);
            }
            g_runtime.arena_owned = false;
            g_runtime.arena_alloc = {nullptr, nullptr, nullptr};
        }

        g_runtime.arena_base = arena_base;
        g_runtime.arena_size = static_cast<size_t>(arena_size_u64);
        g_runtime.enable_hugepages = false;

        // Partition the arena for worker scratch + runtime heap + per-session scratch blocks.
        uint8_t* cursor = static_cast<uint8_t*>(arena_base);
        uint8_t* end = cursor + g_runtime.arena_size;

        // Worker scratch: per worker thread (or 1 in threads-disabled builds).
        const uint32_t scratch_workers =
#if ASTRAL_ENABLE_THREADS
            (g_runtime.thread_count != 0 ? g_runtime.thread_count : 1u);
#else
            1u;
#endif

        const uint32_t scratch_bytes_per_worker =
            cfg2->arena._reserved[0] != 0 ? cfg2->arena._reserved[0] : (256u * 1024u);
        const size_t scratch_total =
            static_cast<size_t>(scratch_bytes_per_worker) * static_cast<size_t>(scratch_workers);

        g_runtime.worker_scratch_base = nullptr;
        g_runtime.worker_scratch_bytes = scratch_bytes_per_worker;

        if (scratch_total != 0) {
            cursor = static_cast<uint8_t*>(align_up_ptr(cursor, 64));
            if (static_cast<size_t>(end - cursor) < scratch_total) {
                return fail_init(ASTRAL_E_NOMEM);
            }
            g_runtime.worker_scratch_base = cursor;
            cursor += scratch_total;
        }

        // Runtime heap (size-class allocator) for deterministic non-hot allocations.
        const uint32_t heap_bytes = cfg2->arena._reserved[1] != 0 ? cfg2->arena._reserved[1] : (2u * 1024u * 1024u);
        g_runtime.arena_heap.reset();
        if (heap_bytes != 0) {
            cursor = static_cast<uint8_t*>(align_up_ptr(cursor, 64));
            if (static_cast<size_t>(end - cursor) < static_cast<size_t>(heap_bytes)) {
                return fail_init(ASTRAL_E_NOMEM);
            }
            g_runtime.arena_heap.init(cursor, static_cast<size_t>(heap_bytes));
            cursor += static_cast<size_t>(heap_bytes);
        }

        // Initialize deterministic per-session scratch blocks inside the remaining arena.
        constexpr uint32_t kDefaultSessionBlockSize = 2u * 1024u * 1024u;
        const uint32_t block_size = cfg2->arena.session_block_size != 0 ? cfg2->arena.session_block_size : kDefaultSessionBlockSize;
        const uint32_t block_count = cfg2->arena.session_block_count;
        if (!session_pool_init_from_arena(cursor, static_cast<size_t>(end - cursor), block_size, block_count)) {
            return fail_init(ASTRAL_E_NOMEM);
        }
    } else {
        return fail_init(ASTRAL_E_INVALID);
    }

    // Reset work queue state (tests may init/shutdown multiple times in one process).
    if (g_runtime.work_queues != nullptr) {
        astral::core::free_array(g_runtime.sys_alloc, g_runtime.work_queues, g_runtime.worker_alloc_count);
        g_runtime.work_queues = nullptr;
    }
    g_runtime.work_queue_rr.store(0, std::memory_order_relaxed);

#if ASTRAL_ENABLE_THREADS
    {
        astral::core::ScopedArray<WorkQueue> queues(g_runtime.sys_alloc, g_runtime.thread_count);
        if (queues.get() == nullptr) {
            return fail_init(ASTRAL_E_NOMEM);
        }
        g_runtime.work_queues = queues.release();
    }

    {
        astral::core::ScopedArray<astral::platform::Thread> workers(g_runtime.sys_alloc, g_runtime.thread_count);
        if (workers.get() == nullptr) {
            return fail_init(ASTRAL_E_NOMEM);
        }
        g_runtime.workers = workers.release();
    }
    g_runtime.worker_alloc_count = g_runtime.thread_count;

    uint32_t started = 0;
    for (; started < g_runtime.thread_count; ++started) {
        if (!astral::platform::thread_start(
                &g_runtime.workers[started],
                worker_loop_thunk,
                reinterpret_cast<void*>(static_cast<uintptr_t>(started)))) {
            for (uint32_t i = 0; i < started; ++i) {
                astral::platform::thread_join(&g_runtime.workers[i]);
            }
            astral::core::free_array(g_runtime.sys_alloc, g_runtime.workers, g_runtime.worker_alloc_count);
            g_runtime.workers = nullptr;
            g_runtime.worker_alloc_count = 0;
            return fail_init(ASTRAL_E_BACKEND);
        }
    }
#else
    g_runtime.thread_count = 0;
    g_runtime.workers = nullptr;
    g_runtime.work_queues = nullptr;
    g_runtime.worker_alloc_count = 0;
    bind_worker_tls(0);
#endif

    if (cfg->log_cb) {
        set_callback(logging_callback_adapter, cfg->log_user);
    }

    if (g_runtime.memory_mode == ASTRAL_MEMMODE_VM) {
        info("Astral runtime initialized: vm_reserve=%zu MB, vm_commit=%zu MB, threads=%u",
             g_runtime.vm_size / (1024 * 1024),
             g_runtime.vm_committed / (1024 * 1024),
             g_runtime.thread_count);
    } else {
        info("Astral runtime initialized: arena=%zu MB, session_blocks=%u x %u KB, threads=%u",
             g_runtime.arena_size / (1024 * 1024),
             g_runtime.session_block_count,
             g_runtime.session_block_size / 1024u,
             g_runtime.thread_count);
    }

    (void)astral::platform::tick_clock();
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_shutdown(void) {
    ASTRAL_ABI_TRY_BEGIN
    using namespace astral::core;
    using namespace astral::logging;

    // Check if initialized
    if (!g_runtime.initialized.load(std::memory_order_acquire)) {
        return;
    }

    // Log shutdown
    info("Astral runtime shutting down");

    // Stop worker pool.
    if (g_runtime.workers != nullptr) {
        const uint32_t n_alloc = g_runtime.worker_alloc_count;
        for (uint32_t i = 0; i < g_runtime.thread_count; ++i) {
            if (g_runtime.work_queues) {
                g_runtime.work_queues[i].enqueue_wait(WorkItem{nullptr, nullptr});
            }
        }
        for (uint32_t i = 0; i < g_runtime.thread_count; ++i) {
            astral::platform::thread_join(&g_runtime.workers[i]);
        }
        astral::core::free_array(g_runtime.sys_alloc, g_runtime.workers, n_alloc);
        g_runtime.workers = nullptr;
    }
    if (g_runtime.work_queues != nullptr) {
        const uint32_t n_alloc = g_runtime.worker_alloc_count;
        astral::core::free_array(g_runtime.sys_alloc, g_runtime.work_queues, n_alloc);
        g_runtime.work_queues = nullptr;
    }
    g_runtime.worker_alloc_count = 0;

    // Release memory backing
    if (g_runtime.memory_mode == ASTRAL_MEMMODE_VM) {
#if ASTRAL_ENABLE_VIRTUAL_MEMORY
        if (g_runtime.vm_base) {
            astral::platform::vm_release(g_runtime.vm_base, g_runtime.vm_size);
        }
#endif
    } else if (g_runtime.memory_mode == ASTRAL_MEMMODE_ARENA_OWNED) {
        if (g_runtime.arena_owned && g_runtime.arena_base) {
            AstralAllocator a = g_runtime.arena_alloc;
            if (a.free) {
                a.free(a.user, g_runtime.arena_base, g_runtime.arena_size, 16);
            } else {
                ::free(g_runtime.arena_base);
            }
        }
    }

    // Clear global state
    g_runtime.vm_base = nullptr;
    g_runtime.vm_size = 0;
    g_runtime.vm_committed = 0;
    g_runtime.sys_alloc = {nullptr, nullptr, nullptr};
    g_runtime.memory_mode = ASTRAL_MEMMODE_VM;
    g_runtime.arena_base = nullptr;
    g_runtime.arena_size = 0;
    g_runtime.arena_alloc = {nullptr, nullptr, nullptr};
    g_runtime.arena_owned = false;
    g_runtime.arena_heap.reset();
    g_runtime.worker_scratch_base = nullptr;
    g_runtime.worker_scratch_bytes = 0;
    session_pool_reset();
    g_runtime.log_cb = nullptr;
    g_runtime.log_user = nullptr;
    g_runtime.thread_count = 0;
    g_runtime.worker_alloc_count = 0;
    g_runtime.work_queue_rr.store(0, std::memory_order_relaxed);
    g_runtime.numa_node = 0xFFFFFFFF;

    // Clear logging callback
    set_callback(nullptr, nullptr);

    // Mark as uninitialized
    g_runtime.initialized.store(false, std::memory_order_release);

    // Clear TLS view for the calling thread (worker threads are already joined).
    g_tls_worker_id = 0xFFFFFFFFu;
    g_tls_worker_scratch_base = nullptr;
    g_tls_worker_scratch_size = 0;
    g_tls_worker_scratch_used = 0;
    ASTRAL_ABI_CATCH_END_VOID()
}

} // extern "C"
