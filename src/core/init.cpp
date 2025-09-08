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
#include "../platform/time.h"
#include "handles.hpp"
#include "work_queue.hpp"
#include "runtime_state.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

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

    // Logging
    AstralLogFn log_cb;
    void* log_user;

    // Threading
    uint32_t thread_count;
    std::thread* workers;

    // Work queue (MPMC, bounded, CAS-free hot path)
    WorkQueue work_queue;

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
        , log_cb(nullptr)
        , log_user(nullptr)
        , thread_count(0)
        , workers(nullptr)
        , numa_node(0xFFFFFFFF)
        , enable_hugepages(false) {}
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
    if (g_runtime.sys_alloc.alloc) {
        return g_runtime.sys_alloc.alloc(g_runtime.sys_alloc.user, size, align);
    } else {
        // Fallback: Use aligned_alloc if available
        // For simplicity, use malloc (alignment not guaranteed)
        // Production code should use platform-specific aligned malloc
        void* ptr = ::malloc(size);
        return ptr;
    }
}

void internal_free(void* ptr, size_t size, size_t align) {
    if (g_runtime.sys_alloc.free) {
        g_runtime.sys_alloc.free(g_runtime.sys_alloc.user, ptr, size, align);
    } else {
        ::free(ptr);
    }
}

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
        cb(user, level, span);
    }
}

namespace {

void worker_loop() {
    for (;;) {
        WorkItem item{};
        g_runtime.work_queue.dequeue_wait(&item);
        if (item.fn == nullptr) {
            return;
        }
        item.fn(item.user);
    }
}

} // namespace

AstralErr submit_work(WorkFn fn, void* user) {
    if (fn == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (!g_runtime.initialized.load(std::memory_order_acquire)) {
        return ASTRAL_E_STATE;
    }

    g_runtime.work_queue.enqueue_wait(WorkItem{fn, user});
    return ASTRAL_OK;
}

bool runtime_initialized() {
    return g_runtime.initialized.load(std::memory_order_acquire);
}

uint32_t runtime_thread_count() {
    return g_runtime.thread_count;
}

void runtime_memory_stats(uint64_t* out_committed, uint64_t* out_reserved) {
    if (out_committed) {
        *out_committed = static_cast<uint64_t>(g_runtime.vm_committed);
    }
    if (out_reserved) {
        *out_reserved = static_cast<uint64_t>(g_runtime.vm_size);
    }
}

bool runtime_hugepages_enabled() {
    return g_runtime.enable_hugepages;
}

} // namespace astral::core

// ============================================================================
// C ABI Implementation
// ============================================================================

extern "C" {

ASTRAL_API AstralErr ASTRAL_CALL astral_init(const AstralInit* cfg) {
    using namespace astral::core;
    using namespace astral::platform;
    using namespace astral::logging;

    // Validate parameters
    if (!cfg) {
        return ASTRAL_E_INVALID;
    }

    // Check if already initialized
    if (g_runtime.initialized.exchange(true, std::memory_order_acq_rel)) {
        return ASTRAL_E_STATE;
    }

    // Determine reserve size
    size_t reserve_size = cfg->reserve_bytes;
    if (reserve_size == 0) {
        // Default: 2GB
        reserve_size = 2ULL << 30;
    }

    // If hugepages are enabled, prefer a 2MB-aligned reservation and a 2MB-sized region
    // so we can use explicit hugepage mappings later (and maximize THP promotion chances).
    if (cfg->enable_hugepages) {
        constexpr size_t kHugeAlign = 2 * 1024 * 1024;
        reserve_size = (reserve_size + (kHugeAlign - 1)) & ~(kHugeAlign - 1);
    }

    // Reserve virtual memory
    void* vm_base = cfg->enable_hugepages ? vm_reserve_aligned(reserve_size, 2 * 1024 * 1024) : vm_reserve(reserve_size);
    if (!vm_base) {
        g_runtime.initialized.store(false, std::memory_order_release);
        return ASTRAL_E_NOMEM;
    }

    // Commit initial pages (2MB)
    // This avoids page faults during early initialization
    size_t initial_commit = 2 * 1024 * 1024; // 2MB
    if (initial_commit > reserve_size) {
        initial_commit = reserve_size;
    }
    vm_commit(vm_base, initial_commit);

    // Try huge pages if requested
    if (cfg->enable_hugepages) {
        // Best-effort: Ignore failure
        vm_try_hugepages(vm_base, reserve_size);
    }

    // Initialize global runtime state
    g_runtime.vm_base = vm_base;
    g_runtime.vm_size = reserve_size;
    g_runtime.vm_committed = initial_commit;
    g_runtime.sys_alloc = cfg->sys_alloc;
    g_runtime.log_cb = cfg->log_cb;
    g_runtime.log_user = cfg->log_user;
    g_runtime.numa_node = cfg->numa_node;
    g_runtime.enable_hugepages = (cfg->enable_hugepages != 0);

    // Determine thread count
    g_runtime.thread_count = cfg->thread_count;
    if (g_runtime.thread_count == 0) {
        // Auto-detect: Use hardware_concurrency(); fall back to 4.
        const uint32_t hw = std::thread::hardware_concurrency();
        g_runtime.thread_count = hw > 0 ? hw : 4;
    }

    // Reset work queue state (tests may init/shutdown multiple times in one process).
    g_runtime.work_queue.~WorkQueue();
    new (&g_runtime.work_queue) WorkQueue();

    // Start worker pool (best-effort; init is not a hot path).
    g_runtime.workers = new (std::nothrow) std::thread[g_runtime.thread_count];
    if (g_runtime.workers == nullptr) {
        vm_release(vm_base, reserve_size);
        g_runtime.initialized.store(false, std::memory_order_release);
        return ASTRAL_E_NOMEM;
    }

    uint32_t started = 0;
    try {
        for (; started < g_runtime.thread_count; ++started) {
            g_runtime.workers[started] = std::thread(worker_loop);
        }
    } catch (...) {
        // Join threads we managed to start, then fail init.
        for (uint32_t i = 0; i < started; ++i) {
            if (g_runtime.workers[i].joinable()) {
                g_runtime.workers[i].join();
            }
        }
        delete[] g_runtime.workers;
        g_runtime.workers = nullptr;

        vm_release(vm_base, reserve_size);
        g_runtime.initialized.store(false, std::memory_order_release);
        return ASTRAL_E_BACKEND;
    }

    // Set logging callback
    if (cfg->log_cb) {
        set_callback(logging_callback_adapter, cfg->log_user);
    }

    // Log initialization success
    info("Astral runtime initialized: reserve=%zu MB, commit=%zu MB, threads=%u",
         reserve_size / (1024 * 1024),
         initial_commit / (1024 * 1024),
         g_runtime.thread_count);

    // Prime monotonic tick calibration so hot/wait paths don't pay first-use costs.
    (void)astral::platform::tick_clock();

    return ASTRAL_OK;
}

ASTRAL_API void ASTRAL_CALL astral_shutdown(void) {
    using namespace astral::core;
    using namespace astral::platform;
    using namespace astral::logging;

    // Check if initialized
    if (!g_runtime.initialized.load(std::memory_order_acquire)) {
        return;
    }

    // Log shutdown
    info("Astral runtime shutting down");

    // Stop worker pool.
    if (g_runtime.workers != nullptr) {
        for (uint32_t i = 0; i < g_runtime.thread_count; ++i) {
            g_runtime.work_queue.enqueue_wait(WorkItem{nullptr, nullptr});
        }
        for (uint32_t i = 0; i < g_runtime.thread_count; ++i) {
            if (g_runtime.workers[i].joinable()) {
                g_runtime.workers[i].join();
            }
        }
        delete[] g_runtime.workers;
        g_runtime.workers = nullptr;
    }

    // Release virtual memory
    if (g_runtime.vm_base) {
        vm_release(g_runtime.vm_base, g_runtime.vm_size);
    }

    // Clear global state
    g_runtime.vm_base = nullptr;
    g_runtime.vm_size = 0;
    g_runtime.vm_committed = 0;
    g_runtime.sys_alloc = {nullptr, nullptr, nullptr};
    g_runtime.log_cb = nullptr;
    g_runtime.log_user = nullptr;
    g_runtime.thread_count = 0;
    g_runtime.work_queue.~WorkQueue();
    new (&g_runtime.work_queue) WorkQueue();
    g_runtime.numa_node = 0xFFFFFFFF;

    // Clear logging callback
    set_callback(nullptr, nullptr);

    // Mark as uninitialized
    g_runtime.initialized.store(false, std::memory_order_release);
}

} // extern "C"
