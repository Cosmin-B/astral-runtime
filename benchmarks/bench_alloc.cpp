#include "bench_clock.hpp"
#include "bench_common.hpp"

#include "../include/astral_rt.h"
#include "../src/core/runtime_state.hpp"
#include "../src/memory/frame_allocator.hpp"
#include "../src/memory/object_pool.hpp"
#include "../src/platform/atomics.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace astral::bench {

namespace {

struct ArenaBuffer {
    void* base = nullptr;
    size_t size = 0;

    bool alloc(size_t bytes) {
        free();
        base = std::malloc(bytes);
        if (!base) {
            return false;
        }
        size = bytes;
        std::memset(base, 0, bytes);
        return true;
    }

    void free() {
        if (base) {
            std::free(base);
        }
        base = nullptr;
        size = 0;
    }

    ~ArenaBuffer() { free(); }
};

static AstralErr init_vm_runtime() {
    AstralInit cfg{};
    cfg.reserve_bytes = 128ull * 1024ull * 1024ull;
    cfg.thread_count = 1;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    return astral_init(&cfg);
}

static AstralErr init_arena_runtime(ArenaBuffer& arena) {
    if (!arena.alloc(128ull * 1024ull * 1024ull)) {
        return ASTRAL_E_NOMEM;
    }

    AstralInit2 cfg2{};
    cfg2.base.reserve_bytes = 0;
    cfg2.base.thread_count = 1;
    cfg2.base.numa_node = 0xFFFFFFFFu;
    cfg2.base.enable_hugepages = 0;
    cfg2.memory_mode = ASTRAL_MEMMODE_ARENA_BORROWED;
    cfg2.arena.base = arena.base;
    cfg2.arena.size = static_cast<uint64_t>(arena.size);
    cfg2.arena.session_block_size = 0;
    cfg2.arena.session_block_count = 0;
    return astral_init2(&cfg2);
}

struct PoolNode {
    uint64_t a;
    uint64_t b;
    uint8_t payload[48];
};

inline void do_not_optimize_ptr(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::"r"(p) : "memory");
#else
    (void)p;
#endif
}

inline void do_not_optimize_u64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::"r"(v) : "memory");
#else
    (void)v;
#endif
}

} // namespace

BenchResult bench_frame_allocator(uint64_t iters, uint32_t size) {
    constexpr size_t kArenaSize = 128ull * 1024ull * 1024ull;
    ArenaBuffer arena{};
    BenchResult r{};
    r.name = "frame_allocator alloc/reset";
    r.ops = iters;

    if (!arena.alloc(kArenaSize)) {
        r.ops = 0;
        return r;
    }

    astral::memory::FrameAllocator alloc(arena.base, arena.size);
    volatile void* sink = nullptr;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        void* p = alloc.alloc(static_cast<size_t>(size), 64);
        if (p == nullptr) {
            alloc.reset();
            p = alloc.alloc(static_cast<size_t>(size), 64);
        }
        sink = p;
        do_not_optimize_ptr(p);
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    (void)sink;
    do_not_optimize_u64(static_cast<uint64_t>(alloc.used()));

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

BenchResult bench_object_pool_single(uint64_t iters) {
    astral::memory::ObjectPool<PoolNode, 4096> pool;
    BenchResult r{};
    r.name = "object_pool acquire/release";
    r.ops = iters * 2;

    volatile PoolNode* sink = nullptr;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        PoolNode* node = pool.acquire();
        if (node != nullptr) {
            node->a = i;
            sink = node;
            do_not_optimize_ptr(node);
            pool.release(node);
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    (void)sink;

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

BenchResult bench_object_pool_contended(uint64_t iters, uint32_t threads) {
    constexpr uint32_t kMaxThreads = 16;
    astral::memory::ObjectPool<PoolNode, 4096> pool;
    BenchResult r{};
    r.name = "object_pool contended";

    if (threads == 0) {
        threads = 1;
    }
    if (threads > kMaxThreads) {
        threads = kMaxThreads;
    }

    const uint64_t per_thread = iters / threads;
    r.ops = per_thread * threads * 2;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (uint32_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            for (uint64_t i = 0; i < per_thread; ++i) {
                PoolNode* node = pool.acquire();
                if (node != nullptr) {
                    node->a = (static_cast<uint64_t>(t) << 32) | i;
                    do_not_optimize_ptr(node);
                    pool.release(node);
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != threads) {
        astral::platform::cpu_pause();
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    start.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

BenchResult bench_runtime_alloc_free(uint64_t iters, uint32_t size, bool arena_mode) {
    BenchResult r{};
    r.name = arena_mode ? "runtime_alloc/free (arena)" : "runtime_alloc/free (vm)";
    r.ops = iters;

    ArenaBuffer arena{};
    const AstralErr init_err = arena_mode ? init_arena_runtime(arena) : init_vm_runtime();
    if (init_err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    volatile void* sink = nullptr;
    for (uint64_t i = 0; i < iters; ++i) {
        void* p = astral::core::runtime_alloc(static_cast<size_t>(size), 16);
        sink = p;
        astral::core::runtime_free(p, static_cast<size_t>(size), 16);
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    (void)sink;

    r.ticks = t1 - t0;
    r.ns = n1 - n0;

    astral_shutdown();
    return r;
}

} // namespace astral::bench
