#include "bench_clock.hpp"
#include "bench_common.hpp"

#include "../include/astral_rt.h"
#include "../src/core/runtime_state.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>

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

} // namespace

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

