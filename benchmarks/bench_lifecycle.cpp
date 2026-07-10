#include "bench_clock.hpp"
#include "bench_common.hpp"

#include "../include/astral_rt.h"

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
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 256ull * 1024ull * 1024ull;
    cfg.thread_count = 1;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    return astral_init(&cfg);
}

static AstralErr init_arena_runtime(ArenaBuffer& arena) {
    if (!arena.alloc(256ull * 1024ull * 1024ull)) {
        return ASTRAL_E_NOMEM;
    }

    AstralInit cfg{};
    cfg.size = sizeof(AstralInit);
    cfg.thread_count = 1;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.memory_mode = ASTRAL_MEMMODE_ARENA_BORROWED;
    cfg.arena.base = arena.base;
    cfg.arena.size = static_cast<uint64_t>(arena.size);
    return astral_init(&cfg);
}

static AstralHandle load_mock_model() {
    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    desc.gpu_layers = 0;
    desc.n_ctx = 128;
    desc.n_batch = 32;
    desc.n_threads = 0;
    desc.embeddings_only = 0;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&desc, &model);
    if (err != ASTRAL_OK) {
        return 0;
    }
    return model;
}

} // namespace

BenchResult bench_model_load_release(uint64_t iters, bool arena_mode) {
    BenchResult r{};
    r.name = arena_mode ? "model_load/release (arena)" : "model_load/release (vm)";
    r.ops = iters;

    ArenaBuffer arena{};
    const AstralErr init_err = arena_mode ? init_arena_runtime(arena) : init_vm_runtime();
    if (init_err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        AstralHandle model = load_mock_model();
        if (model == 0) {
            break;
        }
        astral_model_release(model);
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;

    astral_shutdown();
    return r;
}

BenchResult bench_session_create_destroy(uint64_t iters, bool arena_mode) {
    BenchResult r{};
    r.name = arena_mode ? "session_create/destroy (arena)" : "session_create/destroy (vm)";
    r.ops = iters;

    ArenaBuffer arena{};
    const AstralErr init_err = arena_mode ? init_arena_runtime(arena) : init_vm_runtime();
    if (init_err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    const AstralHandle model = load_mock_model();
    if (model == 0) {
        astral_shutdown();
        r.ops = 0;
        return r;
    }

    AstralSessionDesc sdesc{};
    sdesc.model = model;
    sdesc.max_tokens = 8;
    sdesc.temperature = 0.0f;
    sdesc.top_k = 0;
    sdesc.top_p = 1.0f;
    sdesc.stream_enabled = 0;
    sdesc.seed = 1;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        AstralHandle session = 0;
        const AstralErr err = astral_session_create(&sdesc, &session);
        if (err != ASTRAL_OK || session == 0) {
            break;
        }
        astral_session_destroy(session);
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;

    astral_model_release(model);
    astral_shutdown();
    return r;
}

} // namespace astral::bench
