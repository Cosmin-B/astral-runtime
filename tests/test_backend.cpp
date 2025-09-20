/**
 * test_backend.cpp - Backend tests
 *
 * Tests for backend registration, selection, and CPU backend.
 * Validates: backend registration, selection by gpu_layers, lookup by name.
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <cstring>

namespace {

std::atomic<uint64_t> g_new_calls{0};

static void* alloc_aligned(std::size_t size, std::size_t alignment) noexcept {
    if (size == 0) {
        size = 1;
    }

    if (alignment < alignof(void*)) {
        alignment = alignof(void*);
    }

#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

static void free_aligned(void* ptr) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace

void* operator new(std::size_t size) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void* operator new[](std::size_t size) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void* operator new(std::size_t size, std::align_val_t align) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    void* p = alloc_aligned(size, static_cast<std::size_t>(align));
    if (p) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr, std::align_val_t) noexcept {
    free_aligned(ptr);
}

void* operator new[](std::size_t size, std::align_val_t align) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    void* p = alloc_aligned(size, static_cast<std::size_t>(align));
    if (p) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete[](void* ptr, std::align_val_t) noexcept {
    free_aligned(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
    free_aligned(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
    free_aligned(ptr);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
}

//
// Backend Registration Tests
//

TEST(backend_cpu_auto_registration) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // CPU backend should be auto-registered
    // We can verify this by loading a model with gpu_layers=0

    astral_shutdown();
}

//
// Backend Selection Tests
//

TEST(backend_selection_cpu) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Create model with cpu_layers=0 (CPU only)
    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "models/test.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.gpu_layers = 0; // CPU only
    model_desc.embeddings_only = 0;

    AstralHandle model;
    AstralErr err = astral_model_load(&model_desc, &model);

    // May fail if model file doesn't exist, but should not crash
    if (err == ASTRAL_OK) {
        astral_model_release(model);
    }

    astral_shutdown();
}

TEST(backend_selection_gpu) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Create model with gpu_layers > 0
    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "models/test.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.gpu_layers = 10; // Try GPU
    model_desc.embeddings_only = 0;

    AstralHandle model;
    AstralErr err = astral_model_load(&model_desc, &model);

    // May fail if no GPU backend available, which is expected
    // Just verify it doesn't crash
    if (err == ASTRAL_OK) {
        astral_model_release(model);
    }

    astral_shutdown();
}

TEST(backend_cuda_registration_surface) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Force the backend by name, but keep model_path empty. This distinguishes:
    // - backend present: backend load runs and returns ASTRAL_E_INVALID (missing path)
    // - backend absent: registry lookup fails and Astral returns ASTRAL_E_BACKEND
    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "cuda";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(strlen(backend));
    model_desc.model_path.data = nullptr;
    model_desc.model_path.len = 0;
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.gpu_layers = 10;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&model_desc, &model);

#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
    ASSERT_EQ(err, ASTRAL_E_INVALID);
#else
    ASSERT_EQ(err, ASTRAL_E_BACKEND);
#endif

    if (err == ASTRAL_OK) {
        astral_model_release(model);
    }

    astral_shutdown();
}

//
// Model Load/Release Tests
//

TEST(backend_model_load_invalid_path) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "nonexistent/path/model.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));
    model_desc.n_ctx = 512;

    AstralHandle model;
    AstralErr err = astral_model_load(&model_desc, &model);

    // Should fail (file doesn't exist)
    ASSERT_NE(err, ASTRAL_OK);

    astral_shutdown();
}

TEST(backend_model_load_null_desc) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralHandle model;
    AstralErr err = astral_model_load(nullptr, &model);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(backend_model_load_null_output) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "test.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));

    AstralErr err = astral_model_load(&model_desc, nullptr);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(backend_mock_provider_end_to_end) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(strlen(backend));

    // Mock backend ignores model_path; leave empty.
    model_desc.n_ctx = 128;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    AstralModelInfo info{};
    err = astral_model_info(model, &info);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(info.vocab_size, 0u);

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 32;
    session_desc.temperature = 0.0f; // greedy (deterministic)
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    const char* prompt = "hi";
    AstralSpanU8 chunk = {};
    chunk.data = reinterpret_cast<const uint8_t*>(prompt);
    chunk.len = static_cast<uint32_t>(strlen(prompt));
    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    const uint64_t allocs_before_decode = g_new_calls.load(std::memory_order_relaxed);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    const uint64_t allocs_after_decode = g_new_calls.load(std::memory_order_relaxed);
    ASSERT_EQ(allocs_after_decode, allocs_before_decode);

    // Drain output (mock produces a short fixed message).
    uint8_t buf[128];
    uint32_t total = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        AstralMutSpanU8 out = {};
        out.data = buf + total;
        out.len = sizeof(buf) - total;

        int32_t n = astral_stream_read(session, out, 1000);
        if (n < 0) {
            ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
            continue;
        }
        if (n == 0) {
            break;
        }
        total += static_cast<uint32_t>(n);
        if (total == sizeof(buf)) {
            break;
        }
    }

    AstralStats stats{};
    err = astral_session_stats(session, &stats);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(stats.tok_per_s, 0.0);

    ASSERT_GT(total, 0u);

    // Reset and run a second decode to validate reuse semantics.
    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    const uint64_t allocs_before_decode2 = g_new_calls.load(std::memory_order_relaxed);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    const uint64_t allocs_after_decode2 = g_new_calls.load(std::memory_order_relaxed);
    ASSERT_EQ(allocs_after_decode2, allocs_before_decode2);

    uint8_t buf2[128];
    uint32_t total2 = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        AstralMutSpanU8 out = {};
        out.data = buf2 + total2;
        out.len = sizeof(buf2) - total2;

        int32_t n = astral_stream_read(session, out, 1000);
        if (n < 0) {
            ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
            continue;
        }
        if (n == 0) {
            break;
        }
        total2 += static_cast<uint32_t>(n);
        if (total2 == sizeof(buf2)) {
            break;
        }
    }

    ASSERT_GT(total2, 0u);
    ASSERT_EQ(total2, total);
    ASSERT_EQ(std::memcmp(buf2, buf, total), 0);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}
