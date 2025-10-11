/**
 * gate_allocations.cpp - Hot-path allocation validation gate
 *
 * Goal:
 * - Fail the test suite if steady-state decode/stream performs any heap allocations.
 *
 * Strategy:
 * - Run a warmup decode first (to trigger any one-time lazy init in providers/libs).
 * - Reset the session, then enable allocation tracking and run decode+stream again.
 * - Require zero allocation calls while tracking is enabled.
 *
 * Tracking implementation:
 * - On platforms that support linker symbol wrapping, wrap malloc/calloc/realloc/free
 *   (and aligned_alloc/posix_memalign when available).
 * - On platforms without linker wrapping, track C++ operator new/new[] calls only.
 *
 * Notes:
 * - Provider-agnostic: always runs against the mock backend; runs CPU backend only
 *   when explicitly enabled (env ASTRAL_GATE_CPU_ALLOC=1) and a GGUF model is available.
 * - Embeddings: always gated on mock; CPU embeddings gating is opt-in because provider
 *   embedding paths can allocate internally after warmup.
 */

#include "test_framework.hpp"
#include "astral_rt.h"

#include <atomic>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sys/stat.h>

#if defined(_WIN32)
  #include <malloc.h>
#endif

namespace {

// Allocation counters (count calls, not bytes-only).
std::atomic<uint64_t> g_alloc_calls{0};
std::atomic<uint64_t> g_alloc_bytes{0};

std::atomic<uint64_t> g_new_calls{0};
std::atomic<uint64_t> g_new_bytes{0};

std::atomic<bool> g_tracking_enabled{false};

thread_local bool g_wrap_reentry = false;

static void tracking_reset() {
    g_alloc_calls.store(0, std::memory_order_relaxed);
    g_alloc_bytes.store(0, std::memory_order_relaxed);
    g_new_calls.store(0, std::memory_order_relaxed);
    g_new_bytes.store(0, std::memory_order_relaxed);
}

static void tracking_set_enabled(bool enabled) {
    g_tracking_enabled.store(enabled, std::memory_order_relaxed);
}

static uint64_t tracking_total_calls() {
    const uint64_t a = g_alloc_calls.load(std::memory_order_relaxed);
    const uint64_t n = g_new_calls.load(std::memory_order_relaxed);
    return a + n;
}

static bool file_exists_min_size(const char* path, uint64_t min_bytes) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }

    return static_cast<uint64_t>(st.st_size) >= min_bytes;
}

static uint64_t parse_u64_env(const char* key, uint64_t fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v) {
        return fallback;
    }
    return static_cast<uint64_t>(x);
}

static bool parse_bool_env(const char* key, bool fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }

    if (v[0] == '1' && v[1] == '\0') return true;
    if (v[0] == '0' && v[1] == '\0') return false;

    // Accept common truthy values.
    if (std::strcmp(v, "true") == 0) return true;
    if (std::strcmp(v, "TRUE") == 0) return true;
    if (std::strcmp(v, "yes") == 0) return true;
    if (std::strcmp(v, "YES") == 0) return true;
    if (std::strcmp(v, "on") == 0) return true;
    if (std::strcmp(v, "ON") == 0) return true;

    return fallback;
}

static const char* find_test_model_path() {
    // For auto-detection, match the integration test: rely on ASTRAL_MODEL_MIN_BYTES,
    // with a conservative default that covers the default downloader output.
    const uint64_t min_bytes = parse_u64_env("ASTRAL_MODEL_MIN_BYTES", 70000000ULL);

    // Prefer a decode-capable model from known local paths (gpt2/tinyllama).
    // Users can override explicitly with ASTRAL_TEST_DECODE_MODEL.
    const char* env_decode = std::getenv("ASTRAL_TEST_DECODE_MODEL");
    if (file_exists_min_size(env_decode, 10ULL * 1024ULL * 1024ULL)) {
        return env_decode;
    }

    // Keep these paths in sync with tests/model_downloader.sh and test_integration.cpp.
    static const char* paths[] = {
        "tests/models/gpt2.Q2_K.gguf",
        "../tests/models/gpt2.Q2_K.gguf",
        "../../tests/models/gpt2.Q2_K.gguf",
        "../../../tests/models/gpt2.Q2_K.gguf",
        "../../../../tests/models/gpt2.Q2_K.gguf",

        "tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../../../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../../../../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",

        "tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../../../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../../../../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    };

    for (const char* p : paths) {
        if (file_exists_min_size(p, min_bytes)) {
            return p;
        }
    }

    // Fallback: allow ASTRAL_TEST_MODEL if nothing else is present.
    const char* env_path = std::getenv("ASTRAL_TEST_MODEL");
    if (file_exists_min_size(env_path, 10ULL * 1024ULL * 1024ULL)) {
        return env_path;
    }

    return nullptr;
}

static void drain_stream(AstralHandle session) {
    uint8_t buf[256];
    for (;;) {
        AstralMutSpanU8 out{};
        out.data = buf;
        out.len = sizeof(buf);

        const int32_t n = astral_stream_read(session, out, 10);
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        if (n <= 0) {
            break;
        }
    }

    // Ensure terminal state observed.
    (void)astral_session_wait(session, 5000);
}

static void run_no_alloc_gate_for_backend(const char* backend_name, const char* model_path) {
    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    if (backend_name != nullptr) {
        model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend_name);
        model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend_name));
    }

    if (model_path != nullptr) {
        model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
        model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));
    }

    model_desc.n_ctx = 256;
    model_desc.n_batch = 128;
    model_desc.n_threads = 2;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    AstralErr err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = 256;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    // Warmup: run once without tracking to allow one-time init.
    {
        const char* prompt = "";
        AstralSpanU8 chunk{};
        chunk.data = reinterpret_cast<const uint8_t*>(prompt);
        chunk.len = 0;
        err = astral_session_feed(session, chunk, 1);
        ASSERT_EQ(err, ASTRAL_OK);

        err = astral_session_decode(session);
        ASSERT_EQ(err, ASTRAL_OK);
        drain_stream(session);
    }

    // Reset, then enforce allocation-free decode/stream.
    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_OK);

    {
        const char* prompt = "";
        AstralSpanU8 chunk{};
        chunk.data = reinterpret_cast<const uint8_t*>(prompt);
        chunk.len = 0;
        err = astral_session_feed(session, chunk, 1);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    tracking_reset();
    tracking_set_enabled(true);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    drain_stream(session);

    tracking_set_enabled(false);

    const uint64_t alloc_calls = g_alloc_calls.load(std::memory_order_relaxed);
    const uint64_t alloc_bytes = g_alloc_bytes.load(std::memory_order_relaxed);
    const uint64_t new_calls = g_new_calls.load(std::memory_order_relaxed);
    const uint64_t new_bytes = g_new_bytes.load(std::memory_order_relaxed);
    const uint64_t total_calls = alloc_calls + new_calls;

    if (total_calls != 0ULL) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "allocation gate: total=%llu malloc=%llu (%llu bytes) new=%llu (%llu bytes)",
                      static_cast<unsigned long long>(total_calls),
                      static_cast<unsigned long long>(alloc_calls),
                      static_cast<unsigned long long>(alloc_bytes),
                      static_cast<unsigned long long>(new_calls),
                      static_cast<unsigned long long>(new_bytes));
        ::astral::testing::test_fail_msg(__FILE__, __LINE__, msg);
    }

    astral_session_destroy(session);
    astral_model_release(model);
}

static void run_no_alloc_gate_for_embeddings(const char* backend_name, const char* model_path) {
    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    if (backend_name != nullptr) {
        model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend_name);
        model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend_name));
    }

    if (model_path != nullptr) {
        model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
        model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));
    }

    model_desc.n_ctx = 256;
    model_desc.n_batch = 128;
    model_desc.n_threads = 2;
    model_desc.gpu_layers = 0;
    model_desc.embeddings_only = 1;

    AstralHandle model = 0;
    AstralErr err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(dim, 0u);
    ASSERT_LE(dim, 8192u);

    AstralHandle emb = 0;
    err = astral_embed_create(model, &emb);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* text = "abc";
    AstralSpanU8 text_span{};
    text_span.data = reinterpret_cast<const uint8_t*>(text);
    text_span.len = static_cast<uint32_t>(std::strlen(text));

    static float vec[8192];
    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = static_cast<uint32_t>(sizeof(vec));

    // Warmup: allow one-time init in provider/library.
    {
        uint64_t ticket = 0;
        err = astral_embed_enqueue(emb, text_span, &ticket);
        ASSERT_EQ(err, ASTRAL_OK);
        err = astral_embed_collect(emb, ticket, out);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    {
        uint64_t ticket = 0;
        err = astral_embed_enqueue(emb, text_span, &ticket);
        ASSERT_EQ(err, ASTRAL_OK);

        // Match the decode/stream gate: exclude tokenization/enqueue from the tracked window and
        // validate that the steady-state embedding compute/collection does not allocate.
        tracking_reset();
        tracking_set_enabled(true);

        err = astral_embed_collect(emb, ticket, out);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    tracking_set_enabled(false);

    const uint64_t alloc_calls = g_alloc_calls.load(std::memory_order_relaxed);
    const uint64_t alloc_bytes = g_alloc_bytes.load(std::memory_order_relaxed);
    const uint64_t new_calls = g_new_calls.load(std::memory_order_relaxed);
    const uint64_t new_bytes = g_new_bytes.load(std::memory_order_relaxed);
    const uint64_t total_calls = alloc_calls + new_calls;

    if (total_calls != 0ULL) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "allocation gate (embeddings): total=%llu malloc=%llu (%llu bytes) new=%llu (%llu bytes)",
                      static_cast<unsigned long long>(total_calls),
                      static_cast<unsigned long long>(alloc_calls),
                      static_cast<unsigned long long>(alloc_bytes),
                      static_cast<unsigned long long>(new_calls),
                      static_cast<unsigned long long>(new_bytes));
        ::astral::testing::test_fail_msg(__FILE__, __LINE__, msg);
    }

    astral_embed_destroy(emb);
    astral_model_release(model);
}

} // namespace

// Best-effort global new tracking (also useful when malloc wrapping isn't available).
void* operator new(std::size_t size) {
    if (g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_new_calls.fetch_add(1, std::memory_order_relaxed);
        g_new_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }

    void* p = std::malloc(size == 0 ? 1 : size);
    if (p) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

// Sized delete overloads are required on some toolchains/configs (C++14+).
// Without these, deallocations can bypass our hooks and/or trigger mismatches.
void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void* operator new[](std::size_t size) {
    if (g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_new_calls.fetch_add(1, std::memory_order_relaxed);
        g_new_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }

    void* p = std::malloc(size == 0 ? 1 : size);
    if (p) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

#if defined(__cpp_aligned_new)

void* operator new(std::size_t size, std::align_val_t align) {
    if (g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_new_calls.fetch_add(1, std::memory_order_relaxed);
        g_new_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }

    const std::size_t a = static_cast<std::size_t>(align);
#if defined(_WIN32)
    void* p = _aligned_malloc(size == 0 ? 1 : size, a);
    if (p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
#else
    void* p = nullptr;
    if (posix_memalign(&p, a, size == 0 ? 1 : size) == 0 && p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
#endif
}

void operator delete(void* ptr, std::align_val_t) noexcept {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void* operator new[](std::size_t size, std::align_val_t align) {
    if (g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_new_calls.fetch_add(1, std::memory_order_relaxed);
        g_new_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }

    const std::size_t a = static_cast<std::size_t>(align);
#if defined(_WIN32)
    void* p = _aligned_malloc(size == 0 ? 1 : size, a);
    if (p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
#else
    void* p = nullptr;
    if (posix_memalign(&p, a, size == 0 ? 1 : size) == 0 && p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
#endif
}

void operator delete[](void* ptr, std::align_val_t) noexcept {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

#endif

#if defined(ASTRAL_ALLOC_WRAP)

extern "C" void* __real_malloc(std::size_t size);
extern "C" void* __real_calloc(std::size_t n, std::size_t size);
extern "C" void* __real_realloc(void* ptr, std::size_t size);
extern "C" void __real_free(void* ptr);
extern "C" int __real_posix_memalign(void** memptr, std::size_t alignment, std::size_t size);

extern "C" void* __wrap_malloc(std::size_t size) {
    if (g_wrap_reentry) {
        return __real_malloc(size);
    }
    g_wrap_reentry = true;

    if (g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        g_alloc_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }

    void* p = __real_malloc(size);
    g_wrap_reentry = false;
    return p;
}

extern "C" void* __wrap_calloc(std::size_t n, std::size_t size) {
    if (g_wrap_reentry) {
        return __real_calloc(n, size);
    }
    g_wrap_reentry = true;

    if (g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        g_alloc_bytes.fetch_add(static_cast<uint64_t>(n) * static_cast<uint64_t>(size),
                                std::memory_order_relaxed);
    }

    void* p = __real_calloc(n, size);
    g_wrap_reentry = false;
    return p;
}

extern "C" void* __wrap_realloc(void* ptr, std::size_t size) {
    if (g_wrap_reentry) {
        return __real_realloc(ptr, size);
    }
    g_wrap_reentry = true;

    if (g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        g_alloc_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }

    void* p = __real_realloc(ptr, size);
    g_wrap_reentry = false;
    return p;
}

extern "C" void __wrap_free(void* ptr) {
    __real_free(ptr);
}

extern "C" int __wrap_posix_memalign(void** memptr, std::size_t alignment, std::size_t size) {
    if (g_wrap_reentry) {
        return __real_posix_memalign(memptr, alignment, size);
    }
    g_wrap_reentry = true;

    const int rc = __real_posix_memalign(memptr, alignment, size);
    if (rc == 0 && g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        g_alloc_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }

    g_wrap_reentry = false;
    return rc;
}

#endif

TEST(gate_no_alloc_decode_stream_hotpath) {
    tracking_set_enabled(false);

    AstralInit cfg{};
    cfg.reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Always gate against mock backend (deterministic, no model file required).
    // Use the "infinite" mode so we exercise steady-state token generation.
    run_no_alloc_gate_for_backend("mock", "infinite");
    run_no_alloc_gate_for_embeddings("mock", "infinite");

    // CPU backend: enforce steady-state no-alloc when a GGUF is available.
    // You can disable this locally via ASTRAL_GATE_CPU_ALLOC=0.
    if (parse_bool_env("ASTRAL_GATE_CPU_ALLOC", true)) {
        const char* gguf_path = find_test_model_path();
        if (gguf_path != nullptr) {
            run_no_alloc_gate_for_backend("cpu", gguf_path);
            // CPU embeddings gating is optional (enabled via ASTRAL_GATE_CPU_EMB_ALLOC=1).
            if (parse_bool_env("ASTRAL_GATE_CPU_EMB_ALLOC", false)) {
                run_no_alloc_gate_for_embeddings("cpu", gguf_path);
            }
        }
    }

    astral_shutdown();
}
