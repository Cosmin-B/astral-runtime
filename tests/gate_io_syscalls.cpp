/**
 * gate_io_syscalls.cpp - Steady-state I/O syscall validation gate
 *
 * Goal:
 * - Fail the test suite if steady-state decode/stream performs any file or VM I/O calls
 *   that would imply syscalls (e.g., read/write/mmap).
 *
 * Scope:
 * - This gate intentionally targets I/O-like libc entry points (read/write/mmap/...),
 *   not kernel scheduling primitives.
 *
 * Strategy:
 * - Warmup decode first (to allow one-time init).
 * - Reset, then enable I/O tracking and run decode+stream again.
 * - Require zero tracked I/O calls while tracking is enabled.
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

#if defined(__linux__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <sys/types.h>
  #include <unistd.h>
#endif

namespace {

std::atomic<uint64_t> g_io_calls{0};
std::atomic<bool> g_tracking_enabled{false};

thread_local bool g_wrap_reentry = false;

static void tracking_reset() {
    g_io_calls.store(0, std::memory_order_relaxed);
}

static void tracking_set_enabled(bool enabled) {
    g_tracking_enabled.store(enabled, std::memory_order_relaxed);
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

    if (std::strcmp(v, "true") == 0) return true;
    if (std::strcmp(v, "TRUE") == 0) return true;
    if (std::strcmp(v, "yes") == 0) return true;
    if (std::strcmp(v, "YES") == 0) return true;
    if (std::strcmp(v, "on") == 0) return true;
    if (std::strcmp(v, "ON") == 0) return true;

    return fallback;
}

static const char* find_test_model_path() {
    const uint64_t min_bytes = parse_u64_env("ASTRAL_MODEL_MIN_BYTES", 70000000ULL);

    const char* env_decode = std::getenv("ASTRAL_TEST_DECODE_MODEL");
    if (file_exists_min_size(env_decode, 10ULL * 1024ULL * 1024ULL)) {
        return env_decode;
    }

    static const char* paths[] = {
        "tests/models/gpt2.Q2_K.gguf",
        "../tests/models/gpt2.Q2_K.gguf",
        "../../tests/models/gpt2.Q2_K.gguf",
        "../../../tests/models/gpt2.Q2_K.gguf",
        "../../../../tests/models/gpt2.Q2_K.gguf",
    };

    for (const char* p : paths) {
        if (file_exists_min_size(p, min_bytes)) {
            return p;
        }
    }

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

    (void)astral_session_wait(session, 5000);
}

static void run_no_io_gate_for_backend(const char* backend_name, const char* model_path) {
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

    // Warmup.
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

    // Reset, then enforce no tracked I/O calls in steady state.
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

    const uint64_t io_calls = g_io_calls.load(std::memory_order_relaxed);
    if (io_calls != 0ULL) {
        ::astral::testing::test_fail_msg(__FILE__, __LINE__, "I/O gate: steady-state I/O calls detected");
    }

    astral_session_destroy(session);
    astral_model_release(model);
}

static void run_no_io_gate_for_embeddings(const char* backend_name, const char* model_path) {
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
    ASSERT_TRUE(astral_handle_valid(model));

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

    // Warmup (allow one-time init / model page faults).
    {
        uint64_t ticket = 0;
        err = astral_embed_enqueue(emb, text_span, &ticket);
        ASSERT_EQ(err, ASTRAL_OK);
        err = astral_embed_collect(emb, ticket, out);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    // Gate the compute/collection portion (steady-state). Keep enqueue/tokenize outside the window.
    {
        uint64_t ticket = 0;
        err = astral_embed_enqueue(emb, text_span, &ticket);
        ASSERT_EQ(err, ASTRAL_OK);

        tracking_reset();
        tracking_set_enabled(true);

        err = astral_embed_collect(emb, ticket, out);
        ASSERT_EQ(err, ASTRAL_OK);

        tracking_set_enabled(false);
    }

    const uint64_t io_calls = g_io_calls.load(std::memory_order_relaxed);
    if (io_calls != 0ULL) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "I/O gate (embeddings): calls=%llu", static_cast<unsigned long long>(io_calls));
        ::astral::testing::test_fail_msg(__FILE__, __LINE__, msg);
    }

    astral_embed_destroy(emb);
    astral_model_release(model);
}

} // namespace

#if defined(ASTRAL_IO_WRAP)

extern "C" ssize_t __real_read(int fd, void* buf, size_t count);
extern "C" ssize_t __real_write(int fd, const void* buf, size_t count);
extern "C" ssize_t __real_pread(int fd, void* buf, size_t count, off_t offset);
extern "C" ssize_t __real_pwrite(int fd, const void* buf, size_t count, off_t offset);
extern "C" off_t __real_lseek(int fd, off_t offset, int whence);
extern "C" int __real_close(int fd);
extern "C" void* __real_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset);
extern "C" int __real_munmap(void* addr, size_t length);
extern "C" int __real_mprotect(void* addr, size_t length, int prot);
extern "C" int __real_madvise(void* addr, size_t length, int advice);

static inline void io_count() {
    if (g_tracking_enabled.load(std::memory_order_relaxed)) {
        g_io_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

extern "C" ssize_t __wrap_read(int fd, void* buf, size_t count) {
    if (g_wrap_reentry) {
        return __real_read(fd, buf, count);
    }
    g_wrap_reentry = true;
    io_count();
    const ssize_t rc = __real_read(fd, buf, count);
    g_wrap_reentry = false;
    return rc;
}

extern "C" ssize_t __wrap_write(int fd, const void* buf, size_t count) {
    if (g_wrap_reentry) {
        return __real_write(fd, buf, count);
    }
    g_wrap_reentry = true;
    io_count();
    const ssize_t rc = __real_write(fd, buf, count);
    g_wrap_reentry = false;
    return rc;
}

extern "C" ssize_t __wrap_pread(int fd, void* buf, size_t count, off_t offset) {
    if (g_wrap_reentry) {
        return __real_pread(fd, buf, count, offset);
    }
    g_wrap_reentry = true;
    io_count();
    const ssize_t rc = __real_pread(fd, buf, count, offset);
    g_wrap_reentry = false;
    return rc;
}

extern "C" ssize_t __wrap_pwrite(int fd, const void* buf, size_t count, off_t offset) {
    if (g_wrap_reentry) {
        return __real_pwrite(fd, buf, count, offset);
    }
    g_wrap_reentry = true;
    io_count();
    const ssize_t rc = __real_pwrite(fd, buf, count, offset);
    g_wrap_reentry = false;
    return rc;
}

extern "C" off_t __wrap_lseek(int fd, off_t offset, int whence) {
    if (g_wrap_reentry) {
        return __real_lseek(fd, offset, whence);
    }
    g_wrap_reentry = true;
    io_count();
    const off_t rc = __real_lseek(fd, offset, whence);
    g_wrap_reentry = false;
    return rc;
}

extern "C" int __wrap_close(int fd) {
    if (g_wrap_reentry) {
        return __real_close(fd);
    }
    g_wrap_reentry = true;
    io_count();
    const int rc = __real_close(fd);
    g_wrap_reentry = false;
    return rc;
}

extern "C" void* __wrap_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (g_wrap_reentry) {
        return __real_mmap(addr, length, prot, flags, fd, offset);
    }
    g_wrap_reentry = true;
    io_count();
    void* rc = __real_mmap(addr, length, prot, flags, fd, offset);
    g_wrap_reentry = false;
    return rc;
}

extern "C" int __wrap_munmap(void* addr, size_t length) {
    if (g_wrap_reentry) {
        return __real_munmap(addr, length);
    }
    g_wrap_reentry = true;
    io_count();
    const int rc = __real_munmap(addr, length);
    g_wrap_reentry = false;
    return rc;
}

extern "C" int __wrap_mprotect(void* addr, size_t length, int prot) {
    if (g_wrap_reentry) {
        return __real_mprotect(addr, length, prot);
    }
    g_wrap_reentry = true;
    io_count();
    const int rc = __real_mprotect(addr, length, prot);
    g_wrap_reentry = false;
    return rc;
}

extern "C" int __wrap_madvise(void* addr, size_t length, int advice) {
    if (g_wrap_reentry) {
        return __real_madvise(addr, length, advice);
    }
    g_wrap_reentry = true;
    io_count();
    const int rc = __real_madvise(addr, length, advice);
    g_wrap_reentry = false;
    return rc;
}

#endif

TEST(gate_no_io_decode_stream_hotpath) {
    tracking_set_enabled(false);

    AstralInit cfg{};
    cfg.reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    run_no_io_gate_for_backend("mock", "infinite");
    run_no_io_gate_for_embeddings("mock", "infinite");

    if (parse_bool_env("ASTRAL_GATE_CPU_IO", true)) {
        const char* gguf_path = find_test_model_path();
        if (gguf_path != nullptr) {
            run_no_io_gate_for_backend("cpu", gguf_path);
            if (parse_bool_env("ASTRAL_GATE_CPU_EMB_IO", false)) {
                run_no_io_gate_for_embeddings("cpu", gguf_path);
            }
        }
    }

    astral_shutdown();
}
