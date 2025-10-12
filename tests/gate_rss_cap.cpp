/**
 * gate_rss_cap.cpp - Process RSS cap (embedded validation)
 *
 * Goal:
 * - Provide an opt-in RSS cap gate for embedded/edge validation.
 *
 * Notes:
 * - RSS is OS-dependent; this gate currently enforces on Linux via /proc.
 * - Default cap is intentionally high to avoid surprising failures; set
 *   ASTRAL_RSS_MAX_MB to tighten.
 */

#include "test_framework.hpp"
#include "astral_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
  #include <sys/stat.h>
#endif

namespace {

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

#if defined(__linux__)
static uint64_t read_rss_kb_linux() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (f == nullptr) {
        return 0;
    }

    char line[512];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            const char* p = line + 6;
            while (*p == ' ' || *p == '\t') ++p;
            kb = std::strtoull(p, nullptr, 10);
            break;
        }
    }

    std::fclose(f);
    return kb;
}
#endif

static bool file_exists_min_size(const char* path, uint64_t min_bytes) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
#if defined(__linux__)
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    return static_cast<uint64_t>(st.st_size) >= min_bytes;
#else
    (void)min_bytes;
    return true;
#endif
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

} // namespace

TEST(gate_rss_cap) {
#if !defined(__linux__)
    // Not enforced on this platform in v0.1.
    ASSERT_TRUE(true);
    return;
#else
    const uint64_t max_mb = parse_u64_env("ASTRAL_RSS_MAX_MB", 8192ULL);
    const uint64_t max_kb = max_mb * 1024ULL;

    const char* gguf = find_test_model_path();
    if (gguf == nullptr) {
        // Model RSS sampling only runs when a GGUF fixture is configured; mock-default lanes have no model load.
        ASSERT_TRUE(true);
        return;
    }

    AstralInit cfg{};
    cfg.reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    uint64_t rss_peak = read_rss_kb_linux();

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(gguf);
    model_desc.model_path.len = static_cast<uint32_t>(std::strlen(gguf));
    model_desc.n_ctx = 256;
    model_desc.n_batch = 128;
    model_desc.n_threads = 2;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    {
        const uint64_t rss = read_rss_kb_linux();
        if (rss > rss_peak) rss_peak = rss;
    }

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = 64;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    {
        const uint64_t rss = read_rss_kb_linux();
        if (rss > rss_peak) rss_peak = rss;
    }

    // Warmup once.
    AstralSpanU8 chunk{};
    chunk.data = reinterpret_cast<const uint8_t*>("");
    chunk.len = 0;
    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    (void)astral_session_wait(session, 60000);

    // Drain any stream output (bounded).
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

    {
        const uint64_t rss = read_rss_kb_linux();
        if (rss > rss_peak) rss_peak = rss;
    }

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();

    if (rss_peak > max_kb) {
        char msg[256];
        std::snprintf(msg,
                      sizeof(msg),
                      "RSS cap exceeded: rss=%llu KB cap=%llu KB",
                      static_cast<unsigned long long>(rss_peak),
                      static_cast<unsigned long long>(max_kb));
        ::astral::testing::test_fail_msg(__FILE__, __LINE__, msg);
    }
#endif
}
