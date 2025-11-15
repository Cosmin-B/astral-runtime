/**
 * gate_model_churn_soak.cpp - Repeated model/session churn with RSS sampling.
 *
 * Default coverage is mock-backed and quick enough for the release-with-tests
 * lane. Set ASTRAL_SOAK_MODEL to a GGUF path on release runners for real CPU
 * model churn.
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

struct ChurnConfig
{
    const char* Backend = "mock";
    const char* ModelPath = nullptr;
    uint32_t Cycles = 8;
    uint64_t MaxRssDriftKb = 64ULL * 1024ULL;
};

static uint64_t parse_u64_env(const char* key, uint64_t fallback)
{
    const char* value = std::getenv(key);
    if (value == nullptr || value[0] == '\0')
    {
        return fallback;
    }

    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value)
    {
        return fallback;
    }
    return static_cast<uint64_t>(parsed);
}

#if defined(__linux__)
static uint64_t read_rss_kb()
{
    FILE* file = std::fopen("/proc/self/status", "r");
    if (file == nullptr)
    {
        return 0;
    }

    char line[512];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr)
    {
        if (std::strncmp(line, "VmRSS:", 6) == 0)
        {
            const char* cursor = line + 6;
            while (*cursor == ' ' || *cursor == '\t')
            {
                ++cursor;
            }
            kb = std::strtoull(cursor, nullptr, 10);
            break;
        }
    }

    std::fclose(file);
    return kb;
}

static bool file_exists(const char* path)
{
    if (path == nullptr || path[0] == '\0')
    {
        return false;
    }

    struct stat st;
    return stat(path, &st) == 0 && st.st_size > 0;
}
#endif

static void sample_rss(uint64_t* peak)
{
#if defined(__linux__)
    const uint64_t rss = read_rss_kb();
    if (rss > *peak)
    {
        *peak = rss;
    }
#else
    (void)peak;
#endif
}

static AstralHandle load_model(const ChurnConfig& cfg)
{
    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    desc.backend_name.data = reinterpret_cast<const uint8_t*>(cfg.Backend);
    desc.backend_name.len = static_cast<uint32_t>(std::strlen(cfg.Backend));
    desc.n_ctx = 128;
    desc.n_batch = 64;
    desc.n_threads = 2;
    desc.gpu_layers = 0;

    if (cfg.ModelPath != nullptr)
    {
        desc.model_path.data = reinterpret_cast<const uint8_t*>(cfg.ModelPath);
        desc.model_path.len = static_cast<uint32_t>(std::strlen(cfg.ModelPath));
    }

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));
    return model;
}

static uint32_t drain_stream(AstralHandle session)
{
    uint8_t buf[256];
    uint32_t total = 0;
    for (uint32_t i = 0; i < 512; ++i)
    {
        AstralMutSpanU8 out{};
        out.data = buf;
        out.len = sizeof(buf);

        const int32_t n = astral_stream_read(session, out, 100);
        if (n == 0)
        {
            break;
        }
        if (n == ASTRAL_E_TIMEOUT)
        {
            continue;
        }
        ASSERT_GT(n, 0);
        total += static_cast<uint32_t>(n);
    }
    return total;
}

static void decode_once(AstralHandle session)
{
    const char* prompt = "churn";
    AstralSpanU8 chunk{};
    chunk.data = reinterpret_cast<const uint8_t*>(prompt);
    chunk.len = static_cast<uint32_t>(std::strlen(prompt));

    AstralErr err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 30000);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(drain_stream(session), 0u);
}

static void run_session_churn(AstralHandle model, uint32_t seed)
{
    AstralSessionDesc desc{};
    desc.model = model;
    desc.max_tokens = 16;
    desc.temperature = 0.0f;
    desc.top_k = 0;
    desc.top_p = 1.0f;
    desc.stream_enabled = 1;
    desc.seed = seed;

    AstralHandle session = 0;
    AstralErr err = astral_session_create(&desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    decode_once(session);

    err = astral_session_reset(session, &desc);
    ASSERT_EQ(err, ASTRAL_OK);
    decode_once(session);

    astral_session_destroy(session);
}

static void run_model_churn_soak(const ChurnConfig& cfg)
{
    uint64_t rss_start = 0;
    uint64_t rss_peak = 0;
    sample_rss(&rss_start);
    rss_peak = rss_start;

    for (uint32_t cycle = 0; cycle < cfg.Cycles; ++cycle)
    {
        AstralInit init{};
        init.reserve_bytes = 128ULL * 1024ULL * 1024ULL;
        init.thread_count = 2;
        init.numa_node = 0xFFFFFFFFu;
        init.enable_hugepages = 0;

        AstralErr err = astral_init(&init);
        ASSERT_EQ(err, ASTRAL_OK);

        AstralHandle model = load_model(cfg);
        sample_rss(&rss_peak);

        run_session_churn(model, 1000u + cycle);
        sample_rss(&rss_peak);

        astral_model_release(model);
        astral_shutdown();
        sample_rss(&rss_peak);
    }

#if defined(__linux__)
    if (rss_start != 0 && rss_peak > rss_start + cfg.MaxRssDriftKb)
    {
        char msg[256];
        std::snprintf(msg,
                      sizeof(msg),
                      "model churn RSS drift exceeded: start=%llu KB peak=%llu KB cap=%llu KB",
                      static_cast<unsigned long long>(rss_start),
                      static_cast<unsigned long long>(rss_peak),
                      static_cast<unsigned long long>(cfg.MaxRssDriftKb));
        ::astral::testing::test_fail_msg(__FILE__, __LINE__, msg);
    }
#endif
}

} // namespace

TEST(gate_model_churn_soak_mock)
{
    ChurnConfig cfg{};
    cfg.Backend = "mock";
    cfg.ModelPath = nullptr;
    cfg.Cycles = static_cast<uint32_t>(parse_u64_env("ASTRAL_SOAK_MOCK_CYCLES", 8));
    cfg.MaxRssDriftKb = parse_u64_env("ASTRAL_SOAK_RSS_DRIFT_MB", 64) * 1024ULL;
    run_model_churn_soak(cfg);
}

TEST(gate_model_churn_soak_real_model_probe)
{
    const char* model_path = std::getenv("ASTRAL_SOAK_MODEL");
    if (model_path == nullptr || model_path[0] == '\0')
    {
        SKIP_TEST("ASTRAL_SOAK_MODEL is required for real GGUF churn soak");
    }

#if defined(__linux__)
    if (!file_exists(model_path))
    {
        SKIP_TEST("ASTRAL_SOAK_MODEL does not point to a readable model");
    }
#endif

    ChurnConfig cfg{};
    cfg.Backend = "cpu";
    cfg.ModelPath = model_path;
    cfg.Cycles = static_cast<uint32_t>(parse_u64_env("ASTRAL_SOAK_REAL_CYCLES", 2));
    cfg.MaxRssDriftKb = parse_u64_env("ASTRAL_SOAK_RSS_DRIFT_MB", 512) * 1024ULL;
    run_model_churn_soak(cfg);
}
