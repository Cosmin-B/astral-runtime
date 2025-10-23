/**
 * test_provider_harness.cpp - Provider swap harness tests
 *
 * Purpose:
 * - Validate that the core runtime remains provider-agnostic:
 *   backend selection happens at model load, and decode/stream uses a cached ops table.
 * - Provide a single “harness” path that can be reused for new providers without
 *   changing call sites.
 *
 * Notes:
 * - Always exercises the mock backend (no GGUF required).
 * - Exercises the CPU backend only when a GGUF is available.
 * - A placeholder "next" provider name is validated to fail cleanly.
 */

#include "test_framework.hpp"
#include "astral_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

namespace {

static bool file_is_large_enough(const char* path, uint64_t min_bytes) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    struct stat st {};
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

static const char* find_test_model_path() {
    const uint64_t min_bytes = parse_u64_env("ASTRAL_MODEL_MIN_BYTES", 70000000ULL);

    const char* env_decode = std::getenv("ASTRAL_TEST_DECODE_MODEL");
    if (env_decode && env_decode[0] != '\0') {
        if (file_is_large_enough(env_decode, 10ULL * 1024ULL * 1024ULL)) {
            return env_decode;
        }
        return nullptr;
    }

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
        if (file_is_large_enough(p, min_bytes)) {
            return p;
        }
    }

    // Fallback: allow ASTRAL_TEST_MODEL if no decode model is present.
    const char* env = std::getenv("ASTRAL_TEST_MODEL");
    if (env && env[0] != '\0') {
        if (file_is_large_enough(env, 10ULL * 1024ULL * 1024ULL)) {
            return env;
        }
    }

    return nullptr;
}

struct ProviderCase {
    const char* name;
    const char* model_path; // may be null for providers that don't need a file
};

static void run_provider_case(const ProviderCase& c) {
    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;

    if (c.name != nullptr) {
        model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(c.name);
        model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(c.name));
    }

    if (c.model_path != nullptr) {
        model_desc.model_path.data = reinterpret_cast<const uint8_t*>(c.model_path);
        model_desc.model_path.len = static_cast<uint32_t>(std::strlen(c.model_path));
    }

    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.n_threads = 4;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    AstralErr err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = 32;
    session_desc.temperature = 0.0f; // greedy
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    const char* prompt = "hi";
    AstralSpanU8 chunk{};
    chunk.data = reinterpret_cast<const uint8_t*>(prompt);
    chunk.len = static_cast<uint32_t>(std::strlen(prompt));
    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    uint8_t buf[256];
    uint32_t total = 0;
    for (uint32_t i = 0; i < 256; ++i) {
        AstralMutSpanU8 out{};
        out.data = buf + total;
        out.len = sizeof(buf) - total;

        const int32_t n = astral_stream_read(session, out, 1000);
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        if (n < 0) {
            ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
        }
        if (n == 0) {
            break;
        }

        total += static_cast<uint32_t>(n);
        if (total == sizeof(buf)) {
            break;
        }
    }

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(total, 0u);

    // Reset and re-run to validate lifecycle edge: reuse is supported.
    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    // Just confirm we can drain again.
    for (uint32_t i = 0; i < 256; ++i) {
        AstralMutSpanU8 out{};
        out.data = buf;
        out.len = sizeof(buf);

        const int32_t n = astral_stream_read(session, out, 1000);
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        if (n <= 0) {
            break;
        }
    }

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
}

} // namespace

TEST(provider_swap_harness_mock_and_cpu) {
    AstralInit cfg{};
    cfg.reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Mock backend is always available and requires no model file.
    run_provider_case(ProviderCase{"mock", nullptr});

    // CPU backend requires a GGUF. Skip if none found.
    const char* gguf = find_test_model_path();
    if (gguf != nullptr) {
        run_provider_case(ProviderCase{"cpu", gguf});
    }

    astral_shutdown();
}

TEST(provider_plugin_sample_load_and_run) {
#if defined(ASTRAL_SAMPLE_BACKEND_PLUGIN_PATH)
    const char* plugin_path = ASTRAL_SAMPLE_BACKEND_PLUGIN_PATH;
    ASSERT_TRUE(plugin_path != nullptr);

    AstralSpanU8 path{};
    path.data = reinterpret_cast<const uint8_t*>(plugin_path);
    path.len = static_cast<uint32_t>(std::strlen(plugin_path));

    AstralErr err = astral_backend_load_plugin(path);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralInit cfg{};
    cfg.reserve_bytes = 128ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    run_provider_case(ProviderCase{"sample", nullptr});

    astral_shutdown();
#else
    // Sample plugin not built in this configuration.
    ASSERT_TRUE(true);
#endif
}

TEST(provider_plugin_loader_rejects_relative_path) {
    const char* plugin_path = "astral_backend_sample_plugin.so";
    AstralSpanU8 path{};
    path.data = reinterpret_cast<const uint8_t*>(plugin_path);
    path.len = static_cast<uint32_t>(std::strlen(plugin_path));

    AstralErr err = astral_backend_load_plugin(path);
    ASSERT_EQ(err, ASTRAL_E_INVALID);
}

TEST(provider_plugin_loader_rejects_embedded_nul_path) {
    const uint8_t plugin_path[] = {'/', 't', 'm', 'p', '/', 'a', 0, 'b'};
    AstralSpanU8 path{};
    path.data = plugin_path;
    path.len = static_cast<uint32_t>(sizeof(plugin_path));

    AstralErr err = astral_backend_load_plugin(path);
    ASSERT_EQ(err, ASTRAL_E_INVALID);
}

TEST(provider_plugin_cpu_llama_load_and_run) {
#if defined(ASTRAL_CPU_LLAMA_BACKEND_PLUGIN_PATH)
    const char* plugin_path = ASTRAL_CPU_LLAMA_BACKEND_PLUGIN_PATH;
    ASSERT_TRUE(plugin_path != nullptr);

    const char* gguf = find_test_model_path();
    if (gguf == nullptr) {
        // CPU llama runtime coverage needs a GGUF fixture; sample-provider loading is covered above.
        ASSERT_TRUE(true);
        return;
    }

    AstralSpanU8 path{};
    path.data = reinterpret_cast<const uint8_t*>(plugin_path);
    path.len = static_cast<uint32_t>(std::strlen(plugin_path));

    AstralErr err = astral_backend_load_plugin(path);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralInit cfg{};
    cfg.reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    run_provider_case(ProviderCase{"cpu_llama", gguf});

    astral_shutdown();
#else
    // CPU llama plugin not built in this configuration.
    ASSERT_TRUE(true);
#endif
}

TEST(provider_plugin_toy_load_and_run) {
#if defined(ASTRAL_TOY_BACKEND_PLUGIN_PATH)
    const char* plugin_path = ASTRAL_TOY_BACKEND_PLUGIN_PATH;
    ASSERT_TRUE(plugin_path != nullptr);

    AstralSpanU8 path{};
    path.data = reinterpret_cast<const uint8_t*>(plugin_path);
    path.len = static_cast<uint32_t>(std::strlen(plugin_path));

    AstralErr err = astral_backend_load_plugin(path);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralInit cfg{};
    cfg.reserve_bytes = 64ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Use a fixed embedding dimension config string for determinism.
    run_provider_case(ProviderCase{"toy", "dim=256"});

    astral_shutdown();
#else
    ASSERT_TRUE(true);
#endif
}

TEST(provider_override_unknown_fails_cleanly) {
    AstralInit cfg{};
    cfg.reserve_bytes = 64ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* unknown = "next";
    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(unknown);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(unknown));
    model_desc.n_ctx = 128;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_E_BACKEND);

    astral_shutdown();
}
