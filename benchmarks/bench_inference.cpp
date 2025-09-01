#include "bench_clock.hpp"

#include "../include/astral_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

namespace astral::bench {

namespace {

struct InferenceBenchResult {
    const char* name;
    uint64_t ttft_ticks;
    uint64_t ttft_ns;
    uint64_t total_ticks;
    uint64_t total_ns;
    uint64_t tokens;
    uint64_t bytes;
};

static uint32_t parse_u32_env(const char* key, uint32_t fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    unsigned long x = std::strtoul(v, &end, 10);
    if (end == v) {
        return fallback;
    }
    return static_cast<uint32_t>(x);
}

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

static const char* find_bench_model_path() {
    const uint64_t min_bytes = []() -> uint64_t {
        const char* v = std::getenv("ASTRAL_MODEL_MIN_BYTES");
        if (v == nullptr || v[0] == '\0') {
            // Keep in sync with tests/model_downloader.sh default.
            return 70000000ULL;
        }
        char* end = nullptr;
        unsigned long long x = std::strtoull(v, &end, 10);
        if (end == v) {
            return 70000000ULL;
        }
        return static_cast<uint64_t>(x);
    }();

    // Prefer explicit env vars (allows using a smaller model).
    const char* env = std::getenv("ASTRAL_BENCH_MODEL");
    if (env && env[0] != '\0') {
        if (file_is_large_enough(env, 10ull * 1024ull * 1024ull)) {
            return env;
        }
        std::fprintf(stderr, "[bench] ASTRAL_BENCH_MODEL set but file missing/too small: %s\n", env);
        return nullptr;
    }

    env = std::getenv("ASTRAL_TEST_MODEL");
    if (env && env[0] != '\0') {
        if (file_is_large_enough(env, 10ull * 1024ull * 1024ull)) {
            return env;
        }
        std::fprintf(stderr, "[bench] ASTRAL_TEST_MODEL set but file missing/too small: %s\n", env);
        return nullptr;
    }

    // Repo default (gpt2 Q2_K, small). We check a few likely run dirs.
    static const char* paths[] = {
        "tests/models/gpt2.Q2_K.gguf",
        "../tests/models/gpt2.Q2_K.gguf",
        "../../tests/models/gpt2.Q2_K.gguf",
        "../../../tests/models/gpt2.Q2_K.gguf",

        "tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../../../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",

        // Legacy filename.
        "tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../../../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    };

    for (const char* p : paths) {
        if (file_is_large_enough(p, min_bytes)) {
            return p;
        }
    }

    return nullptr;
}

static InferenceBenchResult run_streamed_decode(AstralHandle session) {
    InferenceBenchResult r{};
    r.name = "CPU inference (streamed)";

    const char* prompt = "Once upon a time";
    AstralSpanU8 prompt_span{};
    prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
    prompt_span.len = static_cast<uint32_t>(std::strlen(prompt));

    AstralErr err = astral_session_feed(session, prompt_span, 1);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_session_feed failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    err = astral_session_decode(session);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_session_decode failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        return r;
    }

    bool saw_first = false;
    uint64_t first_ticks = 0;
    uint64_t first_ns = 0;

    uint64_t tokens = 0;
    uint64_t bytes = 0;
    uint8_t buf[64];

    const uint64_t deadline_ns = n0 + 120ull * 1000ull * 1000ull * 1000ull;

    for (;;) {
        AstralMutSpanU8 out{};
        out.data = buf;
        out.len = sizeof(buf);

        const int32_t n = astral_stream_read(session, out, 1000);
        if (n == ASTRAL_E_TIMEOUT) {
            if (ns_now() > deadline_ns) {
                std::fprintf(stderr, "[bench] stream_read timed out waiting for output\n");
                break;
            }
            continue;
        }
        if (n < 0) {
            std::fprintf(stderr, "[bench] astral_stream_read failed: %s (%s)\n",
                         astral_error_string(static_cast<AstralErr>(n)),
                         astral_last_error());
            break;
        }
        if (n == 0) {
            break;
        }

        if (!saw_first) {
            first_ticks = ticks_now();
            first_ns = ns_now();
            saw_first = true;
        }

        ++tokens;
        bytes += static_cast<uint64_t>(n);
    }

    // Wait for final status (should be immediate after EOF).
    const AstralErr wait_err = astral_session_wait(session, 60000);
    if (wait_err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_session_wait returned: %s (%s)\n",
                     astral_error_string(wait_err),
                     astral_last_error());
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    if (saw_first) {
        r.ttft_ticks = first_ticks - t0;
        r.ttft_ns = first_ns - n0;
    }
    r.total_ticks = t1 - t0;
    r.total_ns = n1 - n0;
    r.tokens = tokens;
    r.bytes = bytes;

    return r;
}

} // namespace

void bench_inference_print(uint32_t warmup_tokens, uint32_t measure_tokens) {
    const char* model_path = find_bench_model_path();
    if (model_path == nullptr) {
        std::printf("[bench] Inference: SKIP (set ASTRAL_BENCH_MODEL or place a model under tests/models/)\n");
        return;
    }

    AstralInit cfg{};
    cfg.reserve_bytes = 2ull << 30; // 2GB address space (virtual reserve).
    cfg.thread_count = parse_u32_env("ASTRAL_BENCH_THREADS", 0);
    cfg.numa_node = 0xFFFFFFFFu;    // any
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_init failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        return;
    }

    AstralModelDesc model_desc{};
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));
    // Let the backend pick a safe/default context size for the model.
    model_desc.n_ctx = 0;
    model_desc.n_batch = 0;
    model_desc.n_threads = parse_u32_env("ASTRAL_BENCH_THREADS", 0);
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_model_load failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        astral_shutdown();
        return;
    }

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = warmup_tokens > 0 ? warmup_tokens : measure_tokens;
    session_desc.temperature = 0.8f;
    session_desc.top_k = 40;
    session_desc.top_p = 0.95f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_session_create failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        astral_model_release(model);
        astral_shutdown();
        return;
    }

    // Warm up (allows provider to allocate internal caches before measurement).
    if (warmup_tokens > 0) {
        (void)run_streamed_decode(session);
    }

    session_desc.max_tokens = measure_tokens;
    session_desc.seed = 1234;
    err = astral_session_reset(session, &session_desc);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_session_reset failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        astral_session_destroy(session);
        astral_model_release(model);
        astral_shutdown();
        return;
    }

    const InferenceBenchResult r = run_streamed_decode(session);

    if (r.tokens == 0 || r.total_ns == 0) {
        std::printf("[bench] Inference: FAILED (no tokens)\n");
        astral_session_destroy(session);
        astral_model_release(model);
        astral_shutdown();
        return;
    }

    const double total_s = static_cast<double>(r.total_ns) * 1e-9;
    const double tok_per_s = total_s > 0.0 ? static_cast<double>(r.tokens) / total_s : 0.0;
    const double ticks_per_tok = r.tokens > 0 ? static_cast<double>(r.total_ticks) / static_cast<double>(r.tokens)
                                              : 0.0;

    std::printf("\nInference benchmark (ticks: %s)\n", clock_info().name);
    std::printf("Model: %s\n", model_path);

    if (r.ttft_ns > 0) {
        std::printf("TTFT:  %.2f ms  (%llu ticks)\n",
                    static_cast<double>(r.ttft_ns) / 1e6,
                    static_cast<unsigned long long>(r.ttft_ticks));
    } else {
        std::printf("TTFT:  n/a\n");
    }

    std::printf("Total: %.2f ms  (%llu ticks)\n",
                static_cast<double>(r.total_ns) / 1e6,
                static_cast<unsigned long long>(r.total_ticks));
    std::printf("Tok/s: %.2f  (%.2f ticks/token)\n", tok_per_s, ticks_per_tok);
    std::printf("Tokens: %llu, bytes: %llu\n\n",
                static_cast<unsigned long long>(r.tokens),
                static_cast<unsigned long long>(r.bytes));

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

} // namespace astral::bench
