#include "bench_clock.hpp"

#include "../include/astral_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
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

static uint32_t bench_runtime_threads() {
    // The benchmark only needs a small worker pool to run the decode job.
    // Default to 1 to avoid idle worker oversubscription affecting model scaling.
    return parse_u32_env("ASTRAL_BENCH_RUNTIME_THREADS", 1);
}

static uint32_t bench_model_threads() {
    // Prefer an explicit model-thread override, but accept the older ASTRAL_BENCH_THREADS
    // as a compatibility alias.
    const uint32_t explicit_model = parse_u32_env("ASTRAL_BENCH_MODEL_THREADS", 0);
    if (explicit_model != 0) {
        return explicit_model;
    }
    return parse_u32_env("ASTRAL_BENCH_THREADS", 0);
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

    uint64_t reads = 0;
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

        ++reads;
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
    // Note: `astral_stream_read()` returns bytes, not tokens. We record read count for debug only;
    // throughput should come from `astral_session_stats()` to avoid dependence on consumer buffer sizing.
    r.tokens = reads;
    r.bytes = bytes;

    return r;
}

static InferenceBenchResult run_continuous_batching_decode(
    AstralHandle model,
    uint32_t slots,
    uint32_t max_batch_tokens,
    uint32_t warmup_tokens,
    uint32_t measure_tokens
) {
    InferenceBenchResult r{};
    r.name = "CPU inference (continuous batching)";

    AstralExecutorDesc ex{};
    ex.size = sizeof(AstralExecutorDesc);
    ex.max_slots = slots;
    ex.max_batch_tokens = max_batch_tokens;
    ex.worker_hint = 0;

    AstralErr err = astral_model_executor_configure(model, &ex);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_model_executor_configure failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        return r;
    }

    const uint32_t prompt_repeat = parse_u32_env("ASTRAL_BENCH_PROMPT_REPEAT", 1);
    const uint32_t prompt_heavy_slots = parse_u32_env("ASTRAL_BENCH_PROMPT_HEAVY_SLOTS", 0);

    const char* prompt = "Once upon a time";
    const size_t base_len = std::strlen(prompt);
    std::string prompt_heavy;
    if (prompt_repeat > 1) {
        prompt_heavy.reserve(base_len * static_cast<size_t>(prompt_repeat) + static_cast<size_t>(prompt_repeat));
        for (uint32_t i = 0; i < prompt_repeat; ++i) {
            if (i) prompt_heavy.push_back(' ');
            prompt_heavy.append(prompt);
        }
    }

    AstralSpanU8 prompt_span{};
    prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
    prompt_span.len = static_cast<uint32_t>(base_len);

    AstralSpanU8 prompt_heavy_span{};
    if (!prompt_heavy.empty()) {
        prompt_heavy_span.data = reinterpret_cast<const uint8_t*>(prompt_heavy.data());
        prompt_heavy_span.len = static_cast<uint32_t>(prompt_heavy.size());
    } else {
        prompt_heavy_span = prompt_span;
    }

    AstralConvDesc conv_desc{};
    conv_desc.size = sizeof(AstralConvDesc);
    conv_desc.model = model;
    conv_desc.max_tokens = warmup_tokens > 0 ? warmup_tokens : measure_tokens;
    conv_desc.temperature = 0.8f;
    conv_desc.top_k = 40;
    conv_desc.top_p = 0.95f;
    conv_desc.stream_enabled = 0;
    conv_desc.seed = 1;

    std::vector<AstralHandle> convs(slots, 0);
    for (uint32_t i = 0; i < slots; ++i) {
        err = astral_conv_create(&conv_desc, &convs[i]);
        if (err != ASTRAL_OK) {
            std::fprintf(stderr, "[bench] astral_conv_create failed: %s (%s)\n",
                         astral_error_string(err),
                         astral_last_error());
            for (uint32_t j = 0; j < i; ++j) {
                astral_conv_destroy(convs[j]);
            }
            return r;
        }
    }

    auto run_round = [&](uint32_t max_tokens, uint32_t seed, bool measure, double* out_p50_ms, double* out_p95_ms) {
        AstralConvDesc desc = conv_desc;
        desc.max_tokens = max_tokens;
        desc.seed = seed;
        for (uint32_t i = 0; i < slots; ++i) {
            const AstralErr rr = astral_conv_reset(convs[i], &desc);
            if (rr != ASTRAL_OK) {
                std::fprintf(stderr, "[bench] astral_conv_reset failed: %s (%s)\n",
                             astral_error_string(rr),
                             astral_last_error());
                return false;
            }
            const AstralSpanU8 use_prompt =
                (prompt_heavy_slots > 0 && i < prompt_heavy_slots) ? prompt_heavy_span : prompt_span;
            const AstralErr fr = astral_conv_feed(convs[i], use_prompt, 1);
            if (fr != ASTRAL_OK) {
                std::fprintf(stderr, "[bench] astral_conv_feed failed: %s (%s)\n",
                             astral_error_string(fr),
                             astral_last_error());
                return false;
            }
        }

        const uint64_t t0 = ticks_now();
        const uint64_t n0 = ns_now();

        for (uint32_t i = 0; i < slots; ++i) {
            const AstralErr dr = astral_conv_decode(convs[i]);
            if (dr != ASTRAL_OK) {
                std::fprintf(stderr, "[bench] astral_conv_decode failed: %s (%s)\n",
                             astral_error_string(dr),
                             astral_last_error());
                return false;
            }
        }

        // Poll completion to extract a basic per-slot completion-time distribution without
        // serializing completion via per-slot blocking waits.
        std::vector<uint64_t> done_ns(slots, 0);
        uint32_t remaining = slots;
        while (remaining > 0) {
            for (uint32_t i = 0; i < slots; ++i) {
                if (done_ns[i] != 0) continue;
                AstralSessionState st = ASTRAL_SESSION_IDLE;
                const AstralErr sr = astral_conv_state(convs[i], &st);
                if (sr != ASTRAL_OK) {
                    std::fprintf(stderr, "[bench] astral_conv_state failed: %s (%s)\n",
                                 astral_error_string(sr),
                                 astral_last_error());
                    return false;
                }
                if (st == ASTRAL_SESSION_COMPLETED || st == ASTRAL_SESSION_CANCELED || st == ASTRAL_SESSION_FAILED) {
                    done_ns[i] = ns_now();
                    --remaining;
                }
            }
        }

        for (uint32_t i = 0; i < slots; ++i) {
            const AstralErr wr = astral_conv_wait(convs[i], 60000);
            if (wr != ASTRAL_OK) {
                std::fprintf(stderr, "[bench] astral_conv_wait failed: %s (%s)\n",
                             astral_error_string(wr),
                             astral_last_error());
                return false;
            }
        }

        const uint64_t t1 = ticks_now();
        const uint64_t n1 = ns_now();

        if (measure) {
            r.total_ticks = t1 - t0;
            r.total_ns = n1 - n0;
            r.tokens = static_cast<uint64_t>(slots) * static_cast<uint64_t>(max_tokens);

            std::vector<uint64_t> deltas_ns;
            deltas_ns.reserve(slots);
            for (uint32_t i = 0; i < slots; ++i) {
                const uint64_t dt = done_ns[i] > n0 ? (done_ns[i] - n0) : 0;
                deltas_ns.push_back(dt);
            }
            std::sort(deltas_ns.begin(), deltas_ns.end());
            auto q = [&](double quant) -> uint64_t {
                if (deltas_ns.empty()) return 0;
                const double idx = quant * static_cast<double>(deltas_ns.size() - 1);
                const size_t i0 = static_cast<size_t>(idx);
                return deltas_ns[i0];
            };
            const uint64_t p50 = q(0.50);
            const uint64_t p95 = q(0.95);
            if (out_p50_ms) *out_p50_ms = static_cast<double>(p50) / 1e6;
            if (out_p95_ms) *out_p95_ms = static_cast<double>(p95) / 1e6;
        }
        return true;
    };

    // Warmup (allows provider to allocate internal caches before measurement).
    if (warmup_tokens > 0) {
        (void)run_round(warmup_tokens, 1, false, nullptr, nullptr);
    }

    double p50_ms = 0.0;
    double p95_ms = 0.0;
    (void)run_round(measure_tokens, 1234, true, &p50_ms, &p95_ms);
    if (p50_ms > 0.0 || p95_ms > 0.0) {
        std::printf("[bench] Continuous batching completion: p50=%.2f ms, p95=%.2f ms\n", p50_ms, p95_ms);
        if (prompt_heavy_slots > 0 && prompt_repeat > 1) {
            std::printf("[bench] Mixed prompts: heavy_slots=%u, prompt_repeat=%u\n", prompt_heavy_slots, prompt_repeat);
        }
    }

    for (uint32_t i = 0; i < slots; ++i) {
        astral_conv_destroy(convs[i]);
    }

    return r;
}

} // namespace

static bool feed_default_prompt(AstralHandle session) {
    const char* prompt = "Once upon a time";
    AstralSpanU8 prompt_span{};
    prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
    prompt_span.len = static_cast<uint32_t>(std::strlen(prompt));
    const AstralErr err = astral_session_feed(session, prompt_span, 1);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_session_feed failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        return false;
    }
    return true;
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

void bench_inference_print(uint32_t warmup_tokens, uint32_t measure_tokens) {
    const char* model_path = find_bench_model_path();
    if (model_path == nullptr) {
        std::printf("[bench] Inference: SKIP (set ASTRAL_BENCH_MODEL or place a model under tests/models/)\n");
        return;
    }

    const uint32_t slots = parse_u32_env("ASTRAL_BENCH_SLOTS", 1);
    const uint32_t max_batch_tokens = parse_u32_env("ASTRAL_BENCH_MAX_BATCH_TOKENS", 64);

    AstralInit cfg{};
    cfg.reserve_bytes = 2ull << 30; // 2GB address space (virtual reserve).
    cfg.thread_count = bench_runtime_threads();
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

    // Let the backend pick a safe/default context size for the model, unless we
    // explicitly run multi-slot continuous batching. In that mode, we want a
    // per-slot ctx >= max_tokens + prompt headroom.
    model_desc.n_ctx = 0;
    if (slots > 1) {
        const uint32_t env_ctx = parse_u32_env("ASTRAL_BENCH_CTX", 0);
        if (env_ctx != 0) {
            model_desc.n_ctx = env_ctx;
        } else {
            const uint32_t per_slot = measure_tokens + 128; // prompt + BOS + safety
            const uint64_t total64 = static_cast<uint64_t>(per_slot) * static_cast<uint64_t>(slots);
            const uint64_t rounded64 = ((total64 + slots - 1u) / slots) * slots;
            const uint64_t clamped64 = rounded64 > 0xFFFFFFFFu ? 0xFFFFFFFFu : rounded64;
            model_desc.n_ctx = static_cast<uint32_t>(clamped64);
        }
    }

    model_desc.n_batch = 0;
    if (slots > 1) {
        model_desc.n_batch = max_batch_tokens;
    }
    model_desc.n_threads = bench_model_threads();
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

    if (slots > 1) {
        InferenceBenchResult r =
            run_continuous_batching_decode(model, slots, max_batch_tokens, warmup_tokens, measure_tokens);

        if (r.tokens == 0 || r.total_ns == 0) {
            std::printf("[bench] Inference: FAILED (no tokens)\n");
            astral_model_release(model);
            astral_shutdown();
            return;
        }

        const double total_s = static_cast<double>(r.total_ns) * 1e-9;
        const double tok_s = total_s > 0.0 ? static_cast<double>(r.tokens) / total_s : 0.0;

        std::printf("\nInference benchmark (ticks: %s)\n", clock_info().name);
        std::printf("Model: %s\n", model_path);
        std::printf("Mode:  continuous batching\n");
        std::printf("Slots: %u, max_batch_tokens: %u\n", slots, max_batch_tokens);
        std::printf("Total: %.2f ms  (%llu ticks)\n",
                    static_cast<double>(r.total_ns) / 1e6,
                    static_cast<unsigned long long>(r.total_ticks));
        std::printf("Tok/s: %.2f  (aggregate across slots)\n\n", tok_s);

        astral_model_release(model);
        astral_shutdown();
        return;
    }

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = warmup_tokens > 0 ? warmup_tokens : measure_tokens;
    session_desc.temperature = 0.8f;
    session_desc.top_k = 40;
    session_desc.top_p = 0.95f;
    const bool want_stream = parse_bool_env("ASTRAL_BENCH_INFER_STREAM", true);
    session_desc.stream_enabled = want_stream ? 1 : 0;
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
        if (want_stream) {
            (void)run_streamed_decode(session);
        } else {
            if (!feed_default_prompt(session)) {
                astral_session_destroy(session);
                astral_model_release(model);
                astral_shutdown();
                return;
            }
            (void)astral_session_decode(session);
            (void)astral_session_wait(session, 60000);
        }
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

    InferenceBenchResult r{};
    r.name = want_stream ? "CPU inference (streamed)" : "CPU inference (no-stream)";
    if (want_stream) {
        r = run_streamed_decode(session);
    } else {
        if (!feed_default_prompt(session)) {
            astral_session_destroy(session);
            astral_model_release(model);
            astral_shutdown();
            return;
        }
        const uint64_t t0 = ticks_now();
        const uint64_t n0 = ns_now();

        AstralErr dec_err = astral_session_decode(session);
        if (dec_err != ASTRAL_OK) {
            std::fprintf(stderr, "[bench] astral_session_decode failed: %s (%s)\n",
                         astral_error_string(dec_err),
                         astral_last_error());
        } else {
            (void)astral_session_wait(session, 60000);
        }

        const uint64_t t1 = ticks_now();
        const uint64_t n1 = ns_now();
        r.total_ticks = t1 - t0;
        r.total_ns = n1 - n0;

        AstralStats stats{};
        const AstralErr st_err = astral_session_stats(session, &stats);
        if (st_err == ASTRAL_OK) {
            r.tokens = measure_tokens;
            r.ttft_ns = static_cast<uint64_t>(stats.t_first_token_ms * 1e6);
            r.ttft_ticks = 0;
        } else {
            r.tokens = 0;
        }
    }

    if (r.tokens == 0 || r.total_ns == 0) {
        std::printf("[bench] Inference: FAILED (no tokens)\n");
        astral_session_destroy(session);
        astral_model_release(model);
        astral_shutdown();
        return;
    }

    AstralStats stats{};
    const AstralErr stats_err = astral_session_stats(session, &stats);
    if (stats_err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_session_stats failed: %s (%s)\n",
                     astral_error_string(stats_err),
                     astral_last_error());
    }

    const double total_s = static_cast<double>(r.total_ns) * 1e-9;
    const double ticks_per_read = r.tokens > 0 ? static_cast<double>(r.total_ticks) / static_cast<double>(r.tokens)
                                               : 0.0;

    std::printf("\nInference benchmark (ticks: %s)\n", clock_info().name);
    std::printf("Model: %s\n", model_path);
    std::printf("Runtime threads (requested): %u\n", cfg.thread_count);
    std::printf("Model threads (requested):   %u\n", model_desc.n_threads);

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
    if (stats_err == ASTRAL_OK) {
        std::printf("Tok/s: %.2f  (ttft: %.2f ms)\n", stats.tok_per_s, stats.t_first_token_ms);
    } else {
        std::printf("Tok/s: n/a\n");
    }
    std::printf("Stream: %llu reads, %llu bytes  (%.2f ticks/read)\n\n",
                static_cast<unsigned long long>(r.tokens),
                static_cast<unsigned long long>(r.bytes),
                ticks_per_read);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

} // namespace astral::bench
