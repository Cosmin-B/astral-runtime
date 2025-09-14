/**
 * test_cuda_e2e.cpp - End-to-end feature validation for CUDA offload
 *
 * This suite focuses on API surface correctness when running llama.cpp with GPU offload enabled
 * (backend_name="cuda", gpu_layers>0).
 *
 * It always runs CPU checks (when the test model is available) and conditionally runs CUDA checks
 * when built with ASTRAL_ENABLE_CUDA and ASTRAL_TEST_CUDA_E2E=1.
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

static bool env_is_true(const char* key) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return (std::strcmp(v, "1") == 0) || (std::strcmp(v, "true") == 0) || (std::strcmp(v, "TRUE") == 0) ||
           (std::strcmp(v, "on") == 0) || (std::strcmp(v, "ON") == 0) || (std::strcmp(v, "yes") == 0) ||
           (std::strcmp(v, "YES") == 0);
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

static const char* find_test_model_path() {
    const char* env = std::getenv("ASTRAL_TEST_MODEL");
    if (file_exists_min_size(env, 5ULL * 1024ULL * 1024ULL)) {
        return env;
    }

    static const char* paths[] = {
        "tests/models/gpt2.Q2_K.gguf",
        "../tests/models/gpt2.Q2_K.gguf",
        "../../tests/models/gpt2.Q2_K.gguf",
        "../../../tests/models/gpt2.Q2_K.gguf",
        "../../../../tests/models/gpt2.Q2_K.gguf",
    };
    for (const char* p : paths) {
        if (file_exists_min_size(p, 5ULL * 1024ULL * 1024ULL)) {
            return p;
        }
    }
    return nullptr;
}

static AstralSpanU8 span_from_cstr(const char* s) {
    AstralSpanU8 out{};
    out.data = reinterpret_cast<const uint8_t*>(s);
    out.len = s ? static_cast<uint32_t>(std::strlen(s)) : 0u;
    return out;
}

static AstralHandle load_model(const char* backend, const char* model_path, uint32_t gpu_layers, uint8_t embeddings_only) {
    AstralModelDesc desc{};
    desc.backend_name = span_from_cstr(backend);
    desc.model_path = span_from_cstr(model_path);
    desc.gpu_layers = gpu_layers;
    desc.n_ctx = 512;
    desc.n_batch = 128;
    desc.n_threads = 2;
    desc.embeddings_only = embeddings_only;

    AstralHandle model = 0;
    ASSERT_EQ(astral_model_load(&desc, &model), ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));
    return model;
}

static std::vector<AstralTokenMeta> drain_meta(AstralHandle session, uint32_t max_events) {
    std::vector<AstralTokenMeta> out;
    out.reserve(max_events);

    for (;;) {
        AstralTokenMeta evs[8]{};
        const int32_t n = astral_stream_read_meta(session, evs, 8, 250);
        if (n == 0) {
            break;
        }
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        ASSERT_GT(n, 0);
        for (int32_t i = 0; i < n; ++i) {
            out.push_back(evs[i]);
            if (out.size() >= max_events) {
                return out;
            }
        }
    }

    return out;
}

static std::string drain_utf8(AstralHandle session, uint32_t max_bytes) {
    std::string out;
    out.reserve(max_bytes);

    uint8_t buf[256];
    for (;;) {
        AstralMutSpanU8 span{};
        span.data = buf;
        span.len = sizeof(buf);
        const int32_t n = astral_stream_read(session, span, 250);
        if (n == 0) {
            break;
        }
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        ASSERT_GT(n, 0);
        out.append(reinterpret_cast<const char*>(buf), reinterpret_cast<const char*>(buf) + n);
        if (out.size() >= max_bytes) {
            break;
        }
    }

    return out;
}

static void assert_logprobs_shape_consistent(const std::vector<AstralTokenMeta>& events, uint32_t want_n) {
    ASSERT_FALSE(events.empty());

    for (size_t i = 0; i < events.size(); ++i) {
        const AstralTokenMeta& ev = events[i];
        ASSERT_GT(ev.top_n, 0u);
        ASSERT_LE(ev.top_n, want_n);

        for (uint32_t j = 0; j < ev.top_n; ++j) {
            const uint32_t id = ev.top_token_ids[j];
            // Unique ids in the top list.
            for (uint32_t k = 0; k < j; ++k) {
                ASSERT_NE(id, ev.top_token_ids[k]);
            }

            const float lp = ev.top_logprobs[j];
            ASSERT_TRUE(std::isfinite(lp));
            if (j > 0) {
                // Non-increasing (by probability) logprobs.
                ASSERT_TRUE(ev.top_logprobs[j - 1] >= lp - 1e-4f);
            }
        }

        ASSERT_TRUE(std::isfinite(ev.logprob));

        // If the chosen token is within the reported top-N list, its logprob should match.
        for (uint32_t j = 0; j < ev.top_n; ++j) {
            if (ev.top_token_ids[j] == ev.token_id) {
                ASSERT_TRUE(std::fabs(ev.logprob - ev.top_logprobs[j]) <= 1e-3f ||
                            (ev.logprob == 0.0f && ev.top_logprobs[j] == 0.0f));
                break;
            }
        }

        // Chosen token must not have higher probability than the top-1 entry.
        ASSERT_TRUE(ev.logprob <= ev.top_logprobs[0] + 1e-3f);
    }
}

static std::vector<uint32_t> to_token_ids(const std::vector<AstralTokenMeta>& events) {
    std::vector<uint32_t> out;
    out.reserve(events.size());
    for (const auto& e : events) {
        out.push_back(e.token_id);
    }
    return out;
}

static std::vector<uint32_t> run_tokens_for_kv_ref(
    AstralHandle model,
    const char* prompt,
    uint32_t max_tokens,
    float temperature,
    uint32_t top_k,
    float top_p,
    uint32_t seed
) {
    AstralSessionDesc sdesc{};
    sdesc.model = model;
    sdesc.max_tokens = max_tokens;
    sdesc.temperature = temperature;
    sdesc.top_k = top_k;
    sdesc.top_p = top_p;
    sdesc.stream_enabled = 1;
    sdesc.seed = seed;

    AstralHandle session = 0;
    ASSERT_EQ(astral_session_create(&sdesc, &session), ASTRAL_OK);
    ASSERT_EQ(astral_session_set_logprobs(session, 8), ASTRAL_OK);

    ASSERT_EQ(astral_session_feed(session, span_from_cstr(prompt), 1), ASTRAL_OK);
    ASSERT_EQ(astral_session_decode(session), ASTRAL_OK);
    ASSERT_EQ(astral_session_wait(session, 10000), ASTRAL_OK);

    const std::vector<AstralTokenMeta> evs = drain_meta(session, max_tokens);
    astral_session_destroy(session);
    return to_token_ids(evs);
}

static std::vector<uint32_t> run_tokens_from_loaded_state(
    AstralHandle model,
    AstralSpanU8 state_bytes,
    uint32_t max_tokens,
    float temperature,
    uint32_t top_k,
    float top_p,
    uint32_t seed
) {
    AstralSessionDesc sdesc{};
    sdesc.model = model;
    sdesc.max_tokens = max_tokens;
    sdesc.temperature = temperature;
    sdesc.top_k = top_k;
    sdesc.top_p = top_p;
    sdesc.stream_enabled = 1;
    sdesc.seed = seed;

    AstralHandle session = 0;
    ASSERT_EQ(astral_session_create(&sdesc, &session), ASTRAL_OK);
    ASSERT_EQ(astral_session_set_logprobs(session, 8), ASTRAL_OK);

    // Load backend+sampler state, then transition to FeedingPrompt without adding special tokens.
    ASSERT_EQ(astral_session_state_load(session, state_bytes), ASTRAL_OK);

    AstralSpanU8 empty{};
    ASSERT_EQ(astral_session_feed(session, empty, 0), ASTRAL_OK);

    ASSERT_EQ(astral_session_decode(session), ASTRAL_OK);
    ASSERT_EQ(astral_session_wait(session, 10000), ASTRAL_OK);

    const std::vector<AstralTokenMeta> evs = drain_meta(session, max_tokens);
    astral_session_destroy(session);
    return to_token_ids(evs);
}

static void run_backend_e2e(const char* backend, uint32_t gpu_layers, const char* model_path) {
    // -----------------------------
    // Logprobs shape / consistency
    // -----------------------------
    {
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/0);

        AstralSessionDesc sdesc{};
        sdesc.model = model;
        sdesc.max_tokens = 4;
        sdesc.temperature = 1.0f;
        sdesc.top_k = 32;
        sdesc.top_p = 1.0f;
        sdesc.stream_enabled = 1;
        sdesc.seed = 123;

        AstralHandle session = 0;
        ASSERT_EQ(astral_session_create(&sdesc, &session), ASTRAL_OK);
        ASSERT_EQ(astral_session_set_logprobs(session, 8), ASTRAL_OK);

        ASSERT_EQ(astral_session_feed(session, span_from_cstr("Hello"), 1), ASTRAL_OK);
        ASSERT_EQ(astral_session_decode(session), ASTRAL_OK);
        ASSERT_EQ(astral_session_wait(session, 10000), ASTRAL_OK);

        const auto meta = drain_meta(session, 16);
        assert_logprobs_shape_consistent(meta, 8);

        astral_session_destroy(session);
        astral_model_release(model);
    }

    // -----------------------------
    // Grammar (GBNF) end-to-end
    // -----------------------------
    {
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/0);

        AstralSessionDesc sdesc{};
        sdesc.model = model;
        sdesc.max_tokens = 8;
        sdesc.temperature = 0.0f; // greedy
        sdesc.top_k = 0;
        sdesc.top_p = 1.0f;
        sdesc.stream_enabled = 1;
        sdesc.seed = 1;

        AstralHandle session = 0;
        ASSERT_EQ(astral_session_create(&sdesc, &session), ASTRAL_OK);

        // Force output to be repetitions of " a" (space + 'a').
        const char* gbnf =
            "root ::= piece root | piece\n"
            "piece ::= \" a\"\n";
        ASSERT_EQ(astral_session_set_grammar_gbnf(session, span_from_cstr(gbnf), AstralSpanU8{}), ASTRAL_OK);

        // Use a non-empty prompt to ensure logits are available for the first sampling step.
        ASSERT_EQ(astral_session_feed(session, span_from_cstr("Hello"), 1), ASTRAL_OK);
        ASSERT_EQ(astral_session_decode(session), ASTRAL_OK);
        {
            const AstralErr w = astral_session_wait(session, 10000);
            if (w != ASTRAL_OK) {
                char msg[512];
                const char* le = astral_last_error();
                snprintf(msg, sizeof(msg), "session_wait failed: %d last_error=%s", (int)w, le ? le : "(null)");
                ::astral::testing::test_fail_msg(__FILE__, __LINE__, msg);
            }
        }

        const std::string out = drain_utf8(session, 256);
        ASSERT_FALSE(out.empty());
        for (char c : out) {
            ASSERT_TRUE(c == ' ' || c == 'a');
        }
        // Expect a leading space in the constrained output.
        ASSERT_EQ(out[0], ' ');

        astral_session_destroy(session);
        astral_model_release(model);
    }

    // -----------------------------
    // KV save/load continuation
    // -----------------------------
    {
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/0);

        constexpr float kTemp = 1.0f;
        constexpr uint32_t kTopK = 32;
        constexpr float kTopP = 1.0f;
        constexpr uint32_t kSeed = 777;

        // Reference run: 8 tokens in one go.
        const std::vector<uint32_t> ref = run_tokens_for_kv_ref(model, "Hello", 8, kTemp, kTopK, kTopP, kSeed);
        ASSERT_GE(ref.size(), 8u);

        // First stage: 4 tokens, then save state.
        AstralSessionDesc sdesc{};
        sdesc.model = model;
        sdesc.max_tokens = 4;
        sdesc.temperature = kTemp;
        sdesc.top_k = kTopK;
        sdesc.top_p = kTopP;
        sdesc.stream_enabled = 1;
        sdesc.seed = kSeed;

        AstralHandle session = 0;
        ASSERT_EQ(astral_session_create(&sdesc, &session), ASTRAL_OK);
        ASSERT_EQ(astral_session_set_logprobs(session, 8), ASTRAL_OK);

        ASSERT_EQ(astral_session_feed(session, span_from_cstr("Hello"), 1), ASTRAL_OK);
        ASSERT_EQ(astral_session_decode(session), ASTRAL_OK);
        ASSERT_EQ(astral_session_wait(session, 10000), ASTRAL_OK);
        (void)drain_meta(session, 16);

        uint64_t bytes = 0;
        ASSERT_EQ(astral_session_state_size(session, &bytes), ASTRAL_OK);
        ASSERT_GT(bytes, 16u);

        std::vector<uint8_t> state(static_cast<size_t>(bytes));
        AstralMutSpanU8 out{};
        out.data = state.data();
        out.len = static_cast<uint32_t>(state.size());
        uint64_t written = 0;
        ASSERT_EQ(astral_session_state_save(session, out, &written), ASTRAL_OK);
        ASSERT_GT(written, 0u);

        astral_session_destroy(session);

        AstralSpanU8 state_in{};
        state_in.data = state.data();
        state_in.len = static_cast<uint32_t>(written);

        // Second stage: load state and generate 4 more tokens; should match ref tokens [4..7].
        const std::vector<uint32_t> cont = run_tokens_from_loaded_state(model, state_in, 4, kTemp, kTopK, kTopP, kSeed);
        ASSERT_GE(cont.size(), 4u);
        for (size_t i = 0; i < 4; ++i) {
            ASSERT_EQ(cont[i], ref[i + 4]);
        }

        astral_model_release(model);
    }

    // -----------------------------
    // Embeddings end-to-end
    // -----------------------------
    {
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/1);
        uint32_t dim = 0;
        ASSERT_EQ(astral_model_embedding_dim(model, &dim), ASTRAL_OK);
        ASSERT_GT(dim, 0u);
        ASSERT_LT(dim, 8192u);

        AstralHandle emb = 0;
        ASSERT_EQ(astral_embed_create(model, &emb), ASTRAL_OK);
        ASSERT_TRUE(astral_handle_valid(emb));

        uint64_t ticket = 0;
        ASSERT_EQ(astral_embed_enqueue(emb, span_from_cstr("hi"), &ticket), ASTRAL_OK);
        ASSERT_NE(ticket, 0ULL);

        static float vec[8192];
        AstralMutSpanU8 vec_out{};
        vec_out.data = reinterpret_cast<uint8_t*>(vec);
        vec_out.len = static_cast<uint32_t>(sizeof(vec));

        ASSERT_EQ(astral_embed_collect(emb, ticket, vec_out), ASTRAL_OK);

        double sum_abs = 0.0;
        for (uint32_t i = 0; i < dim; ++i) {
            const float v = vec[i];
            sum_abs += (v >= 0.0f) ? v : -v;
        }
        ASSERT_GT(sum_abs, 0.0);

        astral_embed_destroy(emb);
        astral_model_release(model);
    }
}

} // namespace

TEST(cuda_e2e_features) {
    const char* model_path = find_test_model_path();
    if (model_path == nullptr) {
        return;
    }

    AstralInit cfg{};
    cfg.reserve_bytes = 2ULL << 30;
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    // Always validate CPU path on the local machine.
    run_backend_e2e("cpu", 0, model_path);

#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
    if (env_is_true("ASTRAL_TEST_CUDA_E2E")) {
        run_backend_e2e("cuda", 8, model_path);
    }
#endif

    astral_shutdown();
}
