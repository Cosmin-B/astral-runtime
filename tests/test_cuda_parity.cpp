/**
 * test_cuda_parity.cpp - CUDA backend parity scaffolding
 *
 * Goals:
 * - Provide a deterministic “parity harness” that can be run on a CUDA-capable machine.
 * - Keep CPU-only builds unaffected (no CUDA toolkit required; tests still compile/run).
 *
 * Notes:
 * - This test does NOT require a GPU unless the optional inference parity section is enabled.
 * - Enable the inference parity run by setting ASTRAL_TEST_CUDA_PARITY_INFER=1.
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

static AstralSpanU8 span_from_cstr(const char* s) {
    AstralSpanU8 out{};
    out.data = reinterpret_cast<const uint8_t*>(s);
    out.len = s ? static_cast<uint32_t>(std::strlen(s)) : 0u;
    return out;
}

static bool env_is_true(const char* key) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return (std::strcmp(v, "1") == 0) || (std::strcmp(v, "true") == 0) || (std::strcmp(v, "TRUE") == 0) ||
           (std::strcmp(v, "on") == 0) || (std::strcmp(v, "ON") == 0) || (std::strcmp(v, "yes") == 0) ||
           (std::strcmp(v, "YES") == 0);
}

static std::vector<AstralTokenMeta> read_meta_events(AstralHandle session, uint32_t max_events) {
    std::vector<AstralTokenMeta> out;
    out.reserve(max_events);

    AstralTokenMeta evs[8]{};
    for (;;) {
        const int32_t n = astral_stream_read_meta(session, evs, 8, 100 /*ms*/);
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

static AstralHandle load_model_by_backend(const char* backend_name, const char* model_path, uint32_t gpu_layers) {
    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    if (backend_name != nullptr) {
        desc.backend_name = span_from_cstr(backend_name);
    }
    if (model_path != nullptr) {
        desc.model_path = span_from_cstr(model_path);
    }
    desc.gpu_layers = gpu_layers;
    desc.n_ctx = 256;
    desc.n_batch = 128;
    desc.n_threads = 2;
    desc.embeddings_only = 0;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));
    return model;
}

static std::vector<AstralTokenMeta> run_greedy_meta(
    const char* backend_name,
    const char* model_path,
    uint32_t gpu_layers,
    const char* prompt
) {
    const AstralHandle model = load_model_by_backend(backend_name, model_path, gpu_layers);

    AstralSessionDesc sdesc{};
    sdesc.model = model;
    sdesc.max_tokens = 32;
    sdesc.temperature = 0.0f;
    sdesc.top_k = 0;
    sdesc.top_p = 1.0f;
    sdesc.stream_enabled = 1;
    sdesc.seed = 1;

    AstralHandle session = 0;
    ASSERT_EQ(astral_session_create(&sdesc, &session), ASTRAL_OK);
    // Capture enough distribution mass to compare “near parity” even when top-1 differs across CPU/CUDA.
    ASSERT_EQ(astral_session_set_logprobs(session, 8), ASTRAL_OK);

    const AstralSpanU8 chunk = span_from_cstr(prompt);
    ASSERT_EQ(astral_session_feed(session, chunk, 1), ASTRAL_OK);

    ASSERT_EQ(astral_session_decode(session), ASTRAL_OK);
    const std::vector<AstralTokenMeta> out = read_meta_events(session, 32);
    ASSERT_EQ(astral_session_wait(session, 5000), ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    return out;
}

static bool token_in_top(const AstralTokenMeta& ev, uint32_t token, uint32_t* out_rank) {
    const uint32_t n = (ev.top_n > ASTRAL_LOGPROBS_MAX) ? ASTRAL_LOGPROBS_MAX : ev.top_n;
    for (uint32_t i = 0; i < n; ++i) {
        if (ev.top_token_ids[i] == token) {
            if (out_rank != nullptr) {
                *out_rank = i;
            }
            return true;
        }
    }
    return false;
}

} // namespace

TEST(cuda_parity_scaffold) {
    AstralInit cfg{};
    cfg.reserve_bytes = 512ull * 1024ull * 1024ull;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    // Surface: if CUDA is not built in, trying to force the backend should report ASTRAL_E_BACKEND.
    // If CUDA is built in, the backend exists; missing model_path then yields ASTRAL_E_INVALID.
    {
        AstralModelDesc model_desc{};
        model_desc.size = sizeof(AstralModelDesc);
        model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
        const char* backend = "cuda";
        model_desc.backend_name = span_from_cstr(backend);
        model_desc.gpu_layers = 1;
        model_desc.n_ctx = 256;
        model_desc.n_batch = 128;

        AstralHandle model = 0;
        const AstralErr err = astral_model_load(&model_desc, &model);
#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
        ASSERT_EQ(err, ASTRAL_E_INVALID);
#else
        ASSERT_EQ(err, ASTRAL_E_BACKEND);
#endif
    }

#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
    // Optional: deterministic inference parity check. Intended to run on a CUDA machine.
    // This is guarded because it depends on having a real GGUF model available and can be slow.
    if (env_is_true("ASTRAL_TEST_CUDA_PARITY_INFER")) {
        const char* model_path = ASTRAL_TEST_SOURCE_DIR "/tests/models/gpt2.Q2_K.gguf";

        const std::vector<AstralTokenMeta> out_cpu = run_greedy_meta("cpu", model_path, 0, "Hello");
        const std::vector<AstralTokenMeta> out_cuda = run_greedy_meta("cuda", model_path, 8, "Hello");

        ASSERT_FALSE(out_cpu.empty());
        ASSERT_FALSE(out_cuda.empty());

        // Strict mode is opt-in: exact top-1 parity across CPU and CUDA is not always guaranteed
        // (depending on GPU kernels / math flags / driver/toolkit combos).
        //
        // In strict mode, enforce that each backend’s chosen token is at least “plausible”
        // under the other backend’s distribution (within the captured top-N).
        if (env_is_true("ASTRAL_TEST_CUDA_PARITY_STRICT")) {
            const size_t n = std::min(out_cpu.size(), out_cuda.size());
            ASSERT_GE(n, 1u);

            // If you want exact token-id parity (useful for debugging), set this too.
            const bool exact = env_is_true("ASTRAL_TEST_CUDA_PARITY_EXACT");
            if (exact) {
                ASSERT_EQ(out_cpu[0].token_id, out_cuda[0].token_id);
            } else {
                // Compare only the first token: once tokens diverge, later steps are conditioning on different
                // contexts and the comparison is no longer meaningful.
                const uint32_t cpu_tok = out_cpu[0].token_id;
                const uint32_t cuda_tok = out_cuda[0].token_id;

                uint32_t cpu_rank_in_cuda = 0xFFFFFFFFu;
                uint32_t cuda_rank_in_cpu = 0xFFFFFFFFu;

                const bool cpu_ok = token_in_top(out_cuda[0], cpu_tok, &cpu_rank_in_cuda);
                const bool cuda_ok = token_in_top(out_cpu[0], cuda_tok, &cuda_rank_in_cpu);

                ASSERT_TRUE(cpu_ok);
                ASSERT_TRUE(cuda_ok);

                // Heuristic: each chosen token should be within the other backend’s top-8.
                ASSERT_LE(cpu_rank_in_cuda, 7u);
                ASSERT_LE(cuda_rank_in_cpu, 7u);
            }
        }
    }
#endif

    astral_shutdown();
}
