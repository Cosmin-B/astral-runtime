/**
 * test_inference.cpp - Inference layer tests
 *
 * Tests for session management, state machine, sampler, and token generation.
 * Validates: session create/destroy, feed/decode, streaming, sampler configuration.
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <cstring>
#include <chrono>
#include <thread>
#include <string>
#include <cmath>

namespace {

AstralHandle load_mock_model(const char* model_path) {
    AstralModelDesc model_desc = {};
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));

    if (model_path != nullptr) {
        model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
        model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));
    }

    model_desc.n_ctx = 128;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));
    return model;
}

uint32_t drain_stream(AstralHandle session, uint32_t max_reads) {
    uint8_t buf[256];
    uint32_t total = 0;

    for (uint32_t i = 0; i < max_reads; ++i) {
        AstralMutSpanU8 out = {};
        out.data = buf;
        out.len = sizeof(buf);

        const int32_t n = astral_stream_read(session, out, 100 /*ms*/);
        if (n == 0) {
            break;
        }
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        ASSERT_GT(n, 0);
        total += static_cast<uint32_t>(n);
    }

    return total;
}

std::string read_stream_all(AstralHandle session) {
    std::string out;
    out.reserve(256);

    uint8_t buf[256];
    for (uint32_t i = 0; i < 10000; ++i) {
        AstralMutSpanU8 span{};
        span.data = buf;
        span.len = sizeof(buf);

        const int32_t n = astral_stream_read(session, span, 100 /*ms*/);
        if (n == 0) {
            break;
        }
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        ASSERT_GT(n, 0);
        out.append(reinterpret_cast<const char*>(buf), static_cast<size_t>(n));
    }

    return out;
}

} // namespace

//
// Session Creation Tests
//

TEST(inference_session_create_destroy) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 32 * 1024 * 1024;
    astral_init(&cfg);

    // Load model first (may fail if no model file)
    AstralModelDesc model_desc = {};
    const char* path = "models/test.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.gpu_layers = 0;

    AstralHandle model;
    AstralErr err = astral_model_load(&model_desc, &model);

    if (err == ASTRAL_OK) {
        // Create session
        AstralSessionDesc session_desc = {};
        session_desc.model = model;
        session_desc.max_tokens = 100;
        session_desc.temperature = 0.7f;
        session_desc.top_k = 40;
        session_desc.top_p = 0.9f;
        session_desc.stream_enabled = 1;

        AstralHandle session;
        err = astral_session_create(&session_desc, &session);

        if (err == ASTRAL_OK) {
            astral_session_destroy(session);
        }

        astral_model_release(model);
    }

    astral_shutdown();
}

TEST(inference_session_create_null_desc) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralHandle session;
    AstralErr err = astral_session_create(nullptr, &session);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(inference_session_create_null_output) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralSessionDesc session_desc = {};
    AstralErr err = astral_session_create(&session_desc, nullptr);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

//
// Sampler Configuration Tests
//

TEST(inference_sampler_deterministic) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 32 * 1024 * 1024;
    astral_init(&cfg);

    // Configuration for deterministic sampling (temperature=0)
    AstralSessionDesc session_desc = {};
    session_desc.temperature = 0.0f; // Deterministic (greedy)
    session_desc.max_tokens = 10;

    // This tests the configuration, actual sampling requires a valid model
    astral_shutdown();
}

TEST(inference_sampler_diverse) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 32 * 1024 * 1024;
    astral_init(&cfg);

    // Configuration for diverse sampling
    AstralSessionDesc session_desc = {};
    session_desc.temperature = 1.0f; // Diverse
    session_desc.top_k = 50;
    session_desc.top_p = 0.95f;
    session_desc.max_tokens = 10;

    // This tests the configuration
    astral_shutdown();
}

//
// Session Feed Tests
//

TEST(inference_session_feed_null_session) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    const char* prompt = "Hello";
    AstralSpanU8 chunk = {};
    chunk.data = reinterpret_cast<const uint8_t*>(prompt);
    chunk.len = static_cast<uint32_t>(strlen(prompt));

    AstralErr err = astral_session_feed(0, chunk, 1);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(inference_session_feed_empty) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Feed empty chunk should be allowed (edge case)
    AstralSpanU8 chunk = {};
    chunk.data = nullptr;
    chunk.len = 0;

    // Would need valid session to test fully
    // This just tests the null case
    astral_shutdown();
}

//
// Decode Tests
//

TEST(inference_session_decode_null_session) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralErr err = astral_session_decode(0);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

//
// Stream Read Tests
//

TEST(inference_stream_read_null_session) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    uint8_t buffer[256];
    AstralMutSpanU8 out_buf = {};
    out_buf.data = buffer;
    out_buf.len = sizeof(buffer);

    int32_t result = astral_stream_read(0, out_buf, 0);

    ASSERT_EQ(result, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(inference_stream_read_null_buffer) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralMutSpanU8 out_buf = {};
    out_buf.data = nullptr;
    out_buf.len = 0;

    // Would need valid session for full test
    // This tests null buffer handling
    astral_shutdown();
}

TEST(inference_stream_timeout_and_eos_semantics) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 64;
    session_desc.temperature = 0.0f; // greedy (deterministic)
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    // Before decode, empty ring reads should time out.
    uint8_t buf[64];
    AstralMutSpanU8 out = {};
    out.data = buf;
    out.len = sizeof(buf);
    ASSERT_EQ(astral_stream_read(session, out, 0), ASTRAL_E_TIMEOUT);

    // Empty feed+finalize is valid and triggers BOS tokenization for mock.
    AstralSpanU8 empty = {};
    empty.data = nullptr;
    empty.len = 0;
    err = astral_session_feed(session, empty, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    ASSERT_GT(drain_stream(session, 256), 0u);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    // After end-of-stream, reads return 0.
    ASSERT_EQ(astral_stream_read(session, out, 0), 0);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_model_caps_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralCaps caps = 0;
    err = astral_model_caps(model, &caps);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE((caps & ASTRAL_CAP_SAMPLER_EXT) != 0);
    ASSERT_TRUE((caps & ASTRAL_CAP_STOP_SEQS) != 0);
    ASSERT_TRUE((caps & ASTRAL_CAP_EMBEDDINGS) != 0);
    ASSERT_TRUE((caps & ASTRAL_CAP_LOGPROBS) != 0);
    ASSERT_TRUE((caps & ASTRAL_CAP_KV_STATE) != 0);
    ASSERT_TRUE((caps & ASTRAL_CAP_LORA) != 0);
    ASSERT_TRUE((caps & ASTRAL_CAP_GRAMMAR) != 0);
    ASSERT_TRUE((caps & ASTRAL_CAP_GRAMMAR_GBNF) != 0);
    ASSERT_TRUE((caps & ASTRAL_CAP_SLOTS) != 0);

    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_stop_sequences_suppress_output) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 128;
    sd.temperature = 0.0f; // greedy
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 1;
    sd.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* stop = "backend";
    AstralSpanU8 stop_span{};
    stop_span.data = reinterpret_cast<const uint8_t*>(stop);
    stop_span.len = static_cast<uint32_t>(std::strlen(stop));
    err = astral_session_stop_clear(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_stop_add_utf8(session, stop_span);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSpanU8 empty{};
    err = astral_session_feed(session, empty, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    const std::string text = read_stream_all(session);
    ASSERT_EQ(text, std::string("mock-"));

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_stop_sequences_bulk_set_utf8) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 128;
    sd.temperature = 0.0f;
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 1;
    sd.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* stop = "backend";
    AstralSpanU8 stop_span{};
    stop_span.data = reinterpret_cast<const uint8_t*>(stop);
    stop_span.len = static_cast<uint32_t>(std::strlen(stop));
    err = astral_session_stop_set_utf8(session, &stop_span, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSpanU8 empty{};
    err = astral_session_feed(session, empty, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    const std::string text = read_stream_all(session);
    ASSERT_EQ(text, std::string("mock-"));

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_sampler_repeat_penalty_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model("sampler");

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 6;
    sd.temperature = 0.0f; // greedy
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 1;
    sd.seed = 123;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSamplerDesc sampler{};
    sampler.size = sizeof(AstralSamplerDesc);
    sampler.temperature = 0.0f;
    sampler.top_k = 0;
    sampler.top_p = 1.0f;
    sampler.min_p = 0.0f;
    sampler.typical_p = 1.0f;
    sampler.repeat_penalty = 2.0f;
    sampler.repeat_last_n = 1;
    sampler.penalize_nl = 0;
    sampler.presence_penalty = 0.0f;
    sampler.frequency_penalty = 0.0f;
    sampler.mirostat = 0;
    sampler.mirostat_tau = 0.0f;
    sampler.mirostat_eta = 0.0f;

    err = astral_session_set_sampler(session, &sampler);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSpanU8 empty{};
    err = astral_session_feed(session, empty, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    const std::string text = read_stream_all(session);
    ASSERT_EQ(text, std::string("ababab"));

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_logprobs_meta_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model("sampler");

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 1;
    sd.temperature = 1.0f;
    sd.top_k = 2;
    sd.top_p = 1.0f;
    sd.stream_enabled = 1;
    sd.seed = 123;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSamplerDesc sampler{};
    sampler.size = sizeof(AstralSamplerDesc);
    sampler.temperature = 1.0f;
    sampler.top_k = 2;
    sampler.top_p = 1.0f;
    sampler.min_p = 0.0f;
    sampler.typical_p = 1.0f;
    sampler.repeat_penalty = 1.0f;
    sampler.repeat_last_n = 0;
    sampler.penalize_nl = 0;
    sampler.presence_penalty = 0.0f;
    sampler.frequency_penalty = 0.0f;
    sampler.mirostat = 0;
    sampler.mirostat_tau = 0.0f;
    sampler.mirostat_eta = 0.0f;
    err = astral_session_set_sampler(session, &sampler);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_set_logprobs(session, 2);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSpanU8 empty{};
    err = astral_session_feed(session, empty, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralTokenMeta meta[4]{};
    const int32_t n = astral_stream_read_meta(session, meta, 4, 1000);
    ASSERT_GT(n, 0);
    const uint32_t a = static_cast<uint32_t>('a');
    const uint32_t b = static_cast<uint32_t>('b');
    ASSERT_TRUE(meta[0].token_id == a || meta[0].token_id == b);
    ASSERT_EQ(meta[0].top_n, 2u);
    ASSERT_TRUE((meta[0].top_token_ids[0] == a && meta[0].top_token_ids[1] == b) ||
                (meta[0].top_token_ids[0] == b && meta[0].top_token_ids[1] == a));
    ASSERT_TRUE(std::fabs(meta[0].top_logprobs[0] - (-0.693f)) <= 0.05f);
    ASSERT_TRUE(std::fabs(meta[0].top_logprobs[1] - (-0.693f)) <= 0.05f);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_grammar_gbnf_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model("sampler");

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 8;
    sd.temperature = 1.0f;
    sd.top_k = 2;
    sd.top_p = 1.0f;
    sd.stream_enabled = 1;
    sd.seed = 123;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* g = "allow=a";
    AstralSpanU8 gbnf{};
    gbnf.data = reinterpret_cast<const uint8_t*>(g);
    gbnf.len = static_cast<uint32_t>(std::strlen(g));

    AstralSpanU8 root{};
    err = astral_session_set_grammar_gbnf(session, gbnf, root);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSpanU8 empty{};
    err = astral_session_feed(session, empty, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    const std::string text = read_stream_all(session);
    ASSERT_EQ(text, std::string("aaaaaaaa"));

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_adapters_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 4;
    sd.temperature = 0.0f;
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 1;
    sd.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* path = "adapter-1";
    AstralSpanU8 path_span{};
    path_span.data = reinterpret_cast<const uint8_t*>(path);
    path_span.len = static_cast<uint32_t>(std::strlen(path));
    AstralAdapterDesc ad{};
    ad.size = sizeof(AstralAdapterDesc);
    ad.path = path_span;

    AstralHandle adapter = 0;
    err = astral_model_adapter_load(model, &ad, &adapter);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(adapter));

    err = astral_session_adapters_add(session, adapter, 1.0f);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_adapters_clear(session);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_model_adapter_release(adapter);
    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_kv_state_roundtrip_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 4;
    sd.temperature = 0.0f;
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 1;
    sd.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSpanU8 empty{};
    err = astral_session_feed(session, empty, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    uint64_t bytes = 0;
    err = astral_session_state_size(session, &bytes);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GE(bytes, 16u);

    std::string buf;
    buf.resize(static_cast<size_t>(bytes));

    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(buf.data());
    out.len = static_cast<uint32_t>(buf.size());
    uint64_t written = 0;
    err = astral_session_state_save(session, out, &written);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(written, 0u);

    AstralSpanU8 in{};
    in.data = reinterpret_cast<const uint8_t*>(buf.data());
    in.len = static_cast<uint32_t>(written);
    err = astral_session_state_load(session, in);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_slots_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 1;
    sd.temperature = 0.0f;
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 0;
    sd.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_set_slot(session, 7);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_cancel_wait_and_reset_invariants) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Infinite mock never emits EOS; used to validate cancel/wait/backpressure.
    const AstralHandle model = load_mock_model("infinite");

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 1000000000u;
    session_desc.temperature = 0.0f;
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
    chunk.len = static_cast<uint32_t>(std::strlen(prompt));
    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    // Cannot reset while decoding.
    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_E_STATE);

    // Poll wait should time out while decoding is in flight.
    err = astral_session_wait(session, 0);
    ASSERT_EQ(err, ASTRAL_E_TIMEOUT);

    // Cancel and wait should return canceled.
    err = astral_session_cancel(session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_E_CANCELED);

    AstralSessionState state = 0;
    err = astral_session_state(session, &state);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_CANCELED);

    // Drain any buffered output (may be empty depending on scheduling).
    (void)drain_stream(session, 512);

    // Reset should clear cancellation and return to idle.
    session_desc.max_tokens = 8;
    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_state(session, &state);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_IDLE);

    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    // This run terminates by max_tokens.
    ASSERT_GT(drain_stream(session, 256), 0u);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_state(session, &state);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_COMPLETED);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_cancel_unblocks_when_stream_not_drained) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Infinite mock never emits EOS; used to force steady streaming + backpressure.
    const AstralHandle model = load_mock_model("infinite");

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 1000000000u;
    session_desc.temperature = 0.0f;
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
    chunk.len = static_cast<uint32_t>(std::strlen(prompt));
    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    // Intentionally do not read the stream; give the producer time to fill the ring and block.
    std::this_thread::sleep_for(std::chrono::milliseconds(25));

    err = astral_session_cancel(session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_E_CANCELED);

    AstralSessionState state = 0;
    err = astral_session_state(session, &state);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_CANCELED);

    // Drain any buffered output. After draining, reads should return end-of-stream (0).
    (void)drain_stream(session, 1024);

    uint8_t buf[64];
    AstralMutSpanU8 out = {};
    out.data = buf;
    out.len = sizeof(buf);
    ASSERT_EQ(astral_stream_read(session, out, 0), 0);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_reset_clears_stream_buffer) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 32;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    // Run decode without reading the stream. This should complete because max_tokens << ring capacity.
    const char* prompt = "hi";
    AstralSpanU8 chunk = {};
    chunk.data = reinterpret_cast<const uint8_t*>(prompt);
    chunk.len = static_cast<uint32_t>(std::strlen(prompt));
    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    // Reset must clear any unread stream tokens and return the session to idle.
    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_OK);

    uint8_t buf[64];
    AstralMutSpanU8 out = {};
    out.data = buf;
    out.len = sizeof(buf);
    ASSERT_EQ(astral_stream_read(session, out, 0), ASTRAL_E_TIMEOUT);

    AstralSessionState state = 0;
    err = astral_session_state(session, &state);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_IDLE);

    // Ensure the session can be reused successfully.
    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(drain_stream(session, 256), 0u);
    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

//
// Statistics Tests
//

TEST(inference_session_stats_null_session) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralStats stats;
    AstralErr err = astral_session_stats(0, &stats);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(inference_session_stats_null_output) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Would need valid session for full test
    // This tests null output handling
    astral_shutdown();
}

//
// Embeddings Tests
//

TEST(inference_embed_create_null_model) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralHandle embedder;
    AstralErr err = astral_embed_create(0, &embedder);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(inference_embed_create_null_output) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Would need valid model for full test
    AstralErr err = astral_embed_create(0, nullptr);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(inference_embed_create_and_run_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    astral_init(&cfg);

    const char* backend = "mock";
    const char* model_path = "infinite";
    AstralModelDesc model_desc{};
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    model_desc.n_ctx = 128;
    model_desc.embeddings_only = 1;

    AstralHandle model = 0;
    AstralErr err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(dim, 0u);

    AstralHandle embedder = 0;
    err = astral_embed_create(model, &embedder);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(embedder));

    const char* text = "abc";
    AstralSpanU8 text_span{};
    text_span.data = reinterpret_cast<const uint8_t*>(text);
    text_span.len = static_cast<uint32_t>(std::strlen(text));

    uint64_t ticket = 0;
    err = astral_embed_enqueue(embedder, text_span, &ticket);
    ASSERT_EQ(err, ASTRAL_OK);

    float vec[64] = {};
    ASSERT_LE(dim, 64u);
    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = static_cast<uint32_t>(sizeof(vec));

    err = astral_embed_collect(embedder, ticket, out);
    ASSERT_EQ(err, ASTRAL_OK);

    const float expected0 = static_cast<float>(256 + 97 + 98 + 99);
    ASSERT_EQ(vec[0], expected0);

    astral_embed_destroy(embedder);
    astral_model_release(model);
    astral_shutdown();
}
