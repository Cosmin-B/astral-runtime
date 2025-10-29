/**
 * test_continuous_batching.cpp - Continuous batching (v0.2+) tests
 *
 * Validates:
 * - model_executor_configure + conv_create for multiple slots
 * - concurrent progress across multiple conversations (no starvation)
 * - conv_stats surface is callable during decoding
 */

#include "test_framework.hpp"
#include "astral_rt.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <sys/stat.h>

namespace {

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

static AstralHandle load_mock_model() {
    AstralModelDesc md{};
    md.size = sizeof(AstralModelDesc);
    md.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    md.backend_name = span_from_cstr("mock");
    md.n_ctx = 128;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&md, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));
    return model;
}

} // namespace

TEST(continuous_batching_mock_slot_fairness) {
    constexpr uint32_t kSlots = 4;

    AstralInit cfg{};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model();

    AstralExecutorDesc exd{};
    exd.size = sizeof(AstralExecutorDesc);
    exd.max_slots = kSlots;
    exd.max_batch_tokens = 16;
    exd.worker_hint = 0;
    err = astral_model_executor_configure(model, &exd);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle convs[kSlots] = {0, 0, 0, 0};
    const char* prompts[kSlots] = {"slot zero", "slot one", "slot two", "slot three"};

    for (uint32_t i = 0; i < kSlots; ++i) {
        AstralConvDesc cd{};
        cd.size = sizeof(AstralConvDesc);
        cd.model = model;
        cd.max_tokens = 12;
        cd.temperature = 0.0f;
        cd.top_k = 0;
        cd.top_p = 1.0f;
        cd.stream_enabled = 1;
        cd.seed = 1u + i;

        err = astral_conv_create(&cd, &convs[i]);
        ASSERT_EQ(err, ASTRAL_OK);
        ASSERT_TRUE(astral_handle_valid(convs[i]));

        err = astral_conv_feed(convs[i], span_from_cstr(prompts[i]), 1);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    AstralConvDesc overflow_desc{};
    overflow_desc.size = sizeof(AstralConvDesc);
    overflow_desc.model = model;
    overflow_desc.max_tokens = 4;
    overflow_desc.stream_enabled = 1;
    AstralHandle overflow = 0;
    err = astral_conv_create(&overflow_desc, &overflow);
    ASSERT_EQ(err, ASTRAL_E_NOMEM);
    ASSERT_EQ(overflow, 0u);

    for (uint32_t i = 0; i < kSlots; ++i) {
        err = astral_conv_decode(convs[i]);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    bool got_any[kSlots] = {false, false, false, false};
    uint32_t done = 0;
    uint32_t spins = 0;
    while (done < kSlots && spins < 400) {
        done = 0;
        for (uint32_t i = 0; i < kSlots; ++i) {
            uint8_t buf[64];
            AstralMutSpanU8 out{};
            out.data = buf;
            out.len = sizeof(buf);
            const int32_t n = astral_conv_stream_read(convs[i], out, 0);
            if (n > 0) {
                got_any[i] = true;
            } else if (n != 0 && n != ASTRAL_E_TIMEOUT) {
                ASSERT_TRUE(false);
            }

            AstralConvStats stats{};
            err = astral_conv_stats(convs[i], &stats);
            ASSERT_EQ(err, ASTRAL_OK);
            ASSERT_LT(stats.slot_id, kSlots);
            ASSERT_GT(stats.prompt_tokens, 0u);

            AstralSessionState state = ASTRAL_SESSION_FAILED;
            err = astral_conv_state(convs[i], &state);
            ASSERT_EQ(err, ASTRAL_OK);
            if (state == ASTRAL_SESSION_COMPLETED) {
                ++done;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ++spins;
    }

    ASSERT_EQ(done, kSlots);
    for (uint32_t i = 0; i < kSlots; ++i) {
        ASSERT_TRUE(got_any[i]);
        err = astral_conv_wait(convs[i], 1000);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    astral_conv_destroy(convs[1]);
    convs[1] = 0;

    AstralConvDesc reuse_desc{};
    reuse_desc.size = sizeof(AstralConvDesc);
    reuse_desc.model = model;
    reuse_desc.max_tokens = 8;
    reuse_desc.temperature = 0.0f;
    reuse_desc.top_k = 0;
    reuse_desc.top_p = 1.0f;
    reuse_desc.stream_enabled = 1;
    reuse_desc.seed = 99;

    err = astral_conv_create(&reuse_desc, &convs[1]);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_conv_feed(convs[1], span_from_cstr("reused slot"), 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_conv_decode(convs[1]);
    ASSERT_EQ(err, ASTRAL_OK);

    bool reuse_output = false;
    for (uint32_t i = 0; i < 128; ++i) {
        uint8_t buf[64];
        AstralMutSpanU8 out{};
        out.data = buf;
        out.len = sizeof(buf);
        const int32_t n = astral_conv_stream_read(convs[1], out, 0);
        if (n > 0) {
            reuse_output = true;
        } else if (n != 0 && n != ASTRAL_E_TIMEOUT) {
            ASSERT_TRUE(false);
        }

        AstralSessionState state = ASTRAL_SESSION_FAILED;
        err = astral_conv_state(convs[1], &state);
        ASSERT_EQ(err, ASTRAL_OK);
        if (state == ASTRAL_SESSION_COMPLETED) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(reuse_output);
    err = astral_conv_wait(convs[1], 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    for (uint32_t i = 0; i < kSlots; ++i) {
        if (convs[i] != 0) {
            astral_conv_destroy(convs[i]);
        }
    }
    astral_model_release(model);
    astral_shutdown();
}

TEST(continuous_batching_multi_conv_cpu) {
    const char* model_path = find_test_model_path();
    if (model_path == nullptr) {
        return;
    }

    AstralInit cfg{};
    cfg.reserve_bytes = 2ULL << 30;
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralModelDesc md{};
    md.size = sizeof(AstralModelDesc);
    md.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    md.backend_name = span_from_cstr("cpu");
    md.model_path = span_from_cstr(model_path);
    md.n_ctx = 512;
    md.n_batch = 128;
    md.n_threads = 0;
    md.embeddings_only = 0;

    AstralHandle model = 0;
    err = astral_model_load(&md, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralExecutorDesc exd{};
    exd.size = sizeof(AstralExecutorDesc);
    exd.max_slots = 4;
    exd.max_batch_tokens = 64;
    exd.worker_hint = 0;
    err = astral_model_executor_configure(model, &exd);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle convs[3] = {0, 0, 0};
    const char* prompts[3] = {"Hello", "Once upon a time", "The meaning of life is"};

    for (int i = 0; i < 3; ++i) {
        AstralConvDesc cd{};
        cd.size = sizeof(AstralConvDesc);
        cd.model = model;
        cd.max_tokens = 16;
        cd.temperature = 0.0f;
        cd.top_k = 0;
        cd.top_p = 1.0f;
        cd.stream_enabled = 1;
        cd.seed = 1u + static_cast<uint32_t>(i);

        err = astral_conv_create(&cd, &convs[i]);
        ASSERT_EQ(err, ASTRAL_OK);
        ASSERT_TRUE(astral_handle_valid(convs[i]));

        err = astral_conv_feed(convs[i], span_from_cstr(prompts[i]), /*finalize=*/1);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    for (int i = 0; i < 3; ++i) {
        err = astral_conv_decode(convs[i]);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    bool got_any[3] = {false, false, false};
    uint32_t done = 0;
    uint32_t spins = 0;

    // Poll/consume round-robin until all conversations finish (or time out).
    while (done < 3 && spins < 2000) { // ~2000 * 5ms = 10s budget
        done = 0;
        for (int i = 0; i < 3; ++i) {
            AstralSessionState st = ASTRAL_SESSION_FAILED;
            err = astral_conv_state(convs[i], &st);
            ASSERT_EQ(err, ASTRAL_OK);

            if (st == ASTRAL_SESSION_COMPLETED || st == ASTRAL_SESSION_CANCELED || st == ASTRAL_SESSION_FAILED) {
                ++done;
            }

            uint8_t buf[128];
            AstralMutSpanU8 out{};
            out.data = buf;
            out.len = sizeof(buf);

            const int32_t n = astral_conv_stream_read(convs[i], out, /*timeout_ms=*/0);
            if (n > 0) {
                got_any[i] = true;
            } else if (n != 0 && n != ASTRAL_E_TIMEOUT) {
                ASSERT_TRUE(false);
            }

            AstralConvStats cs{};
            err = astral_conv_stats(convs[i], &cs);
            ASSERT_EQ(err, ASTRAL_OK);
            ASSERT_LT(cs.slot_id, exd.max_slots);
            ASSERT_GT(cs.prompt_tokens, 0u);
        }

        // Make sure everyone makes progress (no starvation) within the first few iterations.
        if (spins == 200) {
            ASSERT_TRUE(got_any[0]);
            ASSERT_TRUE(got_any[1]);
            ASSERT_TRUE(got_any[2]);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ++spins;
    }

    ASSERT_EQ(done, 3u);

    for (int i = 0; i < 3; ++i) {
        (void)astral_conv_wait(convs[i], 1000);
        astral_conv_destroy(convs[i]);
    }

    astral_model_release(model);
    astral_shutdown();
}
