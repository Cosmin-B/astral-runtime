/**
 * test_media.cpp - Vision/audio surface tests (mock + optional CPU init)
 */

#include "test_framework.hpp"
#include "astral_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

namespace {

static AstralSpanU8 span_from_cstr(const char* s) {
    AstralSpanU8 out{};
    out.data = reinterpret_cast<const uint8_t*>(s);
    out.len = s ? static_cast<uint32_t>(std::strlen(s)) : 0u;
    return out;
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

static bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool media_fixture_ready(const char* label,
                                const char* model_path,
                                const char* media_path) {
    const bool model_ok = file_exists_min_size(model_path, 100ULL * 1024ULL * 1024ULL);
    const bool media_ok = file_exists_min_size(media_path, 1ULL * 1024ULL * 1024ULL);
    if (model_ok && media_ok) {
        return true;
    }

    if (env_enabled("ASTRAL_TEST_REQUIRE_MEDIA")) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "%s media fixture missing or too small (model >=100MiB, media >=1MiB)",
                      label);
        astral::testing::test_fail_msg(__FILE__, __LINE__, msg);
    }
    return false;
}

static AstralHandle load_mock_model(const char* model_path, uint8_t embeddings_only) {
    const char* backend = "mock";
    const char* path = model_path ? model_path : "infinite";

    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    desc.backend_name = span_from_cstr(backend);
    desc.model_path = span_from_cstr(path);
    desc.embeddings_only = embeddings_only;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    return model;
}

static AstralImageDesc make_tiny_image() {
    static uint8_t pixels[3] = { 1, 2, 3 };
    AstralImageDesc img{};
    img.size = sizeof(AstralImageDesc);
    img.format = ASTRAL_IMAGE_FORMAT_RGB8;
    img.width = 1;
    img.height = 1;
    img.row_stride = 0;
    img.flags = 0;
    img.pixels.data = pixels;
    img.pixels.len = sizeof(pixels);
    return img;
}

static AstralAudioDesc make_tiny_audio() {
    static uint8_t samples[4] = { 4, 5, 6, 7 };
    AstralAudioDesc audio{};
    audio.size = sizeof(AstralAudioDesc);
    audio.format = ASTRAL_AUDIO_FORMAT_I16;
    audio.channels = 1;
    audio.sample_rate = 16000;
    audio.frame_count = 2;
    audio.samples.data = samples;
    audio.samples.len = sizeof(samples);
    audio.flags = 0;
    return audio;
}

#if ASTRAL_ENABLE_MTMD
static void run_cpu_media_init_smoke(const char* label,
                                     const char* model_env,
                                     const char* media_env,
                                     bool expect_image,
                                     bool expect_audio) {
    const char* model_path = std::getenv(model_env);
    const char* media_path = std::getenv(media_env);
    if (!media_fixture_ready(label, model_path, media_path)) {
        return;
    }

    AstralInit cfg{};
    cfg.reserve_bytes = 2ULL << 30;
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    desc.backend_name = span_from_cstr("cpu");
    desc.model_path = span_from_cstr(model_path);
    desc.n_ctx = 0;
    desc.n_batch = 0;
    desc.n_threads = 0;
    desc.embeddings_only = 0;

    AstralHandle model = 0;
    ASSERT_EQ(astral_model_load(&desc, &model), ASTRAL_OK);

    AstralModelMediaDesc media{};
    media.size = sizeof(AstralModelMediaDesc);
    media.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    media.media_path = span_from_cstr(media_path);
    ASSERT_EQ(astral_model_media_init(model, &media), ASTRAL_OK);

    AstralMediaInfo info{};
    info.size = sizeof(AstralMediaInfo);
    ASSERT_EQ(astral_model_media_info(model, &info), ASTRAL_OK);
    if (expect_image) {
        ASSERT_TRUE(info.supports_image != 0);
    }
    if (expect_audio) {
        ASSERT_TRUE(info.supports_audio != 0);
    }

    astral_model_release(model);
    astral_shutdown();
}
#endif

} // namespace

TEST(media_mock_session_conv) {
    AstralInit cfg{};
    cfg.reserve_bytes = 64ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = load_mock_model("infinite", /*embeddings_only=*/0);

    AstralModelMediaDesc media_desc{};
    media_desc.size = sizeof(AstralModelMediaDesc);
    media_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    ASSERT_EQ(astral_model_media_init(model, &media_desc), ASTRAL_OK);

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = 8;
    sd.temperature = 0.0f;
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 0;

    AstralHandle session = 0;
    ASSERT_EQ(astral_session_create(&sd, &session), ASTRAL_OK);

    const char* text = "hi";
    AstralSpanU8 text_span = span_from_cstr(text);
    ASSERT_EQ(astral_session_feed(session, text_span, 0), ASTRAL_OK);

    AstralImageDesc img = make_tiny_image();
    ASSERT_EQ(astral_session_feed_image(session, &img, 1), ASTRAL_OK);

    ASSERT_EQ(astral_session_decode(session), ASTRAL_OK);
    ASSERT_EQ(astral_session_wait(session, 1000), ASTRAL_OK);

    astral_session_destroy(session);

    AstralExecutorDesc ex{};
    ex.size = sizeof(AstralExecutorDesc);
    ex.max_slots = 1;
    ex.max_batch_tokens = 8;
    ex.worker_hint = 0;
    ASSERT_EQ(astral_model_executor_configure(model, &ex), ASTRAL_OK);

    AstralConvDesc cd{};
    cd.size = sizeof(AstralConvDesc);
    cd.model = model;
    cd.max_tokens = 8;
    cd.temperature = 0.0f;
    cd.top_k = 0;
    cd.top_p = 1.0f;
    cd.stream_enabled = 0;
    cd.seed = 1;

    AstralHandle conv = 0;
    ASSERT_EQ(astral_conv_create(&cd, &conv), ASTRAL_OK);

    AstralAudioDesc audio = make_tiny_audio();
    ASSERT_EQ(astral_conv_feed_audio(conv, &audio, 1), ASTRAL_OK);
    ASSERT_EQ(astral_conv_decode(conv), ASTRAL_OK);
    ASSERT_EQ(astral_conv_wait(conv, 1000), ASTRAL_OK);

    astral_conv_destroy(conv);
    astral_model_release(model);
    astral_shutdown();
}

TEST(media_mock_embeddings) {
    AstralInit cfg{};
    cfg.reserve_bytes = 64ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = load_mock_model("infinite", /*embeddings_only=*/1);

    AstralHandle emb = 0;
    ASSERT_EQ(astral_embed_create(model, &emb), ASTRAL_OK);

    AstralImageDesc img = make_tiny_image();
    AstralAudioDesc audio = make_tiny_audio();

    float vec[16] = {};
    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = sizeof(vec);

    uint64_t ticket = 0;
    ASSERT_EQ(astral_embed_enqueue_image(emb, &img, &ticket), ASTRAL_OK);
    ASSERT_NE(ticket, 0ULL);
    ASSERT_EQ(astral_embed_collect(emb, ticket, out), ASTRAL_OK);
    ASSERT_EQ(vec[0], 17.0f);

    ticket = 0;
    ASSERT_EQ(astral_embed_enqueue_audio(emb, &audio, &ticket), ASTRAL_OK);
    ASSERT_NE(ticket, 0ULL);
    ASSERT_EQ(astral_embed_collect(emb, ticket, out), ASTRAL_OK);
    ASSERT_EQ(vec[0], 51.0f);

    const char* text = "hi";
    AstralSpanU8 text_span = span_from_cstr(text);
    ticket = 0;
    ASSERT_EQ(astral_embed_enqueue_multimodal(emb, text_span, &img, nullptr, &ticket), ASTRAL_OK);
    ASSERT_NE(ticket, 0ULL);
    ASSERT_EQ(astral_embed_collect(emb, ticket, out), ASTRAL_OK);
    ASSERT_EQ(vec[0], 218.0f);

    astral_embed_destroy(emb);
    astral_model_release(model);
    astral_shutdown();
}

TEST(media_cpu_vision_init_smoke) {
#if ASTRAL_ENABLE_MTMD
    run_cpu_media_init_smoke("vision", "ASTRAL_TEST_VISION_MODEL", "ASTRAL_TEST_VISION_MEDIA",
                             /*expect_image=*/true, /*expect_audio=*/false);
#else
    if (env_enabled("ASTRAL_TEST_REQUIRE_MEDIA")) {
        astral::testing::test_fail_msg(__FILE__, __LINE__, "ASTRAL_TEST_REQUIRE_MEDIA needs ASTRAL_ENABLE_MTMD=ON");
    }
#endif
}

TEST(media_cpu_audio_init_smoke) {
#if ASTRAL_ENABLE_MTMD
    run_cpu_media_init_smoke("audio", "ASTRAL_TEST_AUDIO_MODEL", "ASTRAL_TEST_AUDIO_MEDIA",
                             /*expect_image=*/false, /*expect_audio=*/true);
#else
    if (env_enabled("ASTRAL_TEST_REQUIRE_MEDIA")) {
        astral::testing::test_fail_msg(__FILE__, __LINE__, "ASTRAL_TEST_REQUIRE_MEDIA needs ASTRAL_ENABLE_MTMD=ON");
    }
#endif
}
