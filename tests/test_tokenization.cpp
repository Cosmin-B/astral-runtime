#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <cstring>

namespace {

AstralSpanU8 span_from_cstr(const char* text) {
    AstralSpanU8 span{};
    span.data = reinterpret_cast<const uint8_t*>(text);
    span.len = static_cast<uint32_t>(std::strlen(text));
    return span;
}

AstralHandle load_mock_model() {
    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    desc.n_ctx = 128;
    desc.n_batch = 64;

    AstralHandle model = 0;
    ASSERT_EQ(astral_model_load(&desc, &model), ASTRAL_OK);
    return model;
}

void init_runtime() {
    AstralInit cfg{};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    cfg.thread_count = 1;
    cfg.numa_node = 0xFFFFFFFFu;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);
}

} // namespace

TEST(tokenization_count_and_roundtrip) {
    init_runtime();
    AstralHandle model = load_mock_model();

    const AstralSpanU8 text = span_from_cstr("hello");
    uint32_t count = 0;
    ASSERT_EQ(astral_tokenize_count(model, text, 1, 0, &count), ASTRAL_OK);
    ASSERT_EQ(count, 6u);

    int32_t tokens[8]{};
    uint32_t written = 0;
    ASSERT_EQ(astral_tokenize(model, text, tokens, 8, 1, 0, &written), ASTRAL_OK);
    ASSERT_EQ(written, count);
    ASSERT_EQ(tokens[0], 256);
    ASSERT_EQ(tokens[1], static_cast<int32_t>('h'));

    uint32_t bytes = 0;
    ASSERT_EQ(astral_detokenize_count(model, tokens, written, &bytes), ASTRAL_OK);
    ASSERT_EQ(bytes, 5u);

    uint8_t out[8]{};
    AstralMutSpanU8 out_span{};
    out_span.data = out;
    out_span.len = static_cast<uint32_t>(sizeof(out));
    uint32_t out_len = 0;
    ASSERT_EQ(astral_detokenize(model, tokens, written, out_span, &out_len), ASTRAL_OK);
    ASSERT_EQ(out_len, bytes);
    ASSERT_EQ(std::memcmp(out, "hello", 5), 0);

    astral_model_release(model);
    astral_shutdown();
}

TEST(tokenization_empty_and_utf8_count) {
    init_runtime();
    AstralHandle model = load_mock_model();

    AstralSpanU8 empty{};
    uint32_t count = 99;
    ASSERT_EQ(astral_tokenize_count(model, empty, 0, 0, &count), ASTRAL_OK);
    ASSERT_EQ(count, 0u);
    ASSERT_EQ(astral_tokenize_count(model, empty, 1, 0, &count), ASTRAL_OK);
    ASSERT_EQ(count, 1u);

    const AstralSpanU8 utf8 = span_from_cstr(u8"hé");
    ASSERT_EQ(astral_tokenize_count(model, utf8, 0, 0, &count), ASTRAL_OK);
    ASSERT_EQ(count, utf8.len);

    int32_t tokens[8]{};
    uint32_t written = 0;
    ASSERT_EQ(astral_tokenize(model, utf8, tokens, 8, 0, 0, &written), ASTRAL_OK);
    ASSERT_EQ(written, count);

    uint32_t bytes = 0;
    ASSERT_EQ(astral_detokenize_count(model, tokens, written, &bytes), ASTRAL_OK);
    ASSERT_EQ(bytes, utf8.len);

    uint8_t out[8]{};
    AstralMutSpanU8 out_span{};
    out_span.data = out;
    out_span.len = static_cast<uint32_t>(sizeof(out));
    uint32_t out_len = 0;
    ASSERT_EQ(astral_detokenize(model, tokens, written, out_span, &out_len), ASTRAL_OK);
    ASSERT_EQ(out_len, utf8.len);
    ASSERT_EQ(std::memcmp(out, utf8.data, utf8.len), 0);

    astral_model_release(model);
    astral_shutdown();
}

TEST(tokenization_batch_offsets_and_sizing) {
    constexpr uint32_t kRequestCount = 3;
    constexpr uint32_t kOffsetCount = kRequestCount + 1;
    constexpr uint32_t kTotalTokens = 6;
    constexpr uint32_t kFullTokenCapacity = 8;
    constexpr uint32_t kSmallTokenCapacity = 5;
    constexpr uint32_t kFirstOffset = 3;
    constexpr uint32_t kSecondOffset = 3;
    constexpr uint32_t kThirdOffset = kTotalTokens;

    init_runtime();
    AstralHandle model = load_mock_model();

    AstralTokenizeRequest requests[kRequestCount]{};
    requests[0].text = span_from_cstr("ab");
    requests[0].add_special = 1;
    requests[1].text = span_from_cstr("");
    requests[2].text = span_from_cstr("xyz");

    uint32_t offsets[kOffsetCount]{};
    uint32_t total = 0;
    ASSERT_EQ(astral_tokenize_batch(model, requests, kRequestCount, offsets, nullptr, 0, &total), ASTRAL_OK);
    ASSERT_EQ(total, kTotalTokens);
    ASSERT_EQ(offsets[0], 0u);
    ASSERT_EQ(offsets[1], kFirstOffset);
    ASSERT_EQ(offsets[2], kSecondOffset);
    ASSERT_EQ(offsets[3], kThirdOffset);

    int32_t tokens[kFullTokenCapacity]{};
    ASSERT_EQ(astral_tokenize_batch(model, requests, kRequestCount, offsets, tokens, kFullTokenCapacity, &total), ASTRAL_OK);
    ASSERT_EQ(total, kTotalTokens);
    ASSERT_EQ(tokens[0], 256);
    ASSERT_EQ(tokens[1], static_cast<int32_t>('a'));
    ASSERT_EQ(tokens[2], static_cast<int32_t>('b'));
    ASSERT_EQ(tokens[3], static_cast<int32_t>('x'));
    ASSERT_EQ(tokens[5], static_cast<int32_t>('z'));

    offsets[0] = kTotalTokens;
    offsets[1] = kTotalTokens;
    offsets[2] = kTotalTokens;
    offsets[3] = kTotalTokens;
    ASSERT_EQ(astral_tokenize_batch(model, requests, kRequestCount, offsets, tokens, kSmallTokenCapacity, &total), ASTRAL_E_NOMEM);
    ASSERT_EQ(total, kTotalTokens);
    ASSERT_EQ(offsets[0], 0u);
    ASSERT_EQ(offsets[1], kFirstOffset);
    ASSERT_EQ(offsets[2], kSecondOffset);
    ASSERT_EQ(offsets[3], kThirdOffset);

    astral_model_release(model);
    astral_shutdown();
}
