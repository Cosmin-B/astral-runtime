/**
 * test_abi_invalid_args.cpp - Public C ABI boundary validation
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

namespace {

AstralSpanU8 null_span() {
    AstralSpanU8 s{};
    return s;
}

AstralMutSpanU8 null_mut_span() {
    AstralMutSpanU8 s{};
    return s;
}

AstralInit small_init() {
    AstralInit cfg{};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    cfg.thread_count = 1;
    cfg.numa_node = 0xFFFFFFFFu;
    return cfg;
}

} // namespace

TEST(abi_invalid_args_core_and_backend) {
    astral_version(nullptr, nullptr, nullptr);

    ASSERT_EQ(astral_init(nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_init2(nullptr), ASTRAL_E_INVALID);

    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    ASSERT_EQ(astral_backend_load_plugin(null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_handle_valid(0), 0);

    astral_clear_last_error();
    ASSERT_EQ(astral_last_error()[0], '\0');

    astral_shutdown();
}

TEST(abi_invalid_args_model_surface) {
    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = 0;
    ASSERT_EQ(astral_model_load(nullptr, &model), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_load2(nullptr, &model), ASTRAL_E_INVALID);

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    ASSERT_EQ(astral_model_load(&model_desc, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_load2(&model_desc, nullptr), ASTRAL_E_INVALID);

    AstralModelInfo info{};
    AstralCaps caps = 0;
    AstralModelLimits limits{};
    AstralMediaInfo media_info{};
    AstralAdapterDesc adapter_desc{};
    AstralHandle adapter = 0;
    int32_t tokens[2] = {1, 2};
    uint32_t token_count = 0;
    uint8_t text_buf[16] = {};
    AstralMutSpanU8 text_out{};
    text_out.data = text_buf;
    text_out.len = static_cast<uint32_t>(sizeof(text_buf));
    uint32_t dim = 0;

    ASSERT_EQ(astral_model_info(0, &info), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_info(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_caps(0, &caps), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_caps(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_limits(0, &limits), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_limits(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_media_init(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_media_info(0, &media_info), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_media_info(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_embedding_dim(0, &dim), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_embedding_dim(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_tokenize(0, null_span(), tokens, 2, 0, 0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_tokenize(0, null_span(), nullptr, 2, 0, 0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_tokenize(0, null_span(), tokens, 0, 0, 0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_tokenize(0, null_span(), tokens, 2, 0, 0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(0, tokens, 2, text_out, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(0, nullptr, 2, text_out, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(0, tokens, 2, null_mut_span(), &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(0, tokens, 2, text_out, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_adapter_load(0, &adapter_desc, &adapter), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_adapter_load(0, nullptr, &adapter), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_adapter_load(0, &adapter_desc, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_executor_configure(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_executor_tune(0, nullptr), ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(abi_invalid_args_session_surface) {
    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle session = 0;
    AstralSessionDesc session_desc{};
    session_desc.model = 0;

    ASSERT_EQ(astral_session_create(nullptr, &session), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_create(&session_desc, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_cancel(0), ASTRAL_E_INVALID);

    AstralSessionState state = ASTRAL_SESSION_IDLE;
    AstralStats stats{};
    uint64_t bytes = 0;
    uint64_t written = 0;
    AstralSamplerDesc sampler{};
    int32_t tokens[2] = {1, 2};

    ASSERT_EQ(astral_session_state(0, &state), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_wait(0, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_reset(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_sampler(0, &sampler), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_sampler(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_penalty_prompt_set_tokens(0, tokens, 2), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_stop_clear(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_stop_add_utf8(0, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_stop_set_utf8(0, nullptr, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_logprobs(0, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state_size(0, &bytes), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state_size(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state_save(0, null_mut_span(), &written), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state_save(0, null_mut_span(), nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state_load(0, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_adapters_clear(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_adapters_add(0, 1, 1.0f), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_adapters_add(1, 0, 1.0f), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_grammar_gbnf(0, null_span(), null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_grammar_json_schema(0, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_clear_grammar(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_slot(0, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_feed(0, null_span(), 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_feed_image(0, nullptr, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_feed_audio(0, nullptr, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_decode(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_stream_read(0, null_mut_span(), 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_stream_read_meta(0, nullptr, 0, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_stats(0, &stats), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_stats(0, nullptr), ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(abi_invalid_args_conversation_surface) {
    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle conv = 0;
    AstralConvDesc conv_desc{};
    conv_desc.size = sizeof(AstralConvDesc);
    AstralSessionState state = ASTRAL_SESSION_IDLE;
    AstralConvStats stats{};
    AstralSamplerDesc sampler{};
    int32_t tokens[2] = {1, 2};

    ASSERT_EQ(astral_conv_create(nullptr, &conv), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_create(&conv_desc, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_feed(0, null_span(), 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_feed_image(0, nullptr, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_feed_audio(0, nullptr, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_decode(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_cancel(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_state(0, &state), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_state(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_wait(0, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_reset(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_set_sampler(0, &sampler), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_set_sampler(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_penalty_prompt_set_tokens(0, tokens, 2), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stop_clear(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stop_add_utf8(0, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stop_set_utf8(0, nullptr, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_set_logprobs(0, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_grammar_set_gbnf(0, null_span(), null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_grammar_set_json_schema(0, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_grammar_clear(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stream_read(0, null_mut_span(), 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stream_read_meta(0, nullptr, 0, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stats(0, &stats), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stats(0, nullptr), ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(abi_invalid_args_embedding_surface) {
    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle embedder = 0;
    uint64_t ticket = 0;

    ASSERT_EQ(astral_embed_create(0, &embedder), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_create(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_enqueue(0, null_span(), &ticket), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_enqueue(0, null_span(), nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_enqueue_image(0, nullptr, &ticket), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_enqueue_image(0, nullptr, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_enqueue_audio(0, nullptr, &ticket), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_enqueue_audio(0, nullptr, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_enqueue_multimodal(0, null_span(), nullptr, nullptr, &ticket), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_enqueue_multimodal(0, null_span(), nullptr, nullptr, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_collect(0, 0, null_mut_span()), ASTRAL_E_INVALID);

    astral_shutdown();
}
