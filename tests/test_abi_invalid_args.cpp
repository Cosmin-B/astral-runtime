/**
 * test_abi_invalid_args.cpp - Public C ABI boundary validation
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <cstring>

namespace {

AstralSpanU8 null_span() {
    AstralSpanU8 s{};
    return s;
}

AstralSpanU8 span_from_cstr(const char* text) {
    AstralSpanU8 s{};
    s.data = reinterpret_cast<const uint8_t*>(text);
    s.len = static_cast<uint32_t>(std::strlen(text));
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

AstralModelDesc mock_model_desc() {
    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;

    const char* backend = "mock";
    desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    desc.n_ctx = 128;
    desc.n_batch = 64;
    desc.gpu_layers = 0;
    return desc;
}

AstralSessionDesc mock_session_desc(AstralHandle model) {
    AstralSessionDesc desc{};
    desc.model = model;
    desc.max_tokens = 8;
    desc.temperature = 0.0f;
    desc.top_k = 0;
    desc.top_p = 1.0f;
    desc.stream_enabled = 1;
    desc.seed = 1;
    return desc;
}

AstralConvDesc mock_conv_desc(AstralHandle model) {
    AstralConvDesc desc{};
    desc.size = sizeof(AstralConvDesc);
    desc.model = model;
    desc.max_tokens = 8;
    desc.temperature = 0.0f;
    desc.top_k = 0;
    desc.top_p = 1.0f;
    desc.stream_enabled = 1;
    desc.seed = 1;
    return desc;
}

AstralImageDesc invalid_image_desc() {
    static const uint8_t pixel[3] = {0, 0, 0};
    AstralImageDesc image{};
    image.size = 0;
    image.format = ASTRAL_IMAGE_FORMAT_RGB8;
    image.width = 1;
    image.height = 1;
    image.pixels.data = pixel;
    image.pixels.len = static_cast<uint32_t>(sizeof(pixel));
    return image;
}

AstralAudioDesc invalid_audio_desc() {
    static const int16_t sample[1] = {0};
    AstralAudioDesc audio{};
    audio.size = 0;
    audio.format = ASTRAL_AUDIO_FORMAT_I16;
    audio.channels = 1;
    audio.sample_rate = 16000;
    audio.frame_count = 1;
    audio.samples.data = reinterpret_cast<const uint8_t*>(sample);
    audio.samples.len = static_cast<uint32_t>(sizeof(sample));
    return audio;
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
    constexpr uint32_t kToolId = 1;
    constexpr uint32_t kToolCount = 1;
    AstralToolDesc tool_desc{};
    tool_desc.size = sizeof(AstralToolDesc);
    tool_desc.tool_id = kToolId;
    tool_desc.name = span_from_cstr("search");
    tool_desc.json_schema = span_from_cstr("{\"type\":\"object\"}");
    AstralToolsetDesc toolset_desc{};
    toolset_desc.size = sizeof(AstralToolsetDesc);
    toolset_desc.tool_count = kToolCount;
    toolset_desc.choice_mode = ASTRAL_TOOL_CHOICE_AUTO;
    toolset_desc.tools = &tool_desc;
    AstralHandle toolset = 0;
    AstralToolInfo tool_info{};
    tool_info.size = sizeof(AstralToolInfo);
    AstralToolCallResult tool_call{};
    tool_call.size = sizeof(AstralToolCallResult);
    constexpr uint32_t kChunkMaxUnits = 2;
    constexpr uint32_t kInvalidTokenCount = 1;
    constexpr uint32_t kRangeCapacity = 1;
    AstralChunkerDesc chunker{};
    chunker.size = sizeof(AstralChunkerDesc);
    chunker.mode = ASTRAL_CHUNK_MODE_WORD;
    chunker.max_units = kChunkMaxUnits;
    AstralChunkRange chunk_range{};
    chunk_range.size = sizeof(AstralChunkRange);
    constexpr uint32_t kMemoryDim = 4;
    constexpr uint32_t kMemoryCapacity = 4;
    constexpr uint64_t kMemoryKey = 1;
    constexpr uint32_t kMemoryTopK = 1;
    constexpr uint32_t kMemoryFetchCapacity = 1;
    AstralMemoryIndexDesc memory_desc{};
    memory_desc.size = sizeof(AstralMemoryIndexDesc);
    memory_desc.dim = kMemoryDim;
    memory_desc.capacity = kMemoryCapacity;
    memory_desc.metric = ASTRAL_MEMORY_METRIC_DOT;
    memory_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
    AstralMemoryRecord memory_record{};
    memory_record.size = sizeof(AstralMemoryRecord);
    memory_record.key = kMemoryKey;
    AstralMemorySearchDesc memory_search{};
    memory_search.size = sizeof(AstralMemorySearchDesc);
    memory_search.top_k = kMemoryTopK;
    memory_search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    AstralMemorySearchResult memory_result{};
    AstralAgentDesc agent_desc{};
    agent_desc.size = sizeof(AstralAgentDesc);
    AstralAgentMessage agent_message{};
    agent_message.size = sizeof(AstralAgentMessage);
    agent_message.role = ASTRAL_AGENT_ROLE_USER;
    AstralAgentChatDesc agent_chat{};
    agent_chat.size = sizeof(AstralAgentChatDesc);
    AstralAgentChatResult agent_result{};
    agent_result.size = sizeof(AstralAgentChatResult);
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
    ASSERT_EQ(astral_tokenize_count(0, null_span(), 0, 0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_tokenize_count(0, null_span(), 0, 0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_tokenize_batch(0, nullptr, 0, nullptr, nullptr, 0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(0, tokens, 2, text_out, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(0, nullptr, 2, text_out, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(0, tokens, 2, null_mut_span(), &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(0, tokens, 2, text_out, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize_count(0, tokens, 2, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize_count(0, tokens, 2, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_adapter_load(0, &adapter_desc, &adapter), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_adapter_load(0, nullptr, &adapter), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_model_adapter_load(0, &adapter_desc, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_toolset_create(nullptr, &toolset), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_toolset_create(&toolset_desc, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_toolset_count(0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_toolset_count(1, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_toolset_get(0, 0, &tool_info), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_toolset_get(1, 0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_toolset_parse_call(0, null_span(), &tool_call), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_toolset_parse_call(1, null_span(), nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_chunk_count(nullptr, null_span(), &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_chunk_count(&chunker, null_span(), nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_chunk_ranges(nullptr, null_span(), &chunk_range, kRangeCapacity, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_chunk_ranges(&chunker, null_span(), nullptr, kRangeCapacity, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_chunk_text_copy(null_span(), nullptr, null_mut_span(), &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_chunk_text_copy(null_span(), &chunk_range, null_mut_span(), nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_token_chunk_count(nullptr, kInvalidTokenCount, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_token_chunk_count(&chunker, kInvalidTokenCount, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_token_chunk_ranges(nullptr, kInvalidTokenCount, &chunk_range, kRangeCapacity, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_token_chunk_ranges(&chunker, kInvalidTokenCount, nullptr, kRangeCapacity, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_create(nullptr, &toolset), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_create(&memory_desc, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_count(0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_count(1, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_clear(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_add_batch(0, &memory_record, nullptr, kMemoryTopK), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_remove(0, kMemoryKey), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_search(0, &memory_search, nullptr, &memory_result, kMemoryTopK, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_search_begin(0, &memory_search, nullptr, &toolset), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_search_fetch(0, &memory_result, kMemoryFetchCapacity, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_save_size(0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_save(0, null_mut_span(), nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_load(nullptr, null_span(), &toolset), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_memory_load(&memory_desc, null_span(), nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_create(nullptr, &toolset), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_create(&agent_desc, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_set_system_prompt(0, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_get_system_prompt_size(0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_get_system_prompt(0, text_out, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_message_add(0, &agent_message), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_history_clear(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_history_count(0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_history_save_size(0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_history_save(0, text_out, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_history_load(0, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_chat_enqueue(0, &agent_chat), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_chat_cancel(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_chat_stream_read(0, text_out, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_agent_chat_result(0, &agent_result), ASTRAL_E_INVALID);
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
    uint32_t adapter_count = 0;
    AstralHandle adapter = 0;
    float adapter_scale = 0.0f;

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
    ASSERT_EQ(astral_session_adapters_count(0, &adapter_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_adapters_count(1, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_adapters_get(0, 0, &adapter, &adapter_scale), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_adapters_get(1, 0, nullptr, &adapter_scale), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_adapters_get(1, 0, &adapter, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_grammar_gbnf(0, null_span(), null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_grammar_json_schema(0, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_clear_grammar(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_toolset(0, 1, ASTRAL_TOOL_CHOICE_AUTO), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_set_toolset(1, 0, ASTRAL_TOOL_CHOICE_AUTO), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_clear_toolset(0), ASTRAL_E_INVALID);
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

TEST(abi_invalid_args_valid_handle_buffer_surface) {
    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = 0;
    AstralModelDesc model_desc = mock_model_desc();
    ASSERT_EQ(astral_model_load(&model_desc, &model), ASTRAL_OK);

    AstralHandle session = 0;
    AstralSessionDesc session_desc = mock_session_desc(model);
    ASSERT_EQ(astral_session_create(&session_desc, &session), ASTRAL_OK);

    int32_t tokens[2] = {1, 2};
    uint64_t written = 0;
    AstralTokenMeta meta{};
    uint8_t bytes[16] = {};
    AstralSpanU8 text{};
    text.data = reinterpret_cast<const uint8_t*>("stop");
    text.len = 4;
    AstralSpanU8 seqs[1] = {text};
    AstralMutSpanU8 out{};
    out.data = bytes;
    out.len = static_cast<uint32_t>(sizeof(bytes));
    uint32_t token_count = 0;

    ASSERT_EQ(astral_tokenize(model, null_span(), nullptr, 2, 0, 0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_tokenize_count(model, null_span(), 0, 0, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_tokenize_batch(model, nullptr, 1, nullptr, nullptr, 0, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize(model, nullptr, 1, out, nullptr), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_detokenize_count(model, nullptr, 1, &token_count), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_penalty_prompt_set_tokens(session, nullptr, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_stop_add_utf8(session, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_stop_set_utf8(session, nullptr, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state_save(session, null_mut_span(), &written), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state_load(session, null_span()), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_stream_read(session, null_mut_span(), 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_stream_read_meta(session, nullptr, 1, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_stream_read_meta(session, &meta, 0, 0), ASTRAL_E_INVALID);

    ASSERT_EQ(astral_session_stop_set_utf8(session, seqs, 1), ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(abi_invalid_args_session_media_rejects_before_state_change) {
    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = 0;
    AstralModelDesc model_desc = mock_model_desc();
    ASSERT_EQ(astral_model_load(&model_desc, &model), ASTRAL_OK);

    AstralHandle session = 0;
    AstralSessionDesc session_desc = mock_session_desc(model);
    ASSERT_EQ(astral_session_create(&session_desc, &session), ASTRAL_OK);

    AstralImageDesc image = invalid_image_desc();
    AstralAudioDesc audio = invalid_audio_desc();
    AstralSessionState state = ASTRAL_SESSION_FAILED;

    ASSERT_EQ(astral_session_feed_image(session, &image, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state(session, &state), ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_IDLE);

    ASSERT_EQ(astral_session_feed_audio(session, &audio, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_session_state(session, &state), ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_IDLE);

    astral_session_destroy(session);
    astral_model_release(model);
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
    ASSERT_EQ(astral_conv_set_toolset(0, 1, ASTRAL_TOOL_CHOICE_AUTO), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_set_toolset(1, 0, ASTRAL_TOOL_CHOICE_AUTO), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_clear_toolset(0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stream_read(0, null_mut_span(), 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stream_read_meta(0, nullptr, 0, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stats(0, &stats), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stats(0, nullptr), ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(abi_invalid_args_valid_conversation_buffer_surface) {
    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = 0;
    AstralModelDesc model_desc = mock_model_desc();
    ASSERT_EQ(astral_model_load(&model_desc, &model), ASTRAL_OK);

    AstralExecutorDesc executor{};
    executor.size = sizeof(AstralExecutorDesc);
    executor.max_slots = 1;
    executor.max_batch_tokens = 8;
    executor.worker_hint = 0;
    ASSERT_EQ(astral_model_executor_configure(model, &executor), ASTRAL_OK);

    AstralHandle conv = 0;
    AstralConvDesc conv_desc = mock_conv_desc(model);
    ASSERT_EQ(astral_conv_create(&conv_desc, &conv), ASTRAL_OK);

    AstralTokenMeta meta{};
    ASSERT_EQ(astral_conv_stream_read(conv, null_mut_span(), 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stream_read_meta(conv, nullptr, 1, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_stream_read_meta(conv, &meta, 0, 0), ASTRAL_E_INVALID);

    astral_conv_destroy(conv);
    astral_model_release(model);
    astral_shutdown();
}

TEST(abi_invalid_args_conversation_media_rejects_before_state_change) {
    AstralInit cfg = small_init();
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = 0;
    AstralModelDesc model_desc = mock_model_desc();
    ASSERT_EQ(astral_model_load(&model_desc, &model), ASTRAL_OK);

    AstralExecutorDesc executor{};
    executor.size = sizeof(AstralExecutorDesc);
    executor.max_slots = 1;
    executor.max_batch_tokens = 8;
    executor.worker_hint = 0;
    ASSERT_EQ(astral_model_executor_configure(model, &executor), ASTRAL_OK);

    AstralHandle conv = 0;
    AstralConvDesc conv_desc = mock_conv_desc(model);
    ASSERT_EQ(astral_conv_create(&conv_desc, &conv), ASTRAL_OK);

    AstralImageDesc image = invalid_image_desc();
    AstralAudioDesc audio = invalid_audio_desc();
    AstralSessionState state = ASTRAL_SESSION_FAILED;

    ASSERT_EQ(astral_conv_feed_image(conv, &image, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_state(conv, &state), ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_IDLE);

    ASSERT_EQ(astral_conv_feed_audio(conv, &audio, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_conv_state(conv, &state), ASTRAL_OK);
    ASSERT_EQ(state, ASTRAL_SESSION_IDLE);

    astral_conv_destroy(conv);
    astral_model_release(model);
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
    ASSERT_EQ(astral_embed_cancel(0, 1), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_cancel(0, 0), ASTRAL_E_INVALID);
    ASSERT_EQ(astral_embed_collect(0, 0, null_mut_span()), ASTRAL_E_INVALID);

    astral_shutdown();
}
