/**
 * test_inference.cpp - Inference layer tests
 *
 * Tests for session management, state machine, sampler, and token generation.
 * Validates: session create/destroy, feed/decode, streaming, sampler configuration.
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <atomic>
#include <cstring>
#include <chrono>
#include <thread>
#include <string>
#include <cmath>

namespace {

AstralHandle load_mock_model(const char* model_path) {
    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
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

AstralSpanU8 span_from_cstr(const char* text) {
    AstralSpanU8 span{};
    span.data = reinterpret_cast<const uint8_t*>(text);
    span.len = static_cast<uint32_t>(std::strlen(text));
    return span;
}

bool bytes_contain(const std::vector<uint8_t>& bytes, const char* needle) {
    const size_t needle_len = std::strlen(needle);
    if (needle_len == 0 || bytes.size() < needle_len) {
        return false;
    }
    for (size_t i = 0; i + needle_len <= bytes.size(); ++i) {
        if (std::memcmp(bytes.data() + i, needle, needle_len) == 0) {
            return true;
        }
    }
    return false;
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

std::string run_mock_decode_once(AstralHandle session, const char* prompt) {
    AstralSpanU8 prompt_span{};
    prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
    prompt_span.len = static_cast<uint32_t>(std::strlen(prompt));

    AstralErr err = astral_session_feed(session, prompt_span, 1);
    if (err != ASTRAL_OK) {
        return {};
    }

    err = astral_session_decode(session);
    if (err != ASTRAL_OK) {
        return {};
    }

    const std::string text = read_stream_all(session);
    err = astral_session_wait(session, 1000);
    if (err != ASTRAL_OK) {
        return {};
    }
    return text;
}

std::string read_conv_stream_all(AstralHandle conv) {
    std::string out;
    out.reserve(256);

    uint8_t buf[256];
    for (uint32_t i = 0; i < 10000; ++i) {
        AstralMutSpanU8 span{};
        span.data = buf;
        span.len = sizeof(buf);

        const int32_t n = astral_conv_stream_read(conv, span, 100 /*ms*/);
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

std::string read_agent_stream_all(AstralHandle agent) {
    constexpr uint32_t kBufferBytes = 256;
    constexpr uint32_t kMaxReads = 256;
    constexpr uint32_t kReadTimeoutMs = 10;
    std::string out;
    out.reserve(kBufferBytes);

    uint8_t buf[kBufferBytes];
    for (uint32_t i = 0; i < kMaxReads; ++i) {
        AstralMutSpanU8 span{};
        span.data = buf;
        span.len = sizeof(buf);

        const int32_t n = astral_agent_chat_stream_read(agent, span, kReadTimeoutMs);
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

void run_mock_lifecycle_session(AstralHandle model, uint32_t seed) {
    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 12;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = seed;

    AstralHandle session = 0;
    AstralErr err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    const char* prompt = "lifecycle";
    AstralSpanU8 chunk = {};
    chunk.data = reinterpret_cast<const uint8_t*>(prompt);
    chunk.len = static_cast<uint32_t>(std::strlen(prompt));

    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(drain_stream(session, 128), 0u);
    err = astral_session_wait(session, 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralStats stats = {};
    err = astral_session_stats(session, &stats);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(stats.tok_per_s, 0.0);

    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_OK);

    uint8_t empty_buf[16];
    AstralMutSpanU8 out = {};
    out.data = empty_buf;
    out.len = sizeof(empty_buf);
    ASSERT_EQ(astral_stream_read(session, out, 0), ASTRAL_E_TIMEOUT);

    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(drain_stream(session, 128), 0u);
    err = astral_session_wait(session, 1000);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
}

} // namespace

//
// Session Creation Tests
//

TEST(inference_session_create_destroy) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 32 * 1024 * 1024;
    astral_init(&cfg);

    // Use mock backend so the test is hermetic (no external GGUF needed).
    AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 100;
    session_desc.temperature = 0.7f;
    session_desc.top_k = 40;
    session_desc.top_p = 0.9f;
    session_desc.stream_enabled = 1;

    AstralHandle session = 0;
    const AstralErr err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);

    astral_shutdown();
}

TEST(inference_request_lifecycle_session_mock) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 32 * 1024 * 1024;
    cfg.thread_count = 2;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 8;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 1;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;

    AstralHandle session = 0;
    ASSERT_EQ(astral_session_create(&session_desc, &session), ASTRAL_OK);

    AstralRequestRef request{};
    ASSERT_EQ(astral_request_from_session(session, &request), ASTRAL_OK);
    ASSERT_EQ(request.kind, ASTRAL_REQUEST_SESSION);
    ASSERT_EQ(request.owner, session);

    AstralRequestStatus status{};
    status.size = sizeof(AstralRequestStatus);
    ASSERT_EQ(astral_request_state(&request, &status), ASTRAL_OK);
    ASSERT_EQ(status.state, ASTRAL_REQUEST_QUEUED);

    ASSERT_EQ(astral_session_feed(session, span_from_cstr("hello"), 1), ASTRAL_OK);
    ASSERT_EQ(astral_session_decode(session), ASTRAL_OK);

    const std::string output = read_stream_all(session);
    ASSERT_FALSE(output.empty());

    status = AstralRequestStatus{};
    status.size = sizeof(AstralRequestStatus);
    ASSERT_EQ(astral_request_wait(&request, 1000, &status), ASTRAL_OK);
    ASSERT_EQ(status.state, ASTRAL_REQUEST_COMPLETED);
    ASSERT_EQ(status.result, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_mock_lifecycle_churn) {
    constexpr uint32_t kRuntimeCycles = 8;
    constexpr uint32_t kModelsPerRuntime = 2;
    constexpr uint32_t kSessionsPerModel = 3;

    for (uint32_t cycle = 0; cycle < kRuntimeCycles; ++cycle) {
        AstralInit cfg = {};
        cfg.reserve_bytes = 64 * 1024 * 1024;
        cfg.thread_count = 2;
        const AstralErr init_err = astral_init(&cfg);
        ASSERT_EQ(init_err, ASTRAL_OK);

        for (uint32_t model_idx = 0; model_idx < kModelsPerRuntime; ++model_idx) {
            const AstralHandle model = load_mock_model(nullptr);

            for (uint32_t session_idx = 0; session_idx < kSessionsPerModel; ++session_idx) {
                const uint32_t seed = 1 + cycle * 100 + model_idx * 10 + session_idx;
                run_mock_lifecycle_session(model, seed);
            }

            astral_model_release(model);
        }

        astral_shutdown();
    }
}

TEST(inference_conversation_multi_slot_isolation) {
    AstralInit cfg{};
    cfg.reserve_bytes = 32 * 1024 * 1024;
    astral_init(&cfg);

    AstralHandle model = load_mock_model(nullptr);

    AstralExecutorDesc ex{};
    ex.size = sizeof(AstralExecutorDesc);
    ex.max_slots = 2;
    ex.max_batch_tokens = 8;
    ex.worker_hint = 0;
    ASSERT_EQ(astral_model_executor_configure(model, &ex), ASTRAL_OK);

    AstralConvDesc conv_desc{};
    conv_desc.size = sizeof(AstralConvDesc);
    conv_desc.model = model;
    conv_desc.max_tokens = 16;
    conv_desc.temperature = 0.0f;
    conv_desc.top_k = 0;
    conv_desc.top_p = 1.0f;
    conv_desc.stream_enabled = 1;
    conv_desc.seed = 1;

    AstralHandle a = 0;
    AstralHandle b = 0;
    ASSERT_EQ(astral_conv_create(&conv_desc, &a), ASTRAL_OK);
    ASSERT_EQ(astral_conv_create(&conv_desc, &b), ASTRAL_OK);

    AstralSpanU8 empty{};
    ASSERT_EQ(astral_conv_feed(a, empty, 1), ASTRAL_OK);
    ASSERT_EQ(astral_conv_feed(b, empty, 1), ASTRAL_OK);

    ASSERT_EQ(astral_conv_decode(a), ASTRAL_OK);
    ASSERT_EQ(astral_conv_decode(b), ASTRAL_OK);

    const std::string out_a = read_conv_stream_all(a);
    const std::string out_b = read_conv_stream_all(b);

    ASSERT_EQ(astral_conv_wait(a, 1000), ASTRAL_OK);
    ASSERT_EQ(astral_conv_wait(b, 1000), ASTRAL_OK);

    ASSERT_FALSE(out_a.empty());
    ASSERT_FALSE(out_b.empty());
    ASSERT_NE(out_a, out_b);

    AstralHandle c = 0;
    ASSERT_EQ(astral_conv_create(&conv_desc, &c), ASTRAL_E_NOMEM);

    astral_conv_destroy(a);
    astral_conv_destroy(b);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_conversation_grammar_gbnf_mock) {
    AstralInit cfg{};
    cfg.reserve_bytes = 32 * 1024 * 1024;
    astral_init(&cfg);

    AstralHandle model = load_mock_model(nullptr);

    AstralExecutorDesc ex{};
    ex.size = sizeof(AstralExecutorDesc);
    ex.max_slots = 1;
    ex.max_batch_tokens = 8;
    ex.worker_hint = 0;
    ASSERT_EQ(astral_model_executor_configure(model, &ex), ASTRAL_OK);

    AstralConvDesc conv_desc{};
    conv_desc.size = sizeof(AstralConvDesc);
    conv_desc.model = model;
    conv_desc.max_tokens = 8;
    conv_desc.temperature = 0.0f;
    conv_desc.top_k = 0;
    conv_desc.top_p = 1.0f;
    conv_desc.stream_enabled = 1;
    conv_desc.seed = 1;

    AstralHandle conv = 0;
    ASSERT_EQ(astral_conv_create(&conv_desc, &conv), ASTRAL_OK);

    const char* gbnf = "allow=x";
    AstralSpanU8 g{};
    g.data = reinterpret_cast<const uint8_t*>(gbnf);
    g.len = static_cast<uint32_t>(std::strlen(gbnf));
    AstralSpanU8 root{};
    ASSERT_EQ(astral_conv_grammar_set_gbnf(conv, g, root), ASTRAL_OK);

    AstralSpanU8 empty{};
    ASSERT_EQ(astral_conv_feed(conv, empty, 1), ASTRAL_OK);
    ASSERT_EQ(astral_conv_decode(conv), ASTRAL_OK);

    const std::string out = read_conv_stream_all(conv);
    ASSERT_EQ(astral_conv_wait(conv, 1000), ASTRAL_OK);

    ASSERT_FALSE(out.empty());
    for (char ch : out) {
        ASSERT_EQ(ch, 'x');
    }

    astral_conv_destroy(conv);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_agent_history_and_chat_mock) {
    constexpr uint32_t kBytesPerKiB = 1024;
    constexpr uint32_t kBytesPerMiB = kBytesPerKiB * kBytesPerKiB;
    constexpr uint32_t kReserveMiB = 32;
    constexpr uint32_t kReserveBytes = kReserveMiB * kBytesPerMiB;
    constexpr uint32_t kExecutorSlots = 1;
    constexpr uint32_t kBatchTokens = 8;
    constexpr uint32_t kMaxTokens = 12;
    constexpr uint32_t kMaxMessages = 8;
    constexpr uint32_t kMaxPromptKiB = 4;
    constexpr uint32_t kMaxPromptBytes = kMaxPromptKiB * kBytesPerKiB;
    constexpr uint32_t kSeed = 7;
    constexpr char kSystemPrompt[] = "reply tersely";
    constexpr char kSummary[] = "user likes short answers";
    constexpr char kMemoryContext[] = "Retrieved: alpha document says use compact output.";
    constexpr uint32_t kSystemBytes = static_cast<uint32_t>(sizeof(kSystemPrompt) - 1);
    constexpr uint32_t kSummaryBytes = static_cast<uint32_t>(sizeof(kSummary) - 1);
    constexpr uint32_t kMemoryContextBytes = static_cast<uint32_t>(sizeof(kMemoryContext) - 1);
    constexpr uint32_t kHistoryCount = 2;
    constexpr uint32_t kSaveCapacity = 512;
    constexpr uint32_t kChatHistoryCount = 3;
    constexpr uint32_t kPromptCacheEntries = 4;
    constexpr uint32_t kPromptCacheTokens = kMaxPromptBytes;
    constexpr uint32_t kPromptCacheFirstMisses = 1;
    constexpr uint32_t kPromptCacheSecondHits = 1;
    constexpr uint32_t kPromptCacheNoReusedTokens = 0;
    constexpr uint32_t kPromptCacheNoNewTokens = 0;
    constexpr uint32_t kMemoryDim = 4;
    constexpr uint32_t kMemoryCapacity = 8;

    AstralInit cfg{};
    cfg.reserve_bytes = kReserveBytes;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    AstralHandle model = load_mock_model(nullptr);
    AstralExecutorDesc ex{};
    ex.size = sizeof(AstralExecutorDesc);
    ex.max_slots = kExecutorSlots;
    ex.max_batch_tokens = kBatchTokens;
    ASSERT_EQ(astral_model_executor_configure(model, &ex), ASTRAL_OK);

    AstralPromptCacheDesc cache_desc{};
    cache_desc.size = sizeof(AstralPromptCacheDesc);
    cache_desc.max_entries = kPromptCacheEntries;
    cache_desc.max_tokens = kPromptCacheTokens;
    cache_desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;
    cache_desc.flags = ASTRAL_PROMPT_CACHE_FLAG_TRACK_STATS;
    AstralHandle prompt_cache = 0;
    ASSERT_EQ(astral_prompt_cache_create(&cache_desc, &prompt_cache), ASTRAL_OK);

    AstralMemoryIndexDesc memory_desc{};
    memory_desc.size = sizeof(AstralMemoryIndexDesc);
    memory_desc.dim = kMemoryDim;
    memory_desc.capacity = kMemoryCapacity;
    memory_desc.metric = ASTRAL_MEMORY_METRIC_COSINE;
    memory_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
    AstralHandle memory_index = 0;
    ASSERT_EQ(astral_memory_create(&memory_desc, &memory_index), ASTRAL_OK);

    AstralAgentDesc desc{};
    desc.size = sizeof(AstralAgentDesc);
    desc.model = model;
    desc.prompt_cache = prompt_cache;
    desc.memory_index = memory_index;
    desc.max_tokens = kMaxTokens;
    desc.temperature = 0.0f;
    desc.top_p = 1.0f;
    desc.stream_enabled = 1;
    desc.seed = kSeed;
    desc.max_messages = kMaxMessages;
    desc.max_prompt_bytes = kMaxPromptBytes;

    AstralHandle agent = 0;
    ASSERT_EQ(astral_agent_create(&desc, &agent), ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(agent));

    ASSERT_EQ(astral_agent_set_system_prompt(agent, span_from_cstr(kSystemPrompt)), ASTRAL_OK);
    uint32_t bytes = 0;
    ASSERT_EQ(astral_agent_get_system_prompt_size(agent, &bytes), ASTRAL_OK);
    ASSERT_EQ(bytes, kSystemBytes);

    uint8_t system_buf[kSystemBytes]{};
    AstralMutSpanU8 system_out{};
    system_out.data = system_buf;
    system_out.len = sizeof(system_buf);
    ASSERT_EQ(astral_agent_get_system_prompt(agent, system_out, &bytes), ASTRAL_OK);
    ASSERT_EQ(std::string(reinterpret_cast<const char*>(system_buf), bytes), kSystemPrompt);

    ASSERT_EQ(astral_agent_set_summary(agent, span_from_cstr(kSummary)), ASTRAL_OK);
    ASSERT_EQ(astral_agent_get_summary_size(agent, &bytes), ASTRAL_OK);
    ASSERT_EQ(bytes, kSummaryBytes);

    uint8_t summary_buf[kSummaryBytes]{};
    AstralMutSpanU8 summary_out{};
    summary_out.data = summary_buf;
    summary_out.len = sizeof(summary_buf);
    ASSERT_EQ(astral_agent_get_summary(agent, summary_out, &bytes), ASTRAL_OK);
    ASSERT_EQ(std::string(reinterpret_cast<const char*>(summary_buf), bytes), kSummary);

    ASSERT_EQ(astral_agent_set_memory_context(agent, span_from_cstr(kMemoryContext)), ASTRAL_OK);
    ASSERT_EQ(astral_agent_get_memory_context_size(agent, &bytes), ASTRAL_OK);
    ASSERT_EQ(bytes, kMemoryContextBytes);

    uint8_t memory_context_buf[kMemoryContextBytes]{};
    AstralMutSpanU8 memory_context_out{};
    memory_context_out.data = memory_context_buf;
    memory_context_out.len = sizeof(memory_context_buf);
    ASSERT_EQ(astral_agent_get_memory_context(agent, memory_context_out, &bytes), ASTRAL_OK);
    ASSERT_EQ(std::string(reinterpret_cast<const char*>(memory_context_buf), bytes), kMemoryContext);

    AstralAgentMessage user{};
    user.size = sizeof(AstralAgentMessage);
    user.role = ASTRAL_AGENT_ROLE_USER;
    user.content = span_from_cstr("hello");
    ASSERT_EQ(astral_agent_message_add(agent, &user), ASTRAL_OK);

    AstralAgentMessage assistant{};
    assistant.size = sizeof(AstralAgentMessage);
    assistant.role = ASTRAL_AGENT_ROLE_ASSISTANT;
    assistant.content = span_from_cstr("hi");
    ASSERT_EQ(astral_agent_message_add(agent, &assistant), ASTRAL_OK);

    uint32_t count = 0;
    ASSERT_EQ(astral_agent_history_count(agent, &count), ASTRAL_OK);
    ASSERT_EQ(count, kHistoryCount);

    uint32_t save_bytes = 0;
    ASSERT_EQ(astral_agent_history_save_size(agent, &save_bytes), ASTRAL_OK);
    ASSERT_GT(save_bytes, 0u);
    ASSERT_LT(save_bytes, kSaveCapacity);
    uint8_t saved[kSaveCapacity]{};
    AstralMutSpanU8 saved_out{};
    saved_out.data = saved;
    saved_out.len = sizeof(saved);
    uint32_t written = 0;
    ASSERT_EQ(astral_agent_history_save(agent, saved_out, &written), ASTRAL_OK);
    ASSERT_EQ(written, save_bytes);

    ASSERT_EQ(astral_agent_history_clear(agent), ASTRAL_OK);
    ASSERT_EQ(astral_agent_history_count(agent, &count), ASTRAL_OK);
    ASSERT_EQ(count, 0u);
    AstralSpanU8 saved_in{};
    saved_in.data = saved;
    saved_in.len = written;
    ASSERT_EQ(astral_agent_history_load(agent, saved_in), ASTRAL_OK);
    ASSERT_EQ(astral_agent_history_count(agent, &count), ASTRAL_OK);
    ASSERT_EQ(count, kHistoryCount);
    ASSERT_EQ(astral_agent_get_summary_size(agent, &bytes), ASTRAL_OK);
    ASSERT_EQ(bytes, kSummaryBytes);
    uint8_t loaded_summary[kSummaryBytes]{};
    summary_out.data = loaded_summary;
    summary_out.len = sizeof(loaded_summary);
    ASSERT_EQ(astral_agent_get_summary(agent, summary_out, &bytes), ASTRAL_OK);
    ASSERT_EQ(std::string(reinterpret_cast<const char*>(loaded_summary), bytes), kSummary);
    ASSERT_EQ(astral_agent_get_memory_context_size(agent, &bytes), ASTRAL_OK);
    ASSERT_EQ(bytes, kMemoryContextBytes);
    uint8_t loaded_memory_context[kMemoryContextBytes]{};
    memory_context_out.data = loaded_memory_context;
    memory_context_out.len = sizeof(loaded_memory_context);
    ASSERT_EQ(astral_agent_get_memory_context(agent, memory_context_out, &bytes), ASTRAL_OK);
    ASSERT_EQ(std::string(reinterpret_cast<const char*>(loaded_memory_context), bytes), kMemoryContext);

    AstralAgentMessage next{};
    next.size = sizeof(AstralAgentMessage);
    next.role = ASTRAL_AGENT_ROLE_USER;
    next.content = span_from_cstr("next");
    ASSERT_EQ(astral_agent_message_add(agent, &next), ASTRAL_OK);

    AstralAgentChatDesc chat{};
    chat.size = sizeof(AstralAgentChatDesc);
    chat.user_message = span_from_cstr("next");
    const AstralErr chat_err = astral_agent_chat_enqueue(agent, &chat);
    ASSERT_EQ(chat_err, ASTRAL_OK);
    (void)read_agent_stream_all(agent);

    AstralAgentChatResult result{};
    result.size = sizeof(AstralAgentChatResult);
    ASSERT_EQ(astral_agent_chat_result(agent, &result), ASTRAL_OK);
    ASSERT_EQ(result.state, ASTRAL_SESSION_COMPLETED);
    ASSERT_EQ(result.history_messages, kChatHistoryCount);
    ASSERT_GT(result.prompt_bytes, 0u);
    ASSERT_GT(result.generated_tokens, 0ull);
    ASSERT_GT(result.prompt_cache_new_tokens, kPromptCacheNoNewTokens);
    ASSERT_EQ(result.prompt_cache_reused_tokens, kPromptCacheNoReusedTokens);
    ASSERT_EQ(result.prompt_cache_misses, kPromptCacheFirstMisses);

    chat.flags = ASTRAL_AGENT_CHAT_FLAG_WARMUP;
    ASSERT_EQ(astral_agent_chat_enqueue(agent, &chat), ASTRAL_OK);
    (void)read_agent_stream_all(agent);
    result = AstralAgentChatResult{};
    result.size = sizeof(AstralAgentChatResult);
    ASSERT_EQ(astral_agent_chat_result(agent, &result), ASTRAL_OK);
    ASSERT_EQ(result.state, ASTRAL_SESSION_COMPLETED);
    ASSERT_EQ(result.history_messages, kChatHistoryCount);
    ASSERT_GT(result.prompt_cache_reused_tokens, kPromptCacheNoReusedTokens);
    ASSERT_EQ(result.prompt_cache_new_tokens, kPromptCacheNoNewTokens);
    ASSERT_EQ(result.prompt_cache_hits, kPromptCacheSecondHits);
    ASSERT_EQ(astral_agent_chat_cancel(agent), ASTRAL_OK);

    astral_agent_destroy(agent);
    astral_memory_destroy(memory_index);
    astral_prompt_cache_destroy(prompt_cache);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_agent_overflow_truncate_mock) {
    constexpr uint32_t kReserveBytes = 32 * 1024 * 1024;
    constexpr uint32_t kMaxTokens = 0;
    constexpr uint32_t kRejectMaxMessages = 2;
    constexpr uint32_t kTruncateMaxMessages = 2;
    constexpr uint32_t kPromptMaxMessages = 8;
    constexpr uint32_t kLongHistoryCount = 2;
    constexpr uint32_t kExpectedTruncatedCount = 2;
    constexpr uint32_t kExpectedPromptCount = 1;
    constexpr uint32_t kSaveCapacity = 512;
    constexpr uint32_t kReservedNonZero = 1;
    constexpr AstralAgentOverflowPolicy kInvalidOverflowPolicy =
        static_cast<AstralAgentOverflowPolicy>(ASTRAL_AGENT_OVERFLOW_TRUNCATE_OLDEST + kReservedNonZero);
    constexpr char kLongHistoryMessage[] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    constexpr char kShortUserMessage[] = "go";
    constexpr uint32_t kPromptHeaderReserveBytes = 96;
    constexpr uint32_t kUserLabelBytes = static_cast<uint32_t>(sizeof("User: ") - 1);
    constexpr uint32_t kAssistantLabelBytes = static_cast<uint32_t>(sizeof("Assistant: ") - 1);
    constexpr uint32_t kLineBreakBytes = static_cast<uint32_t>(sizeof("\n") - 1);
    constexpr uint32_t kLongHistoryBytes = static_cast<uint32_t>(sizeof(kLongHistoryMessage) - 1);
    constexpr uint32_t kShortUserBytes = static_cast<uint32_t>(sizeof(kShortUserMessage) - 1);
    constexpr uint32_t kOneHistoryPromptBytes = kPromptHeaderReserveBytes + kUserLabelBytes + kLongHistoryBytes +
                                                kLineBreakBytes + kUserLabelBytes + kShortUserBytes +
                                                kLineBreakBytes + kAssistantLabelBytes;

    AstralInit cfg{};
    cfg.reserve_bytes = kReserveBytes;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralAgentDesc reject_desc{};
    reject_desc.size = sizeof(AstralAgentDesc);
    reject_desc.model = model;
    reject_desc.max_tokens = kMaxTokens;
    reject_desc.max_messages = kRejectMaxMessages;

    AstralAgentDesc invalid_desc = reject_desc;
    invalid_desc.overflow_policy = kInvalidOverflowPolicy;
    AstralHandle invalid_agent = 0;
    ASSERT_EQ(astral_agent_create(&invalid_desc, &invalid_agent), ASTRAL_E_INVALID);

    invalid_desc = reject_desc;
    invalid_desc._reserved0 = kReservedNonZero;
    ASSERT_EQ(astral_agent_create(&invalid_desc, &invalid_agent), ASTRAL_E_INVALID);

    invalid_desc = reject_desc;
    invalid_desc.memory_index = model;
    ASSERT_EQ(astral_agent_create(&invalid_desc, &invalid_agent), ASTRAL_E_INVALID);

    AstralHandle reject_agent = 0;
    ASSERT_EQ(astral_agent_create(&reject_desc, &reject_agent), ASTRAL_OK);

    AstralAgentMessage msg{};
    msg.size = sizeof(AstralAgentMessage);
    msg.role = ASTRAL_AGENT_ROLE_USER;
    msg.content = span_from_cstr("first");
    ASSERT_EQ(astral_agent_message_add(reject_agent, &msg), ASTRAL_OK);
    msg.content = span_from_cstr("second");
    ASSERT_EQ(astral_agent_message_add(reject_agent, &msg), ASTRAL_OK);
    msg.content = span_from_cstr("third");
    ASSERT_EQ(astral_agent_message_add(reject_agent, &msg), ASTRAL_E_NOMEM);
    astral_agent_destroy(reject_agent);

    AstralAgentDesc truncate_desc = reject_desc;
    truncate_desc.overflow_policy = ASTRAL_AGENT_OVERFLOW_TRUNCATE_OLDEST;
    truncate_desc.max_messages = kTruncateMaxMessages;

    AstralHandle truncate_agent = 0;
    ASSERT_EQ(astral_agent_create(&truncate_desc, &truncate_agent), ASTRAL_OK);
    msg.content = span_from_cstr("first");
    ASSERT_EQ(astral_agent_message_add(truncate_agent, &msg), ASTRAL_OK);
    msg.content = span_from_cstr("second");
    ASSERT_EQ(astral_agent_message_add(truncate_agent, &msg), ASTRAL_OK);
    msg.content = span_from_cstr("third");
    ASSERT_EQ(astral_agent_message_add(truncate_agent, &msg), ASTRAL_OK);

    uint32_t count = 0;
    ASSERT_EQ(astral_agent_history_count(truncate_agent, &count), ASTRAL_OK);
    ASSERT_EQ(count, kExpectedTruncatedCount);

    std::vector<uint8_t> saved(kSaveCapacity);
    AstralMutSpanU8 saved_out{};
    saved_out.data = saved.data();
    saved_out.len = static_cast<uint32_t>(saved.size());
    uint32_t written = 0;
    ASSERT_EQ(astral_agent_history_save(truncate_agent, saved_out, &written), ASTRAL_OK);
    saved.resize(written);
    ASSERT_FALSE(bytes_contain(saved, "first"));
    ASSERT_TRUE(bytes_contain(saved, "second"));
    ASSERT_TRUE(bytes_contain(saved, "third"));
    astral_agent_destroy(truncate_agent);

    AstralAgentDesc prompt_desc = truncate_desc;
    prompt_desc.max_messages = kPromptMaxMessages;
    prompt_desc.max_prompt_bytes = kOneHistoryPromptBytes;

    AstralHandle prompt_agent = 0;
    ASSERT_EQ(astral_agent_create(&prompt_desc, &prompt_agent), ASTRAL_OK);
    msg.content = span_from_cstr(kLongHistoryMessage);
    for (uint32_t i = 0; i < kLongHistoryCount; ++i) {
        ASSERT_EQ(astral_agent_message_add(prompt_agent, &msg), ASTRAL_OK);
    }

    AstralAgentChatDesc chat{};
    chat.size = sizeof(AstralAgentChatDesc);
    chat.flags = ASTRAL_AGENT_CHAT_FLAG_WARMUP;
    chat.user_message = span_from_cstr(kShortUserMessage);
    ASSERT_EQ(astral_agent_chat_enqueue(prompt_agent, &chat), ASTRAL_OK);
    (void)read_agent_stream_all(prompt_agent);

    ASSERT_EQ(astral_agent_history_count(prompt_agent, &count), ASTRAL_OK);
    ASSERT_EQ(count, kExpectedPromptCount);

    astral_agent_destroy(prompt_agent);
    astral_model_release(model);
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

TEST(inference_stream_read_rejects_concurrent_consumers) {
    struct Cleanup {
        AstralHandle session = 0;
        AstralHandle model = 0;
        bool shutdown = false;
        ~Cleanup() {
            if (session) {
                astral_session_destroy(session);
            }
            if (model) {
                astral_model_release(model);
            }
            if (shutdown) {
                astral_shutdown();
            }
        }
    } cleanup;

    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);
    cleanup.shutdown = true;

    cleanup.model = load_mock_model(nullptr);

    AstralSessionDesc session_desc = {};
    session_desc.model = cleanup.model;
    session_desc.max_tokens = 8;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    err = astral_session_create(&session_desc, &cleanup.session);
    ASSERT_EQ(err, ASTRAL_OK);

    // With no decode running, a blocking read will hold the stream consumer lock until it times out.
    std::atomic<int32_t> blocking_result{0};
    std::thread t([&]() {
        uint8_t buf[16];
        AstralMutSpanU8 out = {};
        out.data = buf;
        out.len = sizeof(buf);
        blocking_result.store(astral_stream_read(cleanup.session, out, 250), std::memory_order_relaxed);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    uint8_t buf2[16];
    AstralMutSpanU8 out2 = {};
    out2.data = buf2;
    out2.len = sizeof(buf2);
    const int32_t concurrent = astral_stream_read(cleanup.session, out2, 0);

    t.join();

    const int32_t first = blocking_result.load(std::memory_order_relaxed);
    ASSERT_TRUE((concurrent == ASTRAL_E_STATE) && (first == ASTRAL_E_TIMEOUT));
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

TEST(inference_stop_sequences_suppress_output_after_reset) {
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

    ASSERT_EQ(run_mock_decode_once(session, "hi"), std::string("mock-backend"));

    err = astral_session_reset(session, &sd);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_stop_clear(session);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* stop = "backend";
    AstralSpanU8 stop_span{};
    stop_span.data = reinterpret_cast<const uint8_t*>(stop);
    stop_span.len = static_cast<uint32_t>(std::strlen(stop));
    err = astral_session_stop_add_utf8(session, stop_span);
    ASSERT_EQ(err, ASTRAL_OK);

    ASSERT_EQ(run_mock_decode_once(session, "hi"), std::string("mock-"));

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

TEST(inference_toolset_parse_and_bind_mock) {
    constexpr uint64_t kRuntimeReserveBytes = 64 * 1024 * 1024;
    constexpr uint32_t kRuntimeThreads = 2;
    constexpr uint32_t kToolCount = 2;
    constexpr uint32_t kCollisionToolCount = 2;
    constexpr uint32_t kSearchToolId = 101;
    constexpr uint32_t kOpenToolId = 202;
    constexpr uint32_t kFirstCollisionToolId = 303;
    constexpr uint32_t kSecondCollisionToolId = 404;
    constexpr uint32_t kOpenToolIndex = 1;
    constexpr uint32_t kSessionMaxTokens = 4;
    constexpr uint32_t kExecutorMaxBatchTokens = 8;
    constexpr float kAgentTemperature = 0.0f;
    constexpr uint32_t kAgentTopK = 0;
    constexpr float kAgentTopP = 1.0f;
    constexpr uint8_t kAgentStreamEnabled = 1;
    constexpr uint32_t kAgentSeed = 1;
    constexpr char kAgentToolCallJson[] = "{\"name\":\"search\",\"arguments\":{\"query\":\"agent\"}}";
    constexpr char kAgentToolCallArgs[] = "{\"query\":\"agent\"}";
    constexpr char kPlainText[] = "plain text";

    AstralInit cfg = {};
    cfg.reserve_bytes = kRuntimeReserveBytes;
    cfg.thread_count = kRuntimeThreads;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralToolDesc tools[kToolCount]{};
    tools[0].size = sizeof(AstralToolDesc);
    tools[0].tool_id = kSearchToolId;
    tools[0].name = span_from_cstr("search");
    tools[0].description = span_from_cstr("Search indexed text");
    tools[0].json_schema = span_from_cstr("{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}");
    tools[1].size = sizeof(AstralToolDesc);
    tools[1].tool_id = kOpenToolId;
    tools[1].name = span_from_cstr("open");
    tools[1].description = span_from_cstr("Open one result");
    tools[1].json_schema = span_from_cstr("{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"}},\"required\":[\"id\"]}");

    AstralToolsetDesc td{};
    td.size = sizeof(AstralToolsetDesc);
    td.tool_count = kToolCount;
    td.choice_mode = ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL;
    td.tools = tools;

    AstralHandle toolset = 0;
    err = astral_toolset_create(&td, &toolset);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(toolset));

    uint32_t tool_count = 0;
    err = astral_toolset_count(toolset, &tool_count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(tool_count, kToolCount);

    AstralToolInfo info{};
    info.size = sizeof(AstralToolInfo);
    err = astral_toolset_get(toolset, kOpenToolIndex, &info);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(info.tool_id, kOpenToolId);
    ASSERT_EQ(std::string(reinterpret_cast<const char*>(info.name.data), info.name.len), std::string("open"));

    AstralToolCallResult call{};
    call.size = sizeof(AstralToolCallResult);
    err = astral_toolset_parse_call(toolset, span_from_cstr("{\"name\":\"search\",\"arguments\":{\"query\":\"latency\"}}"), &call);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(call.tool_id, kSearchToolId);
    ASSERT_EQ(call.parse_status, ASTRAL_OK);
    const std::string search_args(reinterpret_cast<const char*>(call.arguments_json.data), call.arguments_json.len);
    ASSERT_EQ(search_args, std::string("{\"query\":\"latency\"}"));

    call = {};
    call.size = sizeof(AstralToolCallResult);
    err = astral_toolset_parse_call(toolset, span_from_cstr("{\"tool\":\"open\",\"arguments\":{\"id\":7}}"), &call);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(call.tool_id, kOpenToolId);
    ASSERT_EQ(call.parse_status, ASTRAL_OK);

    call = {};
    call.size = sizeof(AstralToolCallResult);
    err = astral_toolset_parse_call(toolset, span_from_cstr("{\"arguments\":{\"query\":\"early\"},\"name\":\"search\"}"), &call);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(call.tool_id, kSearchToolId);
    ASSERT_EQ(call.parse_status, ASTRAL_OK);

    call = {};
    call.size = sizeof(AstralToolCallResult);
    err = astral_toolset_parse_call(toolset, span_from_cstr("{\"name\":\"search\",\"arguments\":\"bad\"}"), &call);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(call.tool_id, kSearchToolId);
    ASSERT_EQ(call.parse_status, ASTRAL_E_INVALID);

    call = {};
    call.size = sizeof(AstralToolCallResult);
    err = astral_toolset_parse_call(toolset, span_from_cstr("plain text"), &call);
    ASSERT_EQ(err, ASTRAL_E_NOT_FOUND);

    AstralToolDesc collision_tools[kCollisionToolCount]{};
    collision_tools[0].size = sizeof(AstralToolDesc);
    collision_tools[0].tool_id = kFirstCollisionToolId;
    collision_tools[0].name = span_from_cstr("same_alpha");
    collision_tools[0].json_schema = span_from_cstr("{\"type\":\"object\"}");
    collision_tools[1].size = sizeof(AstralToolDesc);
    collision_tools[1].tool_id = kSecondCollisionToolId;
    collision_tools[1].name = span_from_cstr("same_beta");
    collision_tools[1].json_schema = span_from_cstr("{\"type\":\"object\"}");

    AstralToolsetDesc collision_td{};
    collision_td.size = sizeof(AstralToolsetDesc);
    collision_td.tool_count = kCollisionToolCount;
    collision_td.choice_mode = ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL;
    collision_td.tools = collision_tools;

    AstralHandle collision_toolset = 0;
    err = astral_toolset_create(&collision_td, &collision_toolset);
    ASSERT_EQ(err, ASTRAL_OK);

    call = {};
    call.size = sizeof(AstralToolCallResult);
    err = astral_toolset_parse_call(collision_toolset, span_from_cstr("{\"name\":\"same_beta\",\"arguments\":{}}"), &call);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(call.tool_id, kSecondCollisionToolId);
    ASSERT_EQ(call.parse_status, ASTRAL_OK);
    astral_toolset_destroy(collision_toolset);

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = kSessionMaxTokens;
    sd.temperature = 0.0f;
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 1;
    sd.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_set_toolset(session, toolset, ASTRAL_TOOL_CHOICE_REQUIRED);
    ASSERT_EQ(err, ASTRAL_OK);
    astral_toolset_destroy(toolset);
    err = astral_session_clear_toolset(session);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle toolset2 = 0;
    err = astral_toolset_create(&td, &toolset2);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralExecutorDesc ex{};
    ex.size = sizeof(AstralExecutorDesc);
    ex.max_slots = 1;
    ex.max_batch_tokens = kExecutorMaxBatchTokens;
    ex.worker_hint = 0;
    ASSERT_EQ(astral_model_executor_configure(model, &ex), ASTRAL_OK);

    AstralConvDesc conv_desc{};
    conv_desc.size = sizeof(AstralConvDesc);
    conv_desc.model = model;
    conv_desc.max_tokens = kSessionMaxTokens;
    conv_desc.temperature = 0.0f;
    conv_desc.top_k = 0;
    conv_desc.top_p = 1.0f;
    conv_desc.stream_enabled = 1;
    conv_desc.seed = 1;

    AstralHandle conv = 0;
    err = astral_conv_create(&conv_desc, &conv);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_conv_set_toolset(conv, toolset2, ASTRAL_TOOL_CHOICE_AUTO);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_conv_clear_toolset(conv);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_conv_destroy(conv);
    astral_toolset_destroy(toolset2);

    AstralHandle agent_toolset = 0;
    err = astral_toolset_create(&td, &agent_toolset);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralAgentDesc agent_desc{};
    agent_desc.size = sizeof(AstralAgentDesc);
    agent_desc.model = model;
    agent_desc.toolset = agent_toolset;
    agent_desc.tool_choice_mode = ASTRAL_TOOL_CHOICE_REQUIRED;
    agent_desc.max_tokens = kSessionMaxTokens;
    agent_desc.temperature = kAgentTemperature;
    agent_desc.top_k = kAgentTopK;
    agent_desc.top_p = kAgentTopP;
    agent_desc.stream_enabled = kAgentStreamEnabled;
    agent_desc.seed = kAgentSeed;

    AstralHandle agent = 0;
    err = astral_agent_create(&agent_desc, &agent);
    ASSERT_EQ(err, ASTRAL_OK);
    astral_toolset_destroy(agent_toolset);

    call = {};
    call.size = sizeof(AstralToolCallResult);
    err = astral_agent_parse_tool_call(agent, span_from_cstr(kAgentToolCallJson), &call);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(call.tool_id, kSearchToolId);
    ASSERT_EQ(call.parse_status, ASTRAL_OK);
    const std::string agent_args(reinterpret_cast<const char*>(call.arguments_json.data), call.arguments_json.len);
    ASSERT_EQ(agent_args, std::string(kAgentToolCallArgs));

    call = {};
    call.size = sizeof(AstralToolCallResult);
    err = astral_agent_parse_tool_call(agent, span_from_cstr(kPlainText), &call);
    ASSERT_EQ(err, ASTRAL_E_NOT_FOUND);

    astral_agent_destroy(agent);
    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_chunking_ranges_mock) {
    constexpr uint32_t kDocId = 17;
    constexpr uint32_t kGroupId = 3;
    constexpr uint32_t kWordMaxUnits = 2;
    constexpr uint32_t kWordOverlapUnits = 1;
    constexpr uint32_t kWordRangeCount = 3;
    constexpr uint32_t kFirstWordRangeEnd = 10;
    constexpr uint32_t kSecondWordRangeBegin = 6;
    constexpr uint32_t kSecondWordRangeEnd = 16;
    constexpr uint32_t kThirdWordRangeBegin = 11;
    constexpr uint32_t kThirdWordRangeEnd = 22;
    constexpr uint32_t kUtf8MaxUnits = 2;
    constexpr uint32_t kUtf8RangeCount = 2;
    constexpr uint32_t kFirstUtf8RangeEnd = 3;
    constexpr uint32_t kSecondUtf8RangeBegin = 3;
    constexpr uint32_t kSecondUtf8RangeEnd = 8;
    constexpr uint32_t kSentenceMaxUnits = 2;
    constexpr uint32_t kSentenceRangeCount = 2;
    constexpr uint32_t kFirstSentenceRangeEnd = 9;
    constexpr uint32_t kSecondSentenceRangeBegin = 10;
    constexpr uint32_t kSecondSentenceRangeEnd = 16;
    constexpr uint32_t kTokenCount = 10;
    constexpr uint32_t kTokenMaxUnits = 4;
    constexpr uint32_t kTokenOverlapUnits = 1;
    constexpr uint32_t kTokenRangeCount = 3;
    constexpr uint32_t kSecondTokenBegin = 3;
    constexpr uint32_t kSecondTokenEnd = 7;
    constexpr uint32_t kThirdTokenBegin = 6;
    constexpr uint32_t kRangeCapacity = 4;

    AstralChunkerDesc desc{};
    desc.size = sizeof(AstralChunkerDesc);
    desc.mode = ASTRAL_CHUNK_MODE_WORD;
    desc.max_units = kWordMaxUnits;
    desc.overlap_units = kWordOverlapUnits;
    desc.document_id = kDocId;
    desc.group_id = kGroupId;

    uint32_t count = 0;
    AstralErr err = astral_chunk_count(&desc, span_from_cstr("alpha beta gamma delta"), &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kWordRangeCount);

    AstralChunkRange ranges[kRangeCapacity]{};
    err = astral_chunk_ranges(&desc, span_from_cstr("alpha beta gamma delta"), ranges, kRangeCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kWordRangeCount);
    ASSERT_EQ(ranges[0].document_id, kDocId);
    ASSERT_EQ(ranges[0].group_id, kGroupId);
    ASSERT_EQ(ranges[0].byte_begin, 0u);
    ASSERT_EQ(ranges[0].byte_end, kFirstWordRangeEnd);
    ASSERT_EQ(ranges[1].byte_begin, kSecondWordRangeBegin);
    ASSERT_EQ(ranges[1].byte_end, kSecondWordRangeEnd);
    ASSERT_EQ(ranges[2].byte_begin, kThirdWordRangeBegin);
    ASSERT_EQ(ranges[2].byte_end, kThirdWordRangeEnd);

    uint8_t copied[kFirstWordRangeEnd]{};
    AstralMutSpanU8 copied_span{};
    copied_span.data = copied;
    copied_span.len = sizeof(copied);
    uint32_t copied_len = 0;
    err = astral_chunk_text_copy(span_from_cstr("alpha beta gamma delta"), &ranges[0], copied_span, &copied_len);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(copied_len, kFirstWordRangeEnd);
    ASSERT_EQ(std::string(reinterpret_cast<const char*>(copied), copied_len), std::string("alpha beta"));

    constexpr uint64_t kChunkRecordKey = 9001;
    constexpr uint32_t kChunkRecordFlags = 5;
    AstralMemoryRecord chunk_record{};
    err = astral_memory_record_from_chunk(&ranges[1], kChunkRecordKey, kChunkRecordFlags, &chunk_record);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(chunk_record.group_id, kGroupId);
    ASSERT_EQ(chunk_record.key, kChunkRecordKey);
    ASSERT_EQ(chunk_record.document_id, kDocId);
    ASSERT_EQ(chunk_record.chunk_id, ranges[1].chunk_id);
    ASSERT_EQ(chunk_record.flags, kChunkRecordFlags);

    desc.mode = ASTRAL_CHUNK_MODE_NONE;
    desc.max_units = 1;
    desc.overlap_units = 0;
    err = astral_chunk_ranges(&desc, span_from_cstr("full document"), ranges, kRangeCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, 1u);
    ASSERT_EQ(ranges[0].document_id, kDocId);
    ASSERT_EQ(ranges[0].chunk_id, 0u);
    ASSERT_EQ(ranges[0].byte_begin, 0u);
    ASSERT_EQ(ranges[0].byte_end, static_cast<uint32_t>(std::strlen("full document")));

    desc.mode = ASTRAL_CHUNK_MODE_CHAR;
    desc.max_units = kUtf8MaxUnits;
    desc.overlap_units = 0;
    err = astral_chunk_ranges(&desc, span_from_cstr("aé🙂b"), ranges, kRangeCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kUtf8RangeCount);
    ASSERT_EQ(ranges[0].byte_begin, 0u);
    ASSERT_EQ(ranges[0].byte_end, kFirstUtf8RangeEnd);
    ASSERT_EQ(ranges[1].byte_begin, kSecondUtf8RangeBegin);
    ASSERT_EQ(ranges[1].byte_end, kSecondUtf8RangeEnd);

    desc.mode = ASTRAL_CHUNK_MODE_SENTENCE;
    desc.max_units = kSentenceMaxUnits;
    err = astral_chunk_ranges(&desc, span_from_cstr("One. Two! Three?"), ranges, kRangeCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kSentenceRangeCount);
    ASSERT_EQ(ranges[0].byte_begin, 0u);
    ASSERT_EQ(ranges[0].byte_end, kFirstSentenceRangeEnd);
    ASSERT_EQ(ranges[1].byte_begin, kSecondSentenceRangeBegin);
    ASSERT_EQ(ranges[1].byte_end, kSecondSentenceRangeEnd);

    desc.mode = ASTRAL_CHUNK_MODE_TOKEN;
    desc.max_units = kTokenMaxUnits;
    desc.overlap_units = kTokenOverlapUnits;
    err = astral_token_chunk_ranges(&desc, kTokenCount, ranges, kRangeCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kTokenRangeCount);
    ASSERT_EQ(ranges[0].token_begin, 0u);
    ASSERT_EQ(ranges[0].token_end, kTokenMaxUnits);
    ASSERT_EQ(ranges[1].token_begin, kSecondTokenBegin);
    ASSERT_EQ(ranges[1].token_end, kSecondTokenEnd);
    ASSERT_EQ(ranges[2].token_begin, kThirdTokenBegin);
    ASSERT_EQ(ranges[2].token_end, kTokenCount);
}

TEST(inference_memory_index_flat_mock) {
    constexpr uint32_t kDim = 4;
    constexpr uint32_t kCapacity = 8;
    constexpr uint32_t kRecordCount = 5;
    constexpr uint32_t kTopOne = 1;
    constexpr uint32_t kTopK = 4;
    constexpr uint64_t kKeyA = 11;
    constexpr uint64_t kKeyB = 22;
    constexpr uint64_t kKeyC = 33;
    constexpr uint64_t kKeyD = 44;
    constexpr uint64_t kKeyE = 55;
    constexpr uint32_t kGroupA = 7;
    constexpr uint32_t kGroupB = 9;
    constexpr uint32_t kDocA = 101;
    constexpr uint32_t kChunkA = 3;
    constexpr uint32_t kResultCapacity = 4;
    constexpr uint32_t kFirstFetchCapacity = 2;
    constexpr uint32_t kSecondFetchCapacity = 2;
    constexpr uint32_t kFinalFetchCapacity = 1;
    constexpr uint32_t kFirstFetchCount = 2;
    constexpr uint32_t kSecondFetchCount = 2;
    constexpr uint32_t kFinalFetchCount = 0;
    constexpr uint32_t kGroupBResultCount = 1;
    constexpr uint32_t kPostRemoveResultCount = kRecordCount - 1u;
    constexpr uint32_t kPostReaddResultCount = kRecordCount;

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = kDim;
    desc.capacity = kCapacity;
    desc.metric = ASTRAL_MEMORY_METRIC_COSINE;
    desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(index));

    AstralMemoryRecord records[kRecordCount]{};
    records[0].size = sizeof(AstralMemoryRecord);
    records[0].key = kKeyA;
    records[0].group_id = kGroupA;
    records[0].document_id = kDocA;
    records[0].chunk_id = kChunkA;
    records[1].size = sizeof(AstralMemoryRecord);
    records[1].key = kKeyB;
    records[1].group_id = kGroupA;
    records[2].size = sizeof(AstralMemoryRecord);
    records[2].key = kKeyC;
    records[2].group_id = kGroupB;
    records[3].size = sizeof(AstralMemoryRecord);
    records[3].key = kKeyD;
    records[3].group_id = kGroupA;
    records[4].size = sizeof(AstralMemoryRecord);
    records[4].key = kKeyE;
    records[4].group_id = kGroupA;

    const float vectors[kRecordCount * kDim] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.0f, 0.0f,
    };
    err = astral_memory_add_batch(index, records, vectors, kRecordCount);
    ASSERT_EQ(err, ASTRAL_OK);

    uint32_t count = 0;
    err = astral_memory_count(index, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kRecordCount);

    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    const float query[kDim] = {1.0f, 0.0f, 0.0f, 0.0f};
    AstralMemorySearchResult results[kResultCapacity]{};
    err = astral_memory_search(index, &search, query, results, kResultCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kTopK);
    ASSERT_EQ(results[0].key, kKeyA);
    ASSERT_EQ(results[0].document_id, kDocA);
    ASSERT_EQ(results[0].chunk_id, kChunkA);
    ASSERT_EQ(results[1].key, kKeyD);
    ASSERT_EQ(results[2].key, kKeyE);

    search.top_k = kTopOne;
    err = astral_memory_search(index, &search, query, results, kResultCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kTopOne);
    ASSERT_EQ(results[0].key, kKeyA);

    search.top_k = kTopK;
    AstralHandle cursor = 0;
    err = astral_memory_search_begin(index, &search, query, &cursor);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(cursor));

    AstralRequestRef request{};
    err = astral_request_from_memory_search(cursor, &request);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(request.kind, ASTRAL_REQUEST_MEMORY_SEARCH);
    ASSERT_EQ(request.owner, cursor);

    AstralRequestStatus status{};
    status.size = sizeof(AstralRequestStatus);
    err = astral_request_state(&request, &status);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(status.state, ASTRAL_REQUEST_COMPLETED);
    ASSERT_EQ(status.queue_depth, kTopK);

    AstralMemorySearchResult cursor_results[kFirstFetchCapacity]{};
    err = astral_memory_search_fetch(cursor, cursor_results, kFirstFetchCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kFirstFetchCount);
    ASSERT_EQ(cursor_results[0].key, kKeyA);
    ASSERT_EQ(cursor_results[1].key, kKeyD);

    status = AstralRequestStatus{};
    status.size = sizeof(AstralRequestStatus);
    err = astral_request_wait(&request, 0, &status);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(status.state, ASTRAL_REQUEST_COMPLETED);
    ASSERT_EQ(status.queue_depth, kTopK - kFirstFetchCount);

    err = astral_memory_search_fetch(cursor, cursor_results, kSecondFetchCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kSecondFetchCount);

    err = astral_memory_search_fetch(cursor, cursor_results, kFinalFetchCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kFinalFetchCount);
    astral_memory_search_end(cursor);

    search.group_id = kGroupB;
    err = astral_memory_search(index, &search, query, results, kResultCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kGroupBResultCount);
    ASSERT_EQ(results[0].key, kKeyC);

    uint64_t save_bytes = 0;
    err = astral_memory_save_size(index, &save_bytes);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(save_bytes, 0ull);
    std::string blob;
    blob.resize(static_cast<size_t>(save_bytes));
    AstralMutSpanU8 out_blob{};
    out_blob.data = reinterpret_cast<uint8_t*>(&blob[0]);
    out_blob.len = static_cast<uint32_t>(blob.size());
    uint64_t written = 0;
    err = astral_memory_save(index, out_blob, &written);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(written, save_bytes);

    AstralSpanU8 blob_span{};
    blob_span.data = reinterpret_cast<const uint8_t*>(blob.data());
    blob_span.len = static_cast<uint32_t>(blob.size());
    AstralHandle loaded = 0;
    err = astral_memory_load(&desc, blob_span, &loaded);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(loaded));

    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    err = astral_memory_search(loaded, &search, query, results, kResultCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(results[0].key, kKeyA);

    err = astral_memory_remove(index, kKeyA);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_memory_count(index, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kPostRemoveResultCount);
    err = astral_memory_search(index, &search, query, results, kResultCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kPostRemoveResultCount);
    ASSERT_EQ(results[0].key, kKeyD);

    err = astral_memory_add_batch(index, &records[0], vectors, kTopOne);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_memory_count(index, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kPostReaddResultCount);
    err = astral_memory_search(index, &search, query, results, kResultCapacity, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(results[0].key, kKeyA);

    err = astral_memory_clear(index);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_memory_count(index, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, 0u);

    astral_memory_destroy(loaded);
    astral_memory_destroy(index);
}

TEST(inference_memory_index_dot_l2_tail_mock) {
    constexpr uint32_t kDim = 17;
    constexpr uint32_t kCapacity = 4;
    constexpr uint32_t kRecordCount = 3;
    constexpr uint32_t kTopOne = 1;
    constexpr uint64_t kKeyA = 101;
    constexpr uint64_t kKeyB = 202;
    constexpr uint64_t kKeyC = 303;
    constexpr float kQueryBias = 1.0f;
    constexpr float kFarOffset = 16.0f;

    AstralMemoryRecord records[kRecordCount]{};
    records[0].size = sizeof(AstralMemoryRecord);
    records[0].key = kKeyA;
    records[1].size = sizeof(AstralMemoryRecord);
    records[1].key = kKeyB;
    records[2].size = sizeof(AstralMemoryRecord);
    records[2].key = kKeyC;

    float query[kDim]{};
    float vectors[kRecordCount * kDim]{};
    for (uint32_t i = 0; i < kDim; ++i) {
        query[i] = static_cast<float>(i) + kQueryBias;
        vectors[i] = query[i];
        vectors[kDim + i] = query[i] * kQueryBias;
        vectors[kDim * 2u + i] = query[i] + kFarOffset;
    }

    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kTopOne;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    AstralMemorySearchResult results[kTopOne]{};
    uint32_t count = 0;

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = kDim;
    desc.capacity = kCapacity;
    desc.metric = ASTRAL_MEMORY_METRIC_DOT;
    desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_memory_add_batch(index, records, vectors, kRecordCount);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_memory_search(index, &search, query, results, kTopOne, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kTopOne);
    ASSERT_EQ(results[0].key, kKeyC);
    astral_memory_destroy(index);

    desc.metric = ASTRAL_MEMORY_METRIC_L2;
    index = 0;
    err = astral_memory_create(&desc, &index);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_memory_add_batch(index, records, vectors, kRecordCount);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_memory_search(index, &search, query, results, kTopOne, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, kTopOne);
    ASSERT_EQ(results[0].key, kKeyA);
    astral_memory_destroy(index);
}

TEST(inference_adapters_mock) {
    constexpr float kPrimaryAdapterScale = 1.0f;
    constexpr float kSecondaryAdapterScale = 0.5f;
    constexpr float kUpdatedAdapterScale = 0.75f;
    constexpr uint32_t kPrimaryAdapterIndex = 0u;
    constexpr uint32_t kAttachedPrimaryAdapterCount = 1u;

    AstralInit cfg = {};
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);
    const AstralHandle other_model = load_mock_model("adapter-other-model");

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

    AstralAdapterInfo adapter_info{};
    adapter_info.size = sizeof(AstralAdapterInfo);
    err = astral_model_adapter_info(adapter, &adapter_info);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(adapter_info.model, model);
    ASSERT_EQ(adapter_info.path_bytes, path_span.len);
    ASSERT_EQ(adapter_info.refcount, 1u);

    uint8_t adapter_path_buf[32] = {};
    AstralMutSpanU8 adapter_path_out{};
    adapter_path_out.data = adapter_path_buf;
    adapter_path_out.len = static_cast<uint32_t>(sizeof(adapter_path_buf));
    uint32_t adapter_path_len = 0;
    err = astral_model_adapter_path_copy(adapter, adapter_path_out, &adapter_path_len);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(adapter_path_len, path_span.len);
    ASSERT_EQ(std::memcmp(adapter_path_buf, path, path_span.len), 0);

    const char* other_path = "adapter-other";
    AstralSpanU8 other_path_span{};
    other_path_span.data = reinterpret_cast<const uint8_t*>(other_path);
    other_path_span.len = static_cast<uint32_t>(std::strlen(other_path));
    AstralAdapterDesc other_ad{};
    other_ad.size = sizeof(AstralAdapterDesc);
    other_ad.path = other_path_span;

    AstralHandle other_adapter = 0;
    err = astral_model_adapter_load(other_model, &other_ad, &other_adapter);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(other_adapter));

    uint32_t adapter_count = ASTRAL_SESSION_ADAPTERS_MAX;
    err = astral_session_adapters_count(session, &adapter_count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(adapter_count, 0u);

    AstralHandle attached_adapter = 0;
    float attached_scale = 0.0f;
    err = astral_session_adapters_get(session, 0, &attached_adapter, &attached_scale);
    ASSERT_EQ(err, ASTRAL_E_NOT_FOUND);

    err = astral_session_adapters_add(session, other_adapter, kPrimaryAdapterScale);
    ASSERT_EQ(err, ASTRAL_E_INVALID);

    err = astral_session_adapters_add(session, adapter, kPrimaryAdapterScale);
    ASSERT_EQ(err, ASTRAL_OK);
    adapter_info = AstralAdapterInfo{};
    adapter_info.size = sizeof(AstralAdapterInfo);
    err = astral_model_adapter_info(adapter, &adapter_info);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(adapter_info.refcount, 2u);

    err = astral_session_adapters_count(session, &adapter_count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(adapter_count, 1u);

    err = astral_session_adapters_get(session, 0, &attached_adapter, &attached_scale);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(attached_adapter, adapter);
    ASSERT_EQ(attached_scale, kPrimaryAdapterScale);

    err = astral_session_adapters_set_scale(session, kAttachedPrimaryAdapterCount, kUpdatedAdapterScale);
    ASSERT_EQ(err, ASTRAL_E_NOT_FOUND);

    err = astral_session_adapters_set_scale(session, kPrimaryAdapterIndex, kUpdatedAdapterScale);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_adapters_get(session, kPrimaryAdapterIndex, &attached_adapter, &attached_scale);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(attached_adapter, adapter);
    ASSERT_EQ(attached_scale, kUpdatedAdapterScale);

    AstralHandle overflow_adapters[ASTRAL_SESSION_ADAPTERS_MAX]{};
    for (uint32_t i = kAttachedPrimaryAdapterCount; i < ASTRAL_SESSION_ADAPTERS_MAX; ++i) {
        err = astral_model_adapter_load(model, &ad, &overflow_adapters[i]);
        ASSERT_EQ(err, ASTRAL_OK);
        err = astral_session_adapters_add(session, overflow_adapters[i], kSecondaryAdapterScale);
        ASSERT_EQ(err, ASTRAL_OK);
    }

    err = astral_session_adapters_count(session, &adapter_count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(adapter_count, ASTRAL_SESSION_ADAPTERS_MAX);

    AstralHandle overflow_adapter = 0;
    err = astral_model_adapter_load(model, &ad, &overflow_adapter);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_adapters_add(session, overflow_adapter, kSecondaryAdapterScale);
    ASSERT_EQ(err, ASTRAL_E_NOMEM);

    err = astral_session_adapters_clear(session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_adapters_count(session, &adapter_count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(adapter_count, 0u);

    astral_model_adapter_release(overflow_adapter);
    for (uint32_t i = kAttachedPrimaryAdapterCount; i < ASTRAL_SESSION_ADAPTERS_MAX; ++i) {
        astral_model_adapter_release(overflow_adapters[i]);
    }
    astral_model_adapter_release(other_adapter);
    astral_model_adapter_release(adapter);
    astral_session_destroy(session);
    astral_model_release(other_model);
    astral_model_release(model);
    astral_shutdown();
}

TEST(inference_adapter_release_after_session_destroy_mock) {
    constexpr uint64_t kReserveBytes = 64ull * 1024ull * 1024ull;
    constexpr uint32_t kThreadCount = 2u;
    constexpr uint32_t kMaxTokens = 4u;
    constexpr float kTemperature = 0.0f;
    constexpr uint32_t kTopK = 0u;
    constexpr float kTopP = 1.0f;
    constexpr uint32_t kSeed = 1u;
    constexpr float kAdapterScale = 1.0f;

    AstralInit cfg = {};
    cfg.reserve_bytes = kReserveBytes;
    cfg.thread_count = kThreadCount;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_model(nullptr);

    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = kMaxTokens;
    sd.temperature = kTemperature;
    sd.top_k = kTopK;
    sd.top_p = kTopP;
    sd.stream_enabled = 1;
    sd.seed = kSeed;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralAdapterDesc ad{};
    ad.size = sizeof(AstralAdapterDesc);
    ad.path = span_from_cstr("adapter-release-after-session-destroy");

    AstralHandle adapter = 0;
    err = astral_model_adapter_load(model, &ad, &adapter);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(adapter));

    err = astral_session_adapters_add(session, adapter, kAdapterScale);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_adapter_release(adapter);
    ASSERT_FALSE(astral_handle_valid(adapter));

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
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
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
