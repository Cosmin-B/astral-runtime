#include "agent.hpp"

#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"

#include "../concurrency/spsc_ring.hpp"
#include "../utils/trace.hpp"

#include "conversation_runtime.hpp"
#include "tooling.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace astral::inference {

namespace {

constexpr uint32_t kDefaultMaxMessages = 64;
constexpr uint32_t kBytesPerKiB = 1024;
constexpr uint32_t kDefaultMaxPromptKiB = 64;
constexpr uint32_t kDefaultMaxPromptBytes = kDefaultMaxPromptKiB * kBytesPerKiB;
constexpr uint32_t kInitialMessageCapacity = 8;
constexpr uint32_t kCapacityGrowthFactor = 2;
constexpr uint32_t kSaveMagic = 0x41414754u;
constexpr uint32_t kSaveVersionLegacy = 1;
constexpr uint32_t kSaveVersionSummary = 2;
constexpr uint32_t kSaveVersion = 3;
constexpr uint32_t kHeaderReserveBytes = 96;
constexpr uint32_t kPromptCacheGeneration = 1;
constexpr uint32_t kPromptCacheHitCount = 1;
constexpr uint32_t kPromptCacheMissCount = 1;
constexpr uint8_t kPromptFinalize = 1;
constexpr uint8_t kPromptAddSpecial = 1;
constexpr uint8_t kPromptParseSpecial = 0;
constexpr uint32_t kOneMessage = 1;
constexpr uint32_t kNoGeneratedCapture = 0;
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

constexpr char kSystemLabel[] = "System: ";
constexpr char kSummaryLabel[] = "Summary: ";
constexpr char kMemoryLabel[] = "Memory: ";
constexpr char kUserLabel[] = "User: ";
constexpr char kAssistantLabel[] = "Assistant: ";
constexpr char kToolLabel[] = "Tool: ";
constexpr char kLineBreak[] = "\n";

struct AgentMessageStorage {
    AstralAgentRole role;
    uint32_t offset;
    uint32_t len;
};

struct AgentHistoryArena {
    uint8_t* bytes;
    uint32_t len;
    uint32_t capacity;
};

struct AgentPromptWorkspace {
    uint8_t* scratch;
    uint32_t scratch_capacity;
    uint8_t* prefix;
    uint32_t prefix_capacity;
};

struct AgentGeneratedCapture {
    uint8_t* bytes;
    uint32_t len;
    uint32_t capacity;
    uint8_t truncated;
};

struct AgentPromptMetadata {
    uint32_t prefix_bytes;
    uint64_t prefix_hash;
    uint8_t valid;
};

struct AgentChatRuntime {
    AgentPromptWorkspace prompt;
    AgentGeneratedCapture generated;
    AgentPromptMetadata prefix;
    uint32_t prompt_bytes;
    uint32_t prompt_tokens;
    uint32_t prompt_cache_reused_tokens;
    uint32_t prompt_cache_new_tokens;
    uint32_t prompt_cache_hits;
    uint32_t prompt_cache_misses;
    AstralErr last_error;
};

struct AgentSaveHeaderV1 {
    uint32_t magic;
    uint32_t version;
    uint32_t system_len;
    uint32_t message_count;
};

struct AgentSaveHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t system_len;
    uint32_t summary_len;
    uint32_t memory_context_len;
    uint32_t message_count;
};

struct AgentSaveRecord {
    uint32_t role;
    uint32_t len;
};

inline bool role_valid(AstralAgentRole role) {
    return role == ASTRAL_AGENT_ROLE_SYSTEM || role == ASTRAL_AGENT_ROLE_USER ||
           role == ASTRAL_AGENT_ROLE_ASSISTANT || role == ASTRAL_AGENT_ROLE_TOOL;
}

inline const char* role_label(AstralAgentRole role, uint32_t* out_len) {
    switch (role) {
    case ASTRAL_AGENT_ROLE_SYSTEM:
        *out_len = static_cast<uint32_t>(sizeof(kSystemLabel) - 1u);
        return kSystemLabel;
    case ASTRAL_AGENT_ROLE_ASSISTANT:
        *out_len = static_cast<uint32_t>(sizeof(kAssistantLabel) - 1u);
        return kAssistantLabel;
    case ASTRAL_AGENT_ROLE_TOOL:
        *out_len = static_cast<uint32_t>(sizeof(kToolLabel) - 1u);
        return kToolLabel;
    case ASTRAL_AGENT_ROLE_USER:
    default:
        *out_len = static_cast<uint32_t>(sizeof(kUserLabel) - 1u);
        return kUserLabel;
    }
}

inline bool span_valid(AstralSpanU8 span) {
    return span.len == 0 || span.data != nullptr;
}

inline bool overflow_policy_valid(AstralAgentOverflowPolicy policy) {
    return policy == ASTRAL_AGENT_OVERFLOW_REJECT || policy == ASTRAL_AGENT_OVERFLOW_TRUNCATE_OLDEST;
}

void invalidate_prompt_metadata(Agent* agent);
void append_bytes_hashed(uint8_t*& dst, const void* src, uint32_t len, uint64_t* hash);

void hash_bytes_into(uint64_t* hash, const void* src, uint32_t len) {
    const uint8_t* bytes = static_cast<const uint8_t*>(src);
    for (uint32_t i = 0; i < len; ++i) {
        *hash ^= static_cast<uint64_t>(bytes[i]);
        *hash *= kFnvPrime;
    }
}

uint8_t* copy_span(AstralSpanU8 span) {
    if (span.len == 0) {
        return nullptr;
    }
    uint8_t* out = core::runtime_alloc_array<uint8_t>(span.len);
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, span.data, span.len);
    return out;
}

void free_bytes(uint8_t*& bytes, uint32_t len) {
    if (bytes != nullptr) {
        core::runtime_free_array(bytes, len);
        bytes = nullptr;
    }
}

} // namespace

struct Agent {
    AstralHandle handle;
    AstralAgentDesc desc;
    AstralHandle conv;
    Conversation* conv_ptr;
    Toolset* toolset;
    uint8_t* system_prompt;
    uint32_t system_prompt_len;
    uint8_t* summary;
    uint32_t summary_len;
    uint8_t* memory_context;
    uint32_t memory_context_len;
    AgentMessageStorage* messages;
    AgentHistoryArena history;
    uint32_t message_count;
    uint32_t message_capacity;
    AgentChatRuntime chat;
    uint32_t max_messages;
    uint32_t max_prompt_bytes;
};

AstralHandle agent_handle(Agent* agent) {
    return agent != nullptr ? agent->handle : 0;
}

namespace {

void invalidate_prompt_metadata(Agent* agent) {
    agent->chat.prefix.valid = 0;
}

void release_history(Agent* agent) {
    if (agent == nullptr || agent->messages == nullptr) {
        if (agent != nullptr) {
            free_bytes(agent->history.bytes, agent->history.capacity);
            agent->history.len = 0;
            agent->history.capacity = 0;
        }
        return;
    }
    core::runtime_free_array(agent->messages, agent->message_capacity);
    agent->messages = nullptr;
    agent->message_count = 0;
    agent->message_capacity = 0;
    free_bytes(agent->history.bytes, agent->history.capacity);
    agent->history.len = 0;
    agent->history.capacity = 0;
}

void destroy_agent_allocations(Agent* agent) {
    if (agent == nullptr) {
        return;
    }
    if (agent->conv != 0) {
        astral_conv_destroy(agent->conv);
        agent->conv = 0;
        agent->conv_ptr = nullptr;
    }
    if (agent->toolset != nullptr) {
        toolset_release(agent->toolset);
        agent->toolset = nullptr;
    }
    free_bytes(agent->system_prompt, agent->system_prompt_len);
    agent->system_prompt_len = 0;
    free_bytes(agent->summary, agent->summary_len);
    agent->summary_len = 0;
    free_bytes(agent->memory_context, agent->memory_context_len);
    agent->memory_context_len = 0;
    free_bytes(agent->chat.prompt.scratch, agent->chat.prompt.scratch_capacity);
    agent->chat.prompt.scratch_capacity = 0;
    free_bytes(agent->chat.prompt.prefix, agent->chat.prompt.prefix_capacity);
    agent->chat.prompt.prefix_capacity = 0;
    free_bytes(agent->chat.generated.bytes, agent->chat.generated.capacity);
    agent->chat.generated.len = 0;
    agent->chat.generated.capacity = 0;
    agent->chat.generated.truncated = 0;
    release_history(agent);
}

AstralErr ensure_message_capacity(Agent* agent, uint32_t required) {
    if (agent == nullptr || required > agent->max_messages) {
        return ASTRAL_E_NOMEM;
    }
    if (required <= agent->message_capacity) {
        return ASTRAL_OK;
    }

    uint32_t next_capacity = agent->message_capacity != 0 ? agent->message_capacity : kInitialMessageCapacity;
    while (next_capacity < required) {
        const uint32_t grown = next_capacity * kCapacityGrowthFactor;
        if (grown <= next_capacity || grown > agent->max_messages) {
            next_capacity = agent->max_messages;
            break;
        }
        next_capacity = grown;
    }

    AgentMessageStorage* next = core::runtime_alloc_array<AgentMessageStorage>(next_capacity);
    if (next == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    std::memset(next, 0, sizeof(AgentMessageStorage) * next_capacity);
    for (uint32_t i = 0; i < agent->message_count; ++i) {
        next[i] = agent->messages[i];
    }
    if (agent->messages != nullptr) {
        core::runtime_free_array(agent->messages, agent->message_capacity);
    }
    agent->messages = next;
    agent->message_capacity = next_capacity;
    return ASTRAL_OK;
}

AstralErr ensure_history_arena_capacity(Agent* agent, uint32_t required) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (required <= agent->history.capacity) {
        return ASTRAL_OK;
    }

    uint32_t next_capacity = agent->history.capacity != 0 ? agent->history.capacity : required;
    while (next_capacity < required) {
        const uint32_t grown = next_capacity * kCapacityGrowthFactor;
        if (grown <= next_capacity) {
            next_capacity = required;
            break;
        }
        next_capacity = grown;
    }

    uint8_t* next = core::runtime_alloc_array<uint8_t>(next_capacity);
    if (next == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    if (agent->history.len != 0) {
        std::memcpy(next, agent->history.bytes, agent->history.len);
    }
    free_bytes(agent->history.bytes, agent->history.capacity);
    agent->history.bytes = next;
    agent->history.capacity = next_capacity;
    return ASTRAL_OK;
}

const uint8_t* message_content(const Agent* agent, const AgentMessageStorage& message) {
    return message.len != 0 ? agent->history.bytes + message.offset : nullptr;
}

void remove_oldest_message(Agent* agent) {
    if (agent->message_count == 0) {
        return;
    }
    const uint32_t removed_begin = agent->messages[0].offset;
    const uint32_t removed_end = removed_begin + agent->messages[0].len;
    const uint32_t tail_bytes = agent->history.len - removed_end;
    if (tail_bytes != 0) {
        std::memmove(agent->history.bytes, agent->history.bytes + removed_end, tail_bytes);
    }
    agent->history.len -= removed_end;
    const uint32_t remaining = agent->message_count - kOneMessage;
    for (uint32_t i = 0; i < remaining; ++i) {
        agent->messages[i] = agent->messages[i + kOneMessage];
        agent->messages[i].offset -= removed_end;
    }
    agent->message_count = remaining;
    if (agent->message_capacity != 0) {
        std::memset(&agent->messages[agent->message_count], 0, sizeof(AgentMessageStorage));
    }
    invalidate_prompt_metadata(agent);
}

uint32_t role_line_bytes(AstralAgentRole role, uint32_t len) {
    uint32_t label_len = 0;
    (void)role_label(role, &label_len);
    return label_len + len + static_cast<uint32_t>(sizeof(kLineBreak) - 1u);
}

uint32_t stable_prefix_bytes(const Agent* agent) {
    uint32_t bytes = 0;
    if (agent->system_prompt_len != 0) {
        bytes += role_line_bytes(ASTRAL_AGENT_ROLE_SYSTEM, agent->system_prompt_len);
    }
    if (agent->summary_len != 0) {
        bytes += static_cast<uint32_t>(sizeof(kSummaryLabel) - 1u) + agent->summary_len +
                 static_cast<uint32_t>(sizeof(kLineBreak) - 1u);
    }
    if (agent->memory_context_len != 0) {
        bytes += static_cast<uint32_t>(sizeof(kMemoryLabel) - 1u) + agent->memory_context_len +
                 static_cast<uint32_t>(sizeof(kLineBreak) - 1u);
    }
    for (uint32_t i = 0; i < agent->message_count; ++i) {
        bytes += role_line_bytes(agent->messages[i].role, agent->messages[i].len);
    }
    return bytes;
}

AstralErr ensure_prefix_cache(Agent* agent, uint32_t required) {
    if (required <= agent->chat.prompt.prefix_capacity) {
        return ASTRAL_OK;
    }
    uint32_t next_capacity = agent->chat.prompt.prefix_capacity != 0 ? agent->chat.prompt.prefix_capacity : required;
    while (next_capacity < required) {
        const uint32_t grown = next_capacity * kCapacityGrowthFactor;
        if (grown <= next_capacity || grown > agent->max_prompt_bytes) {
            next_capacity = required;
            break;
        }
        next_capacity = grown;
    }
    uint8_t* next = core::runtime_alloc_array<uint8_t>(next_capacity);
    if (next == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    free_bytes(agent->chat.prompt.prefix, agent->chat.prompt.prefix_capacity);
    agent->chat.prompt.prefix = next;
    agent->chat.prompt.prefix_capacity = next_capacity;
    return ASTRAL_OK;
}

void append_role_line_hashed(uint8_t*& dst, AstralAgentRole role, const uint8_t* content, uint32_t len, uint64_t* hash) {
    uint32_t label_len = 0;
    const char* label = role_label(role, &label_len);
    append_bytes_hashed(dst, label, label_len, hash);
    append_bytes_hashed(dst, content, len, hash);
    append_bytes_hashed(dst, kLineBreak, static_cast<uint32_t>(sizeof(kLineBreak) - 1u), hash);
}

AstralErr ensure_prompt_metadata(Agent* agent) {
    if (agent->chat.prefix.valid != 0) {
        return ASTRAL_OK;
    }

    const uint32_t bytes = stable_prefix_bytes(agent);
    const AstralErr cache_err = ensure_prefix_cache(agent, bytes);
    if (cache_err != ASTRAL_OK) {
        return cache_err;
    }

    uint8_t* cursor = agent->chat.prompt.prefix;
    uint64_t hash = kFnvOffsetBasis;
    if (agent->system_prompt_len != 0) {
        append_role_line_hashed(cursor, ASTRAL_AGENT_ROLE_SYSTEM, agent->system_prompt, agent->system_prompt_len, &hash);
    }
    if (agent->summary_len != 0) {
        append_bytes_hashed(cursor, kSummaryLabel, static_cast<uint32_t>(sizeof(kSummaryLabel) - 1u), &hash);
        append_bytes_hashed(cursor, agent->summary, agent->summary_len, &hash);
        append_bytes_hashed(cursor, kLineBreak, static_cast<uint32_t>(sizeof(kLineBreak) - 1u), &hash);
    }
    if (agent->memory_context_len != 0) {
        append_bytes_hashed(cursor, kMemoryLabel, static_cast<uint32_t>(sizeof(kMemoryLabel) - 1u), &hash);
        append_bytes_hashed(cursor, agent->memory_context, agent->memory_context_len, &hash);
        append_bytes_hashed(cursor, kLineBreak, static_cast<uint32_t>(sizeof(kLineBreak) - 1u), &hash);
    }
    for (uint32_t i = 0; i < agent->message_count; ++i) {
        append_role_line_hashed(
            cursor,
            agent->messages[i].role,
            message_content(agent, agent->messages[i]),
            agent->messages[i].len,
            &hash
        );
    }
    agent->chat.prefix.prefix_bytes = bytes;
    agent->chat.prefix.prefix_hash = hash;
    agent->chat.prefix.valid = 1;
    return ASTRAL_OK;
}

AstralErr prompt_bytes_required(Agent* agent, AstralSpanU8 user_message, uint32_t* out_bytes) {
    const uint32_t prefix_bytes =
        agent->chat.prefix.valid != 0 ? agent->chat.prefix.prefix_bytes : stable_prefix_bytes(agent);
    uint32_t bytes = kHeaderReserveBytes + prefix_bytes;
    if (user_message.len != 0) {
        bytes += static_cast<uint32_t>(sizeof(kUserLabel) - 1u) + user_message.len +
                 static_cast<uint32_t>(sizeof(kLineBreak) - 1u);
    }
    bytes += static_cast<uint32_t>(sizeof(kAssistantLabel) - 1u);
    *out_bytes = bytes;
    return ASTRAL_OK;
}

AstralErr apply_prompt_overflow(Agent* agent, AstralSpanU8 user_message, uint32_t* out_bytes) {
    uint32_t bytes = 0;
    AstralErr bytes_err = prompt_bytes_required(agent, user_message, &bytes);
    if (bytes_err != ASTRAL_OK) {
        return bytes_err;
    }
    if (bytes <= agent->max_prompt_bytes) {
        *out_bytes = bytes;
        return ASTRAL_OK;
    }
    if (agent->desc.overflow_policy != ASTRAL_AGENT_OVERFLOW_TRUNCATE_OLDEST) {
        return ASTRAL_E_NOMEM;
    }
    while (bytes > agent->max_prompt_bytes && agent->message_count != 0) {
        remove_oldest_message(agent);
        bytes_err = prompt_bytes_required(agent, user_message, &bytes);
        if (bytes_err != ASTRAL_OK) {
            return bytes_err;
        }
    }
    if (bytes > agent->max_prompt_bytes) {
        return ASTRAL_E_NOMEM;
    }
    *out_bytes = bytes;
    return ASTRAL_OK;
}

AstralErr ensure_prompt_scratch(Agent* agent, uint32_t required) {
    if (agent == nullptr || required > agent->max_prompt_bytes) {
        return ASTRAL_E_NOMEM;
    }
    if (required <= agent->chat.prompt.scratch_capacity) {
        return ASTRAL_OK;
    }

    uint32_t next_capacity = agent->chat.prompt.scratch_capacity != 0 ? agent->chat.prompt.scratch_capacity : required;
    while (next_capacity < required) {
        const uint32_t grown = next_capacity * kCapacityGrowthFactor;
        if (grown <= next_capacity || grown > agent->max_prompt_bytes) {
            next_capacity = agent->max_prompt_bytes;
            break;
        }
        next_capacity = grown;
    }

    uint8_t* next = core::runtime_alloc_array<uint8_t>(next_capacity);
    if (next == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    free_bytes(agent->chat.prompt.scratch, agent->chat.prompt.scratch_capacity);
    agent->chat.prompt.scratch = next;
    agent->chat.prompt.scratch_capacity = next_capacity;
    return ASTRAL_OK;
}

void append_bytes(uint8_t*& dst, const void* src, uint32_t len) {
    if (len == 0) {
        return;
    }
    std::memcpy(dst, src, len);
    dst += len;
}

void append_bytes_hashed(uint8_t*& dst, const void* src, uint32_t len, uint64_t* hash) {
    append_bytes(dst, src, len);
    hash_bytes_into(hash, src, len);
}

void append_span(uint8_t*& dst, AstralSpanU8 span) {
    append_bytes(dst, span.data, span.len);
}

void append_span_hashed(uint8_t*& dst, AstralSpanU8 span, uint64_t* hash) {
    append_bytes_hashed(dst, span.data, span.len, hash);
}

AstralErr build_prompt(
    Agent* agent,
    AstralSpanU8 user_message,
    uint8_t** out_bytes,
    uint32_t* out_len,
    uint32_t* out_prefix_len,
    uint64_t* out_prefix_hash,
    uint64_t* out_prompt_hash
) {
    ASTRAL_ZONE_N("astral.agent.build_prompt");
    if (agent == nullptr || out_bytes == nullptr || out_len == nullptr || !span_valid(user_message)) {
        return ASTRAL_E_INVALID;
    }

    uint32_t bytes = 0;
    const AstralErr overflow_err = apply_prompt_overflow(agent, user_message, &bytes);
    if (overflow_err != ASTRAL_OK) {
        return overflow_err;
    }

    const AstralErr scratch_err = ensure_prompt_scratch(agent, bytes);
    if (scratch_err != ASTRAL_OK) {
        return scratch_err;
    }

    const AstralErr meta_err = ensure_prompt_metadata(agent);
    if (meta_err != ASTRAL_OK) {
        return meta_err;
    }

    uint8_t* cursor = agent->chat.prompt.scratch;
    append_bytes(cursor, agent->chat.prompt.prefix, agent->chat.prefix.prefix_bytes);
    if (out_prefix_len != nullptr) {
        *out_prefix_len = agent->chat.prefix.prefix_bytes;
    }
    if (out_prefix_hash != nullptr) {
        *out_prefix_hash = agent->chat.prefix.prefix_hash;
    }

    uint64_t hash = agent->chat.prefix.prefix_hash;
    if (user_message.len != 0) {
        AstralSpanU8 user_label{};
        user_label.data = reinterpret_cast<const uint8_t*>(kUserLabel);
        user_label.len = static_cast<uint32_t>(sizeof(kUserLabel) - 1u);
        if (out_prompt_hash != nullptr) {
            append_span_hashed(cursor, user_label, &hash);
            append_span_hashed(cursor, user_message, &hash);
            append_bytes_hashed(cursor, kLineBreak, static_cast<uint32_t>(sizeof(kLineBreak) - 1u), &hash);
        } else {
            append_span(cursor, user_label);
            append_span(cursor, user_message);
            append_bytes(cursor, kLineBreak, static_cast<uint32_t>(sizeof(kLineBreak) - 1u));
        }
    }
    if (out_prompt_hash != nullptr) {
        append_bytes_hashed(cursor, kAssistantLabel, static_cast<uint32_t>(sizeof(kAssistantLabel) - 1u), &hash);
        *out_prompt_hash = hash;
    } else {
        append_bytes(cursor, kAssistantLabel, static_cast<uint32_t>(sizeof(kAssistantLabel) - 1u));
    }

    *out_bytes = agent->chat.prompt.scratch;
    *out_len = static_cast<uint32_t>(cursor - agent->chat.prompt.scratch);
    return ASTRAL_OK;
}

AstralPromptCacheKey prompt_cache_key(const Agent* agent, uint64_t prompt_hash, AstralPromptSectionKind section_kind) {
    AstralPromptCacheKey key{};
    key.size = sizeof(AstralPromptCacheKey);
    key.section_kind = section_kind;
    key.model = agent->desc.model;
    key.key = prompt_hash;
    key.generation = kPromptCacheGeneration;
    return key;
}

void reset_prompt_cache_stats(Agent* agent) {
    agent->chat.prompt_cache_reused_tokens = 0;
    agent->chat.prompt_cache_new_tokens = 0;
    agent->chat.prompt_cache_hits = 0;
    agent->chat.prompt_cache_misses = 0;
}

AstralErr feed_prompt_with_cache(
    Agent* agent,
    AstralSpanU8 prompt,
    uint32_t prefix_len,
    uint64_t prefix_hash,
    uint64_t prompt_hash
) {
    const AstralPromptCacheKey full_key = prompt_cache_key(agent, prompt_hash, ASTRAL_PROMPT_SECTION_RAW);
    const int32_t* full_tokens = nullptr;
    uint32_t full_count = 0;

    AstralErr err =
        astral_prompt_cache_get_token_view(agent->desc.prompt_cache, &full_key, &full_tokens, &full_count);
    if (err == ASTRAL_OK) {
        agent->chat.prompt_cache_reused_tokens = full_count;
        agent->chat.prompt_cache_hits = kPromptCacheHitCount;
        return conv_feed_tokens(agent->conv_ptr, full_tokens, full_count, kPromptFinalize);
    }
    if (err != ASTRAL_E_NOT_FOUND) {
        return err;
    }

    if (prefix_len == 0) {
        return conv_feed(agent->conv_ptr, prompt, kPromptFinalize);
    }

    AstralSpanU8 prefix{};
    prefix.data = prompt.data;
    prefix.len = prefix_len;
    AstralSpanU8 suffix{};
    suffix.data = prompt.data + prefix_len;
    suffix.len = prompt.len - prefix_len;

    const AstralPromptCacheKey key = prompt_cache_key(agent, prefix_hash, ASTRAL_PROMPT_SECTION_HISTORY);
    const int32_t* cached_tokens = nullptr;
    uint32_t cached_count = 0;

    err = astral_prompt_cache_get_token_view(agent->desc.prompt_cache, &key, &cached_tokens, &cached_count);
    if (err == ASTRAL_OK) {
        agent->chat.prompt_cache_reused_tokens = cached_count;
        agent->chat.prompt_cache_hits = kPromptCacheHitCount;
        err = conv_feed_tokens(agent->conv_ptr, cached_tokens, cached_count, 0);
        if (err != ASTRAL_OK) {
            return err;
        }
        err = conv_feed(agent->conv_ptr, suffix, kPromptFinalize);
        if (err == ASTRAL_OK && agent->conv_ptr->prompt_count >= cached_count) {
            agent->chat.prompt_cache_new_tokens = agent->conv_ptr->prompt_count - cached_count;
        }
        return err;
    }
    if (err != ASTRAL_E_NOT_FOUND) {
        return err;
    }

    uint32_t full_token_count = 0;
    err = astral_tokenize_count(agent->desc.model, prompt, kPromptAddSpecial, kPromptParseSpecial, &full_token_count);
    if (err != ASTRAL_OK) {
        return err;
    }
    int32_t* full_copy = nullptr;
    if (full_token_count != 0) {
        full_copy = core::runtime_alloc_array<int32_t>(full_token_count);
        if (full_copy == nullptr) {
            return ASTRAL_E_NOMEM;
        }
    }
    uint32_t full_written = 0;
    err = astral_tokenize(
        agent->desc.model,
        prompt,
        full_copy,
        full_token_count,
        kPromptAddSpecial,
        kPromptParseSpecial,
        &full_written
    );
    if (err == ASTRAL_OK && full_written != full_token_count) {
        err = ASTRAL_E_BACKEND;
    }
    if (err == ASTRAL_OK) {
        (void)astral_prompt_cache_put_tokens(agent->desc.prompt_cache, &full_key, full_copy, full_written);
        agent->chat.prompt_cache_new_tokens = full_written;
        agent->chat.prompt_cache_misses = kPromptCacheMissCount;
        err = conv_feed_tokens(agent->conv_ptr, full_copy, full_written, kPromptFinalize);
    }
    const AstralErr full_feed_err = err;

    uint32_t token_count = 0;
    const AstralErr prefix_count_err =
        astral_tokenize_count(agent->desc.model, prefix, kPromptAddSpecial, kPromptParseSpecial, &token_count);
    if (prefix_count_err != ASTRAL_OK || full_feed_err != ASTRAL_OK) {
        if (full_copy != nullptr) {
            core::runtime_free_array(full_copy, full_token_count);
        }
        return full_feed_err;
    }

    int32_t* tokens = nullptr;
    if (token_count != 0) {
        tokens = core::runtime_alloc_array<int32_t>(token_count);
        if (tokens == nullptr) {
            if (full_copy != nullptr) {
                core::runtime_free_array(full_copy, full_token_count);
            }
            return full_feed_err;
        }
    }

    uint32_t written = 0;
    err = astral_tokenize(
        agent->desc.model,
        prefix,
        tokens,
        token_count,
        kPromptAddSpecial,
        kPromptParseSpecial,
        &written
    );
    if (err == ASTRAL_OK && written != token_count) {
        err = ASTRAL_E_BACKEND;
    }
    if (err == ASTRAL_OK && written == token_count) {
        err = astral_prompt_cache_put_tokens(agent->desc.prompt_cache, &key, tokens, written);
    }

    if (tokens != nullptr) {
        core::runtime_free_array(tokens, token_count);
    }
    if (full_copy != nullptr) {
        core::runtime_free_array(full_copy, full_token_count);
    }
    return full_feed_err;
}

AstralErr replace_system_prompt(Agent* agent, AstralSpanU8 system_prompt) {
    if (agent == nullptr || !span_valid(system_prompt)) {
        return ASTRAL_E_INVALID;
    }
    uint8_t* copy = copy_span(system_prompt);
    if (system_prompt.len != 0 && copy == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    free_bytes(agent->system_prompt, agent->system_prompt_len);
    agent->system_prompt = copy;
    agent->system_prompt_len = system_prompt.len;
    invalidate_prompt_metadata(agent);
    return ASTRAL_OK;
}

AstralErr replace_summary(Agent* agent, AstralSpanU8 summary) {
    if (agent == nullptr || !span_valid(summary)) {
        return ASTRAL_E_INVALID;
    }
    uint8_t* copy = copy_span(summary);
    if (summary.len != 0 && copy == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    free_bytes(agent->summary, agent->summary_len);
    agent->summary = copy;
    agent->summary_len = summary.len;
    invalidate_prompt_metadata(agent);
    return ASTRAL_OK;
}

AstralErr replace_memory_context(Agent* agent, AstralSpanU8 memory_context) {
    if (agent == nullptr || !span_valid(memory_context)) {
        return ASTRAL_E_INVALID;
    }
    uint8_t* copy = copy_span(memory_context);
    if (memory_context.len != 0 && copy == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    free_bytes(agent->memory_context, agent->memory_context_len);
    agent->memory_context = copy;
    agent->memory_context_len = memory_context.len;
    invalidate_prompt_metadata(agent);
    return ASTRAL_OK;
}

AstralErr replace_memory_context_from_results(Agent* agent, const AstralAgentMemoryContextDesc* desc) {
    if (agent == nullptr || desc == nullptr || desc->size != sizeof(AstralAgentMemoryContextDesc) ||
        !span_valid(desc->document_text) || !span_valid(desc->separator)) {
        return ASTRAL_E_INVALID;
    }
    if ((desc->result_count != 0 && desc->results == nullptr) || (desc->chunk_count != 0 && desc->chunks == nullptr)) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t max_bytes = desc->max_bytes != 0 ? desc->max_bytes : agent->max_prompt_bytes;
    uint32_t bytes = 0;
    uint32_t selected = 0;
    for (uint32_t i = 0; i < desc->result_count; ++i) {
        const AstralMemorySearchResult& result = desc->results[i];
        if (result.size != sizeof(AstralMemorySearchResult) || result.chunk_id >= desc->chunk_count) {
            return ASTRAL_E_INVALID;
        }
        const AstralChunkRange& range = desc->chunks[result.chunk_id];
        if (range.size != sizeof(AstralChunkRange) || range.byte_begin > range.byte_end ||
            range.byte_end > desc->document_text.len) {
            return ASTRAL_E_INVALID;
        }
        const uint32_t chunk_bytes = range.byte_end - range.byte_begin;
        if (chunk_bytes == 0) {
            continue;
        }
        const uint32_t separator_bytes = selected != 0 ? desc->separator.len : 0;
        if (separator_bytes > max_bytes || chunk_bytes > max_bytes ||
            bytes > max_bytes - separator_bytes || bytes + separator_bytes > max_bytes - chunk_bytes) {
            return ASTRAL_E_NOMEM;
        }
        bytes += separator_bytes + chunk_bytes;
        ++selected;
    }

    if (bytes == 0) {
        free_bytes(agent->memory_context, agent->memory_context_len);
        agent->memory_context_len = 0;
        invalidate_prompt_metadata(agent);
        return ASTRAL_OK;
    }

    uint8_t* copy = core::runtime_alloc_array<uint8_t>(bytes);
    if (copy == nullptr) {
        return ASTRAL_E_NOMEM;
    }

    uint8_t* cursor = copy;
    selected = 0;
    for (uint32_t i = 0; i < desc->result_count; ++i) {
        const AstralMemorySearchResult& result = desc->results[i];
        const AstralChunkRange& range = desc->chunks[result.chunk_id];
        const uint32_t chunk_bytes = range.byte_end - range.byte_begin;
        if (chunk_bytes == 0) {
            continue;
        }
        if (selected != 0 && desc->separator.len != 0) {
            std::memcpy(cursor, desc->separator.data, desc->separator.len);
            cursor += desc->separator.len;
        }
        std::memcpy(cursor, desc->document_text.data + range.byte_begin, chunk_bytes);
        cursor += chunk_bytes;
        ++selected;
    }

    free_bytes(agent->memory_context, agent->memory_context_len);
    agent->memory_context = copy;
    agent->memory_context_len = bytes;
    invalidate_prompt_metadata(agent);
    return ASTRAL_OK;
}

AstralErr add_message_storage(Agent* agent, AstralAgentRole role, AstralSpanU8 content) {
    if (agent == nullptr || !role_valid(role) || !span_valid(content)) {
        return ASTRAL_E_INVALID;
    }
    if (agent->message_count == agent->max_messages) {
        if (agent->desc.overflow_policy != ASTRAL_AGENT_OVERFLOW_TRUNCATE_OLDEST) {
            return ASTRAL_E_NOMEM;
        }
        remove_oldest_message(agent);
    }
    const AstralErr cap_err = ensure_message_capacity(agent, agent->message_count + kOneMessage);
    if (cap_err != ASTRAL_OK) {
        return cap_err;
    }
    if (content.len > UINT32_MAX - agent->history.len) {
        return ASTRAL_E_NOMEM;
    }
    const uint32_t offset = agent->history.len;
    const AstralErr arena_err = ensure_history_arena_capacity(agent, offset + content.len);
    if (arena_err != ASTRAL_OK) {
        return arena_err;
    }
    if (content.len != 0) {
        std::memcpy(agent->history.bytes + offset, content.data, content.len);
        agent->history.len += content.len;
    }
    AgentMessageStorage& dst = agent->messages[agent->message_count++];
    dst.role = role;
    dst.offset = offset;
    dst.len = content.len;
    invalidate_prompt_metadata(agent);
    return ASTRAL_OK;
}

AstralErr prepare_generated_capture(Agent* agent, uint32_t max_tokens) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    agent->chat.generated.len = 0;
    agent->chat.generated.truncated = 0;
    if (agent->toolset == nullptr) {
        return ASTRAL_OK;
    }
    if (max_tokens == 0) {
        free_bytes(agent->chat.generated.bytes, agent->chat.generated.capacity);
        agent->chat.generated.capacity = 0;
        return ASTRAL_OK;
    }
    constexpr uint32_t kCaptureBytesPerToken = concurrency::kStreamTokenUtf8Capacity;
    if (max_tokens > UINT32_MAX / kCaptureBytesPerToken) {
        return ASTRAL_E_NOMEM;
    }
    const uint32_t required = max_tokens * kCaptureBytesPerToken;
    if (required <= agent->chat.generated.capacity) {
        return ASTRAL_OK;
    }
    uint8_t* bytes = core::runtime_alloc_array<uint8_t>(required);
    if (bytes == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    free_bytes(agent->chat.generated.bytes, agent->chat.generated.capacity);
    agent->chat.generated.bytes = bytes;
    agent->chat.generated.capacity = required;
    return ASTRAL_OK;
}

void append_generated_capture(Agent& agent, const uint8_t* bytes, uint32_t len) {
    if (agent.chat.generated.bytes == nullptr || bytes == nullptr || len == 0) {
        return;
    }
    const uint32_t remaining = agent.chat.generated.capacity - agent.chat.generated.len;
    const uint32_t copied = len < remaining ? len : remaining;
    if (copied != 0) {
        std::memcpy(agent.chat.generated.bytes + agent.chat.generated.len, bytes, copied);
        agent.chat.generated.len += copied;
    }
    if (copied != len) {
        agent.chat.generated.truncated = 1;
    }
}

} // namespace

AstralErr agent_create(const AstralAgentDesc* desc, Agent** out_agent) {
    if (desc == nullptr || out_agent == nullptr || desc->size != sizeof(AstralAgentDesc) || desc->model == 0 ||
        !span_valid(desc->system_prompt) || !span_valid(desc->summary) || !span_valid(desc->memory_context)) {
        return ASTRAL_E_INVALID;
    }
    if (!overflow_policy_valid(desc->overflow_policy)) {
        return ASTRAL_E_INVALID;
    }

    Agent* agent = core::runtime_new<Agent>();
    if (agent == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    std::memset(agent, 0, sizeof(Agent));
    agent->desc = *desc;
    agent->desc.system_prompt = {};
    agent->desc.summary = {};
    agent->desc.memory_context = {};
    agent->max_messages = desc->max_messages != 0 ? desc->max_messages : kDefaultMaxMessages;
    agent->max_prompt_bytes = desc->max_prompt_bytes != 0 ? desc->max_prompt_bytes : kDefaultMaxPromptBytes;
    agent->chat.last_error = ASTRAL_OK;

    AstralConvDesc conv_desc{};
    conv_desc.size = sizeof(AstralConvDesc);
    conv_desc.model = desc->model;
    conv_desc.max_tokens = desc->max_tokens;
    conv_desc.temperature = desc->temperature;
    conv_desc.top_k = desc->top_k;
    conv_desc.top_p = desc->top_p;
    conv_desc.stream_enabled = desc->stream_enabled;
    conv_desc.seed = desc->seed;

    AstralErr err = conv_create_affine(&conv_desc, desc->slot_affinity, &agent->conv_ptr);
    if (err != ASTRAL_OK) {
        core::runtime_delete(agent);
        return err;
    }
    agent->conv = agent->conv_ptr->handle;
    if (desc->toolset != 0) {
        auto* toolset = static_cast<Toolset*>(core::lookup_handle(desc->toolset, core::HandleKind::Toolset));
        if (toolset == nullptr) {
            destroy_agent_allocations(agent);
            core::runtime_delete(agent);
            return ASTRAL_E_INVALID;
        }
        err = conv_set_toolset(agent->conv_ptr, toolset, desc->tool_choice_mode);
        if (err != ASTRAL_OK) {
            destroy_agent_allocations(agent);
            core::runtime_delete(agent);
            return err;
        }
        toolset_retain(toolset);
        agent->toolset = toolset;
    }
    if (desc->prompt_cache != 0) {
        AstralPromptCacheStats stats{};
        stats.size = sizeof(AstralPromptCacheStats);
        err = astral_prompt_cache_stats(desc->prompt_cache, &stats);
        if (err != ASTRAL_OK) {
            destroy_agent_allocations(agent);
            core::runtime_delete(agent);
            return err;
        }
    }
    if (desc->memory_index != 0) {
        uint32_t memory_count = 0;
        err = astral_memory_count(desc->memory_index, &memory_count);
        if (err != ASTRAL_OK) {
            destroy_agent_allocations(agent);
            core::runtime_delete(agent);
            return err;
        }
    }
    if (desc->system_prompt.len != 0) {
        err = replace_system_prompt(agent, desc->system_prompt);
        if (err != ASTRAL_OK) {
            destroy_agent_allocations(agent);
            core::runtime_delete(agent);
            return err;
        }
    }
    if (desc->summary.len != 0) {
        err = replace_summary(agent, desc->summary);
        if (err != ASTRAL_OK) {
            destroy_agent_allocations(agent);
            core::runtime_delete(agent);
            return err;
        }
    }
    if (desc->memory_context.len != 0) {
        err = replace_memory_context(agent, desc->memory_context);
        if (err != ASTRAL_OK) {
            destroy_agent_allocations(agent);
            core::runtime_delete(agent);
            return err;
        }
    }

    const AstralHandle handle = core::register_handle(core::HandleKind::Agent, agent);
    if (handle == 0) {
        destroy_agent_allocations(agent);
        core::runtime_delete(agent);
        return ASTRAL_E_BUSY;
    }

    agent->handle = handle;
    *out_agent = agent;
    return ASTRAL_OK;
}

void agent_destroy(Agent* agent) {
    if (agent == nullptr) {
        return;
    }
    core::unregister_handle(agent->handle, core::HandleKind::Agent);
    destroy_agent_allocations(agent);
    core::runtime_delete(agent);
}

AstralErr agent_assigned_slot(Agent* agent, uint32_t* out_slot) {
    if (agent == nullptr || out_slot == nullptr || agent->conv_ptr == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_slot = agent->conv_ptr->slot_id.load(std::memory_order_acquire);
    return ASTRAL_OK;
}

AstralErr agent_set_system_prompt(Agent* agent, AstralSpanU8 system_prompt) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    AstralSessionState state = ASTRAL_SESSION_FAILED;
    const AstralErr state_err = conv_state(agent->conv_ptr, &state);
    if (state_err != ASTRAL_OK) {
        return state_err;
    }
    if (state == ASTRAL_SESSION_DECODING) {
        return ASTRAL_E_STATE;
    }
    return replace_system_prompt(agent, system_prompt);
}

AstralErr agent_get_system_prompt_size(Agent* agent, uint32_t* out_bytes) {
    if (agent == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_bytes = agent->system_prompt_len;
    return ASTRAL_OK;
}

AstralErr agent_get_system_prompt(Agent* agent, AstralMutSpanU8 out_text, uint32_t* out_len) {
    if (agent == nullptr || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_len = agent->system_prompt_len;
    if (agent->system_prompt_len == 0) {
        return ASTRAL_OK;
    }
    if (out_text.data == nullptr || out_text.len < agent->system_prompt_len) {
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(out_text.data, agent->system_prompt, agent->system_prompt_len);
    return ASTRAL_OK;
}

AstralErr agent_set_summary(Agent* agent, AstralSpanU8 summary) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    AstralSessionState state = ASTRAL_SESSION_FAILED;
    const AstralErr state_err = conv_state(agent->conv_ptr, &state);
    if (state_err != ASTRAL_OK) {
        return state_err;
    }
    if (state == ASTRAL_SESSION_DECODING) {
        return ASTRAL_E_STATE;
    }
    return replace_summary(agent, summary);
}

AstralErr agent_get_summary_size(Agent* agent, uint32_t* out_bytes) {
    if (agent == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_bytes = agent->summary_len;
    return ASTRAL_OK;
}

AstralErr agent_get_summary(Agent* agent, AstralMutSpanU8 out_text, uint32_t* out_len) {
    if (agent == nullptr || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_len = agent->summary_len;
    if (agent->summary_len == 0) {
        return ASTRAL_OK;
    }
    if (out_text.data == nullptr || out_text.len < agent->summary_len) {
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(out_text.data, agent->summary, agent->summary_len);
    return ASTRAL_OK;
}

AstralErr agent_set_memory_context(Agent* agent, AstralSpanU8 memory_context) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    AstralSessionState state = ASTRAL_SESSION_FAILED;
    const AstralErr state_err = conv_state(agent->conv_ptr, &state);
    if (state_err != ASTRAL_OK) {
        return state_err;
    }
    if (state == ASTRAL_SESSION_DECODING) {
        return ASTRAL_E_STATE;
    }
    return replace_memory_context(agent, memory_context);
}

AstralErr agent_set_memory_context_from_results(Agent* agent, const AstralAgentMemoryContextDesc* desc) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    AstralSessionState state = ASTRAL_SESSION_FAILED;
    const AstralErr state_err = conv_state(agent->conv_ptr, &state);
    if (state_err != ASTRAL_OK) {
        return state_err;
    }
    if (state == ASTRAL_SESSION_DECODING) {
        return ASTRAL_E_STATE;
    }
    return replace_memory_context_from_results(agent, desc);
}

AstralErr agent_get_memory_context_size(Agent* agent, uint32_t* out_bytes) {
    if (agent == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_bytes = agent->memory_context_len;
    return ASTRAL_OK;
}

AstralErr agent_get_memory_context(Agent* agent, AstralMutSpanU8 out_text, uint32_t* out_len) {
    if (agent == nullptr || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_len = agent->memory_context_len;
    if (agent->memory_context_len == 0) {
        return ASTRAL_OK;
    }
    if (out_text.data == nullptr || out_text.len < agent->memory_context_len) {
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(out_text.data, agent->memory_context, agent->memory_context_len);
    return ASTRAL_OK;
}

AstralErr agent_parse_tool_call(Agent* agent, AstralSpanU8 generated_text, AstralToolCallResult* out_result) {
    if (agent == nullptr || out_result == nullptr || out_result->size != sizeof(AstralToolCallResult)) {
        return ASTRAL_E_INVALID;
    }
    if (agent->toolset == nullptr) {
        return ASTRAL_E_NOT_FOUND;
    }
    const AstralErr err = toolset_parse_call(agent->toolset, generated_text, out_result);
    agent->chat.last_error = err;
    return err;
}

AstralErr agent_message_add(Agent* agent, const AstralAgentMessage* message) {
    if (agent == nullptr || message == nullptr || message->size != sizeof(AstralAgentMessage)) {
        return ASTRAL_E_INVALID;
    }
    return add_message_storage(agent, message->role, message->content);
}

AstralErr agent_history_clear(Agent* agent) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    agent->message_count = 0;
    agent->history.len = 0;
    invalidate_prompt_metadata(agent);
    return ASTRAL_OK;
}

AstralErr agent_history_count(Agent* agent, uint32_t* out_count) {
    if (agent == nullptr || out_count == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_count = agent->message_count;
    return ASTRAL_OK;
}

AstralErr agent_history_save_size(Agent* agent, uint32_t* out_bytes) {
    if (agent == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    uint32_t bytes = sizeof(AgentSaveHeader) + agent->system_prompt_len + agent->summary_len +
                     agent->memory_context_len;
    for (uint32_t i = 0; i < agent->message_count; ++i) {
        bytes += sizeof(AgentSaveRecord) + agent->messages[i].len;
    }
    *out_bytes = bytes;
    return ASTRAL_OK;
}

AstralErr agent_history_save(Agent* agent, AstralMutSpanU8 out_bytes, uint32_t* out_len) {
    if (agent == nullptr || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }
    uint32_t required = 0;
    AstralErr err = agent_history_save_size(agent, &required);
    if (err != ASTRAL_OK) {
        return err;
    }
    *out_len = required;
    if (out_bytes.data == nullptr || out_bytes.len < required) {
        return ASTRAL_E_NOMEM;
    }

    AgentSaveHeader header{};
    header.magic = kSaveMagic;
    header.version = kSaveVersion;
    header.system_len = agent->system_prompt_len;
    header.summary_len = agent->summary_len;
    header.memory_context_len = agent->memory_context_len;
    header.message_count = agent->message_count;

    uint8_t* cursor = out_bytes.data;
    append_bytes(cursor, &header, sizeof(header));
    append_bytes(cursor, agent->system_prompt, agent->system_prompt_len);
    append_bytes(cursor, agent->summary, agent->summary_len);
    append_bytes(cursor, agent->memory_context, agent->memory_context_len);
    for (uint32_t i = 0; i < agent->message_count; ++i) {
        AgentSaveRecord record{};
        record.role = agent->messages[i].role;
        record.len = agent->messages[i].len;
        append_bytes(cursor, &record, sizeof(record));
        append_bytes(cursor, message_content(agent, agent->messages[i]), agent->messages[i].len);
    }
    return ASTRAL_OK;
}

AstralErr agent_history_load(Agent* agent, AstralSpanU8 bytes) {
    if (agent == nullptr || bytes.data == nullptr || bytes.len < sizeof(AgentSaveHeaderV1)) {
        return ASTRAL_E_INVALID;
    }

    AgentSaveHeaderV1 v1{};
    std::memcpy(&v1, bytes.data, sizeof(v1));
    if (v1.magic != kSaveMagic) {
        return ASTRAL_E_INVALID;
    }

    uint32_t system_len = 0;
    uint32_t summary_len = 0;
    uint32_t memory_context_len = 0;
    uint32_t message_count = 0;
    size_t header_bytes = 0;
    if (v1.version == kSaveVersionLegacy) {
        system_len = v1.system_len;
        message_count = v1.message_count;
        header_bytes = sizeof(AgentSaveHeaderV1);
    } else if (v1.version == kSaveVersionSummary) {
        struct AgentSaveHeaderV2 {
            uint32_t magic;
            uint32_t version;
            uint32_t system_len;
            uint32_t summary_len;
            uint32_t message_count;
        };
        if (bytes.len < sizeof(AgentSaveHeaderV2)) {
            return ASTRAL_E_INVALID;
        }
        AgentSaveHeaderV2 header{};
        std::memcpy(&header, bytes.data, sizeof(header));
        system_len = header.system_len;
        summary_len = header.summary_len;
        message_count = header.message_count;
        header_bytes = sizeof(AgentSaveHeaderV2);
    } else if (v1.version == kSaveVersion) {
        if (bytes.len < sizeof(AgentSaveHeader)) {
            return ASTRAL_E_INVALID;
        }
        AgentSaveHeader header{};
        std::memcpy(&header, bytes.data, sizeof(header));
        system_len = header.system_len;
        summary_len = header.summary_len;
        memory_context_len = header.memory_context_len;
        message_count = header.message_count;
        header_bytes = sizeof(AgentSaveHeader);
    } else {
        return ASTRAL_E_INVALID;
    }
    if (message_count > agent->max_messages) {
        return ASTRAL_E_INVALID;
    }

    const uint8_t* cursor = bytes.data + header_bytes;
    const uint8_t* end = bytes.data + bytes.len;
    if (static_cast<size_t>(end - cursor) < system_len) {
        return ASTRAL_E_INVALID;
    }

    AstralSpanU8 system{};
    system.data = cursor;
    system.len = system_len;
    cursor += system_len;
    if (static_cast<size_t>(end - cursor) < summary_len) {
        return ASTRAL_E_INVALID;
    }

    AstralSpanU8 summary{};
    summary.data = cursor;
    summary.len = summary_len;
    cursor += summary_len;
    if (static_cast<size_t>(end - cursor) < memory_context_len) {
        return ASTRAL_E_INVALID;
    }

    AstralSpanU8 memory_context{};
    memory_context.data = cursor;
    memory_context.len = memory_context_len;
    cursor += memory_context_len;

    const uint8_t* records_begin = cursor;
    uint32_t history_bytes = 0;
    for (uint32_t i = 0; i < message_count; ++i) {
        if (static_cast<size_t>(end - cursor) < sizeof(AgentSaveRecord)) {
            return ASTRAL_E_INVALID;
        }
        AgentSaveRecord record{};
        std::memcpy(&record, cursor, sizeof(record));
        cursor += sizeof(record);
        if (!role_valid(record.role) || static_cast<size_t>(end - cursor) < record.len ||
            record.len > UINT32_MAX - history_bytes) {
            return ASTRAL_E_INVALID;
        }
        cursor += record.len;
        history_bytes += record.len;
    }
    if (cursor != end) {
        return ASTRAL_E_INVALID;
    }

    uint8_t* system_copy = copy_span(system);
    if (system.len != 0 && system_copy == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    uint8_t* summary_copy = copy_span(summary);
    if (summary.len != 0 && summary_copy == nullptr) {
        free_bytes(system_copy, system.len);
        return ASTRAL_E_NOMEM;
    }
    uint8_t* memory_context_copy = copy_span(memory_context);
    if (memory_context.len != 0 && memory_context_copy == nullptr) {
        free_bytes(system_copy, system.len);
        free_bytes(summary_copy, summary.len);
        return ASTRAL_E_NOMEM;
    }

    AgentMessageStorage* loaded = nullptr;
    uint32_t loaded_capacity = 0;
    if (message_count != 0) {
        loaded = core::runtime_alloc_array<AgentMessageStorage>(message_count);
        if (loaded == nullptr) {
            free_bytes(system_copy, system.len);
            free_bytes(summary_copy, summary.len);
            free_bytes(memory_context_copy, memory_context.len);
            return ASTRAL_E_NOMEM;
        }
        std::memset(loaded, 0, sizeof(AgentMessageStorage) * message_count);
        loaded_capacity = message_count;
    }

    uint8_t* loaded_history = nullptr;
    if (history_bytes != 0) {
        loaded_history = core::runtime_alloc_array<uint8_t>(history_bytes);
        if (loaded_history == nullptr) {
            free_bytes(system_copy, system.len);
            free_bytes(summary_copy, summary.len);
            free_bytes(memory_context_copy, memory_context.len);
            core::runtime_free_array(loaded, loaded_capacity);
            return ASTRAL_E_NOMEM;
        }
    }

    cursor = records_begin;
    uint32_t history_offset = 0;
    for (uint32_t i = 0; i < message_count; ++i) {
        AgentSaveRecord record{};
        std::memcpy(&record, cursor, sizeof(record));
        cursor += sizeof(record);
        const uint8_t* content = cursor;
        cursor += record.len;
        if (record.len != 0) {
            std::memcpy(loaded_history + history_offset, content, record.len);
        }
        loaded[i].role = record.role;
        loaded[i].offset = history_offset;
        loaded[i].len = record.len;
        history_offset += record.len;
    }

    free_bytes(agent->system_prompt, agent->system_prompt_len);
    free_bytes(agent->summary, agent->summary_len);
    free_bytes(agent->memory_context, agent->memory_context_len);
    release_history(agent);
    agent->system_prompt = system_copy;
    agent->system_prompt_len = system.len;
    agent->summary = summary_copy;
    agent->summary_len = summary.len;
    agent->memory_context = memory_context_copy;
    agent->memory_context_len = memory_context.len;
    agent->messages = loaded;
    agent->history.bytes = loaded_history;
    agent->history.len = history_bytes;
    agent->history.capacity = history_bytes;
    agent->message_capacity = loaded_capacity;
    agent->message_count = message_count;
    invalidate_prompt_metadata(agent);
    return ASTRAL_OK;
}

AstralErr agent_chat_enqueue(Agent* agent, const AstralAgentChatDesc* desc) {
    if (agent == nullptr || desc == nullptr || desc->size != sizeof(AstralAgentChatDesc) ||
        !span_valid(desc->user_message)) {
        return ASTRAL_E_INVALID;
    }

    AstralSessionState state = ASTRAL_SESSION_FAILED;
    AstralErr err = conv_state(agent->conv_ptr, &state);
    if (err != ASTRAL_OK) {
        agent->chat.last_error = err;
        return err;
    }
    reset_prompt_cache_stats(agent);
    if (state == ASTRAL_SESSION_DECODING) {
        agent->chat.last_error = ASTRAL_E_STATE;
        return ASTRAL_E_STATE;
    }

    uint8_t* prompt = nullptr;
    uint32_t prompt_len = 0;
    uint32_t prefix_len = 0;
    uint64_t prefix_hash = 0;
    uint64_t prompt_hash = 0;
    uint32_t* prefix_len_out = agent->desc.prompt_cache != 0 ? &prefix_len : nullptr;
    uint64_t* prefix_hash_out = agent->desc.prompt_cache != 0 ? &prefix_hash : nullptr;
    uint64_t* prompt_hash_out = agent->desc.prompt_cache != 0 ? &prompt_hash : nullptr;
    err = build_prompt(agent, desc->user_message, &prompt, &prompt_len, prefix_len_out, prefix_hash_out, prompt_hash_out);
    if (err != ASTRAL_OK) {
        agent->chat.last_error = err;
        return err;
    }

    AstralConvDesc conv_desc{};
    conv_desc.size = sizeof(AstralConvDesc);
    conv_desc.model = agent->desc.model;
    const bool warmup = (desc->flags & ASTRAL_AGENT_CHAT_FLAG_WARMUP) != 0;
    conv_desc.max_tokens = warmup ? 0 : agent->desc.max_tokens;
    conv_desc.temperature = agent->desc.temperature;
    conv_desc.top_k = agent->desc.top_k;
    conv_desc.top_p = agent->desc.top_p;
    conv_desc.stream_enabled = agent->desc.stream_enabled;
    conv_desc.seed = agent->desc.seed;

    err = prepare_generated_capture(agent, conv_desc.max_tokens);
    if (err != ASTRAL_OK) {
        agent->chat.last_error = err;
        return err;
    }

    err = conv_reset(agent->conv_ptr, &conv_desc);
    if (err == ASTRAL_OK && agent->desc.prompt_cache != 0) {
        AstralSpanU8 prompt_span{};
        prompt_span.data = prompt;
        prompt_span.len = prompt_len;
        err = feed_prompt_with_cache(agent, prompt_span, prefix_len, prefix_hash, prompt_hash);
    } else if (err == ASTRAL_OK) {
        AstralSpanU8 prompt_span{};
        prompt_span.data = prompt;
        prompt_span.len = prompt_len;
        err = conv_feed(agent->conv_ptr, prompt_span, kPromptFinalize);
    }
    if (err == ASTRAL_OK) {
        err = conv_decode(agent->conv_ptr);
    }

    if (err != ASTRAL_OK) {
        agent->chat.last_error = err;
        return err;
    }

    agent->chat.prompt_bytes = prompt_len;
    agent->chat.last_error = ASTRAL_OK;
    return ASTRAL_OK;
}

AstralErr agent_chat_cancel(Agent* agent) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = conv_cancel(agent->conv_ptr);
    agent->chat.last_error = err;
    return err;
}

int32_t agent_chat_stream_read(Agent* agent, AstralMutSpanU8 out_buf, uint32_t timeout_ms) {
    if (agent == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const int32_t result = conv_stream_read(agent->conv_ptr, out_buf, timeout_ms);
    if (result > 0) {
        append_generated_capture(*agent, out_buf.data, static_cast<uint32_t>(result));
    }
    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        agent->chat.last_error = static_cast<AstralErr>(result);
    }
    return result;
}

AstralErr agent_chat_tool_call_result(Agent* agent, AstralToolCallResult* out_result) {
    if (agent == nullptr || out_result == nullptr || out_result->size != sizeof(AstralToolCallResult)) {
        return ASTRAL_E_INVALID;
    }
    if (agent->toolset == nullptr || agent->chat.generated.len == kNoGeneratedCapture) {
        return ASTRAL_E_NOT_FOUND;
    }
    if (agent->chat.generated.truncated != 0) {
        out_result->tool_id = 0;
        out_result->parse_status = ASTRAL_E_INVALID;
        out_result->name = {};
        out_result->arguments_json = {};
        agent->chat.last_error = ASTRAL_E_INVALID;
        return ASTRAL_E_INVALID;
    }
    AstralSpanU8 generated{};
    generated.data = agent->chat.generated.bytes;
    generated.len = agent->chat.generated.len;
    const AstralErr err = toolset_parse_call(agent->toolset, generated, out_result);
    agent->chat.last_error = err;
    return err;
}

AstralErr agent_chat_result(Agent* agent, AstralAgentChatResult* out_result) {
    if (agent == nullptr || out_result == nullptr || out_result->size != sizeof(AstralAgentChatResult)) {
        return ASTRAL_E_INVALID;
    }
    AstralConvStats stats{};
    AstralErr stats_err = conv_stats(agent->conv_ptr, &stats);
    AstralSessionState state = ASTRAL_SESSION_FAILED;
    const AstralErr state_err = conv_state(agent->conv_ptr, &state);
    if (state_err != ASTRAL_OK) {
        return state_err;
    }
    out_result->state = state;
    out_result->prompt_bytes = agent->chat.prompt_bytes;
    out_result->history_messages = agent->message_count;
    out_result->prompt_tokens = stats_err == ASTRAL_OK ? stats.prompt_tokens : agent->chat.prompt_tokens;
    out_result->prompt_cache_reused_tokens = agent->chat.prompt_cache_reused_tokens;
    out_result->prompt_cache_new_tokens = agent->chat.prompt_cache_new_tokens;
    out_result->prompt_cache_hits = agent->chat.prompt_cache_hits;
    out_result->prompt_cache_misses = agent->chat.prompt_cache_misses;
    out_result->last_error = agent->chat.last_error;
    out_result->generated_tokens = stats_err == ASTRAL_OK ? stats.generated_tokens : 0;
    out_result->t_first_token_ms = stats_err == ASTRAL_OK ? stats.t_first_token_ms : 0.0;
    out_result->tok_per_s = stats_err == ASTRAL_OK ? stats.tok_per_s : 0.0;
    return ASTRAL_OK;
}

} // namespace astral::inference
