/**
 * api.cpp - C ABI exports for inference functions
 *
 * This file provides the C ABI layer for model and session operations.
 * All functions handle null checks and error propagation.
 */

#include "../../include/astral_rt.h"
#include "model.hpp"
#include "session.hpp"
#include "executor.hpp"
#include "conversation_runtime.hpp"
#include "embedder.hpp"
#include "adapter.hpp"
#include "tooling.hpp"
#include "chunking.hpp"
#include "memory_index.hpp"
#include "agent.hpp"
#include "../core/error.hpp"
#include "../core/abi_guard.hpp"
#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"
#include "../core/runtime_state.hpp"
#include "../core/model_sources.hpp"
#include "../core/model_load_config.hpp"
#include "../platform/atomics.h"
#include "../platform/time.h"
#include "../utils/trace.hpp"

#include <cstdint>
#include <cstring>

namespace {

inline void set_err_invalid(const char* what) {
    astral::core::set_last_errorf("Invalid parameter: %s", what ? what : "");
}

inline void set_err_unsupported(const char* what) {
    astral::core::set_last_errorf("Unsupported: %s", what ? what : "");
}

inline void set_err_code(AstralErr err) {
    astral::core::set_last_error_from_code(err);
}

inline astral::inference::Model* lookup_model(AstralHandle model) {
    return static_cast<astral::inference::Model*>(
        astral::core::lookup_handle(model, astral::core::HandleKind::Model)
    );
}

inline astral::inference::Toolset* lookup_toolset(AstralHandle toolset) {
    return static_cast<astral::inference::Toolset*>(
        astral::core::lookup_handle(toolset, astral::core::HandleKind::Toolset)
    );
}

inline astral::inference::MemoryIndex* lookup_memory_index(AstralHandle index) {
    return static_cast<astral::inference::MemoryIndex*>(
        astral::core::lookup_handle(index, astral::core::HandleKind::MemoryIndex)
    );
}

inline astral::inference::MemorySearchCursor* lookup_memory_search_cursor(AstralHandle cursor) {
    return static_cast<astral::inference::MemorySearchCursor*>(
        astral::core::lookup_handle(cursor, astral::core::HandleKind::MemorySearch)
    );
}

inline astral::inference::Agent* lookup_agent(AstralHandle agent) {
    return static_cast<astral::inference::Agent*>(
        astral::core::lookup_handle(agent, astral::core::HandleKind::Agent)
    );
}

inline astral::inference::Embedder* lookup_embedder(AstralHandle emb) {
    return static_cast<astral::inference::Embedder*>(
        astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder)
    );
}

inline astral::inference::Session* lookup_session(AstralHandle session) {
    return static_cast<astral::inference::Session*>(
        astral::core::lookup_handle(session, astral::core::HandleKind::Session)
    );
}

inline astral::inference::Conversation* lookup_conversation(AstralHandle conv) {
    return static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation)
    );
}

inline AstralRequestState request_state_from_session(AstralSessionState state) {
    switch (state) {
    case ASTRAL_SESSION_IDLE:
        return ASTRAL_REQUEST_QUEUED;
    case ASTRAL_SESSION_FEEDING_PROMPT:
    case ASTRAL_SESSION_DECODING:
        return ASTRAL_REQUEST_RUNNING;
    case ASTRAL_SESSION_COMPLETED:
        return ASTRAL_REQUEST_COMPLETED;
    case ASTRAL_SESSION_CANCELED:
        return ASTRAL_REQUEST_CANCELED;
    case ASTRAL_SESSION_FAILED:
        return ASTRAL_REQUEST_FAILED;
    default:
        return ASTRAL_REQUEST_INVALID;
    }
}

inline bool request_state_terminal(AstralRequestState state) {
    return state == ASTRAL_REQUEST_COMPLETED || state == ASTRAL_REQUEST_CANCELED || state == ASTRAL_REQUEST_FAILED ||
        state == ASTRAL_REQUEST_INVALID;
}

inline AstralErr request_ref_make(AstralRequestKind kind, AstralHandle owner, uint64_t ticket, AstralRequestRef* out) {
    if (out == nullptr) {
        set_err_invalid("out_request");
        return ASTRAL_E_INVALID;
    }
    out->size = sizeof(AstralRequestRef);
    out->kind = kind;
    out->owner = owner;
    out->ticket = ticket;
    return ASTRAL_OK;
}

inline bool request_ref_valid(const AstralRequestRef* request) {
    return request != nullptr && request->size == sizeof(AstralRequestRef) && request->owner != 0;
}

inline void request_status_init(const AstralRequestRef& request, AstralRequestStatus* out_status) {
    out_status->kind = request.kind;
    out_status->state = ASTRAL_REQUEST_INVALID;
    out_status->flags = request.ticket != 0 ? ASTRAL_REQUEST_FLAG_TICKET : ASTRAL_REQUEST_FLAG_STREAM;
    out_status->owner = request.owner;
    out_status->ticket = request.ticket;
    out_status->result = ASTRAL_OK;
    out_status->queue_depth = 0;
}

AstralErr request_state_impl(const AstralRequestRef* request, AstralRequestStatus* out_status) {
    if (!request_ref_valid(request) || out_status == nullptr || out_status->size != sizeof(AstralRequestStatus)) {
        set_err_invalid("request/out_status");
        return ASTRAL_E_INVALID;
    }

    request_status_init(*request, out_status);

    switch (request->kind) {
    case ASTRAL_REQUEST_SESSION: {
        auto* s = lookup_session(request->owner);
        if (s == nullptr) {
            out_status->result = ASTRAL_E_INVALID;
            return ASTRAL_E_INVALID;
        }
        astral::inference::SessionState internal_state{};
        const AstralErr err = astral::inference::session_state(s, &internal_state);
        if (err != ASTRAL_OK) {
            out_status->result = err;
            return err;
        }
        out_status->state = request_state_from_session(static_cast<AstralSessionState>(internal_state));
        return ASTRAL_OK;
    }
    case ASTRAL_REQUEST_CONVERSATION: {
        auto* c = lookup_conversation(request->owner);
        if (c == nullptr) {
            out_status->result = ASTRAL_E_INVALID;
            return ASTRAL_E_INVALID;
        }
        AstralSessionState state = ASTRAL_SESSION_FAILED;
        const AstralErr err = astral::inference::conv_state(c, &state);
        if (err != ASTRAL_OK) {
            out_status->result = err;
            return err;
        }
        out_status->state = request_state_from_session(state);
        return ASTRAL_OK;
    }
    case ASTRAL_REQUEST_AGENT_CHAT: {
        auto* a = lookup_agent(request->owner);
        if (a == nullptr) {
            out_status->result = ASTRAL_E_INVALID;
            return ASTRAL_E_INVALID;
        }
        AstralAgentChatResult result{};
        result.size = sizeof(AstralAgentChatResult);
        const AstralErr err = astral::inference::agent_chat_result(a, &result);
        if (err != ASTRAL_OK) {
            out_status->result = err;
            return err;
        }
        out_status->state = request_state_from_session(result.state);
        out_status->result = result.last_error;
        return ASTRAL_OK;
    }
    case ASTRAL_REQUEST_EMBEDDING: {
        auto* e = lookup_embedder(request->owner);
        if (e == nullptr) {
            out_status->result = ASTRAL_E_INVALID;
            return ASTRAL_E_INVALID;
        }
        return astral::inference::embedder_request_state(e, request->ticket, out_status);
    }
    case ASTRAL_REQUEST_MEMORY_SEARCH: {
        auto* cursor = lookup_memory_search_cursor(request->owner);
        if (cursor == nullptr) {
            out_status->result = ASTRAL_E_INVALID;
            return ASTRAL_E_INVALID;
        }
        AstralRequestState state = ASTRAL_REQUEST_INVALID;
        uint32_t remaining = 0;
        const AstralErr err = astral::inference::memory_search_cursor_status(cursor, &state, &remaining);
        if (err != ASTRAL_OK) {
            out_status->result = err;
            return err;
        }
        out_status->state = state;
        out_status->queue_depth = remaining;
        return ASTRAL_OK;
    }
    default:
        out_status->result = ASTRAL_E_INVALID;
        return ASTRAL_E_INVALID;
    }
}

inline AstralErr require_model_ops(AstralHandle model, astral::inference::Model** out_model) {
    if (model == 0 || out_model == nullptr) {
        set_err_invalid("model");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Model* m = lookup_model(model);
    if (m == nullptr || m->backend == nullptr || m->backend->ops == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    *out_model = m;
    return ASTRAL_OK;
}

constexpr uint8_t kPosixPathSeparator = static_cast<uint8_t>('/');
constexpr uint8_t kWindowsPathSeparator = static_cast<uint8_t>('\\');
constexpr uint8_t kWindowsDriveSeparator = static_cast<uint8_t>(':');
constexpr uint32_t kPathFirstByteOffset = 0;
constexpr uint32_t kPathLastByteBackstep = 1;
constexpr uint32_t kPathNoSeparatorBytes = 0;
constexpr uint32_t kWindowsDriveLetterOffset = 1;
constexpr uint32_t kWindowsDrivePathOffset = 2;
constexpr uint32_t kMinimumWindowsDrivePathBytes = 3;
constexpr uint32_t kPathJoinSeparatorBytes = 1;

inline bool path_span_valid(AstralSpanU8 s) {
    return s.len == 0 || s.data != nullptr;
}

inline bool path_span_has_bytes(AstralSpanU8 s) {
    return s.data != nullptr && s.len != 0;
}

inline bool path_is_separator(uint8_t c) {
    return c == kPosixPathSeparator || c == kWindowsPathSeparator;
}

inline bool path_is_windows_drive_letter(uint8_t c) {
    return (c >= static_cast<uint8_t>('A') && c <= static_cast<uint8_t>('Z')) ||
        (c >= static_cast<uint8_t>('a') && c <= static_cast<uint8_t>('z'));
}

inline bool path_is_absolute(AstralSpanU8 path) {
    if (!path_span_has_bytes(path)) {
        return false;
    }

    if (path_is_separator(path.data[kPathFirstByteOffset])) {
        return true;
    }

    if (path.len < kMinimumWindowsDrivePathBytes) {
        return false;
    }

    return path_is_windows_drive_letter(path.data[kPathFirstByteOffset]) &&
        path.data[kWindowsDriveLetterOffset] == kWindowsDriveSeparator &&
        path_is_separator(path.data[kWindowsDrivePathOffset]);
}

inline bool path_ends_with_separator(AstralSpanU8 path) {
    return path_span_has_bytes(path) && path_is_separator(path.data[path.len - kPathLastByteBackstep]);
}

inline AstralErr path_resolve_root(const AstralModelPathResolveDesc& desc, AstralSpanU8* out_root) {
    switch (desc.root) {
    case ASTRAL_MODEL_PATH_ROOT_CONTENT:
        *out_root = desc.content_root;
        return ASTRAL_OK;
    case ASTRAL_MODEL_PATH_ROOT_SAVED:
        *out_root = desc.saved_root;
        return ASTRAL_OK;
    case ASTRAL_MODEL_PATH_ROOT_CACHE:
        *out_root = desc.cache_root;
        return ASTRAL_OK;
    case ASTRAL_MODEL_PATH_ROOT_DOWNLOAD:
        *out_root = desc.download_root;
        return ASTRAL_OK;
    default:
        return ASTRAL_E_INVALID;
    }
}

inline AstralErr path_copy_span(AstralSpanU8 src, AstralMutSpanU8 out_path, uint32_t* out_len) {
    *out_len = src.len;
    if (out_path.data == nullptr || out_path.len < src.len) {
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(out_path.data, src.data, src.len);
    return ASTRAL_OK;
}

inline AstralErr path_join_spans(
    AstralSpanU8 root,
    AstralSpanU8 path,
    AstralMutSpanU8 out_path,
    uint32_t* out_len) {
    const uint32_t separator_bytes = path_ends_with_separator(root) ? kPathNoSeparatorBytes : kPathJoinSeparatorBytes;
    const uint64_t required =
        static_cast<uint64_t>(root.len) + static_cast<uint64_t>(separator_bytes) + static_cast<uint64_t>(path.len);
    if (required > UINT32_MAX) {
        *out_len = UINT32_MAX;
        return ASTRAL_E_NOMEM;
    }

    *out_len = static_cast<uint32_t>(required);
    if (out_path.data == nullptr || out_path.len < *out_len) {
        return ASTRAL_E_NOMEM;
    }

    uint8_t* cursor = out_path.data;
    std::memcpy(cursor, root.data, root.len);
    cursor += root.len;
    if (separator_bytes != kPathNoSeparatorBytes) {
        *cursor = kPosixPathSeparator;
        ++cursor;
    }
    std::memcpy(cursor, path.data, path.len);
    return ASTRAL_OK;
}

inline AstralErr model_load_impl(const AstralModelDesc* desc, AstralHandle* out_model) {
    if (desc == nullptr || out_model == nullptr) {
        set_err_invalid("desc/out_model");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralModelDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    AstralModelDesc local = *desc;

    if (desc->source_kind == ASTRAL_MODEL_SOURCE_PATH) {
        const bool has_backend_override = desc->backend_name.data != nullptr && desc->backend_name.len != 0;
        if (!has_backend_override && (desc->model_path.data == nullptr || desc->model_path.len == 0)) {
            set_err_invalid("desc.model_path");
            return ASTRAL_E_INVALID;
        }
        astral::inference::Model* model = nullptr;
        astral::core::ModelLoadConfigScope scope(desc);
        const AstralErr err = astral::inference::model_load(&local, &model);
        if (err == ASTRAL_OK) {
            *out_model = model->handle;
        } else {
            set_err_code(err);
        }
        return err;
    }

    if (desc->source_kind == ASTRAL_MODEL_SOURCE_MEMORY) {
        if (desc->model_bytes.data == nullptr || desc->model_bytes.len == 0) {
            set_err_invalid("desc.model_bytes");
            return ASTRAL_E_INVALID;
        }

        astral::core::ModelSource src{};
        src.kind = ASTRAL_MODEL_SOURCE_MEMORY;
        src.bytes = desc->model_bytes;

        char token[64]{};
        uint64_t id = 0;
        const AstralErr reg = astral::core::model_source_register(src, &id, token, sizeof(token));
        if (reg != ASTRAL_OK) {
            set_err_code(reg);
            return reg;
        }

        local.model_path.data = reinterpret_cast<const uint8_t*>(token);
        local.model_path.len = static_cast<uint32_t>(std::strlen(token));
        local.source_kind = ASTRAL_MODEL_SOURCE_PATH;

        astral::inference::Model* model = nullptr;
        astral::core::ModelLoadConfigScope scope(desc);
        const AstralErr err = astral::inference::model_load(&local, &model);
        if (astral::core::model_source_present(id)) {
            astral::core::model_source_release(id);
        }
        if (err == ASTRAL_OK) {
            *out_model = model->handle;
        } else {
            set_err_code(err);
        }
        return err;
    }

    if (desc->source_kind == ASTRAL_MODEL_SOURCE_IO) {
        if (desc->io.size == nullptr || desc->io.read_at == nullptr) {
            set_err_invalid("desc.io.size/read_at");
            return ASTRAL_E_INVALID;
        }

        astral::core::ModelSource src{};
        src.kind = ASTRAL_MODEL_SOURCE_IO;
        src.io = desc->io;

        char token[64]{};
        uint64_t id = 0;
        const AstralErr reg = astral::core::model_source_register(src, &id, token, sizeof(token));
        if (reg != ASTRAL_OK) {
            set_err_code(reg);
            return reg;
        }

        local.model_path.data = reinterpret_cast<const uint8_t*>(token);
        local.model_path.len = static_cast<uint32_t>(std::strlen(token));
        local.source_kind = ASTRAL_MODEL_SOURCE_PATH;

        astral::inference::Model* model = nullptr;
        astral::core::ModelLoadConfigScope scope(desc);
        const AstralErr err = astral::inference::model_load(&local, &model);
        if (astral::core::model_source_present(id)) {
            astral::core::model_source_release(id);
        }
        if (err == ASTRAL_OK) {
            *out_model = model->handle;
        } else {
            set_err_code(err);
        }
        return err;
    }

    set_err_invalid("desc.source_kind");
    return ASTRAL_E_INVALID;
}

inline constexpr uint32_t kPromptCacheDefaultMaxEntries = 64;
inline constexpr uint32_t kPromptCacheBytesPerKiB = 1024;
inline constexpr uint32_t kPromptCacheDefaultTokenKiB = 64;
inline constexpr uint32_t kPromptCacheDefaultMaxTokens = kPromptCacheDefaultTokenKiB * kPromptCacheBytesPerKiB;
inline constexpr uint32_t kPromptCacheMinTableCapacity = 4;
inline constexpr uint32_t kPromptCacheTableLoadFactorDen = 4;
inline constexpr uint64_t kPromptCacheHashModelMul = 0x9E3779B185EBCA87ull;
inline constexpr uint64_t kPromptCacheHashFinalMul = 0xD6E8FEB86659FD93ull;
inline constexpr uint32_t kPromptCacheHashFinalShift = 32;
inline constexpr uint32_t kPromptCacheHashGenerationShift = 32;
inline constexpr uint64_t kPromptCacheSectionHashOffset = 14695981039346656037ull;
inline constexpr uint64_t kPromptCacheSectionHashPrime = 1099511628211ull;
inline constexpr uint8_t kPromptCacheSlotEmpty = 0;
inline constexpr uint8_t kPromptCacheSlotOccupied = 1;
inline constexpr uint32_t kPromptCacheNoSlot = 0xFFFFFFFFu;
inline constexpr uint32_t kPromptCacheSaveMagic = 0x41504348u;
inline constexpr uint32_t kPromptCacheSaveVersion = 1;
inline constexpr uint32_t kPromptCacheU32Max = 0xFFFFFFFFu;

struct PromptCacheSaveHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t entry_count;
    uint32_t token_count;
};

struct PromptCacheSaveRecord {
    AstralPromptCacheKey key;
    uint64_t sequence;
    uint32_t token_count;
    uint32_t _reserved0;
};

struct PromptCacheEntry {
    AstralPromptCacheKey key{};
    int32_t* tokens = nullptr;
    uint32_t token_count = 0;
    uint32_t token_offset = 0;
    uint32_t hash = 0;
    uint32_t fifo_prev = kPromptCacheNoSlot;
    uint32_t fifo_next = kPromptCacheNoSlot;
    uint64_t sequence = 0;
    uint8_t state = kPromptCacheSlotEmpty;
};

struct PromptCache {
    AstralHandle handle = 0;
    PromptCacheEntry* entries = nullptr;
    int32_t* token_storage = nullptr;
    uint32_t max_entries = 0;
    uint32_t table_capacity = 0;
    uint32_t table_mask = 0;
    uint32_t entry_count = 0;
    uint32_t max_tokens = 0;
    uint32_t token_count = 0;
    uint32_t token_write_offset = 0;
    uint32_t max_bytes = 0;
    uint32_t fifo_head = kPromptCacheNoSlot;
    uint32_t fifo_tail = kPromptCacheNoSlot;
    uint32_t last_hit_slot = kPromptCacheNoSlot;
    uint32_t last_hit_hash = 0;
    uint8_t token_arena_fragmented = 0;
    uint8_t track_stats = 0;
    uint64_t next_sequence = 1;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

inline bool prompt_section_kind_valid(uint32_t kind) {
    return kind >= ASTRAL_PROMPT_SECTION_SYSTEM && kind <= ASTRAL_PROMPT_SECTION_RAW;
}

inline bool prompt_cache_key_valid(const AstralPromptCacheKey* key) {
    return key != nullptr && key->size == sizeof(AstralPromptCacheKey) &&
           prompt_section_kind_valid(key->section_kind) && key->model != 0;
}

inline bool prompt_cache_key_equal(const AstralPromptCacheKey& a, const AstralPromptCacheKey& b) {
    return a.key == b.key && a.model == b.model && a.generation == b.generation && a.section_kind == b.section_kind;
}

inline uint32_t prompt_cache_hash(const AstralPromptCacheKey& key) {
    uint64_t x = key.key;
    x ^= key.model * kPromptCacheHashModelMul;
    x ^= static_cast<uint64_t>(key.generation) << kPromptCacheHashGenerationShift;
    x ^= static_cast<uint64_t>(key.section_kind);
    x *= kPromptCacheHashFinalMul;
    x ^= x >> kPromptCacheHashFinalShift;
    return static_cast<uint32_t>(x);
}

inline uint64_t prompt_cache_hash_bytes(AstralSpanU8 bytes) {
    uint64_t hash = kPromptCacheSectionHashOffset;
    for (uint32_t i = 0; i < bytes.len; ++i) {
        hash ^= static_cast<uint64_t>(bytes.data[i]);
        hash *= kPromptCacheSectionHashPrime;
    }
    hash ^= static_cast<uint64_t>(bytes.len);
    hash *= kPromptCacheSectionHashPrime;
    return hash;
}

inline uint32_t prompt_cache_table_capacity(uint32_t max_entries) {
    uint32_t capacity = 1;
    const uint32_t target = max_entries < kPromptCacheTableLoadFactorDen
        ? kPromptCacheMinTableCapacity
        : max_entries * kPromptCacheTableLoadFactorDen;
    while (capacity < target) {
        capacity <<= 1u;
    }
    return capacity;
}

inline PromptCache* lookup_prompt_cache(AstralHandle cache) {
    return static_cast<PromptCache*>(astral::core::lookup_handle(cache, astral::core::HandleKind::PromptCache));
}

inline PromptCacheEntry* prompt_cache_empty_entry(PromptCache* cache, uint32_t hash);

inline uint32_t prompt_cache_slot_index(const PromptCache* cache, const PromptCacheEntry* entry) {
    return static_cast<uint32_t>(entry - cache->entries);
}

inline void prompt_cache_forget_last_hit(PromptCache* cache) {
    cache->last_hit_slot = kPromptCacheNoSlot;
    cache->last_hit_hash = 0;
}

inline void prompt_cache_remember_last_hit(PromptCache* cache, PromptCacheEntry* entry) {
    cache->last_hit_slot = prompt_cache_slot_index(cache, entry);
    cache->last_hit_hash = entry->hash;
}

inline void prompt_cache_fifo_append(PromptCache* cache, PromptCacheEntry& entry) {
    const uint32_t slot = prompt_cache_slot_index(cache, &entry);
    entry.fifo_prev = cache->fifo_tail;
    entry.fifo_next = kPromptCacheNoSlot;
    if (cache->fifo_tail != kPromptCacheNoSlot) {
        cache->entries[cache->fifo_tail].fifo_next = slot;
    } else {
        cache->fifo_head = slot;
    }
    cache->fifo_tail = slot;
}

inline void prompt_cache_fifo_unlink(PromptCache* cache, PromptCacheEntry& entry) {
    const uint32_t slot = prompt_cache_slot_index(cache, &entry);
    if (entry.fifo_prev != kPromptCacheNoSlot) {
        cache->entries[entry.fifo_prev].fifo_next = entry.fifo_next;
    } else if (cache->fifo_head == slot) {
        cache->fifo_head = entry.fifo_next;
    }

    if (entry.fifo_next != kPromptCacheNoSlot) {
        cache->entries[entry.fifo_next].fifo_prev = entry.fifo_prev;
    } else if (cache->fifo_tail == slot) {
        cache->fifo_tail = entry.fifo_prev;
    }

    entry.fifo_prev = kPromptCacheNoSlot;
    entry.fifo_next = kPromptCacheNoSlot;
}

inline void prompt_cache_fifo_relocate(PromptCache* cache, uint32_t old_slot, uint32_t new_slot) {
    PromptCacheEntry& entry = cache->entries[new_slot];
    if (entry.fifo_prev != kPromptCacheNoSlot) {
        cache->entries[entry.fifo_prev].fifo_next = new_slot;
    } else if (cache->fifo_head == old_slot) {
        cache->fifo_head = new_slot;
    }

    if (entry.fifo_next != kPromptCacheNoSlot) {
        cache->entries[entry.fifo_next].fifo_prev = new_slot;
    } else if (cache->fifo_tail == old_slot) {
        cache->fifo_tail = new_slot;
    }
}

inline bool prompt_cache_token_ranges_overlap(
    uint32_t first_offset,
    uint32_t first_count,
    uint32_t second_offset,
    uint32_t second_count
) {
    const uint32_t first_end = first_offset + first_count;
    const uint32_t second_end = second_offset + second_count;
    return first_offset < second_end && second_offset < first_end;
}

inline bool prompt_cache_token_range_any_overlap(PromptCache* cache, uint32_t offset, uint32_t count) {
    for (uint32_t slot = cache->fifo_head; slot != kPromptCacheNoSlot; slot = cache->entries[slot].fifo_next) {
        const PromptCacheEntry& entry = cache->entries[slot];
        if (prompt_cache_token_ranges_overlap(offset, count, entry.token_offset, entry.token_count)) {
            return true;
        }
    }
    return false;
}

inline bool prompt_cache_compact_tokens(PromptCache* cache) {
    uint32_t previous_end = 0;
    for (uint32_t slot = cache->fifo_head; slot != kPromptCacheNoSlot; slot = cache->entries[slot].fifo_next) {
        const PromptCacheEntry& entry = cache->entries[slot];
        if (entry.token_offset < previous_end) {
            return false;
        }
        previous_end = entry.token_offset + entry.token_count;
    }

    uint32_t offset = 0;
    for (uint32_t slot = cache->fifo_head; slot != kPromptCacheNoSlot; slot = cache->entries[slot].fifo_next) {
        PromptCacheEntry& entry = cache->entries[slot];
        if (entry.token_count != 0 && entry.token_offset != offset) {
            std::memmove(cache->token_storage + offset, entry.tokens,
                         static_cast<size_t>(entry.token_count) * sizeof(int32_t));
            entry.tokens = cache->token_storage + offset;
            entry.token_offset = offset;
        }
        offset += entry.token_count;
    }
    cache->token_write_offset = offset;
    cache->token_arena_fragmented = 0;
    return true;
}

inline void prompt_cache_release_entry_payload(PromptCache* cache, PromptCacheEntry& entry, uint8_t next_state) {
    if (entry.state != kPromptCacheSlotOccupied) {
        return;
    }
    prompt_cache_fifo_unlink(cache, entry);
    entry.tokens = nullptr;
    cache->token_count = cache->token_count >= entry.token_count ? cache->token_count - entry.token_count : 0;
    entry.token_count = 0;
    entry.token_offset = 0;
    entry.hash = 0;
    entry.sequence = 0;
    entry.state = next_state;
    entry.key = {};
}

inline void prompt_cache_remove_entry(PromptCache* cache, PromptCacheEntry* entry) {
    if (entry == nullptr || entry->state != kPromptCacheSlotOccupied) {
        return;
    }

    prompt_cache_forget_last_hit(cache);
    const bool compact_after_remove = prompt_cache_slot_index(cache, entry) != cache->fifo_head;
    const uint32_t hole = prompt_cache_slot_index(cache, entry);
    prompt_cache_release_entry_payload(cache, *entry, kPromptCacheSlotEmpty);

    uint32_t slot = (hole + 1u) & cache->table_mask;
    while (cache->entries[slot].state == kPromptCacheSlotOccupied) {
        PromptCacheEntry moved = cache->entries[slot];
        cache->entries[slot] = {};

        PromptCacheEntry* dst = prompt_cache_empty_entry(cache, moved.hash);
        *dst = moved;
        prompt_cache_fifo_relocate(cache, slot, prompt_cache_slot_index(cache, dst));

        slot = (slot + 1u) & cache->table_mask;
    }
    if (compact_after_remove) {
        cache->token_arena_fragmented = prompt_cache_compact_tokens(cache) ? 0u : 1u;
    }
}

inline void prompt_cache_clear_impl(PromptCache* cache) {
    for (uint32_t i = 0; i < cache->table_capacity; ++i) {
        prompt_cache_release_entry_payload(cache, cache->entries[i], kPromptCacheSlotEmpty);
        cache->entries[i].state = kPromptCacheSlotEmpty;
    }
    cache->fifo_head = kPromptCacheNoSlot;
    cache->fifo_tail = kPromptCacheNoSlot;
    cache->token_write_offset = 0;
    cache->token_arena_fragmented = 0;
    cache->entry_count = 0;
    prompt_cache_forget_last_hit(cache);
}

inline void prompt_cache_destroy_impl(PromptCache* cache) {
    if (cache == nullptr) {
        return;
    }
    prompt_cache_clear_impl(cache);
    astral::core::runtime_free_array(cache->token_storage, cache->max_tokens);
    cache->token_storage = nullptr;
    astral::core::runtime_free_array(cache->entries, cache->table_capacity);
    cache->entries = nullptr;
    astral::core::runtime_delete(cache);
}

inline PromptCacheEntry* prompt_cache_find_hashed(PromptCache* cache, const AstralPromptCacheKey* key, uint32_t hash) {
    if (cache->last_hit_slot != kPromptCacheNoSlot && cache->last_hit_hash == hash) {
        PromptCacheEntry& cached = cache->entries[cache->last_hit_slot];
        if (cached.state == kPromptCacheSlotOccupied && prompt_cache_key_equal(cached.key, *key)) {
            return &cached;
        }
    }

    uint32_t slot = hash & cache->table_mask;
    for (uint32_t probe = 0; probe < cache->table_capacity; ++probe) {
        PromptCacheEntry& entry = cache->entries[slot];
        if (entry.state == kPromptCacheSlotEmpty) {
            return nullptr;
        }
        if (entry.hash == hash && prompt_cache_key_equal(entry.key, *key)) {
            prompt_cache_remember_last_hit(cache, &entry);
            return &entry;
        }
        slot = (slot + 1u) & cache->table_mask;
    }
    return nullptr;
}

inline PromptCacheEntry* prompt_cache_find(PromptCache* cache, const AstralPromptCacheKey* key) {
    return prompt_cache_find_hashed(cache, key, prompt_cache_hash(*key));
}

inline PromptCacheEntry* prompt_cache_empty_entry(PromptCache* cache, uint32_t hash) {
    uint32_t slot = hash & cache->table_mask;
    for (uint32_t probe = 0; probe < cache->table_capacity; ++probe) {
        PromptCacheEntry& entry = cache->entries[slot];
        if (entry.state == kPromptCacheSlotEmpty) {
            return &entry;
        }
        slot = (slot + 1u) & cache->table_mask;
    }
    return nullptr;
}

inline PromptCacheEntry* prompt_cache_oldest_entry(PromptCache* cache) {
    return cache->fifo_head == kPromptCacheNoSlot ? nullptr : &cache->entries[cache->fifo_head];
}

inline bool prompt_cache_evict_oldest(PromptCache* cache) {
    PromptCacheEntry* victim = prompt_cache_oldest_entry(cache);
    if (victim == nullptr) {
        return false;
    }
    prompt_cache_remove_entry(cache, victim);
    --cache->entry_count;
    if (cache->track_stats != 0) {
        ++cache->evictions;
    }
    return true;
}

inline bool prompt_cache_head_token_overlap(PromptCache* cache, uint32_t offset, uint32_t count) {
    PromptCacheEntry* head = prompt_cache_oldest_entry(cache);
    return head != nullptr && prompt_cache_token_ranges_overlap(offset, count, head->token_offset, head->token_count);
}

inline bool prompt_cache_reserve_tokens(PromptCache* cache, uint32_t token_count, uint32_t* out_offset) {
    while (cache->entry_count >= cache->max_entries || cache->token_count + token_count > cache->max_tokens) {
        if (!prompt_cache_evict_oldest(cache)) {
            return false;
        }
    }

    uint32_t offset = cache->token_write_offset;
    if (offset + token_count > cache->max_tokens) {
        offset = 0;
    }

    if (cache->token_arena_fragmented != 0) {
        while (prompt_cache_token_range_any_overlap(cache, offset, token_count)) {
            if (!prompt_cache_evict_oldest(cache)) {
                return false;
            }
        }
    } else {
        while (prompt_cache_head_token_overlap(cache, offset, token_count)) {
            if (!prompt_cache_evict_oldest(cache)) {
                return false;
            }
        }
    }

    *out_offset = offset;
    cache->token_write_offset = offset + token_count;
    return true;
}

inline bool prompt_cache_saved_size_impl(PromptCache* cache, uint32_t* out_bytes) {
    uint64_t bytes = sizeof(PromptCacheSaveHeader);
    for (uint32_t i = 0; i < cache->table_capacity; ++i) {
        const PromptCacheEntry& entry = cache->entries[i];
        if (entry.state != kPromptCacheSlotOccupied) {
            continue;
        }
        bytes += sizeof(PromptCacheSaveRecord);
        bytes += static_cast<uint64_t>(entry.token_count) * sizeof(int32_t);
        if (bytes > kPromptCacheU32Max) {
            return false;
        }
    }
    *out_bytes = static_cast<uint32_t>(bytes);
    return true;
}

} // namespace

extern "C" {

// ============================================================================
// Model API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_model_path_resolve(
    const AstralModelPathResolveDesc* desc,
    AstralMutSpanU8 out_path,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
    if (desc == nullptr || out_len == nullptr) {
        set_err_invalid("desc/out_len");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralModelPathResolveDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }
    if (desc->flags != ASTRAL_MODEL_PATH_RESOLVE_NONE || desc->_reserved0 != 0u) {
        set_err_invalid("desc.flags");
        return ASTRAL_E_INVALID;
    }
    if (!path_span_has_bytes(desc->path)) {
        set_err_invalid("desc.path");
        return ASTRAL_E_INVALID;
    }
    if (!path_span_valid(desc->content_root) ||
        !path_span_valid(desc->saved_root) ||
        !path_span_valid(desc->cache_root) ||
        !path_span_valid(desc->download_root)) {
        set_err_invalid("desc.root");
        return ASTRAL_E_INVALID;
    }

    if (desc->root == ASTRAL_MODEL_PATH_ROOT_RAW || path_is_absolute(desc->path)) {
        const AstralErr err = path_copy_span(desc->path, out_path, out_len);
        if (err != ASTRAL_OK) {
            set_err_code(err);
        }
        return err;
    }

    AstralSpanU8 root{};
    AstralErr err = path_resolve_root(*desc, &root);
    if (err != ASTRAL_OK || !path_span_has_bytes(root)) {
        set_err_invalid("desc.root");
        return ASTRAL_E_INVALID;
    }

    err = path_join_spans(root, desc->path, out_path, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_load(
    const AstralModelDesc* desc,
    AstralHandle* out_model
) {
    ASTRAL_ABI_TRY_BEGIN
    return model_load_impl(desc, out_model);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_load2(
    const AstralModelDesc* desc,
    AstralHandle* out_model
) {
    ASTRAL_ABI_TRY_BEGIN
    return model_load_impl(desc, out_model);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_model_release(AstralHandle model) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0) {
        set_err_invalid("model");
        return;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return;
    }

    astral::inference::model_release(m);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_adapter_load(
    AstralHandle model,
    const AstralAdapterDesc* desc,
    AstralHandle* out_adapter
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.adapter_load");
    if (model == 0 || desc == nullptr || out_adapter == nullptr) {
        set_err_invalid("model/desc/out_adapter");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Adapter* a = nullptr;
    const AstralErr err = astral::inference::adapter_load(m, desc, &a);
    if (err == ASTRAL_OK) {
        *out_adapter = a->handle;
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_model_adapter_release(AstralHandle adapter) {
    ASTRAL_ABI_TRY_BEGIN
    if (adapter == 0) {
        set_err_invalid("adapter");
        return;
    }

    auto* a =
        static_cast<astral::inference::Adapter*>(astral::core::lookup_handle(adapter, astral::core::HandleKind::Adapter));
    if (a == nullptr) {
        set_err_invalid("adapter (invalid handle)");
        return;
    }

    astral::inference::adapter_release(a);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_adapter_info(AstralHandle adapter, AstralAdapterInfo* out_info) {
    ASTRAL_ABI_TRY_BEGIN
    if (adapter == 0 || out_info == nullptr) {
        set_err_invalid("adapter/out_info");
        return ASTRAL_E_INVALID;
    }
    auto* a =
        static_cast<astral::inference::Adapter*>(astral::core::lookup_handle(adapter, astral::core::HandleKind::Adapter));
    if (a == nullptr) {
        set_err_invalid("adapter (invalid handle)");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::adapter_info(a, out_info);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_adapter_path_copy(
    AstralHandle adapter,
    AstralMutSpanU8 out_path,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
    if (adapter == 0 || out_len == nullptr) {
        set_err_invalid("adapter/out_len");
        return ASTRAL_E_INVALID;
    }
    auto* a =
        static_cast<astral::inference::Adapter*>(astral::core::lookup_handle(adapter, astral::core::HandleKind::Adapter));
    if (a == nullptr) {
        set_err_invalid("adapter (invalid handle)");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::adapter_path_copy(a, out_path, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_create(const AstralToolsetDesc* desc, AstralHandle* out_toolset) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.toolset_create");
    if (desc == nullptr || out_toolset == nullptr) {
        set_err_invalid("desc/out_toolset");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Toolset* toolset = nullptr;
    const AstralErr err = astral::inference::toolset_create(desc, &toolset);
    if (err == ASTRAL_OK) {
        *out_toolset = toolset->handle;
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_toolset_destroy(AstralHandle toolset) {
    ASTRAL_ABI_TRY_BEGIN
    if (toolset == 0) {
        set_err_invalid("toolset");
        return;
    }

    auto* ts = lookup_toolset(toolset);
    if (ts == nullptr) {
        set_err_invalid("toolset (invalid handle)");
        return;
    }
    astral::inference::toolset_release(ts);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_count(AstralHandle toolset, uint32_t* out_count) {
    ASTRAL_ABI_TRY_BEGIN
    if (toolset == 0 || out_count == nullptr) {
        set_err_invalid("toolset/out_count");
        return ASTRAL_E_INVALID;
    }

    auto* ts = lookup_toolset(toolset);
    if (ts == nullptr) {
        set_err_invalid("toolset (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::toolset_count(ts, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_get(AstralHandle toolset, uint32_t index, AstralToolInfo* out_info) {
    ASTRAL_ABI_TRY_BEGIN
    if (toolset == 0 || out_info == nullptr) {
        set_err_invalid("toolset/out_info");
        return ASTRAL_E_INVALID;
    }

    auto* ts = lookup_toolset(toolset);
    if (ts == nullptr) {
        set_err_invalid("toolset (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::toolset_get(ts, index, out_info);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_parse_call(
    AstralHandle toolset,
    AstralSpanU8 generated_text,
    AstralToolCallResult* out_result
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.toolset_parse_call");
    if (toolset == 0 || out_result == nullptr) {
        set_err_invalid("toolset/out_result");
        return ASTRAL_E_INVALID;
    }

    auto* ts = lookup_toolset(toolset);
    if (ts == nullptr) {
        set_err_invalid("toolset (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::toolset_parse_call(ts, generated_text, out_result);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_create(const AstralPromptCacheDesc* desc, AstralHandle* out_cache) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.prompt_cache_create");
    if (desc == nullptr || out_cache == nullptr || desc->size != sizeof(AstralPromptCacheDesc)) {
        set_err_invalid("desc/out_cache");
        return ASTRAL_E_INVALID;
    }
    if (desc->eviction_policy != ASTRAL_PROMPT_CACHE_EVICT_FIFO) {
        set_err_unsupported("prompt cache eviction policy");
        return ASTRAL_E_UNSUPPORTED;
    }

    const uint32_t max_entries = desc->max_entries != 0 ? desc->max_entries : kPromptCacheDefaultMaxEntries;
    uint32_t max_tokens = desc->max_tokens != 0 ? desc->max_tokens : kPromptCacheDefaultMaxTokens;
    const uint32_t byte_token_cap = desc->max_bytes / static_cast<uint32_t>(sizeof(int32_t));
    if (byte_token_cap != 0 && byte_token_cap < max_tokens) {
        max_tokens = byte_token_cap;
    }
    if (max_entries == 0 || max_tokens == 0) {
        set_err_invalid("desc budgets");
        return ASTRAL_E_INVALID;
    }

    PromptCache* cache = astral::core::runtime_new<PromptCache>();
    if (cache == nullptr) {
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }

    const uint32_t table_capacity = prompt_cache_table_capacity(max_entries);
    cache->entries = astral::core::runtime_alloc_array<PromptCacheEntry>(table_capacity);
    if (cache->entries == nullptr) {
        astral::core::runtime_delete(cache);
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }
    cache->token_storage = astral::core::runtime_alloc_array<int32_t>(max_tokens);
    if (cache->token_storage == nullptr) {
        astral::core::runtime_free_array(cache->entries, table_capacity);
        astral::core::runtime_delete(cache);
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }
    for (uint32_t i = 0; i < table_capacity; ++i) {
        new (&cache->entries[i]) PromptCacheEntry();
    }
    cache->max_entries = max_entries;
    cache->table_capacity = table_capacity;
    cache->table_mask = table_capacity - 1u;
    cache->max_tokens = max_tokens;
    cache->max_bytes = max_tokens * static_cast<uint32_t>(sizeof(int32_t));
    cache->track_stats = (desc->flags & ASTRAL_PROMPT_CACHE_FLAG_TRACK_STATS) != 0 ? 1u : 0u;

    const AstralHandle handle = astral::core::register_handle(astral::core::HandleKind::PromptCache, cache);
    if (handle == 0) {
        prompt_cache_destroy_impl(cache);
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }

    cache->handle = handle;
    *out_cache = handle;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_prompt_cache_destroy(AstralHandle cache) {
    ASTRAL_ABI_TRY_BEGIN
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr) {
        set_err_invalid("cache");
        return;
    }
    astral::core::unregister_handle(cache, astral::core::HandleKind::PromptCache);
    prompt_cache_destroy_impl(c);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_clear(AstralHandle cache) {
    ASTRAL_ABI_TRY_BEGIN
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr) {
        set_err_invalid("cache");
        return ASTRAL_E_INVALID;
    }
    prompt_cache_clear_impl(c);
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_stats(AstralHandle cache, AstralPromptCacheStats* out_stats) {
    ASTRAL_ABI_TRY_BEGIN
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr || out_stats == nullptr || out_stats->size != sizeof(AstralPromptCacheStats)) {
        set_err_invalid("cache/out_stats");
        return ASTRAL_E_INVALID;
    }
    out_stats->entries = c->entry_count;
    out_stats->max_entries = c->max_entries;
    out_stats->tokens = c->token_count;
    out_stats->max_tokens = c->max_tokens;
    out_stats->bytes = c->token_count * static_cast<uint32_t>(sizeof(int32_t));
    out_stats->max_bytes = c->max_bytes;
    out_stats->hits = c->hits;
    out_stats->misses = c->misses;
    out_stats->evictions = c->evictions;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_save_size(AstralHandle cache, uint32_t* out_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr || out_bytes == nullptr) {
        set_err_invalid("cache/out_bytes");
        return ASTRAL_E_INVALID;
    }
    if (!prompt_cache_saved_size_impl(c, out_bytes)) {
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_save(AstralHandle cache, AstralMutSpanU8 out_bytes, uint32_t* out_len) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.prompt_cache_save");
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr || out_len == nullptr || out_bytes.data == nullptr) {
        set_err_invalid("cache/out_bytes/out_len");
        return ASTRAL_E_INVALID;
    }

    uint32_t needed = 0;
    if (!prompt_cache_saved_size_impl(c, &needed)) {
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }
    *out_len = needed;
    if (out_bytes.len < needed) {
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }

    PromptCacheSaveHeader header{};
    header.magic = kPromptCacheSaveMagic;
    header.version = kPromptCacheSaveVersion;
    header.entry_count = c->entry_count;
    header.token_count = c->token_count;

    uint8_t* cursor = out_bytes.data;
    std::memcpy(cursor, &header, sizeof(header));
    cursor += sizeof(header);

    for (uint32_t slot = c->fifo_head; slot != kPromptCacheNoSlot; slot = c->entries[slot].fifo_next) {
        const PromptCacheEntry& entry = c->entries[slot];
        PromptCacheSaveRecord record{};
        record.key = entry.key;
        record.sequence = entry.sequence;
        record.token_count = entry.token_count;
        std::memcpy(cursor, &record, sizeof(record));
        cursor += sizeof(record);

        const uint32_t token_bytes = entry.token_count * static_cast<uint32_t>(sizeof(int32_t));
        if (token_bytes != 0) {
            std::memcpy(cursor, entry.tokens, token_bytes);
            cursor += token_bytes;
        }
    }

    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_load(
    const AstralPromptCacheDesc* desc,
    AstralSpanU8 bytes,
    AstralHandle* out_cache
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.prompt_cache_load");
    if (desc == nullptr || out_cache == nullptr || bytes.data == nullptr || bytes.len < sizeof(PromptCacheSaveHeader)) {
        set_err_invalid("desc/bytes/out_cache");
        return ASTRAL_E_INVALID;
    }

    PromptCacheSaveHeader header{};
    const uint8_t* cursor = bytes.data;
    const uint8_t* end = bytes.data + bytes.len;
    std::memcpy(&header, cursor, sizeof(header));
    cursor += sizeof(header);
    if (header.magic != kPromptCacheSaveMagic || header.version != kPromptCacheSaveVersion) {
        set_err_invalid("prompt cache bytes");
        return ASTRAL_E_INVALID;
    }

    uint64_t record_tokens = 0;
    AstralHandle cache = 0;
    AstralErr err = astral_prompt_cache_create(desc, &cache);
    if (err != ASTRAL_OK) {
        return err;
    }

    for (uint32_t i = 0; i < header.entry_count; ++i) {
        if (static_cast<uint64_t>(end - cursor) < sizeof(PromptCacheSaveRecord)) {
            astral_prompt_cache_destroy(cache);
            set_err_invalid("prompt cache record");
            return ASTRAL_E_INVALID;
        }

        PromptCacheSaveRecord record{};
        std::memcpy(&record, cursor, sizeof(record));
        cursor += sizeof(record);
        if (!prompt_cache_key_valid(&record.key)) {
            astral_prompt_cache_destroy(cache);
            set_err_invalid("prompt cache key");
            return ASTRAL_E_INVALID;
        }

        const uint64_t token_bytes = static_cast<uint64_t>(record.token_count) * sizeof(int32_t);
        if (static_cast<uint64_t>(end - cursor) < token_bytes) {
            astral_prompt_cache_destroy(cache);
            set_err_invalid("prompt cache tokens");
            return ASTRAL_E_INVALID;
        }

        err = astral_prompt_cache_put_tokens(cache, &record.key, reinterpret_cast<const int32_t*>(cursor), record.token_count);
        if (err != ASTRAL_OK) {
            astral_prompt_cache_destroy(cache);
            return err;
        }
        PromptCache* c = lookup_prompt_cache(cache);
        PromptCacheEntry* entry = c != nullptr ? prompt_cache_find(c, &record.key) : nullptr;
        if (entry != nullptr) {
            entry->sequence = record.sequence;
            if (c->next_sequence <= record.sequence) {
                c->next_sequence = record.sequence + 1u;
            }
        }
        record_tokens += record.token_count;
        cursor += token_bytes;
    }

    if (cursor != end || record_tokens != header.token_count) {
        astral_prompt_cache_destroy(cache);
        set_err_invalid("prompt cache size");
        return ASTRAL_E_INVALID;
    }

    *out_cache = cache;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_key_from_bytes(
    AstralHandle model,
    AstralPromptSectionKind section_kind,
    uint32_t generation,
    AstralSpanU8 bytes,
    AstralPromptCacheKey* out_key
) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || out_key == nullptr || !prompt_section_kind_valid(section_kind) ||
        (bytes.len != 0 && bytes.data == nullptr)) {
        set_err_invalid("model/section_kind/bytes/out_key");
        return ASTRAL_E_INVALID;
    }
    out_key->size = sizeof(AstralPromptCacheKey);
    out_key->section_kind = section_kind;
    out_key->model = model;
    out_key->key = prompt_cache_hash_bytes(bytes);
    out_key->generation = generation;
    out_key->_reserved0 = 0;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_put_tokens(
    AstralHandle cache,
    const AstralPromptCacheKey* key,
    const int32_t* tokens,
    uint32_t token_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.prompt_cache_put_tokens");
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr || !prompt_cache_key_valid(key) || (token_count != 0 && tokens == nullptr)) {
        set_err_invalid("cache/key/tokens");
        return ASTRAL_E_INVALID;
    }
    if (token_count > c->max_tokens) {
        set_err_invalid("token_count");
        return ASTRAL_E_INVALID;
    }

    const uint32_t hash = prompt_cache_hash(*key);
    PromptCacheEntry* entry = prompt_cache_find_hashed(c, key, hash);
    if (entry != nullptr) {
        prompt_cache_remove_entry(c, entry);
        --c->entry_count;
    }

    uint32_t token_offset = 0;
    if (!prompt_cache_reserve_tokens(c, token_count, &token_offset)) {
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }

    entry = prompt_cache_empty_entry(c, hash);
    if (entry == nullptr) {
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }

    int32_t* copy = nullptr;
    if (token_count != 0) {
        copy = c->token_storage + token_offset;
        std::memcpy(copy, tokens, static_cast<size_t>(token_count) * sizeof(int32_t));
    }

    entry->key = *key;
    entry->tokens = copy;
    entry->token_count = token_count;
    entry->token_offset = token_offset;
    entry->hash = hash;
    entry->sequence = c->next_sequence++;
    entry->state = kPromptCacheSlotOccupied;
    prompt_cache_fifo_append(c, *entry);
    prompt_cache_remember_last_hit(c, entry);
    c->token_count += token_count;
    ++c->entry_count;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_get_tokens(
    AstralHandle cache,
    const AstralPromptCacheKey* key,
    int32_t* out_tokens,
    uint32_t max_tokens,
    uint32_t* out_token_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.prompt_cache_get_tokens");
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr || !prompt_cache_key_valid(key) || out_token_count == nullptr) {
        set_err_invalid("cache/key/out_token_count");
        return ASTRAL_E_INVALID;
    }

    PromptCacheEntry* entry = prompt_cache_find(c, key);
    if (entry == nullptr) {
        *out_token_count = 0;
        if (c->track_stats != 0) {
            ++c->misses;
        }
        set_err_code(ASTRAL_E_NOT_FOUND);
        return ASTRAL_E_NOT_FOUND;
    }

    *out_token_count = entry->token_count;
    if (entry->token_count > max_tokens || (entry->token_count != 0 && out_tokens == nullptr)) {
        if (c->track_stats != 0) {
            ++c->misses;
        }
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }
    if (entry->token_count != 0) {
        std::memcpy(out_tokens, entry->tokens, static_cast<size_t>(entry->token_count) * sizeof(int32_t));
    }
    if (c->track_stats != 0) {
        ++c->hits;
    }
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_get_token_view(
    AstralHandle cache,
    const AstralPromptCacheKey* key,
    const int32_t** out_tokens,
    uint32_t* out_token_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.prompt_cache_get_token_view");
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr || !prompt_cache_key_valid(key) || out_tokens == nullptr || out_token_count == nullptr) {
        set_err_invalid("cache/key/out_tokens/out_token_count");
        return ASTRAL_E_INVALID;
    }

    PromptCacheEntry* entry = prompt_cache_find(c, key);
    if (entry == nullptr) {
        *out_tokens = nullptr;
        *out_token_count = 0;
        if (c->track_stats != 0) {
            ++c->misses;
        }
        set_err_code(ASTRAL_E_NOT_FOUND);
        return ASTRAL_E_NOT_FOUND;
    }

    *out_tokens = entry->tokens;
    *out_token_count = entry->token_count;
    if (c->track_stats != 0) {
        ++c->hits;
    }
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_info(AstralHandle model, AstralModelInfo* out_info) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || out_info == nullptr) {
        set_err_invalid("model/out_info");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    uint32_t vocab_size = 0;
    uint32_t ctx_size = 0;
    AstralErr err = m->backend->ops->model_info(m->backend_model_ctx, &vocab_size, &ctx_size);
    if (err != ASTRAL_OK) {
        set_err_code(err);
        return err;
    }

    int32_t bos = -1;
    int32_t eos = -1;
    if (m->backend->ops->model_special_tokens != nullptr) {
        err = m->backend->ops->model_special_tokens(m->backend_model_ctx, &bos, &eos);
        if (err != ASTRAL_OK) {
            set_err_code(err);
            return err;
        }
    }

    out_info->vocab_size = vocab_size;
    out_info->ctx_size = ctx_size;
    out_info->token_bos = bos;
    out_info->token_eos = eos;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_caps(AstralHandle model, AstralCaps* out_caps) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || out_caps == nullptr) {
        set_err_invalid("model/out_caps");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    AstralCaps caps = 0;
    caps |= ASTRAL_CAP_SAMPLER_EXT;
    caps |= ASTRAL_CAP_STOP_SEQS;
    caps |= ASTRAL_CAP_LOGPROBS;

    if (m->backend != nullptr && m->backend->ops != nullptr) {
        if (m->backend->ops->embedder_create != nullptr && m->backend->ops->embedder_embed != nullptr) {
            caps |= ASTRAL_CAP_EMBEDDINGS;
        }
        if (m->backend->supports_gpu) {
            caps |= ASTRAL_CAP_GPU_OFFLOAD;
        }

        if (m->backend->ops->session_state_size != nullptr && m->backend->ops->session_state_save != nullptr &&
            m->backend->ops->session_state_load != nullptr) {
            caps |= ASTRAL_CAP_KV_STATE;
        }

        if (m->backend->ops->model_adapter_load != nullptr && m->backend->ops->model_adapter_unload != nullptr &&
            m->backend->ops->session_adapter_clear != nullptr && m->backend->ops->session_adapter_add != nullptr) {
            caps |= ASTRAL_CAP_LORA;
        }

        if (m->backend->ops->session_grammar_set_gbnf != nullptr && m->backend->ops->session_grammar_clear != nullptr &&
            m->backend->ops->session_apply_grammar != nullptr) {
            caps |= ASTRAL_CAP_GRAMMAR;
            caps |= ASTRAL_CAP_GRAMMAR_GBNF;
        }

        if (m->backend->ops->session_grammar_set_json_schema != nullptr && m->backend->ops->session_grammar_clear != nullptr &&
            m->backend->ops->session_apply_grammar != nullptr) {
            caps |= ASTRAL_CAP_GRAMMAR;
            caps |= ASTRAL_CAP_GRAMMAR_JSON_SCHEMA;
        }

        if (m->backend->ops->session_set_slot != nullptr) {
            caps |= ASTRAL_CAP_SLOTS;
        }

        if (m->backend->ops->model_media_info != nullptr) {
            AstralMediaInfo info{};
            info.size = sizeof(AstralMediaInfo);
            if (m->backend->ops->model_media_info(m->backend_model_ctx, &info) == ASTRAL_OK) {
                if (info.supports_image != 0) {
                    caps |= ASTRAL_CAP_IMAGE;
                }
                if (info.supports_audio != 0) {
                    caps |= ASTRAL_CAP_AUDIO;
                }
            }
        }

        if ((caps & (ASTRAL_CAP_IMAGE | ASTRAL_CAP_AUDIO)) != 0) {
            if (m->backend->ops->embedder_embed_multimodal != nullptr ||
                m->backend->ops->embedder_embed_image != nullptr ||
                m->backend->ops->embedder_embed_audio != nullptr) {
                caps |= ASTRAL_CAP_MM_EMBEDDINGS;
            }
        }
    }

    *out_caps = caps;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_limits(AstralHandle model, AstralModelLimits* out_limits) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || out_limits == nullptr) {
        set_err_invalid("model/out_limits");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    uint32_t vocab_size = 0;
    uint32_t ctx_size = 0;
    const AstralErr err = m->backend->ops->model_info(m->backend_model_ctx, &vocab_size, &ctx_size);
    if (err != ASTRAL_OK) {
        set_err_code(err);
        return err;
    }

    out_limits->vocab_size = vocab_size;
    out_limits->ctx_size = ctx_size;
    out_limits->max_batch = m->desc.n_batch;
    out_limits->max_slots =
        (m->backend && m->backend->ops &&
         m->backend->ops->session_create_ex != nullptr &&
         m->backend->ops->session_batch_eval != nullptr &&
         m->backend->ops->session_batch_logits != nullptr)
            ? astral::inference::ModelExecutor::kMaxSlotsHard
            : 1u;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_media_init(AstralHandle model, const AstralModelMediaDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || desc == nullptr) {
        set_err_invalid("model/desc");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralModelMediaDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const auto* backend = m->backend;
    if (backend == nullptr || backend->ops == nullptr || backend->ops->model_media_init == nullptr) {
        set_err_unsupported("media init");
        return ASTRAL_E_UNSUPPORTED;
    }

    const AstralErr err = backend->ops->model_media_init(m->backend_model_ctx, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_media_info(AstralHandle model, AstralMediaInfo* out_info) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || out_info == nullptr) {
        set_err_invalid("model/out_info");
        return ASTRAL_E_INVALID;
    }
    if (out_info->size != sizeof(AstralMediaInfo)) {
        set_err_invalid("out_info.size");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const auto* backend = m->backend;
    if (backend == nullptr || backend->ops == nullptr || backend->ops->model_media_info == nullptr) {
        set_err_unsupported("media info");
        return ASTRAL_E_UNSUPPORTED;
    }

    const AstralErr err = backend->ops->model_media_info(m->backend_model_ctx, out_info);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_executor_configure(AstralHandle model, const AstralExecutorDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || desc == nullptr) {
        set_err_invalid("model/desc");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralExecutorDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    if (m->executor.load(std::memory_order_acquire) != nullptr) {
        set_err_code(ASTRAL_E_STATE);
        return ASTRAL_E_STATE;
    }

    m->executor_desc = *desc;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_executor_tune(AstralHandle model, const AstralExecutorTuning* tuning) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || tuning == nullptr) {
        set_err_invalid("model/tuning");
        return ASTRAL_E_INVALID;
    }
    if (tuning->size != sizeof(AstralExecutorTuning)) {
        set_err_invalid("tuning.size");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    auto* ex = m->executor.load(std::memory_order_acquire);
    if (ex == nullptr) {
        set_err_code(ASTRAL_E_STATE);
        return ASTRAL_E_STATE;
    }

    if (tuning->max_prompt_tokens_per_slot_tick != 0) {
        ex->max_prompt_tokens_per_slot_per_tick.store(tuning->max_prompt_tokens_per_slot_tick, std::memory_order_relaxed);
    }

    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_embedding_dim(AstralHandle model, uint32_t* out_dim) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || out_dim == nullptr) {
        set_err_invalid("model/out_dim");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const auto* backend = m->backend;
    if (backend == nullptr || backend->ops == nullptr || backend->ops->model_embedding_dim == nullptr) {
        *out_dim = 0;
        set_err_unsupported("embeddings");
        return ASTRAL_E_UNSUPPORTED;
    }

    const AstralErr err = backend->ops->model_embedding_dim(m->backend_model_ctx, out_dim);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_tokenize(
    AstralHandle model,
    AstralSpanU8 text,
    int32_t* out_tokens,
    uint32_t max_tokens,
    uint8_t add_special,
    uint8_t parse_special,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.tokenize");
    if (model == 0 || out_tokens == nullptr || out_count == nullptr || max_tokens == 0) {
        set_err_invalid("model/out_tokens/out_count/max_tokens");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Model* m = nullptr;
    AstralErr err = require_model_ops(model, &m);
    if (err != ASTRAL_OK) {
        return err;
    }

    err = m->backend->ops->tokenize(
        m->backend_model_ctx,
        text,
        out_tokens,
        max_tokens,
        add_special != 0,
        parse_special != 0,
        out_count
    );

    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_tokenize_count(
    AstralHandle model,
    AstralSpanU8 text,
    uint8_t add_special,
    uint8_t parse_special,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.tokenize_count");
    if (model == 0 || out_count == nullptr) {
        set_err_invalid("model/out_count");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Model* m = nullptr;
    AstralErr err = require_model_ops(model, &m);
    if (err != ASTRAL_OK) {
        return err;
    }

    err = m->backend->ops->tokenize(
        m->backend_model_ctx,
        text,
        nullptr,
        0,
        add_special != 0,
        parse_special != 0,
        out_count
    );
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_tokenize_batch(
    AstralHandle model,
    const AstralTokenizeRequest* requests,
    uint32_t request_count,
    uint32_t* out_offsets,
    int32_t* out_tokens,
    uint32_t max_tokens,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.tokenize_batch");
    if (model == 0 || requests == nullptr || out_offsets == nullptr || out_count == nullptr) {
        set_err_invalid("model/requests/out_offsets/out_count");
        return ASTRAL_E_INVALID;
    }
    if (request_count == 0) {
        *out_count = 0;
        out_offsets[0] = 0;
        return ASTRAL_OK;
    }
    if ((out_tokens == nullptr && max_tokens != 0) || (out_tokens != nullptr && max_tokens == 0)) {
        set_err_invalid("out_tokens/max_tokens");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Model* m = nullptr;
    AstralErr err = require_model_ops(model, &m);
    if (err != ASTRAL_OK) {
        return err;
    }
    auto* const model_ctx = m->backend_model_ctx;
    auto* const tokenize = m->backend->ops->tokenize;

    if (out_tokens == nullptr) {
        uint32_t total = 0;
        out_offsets[0] = 0;
        for (uint32_t i = 0; i < request_count; ++i) {
            uint32_t count = 0;
            err = tokenize(
                model_ctx,
                requests[i].text,
                nullptr,
                0,
                requests[i].add_special != 0,
                requests[i].parse_special != 0,
                &count
            );
            if (err != ASTRAL_OK) {
                set_err_code(err);
                return err;
            }
            if (UINT32_MAX - total < count) {
                set_err_invalid("token count overflow");
                return ASTRAL_E_INVALID;
            }
            total += count;
            out_offsets[i + 1] = total;
        }
        *out_count = total;
        return ASTRAL_OK;
    }

    uint32_t total = 0;
    AstralErr batch_err = ASTRAL_OK;
    out_offsets[0] = 0;
    for (uint32_t i = 0; i < request_count; ++i) {
        uint32_t written = 0;
        int32_t* dst = nullptr;
        uint32_t cap = 0;
        if (batch_err == ASTRAL_OK && total < max_tokens) {
            dst = out_tokens + total;
            cap = max_tokens - total;
        }
        err = tokenize(model_ctx,
                       requests[i].text,
                       dst,
                       cap,
                       requests[i].add_special != 0,
                       requests[i].parse_special != 0,
                       &written);
        if (err != ASTRAL_OK) {
            if (err != ASTRAL_E_NOMEM) {
                set_err_code(err);
                return err;
            }
            batch_err = ASTRAL_E_NOMEM;
        }
        if (err == ASTRAL_OK && dst == nullptr && written != 0) {
            batch_err = ASTRAL_E_NOMEM;
        }
        if (err == ASTRAL_OK && dst != nullptr && written > cap) {
            set_err_code(ASTRAL_E_BACKEND);
            return ASTRAL_E_BACKEND;
        }
        if (UINT32_MAX - total < written) {
            set_err_invalid("token count overflow");
            return ASTRAL_E_INVALID;
        }
        total += written;
        out_offsets[i + 1] = total;
    }

    *out_count = total;
    if (batch_err != ASTRAL_OK) {
        set_err_code(batch_err);
    }

    return batch_err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_detokenize(
    AstralHandle model,
    const int32_t* tokens,
    uint32_t count,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.detokenize");
    if (model == 0 || (tokens == nullptr && count != 0) || out_text.data == nullptr || out_len == nullptr) {
        set_err_invalid("model/tokens/out_text/out_len");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Model* m = nullptr;
    AstralErr err = require_model_ops(model, &m);
    if (err != ASTRAL_OK) {
        return err;
    }

    err = m->backend->ops->detokenize(m->backend_model_ctx, tokens, count, out_text, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_detokenize_count(
    AstralHandle model,
    const int32_t* tokens,
    uint32_t count,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.detokenize_count");
    if (model == 0 || (tokens == nullptr && count != 0) || out_len == nullptr) {
        set_err_invalid("model/tokens/out_len");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Model* m = nullptr;
    AstralErr err = require_model_ops(model, &m);
    if (err != ASTRAL_OK) {
        return err;
    }

    AstralMutSpanU8 out{};
    err = m->backend->ops->detokenize(m->backend_model_ctx, tokens, count, out, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_chunk_count(
    const AstralChunkerDesc* desc,
    AstralSpanU8 text,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.chunk_count");
    const AstralErr err = astral::inference::chunk_count(desc, text, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_chunk_ranges(
    const AstralChunkerDesc* desc,
    AstralSpanU8 text,
    AstralChunkRange* out_ranges,
    uint32_t max_ranges,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.chunk_ranges");
    const AstralErr err = astral::inference::chunk_ranges(desc, text, out_ranges, max_ranges, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_chunk_text_copy(
    AstralSpanU8 text,
    const AstralChunkRange* range,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.chunk_text_copy");
    const AstralErr err = astral::inference::chunk_text_copy(text, range, out_text, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_token_chunk_count(
    const AstralChunkerDesc* desc,
    uint32_t token_count,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.token_chunk_count");
    const AstralErr err = astral::inference::token_chunk_count(desc, token_count, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_token_chunk_ranges(
    const AstralChunkerDesc* desc,
    uint32_t token_count,
    AstralChunkRange* out_ranges,
    uint32_t max_ranges,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.token_chunk_ranges");
    const AstralErr err = astral::inference::token_chunk_ranges(desc, token_count, out_ranges, max_ranges, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_create(const AstralMemoryIndexDesc* desc, AstralHandle* out_index) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_create");
    if (desc == nullptr || out_index == nullptr) {
        set_err_invalid("desc/out_index");
        return ASTRAL_E_INVALID;
    }

    astral::inference::MemoryIndex* index = nullptr;
    const AstralErr err = astral::inference::memory_create(desc, &index);
    if (err == ASTRAL_OK) {
        *out_index = astral::inference::memory_handle(index);
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_memory_destroy(AstralHandle index) {
    ASTRAL_ABI_TRY_BEGIN
    if (index == 0) {
        set_err_invalid("index");
        return;
    }
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr) {
        set_err_invalid("index (invalid handle)");
        return;
    }
    astral::inference::memory_destroy(mem);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_count(AstralHandle index, uint32_t* out_count) {
    ASTRAL_ABI_TRY_BEGIN
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr || out_count == nullptr) {
        set_err_invalid("index/out_count");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_count(mem, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_stats(AstralHandle index, AstralMemoryStats* out_stats) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_stats");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr || out_stats == nullptr) {
        set_err_invalid("index/out_stats");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_stats(mem, out_stats);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_clear(AstralHandle index) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_clear");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr) {
        set_err_invalid("index");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_clear(mem);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_get_record(AstralHandle index, uint64_t key, AstralMemoryRecord* out_record) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_get_record");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr || out_record == nullptr) {
        set_err_invalid("index/out_record");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_get_record(mem, key, out_record);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_update_record(AstralHandle index, uint64_t key, const AstralMemoryRecord* record) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_update_record");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr || record == nullptr) {
        set_err_invalid("index/record");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_update_record(mem, key, record);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_add_batch(
    AstralHandle index,
    const AstralMemoryRecord* records,
    const float* vectors,
    uint32_t count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_add_batch");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr) {
        set_err_invalid("index");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_add_batch(mem, records, vectors, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_remove(AstralHandle index, uint64_t key) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_remove");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr) {
        set_err_invalid("index");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_remove(mem, key);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_search(
    AstralHandle index,
    const AstralMemorySearchDesc* desc,
    const float* query,
    AstralMemorySearchResult* out_results,
    uint32_t max_results,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_search");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr) {
        set_err_invalid("index");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_search(mem, desc, query, out_results, max_results, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_search_begin(
    AstralHandle index,
    const AstralMemorySearchDesc* desc,
    const float* query,
    AstralHandle* out_cursor
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_search_begin");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr || out_cursor == nullptr) {
        set_err_invalid("index/out_cursor");
        return ASTRAL_E_INVALID;
    }

    astral::inference::MemorySearchCursor* cursor = nullptr;
    const AstralErr err = astral::inference::memory_search_begin(mem, desc, query, &cursor);
    if (err == ASTRAL_OK) {
        *out_cursor = astral::inference::memory_search_cursor_handle(cursor);
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_search_fetch(
    AstralHandle cursor,
    AstralMemorySearchResult* out_results,
    uint32_t max_results,
    uint32_t* out_count
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_search_fetch");
    auto* search = lookup_memory_search_cursor(cursor);
    if (search == nullptr) {
        set_err_invalid("cursor");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_search_fetch(search, out_results, max_results, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_memory_search_end(AstralHandle cursor) {
    ASTRAL_ABI_TRY_BEGIN
    if (cursor == 0) {
        set_err_invalid("cursor");
        return;
    }
    auto* search = lookup_memory_search_cursor(cursor);
    if (search == nullptr) {
        set_err_invalid("cursor (invalid handle)");
        return;
    }
    astral::inference::memory_search_end(search);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_save_size(AstralHandle index, uint64_t* out_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr || out_bytes == nullptr) {
        set_err_invalid("index/out_bytes");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_save_size(mem, out_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_save(AstralHandle index, AstralMutSpanU8 out_bytes, uint64_t* out_written) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_save");
    auto* mem = lookup_memory_index(index);
    if (mem == nullptr || out_written == nullptr) {
        set_err_invalid("index/out_written");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::memory_save(mem, out_bytes, out_written);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_load(
    const AstralMemoryIndexDesc* desc,
    AstralSpanU8 bytes,
    AstralHandle* out_index
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.memory_load");
    if (desc == nullptr || bytes.data == nullptr || out_index == nullptr) {
        set_err_invalid("desc/bytes/out_index");
        return ASTRAL_E_INVALID;
    }
    astral::inference::MemoryIndex* index = nullptr;
    const AstralErr err = astral::inference::memory_load(desc, bytes, &index);
    if (err == ASTRAL_OK) {
        *out_index = astral::inference::memory_handle(index);
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_record_from_chunk(
    const AstralChunkRange* range,
    uint64_t key,
    uint32_t flags,
    AstralMemoryRecord* out_record
) {
    ASTRAL_ABI_TRY_BEGIN
    if (range == nullptr || range->size != sizeof(AstralChunkRange) || out_record == nullptr) {
        set_err_invalid("range/out_record");
        return ASTRAL_E_INVALID;
    }
    out_record->size = sizeof(AstralMemoryRecord);
    out_record->group_id = range->group_id;
    out_record->key = key;
    out_record->document_id = range->document_id;
    out_record->chunk_id = range->chunk_id;
    out_record->flags = flags;
    out_record->_reserved0 = 0;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

// ============================================================================
// Session API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_session_create(
    const AstralSessionDesc* desc,
    AstralHandle* out_session
) {
    ASTRAL_ABI_TRY_BEGIN
    if (desc == nullptr || out_session == nullptr) {
        set_err_invalid("desc/out_session");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Session* session = nullptr;
    AstralErr err = astral::inference::session_create(desc, &session);
    if (err == ASTRAL_OK) {
        *out_session = session->handle;
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_session_destroy(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return;
    }

    astral::inference::session_destroy(s);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed(
    AstralHandle session,
    AstralSpanU8 prompt_chunk,
    uint8_t finalize
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_feed(
        s,
        prompt_chunk,
        finalize
    );
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed_tokens(
    AstralHandle session,
    const int32_t* tokens,
    uint32_t token_count,
    uint8_t finalize
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || (token_count != 0 && tokens == nullptr)) {
        set_err_invalid("session/tokens");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_feed_tokens(s, tokens, token_count, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_system_prompt(AstralHandle session, AstralSpanU8 system_prompt) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }
    if (s->prompt_count != 0) {
        set_err_code(ASTRAL_E_STATE);
        return ASTRAL_E_STATE;
    }

    const AstralErr err = astral::inference::session_feed(s, system_prompt, 0);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed_image(
    AstralHandle session,
    const AstralImageDesc* image,
    uint8_t finalize
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || image == nullptr) {
        set_err_invalid("session/image");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_feed_image(s, image, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed_audio(
    AstralHandle session,
    const AstralAudioDesc* audio,
    uint8_t finalize
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || audio == nullptr) {
        set_err_invalid("session/audio");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_feed_audio(s, audio, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_sampler(AstralHandle session, const AstralSamplerDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || desc == nullptr) {
        set_err_invalid("session/desc");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_set_sampler(s, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_penalty_prompt_set_tokens(
    AstralHandle session,
    const int32_t* tokens,
    uint32_t count
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_penalty_prompt_set_tokens(s, tokens, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_clear(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_stop_clear(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_add_utf8(AstralHandle session, AstralSpanU8 utf8) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_stop_add_utf8(s, utf8);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_set_utf8(
    AstralHandle session,
    const AstralSpanU8* seqs,
    uint32_t count
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_stop_set_utf8(s, seqs, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_logprobs(AstralHandle session, uint32_t n_probs) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_set_logprobs(s, n_probs);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_stream_read_meta(
    AstralHandle session,
    AstralTokenMeta* out_events,
    uint32_t capacity,
    uint32_t timeout_ms
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const int32_t result = astral::inference::stream_read_meta(s, out_events, capacity, timeout_ms);
    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }
    return result;
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_size(AstralHandle session, uint64_t* out_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || out_bytes == nullptr) {
        set_err_invalid("session/out_bytes");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_state_size(s, out_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_save(
    AstralHandle session,
    AstralMutSpanU8 out_buf,
    uint64_t* out_written
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || out_written == nullptr) {
        set_err_invalid("session/out_written");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_state_save(s, out_buf, out_written);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_load(AstralHandle session, AstralSpanU8 state_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_state_load(s, state_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_clear(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.session_adapters_clear");
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_adapters_clear(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_add(AstralHandle session, AstralHandle adapter, float scale) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.session_adapters_add");
    if (session == 0 || adapter == 0) {
        set_err_invalid("session/adapter");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_adapters_add(s, adapter, scale);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_count(AstralHandle session, uint32_t* out_count) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || out_count == nullptr) {
        set_err_invalid("session/out_count");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_adapters_count(s, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_get(
    AstralHandle session,
    uint32_t index,
    AstralHandle* out_adapter,
    float* out_scale
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || out_adapter == nullptr || out_scale == nullptr) {
        set_err_invalid("session/out_adapter/out_scale");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_adapters_get(s, index, out_adapter, out_scale);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_set_scale(AstralHandle session, uint32_t index, float scale) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.session_adapters_set_scale");
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_adapters_set_scale(s, index, scale);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_grammar_gbnf(AstralHandle session, AstralSpanU8 gbnf, AstralSpanU8 root) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_set_grammar_gbnf(s, gbnf, root);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_grammar_json_schema(AstralHandle session, AstralSpanU8 json_schema) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_set_grammar_json_schema(s, json_schema);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_clear_grammar(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_clear_grammar(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_toolset(
    AstralHandle session,
    AstralHandle toolset,
    AstralToolChoiceMode choice_mode
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.session_set_toolset");
    if (session == 0 || toolset == 0) {
        set_err_invalid("session/toolset");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    auto* ts = lookup_toolset(toolset);
    if (s == nullptr || ts == nullptr) {
        set_err_invalid("session/toolset (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_set_toolset(s, ts, choice_mode);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_clear_toolset(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_clear_toolset(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_slot(AstralHandle session, uint32_t slot_id) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_set_slot(s, slot_id);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_session(AstralHandle session, AstralRequestRef* out_request) {
    ASTRAL_ABI_TRY_BEGIN
    if (lookup_session(session) == nullptr) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }
    return request_ref_make(ASTRAL_REQUEST_SESSION, session, 0, out_request);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_conversation(AstralHandle conv, AstralRequestRef* out_request) {
    ASTRAL_ABI_TRY_BEGIN
    if (lookup_conversation(conv) == nullptr) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }
    return request_ref_make(ASTRAL_REQUEST_CONVERSATION, conv, 0, out_request);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_agent_chat(AstralHandle agent, AstralRequestRef* out_request) {
    ASTRAL_ABI_TRY_BEGIN
    if (lookup_agent(agent) == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    return request_ref_make(ASTRAL_REQUEST_AGENT_CHAT, agent, 0, out_request);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_embedding(
    AstralHandle emb,
    uint64_t ticket,
    AstralRequestRef* out_request
) {
    ASTRAL_ABI_TRY_BEGIN
    if (lookup_embedder(emb) == nullptr || ticket == 0) {
        set_err_invalid("emb/ticket");
        return ASTRAL_E_INVALID;
    }
    return request_ref_make(ASTRAL_REQUEST_EMBEDDING, emb, ticket, out_request);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_memory_search(
    AstralHandle cursor,
    AstralRequestRef* out_request
) {
    ASTRAL_ABI_TRY_BEGIN
    if (lookup_memory_search_cursor(cursor) == nullptr) {
        set_err_invalid("cursor");
        return ASTRAL_E_INVALID;
    }
    return request_ref_make(ASTRAL_REQUEST_MEMORY_SEARCH, cursor, 0, out_request);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_request_state(
    const AstralRequestRef* request,
    AstralRequestStatus* out_status
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.request_state");
    const AstralErr err = request_state_impl(request, out_status);
    if (err != ASTRAL_OK && err != ASTRAL_E_NOT_FOUND) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_request_cancel(const AstralRequestRef* request) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.request_cancel");
    if (!request_ref_valid(request)) {
        set_err_invalid("request");
        return ASTRAL_E_INVALID;
    }

    AstralErr err = ASTRAL_E_INVALID;
    switch (request->kind) {
    case ASTRAL_REQUEST_SESSION: {
        auto* s = lookup_session(request->owner);
        err = s != nullptr ? astral::inference::session_cancel(s) : ASTRAL_E_INVALID;
        break;
    }
    case ASTRAL_REQUEST_CONVERSATION: {
        auto* c = lookup_conversation(request->owner);
        err = c != nullptr ? astral::inference::conv_cancel(c) : ASTRAL_E_INVALID;
        break;
    }
    case ASTRAL_REQUEST_AGENT_CHAT: {
        auto* a = lookup_agent(request->owner);
        err = a != nullptr ? astral::inference::agent_chat_cancel(a) : ASTRAL_E_INVALID;
        break;
    }
    case ASTRAL_REQUEST_EMBEDDING: {
        auto* e = lookup_embedder(request->owner);
        err = e != nullptr ? astral::inference::embedder_cancel(e, request->ticket) : ASTRAL_E_INVALID;
        break;
    }
    case ASTRAL_REQUEST_MEMORY_SEARCH: {
        auto* cursor = lookup_memory_search_cursor(request->owner);
        err = cursor != nullptr ? astral::inference::memory_search_cancel(cursor) : ASTRAL_E_INVALID;
        break;
    }
    default:
        err = ASTRAL_E_INVALID;
        break;
    }

    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_request_wait(
    const AstralRequestRef* request,
    uint32_t timeout_ms,
    AstralRequestStatus* out_status
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.request_wait");
    if (!request_ref_valid(request) || out_status == nullptr || out_status->size != sizeof(AstralRequestStatus)) {
        set_err_invalid("request/out_status");
        return ASTRAL_E_INVALID;
    }

    AstralErr err = ASTRAL_E_INVALID;
    switch (request->kind) {
    case ASTRAL_REQUEST_SESSION: {
        auto* s = lookup_session(request->owner);
        err = s != nullptr ? astral::inference::session_wait(s, timeout_ms) : ASTRAL_E_INVALID;
        (void)request_state_impl(request, out_status);
        break;
    }
    case ASTRAL_REQUEST_CONVERSATION: {
        auto* c = lookup_conversation(request->owner);
        err = c != nullptr ? astral::inference::conv_wait(c, timeout_ms) : ASTRAL_E_INVALID;
        (void)request_state_impl(request, out_status);
        break;
    }
    case ASTRAL_REQUEST_AGENT_CHAT:
    case ASTRAL_REQUEST_EMBEDDING:
    case ASTRAL_REQUEST_MEMORY_SEARCH: {
        const uint64_t timeout_ns = static_cast<uint64_t>(timeout_ms) * 1000000ull;
        const uint64_t start_ns = astral::platform::monotonic_time_ns();
        uint32_t spins = 0;
        for (;;) {
            err = request_state_impl(request, out_status);
            if (err != ASTRAL_OK || request_state_terminal(out_status->state)) {
                break;
            }
            if (timeout_ms == 0 || astral::platform::monotonic_time_ns() - start_ns >= timeout_ns) {
                err = ASTRAL_E_TIMEOUT;
                break;
            }
            if (spins < 64) {
                astral::platform::cpu_pause();
            } else {
                astral::platform::cpu_wait_for_event();
            }
            if (spins < 1024) {
                ++spins;
            }
        }
        break;
    }
    default:
        err = ASTRAL_E_INVALID;
        break;
    }

    if (err != ASTRAL_OK && err != ASTRAL_E_TIMEOUT) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_decode(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_decode(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_cancel(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_cancel(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state(AstralHandle session, AstralSessionState* out_state) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || out_state == nullptr) {
        set_err_invalid("session/out_state");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    astral::inference::SessionState state{};
    const AstralErr err = astral::inference::session_state(s, &state);
    if (err != ASTRAL_OK) {
        set_err_code(err);
        return err;
    }

    *out_state = static_cast<AstralSessionState>(state);
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_wait(AstralHandle session, uint32_t timeout_ms) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_wait(s, timeout_ms);
    if (err != ASTRAL_OK && err != ASTRAL_E_TIMEOUT) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_reset(AstralHandle session, const AstralSessionDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_reset(s, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_stream_read(
    AstralHandle session,
    AstralMutSpanU8 out_buf,
    uint32_t timeout_ms
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const int32_t result = astral::inference::stream_read(s, out_buf, timeout_ms);

    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }

    return result;
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stats(
    AstralHandle session,
    AstralStats* out_stats
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || out_stats == nullptr) {
        set_err_invalid("session/out_stats");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_stats(
        s,
        out_stats
    );

    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

// ============================================================================
// Conversation API (continuous batching)
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_create(const AstralConvDesc* desc, AstralHandle* out_conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (desc == nullptr || out_conv == nullptr) {
        set_err_invalid("desc/out_conv");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralConvDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Conversation* conv = nullptr;
    const AstralErr err = astral::inference::conv_create(desc, &conv);
    if (err == ASTRAL_OK) {
        *out_conv = conv->handle;
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_conv_destroy(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return;
    }

    astral::inference::conv_destroy(c);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed(AstralHandle conv, AstralSpanU8 prompt_chunk, uint8_t finalize) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_feed(c, prompt_chunk, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed_tokens(
    AstralHandle conv,
    const int32_t* tokens,
    uint32_t token_count,
    uint8_t finalize
) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || (token_count != 0 && tokens == nullptr)) {
        set_err_invalid("conv/tokens");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_feed_tokens(c, tokens, token_count, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_system_prompt(AstralHandle conv, AstralSpanU8 system_prompt) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }
    if (c->prompt_count != 0) {
        set_err_code(ASTRAL_E_STATE);
        return ASTRAL_E_STATE;
    }

    const AstralErr err = astral::inference::conv_feed(c, system_prompt, 0);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed_image(AstralHandle conv, const AstralImageDesc* image, uint8_t finalize) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || image == nullptr) {
        set_err_invalid("conv/image");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_feed_image(c, image, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed_audio(AstralHandle conv, const AstralAudioDesc* audio, uint8_t finalize) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || audio == nullptr) {
        set_err_invalid("conv/audio");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_feed_audio(c, audio, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_decode(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_decode(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_cancel(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_cancel(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_state(AstralHandle conv, AstralSessionState* out_state) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || out_state == nullptr) {
        set_err_invalid("conv/out_state");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_state(c, out_state);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_wait(AstralHandle conv, uint32_t timeout_ms) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_wait(c, timeout_ms);
    if (err != ASTRAL_OK && err != ASTRAL_E_TIMEOUT) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_reset(AstralHandle conv, const AstralConvDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || desc == nullptr) {
        set_err_invalid("conv/desc");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralConvDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_reset(c, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_sampler(AstralHandle conv, const AstralSamplerDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || desc == nullptr) {
        set_err_invalid("conv/desc");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_set_sampler(c, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_penalty_prompt_set_tokens(
    AstralHandle conv, const int32_t* tokens, uint32_t count) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_penalty_prompt_set_tokens(c, tokens, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_clear(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_stop_clear(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_add_utf8(AstralHandle conv, AstralSpanU8 utf8) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_stop_add_utf8(c, utf8);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_set_utf8(
    AstralHandle conv, const AstralSpanU8* seqs, uint32_t count) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_stop_set_utf8(c, seqs, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_logprobs(AstralHandle conv, uint32_t n_probs) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_set_logprobs(c, n_probs);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_set_gbnf(AstralHandle conv, AstralSpanU8 gbnf, AstralSpanU8 root) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_grammar_set_gbnf(c, gbnf, root);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_set_json_schema(AstralHandle conv, AstralSpanU8 json_schema) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_grammar_set_json_schema(c, json_schema);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_clear(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_grammar_clear(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_toolset(
    AstralHandle conv,
    AstralHandle toolset,
    AstralToolChoiceMode choice_mode
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.conv_set_toolset");
    if (conv == 0 || toolset == 0) {
        set_err_invalid("conv/toolset");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    auto* ts = lookup_toolset(toolset);
    if (c == nullptr || ts == nullptr) {
        set_err_invalid("conv/toolset (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_set_toolset(c, ts, choice_mode);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_clear_toolset(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_clear_toolset(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_conv_stream_read(AstralHandle conv, AstralMutSpanU8 out_buf, uint32_t timeout_ms) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const int32_t result = astral::inference::conv_stream_read(c, out_buf, timeout_ms);
    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }
    return result;
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_conv_stream_read_meta(
    AstralHandle conv, AstralTokenMeta* out_events, uint32_t capacity, uint32_t timeout_ms) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const int32_t result = astral::inference::conv_stream_read_meta(c, out_events, capacity, timeout_ms);
    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }
    return result;
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stats(AstralHandle conv, AstralConvStats* out_stats) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || out_stats == nullptr) {
        set_err_invalid("conv/out_stats");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_stats(c, out_stats);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

// ============================================================================
// Agent API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_create(const AstralAgentDesc* desc, AstralHandle* out_agent) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_create");
    if (desc == nullptr || out_agent == nullptr) {
        set_err_invalid("desc/out_agent");
        return ASTRAL_E_INVALID;
    }
    astral::inference::Agent* agent = nullptr;
    const AstralErr err = astral::inference::agent_create(desc, &agent);
    if (err == ASTRAL_OK) {
        *out_agent = astral::inference::agent_handle(agent);
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_assigned_slot(AstralHandle agent, uint32_t* out_slot) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_assigned_slot");
    if (out_slot == nullptr) {
        set_err_invalid("out_slot");
        return ASTRAL_E_INVALID;
    }
    auto* a = static_cast<astral::inference::Agent*>(
        astral::core::lookup_handle(agent, astral::core::HandleKind::Agent)
    );
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_assigned_slot(a, out_slot);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_release_slot(AstralHandle agent) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_release_slot");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_release_slot(a);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_agent_destroy(AstralHandle agent) {
    ASTRAL_ABI_TRY_BEGIN
    if (agent == 0) {
        set_err_invalid("agent");
        return;
    }
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent (invalid handle)");
        return;
    }
    astral::inference::agent_destroy(a);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_set_system_prompt(AstralHandle agent, AstralSpanU8 system_prompt) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_set_system_prompt");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_set_system_prompt(a, system_prompt);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_system_prompt_size(AstralHandle agent, uint32_t* out_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_bytes == nullptr) {
        set_err_invalid("agent/out_bytes");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_get_system_prompt_size(a, out_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_system_prompt(
    AstralHandle agent,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_len == nullptr) {
        set_err_invalid("agent/out_len");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_get_system_prompt(a, out_text, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_set_summary(AstralHandle agent, AstralSpanU8 summary) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_set_summary");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_set_summary(a, summary);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_summary_size(AstralHandle agent, uint32_t* out_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_bytes == nullptr) {
        set_err_invalid("agent/out_bytes");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_get_summary_size(a, out_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_summary(
    AstralHandle agent,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_len == nullptr) {
        set_err_invalid("agent/out_len");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_get_summary(a, out_text, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_set_memory_context(AstralHandle agent, AstralSpanU8 memory_context) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_set_memory_context");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_set_memory_context(a, memory_context);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_set_memory_context_from_results(
    AstralHandle agent,
    const AstralAgentMemoryContextDesc* desc
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_set_memory_context_from_results");
    auto* a = lookup_agent(agent);
    if (a == nullptr || desc == nullptr) {
        set_err_invalid("agent/desc");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_set_memory_context_from_results(a, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_memory_context_size(AstralHandle agent, uint32_t* out_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_bytes == nullptr) {
        set_err_invalid("agent/out_bytes");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_get_memory_context_size(a, out_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_memory_context(
    AstralHandle agent,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_parse_tool_call");
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_len == nullptr) {
        set_err_invalid("agent/out_len");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_get_memory_context(a, out_text, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_parse_tool_call(
    AstralHandle agent,
    AstralSpanU8 generated_text,
    AstralToolCallResult* out_result
) {
    ASTRAL_ABI_TRY_BEGIN
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_result == nullptr) {
        set_err_invalid("agent/out_result");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_parse_tool_call(a, generated_text, out_result);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_message_add(AstralHandle agent, const AstralAgentMessage* message) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_message_add");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_message_add(a, message);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_clear(AstralHandle agent) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_history_clear");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_history_clear(a);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_count(AstralHandle agent, uint32_t* out_count) {
    ASTRAL_ABI_TRY_BEGIN
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_count == nullptr) {
        set_err_invalid("agent/out_count");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_history_count(a, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_save_size(AstralHandle agent, uint32_t* out_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_bytes == nullptr) {
        set_err_invalid("agent/out_bytes");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_history_save_size(a, out_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_save(AstralHandle agent, AstralMutSpanU8 out_bytes, uint32_t* out_len) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_history_save");
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_len == nullptr) {
        set_err_invalid("agent/out_len");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_history_save(a, out_bytes, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_load(AstralHandle agent, AstralSpanU8 bytes) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_history_load");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_history_load(a, bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_chat_enqueue(AstralHandle agent, const AstralAgentChatDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_chat_enqueue");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_chat_enqueue(a, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_chat_cancel(AstralHandle agent) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_chat_cancel");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_chat_cancel(a);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_agent_chat_stream_read(
    AstralHandle agent,
    AstralMutSpanU8 out_buf,
    uint32_t timeout_ms
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_chat_stream_read");
    auto* a = lookup_agent(agent);
    if (a == nullptr) {
        set_err_invalid("agent");
        return ASTRAL_E_INVALID;
    }
    const int32_t result = astral::inference::agent_chat_stream_read(a, out_buf, timeout_ms);
    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }
    return result;
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_chat_tool_call_result(
    AstralHandle agent,
    AstralToolCallResult* out_result
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_chat_tool_call_result");
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_result == nullptr) {
        set_err_invalid("agent/out_result");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_chat_tool_call_result(a, out_result);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_chat_result(AstralHandle agent, AstralAgentChatResult* out_result) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.agent_chat_result");
    auto* a = lookup_agent(agent);
    if (a == nullptr || out_result == nullptr) {
        set_err_invalid("agent/out_result");
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = astral::inference::agent_chat_result(a, out_result);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

// ============================================================================
// Embeddings API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_create(
    AstralHandle model,
    AstralHandle* out_embedder
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.embed_create");
    if (model == 0 || out_embedder == nullptr) {
        set_err_invalid("model/out_embedder");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_create(m, out_embedder);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_embed_destroy(AstralHandle emb) {
    ASTRAL_ABI_TRY_BEGIN
    if (emb == 0) {
        set_err_invalid("emb");
        return;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return;
    }

    astral::inference::embedder_destroy(e);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue(
    AstralHandle emb,
    AstralSpanU8 text,
    uint64_t* out_ticket
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.embed_enqueue_text");
    if (emb == 0 || out_ticket == nullptr) {
        set_err_invalid("emb/out_ticket");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_enqueue(e, text, out_ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_image(
    AstralHandle emb,
    const AstralImageDesc* image,
    uint64_t* out_ticket
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.embed_enqueue_image");
    if (emb == 0 || image == nullptr || out_ticket == nullptr) {
        set_err_invalid("emb/image/out_ticket");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_enqueue_image(e, image, out_ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_audio(
    AstralHandle emb,
    const AstralAudioDesc* audio,
    uint64_t* out_ticket
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.embed_enqueue_audio");
    if (emb == 0 || audio == nullptr || out_ticket == nullptr) {
        set_err_invalid("emb/audio/out_ticket");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_enqueue_audio(e, audio, out_ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_multimodal(
    AstralHandle emb,
    AstralSpanU8 text,
    const AstralImageDesc* image,
    const AstralAudioDesc* audio,
    uint64_t* out_ticket
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.embed_enqueue_multimodal");
    if (emb == 0 || out_ticket == nullptr) {
        set_err_invalid("emb/out_ticket");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_enqueue_multimodal(e, text, image, audio, out_ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_collect(
    AstralHandle emb,
    uint64_t ticket,
    AstralMutSpanU8 out_vector
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.embed_collect");
    if (emb == 0 || out_vector.data == nullptr) {
        set_err_invalid("emb/out_vector");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_collect(e, ticket, out_vector);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_cancel(
    AstralHandle emb,
    uint64_t ticket
) {
    ASTRAL_ABI_TRY_BEGIN
    ASTRAL_ZONE_N("astral.abi.embed_cancel");
    if (emb == 0 || ticket == 0) {
        set_err_invalid("emb/ticket");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_cancel(e, ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

} // extern "C"
