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
#include "../core/error.hpp"
#include "../core/abi_guard.hpp"
#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"
#include "../core/runtime_state.hpp"
#include "../core/model_sources.hpp"
#include "../core/model_load_config.hpp"

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
inline constexpr uint32_t kPromptCacheDefaultMaxTokens = 64 * 1024;
inline constexpr uint32_t kPromptCacheMinTableCapacity = 4;
inline constexpr uint32_t kPromptCacheTableLoadFactorDen = 2;
inline constexpr uint64_t kPromptCacheHashModelMul = 0x9E3779B185EBCA87ull;
inline constexpr uint32_t kPromptCacheHashShift0 = 32;
inline constexpr uint32_t kPromptCacheHashShift1 = 16;
inline constexpr uint8_t kPromptCacheSlotEmpty = 0;
inline constexpr uint8_t kPromptCacheSlotOccupied = 1;
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
    uint32_t hash = 0;
    uint64_t sequence = 0;
    uint8_t state = kPromptCacheSlotEmpty;
};

struct PromptCache {
    AstralHandle handle = 0;
    PromptCacheEntry* entries = nullptr;
    uint32_t max_entries = 0;
    uint32_t table_capacity = 0;
    uint32_t table_mask = 0;
    uint32_t entry_count = 0;
    uint32_t max_tokens = 0;
    uint32_t token_count = 0;
    uint32_t max_bytes = 0;
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
    return a.model == b.model && a.section_kind == b.section_kind && a.key == b.key && a.generation == b.generation;
}

inline uint32_t prompt_cache_hash(const AstralPromptCacheKey& key) {
    uint64_t x = key.key ^ (key.model * kPromptCacheHashModelMul) ^
                 (static_cast<uint64_t>(key.generation) << kPromptCacheHashShift0) ^ key.section_kind;
    x ^= x >> kPromptCacheHashShift0;
    x ^= x >> kPromptCacheHashShift1;
    return static_cast<uint32_t>(x);
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

inline void prompt_cache_release_entry_payload(PromptCache* cache, PromptCacheEntry& entry) {
    if (entry.state != kPromptCacheSlotOccupied) {
        return;
    }
    if (entry.tokens != nullptr) {
        astral::core::runtime_free_array(entry.tokens, entry.token_count);
        entry.tokens = nullptr;
    }
    cache->token_count = cache->token_count >= entry.token_count ? cache->token_count - entry.token_count : 0;
    entry.token_count = 0;
    entry.hash = 0;
    entry.sequence = 0;
    entry.state = kPromptCacheSlotEmpty;
    entry.key = {};
}

inline void prompt_cache_remove_entry(PromptCache* cache, PromptCacheEntry* entry) {
    if (entry == nullptr || entry->state != kPromptCacheSlotOccupied) {
        return;
    }

    uint32_t hole = static_cast<uint32_t>(entry - cache->entries);
    prompt_cache_release_entry_payload(cache, *entry);

    uint32_t slot = (hole + 1u) & cache->table_mask;
    while (cache->entries[slot].state == kPromptCacheSlotOccupied) {
        PromptCacheEntry moved = cache->entries[slot];
        cache->entries[slot] = {};

        PromptCacheEntry* dst = prompt_cache_empty_entry(cache, moved.hash);
        *dst = moved;

        slot = (slot + 1u) & cache->table_mask;
    }
}

inline void prompt_cache_clear_impl(PromptCache* cache) {
    for (uint32_t i = 0; i < cache->table_capacity; ++i) {
        prompt_cache_release_entry_payload(cache, cache->entries[i]);
    }
    cache->entry_count = 0;
}

inline void prompt_cache_destroy_impl(PromptCache* cache) {
    if (cache == nullptr) {
        return;
    }
    prompt_cache_clear_impl(cache);
    astral::core::runtime_free_array(cache->entries, cache->table_capacity);
    cache->entries = nullptr;
    astral::core::runtime_delete(cache);
}

inline PromptCacheEntry* prompt_cache_find(PromptCache* cache, const AstralPromptCacheKey* key) {
    const uint32_t hash = prompt_cache_hash(*key);
    uint32_t slot = hash & cache->table_mask;
    for (uint32_t probe = 0; probe < cache->table_capacity; ++probe) {
        PromptCacheEntry& entry = cache->entries[slot];
        if (entry.state == kPromptCacheSlotEmpty) {
            return nullptr;
        }
        if (entry.state == kPromptCacheSlotOccupied && entry.hash == hash && prompt_cache_key_equal(entry.key, *key)) {
            return &entry;
        }
        slot = (slot + 1u) & cache->table_mask;
    }
    return nullptr;
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
    PromptCacheEntry* oldest = nullptr;
    for (uint32_t i = 0; i < cache->table_capacity; ++i) {
        PromptCacheEntry& entry = cache->entries[i];
        if (entry.state != kPromptCacheSlotOccupied) {
            continue;
        }
        if (oldest == nullptr || entry.sequence < oldest->sequence) {
            oldest = &entry;
        }
    }
    return oldest;
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

ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_create(const AstralToolsetDesc* desc, AstralHandle* out_toolset) {
    ASTRAL_ABI_TRY_BEGIN
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

    for (uint32_t i = 0; i < c->table_capacity; ++i) {
        const PromptCacheEntry& entry = c->entries[i];
        if (entry.state != kPromptCacheSlotOccupied) {
            continue;
        }

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

ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_put_tokens(
    AstralHandle cache,
    const AstralPromptCacheKey* key,
    const int32_t* tokens,
    uint32_t token_count
) {
    ASTRAL_ABI_TRY_BEGIN
    PromptCache* c = lookup_prompt_cache(cache);
    if (c == nullptr || !prompt_cache_key_valid(key) || (token_count != 0 && tokens == nullptr)) {
        set_err_invalid("cache/key/tokens");
        return ASTRAL_E_INVALID;
    }
    if (token_count > c->max_tokens) {
        set_err_invalid("token_count");
        return ASTRAL_E_INVALID;
    }

    PromptCacheEntry* entry = prompt_cache_find(c, key);
    if (entry != nullptr) {
        prompt_cache_remove_entry(c, entry);
        --c->entry_count;
    }

    while (c->entry_count >= c->max_entries || c->token_count + token_count > c->max_tokens) {
        PromptCacheEntry* victim = prompt_cache_oldest_entry(c);
        if (victim == nullptr) {
            break;
        }
        prompt_cache_remove_entry(c, victim);
        --c->entry_count;
        if (c->track_stats != 0) {
            ++c->evictions;
        }
    }

    const uint32_t hash = prompt_cache_hash(*key);
    entry = prompt_cache_empty_entry(c, hash);
    if (entry == nullptr) {
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }

    int32_t* copy = nullptr;
    if (token_count != 0) {
        copy = astral::core::runtime_alloc_array<int32_t>(token_count);
        if (copy == nullptr) {
            set_err_code(ASTRAL_E_NOMEM);
            return ASTRAL_E_NOMEM;
        }
        std::memcpy(copy, tokens, static_cast<size_t>(token_count) * sizeof(int32_t));
    }

    entry->key = *key;
    entry->tokens = copy;
    entry->token_count = token_count;
    entry->hash = hash;
    entry->sequence = c->next_sequence++;
    entry->state = kPromptCacheSlotOccupied;
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

    uint32_t total = 0;
    out_offsets[0] = 0;
    for (uint32_t i = 0; i < request_count; ++i) {
        uint32_t count = 0;
        err = m->backend->ops->tokenize(
            m->backend_model_ctx,
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
    if (out_tokens == nullptr) {
        return ASTRAL_OK;
    }
    if (total > max_tokens) {
        set_err_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }

    for (uint32_t i = 0; i < request_count; ++i) {
        uint32_t written = 0;
        const uint32_t begin = out_offsets[i];
        const uint32_t cap = max_tokens - begin;
        err = m->backend->ops->tokenize(
            m->backend_model_ctx,
            requests[i].text,
            out_tokens + begin,
            cap,
            requests[i].add_special != 0,
            requests[i].parse_special != 0,
            &written
        );
        if (err != ASTRAL_OK) {
            set_err_code(err);
            return err;
        }
        if (written != out_offsets[i + 1] - begin) {
            set_err_code(ASTRAL_E_BACKEND);
            return ASTRAL_E_BACKEND;
        }
    }

    return ASTRAL_OK;
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
    const AstralErr err = astral::inference::token_chunk_ranges(desc, token_count, out_ranges, max_ranges, out_count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_create(const AstralMemoryIndexDesc* desc, AstralHandle* out_index) {
    ASTRAL_ABI_TRY_BEGIN
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

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_clear(AstralHandle index) {
    ASTRAL_ABI_TRY_BEGIN
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

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_add_batch(
    AstralHandle index,
    const AstralMemoryRecord* records,
    const float* vectors,
    uint32_t count
) {
    ASTRAL_ABI_TRY_BEGIN
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
// Embeddings API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_create(
    AstralHandle model,
    AstralHandle* out_embedder
) {
    ASTRAL_ABI_TRY_BEGIN
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
