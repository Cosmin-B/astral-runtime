#include "memory_index.hpp"
#include "memory_index_internal.hpp"

#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"
#include "../core/runtime_state.hpp"
#include "../core/work_queue.hpp"
#include "../platform/atomics.h"
#include "../platform/cpu_features.hpp"
#include "../platform/file_map.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace astral::inference {

constexpr uint32_t kInlineSearchCursorResults = 8;

struct MemorySearchCursor {
  AstralHandle handle;
  uint32_t capacity;
  uint32_t count;
  uint32_t offset;
  std::atomic<uint32_t> canceled;
  AstralMemorySearchResult* results;
  AstralMemorySearchResult inline_results[kInlineSearchCursorResults];
};

namespace {

constexpr uint32_t kNoResults = 0;
constexpr uint32_t kTopOne = 1;
constexpr uint32_t kKeyTableMinCapacity = 4;
constexpr uint32_t kKeyTableLoadFactorDen = 2;
constexpr uint32_t kKeyTableEmpty = 0;
constexpr uint32_t kKeyTableTombstone = kU32Max;
constexpr uint32_t kKeyTableSlotRefBias = 1;
constexpr uint32_t kSlotAdvance = 1;
constexpr uint64_t kKeyHashMixMul0 = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t kKeyHashMixMul1 = 0x94D049BB133111EBull;
constexpr uint32_t kKeyHashMixShift0 = 30;
constexpr uint32_t kKeyHashMixShift1 = 27;
constexpr uint32_t kKeyHashMixShift2 = 31;
constexpr uint32_t kGraphDefaultNeighbors = 64;
constexpr uint32_t kGraphMaxNeighbors = 64;
constexpr uint32_t kGraphBaseNeighborMultiplier = 2;
constexpr uint32_t kGraphMaxBaseNeighbors = kGraphMaxNeighbors * kGraphBaseNeighborMultiplier;
constexpr uint32_t kGraphDefaultEfConstruction = 128;
constexpr uint32_t kGraphDefaultEfSearch = 64;
constexpr uint32_t kGraphMinSearch = 4;
constexpr uint32_t kGraphCandidateReserveMultiplier = 4;
constexpr uint32_t kGraphLongLinkCount = 0;
constexpr uint32_t kGraphNeighborPrefetchDistance = 2;
constexpr uint32_t kGraphF32RerankMinCandidates = 256;
constexpr uint32_t kGraphF32RerankTopKMultiplier = 32;
constexpr uint32_t kGraphDefaultQueryCapacityMultiplier = 1;
constexpr uint32_t kGraphQ8RerankDefaultQueryCapacityMultiplier = 1;
constexpr uint32_t kGraphRerankDefaultQueryCapacityMultiplier = 2;
constexpr uint16_t kGraphVisitGenerationStart = 1;
constexpr uint64_t kBytesPerKiB = 1024;
constexpr uint64_t kBytesPerMiB = kBytesPerKiB * kBytesPerKiB;
constexpr uint64_t kGraphCompactExactSearchMaxBytes = 16 * kBytesPerMiB;
constexpr uint32_t kFlatQ8PrefetchDistance = 4;
constexpr uint32_t kFlatQ8PrefetchMinCount = 32768;
constexpr uint32_t kGraphMaxLevels = 16;
constexpr uint32_t kMemoryBatchStackQueries = 16;
constexpr uint32_t kMemoryAddStackSlots = 256;
constexpr uint32_t kMemoryAddPlanStackEntries = kMemoryAddStackSlots * 2;
constexpr uint32_t kMemoryAddParallelMinCount = 1024;
constexpr uint32_t kMemoryAddParallelMaxWorkers = 8;
constexpr uint32_t kMemorySearchBatchParallelMinQueries = 8;
constexpr uint32_t kMemorySearchBatchParallelMaxRecords = 32768;
constexpr uint32_t kMemorySearchBatchParallelMaxTopK = 8;
constexpr uint32_t kMemorySearchRecordParallelMaxTopK = 10;


inline void prefetch_dense_q8_slot(const int8_t* vectors, const float* scales,
                                   const MemorySlot* slots, uint32_t dim, uint32_t slot) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(vectors + static_cast<size_t>(slot) * dim, 0, 1);
  __builtin_prefetch(scales + slot, 0, 1);
  __builtin_prefetch(slots + slot, 0, 1);
#else
  (void)vectors;
  (void)scales;
  (void)slots;
  (void)dim;
  (void)slot;
#endif
}

inline bool metric_valid(AstralMemoryMetric metric) {
  return metric == ASTRAL_MEMORY_METRIC_DOT || metric == ASTRAL_MEMORY_METRIC_COSINE ||
         metric == ASTRAL_MEMORY_METRIC_L2;
}

inline bool index_kind_valid(AstralMemoryIndexKind kind) {
  return kind == ASTRAL_MEMORY_INDEX_FLAT || kind == ASTRAL_MEMORY_INDEX_GRAPH;
}

inline bool storage_kind_valid(AstralMemoryStorageKind kind) {
  return kind == ASTRAL_MEMORY_STORAGE_F32 || kind == ASTRAL_MEMORY_STORAGE_Q8 ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 || kind == ASTRAL_MEMORY_STORAGE_F8_E5M2 ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E3M2 || kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E2M3_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK;
}


inline bool desc_valid(const AstralMemoryIndexDesc* desc) {
  return desc != nullptr && desc->size == sizeof(AstralMemoryIndexDesc) && desc->dim != 0 &&
         desc->dim <= kMaxDim && desc->capacity != 0 && index_kind_valid(desc->index_kind) &&
         metric_valid(desc->metric) && storage_kind_valid(desc->storage_kind);
}

inline uint32_t graph_neighbors_from_desc(const AstralMemoryIndexDesc* desc) {
  if (desc->index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    return 0;
  }
  const uint32_t requested =
      desc->graph_neighbors != 0 ? desc->graph_neighbors : kGraphDefaultNeighbors;
  return requested < kGraphMaxNeighbors ? requested : kGraphMaxNeighbors;
}

inline uint32_t graph_base_neighbors_from_graph_neighbors(uint32_t neighbors) {
  const uint32_t doubled = neighbors * kGraphBaseNeighborMultiplier;
  return doubled < kGraphMaxBaseNeighbors ? doubled : kGraphMaxBaseNeighbors;
}

inline uint32_t graph_search_from_desc(const AstralMemoryIndexDesc* desc) {
  if (desc->index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    return 0;
  }
  const uint32_t requested =
      desc->graph_search != 0 ? desc->graph_search : kGraphDefaultEfConstruction;
  return requested < kGraphMinSearch ? kGraphMinSearch : requested;
}

inline uint32_t graph_default_query_search_from_desc(const AstralMemoryIndexDesc* desc) {
  const uint32_t neighbors = graph_neighbors_from_desc(desc);
  if (neighbors == 0) {
    return 0;
  }
  uint32_t scaled = desc->capacity / neighbors;
  if ((desc->capacity % neighbors) != 0) {
    ++scaled;
  }
  const uint32_t multiplier = desc->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK
                                  ? kGraphQ8RerankDefaultQueryCapacityMultiplier
                                  : (f32_rerank_storage_kind(desc->storage_kind)
                                         ? kGraphRerankDefaultQueryCapacityMultiplier
                                         : kGraphDefaultQueryCapacityMultiplier);
  if (scaled <= kU32Max / multiplier) {
    scaled *= multiplier;
  } else {
    scaled = kU32Max;
  }
  return scaled > kGraphDefaultEfSearch ? scaled : kGraphDefaultEfSearch;
}

inline uint32_t graph_query_search_from_desc(const AstralMemoryIndexDesc* desc) {
  if (desc->index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    return 0;
  }
  uint32_t requested = desc->graph_query_search != 0 ? desc->graph_query_search
                                                     : graph_default_query_search_from_desc(desc);
  if (requested < kGraphMinSearch) {
    requested = kGraphMinSearch;
  }
  return requested;
}

uint32_t graph_level_capacity_for(uint32_t capacity, uint32_t neighbors) {
  if (neighbors == 0) {
    return 0;
  }
  uint32_t levels = 1;
  uint32_t remaining = capacity;
  while (remaining >= neighbors && levels < kGraphMaxLevels) {
    remaining /= neighbors;
    ++levels;
  }
  return levels;
}

inline float* alloc_vector_storage(uint32_t capacity, uint32_t dim) {
  const size_t count = static_cast<size_t>(capacity) * dim;
  return static_cast<float*>(core::runtime_alloc(count * sizeof(float), kVectorStorageAlign));
}

inline int8_t* alloc_q8_vector_storage(uint32_t capacity, uint32_t dim) {
  const size_t count = static_cast<size_t>(capacity) * dim;
  return static_cast<int8_t*>(core::runtime_alloc(count * sizeof(int8_t), kVectorStorageAlign));
}

inline int16_t* alloc_i16_vector_storage(uint32_t capacity, uint32_t dim) {
  const size_t count = static_cast<size_t>(capacity) * dim;
  return static_cast<int16_t*>(core::runtime_alloc(count * sizeof(int16_t), kVectorStorageAlign));
}

inline bool i16_vector_stride_aligned(uint32_t dim) {
  return ((static_cast<uint64_t>(dim) * sizeof(int16_t)) &
          static_cast<uint64_t>(kVectorStorageAlign - 1u)) == 0;
}

inline void free_vector_storage(float* ptr, uint32_t capacity, uint32_t dim) {
  core::runtime_free(ptr, static_cast<size_t>(capacity) * dim * sizeof(float), kVectorStorageAlign);
}

inline void free_q8_vector_storage(int8_t* ptr, uint32_t capacity, uint32_t dim) {
  core::runtime_free(ptr, static_cast<size_t>(capacity) * dim * sizeof(int8_t),
                     kVectorStorageAlign);
}

inline void free_i16_vector_storage(int16_t* ptr, uint32_t capacity, uint32_t dim) {
  core::runtime_free(ptr, static_cast<size_t>(capacity) * dim * sizeof(int16_t),
                     kVectorStorageAlign);
}

} // namespace

AstralHandle memory_handle(MemoryIndex* index) {
  return index != nullptr ? index->handle : 0;
}

AstralHandle memory_search_cursor_handle(MemorySearchCursor* cursor) {
  return cursor != nullptr ? cursor->handle : 0;
}

namespace {

uint32_t memory_add_plan_capacity(uint32_t count) {
  if (count > (1u << 30)) {
    return 0;
  }
  const uint32_t target = count * 2u;
  uint32_t capacity = 1;
  while (capacity < target) {
    capacity <<= 1u;
  }
  return capacity;
}

struct MemoryAddBatchScratch {
  uint32_t stack_storage[kMemoryAddStackSlots];
  uint32_t* storage = stack_storage;
  uint32_t* slots = stack_storage;
  uint32_t storage_count = kMemoryAddStackSlots;

  MemoryAddBatchScratch() = default;

  bool init(uint32_t count) {
    if (count > kMemoryAddStackSlots) {
      storage_count = count;
      storage = core::runtime_alloc_array<uint32_t>(storage_count);
      if (storage == nullptr) {
        return false;
      }
      slots = storage;
    }
    return true;
  }

  ~MemoryAddBatchScratch() {
    if (storage != nullptr && storage != stack_storage) {
      core::runtime_free_array(storage, storage_count);
    }
  }

  MemoryAddBatchScratch(const MemoryAddBatchScratch&) = delete;
  MemoryAddBatchScratch& operator=(const MemoryAddBatchScratch&) = delete;
};

struct MemoryAddSeenEntry {
  uint32_t slot;
  uint8_t occupied;
};

struct MemoryAddSeenScratch {
  MemoryAddSeenEntry stack_entries[kMemoryAddPlanStackEntries];
  MemoryAddSeenEntry* entries = stack_entries;
  uint32_t capacity = kMemoryAddPlanStackEntries;

  MemoryAddSeenScratch() = default;

  bool init(uint32_t count) {
    const uint32_t required = memory_add_plan_capacity(count);
    if (required == 0) {
      return false;
    }
    capacity = required;
    if (required > kMemoryAddPlanStackEntries) {
      entries = core::runtime_alloc_array<MemoryAddSeenEntry>(required);
      if (entries == nullptr) {
        return false;
      }
    }
    std::memset(entries, 0, sizeof(MemoryAddSeenEntry) * capacity);
    return true;
  }

  bool mark(uint32_t slot) {
    uint32_t pos = (slot * 0x9e3779b1u) & (capacity - 1u);
    while (entries[pos].occupied != 0) {
      if (entries[pos].slot == slot) {
        return false;
      }
      pos = (pos + 1u) & (capacity - 1u);
    }
    entries[pos].slot = slot;
    entries[pos].occupied = 1;
    return true;
  }

  ~MemoryAddSeenScratch() {
    if (entries != nullptr && entries != stack_entries) {
      core::runtime_free_array(entries, capacity);
    }
  }

  MemoryAddSeenScratch(const MemoryAddSeenScratch&) = delete;
  MemoryAddSeenScratch& operator=(const MemoryAddSeenScratch&) = delete;
};

struct MemoryAddPreprocessJob {
  MemoryIndex* index;
  const float* vectors;
  const uint32_t* slots;
  uint32_t begin;
  uint32_t end;
  uint8_t validate_vectors;
  std::atomic<uint32_t>* remaining;
  std::atomic<uint32_t>* invalid;
};

struct MemorySearchBatchJob {
  MemoryIndex* index;
  const AstralMemorySearchDesc* desc;
  const float* queries;
  AstralMemorySearchResult* out_results;
  uint32_t* out_counts;
  uint32_t begin;
  uint32_t end;
  std::atomic<uint32_t>* remaining;
};

struct MemorySearchRecordShardJob {
  MemoryIndex* index;
  const AstralMemorySearchDesc* desc;
  const float* queries;
  uint32_t query_count;
  uint32_t begin;
  uint32_t end;
  std::atomic<uint32_t>* remaining;
  AstralMemorySearchResult
      local_results[kMemoryBatchStackQueries * kMemorySearchBatchParallelMaxTopK];
  uint32_t local_counts[kMemoryBatchStackQueries];
};

struct MemorySearchSingleRecordShardJob {
  MemoryIndex* index;
  const AstralMemorySearchDesc* desc;
  const float* query;
  uint32_t begin;
  uint32_t end;
  std::atomic<uint32_t>* remaining;
  AstralMemorySearchResult local_results[kMemorySearchRecordParallelMaxTopK];
  uint32_t local_count;
};


inline void memory_parallel_job_complete(std::atomic<uint32_t>* remaining) {
  if (remaining->fetch_sub(1, std::memory_order_release) == 1u) {
    astral::platform::cpu_signal_event();
  }
}

} // namespace

namespace {

inline uint32_t slot_to_key_ref(uint32_t slot) {
  return slot + kKeyTableSlotRefBias;
}

inline uint32_t key_ref_to_slot(uint32_t ref) {
  return ref - kKeyTableSlotRefBias;
}

inline uint64_t key_hash_mix(uint64_t x) {
  x ^= x >> kKeyHashMixShift0;
  x *= kKeyHashMixMul0;
  x ^= x >> kKeyHashMixShift1;
  x *= kKeyHashMixMul1;
  x ^= x >> kKeyHashMixShift2;
  return x;
}

uint32_t key_table_capacity_for(uint32_t capacity) {
  uint32_t table_capacity = 1;
  const uint32_t target =
      capacity < kKeyTableLoadFactorDen ? kKeyTableMinCapacity : capacity * kKeyTableLoadFactorDen;
  while (table_capacity < target) {
    table_capacity <<= 1u;
  }
  return table_capacity;
}

uint32_t find_slot_by_key_hashed(const MemoryIndex* index, uint64_t key, uint64_t hash) {
  uint32_t table_pos = static_cast<uint32_t>(hash) & index->key_table_mask;
  for (uint32_t probe = 0; probe < index->key_table_capacity; ++probe) {
    const uint32_t ref = index->key_table[table_pos];
    if (ref == kKeyTableEmpty) {
      return kU32Max;
    }
    if (ref != kKeyTableTombstone) {
      const uint32_t slot = key_ref_to_slot(ref);
      if (index->slots[slot].occupied != 0 && index->slots[slot].record.key == key) {
        return slot;
      }
    }
    table_pos = (table_pos + 1u) & index->key_table_mask;
  }
  return kU32Max;
}

uint32_t find_slot_by_key(const MemoryIndex* index, uint64_t key) {
  return find_slot_by_key_hashed(index, key, key_hash_mix(key));
}

uint32_t find_free_slot(const MemoryIndex* index) {
  uint32_t slot = index->free_slot_hint;
  for (uint32_t probe = 0; probe < index->capacity; ++probe) {
    if (index->slots[slot].occupied == 0) {
      return slot;
    }
    slot += kSlotAdvance;
    if (slot == index->capacity) {
      slot = 0;
    }
  }
  return kU32Max;
}

uint32_t graph_level_for_key(const MemoryIndex* index, uint64_t key) {
  if (!graph_enabled(index)) {
    return 0;
  }
  uint32_t level = 0;
  uint64_t hash = key_hash_mix(key);
  const uint32_t threshold = kU32Max / index->graph_neighbor_capacity;
  while (level + 1u < index->graph_level_capacity && static_cast<uint32_t>(hash) <= threshold) {
    ++level;
    hash = key_hash_mix(hash + key);
  }
  return level;
}

AstralErr key_table_insert_new_hashed(MemoryIndex* index, uint64_t hash, uint32_t slot) {
  uint32_t table_pos = static_cast<uint32_t>(hash) & index->key_table_mask;
  uint32_t tombstone_pos = kU32Max;
  for (uint32_t probe = 0; probe < index->key_table_capacity; ++probe) {
    const uint32_t ref = index->key_table[table_pos];
    if (ref == kKeyTableEmpty) {
      index->key_table[tombstone_pos != kU32Max ? tombstone_pos : table_pos] =
          slot_to_key_ref(slot);
      return ASTRAL_OK;
    }
    if (ref == kKeyTableTombstone) {
      if (tombstone_pos == kU32Max) {
        tombstone_pos = table_pos;
      }
    }
    table_pos = (table_pos + 1u) & index->key_table_mask;
  }
  if (tombstone_pos != kU32Max) {
    index->key_table[tombstone_pos] = slot_to_key_ref(slot);
    return ASTRAL_OK;
  }
  return ASTRAL_E_NOMEM;
}

void key_table_remove(MemoryIndex* index, uint64_t key) {
  uint32_t table_pos = static_cast<uint32_t>(key_hash_mix(key)) & index->key_table_mask;
  for (uint32_t probe = 0; probe < index->key_table_capacity; ++probe) {
    const uint32_t ref = index->key_table[table_pos];
    if (ref == kKeyTableEmpty) {
      return;
    }
    if (ref != kKeyTableTombstone) {
      const uint32_t slot = key_ref_to_slot(ref);
      if (index->slots[slot].occupied != 0 && index->slots[slot].record.key == key) {
        index->key_table[table_pos] = kKeyTableTombstone;
        return;
      }
    }
    table_pos = (table_pos + 1u) & index->key_table_mask;
  }
}

void key_table_rebuild(MemoryIndex* index) {
  std::memset(index->key_table, 0, sizeof(uint32_t) * index->key_table_capacity);
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = index->active_slots[active_pos];
    const uint64_t key = index->slots[slot].record.key;
    const AstralErr err = key_table_insert_new_hashed(index, key_hash_mix(key), slot);
    (void)err;
  }
}


void memory_search_flat(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                        AstralMemorySearchResult* out_results, uint32_t* out_count);
void memory_search_flat_batch(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* queries, uint32_t query_count,
                              AstralMemorySearchResult* out_results, uint32_t* out_counts);
bool memory_search_flat_batch_record_parallel(MemoryIndex* index,
                                              const AstralMemorySearchDesc* desc,
                                              const float* queries, uint32_t query_count,
                                              AstralMemorySearchResult* out_results,
                                              uint32_t* out_counts);
bool memory_search_flat_record_parallel(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                        const float* query, AstralMemorySearchResult* out_results,
                                        uint32_t* out_count);
void memory_search_flat_single_range(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                     const float* query, uint32_t begin_active, uint32_t end_active,
                                     AstralMemorySearchResult* out_results, uint32_t* out_count);
void store_f32_vector(MemoryIndex* index, uint32_t slot, const float* src) {
  float* dst = vector_at(index, slot);
  if (index->metric != ASTRAL_MEMORY_METRIC_COSINE) {
    std::memcpy(dst, src, sizeof(float) * index->dim);
    index->slots[slot].score_scale = 0.0f;
    return;
  }

  normalize_f32_vector(dst, src, index->dim);
  index->slots[slot].score_scale = 1.0f;
}

void memory_search_dot(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                       AstralMemorySearchResult* out_results, uint32_t* out_count) {
  uint32_t filled = 0;
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = dot_f32(query, vector_at(index, slot), index->dim);
    if (filled == desc->top_k &&
        !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
      continue;
    }

    AstralMemorySearchResult candidate;
    fill_result(&candidate, s, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  *out_count = filled;
}

void memory_search_q8(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                      AstralMemorySearchResult* out_results, uint32_t* out_count) {
  const float query_scale =
      index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(query, index->dim) : 0.0f;
  if (desc->group_id == ASTRAL_MEMORY_GROUP_ANY && index->dense_active != 0 &&
      e5m2_storage(index) && !f8_f32_rerank_storage(index)) {
    uint32_t filled = 0;
    MemorySlot* slots = index->slots;
    const uint32_t dim = index->dim;
    const int8_t* vectors = index->q8_vectors;
    const bool use_prefetch = index->count >= kFlatQ8PrefetchMinCount;
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT || index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      const DotE5m2F32Fn dot = index->e5m2_kernels->dot_f32;
      const float* scales = index->compact_score_scales;
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
          prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
        }
        MemorySlot& s = slots[slot];
        float score = dot(vectors + static_cast<size_t>(slot) * dim, query, dim) * scales[slot];
        if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
          score *= query_scale;
        }
        if (filled == desc->top_k &&
            !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
          continue;
        }
        AstralMemorySearchResult candidate{};
        fill_result(&candidate, s, score);
        insert_result(out_results, desc->top_k, &filled, candidate);
      }
    } else {
      const L2E5m2F32Fn l2 = index->e5m2_kernels->l2_f32;
      const float* scales = index->q8_scales;
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
          prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
        }
        MemorySlot& s = slots[slot];
        const float score = l2(vectors + static_cast<size_t>(slot) * dim,
                               compact_value_scale(index, scales[slot]), query, dim);
        if (filled == desc->top_k &&
            !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
          continue;
        }
        AstralMemorySearchResult candidate{};
        fill_result(&candidate, s, score);
        insert_result(out_results, desc->top_k, &filled, candidate);
      }
    }
    *out_count = filled;
    return;
  }
  if (desc->group_id == ASTRAL_MEMORY_GROUP_ANY && index->dense_active != 0 &&
      !e5m2_storage(index) && !i16_storage(index) && !q8_f32_rerank_storage(index)) {
    uint32_t filled = 0;
    MemorySlot* slots = index->slots;
    const uint32_t dim = index->dim;
    const int8_t* vectors = index->q8_vectors;
    const float* scales = index->q8_scales;
    const bool use_prefetch = index->count >= kFlatQ8PrefetchMinCount;
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
          prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
        }
        MemorySlot& s = slots[slot];
        const float score = dot_q8_f32(vectors + static_cast<size_t>(slot) * dim, query, dim) *
                            compact_value_scale(index, scales[slot]);
        if (filled == desc->top_k &&
            !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
          continue;
        }
        AstralMemorySearchResult candidate;
        fill_result(&candidate, s, score);
        insert_result(out_results, desc->top_k, &filled, candidate);
      }
    } else if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
          prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
        }
        MemorySlot& s = slots[slot];
        const float score = dot_q8_f32(vectors + static_cast<size_t>(slot) * dim, query, dim) *
                            compact_value_scale(index, scales[slot]) * query_scale * s.score_scale;
        if (filled == desc->top_k &&
            !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
          continue;
        }
        AstralMemorySearchResult candidate{};
        fill_result(&candidate, s, score);
        insert_result(out_results, desc->top_k, &filled, candidate);
      }
    } else {
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
          prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
        }
        MemorySlot& s = slots[slot];
        const float score = l2_score_q8_f32(vectors + static_cast<size_t>(slot) * dim,
                                            compact_value_scale(index, scales[slot]), query, dim);
        if (filled == desc->top_k &&
            !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
          continue;
        }
        AstralMemorySearchResult candidate{};
        fill_result(&candidate, s, score);
        insert_result(out_results, desc->top_k, &filled, candidate);
      }
    }
    *out_count = filled;
    return;
  }

  uint32_t filled = 0;
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = score_slot(index, query, slot, query_scale);
    if (filled == desc->top_k &&
        !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
      continue;
    }

    AstralMemorySearchResult candidate{};
    fill_result(&candidate, s, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  *out_count = filled;
}

void memory_search_q8_top1(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                           const float* query, AstralMemorySearchResult* out_results,
                           uint32_t* out_count) {
  const float query_scale =
      index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(query, index->dim) : 0.0f;
  if (desc->group_id == ASTRAL_MEMORY_GROUP_ANY) {
    if (index->count == 0) {
      *out_count = kNoResults;
      return;
    }

    if (index->dense_active != 0 && !e5m2_storage(index) && !i16_storage(index) &&
        !q8_f32_rerank_storage(index)) {
      MemorySlot* slots = index->slots;
      const uint32_t dim = index->dim;
      const int8_t* vectors = index->q8_vectors;
      const float* scales = index->q8_scales;
      const bool use_prefetch = index->count >= kFlatQ8PrefetchMinCount;
      MemorySlot* best_slot = &slots[0];
      float best_score = 0.0f;
      if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
        best_score = dot_q8_f32(vectors, query, dim) * compact_value_scale(index, scales[0]);
        for (uint32_t slot = 1; slot < index->count; ++slot) {
          if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
            prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
          }
          MemorySlot& s = slots[slot];
          const float score = dot_q8_f32(vectors + static_cast<size_t>(slot) * dim, query, dim) *
                              compact_value_scale(index, scales[slot]);
          if (score > best_score || (score == best_score && s.record.key < best_slot->record.key)) {
            best_slot = &s;
            best_score = score;
          }
        }
      } else if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        best_score = dot_q8_f32(vectors, query, dim) * compact_value_scale(index, scales[0]) *
                     query_scale * best_slot->score_scale;
        for (uint32_t slot = 1; slot < index->count; ++slot) {
          if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
            prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
          }
          MemorySlot& s = slots[slot];
          const float score = dot_q8_f32(vectors + static_cast<size_t>(slot) * dim, query, dim) *
                              compact_value_scale(index, scales[slot]) * query_scale *
                              s.score_scale;
          if (score > best_score || (score == best_score && s.record.key < best_slot->record.key)) {
            best_slot = &s;
            best_score = score;
          }
        }
      } else {
        best_score = l2_score_q8_f32(vectors, compact_value_scale(index, scales[0]), query, dim);
        for (uint32_t slot = 1; slot < index->count; ++slot) {
          if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
            prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
          }
          MemorySlot& s = slots[slot];
          const float score = l2_score_q8_f32(vectors + static_cast<size_t>(slot) * dim,
                                              compact_value_scale(index, scales[slot]), query, dim);
          if (score > best_score || (score == best_score && s.record.key < best_slot->record.key)) {
            best_slot = &s;
            best_score = score;
          }
        }
      }

      fill_result(out_results, *best_slot, best_score);
      *out_count = kTopOne;
      return;
    }
  }

  const MemorySlot* best_slot = nullptr;
  float best_score = 0.0f;
  uint64_t best_key = 0;
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = score_slot(index, query, slot, query_scale);
    if (best_slot == nullptr || score > best_score ||
        (score == best_score && s.record.key < best_key)) {
      best_slot = &s;
      best_score = score;
      best_key = s.record.key;
    }
  }

  if (best_slot == nullptr) {
    *out_count = kNoResults;
    return;
  }

  fill_result(out_results, *best_slot, best_score);
  *out_count = kTopOne;
}

void memory_search_cosine(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                          const float* query, AstralMemorySearchResult* out_results,
                          uint32_t* out_count) {
  uint32_t filled = 0;
  alignas(kVectorStorageAlign) float normalized_query[kMaxDim];
  normalize_f32_vector(normalized_query, query, index->dim);
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = dot_f32(normalized_query, vector_at(index, slot), index->dim);
    if (filled == desc->top_k &&
        !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
      continue;
    }

    AstralMemorySearchResult candidate{};
    fill_result(&candidate, s, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  *out_count = filled;
}

void memory_search_l2(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                      AstralMemorySearchResult* out_results, uint32_t* out_count) {
  uint32_t filled = 0;
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = l2_score_f32(query, vector_at(index, slot), index->dim);
    if (filled == desc->top_k &&
        !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
      continue;
    }

    AstralMemorySearchResult candidate{};
    fill_result(&candidate, s, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  *out_count = filled;
}

void memory_search_dot_top1(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                            const float* query, AstralMemorySearchResult* out_results,
                            uint32_t* out_count) {
  if (desc->group_id == ASTRAL_MEMORY_GROUP_ANY) {
    if (index->count == 0) {
      *out_count = kNoResults;
      return;
    }

    if (index->dense_active != 0) {
      MemorySlot* slots = index->slots;
      const uint32_t dim = index->dim;
      const float* vectors = index->vectors;
      MemorySlot* best_slot = &slots[0];
      float best_score = dot_f32(query, vectors, dim);
      uint64_t best_key = best_slot->record.key;
      for (uint32_t slot = 1; slot < index->count; ++slot) {
        MemorySlot& s = slots[slot];
        const float score = dot_f32(query, vectors + static_cast<size_t>(slot) * dim, dim);
        if (score > best_score || (score == best_score && s.record.key < best_key)) {
          best_slot = &s;
          best_score = score;
          best_key = s.record.key;
        }
      }

      fill_result(out_results, *best_slot, best_score);
      *out_count = kTopOne;
      return;
    }

    const uint32_t first_slot = active_slot_at(index, 0);
    const MemorySlot* best_slot = &index->slots[first_slot];
    float best_score = dot_f32(query, vector_at(index, first_slot), index->dim);
    uint64_t best_key = best_slot->record.key;
    for (uint32_t active_pos = 1; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const MemorySlot& s = index->slots[slot];
      const float score = dot_f32(query, vector_at(index, slot), index->dim);
      if (score > best_score || (score == best_score && s.record.key < best_key)) {
        best_slot = &s;
        best_score = score;
        best_key = s.record.key;
      }
    }

    fill_result(out_results, *best_slot, best_score);
    *out_count = kTopOne;
    return;
  }

  const MemorySlot* best_slot = nullptr;
  float best_score = 0.0f;
  uint64_t best_key = 0;
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = dot_f32(query, vector_at(index, slot), index->dim);
    if (best_slot == nullptr || score > best_score ||
        (score == best_score && s.record.key < best_key)) {
      best_slot = &s;
      best_score = score;
      best_key = s.record.key;
    }
  }

  if (best_slot == nullptr) {
    *out_count = kNoResults;
    return;
  }

  fill_result(out_results, *best_slot, best_score);
  *out_count = kTopOne;
}

void memory_search_cosine_top1(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                               const float* query, AstralMemorySearchResult* out_results,
                               uint32_t* out_count) {
  alignas(kVectorStorageAlign) float normalized_query[kMaxDim];
  normalize_f32_vector(normalized_query, query, index->dim);
  if (desc->group_id == ASTRAL_MEMORY_GROUP_ANY) {
    if (index->count == 0) {
      *out_count = kNoResults;
      return;
    }

    if (index->dense_active != 0) {
      MemorySlot* slots = index->slots;
      const uint32_t dim = index->dim;
      const float* vectors = index->vectors;
      MemorySlot* best_slot = &slots[0];
      float best_score = dot_f32(normalized_query, vectors, dim);
      uint64_t best_key = best_slot->record.key;
      for (uint32_t slot = 1; slot < index->count; ++slot) {
        MemorySlot& s = slots[slot];
        const float score =
            dot_f32(normalized_query, vectors + static_cast<size_t>(slot) * dim, dim);
        if (score > best_score || (score == best_score && s.record.key < best_key)) {
          best_slot = &s;
          best_score = score;
          best_key = s.record.key;
        }
      }

      fill_result(out_results, *best_slot, best_score);
      *out_count = kTopOne;
      return;
    }

    const uint32_t first_slot = active_slot_at(index, 0);
    const MemorySlot* best_slot = &index->slots[first_slot];
    float best_score = dot_f32(normalized_query, vector_at(index, first_slot), index->dim);
    uint64_t best_key = best_slot->record.key;
    for (uint32_t active_pos = 1; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const MemorySlot& s = index->slots[slot];
      const float score = dot_f32(normalized_query, vector_at(index, slot), index->dim);
      if (score > best_score || (score == best_score && s.record.key < best_key)) {
        best_slot = &s;
        best_score = score;
        best_key = s.record.key;
      }
    }

    fill_result(out_results, *best_slot, best_score);
    *out_count = kTopOne;
    return;
  }

  const MemorySlot* best_slot = nullptr;
  float best_score = 0.0f;
  uint64_t best_key = 0;
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = dot_f32(normalized_query, vector_at(index, slot), index->dim);
    if (best_slot == nullptr || score > best_score ||
        (score == best_score && s.record.key < best_key)) {
      best_slot = &s;
      best_score = score;
      best_key = s.record.key;
    }
  }

  if (best_slot == nullptr) {
    *out_count = kNoResults;
    return;
  }

  fill_result(out_results, *best_slot, best_score);
  *out_count = kTopOne;
}

void memory_search_l2_top1(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                           const float* query, AstralMemorySearchResult* out_results,
                           uint32_t* out_count) {
  if (desc->group_id == ASTRAL_MEMORY_GROUP_ANY) {
    if (index->count == 0) {
      *out_count = kNoResults;
      return;
    }

    if (index->dense_active != 0) {
      MemorySlot* slots = index->slots;
      const uint32_t dim = index->dim;
      const float* vectors = index->vectors;
      MemorySlot* best_slot = &slots[0];
      float best_score = l2_score_f32(query, vectors, dim);
      uint64_t best_key = best_slot->record.key;
      for (uint32_t slot = 1; slot < index->count; ++slot) {
        MemorySlot& s = slots[slot];
        const float score = l2_score_f32(query, vectors + static_cast<size_t>(slot) * dim, dim);
        if (score > best_score || (score == best_score && s.record.key < best_key)) {
          best_slot = &s;
          best_score = score;
          best_key = s.record.key;
        }
      }

      fill_result(out_results, *best_slot, best_score);
      *out_count = kTopOne;
      return;
    }

    const uint32_t first_slot = active_slot_at(index, 0);
    const MemorySlot* best_slot = &index->slots[first_slot];
    float best_score = l2_score_f32(query, vector_at(index, first_slot), index->dim);
    uint64_t best_key = best_slot->record.key;
    for (uint32_t active_pos = 1; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const MemorySlot& s = index->slots[slot];
      const float score = l2_score_f32(query, vector_at(index, slot), index->dim);
      if (score > best_score || (score == best_score && s.record.key < best_key)) {
        best_slot = &s;
        best_score = score;
        best_key = s.record.key;
      }
    }

    fill_result(out_results, *best_slot, best_score);
    *out_count = kTopOne;
    return;
  }

  const MemorySlot* best_slot = nullptr;
  float best_score = 0.0f;
  uint64_t best_key = 0;
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = l2_score_f32(query, vector_at(index, slot), index->dim);
    if (best_slot == nullptr || score > best_score ||
        (score == best_score && s.record.key < best_key)) {
      best_slot = &s;
      best_score = score;
      best_key = s.record.key;
    }
  }

  if (best_slot == nullptr) {
    *out_count = kNoResults;
    return;
  }

  fill_result(out_results, *best_slot, best_score);
  *out_count = kTopOne;
}

void memory_search_flat(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                        AstralMemorySearchResult* out_results, uint32_t* out_count) {
  if (memory_search_flat_record_parallel(index, desc, query, out_results, out_count)) {
    return;
  }
  if (index->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2) {
    memory_search_flat_single_range(index, desc, query, 0u, index->count, out_results, out_count);
    return;
  }
  if (compact_storage(index)) {
    if (desc->top_k == kTopOne) {
      memory_search_q8_top1(index, desc, query, out_results, out_count);
    } else {
      memory_search_q8(index, desc, query, out_results, out_count);
    }
    return;
  }
  if (desc->top_k == kTopOne) {
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
      memory_search_dot_top1(index, desc, query, out_results, out_count);
    } else if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      memory_search_cosine_top1(index, desc, query, out_results, out_count);
    } else {
      memory_search_l2_top1(index, desc, query, out_results, out_count);
    }
  } else if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    memory_search_dot(index, desc, query, out_results, out_count);
  } else if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    memory_search_cosine(index, desc, query, out_results, out_count);
  } else {
    memory_search_l2(index, desc, query, out_results, out_count);
  }
}

void memory_search_flat_batch(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* queries, uint32_t query_count,
                              AstralMemorySearchResult* out_results, uint32_t* out_counts) {
  float query_scales[kMemoryBatchStackQueries];
  for (uint32_t query_i = 0; query_i < query_count; ++query_i) {
    const float* query = queries + static_cast<size_t>(query_i) * index->dim;
    query_scales[query_i] =
        index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(query, index->dim) : 1.0f;
    out_counts[query_i] = kNoResults;
  }

  if (compact_storage(index) && desc->group_id == ASTRAL_MEMORY_GROUP_ANY &&
      index->dense_active != 0 && !e5m2_storage(index) && !i16_storage(index) &&
      !q8_f32_rerank_storage(index)) {
    MemorySlot* slots = index->slots;
    const uint32_t dim = index->dim;
    const int8_t* vectors = index->q8_vectors;
    const float* scales = index->q8_scales;
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        MemorySlot& s = slots[slot];
        const int8_t* vector = vectors + static_cast<size_t>(slot) * dim;
        const float scale = compact_value_scale(index, scales[slot]);
        for (uint32_t query_i = 0; query_i < query_count; ++query_i) {
          AstralMemorySearchResult* results =
              out_results + static_cast<size_t>(query_i) * desc->top_k;
          uint32_t* filled = out_counts + query_i;
          const float* query = queries + static_cast<size_t>(query_i) * dim;
          const float score = dot_q8_f32(vector, query, dim) * scale;
          if (*filled == desc->top_k &&
              !result_better_values(score, s.record.key, results[desc->top_k - 1u])) {
            continue;
          }
          AstralMemorySearchResult candidate{};
          fill_result(&candidate, s, score);
          insert_result(results, desc->top_k, filled, candidate);
        }
      }
    } else if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        MemorySlot& s = slots[slot];
        const int8_t* vector = vectors + static_cast<size_t>(slot) * dim;
        const float scale = compact_value_scale(index, scales[slot]) * s.score_scale;
        for (uint32_t query_i = 0; query_i < query_count; ++query_i) {
          AstralMemorySearchResult* results =
              out_results + static_cast<size_t>(query_i) * desc->top_k;
          uint32_t* filled = out_counts + query_i;
          const float* query = queries + static_cast<size_t>(query_i) * dim;
          const float score = dot_q8_f32(vector, query, dim) * scale * query_scales[query_i];
          if (*filled == desc->top_k &&
              !result_better_values(score, s.record.key, results[desc->top_k - 1u])) {
            continue;
          }
          AstralMemorySearchResult candidate{};
          fill_result(&candidate, s, score);
          insert_result(results, desc->top_k, filled, candidate);
        }
      }
    } else {
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        MemorySlot& s = slots[slot];
        const int8_t* vector = vectors + static_cast<size_t>(slot) * dim;
        const float scale = compact_value_scale(index, scales[slot]);
        for (uint32_t query_i = 0; query_i < query_count; ++query_i) {
          AstralMemorySearchResult* results =
              out_results + static_cast<size_t>(query_i) * desc->top_k;
          uint32_t* filled = out_counts + query_i;
          const float* query = queries + static_cast<size_t>(query_i) * dim;
          const float score = l2_score_q8_f32(vector, scale, query, dim);
          if (*filled == desc->top_k &&
              !result_better_values(score, s.record.key, results[desc->top_k - 1u])) {
            continue;
          }
          AstralMemorySearchResult candidate{};
          fill_result(&candidate, s, score);
          insert_result(results, desc->top_k, filled, candidate);
        }
      }
    }
    return;
  }

  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    for (uint32_t query_i = 0; query_i < query_count; ++query_i) {
      AstralMemorySearchResult* results = out_results + static_cast<size_t>(query_i) * desc->top_k;
      uint32_t* filled = out_counts + query_i;
      const float* query = queries + static_cast<size_t>(query_i) * index->dim;
      const float score = score_slot(index, query, slot, query_scales[query_i]);
      if (*filled == desc->top_k &&
          !result_better_values(score, s.record.key, results[desc->top_k - 1u])) {
        continue;
      }
      AstralMemorySearchResult candidate{};
      fill_result(&candidate, s, score);
      insert_result(results, desc->top_k, filled, candidate);
    }
  }
}

void memory_search_flat_batch_chunked(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                      const float* queries, uint32_t query_count,
                                      AstralMemorySearchResult* out_results, uint32_t* out_counts) {
  for (uint32_t query_base = 0; query_base < query_count; query_base += kMemoryBatchStackQueries) {
    const uint32_t remaining = query_count - query_base;
    const uint32_t chunk =
        remaining < kMemoryBatchStackQueries ? remaining : kMemoryBatchStackQueries;
    memory_search_flat_batch(index, desc, queries + static_cast<size_t>(query_base) * index->dim,
                             chunk, out_results + static_cast<size_t>(query_base) * desc->top_k,
                             out_counts + query_base);
  }
}

void memory_search_flat_batch_range(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                    const float* queries, uint32_t query_count,
                                    uint32_t begin_active, uint32_t end_active,
                                    AstralMemorySearchResult* out_results, uint32_t* out_counts) {
  float query_scales[kMemoryBatchStackQueries];
  for (uint32_t query_i = 0; query_i < query_count; ++query_i) {
    const float* query = queries + static_cast<size_t>(query_i) * index->dim;
    query_scales[query_i] =
        index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(query, index->dim) : 1.0f;
    out_counts[query_i] = kNoResults;
  }

  for (uint32_t active_pos = begin_active; active_pos < end_active; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    for (uint32_t query_i = 0; query_i < query_count; ++query_i) {
      AstralMemorySearchResult* results = out_results + static_cast<size_t>(query_i) * desc->top_k;
      uint32_t* filled = out_counts + query_i;
      const float* query = queries + static_cast<size_t>(query_i) * index->dim;
      const float score = score_slot(index, query, slot, query_scales[query_i]);
      if (*filled == desc->top_k &&
          !result_better_values(score, s.record.key, results[desc->top_k - 1u])) {
        continue;
      }
      AstralMemorySearchResult candidate{};
      fill_result(&candidate, s, score);
      insert_result(results, desc->top_k, filled, candidate);
    }
  }
}

void memory_search_record_shard_work(void* user) {
  MemorySearchRecordShardJob* job = static_cast<MemorySearchRecordShardJob*>(user);
  memory_search_flat_batch_range(job->index, job->desc, job->queries, job->query_count, job->begin,
                                 job->end, job->local_results, job->local_counts);
  memory_parallel_job_complete(job->remaining);
}

void memory_search_flat_single_range(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                     const float* query, uint32_t begin_active, uint32_t end_active,
                                     AstralMemorySearchResult* out_results, uint32_t* out_count) {
  if (index->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2 && index->dense_active != 0 &&
      desc->group_id == ASTRAL_MEMORY_GROUP_ANY && index->metric != ASTRAL_MEMORY_METRIC_L2) {
    uint32_t filled = 0;
    const uint32_t dim = index->dim;
    const int8_t* vectors = index->q8_vectors;
    const float* score_scales = index->compact_score_scales;
    const float query_scale =
        index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(query, dim) : 1.0f;
    const DotE5m2F32Fn dot_kernel = index->e5m2_kernels->dot_f32;
    const bool use_384_kernel =
        dim == 384u && platform::cpu_supports_avx2() && platform::cpu_supports_f16c();

    // Dense slots make the vector address and score scale direct functions of the loop index.
    // Storage and metric are fixed for the shard, so neither needs redispatch for every record.
    for (uint32_t slot = begin_active; slot < end_active; ++slot) {
      const MemorySlot& s = index->slots[slot];
      const int8_t* vector = vectors + static_cast<size_t>(slot) * dim;
      const float dot =
          use_384_kernel ? dot_e5m2_f32_384(vector, query) : dot_kernel(vector, query, dim);
      const float score = dot * score_scales[slot] * query_scale;
      if (filled == desc->top_k &&
          !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
        continue;
      }
      AstralMemorySearchResult candidate{};
      fill_result(&candidate, s, score);
      insert_result(out_results, desc->top_k, &filled, candidate);
    }
    *out_count = filled;
    return;
  }

  memory_search_flat_batch_range(index, desc, query, 1u, begin_active, end_active, out_results,
                                 out_count);
}

void memory_search_single_record_shard_work(void* user) {
  MemorySearchSingleRecordShardJob* job = static_cast<MemorySearchSingleRecordShardJob*>(user);
  memory_search_flat_single_range(job->index, job->desc, job->query, job->begin, job->end,
                                  job->local_results, &job->local_count);
  job->remaining->fetch_sub(1, std::memory_order_release);
  astral::platform::cpu_signal_event();
}

bool memory_search_flat_record_parallel(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                        const float* query, AstralMemorySearchResult* out_results,
                                        uint32_t* out_count) {
  if (index->count <= kMemorySearchBatchParallelMaxRecords ||
      desc->top_k > kMemorySearchRecordParallelMaxTopK || !core::runtime_initialized()) {
    return false;
  }

  const uint32_t runtime_threads = core::runtime_thread_count();
  const bool on_worker = core::runtime_on_worker_thread();
  if (runtime_threads == 0u) {
    return false;
  }

  uint32_t remote_count = on_worker ? runtime_threads - 1u : runtime_threads;
  if (remote_count >= kMemorySearchBatchParallelMaxWorkers) {
    remote_count = kMemorySearchBatchParallelMaxWorkers - 1u;
  }
  if (remote_count == 0u) {
    return false;
  }
  const uint32_t shard_count = remote_count + 1u;

  // The single-query path carries only ten results per shard. Reusing the batch job here would
  // reserve its full query matrix on the caller's stack even though every shard scans one query.
  MemorySearchSingleRecordShardJob jobs[kMemorySearchBatchParallelMaxWorkers]{};
  std::atomic<uint32_t> remaining(shard_count);
  const uint32_t current_worker = on_worker ? core::runtime_worker_id() : kU32Max;
  for (uint32_t shard_i = 0; shard_i < shard_count; ++shard_i) {
    jobs[shard_i].index = index;
    jobs[shard_i].desc = desc;
    jobs[shard_i].query = query;
    jobs[shard_i].begin =
        static_cast<uint32_t>((static_cast<uint64_t>(index->count) * shard_i) / shard_count);
    jobs[shard_i].end =
        static_cast<uint32_t>((static_cast<uint64_t>(index->count) * (shard_i + 1u)) / shard_count);
    jobs[shard_i].remaining = &remaining;
  }

  for (uint32_t worker_i = 0; worker_i < remote_count; ++worker_i) {
    uint32_t target_worker = worker_i;
    if (on_worker && target_worker >= current_worker) {
      ++target_worker;
    }
    const AstralErr err = core::submit_work_affine(
        target_worker, memory_search_single_record_shard_work, &jobs[worker_i]);
    if (err != ASTRAL_OK) {
      memory_search_single_record_shard_work(&jobs[worker_i]);
    }
  }

  // The caller owns the final shard instead of parking while remote workers consume the scan.
  memory_search_single_record_shard_work(&jobs[remote_count]);
  while (remaining.load(std::memory_order_acquire) != 0) {
    astral::platform::cpu_wait_for_event();
  }

  uint32_t filled = 0;
  for (uint32_t shard_i = 0; shard_i < shard_count; ++shard_i) {
    for (uint32_t result_i = 0; result_i < jobs[shard_i].local_count; ++result_i) {
      insert_result(out_results, desc->top_k, &filled, jobs[shard_i].local_results[result_i]);
    }
  }
  *out_count = filled;
  return true;
}

bool memory_search_flat_batch_record_parallel(MemoryIndex* index,
                                              const AstralMemorySearchDesc* desc,
                                              const float* queries, uint32_t query_count,
                                              AstralMemorySearchResult* out_results,
                                              uint32_t* out_counts) {
  if (compact_storage(index) || index->count <= kMemorySearchBatchParallelMaxRecords ||
      query_count > kMemoryBatchStackQueries || desc->top_k > kMemorySearchBatchParallelMaxTopK ||
      !core::runtime_initialized()) {
    return false;
  }

  const uint32_t runtime_threads = core::runtime_thread_count();
  const bool on_worker = core::runtime_on_worker_thread();
  if (runtime_threads < 2u || index->count < runtime_threads) {
    return false;
  }

  uint32_t worker_count = on_worker ? runtime_threads - 1u : runtime_threads;
  if (worker_count > kMemorySearchBatchParallelMaxWorkers) {
    worker_count = kMemorySearchBatchParallelMaxWorkers;
  }
  if (worker_count < 2u) {
    return false;
  }
  if (worker_count > index->count) {
    worker_count = index->count;
  }

  // Each shard owns its top-k during the scan. Sharing one heap would turn every competitive
  // score into cross-core cache traffic; only these bounded shard results are merged below.
  MemorySearchRecordShardJob jobs[kMemorySearchBatchParallelMaxWorkers]{};
  std::atomic<uint32_t> remaining(worker_count);
  const uint32_t current_worker = on_worker ? core::runtime_worker_id() : kU32Max;
  for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
    const uint32_t begin =
        static_cast<uint32_t>((static_cast<uint64_t>(index->count) * worker_i) / worker_count);
    const uint32_t end = static_cast<uint32_t>(
        (static_cast<uint64_t>(index->count) * (worker_i + 1u)) / worker_count);
    jobs[worker_i].index = index;
    jobs[worker_i].desc = desc;
    jobs[worker_i].queries = queries;
    jobs[worker_i].query_count = query_count;
    jobs[worker_i].begin = begin;
    jobs[worker_i].end = end;
    jobs[worker_i].remaining = &remaining;

    uint32_t target_worker = worker_i;
    if (on_worker && target_worker >= current_worker) {
      ++target_worker;
    }
    const AstralErr err =
        core::submit_work_affine(target_worker, memory_search_record_shard_work, &jobs[worker_i]);
    if (err != ASTRAL_OK) {
      memory_search_record_shard_work(&jobs[worker_i]);
    }
  }

  while (remaining.load(std::memory_order_acquire) != 0) {
    astral::platform::cpu_wait_for_event();
  }

  for (uint32_t query_i = 0; query_i < query_count; ++query_i) {
    AstralMemorySearchResult* results = out_results + static_cast<size_t>(query_i) * desc->top_k;
    uint32_t filled = 0;
    for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
      const AstralMemorySearchResult* local =
          jobs[worker_i].local_results + static_cast<size_t>(query_i) * desc->top_k;
      const uint32_t local_count = jobs[worker_i].local_counts[query_i];
      for (uint32_t result_i = 0; result_i < local_count; ++result_i) {
        insert_result(results, desc->top_k, &filled, local[result_i]);
      }
    }
    out_counts[query_i] = filled;
  }
  return true;
}

void memory_search_flat_batch_work(void* user) {
  MemorySearchBatchJob* job = static_cast<MemorySearchBatchJob*>(user);
  const uint32_t count = job->end - job->begin;
  memory_search_flat_batch_chunked(
      job->index, job->desc, job->queries + static_cast<size_t>(job->begin) * job->index->dim,
      count, job->out_results + static_cast<size_t>(job->begin) * job->desc->top_k,
      job->out_counts + job->begin);
  memory_parallel_job_complete(job->remaining);
}

bool memory_search_flat_batch_parallel(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                       const float* queries, uint32_t query_count,
                                       AstralMemorySearchResult* out_results,
                                       uint32_t* out_counts) {
  if (query_count < kMemorySearchBatchParallelMinQueries ||
      index->count > kMemorySearchBatchParallelMaxRecords || !core::runtime_initialized()) {
    return false;
  }

  const uint32_t runtime_threads = core::runtime_thread_count();
  const bool on_worker = core::runtime_on_worker_thread();
  if (runtime_threads < 2u) {
    return false;
  }

  uint32_t worker_count = on_worker ? runtime_threads - 1u : runtime_threads;
  if (worker_count > kMemorySearchBatchParallelMaxWorkers) {
    worker_count = kMemorySearchBatchParallelMaxWorkers;
  }
  if (worker_count < 2u) {
    return false;
  }
  if (worker_count > query_count) {
    worker_count = query_count;
  }

  MemorySearchBatchJob jobs[kMemorySearchBatchParallelMaxWorkers]{};
  std::atomic<uint32_t> remaining(worker_count);
  const uint32_t current_worker = on_worker ? core::runtime_worker_id() : kU32Max;
  for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
    const uint32_t begin =
        static_cast<uint32_t>((static_cast<uint64_t>(query_count) * worker_i) / worker_count);
    const uint32_t end = static_cast<uint32_t>(
        (static_cast<uint64_t>(query_count) * (worker_i + 1u)) / worker_count);
    jobs[worker_i].index = index;
    jobs[worker_i].desc = desc;
    jobs[worker_i].queries = queries;
    jobs[worker_i].out_results = out_results;
    jobs[worker_i].out_counts = out_counts;
    jobs[worker_i].begin = begin;
    jobs[worker_i].end = end;
    jobs[worker_i].remaining = &remaining;

    uint32_t target_worker = worker_i;
    if (on_worker && target_worker >= current_worker) {
      ++target_worker;
    }
    const AstralErr err =
        core::submit_work_affine(target_worker, memory_search_flat_batch_work, &jobs[worker_i]);
    if (err != ASTRAL_OK) {
      memory_search_flat_batch_work(&jobs[worker_i]);
    }
  }

  while (remaining.load(std::memory_order_acquire) != 0) {
    astral::platform::cpu_wait_for_event();
  }
  return true;
}

void destroy_allocations(MemoryIndex* index) {
  if (index->slots != nullptr) {
    core::runtime_free_array(index->slots, index->capacity);
    index->slots = nullptr;
  }
  if (index->vectors != nullptr) {
    free_vector_storage(index->vectors, index->capacity, index->dim);
    index->vectors = nullptr;
  }
  if (index->q8_vectors != nullptr) {
    free_q8_vector_storage(index->q8_vectors, index->capacity, index->dim);
    index->q8_vectors = nullptr;
  }
  if (index->i16_vectors != nullptr) {
    free_i16_vector_storage(index->i16_vectors, index->capacity, index->dim);
    index->i16_vectors = nullptr;
  }
  if (index->q8_scales != nullptr) {
    core::runtime_free_array(index->q8_scales, index->capacity);
    index->q8_scales = nullptr;
  }
  if (index->compact_score_scales != nullptr) {
    core::runtime_free_array(index->compact_score_scales, index->capacity);
    index->compact_score_scales = nullptr;
  }
  if (index->compact_vector_sums != nullptr) {
    core::runtime_free_array(index->compact_vector_sums, index->capacity);
    index->compact_vector_sums = nullptr;
  }
  if (index->active_slots != nullptr) {
    core::runtime_free_array(index->active_slots, index->capacity);
    index->active_slots = nullptr;
  }
  if (index->key_table != nullptr) {
    core::runtime_free_array(index->key_table, index->key_table_capacity);
    index->key_table = nullptr;
  }
  if (index->graph_neighbors != nullptr) {
    core::runtime_free_array(index->graph_neighbors,
                             static_cast<uint32_t>(graph_neighbor_storage_count(index)));
    index->graph_neighbors = nullptr;
  }
  if (index->graph_neighbor_counts != nullptr) {
    core::runtime_free_array(index->graph_neighbor_counts,
                             index->capacity * index->graph_level_capacity);
    index->graph_neighbor_counts = nullptr;
  }
  if (index->graph_levels != nullptr) {
    core::runtime_free_array(index->graph_levels, index->capacity);
    index->graph_levels = nullptr;
  }
  if (index->graph_candidates != nullptr) {
    core::runtime_free_array(index->graph_candidates, index->graph_candidate_capacity);
    index->graph_candidates = nullptr;
  }
  if (index->graph_candidate_scores != nullptr) {
    core::runtime_free_array(index->graph_candidate_scores, index->graph_candidate_capacity);
    index->graph_candidate_scores = nullptr;
  }
  if (index->graph_scratch_slots != nullptr) {
    core::runtime_free_array(index->graph_scratch_slots, index->graph_scratch_capacity);
    index->graph_scratch_slots = nullptr;
  }
  if (index->graph_scratch_scores != nullptr) {
    core::runtime_free_array(index->graph_scratch_scores, index->graph_scratch_capacity);
    index->graph_scratch_scores = nullptr;
  }
  if (index->graph_visited != nullptr) {
    core::runtime_free_array(index->graph_visited, index->capacity);
    index->graph_visited = nullptr;
  }
  for (uint32_t i = 0; i < index->graph_batch_scratch_count; ++i) {
    graph_search_scratch_free(index, &index->graph_batch_scratch[i]);
  }
  index->graph_batch_scratch_count = 0;
}

void destroy_search_cursor(MemorySearchCursor* cursor) {
  if (cursor == nullptr) {
    return;
  }
  if (cursor->results != nullptr && cursor->results != cursor->inline_results) {
    core::runtime_free_array(cursor->results, cursor->capacity);
    cursor->results = nullptr;
  }
  core::runtime_delete(cursor);
}

bool memory_add_preprocess_range(MemoryIndex* index, const float* vectors, const uint32_t* slots,
                                 uint32_t begin, uint32_t end, bool validate_vectors) {
  for (uint32_t i = begin; i < end; ++i) {
    const uint32_t slot = slots[i];
    const float* src = vectors + static_cast<size_t>(i) * index->dim;
    if (validate_vectors && !f32_values_finite(src, index->dim)) {
      return false;
    }
    if (q8_storage(index)) {
      if (f32_rerank_storage(index)) {
        store_f32_vector(index, slot, src);
      }
      quantize_q8_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src, index->dim);
      index->compact_vector_sums[slot] = sum_i8(q8_vector_at(index, slot), index->dim);
    } else if (e2m3_storage(index)) {
      if (f32_rerank_storage(index)) {
        store_f32_vector(index, slot, src);
      }
      if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        quantize_e2m3_cosine_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src,
                                    index->dim);
      } else {
        quantize_e2m3_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src, index->dim);
      }
      index->compact_vector_sums[slot] = sum_i8(q8_vector_at(index, slot), index->dim);
    } else if (e3m2_storage(index)) {
      if (f32_rerank_storage(index)) {
        store_f32_vector(index, slot, src);
      }
      if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        quantize_e3m2_cosine_vector(i16_vector_at(index, slot), &index->q8_scales[slot], src,
                                    index->dim);
      } else {
        quantize_e3m2_vector(i16_vector_at(index, slot), &index->q8_scales[slot], src, index->dim);
      }
    } else if (e5m2_storage(index)) {
      if (f8_f32_rerank_storage(index)) {
        store_f32_vector(index, slot, src);
      }
      if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        quantize_e5m2_cosine_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src,
                                    index->dim);
      } else {
        quantize_e5m2_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src, index->dim);
      }
      index->compact_vector_sums[slot] = sum_i8(q8_vector_at(index, slot), index->dim);
    } else {
      store_f32_vector(index, slot, src);
    }
    if (compact_storage(index)) {
      index->slots[slot].score_scale =
          index->metric == ASTRAL_MEMORY_METRIC_COSINE
              ? (q8_storage(index) ? cosine_scale(src, index->dim) : 1.0f)
              : 0.0f;
      update_compact_score_scale(index, slot);
    }
  }
  return true;
}

void memory_add_preprocess_work(void* user) {
  MemoryAddPreprocessJob* job = static_cast<MemoryAddPreprocessJob*>(user);
  if (!memory_add_preprocess_range(job->index, job->vectors, job->slots, job->begin, job->end,
                                   job->validate_vectors != 0)) {
    job->invalid->store(1u, std::memory_order_release);
  }
  memory_parallel_job_complete(job->remaining);
}

bool memory_add_preprocess_parallel(MemoryIndex* index, const float* vectors, const uint32_t* slots,
                                    uint32_t count, bool validate_vectors, bool* out_valid) {
  if (count < kMemoryAddParallelMinCount || !core::runtime_initialized()) {
    return false;
  }

  const uint32_t runtime_threads = core::runtime_thread_count();
  const bool on_worker = core::runtime_on_worker_thread();
  if (runtime_threads < 2u) {
    return false;
  }

  uint32_t worker_count = on_worker ? runtime_threads - 1u : runtime_threads;
  if (worker_count > kMemoryAddParallelMaxWorkers) {
    worker_count = kMemoryAddParallelMaxWorkers;
  }
  if (worker_count < 2u) {
    return false;
  }
  if (worker_count > count) {
    worker_count = count;
  }

  MemoryAddPreprocessJob jobs[kMemoryAddParallelMaxWorkers]{};
  std::atomic<uint32_t> remaining(worker_count);
  std::atomic<uint32_t> invalid(0);
  const uint32_t chunk = (count + worker_count - 1u) / worker_count;
  const uint32_t current_worker = on_worker ? core::runtime_worker_id() : kU32Max;
  for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
    const uint32_t begin = worker_i * chunk;
    uint32_t end = begin + chunk;
    if (end > count) {
      end = count;
    }
    jobs[worker_i].index = index;
    jobs[worker_i].vectors = vectors;
    jobs[worker_i].slots = slots;
    jobs[worker_i].begin = begin;
    jobs[worker_i].end = end;
    jobs[worker_i].validate_vectors = validate_vectors ? 1u : 0u;
    jobs[worker_i].remaining = &remaining;
    jobs[worker_i].invalid = &invalid;

    uint32_t target_worker = worker_i;
    if (on_worker && target_worker >= current_worker) {
      ++target_worker;
    }
    const AstralErr err =
        core::submit_work_affine(target_worker, memory_add_preprocess_work, &jobs[worker_i]);
    if (err != ASTRAL_OK) {
      memory_add_preprocess_work(&jobs[worker_i]);
    }
  }

  while (remaining.load(std::memory_order_acquire) != 0) {
    astral::platform::cpu_wait_for_event();
  }
  *out_valid = invalid.load(std::memory_order_acquire) == 0;
  return true;
}

void memory_add_rollback_new_slots(MemoryIndex* index, uint32_t initial_count,
                                   uint32_t staged_count, uint32_t initial_free_slot_hint,
                                   uint8_t initial_dense_active) {
  for (uint32_t active_pos = initial_count; active_pos < staged_count; ++active_pos) {
    const uint32_t slot = index->active_slots[active_pos];
    index->active_slots[active_pos] = 0;
    index->slots[slot] = MemorySlot{};
    if (graph_enabled(index)) {
      index->graph_levels[slot] = 0;
    }
  }
  index->free_slot_hint = initial_free_slot_hint;
  index->dense_active = initial_dense_active;
  key_table_rebuild(index);
}

} // namespace

void memory_search_flat_fallback(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                 const float* query,
                                 AstralMemorySearchResult* out_results,
                                 uint32_t* out_count) {
  memory_search_flat(index, desc, query, out_results, out_count);
}

AstralErr memory_create(const AstralMemoryIndexDesc* desc, MemoryIndex** out_index) {
  if (!desc_valid(desc) || out_index == nullptr) {
    return ASTRAL_E_INVALID;
  }
  if (desc->capacity > kU32Max / desc->dim || desc->capacity > kU32Max / kKeyTableLoadFactorDen) {
    return ASTRAL_E_NOMEM;
  }
  const uint32_t key_table_capacity = key_table_capacity_for(desc->capacity);
  const uint32_t graph_neighbor_capacity = graph_neighbors_from_desc(desc);
  const uint32_t graph_base_neighbor_capacity =
      graph_neighbor_capacity != 0
          ? graph_base_neighbors_from_graph_neighbors(graph_neighbor_capacity)
          : 0;
  const uint32_t graph_level_capacity =
      graph_level_capacity_for(desc->capacity, graph_neighbor_capacity);
  const uint32_t requested_graph_search = graph_search_from_desc(desc);
  const uint32_t graph_search_capacity =
      requested_graph_search > desc->capacity ? desc->capacity : requested_graph_search;
  const uint32_t requested_query_search = graph_query_search_from_desc(desc);
  const uint32_t graph_query_search_capacity =
      requested_query_search > desc->capacity ? desc->capacity : requested_query_search;
  uint32_t requested_candidate_capacity = graph_query_search_capacity > graph_search_capacity
                                              ? graph_query_search_capacity
                                              : graph_search_capacity;
  if (requested_candidate_capacity <= kU32Max / kGraphCandidateReserveMultiplier) {
    const uint32_t reserve_candidate_capacity =
        requested_candidate_capacity * kGraphCandidateReserveMultiplier;
    if (reserve_candidate_capacity > requested_candidate_capacity) {
      requested_candidate_capacity = reserve_candidate_capacity;
    }
  }
  const uint32_t graph_candidate_capacity =
      requested_candidate_capacity > desc->capacity ? desc->capacity : requested_candidate_capacity;
  uint32_t graph_scratch_capacity = graph_candidate_capacity > graph_neighbor_capacity
                                        ? graph_candidate_capacity
                                        : graph_neighbor_capacity;
  const uint32_t reverse_refine_capacity = graph_base_neighbor_capacity + 1u;
  if (reverse_refine_capacity > graph_scratch_capacity) {
    graph_scratch_capacity = reverse_refine_capacity;
  }

  MemoryIndex* index = core::runtime_new<MemoryIndex>();
  if (index == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  index->handle = 0;
  index->e5m2_kernels = select_e5m2_kernels();
  index->dim = desc->dim;
  index->capacity = desc->capacity;
  index->count = 0;
  index->metric = desc->metric;
  index->index_kind = desc->index_kind;
  index->storage_kind = desc->storage_kind;
  index->slots = core::runtime_alloc_array<MemorySlot>(desc->capacity);
  index->vectors = (desc->storage_kind == ASTRAL_MEMORY_STORAGE_F32 ||
                    f32_rerank_storage_kind(desc->storage_kind))
                       ? alloc_vector_storage(desc->capacity, desc->dim)
                       : nullptr;
  index->q8_vectors =
      compact_storage_kind(desc->storage_kind) && !i16_storage_kind(desc->storage_kind)
          ? alloc_q8_vector_storage(desc->capacity, desc->dim)
          : nullptr;
  index->i16_vectors = i16_storage_kind(desc->storage_kind)
                           ? alloc_i16_vector_storage(desc->capacity, desc->dim)
                           : nullptr;
  index->q8_scales = compact_storage_kind(desc->storage_kind)
                         ? core::runtime_alloc_array<float>(desc->capacity)
                         : nullptr;
  index->compact_score_scales = compact_storage_kind(desc->storage_kind)
                                    ? core::runtime_alloc_array<float>(desc->capacity)
                                    : nullptr;
  index->compact_vector_sums =
      compact_storage_kind(desc->storage_kind) && !i16_storage_kind(desc->storage_kind)
          ? core::runtime_alloc_array<int32_t>(desc->capacity)
          : nullptr;
  index->active_slots = core::runtime_alloc_array<uint32_t>(desc->capacity);
  index->key_table = core::runtime_alloc_array<uint32_t>(key_table_capacity);
  index->graph_neighbors = nullptr;
  index->graph_neighbor_counts = nullptr;
  index->graph_levels = nullptr;
  index->graph_candidates = nullptr;
  index->graph_candidate_scores = nullptr;
  index->graph_scratch_slots = nullptr;
  index->graph_scratch_scores = nullptr;
  index->graph_visited = nullptr;
  for (uint32_t i = 0; i < kMemorySearchBatchParallelMaxWorkers; ++i) {
    index->graph_batch_scratch[i] = GraphSearchScratch{};
  }
  index->key_table_capacity = key_table_capacity;
  index->key_table_mask = key_table_capacity - 1u;
  index->graph_neighbor_capacity = graph_neighbor_capacity;
  index->graph_base_neighbor_capacity = graph_base_neighbor_capacity;
  index->graph_search_capacity = graph_search_capacity;
  index->graph_query_search_capacity = graph_query_search_capacity;
  index->graph_candidate_capacity = graph_candidate_capacity;
  index->graph_scratch_capacity = graph_scratch_capacity;
  index->graph_level_capacity = graph_level_capacity;
  index->graph_entry_slot = kU32Max;
  index->graph_max_level = 0;
  index->graph_visit_generation = 0;
  index->graph_candidate_worst_pos = 0;
  index->graph_candidate_worst_slot = kU32Max;
  index->graph_candidate_worst_score = kWorstScore;
  index->graph_candidate_worst_valid = 0;
  index->graph_build_score_evals = 0;
  index->graph_build_candidate_visits = 0;
  index->graph_batch_scratch_claimed.store(0u, std::memory_order_relaxed);
  index->graph_batch_scratch_count = 0;
  index->dense_active = 1u;
  index->i16_vectors_aligned =
      i16_storage_kind(desc->storage_kind) && i16_vector_stride_aligned(desc->dim) ? 1u : 0u;
  if (graph_neighbor_capacity != 0) {
    const uint64_t graph_neighbor_slots =
        static_cast<uint64_t>(desc->capacity) * graph_base_neighbor_capacity +
        static_cast<uint64_t>(desc->capacity) * (graph_level_capacity - 1u) *
            graph_neighbor_capacity;
    if (graph_neighbor_slots > kU32Max || desc->capacity > kU32Max / graph_base_neighbor_capacity ||
        desc->capacity > kU32Max / graph_neighbor_capacity ||
        graph_level_capacity > kU32Max / desc->capacity ||
        desc->capacity * (graph_level_capacity - 1u) > kU32Max / graph_neighbor_capacity) {
      destroy_allocations(index);
      core::runtime_delete(index);
      return ASTRAL_E_NOMEM;
    }
    index->graph_neighbors =
        core::runtime_alloc_array<uint32_t>(static_cast<uint32_t>(graph_neighbor_slots));
    index->graph_neighbor_counts =
        core::runtime_alloc_array<uint8_t>(desc->capacity * graph_level_capacity);
    index->graph_levels = core::runtime_alloc_array<uint8_t>(desc->capacity);
    index->graph_candidates = core::runtime_alloc_array<uint32_t>(graph_candidate_capacity);
    index->graph_candidate_scores = core::runtime_alloc_array<float>(graph_candidate_capacity);
    index->graph_scratch_slots = core::runtime_alloc_array<uint32_t>(graph_scratch_capacity);
    index->graph_scratch_scores = core::runtime_alloc_array<float>(graph_scratch_capacity);
    index->graph_visited = core::runtime_alloc_array<uint16_t>(desc->capacity);
    if (core::runtime_initialized()) {
      uint32_t scratch_count = core::runtime_thread_count();
      if (scratch_count > kMemorySearchBatchParallelMaxWorkers) {
        scratch_count = kMemorySearchBatchParallelMaxWorkers;
      }
      if (scratch_count > 1u) {
        uint32_t allocated = 0;
        for (; allocated < scratch_count; ++allocated) {
          if (!graph_search_scratch_alloc(index, &index->graph_batch_scratch[allocated])) {
            for (uint32_t i = 0; i < allocated; ++i) {
              graph_search_scratch_free(index, &index->graph_batch_scratch[i]);
            }
            allocated = 0;
            break;
          }
        }
        index->graph_batch_scratch_count = static_cast<uint8_t>(allocated);
      }
    }
  }
  if (index->slots == nullptr || index->active_slots == nullptr || index->key_table == nullptr ||
      ((desc->storage_kind == ASTRAL_MEMORY_STORAGE_F32 ||
        f32_rerank_storage_kind(desc->storage_kind)) &&
       index->vectors == nullptr) ||
      (compact_storage_kind(desc->storage_kind) &&
       (index->q8_scales == nullptr || index->compact_score_scales == nullptr)) ||
      (compact_storage_kind(desc->storage_kind) && !i16_storage_kind(desc->storage_kind) &&
       index->compact_vector_sums == nullptr) ||
      (compact_storage_kind(desc->storage_kind) && !i16_storage_kind(desc->storage_kind) &&
       index->q8_vectors == nullptr) ||
      (i16_storage_kind(desc->storage_kind) && index->i16_vectors == nullptr) ||
      (graph_neighbor_capacity != 0 &&
       (index->graph_neighbors == nullptr || index->graph_neighbor_counts == nullptr ||
        index->graph_levels == nullptr || index->graph_candidates == nullptr ||
        index->graph_candidate_scores == nullptr || index->graph_scratch_slots == nullptr ||
        index->graph_scratch_scores == nullptr || index->graph_visited == nullptr))) {
    destroy_allocations(index);
    core::runtime_delete(index);
    return ASTRAL_E_NOMEM;
  }

  std::memset(index->slots, 0, sizeof(MemorySlot) * desc->capacity);
  std::memset(index->active_slots, 0, sizeof(uint32_t) * desc->capacity);
  std::memset(index->key_table, 0, sizeof(uint32_t) * key_table_capacity);
  if (compact_storage_kind(desc->storage_kind)) {
    if (i16_storage_kind(desc->storage_kind)) {
      std::memset(index->i16_vectors, 0,
                  static_cast<size_t>(desc->capacity) * desc->dim * sizeof(int16_t));
    } else {
      std::memset(index->q8_vectors, 0,
                  static_cast<size_t>(desc->capacity) * desc->dim * sizeof(int8_t));
    }
    std::memset(index->q8_scales, 0, sizeof(float) * desc->capacity);
    std::memset(index->compact_score_scales, 0, sizeof(float) * desc->capacity);
    if (!i16_storage_kind(desc->storage_kind)) {
      std::memset(index->compact_vector_sums, 0, sizeof(int32_t) * desc->capacity);
    }
  }
  if (graph_neighbor_capacity != 0) {
    std::memset(index->graph_neighbors, 0, sizeof(uint32_t) * graph_neighbor_storage_count(index));
    std::memset(index->graph_neighbor_counts, 0,
                sizeof(uint8_t) * desc->capacity * graph_level_capacity);
    std::memset(index->graph_levels, 0, sizeof(uint8_t) * desc->capacity);
    std::memset(index->graph_candidates, 0, sizeof(uint32_t) * graph_candidate_capacity);
    std::memset(index->graph_candidate_scores, 0, sizeof(float) * graph_candidate_capacity);
    std::memset(index->graph_scratch_slots, 0, sizeof(uint32_t) * graph_scratch_capacity);
    std::memset(index->graph_scratch_scores, 0, sizeof(float) * graph_scratch_capacity);
    std::memset(index->graph_visited, 0, sizeof(uint16_t) * desc->capacity);
  }
  const AstralHandle handle = core::register_handle(core::HandleKind::MemoryIndex, index);
  if (handle == 0) {
    destroy_allocations(index);
    core::runtime_delete(index);
    return ASTRAL_E_BUSY;
  }

  index->handle = handle;
  *out_index = index;
  return ASTRAL_OK;
}

void memory_destroy(MemoryIndex* index) {
  if (index == nullptr) {
    return;
  }
  core::unregister_handle(index->handle, core::HandleKind::MemoryIndex);
  destroy_allocations(index);
  core::runtime_delete(index);
}

AstralErr memory_count(MemoryIndex* index, uint32_t* out_count) {
  if (index == nullptr || out_count == nullptr) {
    return ASTRAL_E_INVALID;
  }
  *out_count = index->count;
  return ASTRAL_OK;
}

AstralErr memory_stats(MemoryIndex* index, AstralMemoryStats* out_stats) {
  if (index == nullptr || out_stats == nullptr || out_stats->size != sizeof(AstralMemoryStats)) {
    return ASTRAL_E_INVALID;
  }

  const uint64_t vector_bytes =
      compact_storage(index)
          ? static_cast<uint64_t>(index->capacity) * index->dim *
                    (i16_storage(index) ? sizeof(int16_t) : sizeof(int8_t)) +
                static_cast<uint64_t>(index->capacity) * sizeof(float) +
                (f32_rerank_storage(index)
                     ? static_cast<uint64_t>(index->capacity) * index->dim * sizeof(float)
                     : 0ull)
          : static_cast<uint64_t>(index->capacity) * index->dim * sizeof(float);
  const uint64_t metadata_bytes =
      sizeof(MemoryIndex) + static_cast<uint64_t>(index->capacity) * sizeof(MemorySlot) +
      static_cast<uint64_t>(index->capacity) * sizeof(uint32_t) +
      static_cast<uint64_t>(index->key_table_capacity) * sizeof(uint32_t);
  uint64_t graph_bytes = 0;
  uint64_t graph_base_edges = 0;
  uint64_t graph_upper_edges = 0;
  if (graph_enabled(index)) {
    graph_bytes += static_cast<uint64_t>(graph_neighbor_storage_count(index)) * sizeof(uint32_t);
    graph_bytes +=
        static_cast<uint64_t>(index->capacity) * index->graph_level_capacity * sizeof(uint8_t);
    graph_bytes += static_cast<uint64_t>(index->capacity) * sizeof(uint8_t);
    graph_bytes += static_cast<uint64_t>(index->graph_candidate_capacity) * sizeof(uint32_t);
    graph_bytes += static_cast<uint64_t>(index->graph_candidate_capacity) * sizeof(float);
    graph_bytes += static_cast<uint64_t>(index->graph_scratch_capacity) * sizeof(uint32_t);
    graph_bytes += static_cast<uint64_t>(index->graph_scratch_capacity) * sizeof(float);
    graph_bytes += static_cast<uint64_t>(index->capacity) * sizeof(uint16_t);
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      graph_base_edges += graph_neighbor_count_at_level(index, slot, 0);
      for (uint32_t level = 1; level < index->graph_level_capacity; ++level) {
        graph_upper_edges += graph_neighbor_count_at_level(index, slot, level);
      }
    }
  }

  out_stats->dim = index->dim;
  out_stats->capacity = index->capacity;
  out_stats->count = index->count;
  out_stats->metric = index->metric;
  out_stats->index_kind = index->index_kind;
  out_stats->graph_neighbors = index->graph_neighbor_capacity;
  out_stats->graph_search = index->graph_search_capacity;
  out_stats->graph_query_search = index->graph_query_search_capacity;
  out_stats->graph_levels = index->graph_level_capacity;
  out_stats->storage_kind = index->storage_kind;
  out_stats->vector_bytes = vector_bytes;
  out_stats->metadata_bytes = metadata_bytes;
  out_stats->graph_bytes = graph_bytes;
  out_stats->graph_edges = graph_base_edges + graph_upper_edges;
  out_stats->graph_base_edges = graph_base_edges;
  out_stats->graph_upper_edges = graph_upper_edges;
  out_stats->graph_build_score_evals = index->graph_build_score_evals;
  out_stats->graph_build_candidate_visits = index->graph_build_candidate_visits;
  out_stats->total_bytes = vector_bytes + metadata_bytes + graph_bytes;
  out_stats->save_bytes = memory_save_byte_count(index);
  return ASTRAL_OK;
}

AstralErr memory_clear(MemoryIndex* index) {
  if (index == nullptr) {
    return ASTRAL_E_INVALID;
  }
  std::memset(index->slots, 0, sizeof(MemorySlot) * index->capacity);
  std::memset(index->active_slots, 0, sizeof(uint32_t) * index->capacity);
  std::memset(index->key_table, 0, sizeof(uint32_t) * index->key_table_capacity);
  if (graph_enabled(index)) {
    std::memset(index->graph_neighbor_counts, 0,
                sizeof(uint8_t) * index->capacity * index->graph_level_capacity);
    std::memset(index->graph_levels, 0, sizeof(uint8_t) * index->capacity);
    std::memset(index->graph_visited, 0, sizeof(uint16_t) * index->capacity);
    index->graph_entry_slot = kU32Max;
    index->graph_max_level = 0;
    index->graph_visit_generation = 0;
    index->graph_build_score_evals = 0;
    index->graph_build_candidate_visits = 0;
  }
  index->count = 0;
  index->free_slot_hint = 0;
  index->dense_active = 1u;
  return ASTRAL_OK;
}

AstralErr memory_get_record(MemoryIndex* index, uint64_t key, AstralMemoryRecord* out_record) {
  if (index == nullptr || key == 0 || out_record == nullptr) {
    return ASTRAL_E_INVALID;
  }

  const uint32_t slot = find_slot_by_key(index, key);
  if (slot == kU32Max) {
    return ASTRAL_E_NOT_FOUND;
  }

  *out_record = index->slots[slot].record;
  out_record->size = sizeof(AstralMemoryRecord);
  return ASTRAL_OK;
}

AstralErr memory_update_record(MemoryIndex* index, uint64_t key, const AstralMemoryRecord* record) {
  if (index == nullptr || key == 0 || record == nullptr ||
      record->size != sizeof(AstralMemoryRecord) || record->key == 0) {
    return ASTRAL_E_INVALID;
  }

  const uint32_t slot = find_slot_by_key(index, key);
  if (slot == kU32Max) {
    return ASTRAL_E_NOT_FOUND;
  }

  const bool key_changed = record->key != key;
  if (key_changed) {
    const uint64_t new_hash = key_hash_mix(record->key);
    if (find_slot_by_key_hashed(index, record->key, new_hash) != kU32Max) {
      return ASTRAL_E_STATE;
    }
    key_table_remove(index, key);
    const AstralErr err = key_table_insert_new_hashed(index, new_hash, slot);
    if (err != ASTRAL_OK) {
      const AstralErr restore_err = key_table_insert_new_hashed(index, key_hash_mix(key), slot);
      (void)restore_err;
      return err;
    }
    if (graph_enabled(index)) {
      index->graph_levels[slot] = static_cast<uint8_t>(graph_level_for_key(index, record->key));
    }
  }

  index->slots[slot].record = *record;
  if (key_changed && graph_enabled(index)) {
    graph_rebuild(index);
  }
  return ASTRAL_OK;
}

AstralErr memory_add_batch(MemoryIndex* index, const AstralMemoryRecord* records,
                           const float* vectors, uint32_t count) {
  if (index == nullptr || records == nullptr || vectors == nullptr || count == 0) {
    return ASTRAL_E_INVALID;
  }

  if (count > kU32Max / index->dim) {
    return ASTRAL_E_NOMEM;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (records[i].size != sizeof(AstralMemoryRecord) || records[i].key == 0) {
      return ASTRAL_E_INVALID;
    }
  }
  MemoryAddBatchScratch scratch;
  if (!scratch.init(count)) {
    return ASTRAL_E_NOMEM;
  }

  const uint32_t initial_count = index->count;
  uint32_t staged_count = initial_count;
  const uint32_t initial_free_slot_hint = index->free_slot_hint;
  const uint8_t initial_dense_active = index->dense_active;
  bool has_duplicate_slots = false;
  bool has_existing_updates = false;
  bool seen_existing_initialized = false;
  bool graph_rebuild_needed = false;
  bool graph_has_new_slots = false;

  MemoryAddSeenScratch seen_existing;

  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t key_hash = key_hash_mix(records[i].key);
    uint32_t slot = find_slot_by_key_hashed(index, records[i].key, key_hash);
    const bool is_update = slot != kU32Max;
    if (slot == kU32Max) {
      slot = find_free_slot(index);
      if (slot == kU32Max) {
        memory_add_rollback_new_slots(index, initial_count, staged_count, initial_free_slot_hint,
                                      initial_dense_active);
        return ASTRAL_E_NOMEM;
      }

      if (index->dense_active != 0 && slot != staged_count) {
        index->dense_active = 0;
      }
      index->slots[slot].record = records[i];
      index->slots[slot].active_pos = staged_count;
      index->active_slots[staged_count] = slot;
      index->slots[slot].occupied = 1;
      ++staged_count;
      index->free_slot_hint = slot + kSlotAdvance;
      if (index->free_slot_hint == index->capacity) {
        index->free_slot_hint = 0;
      }
      const AstralErr key_err = key_table_insert_new_hashed(index, key_hash, slot);
      if (key_err != ASTRAL_OK) {
        memory_add_rollback_new_slots(index, initial_count, staged_count, initial_free_slot_hint,
                                      initial_dense_active);
        return key_err;
      }
      if (graph_enabled(index)) {
        index->graph_levels[slot] =
            static_cast<uint8_t>(graph_level_for_key(index, records[i].key));
        graph_has_new_slots = true;
      }
    } else {
      if (index->slots[slot].active_pos >= initial_count) {
        has_duplicate_slots = true;
      } else {
        has_existing_updates = true;
        if (!seen_existing_initialized) {
          if (!seen_existing.init(count)) {
            memory_add_rollback_new_slots(index, initial_count, staged_count,
                                          initial_free_slot_hint, initial_dense_active);
            return ASTRAL_E_NOMEM;
          }
          seen_existing_initialized = true;
        }
        if (!seen_existing.mark(slot)) {
          has_duplicate_slots = true;
        }
      }
      if (graph_enabled(index) && is_update) {
        graph_rebuild_needed = true;
      }
    }
    scratch.slots[i] = slot;
  }

  if (has_existing_updates &&
      !f32_values_finite(vectors, static_cast<size_t>(count) * index->dim)) {
    memory_add_rollback_new_slots(index, initial_count, staged_count, initial_free_slot_hint,
                                  initial_dense_active);
    return ASTRAL_E_INVALID;
  }

  for (uint32_t i = 0; i < count; ++i) {
    index->slots[scratch.slots[i]].record = records[i];
  }

  bool vectors_valid = true;
  const bool validate_during_preprocess = !has_existing_updates;
  if (has_duplicate_slots ||
      !memory_add_preprocess_parallel(index, vectors, scratch.slots, count,
                                      validate_during_preprocess, &vectors_valid)) {
    vectors_valid = memory_add_preprocess_range(index, vectors, scratch.slots, 0, count,
                                                validate_during_preprocess);
  }
  if (!vectors_valid) {
    memory_add_rollback_new_slots(index, initial_count, staged_count, initial_free_slot_hint,
                                  initial_dense_active);
    return ASTRAL_E_INVALID;
  }

  index->count = staged_count;
  if (graph_rebuild_needed) {
    graph_rebuild(index);
  } else if (graph_has_new_slots) {
    const uint32_t final_count = staged_count;
    index->count = initial_count;
    for (uint32_t active_pos = initial_count; active_pos < final_count; ++active_pos) {
      ++index->count;
      graph_connect_slot(index, index->active_slots[active_pos]);
    }
    index->count = final_count;
  }
  return ASTRAL_OK;
}

AstralErr memory_remove(MemoryIndex* index, uint64_t key) {
  if (index == nullptr || key == 0) {
    return ASTRAL_E_INVALID;
  }
  const uint32_t slot = find_slot_by_key(index, key);
  if (slot == kU32Max) {
    return ASTRAL_E_NOT_FOUND;
  }
  const uint32_t active_pos = index->slots[slot].active_pos;
  const uint32_t last_pos = index->count - 1u;
  const uint32_t last_slot = index->active_slots[last_pos];
  key_table_remove(index, key);
  index->free_slot_hint = slot;
  index->active_slots[active_pos] = last_slot;
  index->slots[last_slot].active_pos = active_pos;
  index->slots[slot] = MemorySlot{};
  --index->count;
  if (active_pos != last_pos) {
    index->dense_active = 0;
  }
  graph_rebuild(index);
  return ASTRAL_OK;
}

AstralErr memory_search(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                        AstralMemorySearchResult* out_results, uint32_t max_results,
                        uint32_t* out_count) {
  if (index == nullptr || desc == nullptr || desc->size != sizeof(AstralMemorySearchDesc) ||
      query == nullptr || out_count == nullptr || desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }
  if (max_results < desc->top_k || out_results == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  if (!f32_values_finite(query, index->dim)) {
    return ASTRAL_E_INVALID;
  }

  if (index->index_kind == ASTRAL_MEMORY_INDEX_GRAPH) {
    memory_search_graph(index, desc, query, out_results, out_count);
  } else {
    memory_search_flat(index, desc, query, out_results, out_count);
  }
  return ASTRAL_OK;
}

AstralErr memory_search_batch(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* queries, uint32_t query_count,
                              AstralMemorySearchResult* out_results, uint32_t max_results,
                              uint32_t* out_counts) {
  if (index == nullptr || desc == nullptr || desc->size != sizeof(AstralMemorySearchDesc) ||
      queries == nullptr || out_results == nullptr || out_counts == nullptr || desc->top_k == 0 ||
      query_count == 0) {
    return ASTRAL_E_INVALID;
  }
  if (query_count > kU32Max / desc->top_k) {
    return ASTRAL_E_NOMEM;
  }
  if (query_count > kU32Max / index->dim) {
    return ASTRAL_E_NOMEM;
  }
  const uint32_t result_capacity = query_count * desc->top_k;
  if (max_results < result_capacity) {
    return ASTRAL_E_NOMEM;
  }
  if (!f32_values_finite(queries, static_cast<size_t>(query_count) * index->dim)) {
    return ASTRAL_E_INVALID;
  }

  if (index->index_kind == ASTRAL_MEMORY_INDEX_FLAT) {
    if (!memory_search_flat_batch_record_parallel(index, desc, queries, query_count, out_results,
                                                  out_counts) &&
        !memory_search_flat_batch_parallel(index, desc, queries, query_count, out_results,
                                           out_counts)) {
      memory_search_flat_batch_chunked(index, desc, queries, query_count, out_results, out_counts);
    }
    return ASTRAL_OK;
  }

  if (memory_search_graph_batch_parallel(index, desc, queries, query_count, out_results,
                                         out_counts)) {
    return ASTRAL_OK;
  }

  for (uint32_t i = 0; i < query_count; ++i) {
    uint32_t count = 0;
    const float* query = queries + static_cast<size_t>(i) * index->dim;
    AstralMemorySearchResult* results = out_results + static_cast<size_t>(i) * desc->top_k;
    if (index->index_kind == ASTRAL_MEMORY_INDEX_GRAPH) {
      memory_search_graph(index, desc, query, results, &count);
    } else {
      memory_search_flat(index, desc, query, results, &count);
    }
    out_counts[i] = count;
  }
  return ASTRAL_OK;
}

AstralErr memory_search_begin(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* query, MemorySearchCursor** out_cursor) {
  if (index == nullptr || desc == nullptr || desc->size != sizeof(AstralMemorySearchDesc) ||
      query == nullptr || out_cursor == nullptr || desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }
  if (!f32_values_finite(query, index->dim)) {
    return ASTRAL_E_INVALID;
  }

  const uint32_t capacity = desc->top_k < index->count ? desc->top_k : index->count;
  MemorySearchCursor* cursor = core::runtime_new<MemorySearchCursor>();
  if (cursor == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  cursor->handle = 0;
  cursor->capacity = capacity;
  cursor->count = kNoResults;
  cursor->offset = 0;
  cursor->canceled.store(0, std::memory_order_relaxed);
  cursor->results = capacity <= kInlineSearchCursorResults
                        ? cursor->inline_results
                        : core::runtime_alloc_array<AstralMemorySearchResult>(capacity);
  if (capacity != kNoResults && cursor->results == nullptr) {
    destroy_search_cursor(cursor);
    return ASTRAL_E_NOMEM;
  }

  if (capacity != kNoResults) {
    AstralMemorySearchDesc bounded_desc = *desc;
    bounded_desc.top_k = capacity;
    uint32_t result_count = 0;
    const AstralErr search_err =
        memory_search(index, &bounded_desc, query, cursor->results, capacity, &result_count);
    if (search_err != ASTRAL_OK) {
      destroy_search_cursor(cursor);
      return search_err;
    }
    cursor->count = result_count;
  }

  const AstralHandle handle = core::register_handle(core::HandleKind::MemorySearch, cursor);
  if (handle == 0) {
    destroy_search_cursor(cursor);
    return ASTRAL_E_BUSY;
  }

  cursor->handle = handle;
  *out_cursor = cursor;
  return ASTRAL_OK;
}

AstralErr memory_search_fetch(MemorySearchCursor* cursor, AstralMemorySearchResult* out_results,
                              uint32_t max_results, uint32_t* out_count) {
  if (cursor == nullptr || out_count == nullptr || (max_results != 0 && out_results == nullptr)) {
    return ASTRAL_E_INVALID;
  }
  if (cursor->canceled.load(std::memory_order_acquire) != 0) {
    *out_count = kNoResults;
    return ASTRAL_E_CANCELED;
  }
  if (max_results == 0 || cursor->offset >= cursor->count) {
    *out_count = kNoResults;
    return ASTRAL_OK;
  }

  const uint32_t remaining = cursor->count - cursor->offset;
  const uint32_t to_copy = remaining < max_results ? remaining : max_results;
  std::memcpy(out_results, cursor->results + cursor->offset,
              sizeof(AstralMemorySearchResult) * to_copy);
  cursor->offset += to_copy;
  *out_count = to_copy;
  return ASTRAL_OK;
}

AstralErr memory_search_cancel(MemorySearchCursor* cursor) {
  if (cursor == nullptr) {
    return ASTRAL_E_INVALID;
  }
  cursor->canceled.store(1, std::memory_order_release);
  return ASTRAL_OK;
}

AstralErr memory_search_cursor_status(MemorySearchCursor* cursor, AstralRequestState* out_state,
                                      uint32_t* out_remaining) {
  if (cursor == nullptr || out_state == nullptr || out_remaining == nullptr) {
    return ASTRAL_E_INVALID;
  }
  if (cursor->canceled.load(std::memory_order_acquire) != 0) {
    *out_state = ASTRAL_REQUEST_CANCELED;
    *out_remaining = kNoResults;
    return ASTRAL_OK;
  }
  *out_state = ASTRAL_REQUEST_COMPLETED;
  *out_remaining = cursor->offset < cursor->count ? cursor->count - cursor->offset : kNoResults;
  return ASTRAL_OK;
}

AstralErr memory_search_cursor_remaining(MemorySearchCursor* cursor, uint32_t* out_remaining) {
  if (cursor == nullptr || out_remaining == nullptr) {
    return ASTRAL_E_INVALID;
  }
  *out_remaining = cursor->offset < cursor->count ? cursor->count - cursor->offset : kNoResults;
  return ASTRAL_OK;
}

void memory_search_end(MemorySearchCursor* cursor) {
  if (cursor == nullptr) {
    return;
  }
  core::unregister_handle(cursor->handle, core::HandleKind::MemorySearch);
  destroy_search_cursor(cursor);
}

AstralErr memory_load(const AstralMemoryIndexDesc* desc, AstralSpanU8 bytes,
                      MemoryIndex** out_index) {
  if (!desc_valid(desc) || bytes.data == nullptr || bytes.len < sizeof(SaveHeader) ||
      out_index == nullptr) {
    return ASTRAL_E_INVALID;
  }

  SaveHeader header{};
  std::memcpy(&header, bytes.data, sizeof(header));
  const bool version_valid =
      header.version == kSaveVersionF32 || header.version == kSaveVersionCompactStorage ||
      header.version == kSaveVersionGraphTopology || header.version == kSaveVersionLegacyLayout ||
      header.version == kSaveVersionModernLayout ||
      header.version == kSaveVersionNormalizedF32Cosine ||
      header.version == kSaveVersionGraphQuerySearch ||
      header.version == kSaveVersionCompactScoreScales ||
      header.version == kSaveVersionCompactGraphCounts || header.version == kSaveVersion;
  AstralMemoryStorageKind saved_storage =
      static_cast<AstralMemoryStorageKind>(ASTRAL_MEMORY_STORAGE_F32);
  if (header.version >= kSaveVersionCompactStorage) {
    saved_storage = static_cast<AstralMemoryStorageKind>(header._reserved0);
  }
  if (header.magic != kSaveMagic || !version_valid || header.dim != desc->dim ||
      header.metric != desc->metric || header.index_kind != desc->index_kind ||
      header.count > desc->capacity || !storage_kind_valid(saved_storage)) {
    return ASTRAL_E_INVALID;
  }

  SaveLayout layout{};
  if (!memory_save_layout(header.version, header.dim, header.count, saved_storage,
                          header.index_kind, 0, 0, 0, &layout) ||
      bytes.len < layout.graph_offset) {
    return ASTRAL_E_INVALID;
  }

  MemoryIndex* index = nullptr;
  AstralErr err = memory_create(desc, &index);
  if (err != ASTRAL_OK) {
    return err;
  }

  float vector[kMaxDim];
  for (uint32_t i = 0; i < header.count; ++i) {
    AstralMemoryRecord record{};
    const uint8_t* record_src =
        bytes.data + layout.record_offset + static_cast<uint64_t>(i) * layout.record_stride;
    std::memcpy(&record, record_src, sizeof(record));
    if (record.size != sizeof(AstralMemoryRecord) || record.key == 0) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    const bool saved_compact = compact_storage_kind(saved_storage);
    const bool dst_compact = compact_storage(index);
    float saved_compact_scale = 1.0f;
    const int8_t* saved_compact_vector = nullptr;
    const int16_t* saved_i16_vector = nullptr;
    if (saved_compact) {
      float scale = 1.0f;
      const uint8_t* scale_src =
          bytes.data + layout.scale_offset + static_cast<uint64_t>(i) * layout.scale_stride;
      std::memcpy(&scale, scale_src, sizeof(float));
      const uint8_t* compact_bytes =
          bytes.data + layout.vector_offset + static_cast<uint64_t>(i) * layout.vector_stride;
      const int8_t* compact = reinterpret_cast<const int8_t*>(compact_bytes);
      const int16_t* compact_i16 = reinterpret_cast<const int16_t*>(compact_bytes);
      if (dst_compact && saved_storage == desc->storage_kind) {
        saved_compact_scale = scale;
        if (i16_storage_kind(saved_storage)) {
          saved_i16_vector = compact_i16;
        } else {
          saved_compact_vector = compact;
        }
        if (f32_rerank_storage_kind(saved_storage)) {
          const uint8_t* rerank_src = bytes.data + layout.rerank_vector_offset +
                                      static_cast<uint64_t>(i) * layout.rerank_vector_stride;
          std::memcpy(vector, rerank_src, sizeof(float) * header.dim);
        }
      } else {
        if (f32_rerank_storage_kind(saved_storage)) {
          const uint8_t* rerank_src = bytes.data + layout.rerank_vector_offset +
                                      static_cast<uint64_t>(i) * layout.rerank_vector_stride;
          std::memcpy(vector, rerank_src, sizeof(float) * header.dim);
        } else {
          for (uint32_t dim_i = 0; dim_i < header.dim; ++dim_i) {
            if (saved_storage == ASTRAL_MEMORY_STORAGE_F6_E2M3) {
              vector[dim_i] = e2m3_scaled_to_f32(compact[dim_i], scale);
            } else if (saved_storage == ASTRAL_MEMORY_STORAGE_F6_E3M2) {
              vector[dim_i] = e3m2_scaled_to_f32(compact_i16[dim_i], scale);
            } else if (saved_storage == ASTRAL_MEMORY_STORAGE_F8_E5M2) {
              vector[dim_i] = e5m2_to_f32(static_cast<uint8_t>(compact[dim_i])) * scale;
            } else {
              vector[dim_i] = static_cast<float>(compact[dim_i]) * scale;
            }
          }
        }
      }
    } else {
      const uint8_t* vector_src =
          bytes.data + layout.vector_offset + static_cast<uint64_t>(i) * layout.vector_stride;
      std::memcpy(vector, vector_src, sizeof(float) * header.dim);
    }

    const uint32_t slot = i;
    const uint64_t key_hash = key_hash_mix(record.key);
    if (find_slot_by_key_hashed(index, record.key, key_hash) != kU32Max) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    err = key_table_insert_new_hashed(index, key_hash, slot);
    if (err != ASTRAL_OK) {
      memory_destroy(index);
      return err;
    }
    index->slots[slot].record = record;
    index->slots[slot].score_scale = 0.0f;
    index->slots[slot].active_pos = i;
    index->slots[slot].occupied = 1;
    index->active_slots[i] = slot;
    if (dst_compact) {
      if (saved_i16_vector != nullptr) {
        index->q8_scales[slot] = saved_compact_scale;
        std::memcpy(i16_vector_at(index, slot), saved_i16_vector,
                    static_cast<size_t>(index->dim) * sizeof(int16_t));
        if (f32_rerank_storage(index)) {
          store_f32_vector(index, slot, vector);
        }
        if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
          index->slots[slot].score_scale = 1.0f;
        }
      } else if (saved_compact_vector != nullptr) {
        index->q8_scales[slot] = saved_compact_scale;
        if (e5m2_storage(index)) {
          copy_finite_e5m2(q8_vector_at(index, slot), saved_compact_vector, index->dim);
        } else {
          std::memcpy(q8_vector_at(index, slot), saved_compact_vector,
                      static_cast<size_t>(index->dim) * sizeof(int8_t));
        }
        index->compact_vector_sums[slot] = sum_i8(q8_vector_at(index, slot), index->dim);
        if (f32_rerank_storage(index)) {
          store_f32_vector(index, slot, vector);
        }
        if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
          index->slots[slot].score_scale =
              q8_storage(index)
                  ? cosine_scale_q8(q8_vector_at(index, slot), saved_compact_scale, index->dim)
                  : 1.0f;
        }
      } else {
        if (q8_storage(index)) {
          index->slots[slot].score_scale = index->metric == ASTRAL_MEMORY_METRIC_COSINE
                                               ? cosine_scale(vector, index->dim)
                                               : 0.0f;
          quantize_q8_vector(q8_vector_at(index, slot), &index->q8_scales[slot], vector,
                             index->dim);
          index->compact_vector_sums[slot] = sum_i8(q8_vector_at(index, slot), index->dim);
        } else if (e2m3_storage(index)) {
          index->slots[slot].score_scale =
              index->metric == ASTRAL_MEMORY_METRIC_COSINE ? 1.0f : 0.0f;
          if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
            quantize_e2m3_cosine_vector(q8_vector_at(index, slot), &index->q8_scales[slot], vector,
                                        index->dim);
          } else {
            quantize_e2m3_vector(q8_vector_at(index, slot), &index->q8_scales[slot], vector,
                                 index->dim);
          }
          index->compact_vector_sums[slot] = sum_i8(q8_vector_at(index, slot), index->dim);
        } else if (e3m2_storage(index)) {
          index->slots[slot].score_scale =
              index->metric == ASTRAL_MEMORY_METRIC_COSINE ? 1.0f : 0.0f;
          if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
            quantize_e3m2_cosine_vector(i16_vector_at(index, slot), &index->q8_scales[slot], vector,
                                        index->dim);
          } else {
            quantize_e3m2_vector(i16_vector_at(index, slot), &index->q8_scales[slot], vector,
                                 index->dim);
          }
        } else {
          if (f32_rerank_storage(index)) {
            store_f32_vector(index, slot, vector);
          }
          index->slots[slot].score_scale =
              index->metric == ASTRAL_MEMORY_METRIC_COSINE ? 1.0f : 0.0f;
          if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
            quantize_e5m2_cosine_vector(q8_vector_at(index, slot), &index->q8_scales[slot], vector,
                                        index->dim);
          } else {
            quantize_e5m2_vector(q8_vector_at(index, slot), &index->q8_scales[slot], vector,
                                 index->dim);
          }
          index->compact_vector_sums[slot] = sum_i8(q8_vector_at(index, slot), index->dim);
        }
      }
      update_compact_score_scale(index, slot);
    } else {
      store_f32_vector(index, slot, vector);
    }
    if (graph_enabled(index)) {
      index->graph_levels[slot] = static_cast<uint8_t>(graph_level_for_key(index, record.key));
    }
  }
  index->count = header.count;
  index->free_slot_hint = header.count < index->capacity ? header.count : 0;
  index->dense_active = 1u;

  const uint8_t* cursor = bytes.data + layout.graph_offset;
  bool graph_loaded = false;
  if (graph_enabled(index) && header.version >= kSaveVersionGraphTopology &&
      header._reserved1 == kSaveGraphTopologyFlag) {
    const uint32_t saved_graph_header_bytes = save_graph_header_bytes(header.version);
    if (bytes.len < static_cast<uint64_t>(cursor - bytes.data) + saved_graph_header_bytes) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    SaveGraphHeader graph_header{};
    read_save_graph_header(header.version, cursor, &graph_header);
    cursor += saved_graph_header_bytes;
    if (graph_header.neighbor_capacity == 0 ||
        graph_header.neighbor_capacity > kGraphMaxNeighbors ||
        graph_header.base_neighbor_capacity == 0 ||
        graph_header.base_neighbor_capacity > kGraphMaxBaseNeighbors ||
        graph_header.base_neighbor_capacity < graph_header.neighbor_capacity ||
        graph_header.query_search_capacity == 0 || graph_header.level_capacity == 0 ||
        graph_header.level_capacity > kGraphMaxLevels ||
        graph_header.search_capacity > desc->capacity ||
        graph_header.query_search_capacity > desc->capacity) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    const uint64_t level_count = static_cast<uint64_t>(header.count) * graph_header.level_capacity;
    const uint64_t neighbor_slots =
        static_cast<uint64_t>(header.count) * graph_header.base_neighbor_capacity +
        static_cast<uint64_t>(header.count) * (graph_header.level_capacity - 1u) *
            graph_header.neighbor_capacity;
    const uint32_t graph_count_bytes = save_graph_count_bytes(header.version);
    const uint64_t graph_bytes =
        saved_graph_header_bytes + static_cast<uint64_t>(header.count) * sizeof(uint8_t) +
        level_count * graph_count_bytes + neighbor_slots * sizeof(uint32_t);
    const uint64_t graph_begin = layout.graph_offset;
    if (graph_header.flags != kSaveGraphTopologyFlag || bytes.len < graph_begin + graph_bytes) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    const bool topology_matches =
        graph_header.neighbor_capacity == index->graph_neighbor_capacity &&
        graph_header.base_neighbor_capacity == index->graph_base_neighbor_capacity &&
        graph_header.search_capacity == index->graph_search_capacity &&
        graph_header.level_capacity == index->graph_level_capacity &&
        graph_header.max_level < graph_header.level_capacity &&
        graph_header.entry_active_pos < header.count;
    if (topology_matches) {
      for (uint32_t active_pos = 0; active_pos < header.count; ++active_pos) {
        const uint8_t level = *cursor;
        ++cursor;
        if (level >= index->graph_level_capacity) {
          memory_destroy(index);
          return ASTRAL_E_INVALID;
        }
        index->graph_levels[active_pos] = level;
      }
      for (uint32_t level = 0; level < index->graph_level_capacity; ++level) {
        for (uint32_t active_pos = 0; active_pos < header.count; ++active_pos) {
          uint32_t count = 0;
          if (graph_count_bytes == sizeof(uint8_t)) {
            count = *cursor;
          } else {
            std::memcpy(&count, cursor, sizeof(count));
          }
          cursor += graph_count_bytes;
          if (count > graph_neighbor_capacity_at_level(index, level)) {
            memory_destroy(index);
            return ASTRAL_E_INVALID;
          }
          graph_neighbor_count_ref(index, active_pos, level) = static_cast<uint8_t>(count);
        }
      }
      for (uint32_t level = 0; level < index->graph_level_capacity; ++level) {
        for (uint32_t active_pos = 0; active_pos < header.count; ++active_pos) {
          uint32_t* neighbors = graph_neighbors_at_level(index, active_pos, level);
          const uint32_t capacity = graph_neighbor_capacity_at_level(index, level);
          for (uint32_t i = 0; i < capacity; ++i) {
            uint32_t neighbor_pos = kU32Max;
            std::memcpy(&neighbor_pos, cursor, sizeof(neighbor_pos));
            cursor += sizeof(neighbor_pos);
            if (neighbor_pos != kU32Max && neighbor_pos >= header.count) {
              memory_destroy(index);
              return ASTRAL_E_INVALID;
            }
            neighbors[i] = neighbor_pos;
          }
        }
      }
      index->graph_entry_slot = graph_header.entry_active_pos;
      index->graph_max_level = graph_header.max_level;
      graph_loaded = true;
    }
  }
  if (graph_enabled(index) && !graph_loaded) {
    graph_rebuild(index);
  }

  *out_index = index;
  return ASTRAL_OK;
}

} // namespace astral::inference
