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

namespace {

constexpr uint32_t kSaveMagic = 0x414D454Du;
constexpr uint32_t kSaveVersionF32 = 1;
constexpr uint32_t kSaveVersionCompactStorage = 2;
constexpr uint32_t kSaveVersionGraphTopology = 3;
constexpr uint32_t kSaveVersionLegacyLayout = 4;
constexpr uint32_t kSaveVersionModernLayout = 5;
constexpr uint32_t kSaveVersionNormalizedF32Cosine = 6;
constexpr uint32_t kSaveVersionGraphQuerySearch = 7;
constexpr uint32_t kSaveVersionCompactScoreScales = 8;
constexpr uint32_t kSaveVersionCompactGraphCounts = 9;
constexpr uint32_t kSaveVersionAlignedVectorData = 10;
constexpr uint32_t kSaveVersion = kSaveVersionAlignedVectorData;
constexpr uint32_t kSaveGraphTopologyFlag = 1;
constexpr uint32_t kU32Max = 0xFFFFFFFFu;
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
constexpr uint32_t kInlineSearchCursorResults = 8;
constexpr uint32_t kGraphMaxLevels = 16;
constexpr uint32_t kMemoryBatchStackQueries = 16;
constexpr uint32_t kMemoryAddStackSlots = 256;
constexpr uint32_t kMemoryAddPlanStackEntries = kMemoryAddStackSlots * 2;
constexpr uint32_t kMemoryAddParallelMinCount = 1024;
constexpr uint32_t kMemoryAddParallelMaxWorkers = 8;
constexpr uint32_t kMemorySearchBatchParallelMinQueries = 8;
constexpr uint32_t kMemorySearchBatchParallelMaxRecords = 32768;
constexpr uint32_t kMemorySearchBatchParallelMaxWorkers = 16;
constexpr uint32_t kMemorySearchBatchParallelMaxTopK = 8;
constexpr uint32_t kMemorySearchRecordParallelMaxTopK = 10;

inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  const uint64_t mask = alignment - 1u;
  return (value + mask) & ~mask;
}

inline uint64_t save_payload_align(uint32_t version) {
  return version >= kSaveVersionAlignedVectorData ? static_cast<uint64_t>(kVectorStorageAlign) : 1u;
}

struct MemorySlot {
  AstralMemoryRecord record;
  float score_scale;
  uint32_t active_pos;
  uint8_t occupied;
};

struct SaveHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t dim;
  uint32_t count;
  uint32_t metric;
  uint32_t index_kind;
  uint32_t _reserved0;
  uint32_t _reserved1;
};

struct SaveGraphHeader {
  uint32_t flags;
  uint32_t neighbor_capacity;
  uint32_t base_neighbor_capacity;
  uint32_t search_capacity;
  uint32_t query_search_capacity;
  uint32_t level_capacity;
  uint32_t max_level;
  uint32_t entry_active_pos;
};

struct SaveGraphHeaderV6 {
  uint32_t flags;
  uint32_t neighbor_capacity;
  uint32_t base_neighbor_capacity;
  uint32_t search_capacity;
  uint32_t level_capacity;
  uint32_t max_level;
  uint32_t entry_active_pos;
};

struct SaveGraphHeaderV3 {
  uint32_t flags;
  uint32_t neighbor_capacity;
  uint32_t search_capacity;
  uint32_t level_capacity;
  uint32_t max_level;
  uint32_t entry_active_pos;
};

struct SaveLayout {
  uint64_t record_offset;
  uint64_t record_stride;
  uint64_t scale_offset;
  uint64_t scale_stride;
  uint64_t compact_score_scale_offset;
  uint64_t compact_score_scale_stride;
  uint64_t vector_offset;
  uint64_t vector_stride;
  uint64_t rerank_vector_offset;
  uint64_t rerank_vector_stride;
  uint64_t graph_offset;
  uint64_t graph_bytes;
  uint64_t total_bytes;
};

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

inline bool i16_storage_kind(AstralMemoryStorageKind kind) {
  return kind == ASTRAL_MEMORY_STORAGE_F6_E3M2 || kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK;
}

inline bool f32_rerank_storage_kind(AstralMemoryStorageKind kind) {
  return kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E2M3_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK;
}

inline bool compact_storage_kind(AstralMemoryStorageKind kind) {
  return kind == ASTRAL_MEMORY_STORAGE_Q8 || kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 ||
         kind == ASTRAL_MEMORY_STORAGE_F8_E5M2 || kind == ASTRAL_MEMORY_STORAGE_F6_E3M2 ||
         kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK ||
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

struct GraphSearchScratch {
  uint32_t* candidates;
  float* candidate_scores;
  uint32_t* top_slots;
  float* top_scores;
  uint16_t* visited;
  uint16_t visit_generation;
  uint32_t candidate_worst_pos;
  uint32_t candidate_worst_slot;
  float candidate_worst_score;
  uint8_t candidate_worst_valid;
};

struct MemoryIndex {
  AstralHandle handle;
  const E5m2Kernels* e5m2_kernels;
  uint32_t dim;
  uint32_t capacity;
  uint32_t count;
  AstralMemoryMetric metric;
  AstralMemoryIndexKind index_kind;
  AstralMemoryStorageKind storage_kind;
  MemorySlot* slots;
  float* vectors;
  int8_t* q8_vectors;
  int16_t* i16_vectors;
  float* q8_scales;
  float* compact_score_scales;
  int32_t* compact_vector_sums;
  uint32_t* active_slots;
  uint32_t* key_table;
  uint32_t* graph_neighbors;
  uint8_t* graph_neighbor_counts;
  uint8_t* graph_levels;
  uint32_t* graph_candidates;
  float* graph_candidate_scores;
  uint32_t graph_candidate_worst_pos;
  uint32_t graph_candidate_worst_slot;
  float graph_candidate_worst_score;
  uint8_t graph_candidate_worst_valid;
  uint32_t* graph_scratch_slots;
  float* graph_scratch_scores;
  uint16_t* graph_visited;
  GraphSearchScratch graph_batch_scratch[kMemorySearchBatchParallelMaxWorkers];
  uint32_t key_table_capacity;
  uint32_t key_table_mask;
  uint32_t graph_neighbor_capacity;
  uint32_t graph_base_neighbor_capacity;
  uint32_t graph_search_capacity;
  uint32_t graph_query_search_capacity;
  uint32_t graph_candidate_capacity;
  uint32_t graph_scratch_capacity;
  uint32_t graph_level_capacity;
  uint32_t graph_entry_slot;
  uint32_t graph_max_level;
  uint16_t graph_visit_generation;
  uint64_t graph_build_score_evals;
  uint64_t graph_build_candidate_visits;
  uint32_t free_slot_hint;
  std::atomic<uint32_t> graph_batch_scratch_claimed;
  uint8_t graph_batch_scratch_count;
  uint8_t dense_active;
  uint8_t i16_vectors_aligned;
};

struct MemorySearchCursor {
  AstralHandle handle;
  uint32_t capacity;
  uint32_t count;
  uint32_t offset;
  std::atomic<uint32_t> canceled;
  AstralMemorySearchResult* results;
  AstralMemorySearchResult inline_results[kInlineSearchCursorResults];
};

struct SnapshotGraphLayout {
  SaveGraphHeader header;
  uint64_t levels_offset;
  uint64_t counts_offset;
  uint64_t neighbors_offset;
  uint32_t candidate_capacity;
  uint32_t scratch_capacity;
};

struct MemorySnapshotView {
  AstralHandle handle;
  platform::ReadOnlyFileMap map;
  AstralMemorySnapshotInfo info;
  SnapshotGraphLayout graph;
  GraphSearchScratch graph_scratch;
  uint8_t graph_ready;
  uint8_t e5m2_clamp_non_finite;
};

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

struct MemoryGraphSearchBatchJob {
  MemoryIndex* index;
  const AstralMemorySearchDesc* desc;
  const float* queries;
  AstralMemorySearchResult* out_results;
  uint32_t* out_counts;
  uint32_t begin;
  uint32_t end;
  std::atomic<uint32_t>* remaining;
  GraphSearchScratch* scratch;
};

inline void memory_parallel_job_complete(std::atomic<uint32_t>* remaining) {
  if (remaining->fetch_sub(1, std::memory_order_release) == 1u) {
    astral::platform::cpu_signal_event();
  }
}

inline uint32_t save_graph_count_bytes(uint32_t version) {
  return version >= kSaveVersionCompactGraphCounts ? sizeof(uint8_t) : sizeof(uint32_t);
}

bool memory_save_layout(uint32_t version, uint32_t dim, uint32_t count,
                        AstralMemoryStorageKind storage, uint32_t index_kind,
                        uint32_t graph_base_neighbors, uint32_t graph_neighbors,
                        uint32_t graph_levels, SaveLayout* out_layout);

inline uint64_t memory_save_byte_count(const MemoryIndex* index) {
  SaveLayout layout{};
  const bool has_graph =
      index->index_kind == ASTRAL_MEMORY_INDEX_GRAPH && index->graph_neighbor_capacity != 0;
  const uint32_t graph_base_neighbors = has_graph ? index->graph_base_neighbor_capacity : 0u;
  const uint32_t graph_neighbors = has_graph ? index->graph_neighbor_capacity : 0u;
  const uint32_t graph_levels = has_graph ? index->graph_level_capacity : 0u;
  if (!memory_save_layout(kSaveVersion, index->dim, index->count, index->storage_kind,
                          index->index_kind, graph_base_neighbors, graph_neighbors, graph_levels,
                          &layout)) {
    return 0;
  }
  return layout.total_bytes;
}

bool memory_save_layout(uint32_t version, uint32_t dim, uint32_t count,
                        AstralMemoryStorageKind storage, uint32_t index_kind,
                        uint32_t graph_base_neighbors, uint32_t graph_neighbors,
                        uint32_t graph_levels, SaveLayout* out_layout) {
  if (out_layout == nullptr || !storage_kind_valid(storage)) {
    return false;
  }
  SaveLayout layout{};
  const bool compact = compact_storage_kind(storage);
  const uint64_t compact_element_bytes =
      i16_storage_kind(storage) ? sizeof(int16_t) : sizeof(int8_t);
  layout.record_offset = sizeof(SaveHeader);
  layout.record_stride = sizeof(AstralMemoryRecord);
  if (version >= kSaveVersionModernLayout) {
    uint64_t cursor =
        layout.record_offset + static_cast<uint64_t>(count) * sizeof(AstralMemoryRecord);
    const uint64_t payload_align = save_payload_align(version);
    if (compact) {
      layout.scale_offset = cursor;
      layout.scale_stride = sizeof(float);
      cursor += static_cast<uint64_t>(count) * sizeof(float);
      if (version >= kSaveVersionCompactScoreScales) {
        layout.compact_score_scale_offset = cursor;
        layout.compact_score_scale_stride = sizeof(float);
        cursor += static_cast<uint64_t>(count) * sizeof(float);
      }
      cursor = align_up_u64(cursor, payload_align);
      layout.vector_offset = cursor;
      layout.vector_stride = static_cast<uint64_t>(dim) * compact_element_bytes;
      cursor += static_cast<uint64_t>(count) * layout.vector_stride;
      if (f32_rerank_storage_kind(storage)) {
        cursor = align_up_u64(cursor, payload_align);
        layout.rerank_vector_offset = cursor;
        layout.rerank_vector_stride = static_cast<uint64_t>(dim) * sizeof(float);
        cursor += static_cast<uint64_t>(count) * layout.rerank_vector_stride;
      }
    } else {
      cursor = align_up_u64(cursor, payload_align);
      layout.vector_offset = cursor;
      layout.vector_stride = static_cast<uint64_t>(dim) * sizeof(float);
      cursor += static_cast<uint64_t>(count) * layout.vector_stride;
    }
    cursor = align_up_u64(cursor, payload_align);
    layout.graph_offset = cursor;
  } else {
    layout.scale_offset = compact ? layout.record_offset + sizeof(AstralMemoryRecord) : 0;
    layout.scale_stride =
        compact ? sizeof(AstralMemoryRecord) + sizeof(float) + static_cast<uint64_t>(dim) : 0;
    layout.vector_offset =
        layout.record_offset + sizeof(AstralMemoryRecord) + (compact ? sizeof(float) : 0);
    layout.vector_stride =
        sizeof(AstralMemoryRecord) +
        (compact ? sizeof(float) + static_cast<uint64_t>(dim) * compact_element_bytes
                 : static_cast<uint64_t>(dim) * sizeof(float));
    layout.graph_offset =
        layout.record_offset + static_cast<uint64_t>(count) * layout.vector_stride;
  }
  if (index_kind == ASTRAL_MEMORY_INDEX_GRAPH && graph_neighbors != 0 && graph_levels != 0) {
    const uint64_t graph_header_bytes =
        version >= kSaveVersionGraphQuerySearch ? sizeof(SaveGraphHeader)
        : version >= kSaveVersionLegacyLayout   ? sizeof(SaveGraphHeaderV6)
                                                : sizeof(SaveGraphHeaderV3);
    const uint64_t neighbor_slots =
        static_cast<uint64_t>(count) * graph_base_neighbors +
        static_cast<uint64_t>(count) * (graph_levels - 1u) * graph_neighbors;
    layout.graph_bytes =
        graph_header_bytes + static_cast<uint64_t>(count) * sizeof(uint8_t) +
        static_cast<uint64_t>(count) * graph_levels * save_graph_count_bytes(version) +
        neighbor_slots * sizeof(uint32_t);
  }
  layout.total_bytes = layout.graph_offset + layout.graph_bytes;
  *out_layout = layout;
  return true;
}

inline uint32_t save_graph_header_bytes(uint32_t version) {
  return version >= kSaveVersionGraphQuerySearch ? sizeof(SaveGraphHeader)
         : version >= kSaveVersionLegacyLayout   ? sizeof(SaveGraphHeaderV6)
                                                 : sizeof(SaveGraphHeaderV3);
}

inline void memory_save_skip_to(uint8_t** cursor, uint8_t* target) {
  if (*cursor < target) {
    std::memset(*cursor, 0, static_cast<size_t>(target - *cursor));
    *cursor = target;
  }
}

void read_save_graph_header(uint32_t version, const uint8_t* bytes, SaveGraphHeader* out_header) {
  *out_header = {};
  if (version >= kSaveVersionGraphQuerySearch) {
    std::memcpy(out_header, bytes, sizeof(SaveGraphHeader));
    return;
  }
  if (version >= kSaveVersionLegacyLayout) {
    SaveGraphHeaderV6 header_v6{};
    std::memcpy(&header_v6, bytes, sizeof(header_v6));
    out_header->flags = header_v6.flags;
    out_header->neighbor_capacity = header_v6.neighbor_capacity;
    out_header->base_neighbor_capacity = header_v6.base_neighbor_capacity;
    out_header->search_capacity = header_v6.search_capacity;
    out_header->query_search_capacity = header_v6.search_capacity;
    out_header->level_capacity = header_v6.level_capacity;
    out_header->max_level = header_v6.max_level;
    out_header->entry_active_pos = header_v6.entry_active_pos;
    return;
  }
  SaveGraphHeaderV3 header_v3{};
  std::memcpy(&header_v3, bytes, sizeof(header_v3));
  out_header->flags = header_v3.flags;
  out_header->neighbor_capacity = header_v3.neighbor_capacity;
  out_header->base_neighbor_capacity = header_v3.neighbor_capacity;
  out_header->search_capacity = header_v3.search_capacity;
  out_header->query_search_capacity = header_v3.search_capacity;
  out_header->level_capacity = header_v3.level_capacity;
  out_header->max_level = header_v3.max_level;
  out_header->entry_active_pos = header_v3.entry_active_pos;
}

} // namespace

AstralHandle memory_handle(MemoryIndex* index) {
  return index != nullptr ? index->handle : 0;
}

AstralHandle memory_search_cursor_handle(MemorySearchCursor* cursor) {
  return cursor != nullptr ? cursor->handle : 0;
}

AstralHandle memory_snapshot_view_handle(MemorySnapshotView* view) {
  return view != nullptr ? view->handle : 0;
}

namespace {

inline float* vector_at(MemoryIndex* index, uint32_t slot) {
  return index->vectors + static_cast<size_t>(slot) * index->dim;
}

inline const float* vector_at(const MemoryIndex* index, uint32_t slot) {
  return index->vectors + static_cast<size_t>(slot) * index->dim;
}

inline int8_t* q8_vector_at(MemoryIndex* index, uint32_t slot) {
  return index->q8_vectors + static_cast<size_t>(slot) * index->dim;
}

inline const int8_t* q8_vector_at(const MemoryIndex* index, uint32_t slot) {
  return index->q8_vectors + static_cast<size_t>(slot) * index->dim;
}

inline int16_t* i16_vector_at(MemoryIndex* index, uint32_t slot) {
  return index->i16_vectors + static_cast<size_t>(slot) * index->dim;
}

inline const int16_t* i16_vector_at(const MemoryIndex* index, uint32_t slot) {
  return index->i16_vectors + static_cast<size_t>(slot) * index->dim;
}

inline bool q8_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_Q8 ||
         index->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK;
}

inline bool q8_f32_rerank_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK;
}

inline bool f8_f32_rerank_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK;
}

inline bool f32_rerank_storage(const MemoryIndex* index) {
  return f32_rerank_storage_kind(index->storage_kind);
}

inline bool e2m3_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 ||
         index->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3_F32_RERANK;
}

inline bool e5m2_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2 ||
         index->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK;
}

inline bool e3m2_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E3M2 ||
         index->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK;
}

inline bool i16_storage(const MemoryIndex* index) {
  return i16_storage_kind(index->storage_kind);
}

inline bool compact_storage(const MemoryIndex* index) {
  return compact_storage_kind(index->storage_kind);
}

inline float compact_value_scale(const MemoryIndex* index, float scale) {
  return e2m3_storage(index)   ? scale * kE2M3InvScale
         : e3m2_storage(index) ? scale * kE3M2InvScale
                               : scale;
}

inline float compact_value_scale_kind(AstralMemoryStorageKind kind, float scale) {
  return (kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 || kind == ASTRAL_MEMORY_STORAGE_F6_E2M3_F32_RERANK)
             ? scale * kE2M3InvScale
         : (kind == ASTRAL_MEMORY_STORAGE_F6_E3M2 ||
            kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK)
             ? scale * kE3M2InvScale
             : scale;
}

inline void update_compact_score_scale(MemoryIndex* index, uint32_t slot) {
  float scale = compact_value_scale(index, index->q8_scales[slot]);
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    scale *= index->slots[slot].score_scale;
  }
  index->compact_score_scales[slot] = scale;
}

void quantize_compact_query(const MemoryIndex* index, int8_t* dst, float* out_scale,
                            const float* src) {
  if (e2m3_storage(index)) {
    quantize_e2m3_vector(dst, out_scale, src, index->dim);
    *out_scale *= kE2M3InvScale;
    return;
  }
  if (e5m2_storage(index)) {
    quantize_e5m2_vector(dst, out_scale, src, index->dim);
    return;
  }
  quantize_q8_vector(dst, out_scale, src, index->dim);
}

inline void prefetch_slot_vector(const MemoryIndex* index, uint32_t slot) {
#if defined(__GNUC__) || defined(__clang__)
  if (i16_storage(index)) {
    __builtin_prefetch(i16_vector_at(index, slot), 0, 1);
  } else if (compact_storage(index)) {
    __builtin_prefetch(q8_vector_at(index, slot), 0, 1);
  } else {
    __builtin_prefetch(vector_at(index, slot), 0, 1);
  }
#else
  (void)index;
  (void)slot;
#endif
}

inline uint32_t graph_search_for_query(const MemoryIndex* index,
                                       const AstralMemorySearchDesc* desc) {
  uint32_t requested = desc->graph_search;
  if (requested == 0) {
    requested = index->graph_query_search_capacity;
  }
  if (requested < kGraphMinSearch) {
    requested = kGraphMinSearch;
  }
  return requested < index->graph_scratch_capacity ? requested : index->graph_scratch_capacity;
}

inline uint32_t graph_candidate_search_capacity(const MemoryIndex* index,
                                                uint32_t search_capacity) {
  uint32_t requested = search_capacity;
  if (search_capacity <= kU32Max / kGraphCandidateReserveMultiplier) {
    requested = search_capacity * kGraphCandidateReserveMultiplier;
  }
  return requested < index->graph_candidate_capacity ? requested : index->graph_candidate_capacity;
}

inline bool compact_graph_exact_search_preferred(const MemoryIndex* index) {
  return compact_storage(index) &&
         static_cast<uint64_t>(index->count) * static_cast<uint64_t>(index->dim) <=
             kGraphCompactExactSearchMaxBytes;
}

inline bool snapshot_compact_graph_exact_search_preferred(const AstralMemorySnapshotInfo* info) {
  return compact_storage_kind(info->storage_kind) &&
         static_cast<uint64_t>(info->count) * static_cast<uint64_t>(info->dim) <=
             kGraphCompactExactSearchMaxBytes;
}

inline uint32_t active_slot_at(const MemoryIndex* index, uint32_t active_pos) {
  return index->dense_active != 0 ? active_pos : index->active_slots[active_pos];
}

inline bool graph_enabled(const MemoryIndex* index) {
  return index->index_kind == ASTRAL_MEMORY_INDEX_GRAPH && index->graph_neighbor_capacity != 0;
}

inline uint32_t graph_neighbor_capacity_at_level(const MemoryIndex* index, uint32_t level) {
  return level == 0 ? index->graph_base_neighbor_capacity : index->graph_neighbor_capacity;
}

inline uint32_t graph_outgoing_capacity_at_level(const MemoryIndex* index, uint32_t) {
  return index->graph_neighbor_capacity;
}

inline size_t graph_neighbor_storage_count(const MemoryIndex* index) {
  return static_cast<size_t>(index->capacity) * index->graph_base_neighbor_capacity +
         static_cast<size_t>(index->capacity) * (index->graph_level_capacity - 1u) *
             index->graph_neighbor_capacity;
}

inline size_t graph_level_offset(const MemoryIndex* index, uint32_t slot, uint32_t level) {
  if (level == 0) {
    return static_cast<size_t>(slot) * index->graph_base_neighbor_capacity;
  }
  return static_cast<size_t>(index->capacity) * index->graph_base_neighbor_capacity +
         (static_cast<size_t>(level - 1u) * index->capacity + slot) *
             index->graph_neighbor_capacity;
}

inline uint32_t* graph_neighbors_at_level(MemoryIndex* index, uint32_t slot, uint32_t level) {
  return index->graph_neighbors + graph_level_offset(index, slot, level);
}

inline uint32_t graph_neighbor_count_at_level(const MemoryIndex* index, uint32_t slot,
                                              uint32_t level) {
  return index->graph_neighbor_counts[static_cast<size_t>(level) * index->capacity + slot];
}

inline uint8_t& graph_neighbor_count_ref(MemoryIndex* index, uint32_t slot, uint32_t level) {
  return index->graph_neighbor_counts[static_cast<size_t>(level) * index->capacity + slot];
}

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

inline bool record_matches_group(const AstralMemorySearchDesc* desc, const MemorySlot& slot) {
  return desc->group_id == ASTRAL_MEMORY_GROUP_ANY || slot.record.group_id == desc->group_id;
}

inline bool result_better(const AstralMemorySearchResult& candidate,
                          const AstralMemorySearchResult& existing) {
  return candidate.score > existing.score ||
         (candidate.score == existing.score && candidate.key < existing.key);
}

inline bool result_better_values(float candidate_score, uint64_t candidate_key,
                                 const AstralMemorySearchResult& existing) {
  return candidate_score > existing.score ||
         (candidate_score == existing.score && candidate_key < existing.key);
}

inline bool graph_candidate_better(const MemoryIndex* index, float candidate_score,
                                   uint32_t candidate_slot, float existing_score,
                                   uint32_t existing_slot) {
  return candidate_score > existing_score ||
         (candidate_score == existing_score &&
          index->slots[candidate_slot].record.key < index->slots[existing_slot].record.key);
}

inline bool graph_candidate_worse(const MemoryIndex* index, float candidate_score,
                                  uint32_t candidate_slot, float existing_score,
                                  uint32_t existing_slot) {
  return candidate_score < existing_score ||
         (candidate_score == existing_score &&
          index->slots[candidate_slot].record.key > index->slots[existing_slot].record.key);
}

inline void fill_result(AstralMemorySearchResult* out_result, const MemorySlot& slot, float score) {
  out_result->size = sizeof(AstralMemorySearchResult);
  out_result->group_id = slot.record.group_id;
  out_result->key = slot.record.key;
  out_result->document_id = slot.record.document_id;
  out_result->chunk_id = slot.record.chunk_id;
  out_result->flags = slot.record.flags;
  out_result->score = score;
}

inline float score_slot(MemoryIndex* index, const float* query, uint32_t slot, float query_scale) {
  if (f32_rerank_storage(index)) {
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return dot_f32(query, vector_at(index, slot), index->dim);
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return dot_f32(query, vector_at(index, slot), index->dim) * query_scale;
    }
    return l2_score_f32(query, vector_at(index, slot), index->dim);
  }
  if (compact_storage(index)) {
    const float scale = index->compact_score_scales[slot];
    if (i16_storage(index)) {
      const int16_t* v = i16_vector_at(index, slot);
      if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
        return dot_i16_f32(v, query, index->dim) * scale;
      }
      if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        return dot_i16_f32(v, query, index->dim) * scale * query_scale;
      }
      return l2_score_i16_f32(v, compact_value_scale(index, index->q8_scales[slot]), query,
                              index->dim);
    }
    const int8_t* q8 = q8_vector_at(index, slot);
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return (e5m2_storage(index) ? index->e5m2_kernels->dot_f32(q8, query, index->dim)
                                  : dot_q8_f32(q8, query, index->dim)) *
             scale;
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return (e5m2_storage(index) ? index->e5m2_kernels->dot_f32(q8, query, index->dim)
                                  : dot_q8_f32(q8, query, index->dim)) *
             scale * query_scale;
    }
    return e5m2_storage(index)
               ? index->e5m2_kernels->l2_f32(q8, compact_value_scale(index, index->q8_scales[slot]),
                                             query, index->dim)
               : l2_score_q8_f32(q8, compact_value_scale(index, index->q8_scales[slot]), query,
                                 index->dim);
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    return dot_f32(query, vector_at(index, slot), index->dim);
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return dot_f32(query, vector_at(index, slot), index->dim) * query_scale;
  }
  return l2_score_f32(query, vector_at(index, slot), index->dim);
}

enum class CompactByteQueryStorage : uint8_t {
  q8,
  e5m2,
};

template <CompactByteQueryStorage Storage>
inline float score_slot_compact_query(MemoryIndex* index, const int8_t* query, int32_t query_sum,
                                      float query_scale, uint32_t slot, float cosine_query_scale) {
  const int8_t* q8 = q8_vector_at(index, slot);
  const float scale = index->compact_score_scales[slot];
  if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    if constexpr (Storage == CompactByteQueryStorage::e5m2) {
      return index->e5m2_kernels->dot_e5m2(q8, query, index->dim) * scale * query_scale;
    } else {
      return dot_q8_q8_query_aligned(q8, query, index->dim, query_sum) * scale * query_scale;
    }
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    if constexpr (Storage == CompactByteQueryStorage::e5m2) {
      return index->e5m2_kernels->dot_e5m2(q8, query, index->dim) * scale * query_scale *
             cosine_query_scale;
    } else {
      return dot_q8_q8_query_aligned(q8, query, index->dim, query_sum) * scale * query_scale *
             cosine_query_scale;
    }
  }
  if constexpr (Storage == CompactByteQueryStorage::e5m2) {
    return index->e5m2_kernels->l2_e5m2(q8, compact_value_scale(index, index->q8_scales[slot]),
                                        query, query_scale, index->dim);
  } else {
    return l2_score_q8_q8(q8, compact_value_scale(index, index->q8_scales[slot]), query,
                          query_scale, index->dim);
  }
}

inline float score_slot_compact_query_i16(MemoryIndex* index, const int16_t* query,
                                          float query_scale, uint32_t slot,
                                          float cosine_query_scale) {
  const int16_t* v = i16_vector_at(index, slot);
  const float scale = index->compact_score_scales[slot];
  if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    const float dot = index->i16_vectors_aligned != 0 ? dot_i16_i16_aligned(v, query, index->dim)
                                                      : dot_i16_i16(v, query, index->dim);
    return dot * scale * query_scale;
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    const float dot = index->i16_vectors_aligned != 0 ? dot_i16_i16_aligned(v, query, index->dim)
                                                      : dot_i16_i16(v, query, index->dim);
    return dot * scale * query_scale * cosine_query_scale;
  }
  return l2_score_i16_i16(v, compact_value_scale(index, index->q8_scales[slot]), query, query_scale,
                          index->dim);
}

inline float score_pair(MemoryIndex* index, uint32_t a, uint32_t b) {
  ++index->graph_build_score_evals;
  if (f8_f32_rerank_storage(index)) {
    const float* va = vector_at(index, a);
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return dot_f32(va, vector_at(index, b), index->dim);
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
      return l2_score_f32(va, vector_at(index, b), index->dim);
    }
    return dot_f32(va, vector_at(index, b), index->dim);
  }
  if (compact_storage(index)) {
    const float scale_a = index->compact_score_scales[a];
    const float scale_b = index->compact_score_scales[b];
    if (i16_storage(index)) {
      const int16_t* va = i16_vector_at(index, a);
      const int16_t* vb = i16_vector_at(index, b);
      if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        const float dot = index->i16_vectors_aligned != 0 ? dot_i16_i16_aligned(va, vb, index->dim)
                                                          : dot_i16_i16(va, vb, index->dim);
        return dot * scale_a * scale_b;
      }
      if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
        return l2_score_i16_i16(va, compact_value_scale(index, index->q8_scales[a]), vb,
                                compact_value_scale(index, index->q8_scales[b]), index->dim);
      }
      const float dot = index->i16_vectors_aligned != 0 ? dot_i16_i16_aligned(va, vb, index->dim)
                                                        : dot_i16_i16(va, vb, index->dim);
      return dot * scale_a * scale_b;
    }
    const int8_t* va = q8_vector_at(index, a);
    const int8_t* vb = q8_vector_at(index, b);
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return (e5m2_storage(index)
                  ? index->e5m2_kernels->dot_e5m2(va, vb, index->dim)
                  : dot_q8_q8_query_aligned(va, vb, index->dim, index->compact_vector_sums[b])) *
             scale_a * scale_b;
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
      return e5m2_storage(index)
                 ? index->e5m2_kernels->l2_e5m2(va, compact_value_scale(index, index->q8_scales[a]),
                                                vb, compact_value_scale(index, index->q8_scales[b]),
                                                index->dim)
                 : l2_score_q8_q8(va, compact_value_scale(index, index->q8_scales[a]), vb,
                                  compact_value_scale(index, index->q8_scales[b]), index->dim);
    }
    return (e5m2_storage(index)
                ? index->e5m2_kernels->dot_e5m2(va, vb, index->dim)
                : dot_q8_q8_query_aligned(va, vb, index->dim, index->compact_vector_sums[b])) *
           scale_a * scale_b;
  }
  const float* va = vector_at(index, a);
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return dot_f32(va, vector_at(index, b), index->dim);
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
    return l2_score_f32(va, vector_at(index, b), index->dim);
  }
  return dot_f32(va, vector_at(index, b), index->dim);
}

void memory_search_flat(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                        AstralMemorySearchResult* out_results, uint32_t* out_count);
void memory_search_flat_batch(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* queries, uint32_t query_count,
                              AstralMemorySearchResult* out_results, uint32_t* out_counts);
void memory_search_graph_with_scratch(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                      const float* query, AstralMemorySearchResult* out_results,
                                      uint32_t* out_count, GraphSearchScratch* scratch);
bool memory_search_graph_batch_parallel(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                        const float* queries, uint32_t query_count,
                                        AstralMemorySearchResult* out_results,
                                        uint32_t* out_counts);
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
void graph_begin_visit(MemoryIndex* index);
void graph_add_candidate(MemoryIndex* index, uint32_t capacity, uint32_t slot, float score,
                         uint32_t* candidate_count);
bool graph_pop_candidate(MemoryIndex* index, uint32_t* candidate_count, uint32_t* out_slot,
                         float* out_score);
inline void graph_mark_visited(MemoryIndex* index, uint32_t slot);
inline bool graph_was_visited(const MemoryIndex* index, uint32_t slot);
void graph_query_begin_visit(const MemoryIndex* index, GraphSearchScratch* scratch);
void graph_query_add_candidate(MemoryIndex* index, GraphSearchScratch* scratch, uint32_t capacity,
                               uint32_t slot, float score, uint32_t* candidate_count);
bool graph_query_pop_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                               uint32_t* candidate_count, uint32_t* out_slot, float* out_score);
inline void graph_query_mark_visited(GraphSearchScratch* scratch, uint32_t slot);
inline bool graph_query_was_visited(const GraphSearchScratch* scratch, uint32_t slot);
void insert_graph_build_candidate(MemoryIndex* index, uint32_t capacity, uint32_t* filled,
                                  uint32_t slot, float score);
bool insert_graph_query_top_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                                      uint32_t capacity, uint32_t* filled, uint32_t slot,
                                      float score);
bool graph_neighbor_diverse(MemoryIndex* index, uint32_t candidate_slot, float candidate_score,
                            const uint32_t* neighbors, uint32_t count);
bool graph_neighbor_diverse_except(MemoryIndex* index, uint32_t candidate_slot,
                                   float candidate_score, const uint32_t* neighbors, uint32_t count,
                                   uint32_t except_slot);
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

void insert_result(AstralMemorySearchResult* results, uint32_t top_k, uint32_t* filled,
                   const AstralMemorySearchResult& candidate) {
  uint32_t pos = *filled;
  if (pos < top_k) {
    ++(*filled);
  } else if (top_k != 0 && !result_better(candidate, results[top_k - 1u])) {
    return;
  } else {
    pos = top_k - 1u;
  }

  while (pos > 0 && result_better(candidate, results[pos - 1u])) {
    results[pos] = results[pos - 1u];
    --pos;
  }
  results[pos] = candidate;
}

void refine_graph_neighbor_list(MemoryIndex* index, uint32_t owner_slot, uint32_t candidate_slot,
                                float candidate_score, uint32_t level) {
  if (owner_slot == candidate_slot) {
    return;
  }

  uint32_t* neighbors = graph_neighbors_at_level(index, owner_slot, level);
  uint8_t& count = graph_neighbor_count_ref(index, owner_slot, level);
  const uint32_t capacity = graph_neighbor_capacity_at_level(index, level);
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == candidate_slot) {
      return;
    }
  }
  if (count < capacity) {
    neighbors[count] = candidate_slot;
    ++count;
    return;
  }

  uint32_t weakest_pos = 0;
  uint32_t weakest_slot = neighbors[weakest_pos];
  float weakest_score = score_pair(index, owner_slot, weakest_slot);
  for (uint32_t i = 1; i < count; ++i) {
    const uint32_t neighbor = neighbors[i];
    const float score = score_pair(index, owner_slot, neighbor);
    if (graph_candidate_worse(index, score, neighbor, weakest_score, weakest_slot)) {
      weakest_pos = i;
      weakest_score = score;
      weakest_slot = neighbor;
    }
  }
  if (!graph_candidate_better(index, candidate_score, candidate_slot, weakest_score,
                              weakest_slot)) {
    return;
  }
  if (!graph_neighbor_diverse_except(index, candidate_slot, candidate_score, neighbors, count,
                                     weakest_slot)) {
    return;
  }
  neighbors[weakest_pos] = candidate_slot;
}

void force_graph_neighbor(MemoryIndex* index, uint32_t owner_slot, uint32_t neighbor_slot,
                          uint32_t position, uint32_t level) {
  if (owner_slot == neighbor_slot || position >= graph_neighbor_capacity_at_level(index, level)) {
    return;
  }
  uint32_t* neighbors = graph_neighbors_at_level(index, owner_slot, level);
  uint8_t& count = graph_neighbor_count_ref(index, owner_slot, level);
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == neighbor_slot) {
      return;
    }
  }
  if (position >= count) {
    neighbors[count] = neighbor_slot;
    ++count;
    return;
  }
  neighbors[position] = neighbor_slot;
}

void insert_graph_build_candidate(MemoryIndex* index, uint32_t capacity, uint32_t* filled,
                                  uint32_t slot, float score) {
  uint32_t pos = *filled;
  if (pos < capacity) {
    ++(*filled);
  } else if (score <= index->graph_scratch_scores[capacity - 1u]) {
    return;
  } else {
    pos = capacity - 1u;
  }

  while (pos > 0 && score > index->graph_scratch_scores[pos - 1u]) {
    index->graph_scratch_scores[pos] = index->graph_scratch_scores[pos - 1u];
    index->graph_scratch_slots[pos] = index->graph_scratch_slots[pos - 1u];
    --pos;
  }
  index->graph_scratch_scores[pos] = score;
  index->graph_scratch_slots[pos] = slot;
}

bool graph_neighbor_selected(const uint32_t* neighbors, uint32_t count, uint32_t slot) {
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == slot) {
      return true;
    }
  }
  return false;
}

bool graph_neighbor_diverse(MemoryIndex* index, uint32_t candidate_slot, float candidate_score,
                            const uint32_t* neighbors, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    if (score_pair(index, candidate_slot, neighbors[i]) > candidate_score) {
      return false;
    }
  }
  return true;
}

bool graph_neighbor_diverse_except(MemoryIndex* index, uint32_t candidate_slot,
                                   float candidate_score, const uint32_t* neighbors, uint32_t count,
                                   uint32_t except_slot) {
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == except_slot) {
      continue;
    }
    if (score_pair(index, candidate_slot, neighbors[i]) > candidate_score) {
      return false;
    }
  }
  return true;
}

void graph_select_neighbors(MemoryIndex* index, uint32_t owner_slot, uint32_t level,
                            uint32_t candidate_count, uint32_t* neighbors, float* selected_scores,
                            uint32_t* out_count, uint32_t selection_capacity) {
  (void)level;
  uint32_t selected = 0;
  const uint32_t capacity = selection_capacity;
  for (uint32_t i = 0; i < candidate_count && selected < capacity; ++i) {
    const uint32_t candidate = index->graph_scratch_slots[i];
    const float candidate_score = index->graph_scratch_scores[i];
    if (candidate == owner_slot) {
      continue;
    }
    if (graph_neighbor_diverse(index, candidate, candidate_score, neighbors, selected)) {
      neighbors[selected] = candidate;
      selected_scores[selected] = candidate_score;
      ++selected;
    }
  }

  for (uint32_t i = 0; i < candidate_count && selected < capacity; ++i) {
    const uint32_t candidate = index->graph_scratch_slots[i];
    if (candidate == owner_slot || graph_neighbor_selected(neighbors, selected, candidate)) {
      continue;
    }
    neighbors[selected] = candidate;
    selected_scores[selected] = index->graph_scratch_scores[i];
    ++selected;
  }
  *out_count = selected;
}

bool insert_graph_query_top_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                                      uint32_t capacity, uint32_t* filled, uint32_t slot,
                                      float score) {
  uint32_t count = *filled;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, scratch->top_scores[parent], scratch->top_slots[parent],
                                  score, slot)) {
        break;
      }
      scratch->top_scores[pos] = scratch->top_scores[parent];
      scratch->top_slots[pos] = scratch->top_slots[parent];
      pos = parent;
    }
    scratch->top_scores[pos] = score;
    scratch->top_slots[pos] = slot;
    *filled = count + 1u;
    return true;
  }

  if (!graph_candidate_better(index, score, slot, scratch->top_scores[0], scratch->top_slots[0])) {
    return false;
  }

  uint32_t pos = 0;
  for (;;) {
    const uint32_t left = (pos << 1u) + 1u;
    if (left >= count) {
      break;
    }
    const uint32_t right = left + 1u;
    uint32_t child = left;
    if (right < count &&
        graph_candidate_worse(index, scratch->top_scores[right], scratch->top_slots[right],
                              scratch->top_scores[left], scratch->top_slots[left])) {
      child = right;
    }
    if (!graph_candidate_worse(index, scratch->top_scores[child], scratch->top_slots[child], score,
                               slot)) {
      break;
    }
    scratch->top_scores[pos] = scratch->top_scores[child];
    scratch->top_slots[pos] = scratch->top_slots[child];
    pos = child;
  }
  scratch->top_scores[pos] = score;
  scratch->top_slots[pos] = slot;
  return true;
}

void remove_graph_query_worst_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                                        uint32_t* filled) {
  uint32_t count = *filled;
  if (count == 0) {
    return;
  }

  --count;
  if (count != 0) {
    const uint32_t slot = scratch->top_slots[count];
    const float score = scratch->top_scores[count];
    uint32_t pos = 0;
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count &&
          graph_candidate_worse(index, scratch->top_scores[right], scratch->top_slots[right],
                                scratch->top_scores[left], scratch->top_slots[left])) {
        child = right;
      }
      if (!graph_candidate_worse(index, scratch->top_scores[child], scratch->top_slots[child],
                                 score, slot)) {
        break;
      }
      scratch->top_scores[pos] = scratch->top_scores[child];
      scratch->top_slots[pos] = scratch->top_slots[child];
      pos = child;
    }
    scratch->top_scores[pos] = score;
    scratch->top_slots[pos] = slot;
  }
  *filled = count;
}

uint32_t graph_f32_rerank_capacity(uint32_t top_k, uint32_t top_count) {
  uint32_t requested = kGraphF32RerankMinCandidates;
  if (top_k <= kU32Max / kGraphF32RerankTopKMultiplier) {
    const uint32_t top_k_capacity = top_k * kGraphF32RerankTopKMultiplier;
    if (top_k_capacity > requested) {
      requested = top_k_capacity;
    }
  }
  return requested < top_count ? requested : top_count;
}

void trim_graph_query_top_candidates(MemoryIndex* index, GraphSearchScratch* scratch,
                                     uint32_t* top_count, uint32_t target_count) {
  while (*top_count > target_count) {
    remove_graph_query_worst_candidate(index, scratch, top_count);
  }
}

void graph_collect_neighbors_exact(MemoryIndex* index, uint32_t slot, uint32_t level,
                                   uint32_t* filled) {
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t other = active_slot_at(index, active_pos);
    if (other == slot || index->slots[other].occupied == 0 || index->graph_levels[other] < level) {
      continue;
    }
    ++index->graph_build_candidate_visits;
    insert_graph_build_candidate(index, index->graph_scratch_capacity, filled, other,
                                 score_pair(index, slot, other));
  }
}

void graph_search_layer_pair(MemoryIndex* index, uint32_t slot, uint32_t level, uint32_t entry,
                             uint32_t capacity, uint32_t* filled) {
  if (entry == kU32Max || entry == slot || index->slots[entry].occupied == 0 ||
      index->graph_levels[entry] < level) {
    return;
  }

  graph_begin_visit(index);
  uint32_t candidate_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_mark_visited(index, entry);
  const float entry_score = score_pair(index, slot, entry);
  insert_graph_build_candidate(index, capacity, filled, entry, entry_score);
  graph_add_candidate(index, candidate_capacity, entry, entry_score, &candidate_count);

  while (candidate_count != 0) {
    uint32_t current = kU32Max;
    float current_score = kWorstScore;
    if (!graph_pop_candidate(index, &candidate_count, &current, &current_score)) {
      break;
    }
    if (*filled == capacity && !graph_candidate_better(index, current_score, current,
                                                       index->graph_scratch_scores[capacity - 1u],
                                                       index->graph_scratch_slots[capacity - 1u])) {
      break;
    }

    ++index->graph_build_candidate_visits;

    const uint32_t* neighbors = graph_neighbors_at_level(index, current, level);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, current, level);
    for (uint32_t i = 0; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (neighbor == slot || graph_was_visited(index, neighbor) ||
          index->slots[neighbor].occupied == 0 || index->graph_levels[neighbor] < level) {
        continue;
      }
      graph_mark_visited(index, neighbor);
      const float score = score_pair(index, slot, neighbor);
      if (*filled < capacity ||
          graph_candidate_better(index, score, neighbor, index->graph_scratch_scores[capacity - 1u],
                                 index->graph_scratch_slots[capacity - 1u])) {
        graph_add_candidate(index, candidate_capacity, neighbor, score, &candidate_count);
        insert_graph_build_candidate(index, capacity, filled, neighbor, score);
      }
    }
  }
}

uint32_t graph_greedy_closest_pair(MemoryIndex* index, uint32_t slot, uint32_t entry,
                                   uint32_t begin_level, uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score = score_pair(index, slot, closest);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = graph_neighbors_at_level(index, closest, level);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = neighbors[i];
        if (index->slots[candidate].occupied == 0 || index->graph_levels[candidate] < level) {
          continue;
        }
        const float score = score_pair(index, slot, candidate);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

void graph_search_layer_query(MemoryIndex* index, GraphSearchScratch* scratch, const float* query,
                              float query_scale, uint32_t entry, uint32_t capacity,
                              uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_query_mark_visited(scratch, entry);
  const float entry_score = score_slot(index, query, entry, query_scale);
  insert_graph_query_top_candidate(index, scratch, capacity, &top_count, entry, entry_score);
  graph_query_add_candidate(index, scratch, candidate_capacity, entry, entry_score,
                            &candidate_count);

  while (candidate_count != 0) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!graph_query_pop_candidate(index, scratch, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == capacity &&
        !graph_candidate_better(index, slot_score, slot, scratch->top_scores[0],
                                scratch->top_slots[0])) {
      break;
    }

    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
    const uint32_t prefetch_limit = neighbor_count > kGraphNeighborPrefetchDistance
                                        ? neighbor_count - kGraphNeighborPrefetchDistance
                                        : 0u;
    uint32_t i = 0;
    for (; i < prefetch_limit; ++i) {
      const uint32_t neighbor = neighbors[i];
      prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score = score_slot(index, query, neighbor, query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
    for (; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score = score_slot(index, query, neighbor, query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
  }

  *out_top_count = top_count;
}

template <CompactByteQueryStorage Storage>
void graph_search_layer_compact_query(MemoryIndex* index, GraphSearchScratch* scratch,
                                      const int8_t* query, int32_t query_sum, float query_scale,
                                      float cosine_query_scale, uint32_t entry, uint32_t capacity,
                                      uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_query_mark_visited(scratch, entry);
  const float entry_score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                              entry, cosine_query_scale);
  insert_graph_query_top_candidate(index, scratch, capacity, &top_count, entry, entry_score);
  graph_query_add_candidate(index, scratch, candidate_capacity, entry, entry_score,
                            &candidate_count);

  while (candidate_count != 0) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!graph_query_pop_candidate(index, scratch, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == capacity &&
        !graph_candidate_better(index, slot_score, slot, scratch->top_scores[0],
                                scratch->top_slots[0])) {
      break;
    }

    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
    const uint32_t prefetch_limit = neighbor_count > kGraphNeighborPrefetchDistance
                                        ? neighbor_count - kGraphNeighborPrefetchDistance
                                        : 0u;
    uint32_t i = 0;
    for (; i < prefetch_limit; ++i) {
      const uint32_t neighbor = neighbors[i];
      prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                            neighbor, cosine_query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
    for (; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                            neighbor, cosine_query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
  }

  *out_top_count = top_count;
}

void graph_search_layer_compact_query_i16(MemoryIndex* index, GraphSearchScratch* scratch,
                                          const int16_t* query, float query_scale,
                                          float cosine_query_scale, uint32_t entry,
                                          uint32_t capacity, uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_query_mark_visited(scratch, entry);
  const float entry_score =
      score_slot_compact_query_i16(index, query, query_scale, entry, cosine_query_scale);
  insert_graph_query_top_candidate(index, scratch, capacity, &top_count, entry, entry_score);
  graph_query_add_candidate(index, scratch, candidate_capacity, entry, entry_score,
                            &candidate_count);

  while (candidate_count != 0) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!graph_query_pop_candidate(index, scratch, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == capacity &&
        !graph_candidate_better(index, slot_score, slot, scratch->top_scores[0],
                                scratch->top_slots[0])) {
      break;
    }

    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
    const uint32_t prefetch_limit = neighbor_count > kGraphNeighborPrefetchDistance
                                        ? neighbor_count - kGraphNeighborPrefetchDistance
                                        : 0u;
    uint32_t i = 0;
    for (; i < prefetch_limit; ++i) {
      const uint32_t neighbor = neighbors[i];
      prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score =
          score_slot_compact_query_i16(index, query, query_scale, neighbor, cosine_query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
    for (; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score =
          score_slot_compact_query_i16(index, query, query_scale, neighbor, cosine_query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
  }

  *out_top_count = top_count;
}

uint32_t graph_greedy_closest_query(MemoryIndex* index, const float* query, float query_scale,
                                    uint32_t entry, uint32_t begin_level, uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score = score_slot(index, query, closest, query_scale);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = graph_neighbors_at_level(index, closest, level);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = neighbors[i];
        if (index->slots[candidate].occupied == 0 || index->graph_levels[candidate] < level) {
          continue;
        }
        const float score = score_slot(index, query, candidate, query_scale);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

uint32_t graph_greedy_closest_compact_query_i16(MemoryIndex* index, const int16_t* query,
                                                float query_scale, float cosine_query_scale,
                                                uint32_t entry, uint32_t begin_level,
                                                uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score =
      score_slot_compact_query_i16(index, query, query_scale, closest, cosine_query_scale);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = graph_neighbors_at_level(index, closest, level);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = neighbors[i];
        if (index->slots[candidate].occupied == 0 || index->graph_levels[candidate] < level) {
          continue;
        }
        const float score =
            score_slot_compact_query_i16(index, query, query_scale, candidate, cosine_query_scale);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

template <CompactByteQueryStorage Storage>
uint32_t graph_greedy_closest_compact_query(MemoryIndex* index, const int8_t* query,
                                            int32_t query_sum, float query_scale,
                                            float cosine_query_scale, uint32_t entry,
                                            uint32_t begin_level, uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                          closest, cosine_query_scale);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = graph_neighbors_at_level(index, closest, level);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = neighbors[i];
        if (index->slots[candidate].occupied == 0 || index->graph_levels[candidate] < level) {
          continue;
        }
        const float score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                              candidate, cosine_query_scale);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

void graph_connect_slot(MemoryIndex* index, uint32_t slot) {
  if (!graph_enabled(index) || index->count <= 1u) {
    index->graph_entry_slot = slot;
    index->graph_max_level = graph_enabled(index) ? index->graph_levels[slot] : 0;
    return;
  }

  uint32_t entry = index->graph_entry_slot;
  const uint32_t slot_level = index->graph_levels[slot];
  if (entry == kU32Max || index->slots[entry].occupied == 0) {
    entry = active_slot_at(index, 0);
  }
  if (index->graph_max_level > slot_level) {
    entry = graph_greedy_closest_pair(index, slot, entry, index->graph_max_level, slot_level);
  }

  uint32_t level = slot_level < index->graph_max_level ? slot_level : index->graph_max_level;
  for (;;) {
    uint32_t candidate_count = 0;
    if (index->count <= index->graph_search_capacity) {
      graph_collect_neighbors_exact(index, slot, level, &candidate_count);
    } else {
      graph_search_layer_pair(index, slot, level, entry, index->graph_search_capacity,
                              &candidate_count);
    }

    const uint32_t next_entry = candidate_count != 0 ? index->graph_scratch_slots[0] : entry;
    uint32_t* neighbors = graph_neighbors_at_level(index, slot, level);
    uint32_t filled = 0;
    const uint32_t level_capacity = graph_neighbor_capacity_at_level(index, level);
    const uint32_t outgoing_capacity = graph_outgoing_capacity_at_level(index, level);
    float selected_scores[kGraphMaxBaseNeighbors];
    graph_select_neighbors(index, slot, level, candidate_count, neighbors, selected_scores, &filled,
                           outgoing_capacity);
    graph_neighbor_count_ref(index, slot, level) = static_cast<uint8_t>(filled);
    for (uint32_t i = 0; i < filled; ++i) {
      refine_graph_neighbor_list(index, neighbors[i], slot, selected_scores[i], level);
    }
    if (level == 0 && kGraphLongLinkCount != 0 &&
        level_capacity == index->graph_neighbor_capacity && index->count > level_capacity &&
        level_capacity > kGraphLongLinkCount) {
      const uint32_t long_links =
          kGraphLongLinkCount < level_capacity ? kGraphLongLinkCount : level_capacity;
      const uint32_t stride = index->count / long_links;
      const uint32_t first_long_pos = level_capacity - long_links;
      for (uint32_t i = 0; i < long_links; ++i) {
        const uint32_t linked = active_slot_at(index, i * stride);
        force_graph_neighbor(index, slot, linked, first_long_pos + i, level);
        force_graph_neighbor(index, linked, slot, first_long_pos + i, level);
      }
    }
    if (candidate_count != 0) {
      entry = next_entry;
    }
    if (level == 0) {
      break;
    }
    --level;
  }
  if (slot_level > index->graph_max_level) {
    index->graph_max_level = slot_level;
    index->graph_entry_slot = slot;
  }
}

void graph_rebuild(MemoryIndex* index) {
  if (!graph_enabled(index)) {
    return;
  }
  std::memset(index->graph_neighbor_counts, 0,
              sizeof(uint8_t) * index->capacity * index->graph_level_capacity);
  index->graph_entry_slot = kU32Max;
  index->graph_max_level = 0;
  index->graph_build_score_evals = 0;
  index->graph_build_candidate_visits = 0;
  const uint32_t final_count = index->count;
  index->count = 0;
  for (uint32_t active_pos = 0; active_pos < final_count; ++active_pos) {
    ++index->count;
    graph_connect_slot(index, active_slot_at(index, active_pos));
  }
  index->count = final_count;
}

inline void graph_mark_visited(MemoryIndex* index, uint32_t slot) {
  index->graph_visited[slot] = index->graph_visit_generation;
}

inline bool graph_was_visited(const MemoryIndex* index, uint32_t slot) {
  return index->graph_visited[slot] == index->graph_visit_generation;
}

void graph_begin_visit(MemoryIndex* index) {
  ++index->graph_visit_generation;
  index->graph_candidate_worst_valid = 0;
  if (index->graph_visit_generation == 0) {
    std::memset(index->graph_visited, 0, sizeof(uint16_t) * index->capacity);
    index->graph_visit_generation = kGraphVisitGenerationStart;
  }
}

void graph_refresh_worst_candidate(MemoryIndex* index, uint32_t count) {
  uint32_t worst_pos = 0;
  float worst_score = index->graph_candidate_scores[0];
  for (uint32_t i = 1; i < count; ++i) {
    if (graph_candidate_worse(index, index->graph_candidate_scores[i], index->graph_candidates[i],
                              worst_score, index->graph_candidates[worst_pos])) {
      worst_score = index->graph_candidate_scores[i];
      worst_pos = i;
    }
  }
  index->graph_candidate_worst_pos = worst_pos;
  index->graph_candidate_worst_slot = index->graph_candidates[worst_pos];
  index->graph_candidate_worst_score = worst_score;
  index->graph_candidate_worst_valid = 1;
}

void graph_add_candidate(MemoryIndex* index, uint32_t capacity, uint32_t slot, float score,
                         uint32_t* candidate_count) {
  uint32_t count = *candidate_count;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, score, slot, index->graph_candidate_scores[parent],
                                  index->graph_candidates[parent])) {
        break;
      }
      index->graph_candidates[pos] = index->graph_candidates[parent];
      index->graph_candidate_scores[pos] = index->graph_candidate_scores[parent];
      pos = parent;
    }
    index->graph_candidates[pos] = slot;
    index->graph_candidate_scores[pos] = score;
    *candidate_count = count + 1u;
    return;
  }

  if (index->graph_candidate_worst_valid == 0) {
    graph_refresh_worst_candidate(index, count);
  }
  if (graph_candidate_better(index, score, slot, index->graph_candidate_worst_score,
                             index->graph_candidate_worst_slot)) {
    uint32_t pos = index->graph_candidate_worst_pos;
    bool moved_up = false;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, score, slot, index->graph_candidate_scores[parent],
                                  index->graph_candidates[parent])) {
        break;
      }
      index->graph_candidates[pos] = index->graph_candidates[parent];
      index->graph_candidate_scores[pos] = index->graph_candidate_scores[parent];
      pos = parent;
      moved_up = true;
    }
    if (!moved_up) {
      for (;;) {
        const uint32_t left = (pos << 1u) + 1u;
        if (left >= count) {
          break;
        }
        const uint32_t right = left + 1u;
        uint32_t child = left;
        if (right < count && graph_candidate_better(index, index->graph_candidate_scores[right],
                                                    index->graph_candidates[right],
                                                    index->graph_candidate_scores[left],
                                                    index->graph_candidates[left])) {
          child = right;
        }
        if (!graph_candidate_better(index, index->graph_candidate_scores[child],
                                    index->graph_candidates[child], score, slot)) {
          break;
        }
        index->graph_candidates[pos] = index->graph_candidates[child];
        index->graph_candidate_scores[pos] = index->graph_candidate_scores[child];
        pos = child;
      }
    }
    index->graph_candidates[pos] = slot;
    index->graph_candidate_scores[pos] = score;
    index->graph_candidate_worst_valid = 0;
  }
}

bool graph_pop_candidate(MemoryIndex* index, uint32_t* candidate_count, uint32_t* out_slot,
                         float* out_score) {
  uint32_t count = *candidate_count;
  if (count == 0) {
    return false;
  }

  *out_slot = index->graph_candidates[0];
  *out_score = index->graph_candidate_scores[0];
  --count;
  if (count != 0) {
    const uint32_t slot = index->graph_candidates[count];
    const float score = index->graph_candidate_scores[count];
    uint32_t pos = 0;
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count && graph_candidate_better(index, index->graph_candidate_scores[right],
                                                  index->graph_candidates[right],
                                                  index->graph_candidate_scores[left],
                                                  index->graph_candidates[left])) {
        child = right;
      }
      if (!graph_candidate_better(index, index->graph_candidate_scores[child],
                                  index->graph_candidates[child], score, slot)) {
        break;
      }
      index->graph_candidates[pos] = index->graph_candidates[child];
      index->graph_candidate_scores[pos] = index->graph_candidate_scores[child];
      pos = child;
    }
    index->graph_candidates[pos] = slot;
    index->graph_candidate_scores[pos] = score;
  }
  *candidate_count = count;
  index->graph_candidate_worst_valid = 0;
  return true;
}

inline void graph_query_mark_visited(GraphSearchScratch* scratch, uint32_t slot) {
  scratch->visited[slot] = scratch->visit_generation;
}

inline bool graph_query_was_visited(const GraphSearchScratch* scratch, uint32_t slot) {
  return scratch->visited[slot] == scratch->visit_generation;
}

void graph_query_begin_visit(const MemoryIndex* index, GraphSearchScratch* scratch) {
  ++scratch->visit_generation;
  scratch->candidate_worst_valid = 0;
  if (scratch->visit_generation == 0) {
    std::memset(scratch->visited, 0, sizeof(uint16_t) * index->capacity);
    scratch->visit_generation = kGraphVisitGenerationStart;
  }
}

void graph_query_refresh_worst_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                                         uint32_t count) {
  uint32_t worst_pos = 0;
  float worst_score = scratch->candidate_scores[0];
  for (uint32_t i = 1; i < count; ++i) {
    if (graph_candidate_worse(index, scratch->candidate_scores[i], scratch->candidates[i],
                              worst_score, scratch->candidates[worst_pos])) {
      worst_score = scratch->candidate_scores[i];
      worst_pos = i;
    }
  }
  scratch->candidate_worst_pos = worst_pos;
  scratch->candidate_worst_slot = scratch->candidates[worst_pos];
  scratch->candidate_worst_score = worst_score;
  scratch->candidate_worst_valid = 1;
}

void graph_query_add_candidate(MemoryIndex* index, GraphSearchScratch* scratch, uint32_t capacity,
                               uint32_t slot, float score, uint32_t* candidate_count) {
  uint32_t count = *candidate_count;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, score, slot, scratch->candidate_scores[parent],
                                  scratch->candidates[parent])) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[parent];
      scratch->candidate_scores[pos] = scratch->candidate_scores[parent];
      pos = parent;
    }
    scratch->candidates[pos] = slot;
    scratch->candidate_scores[pos] = score;
    *candidate_count = count + 1u;
    return;
  }

  if (scratch->candidate_worst_valid == 0) {
    graph_query_refresh_worst_candidate(index, scratch, count);
  }
  if (graph_candidate_better(index, score, slot, scratch->candidate_worst_score,
                             scratch->candidate_worst_slot)) {
    const uint32_t worst_pos = scratch->candidate_worst_pos;
    uint32_t pos = worst_pos;
    bool moved_up = false;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, score, slot, scratch->candidate_scores[parent],
                                  scratch->candidates[parent])) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[parent];
      scratch->candidate_scores[pos] = scratch->candidate_scores[parent];
      pos = parent;
      moved_up = true;
    }
    if (!moved_up) {
      for (;;) {
        const uint32_t left = (pos << 1u) + 1u;
        if (left >= count) {
          break;
        }
        const uint32_t right = left + 1u;
        uint32_t child = left;
        if (right < count &&
            graph_candidate_better(index, scratch->candidate_scores[right],
                                   scratch->candidates[right], scratch->candidate_scores[left],
                                   scratch->candidates[left])) {
          child = right;
        }
        if (!graph_candidate_better(index, scratch->candidate_scores[child],
                                    scratch->candidates[child], score, slot)) {
          break;
        }
        scratch->candidates[pos] = scratch->candidates[child];
        scratch->candidate_scores[pos] = scratch->candidate_scores[child];
        pos = child;
      }
    }
    scratch->candidates[pos] = slot;
    scratch->candidate_scores[pos] = score;
    scratch->candidate_worst_valid = 0;
  }
}

bool graph_query_pop_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                               uint32_t* candidate_count, uint32_t* out_slot, float* out_score) {
  uint32_t count = *candidate_count;
  if (count == 0) {
    return false;
  }

  *out_slot = scratch->candidates[0];
  *out_score = scratch->candidate_scores[0];
  --count;
  if (count != 0) {
    const uint32_t slot = scratch->candidates[count];
    const float score = scratch->candidate_scores[count];
    uint32_t pos = 0;
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count && graph_candidate_better(
                               index, scratch->candidate_scores[right], scratch->candidates[right],
                               scratch->candidate_scores[left], scratch->candidates[left])) {
        child = right;
      }
      if (!graph_candidate_better(index, scratch->candidate_scores[child],
                                  scratch->candidates[child], score, slot)) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[child];
      scratch->candidate_scores[pos] = scratch->candidate_scores[child];
      pos = child;
    }
    scratch->candidates[pos] = slot;
    scratch->candidate_scores[pos] = score;
  }
  *candidate_count = count;
  scratch->candidate_worst_valid = 0;
  return true;
}

void memory_search_graph_with_scratch(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                      const float* query, AstralMemorySearchResult* out_results,
                                      uint32_t* out_count, GraphSearchScratch* scratch) {
  if (!graph_enabled(index) || index->count == 0 || desc->group_id != ASTRAL_MEMORY_GROUP_ANY ||
      index->graph_entry_slot == kU32Max || index->slots[index->graph_entry_slot].occupied == 0) {
    memory_search_flat(index, desc, query, out_results, out_count);
    return;
  }
  if (compact_graph_exact_search_preferred(index)) {
    memory_search_flat(index, desc, query, out_results, out_count);
    return;
  }

  alignas(kVectorStorageAlign) float normalized_query[kMaxDim];
  if (!compact_storage(index) && index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    normalize_f32_vector(normalized_query, query, index->dim);
    query = normalized_query;
  }
  const float query_scale = compact_storage(index) && index->metric == ASTRAL_MEMORY_METRIC_COSINE
                                ? cosine_scale(query, index->dim)
                                : 1.0f;
  graph_query_begin_visit(index, scratch);
  if (scratch->visited == index->graph_visited) {
    index->graph_visit_generation = scratch->visit_generation;
  }
  const uint32_t search_capacity = graph_search_for_query(index, desc);

  uint32_t filled = 0;
  uint32_t top_count = 0;
  if (compact_storage(index) && i16_storage(index)) {
    alignas(kVectorStorageAlign) int16_t compact_query[kMaxDim];
    float compact_query_scale = 1.0f;
    quantize_e3m2_vector(compact_query, &compact_query_scale, query, index->dim);
    compact_query_scale *= kE3M2InvScale;
    const uint32_t entry = index->graph_max_level != 0
                               ? graph_greedy_closest_compact_query_i16(
                                     index, compact_query, compact_query_scale, query_scale,
                                     index->graph_entry_slot, index->graph_max_level, 0)
                               : index->graph_entry_slot;
    graph_search_layer_compact_query_i16(index, scratch, compact_query, compact_query_scale,
                                         query_scale, entry, search_capacity, &top_count);
  } else if (compact_storage(index) && !i16_storage(index)) {
    alignas(kVectorStorageAlign) int8_t compact_query[kMaxDim];
    float compact_query_scale = 1.0f;
    float compact_cosine_query_scale = query_scale;
    quantize_compact_query(index, compact_query, &compact_query_scale, query);
    if (e5m2_storage(index)) {
      const uint32_t entry =
          index->graph_max_level != 0
              ? graph_greedy_closest_compact_query<CompactByteQueryStorage::e5m2>(
                    index, compact_query, 0, compact_query_scale, compact_cosine_query_scale,
                    index->graph_entry_slot, index->graph_max_level, 0)
              : index->graph_entry_slot;
      graph_search_layer_compact_query<CompactByteQueryStorage::e5m2>(
          index, scratch, compact_query, 0, compact_query_scale, compact_cosine_query_scale, entry,
          search_capacity, &top_count);
    } else {
      const int32_t compact_query_sum = sum_i8(compact_query, index->dim);
      const uint32_t entry =
          index->graph_max_level != 0
              ? graph_greedy_closest_compact_query<CompactByteQueryStorage::q8>(
                    index, compact_query, compact_query_sum, compact_query_scale,
                    compact_cosine_query_scale, index->graph_entry_slot, index->graph_max_level, 0)
              : index->graph_entry_slot;
      graph_search_layer_compact_query<CompactByteQueryStorage::q8>(
          index, scratch, compact_query, compact_query_sum, compact_query_scale,
          compact_cosine_query_scale, entry, search_capacity, &top_count);
    }
  } else {
    const uint32_t entry =
        index->graph_max_level != 0
            ? graph_greedy_closest_query(index, query, query_scale, index->graph_entry_slot,
                                         index->graph_max_level, 0)
            : index->graph_entry_slot;
    graph_search_layer_query(index, scratch, query, query_scale, entry, search_capacity,
                             &top_count);
  }

  if (f32_rerank_storage(index)) {
    const uint32_t rerank_capacity = graph_f32_rerank_capacity(desc->top_k, top_count);
    trim_graph_query_top_candidates(index, scratch, &top_count, rerank_capacity);
  }

  for (uint32_t i = 0; i < top_count; ++i) {
    const uint32_t slot = scratch->top_slots[i];
    const MemorySlot& s = index->slots[slot];
    const float score = compact_storage(index) ? score_slot(index, query, slot, query_scale)
                                               : scratch->top_scores[i];
    AstralMemorySearchResult candidate{};
    fill_result(&candidate, s, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  if (compact_storage(index) && !f32_rerank_storage(index)) {
    for (uint32_t i = 0; i < top_count; ++i) {
      const uint32_t slot = scratch->top_slots[i];
      const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
      for (uint32_t neighbor_i = 0; neighbor_i < neighbor_count; ++neighbor_i) {
        const uint32_t neighbor = neighbors[neighbor_i];
        if (graph_query_was_visited(scratch, neighbor)) {
          continue;
        }
        graph_query_mark_visited(scratch, neighbor);
        const MemorySlot& s = index->slots[neighbor];
        const float score = score_slot(index, query, neighbor, query_scale);
        AstralMemorySearchResult candidate{};
        fill_result(&candidate, s, score);
        insert_result(out_results, desc->top_k, &filled, candidate);
      }
    }
  }
  *out_count = filled;
}

void memory_search_graph(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                         AstralMemorySearchResult* out_results, uint32_t* out_count) {
  GraphSearchScratch scratch{index->graph_candidates,
                             index->graph_candidate_scores,
                             index->graph_scratch_slots,
                             index->graph_scratch_scores,
                             index->graph_visited,
                             index->graph_visit_generation,
                             0u,
                             kU32Max,
                             kWorstScore,
                             0u};
  memory_search_graph_with_scratch(index, desc, query, out_results, out_count, &scratch);
}

void graph_search_scratch_free(const MemoryIndex* index, GraphSearchScratch* scratch) {
  if (scratch->candidates != nullptr) {
    core::runtime_free_array(scratch->candidates, index->graph_candidate_capacity);
    scratch->candidates = nullptr;
  }
  if (scratch->candidate_scores != nullptr) {
    core::runtime_free_array(scratch->candidate_scores, index->graph_candidate_capacity);
    scratch->candidate_scores = nullptr;
  }
  if (scratch->top_slots != nullptr) {
    core::runtime_free_array(scratch->top_slots, index->graph_scratch_capacity);
    scratch->top_slots = nullptr;
  }
  if (scratch->top_scores != nullptr) {
    core::runtime_free_array(scratch->top_scores, index->graph_scratch_capacity);
    scratch->top_scores = nullptr;
  }
  if (scratch->visited != nullptr) {
    core::runtime_free_array(scratch->visited, index->capacity);
    scratch->visited = nullptr;
  }
  scratch->visit_generation = 0;
}

bool graph_search_scratch_alloc(const MemoryIndex* index, GraphSearchScratch* scratch) {
  *scratch = GraphSearchScratch{};
  scratch->candidates = core::runtime_alloc_array<uint32_t>(index->graph_candidate_capacity);
  scratch->candidate_scores = core::runtime_alloc_array<float>(index->graph_candidate_capacity);
  scratch->top_slots = core::runtime_alloc_array<uint32_t>(index->graph_scratch_capacity);
  scratch->top_scores = core::runtime_alloc_array<float>(index->graph_scratch_capacity);
  scratch->visited = core::runtime_alloc_array<uint16_t>(index->capacity);
  if (scratch->candidates == nullptr || scratch->candidate_scores == nullptr ||
      scratch->top_slots == nullptr || scratch->top_scores == nullptr ||
      scratch->visited == nullptr) {
    graph_search_scratch_free(index, scratch);
    return false;
  }
  std::memset(scratch->visited, 0, sizeof(uint16_t) * index->capacity);
  return true;
}

void snapshot_graph_scratch_free(const AstralMemorySnapshotInfo* info,
                                 const SnapshotGraphLayout* graph, GraphSearchScratch* scratch) {
  if (scratch->candidates != nullptr) {
    core::runtime_free_array(scratch->candidates, graph->candidate_capacity);
    scratch->candidates = nullptr;
  }
  if (scratch->candidate_scores != nullptr) {
    core::runtime_free_array(scratch->candidate_scores, graph->candidate_capacity);
    scratch->candidate_scores = nullptr;
  }
  if (scratch->top_slots != nullptr) {
    core::runtime_free_array(scratch->top_slots, graph->scratch_capacity);
    scratch->top_slots = nullptr;
  }
  if (scratch->top_scores != nullptr) {
    core::runtime_free_array(scratch->top_scores, graph->scratch_capacity);
    scratch->top_scores = nullptr;
  }
  if (scratch->visited != nullptr) {
    core::runtime_free_array(scratch->visited, info->count);
    scratch->visited = nullptr;
  }
  scratch->visit_generation = 0;
}

bool snapshot_graph_scratch_alloc(const AstralMemorySnapshotInfo* info,
                                  const SnapshotGraphLayout* graph, GraphSearchScratch* scratch) {
  *scratch = GraphSearchScratch{};
  scratch->candidates = core::runtime_alloc_array<uint32_t>(graph->candidate_capacity);
  scratch->candidate_scores = core::runtime_alloc_array<float>(graph->candidate_capacity);
  scratch->top_slots = core::runtime_alloc_array<uint32_t>(graph->scratch_capacity);
  scratch->top_scores = core::runtime_alloc_array<float>(graph->scratch_capacity);
  scratch->visited = core::runtime_alloc_array<uint16_t>(info->count);
  if (scratch->candidates == nullptr || scratch->candidate_scores == nullptr ||
      scratch->top_slots == nullptr || scratch->top_scores == nullptr ||
      scratch->visited == nullptr) {
    snapshot_graph_scratch_free(info, graph, scratch);
    return false;
  }
  std::memset(scratch->visited, 0, sizeof(uint16_t) * info->count);
  return true;
}

void memory_search_graph_batch_work(void* user) {
  MemoryGraphSearchBatchJob* job = static_cast<MemoryGraphSearchBatchJob*>(user);
  for (uint32_t i = job->begin; i < job->end; ++i) {
    uint32_t count = 0;
    const float* query = job->queries + static_cast<size_t>(i) * job->index->dim;
    AstralMemorySearchResult* results =
        job->out_results + static_cast<size_t>(i) * job->desc->top_k;
    memory_search_graph_with_scratch(job->index, job->desc, query, results, &count, job->scratch);
    job->out_counts[i] = count;
  }
  memory_parallel_job_complete(job->remaining);
}

bool memory_search_graph_batch_parallel(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                        const float* queries, uint32_t query_count,
                                        AstralMemorySearchResult* out_results,
                                        uint32_t* out_counts) {
  if (!graph_enabled(index) || desc->group_id != ASTRAL_MEMORY_GROUP_ANY ||
      query_count < kMemorySearchBatchParallelMinQueries || !core::runtime_initialized()) {
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

  MemoryGraphSearchBatchJob jobs[kMemorySearchBatchParallelMaxWorkers];
  GraphSearchScratch fallback_scratch[kMemorySearchBatchParallelMaxWorkers];
  bool claimed_index_scratch = false;
  if (index->graph_batch_scratch_count >= worker_count &&
      index->graph_batch_scratch_claimed.exchange(1u, std::memory_order_acquire) == 0u) {
    claimed_index_scratch = true;
    for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
      jobs[worker_i].scratch = &index->graph_batch_scratch[worker_i];
    }
  } else {
    uint32_t allocated = 0;
    for (; allocated < worker_count; ++allocated) {
      if (!graph_search_scratch_alloc(index, &fallback_scratch[allocated])) {
        for (uint32_t i = 0; i < allocated; ++i) {
          graph_search_scratch_free(index, &fallback_scratch[i]);
        }
        return false;
      }
      jobs[allocated].scratch = &fallback_scratch[allocated];
    }
  }

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
        core::submit_work_affine(target_worker, memory_search_graph_batch_work, &jobs[worker_i]);
    if (err != ASTRAL_OK) {
      memory_search_graph_batch_work(&jobs[worker_i]);
    }
  }

  while (remaining.load(std::memory_order_acquire) != 0) {
    astral::platform::cpu_wait_for_event();
  }

  for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
    if (!claimed_index_scratch) {
      graph_search_scratch_free(index, jobs[worker_i].scratch);
    }
  }
  if (claimed_index_scratch) {
    index->graph_batch_scratch_claimed.store(0u, std::memory_order_release);
  }
  return true;
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

AstralErr memory_save_size(MemoryIndex* index, uint64_t* out_bytes) {
  if (index == nullptr || out_bytes == nullptr) {
    return ASTRAL_E_INVALID;
  }
  *out_bytes = memory_save_byte_count(index);
  return ASTRAL_OK;
}

AstralErr memory_save(MemoryIndex* index, AstralMutSpanU8 out_bytes, uint64_t* out_written) {
  if (index == nullptr || out_written == nullptr) {
    return ASTRAL_E_INVALID;
  }

  uint64_t need = 0;
  AstralErr err = memory_save_size(index, &need);
  if (err != ASTRAL_OK) {
    return err;
  }
  *out_written = need;
  if (out_bytes.data == nullptr || out_bytes.len < need) {
    return ASTRAL_E_NOMEM;
  }
  SaveLayout layout{};
  if (!memory_save_layout(kSaveVersion, index->dim, index->count, index->storage_kind,
                          index->index_kind,
                          graph_enabled(index) ? index->graph_base_neighbor_capacity : 0u,
                          graph_enabled(index) ? index->graph_neighbor_capacity : 0u,
                          graph_enabled(index) ? index->graph_level_capacity : 0u, &layout)) {
    return ASTRAL_E_INVALID;
  }

  SaveHeader header{};
  header.magic = kSaveMagic;
  header.version = kSaveVersion;
  header.dim = index->dim;
  header.count = index->count;
  header.metric = index->metric;
  header.index_kind = index->index_kind;
  header._reserved0 = index->storage_kind;
  header._reserved1 = graph_enabled(index) ? kSaveGraphTopologyFlag : 0;

  uint8_t* cursor = out_bytes.data;
  std::memcpy(cursor, &header, sizeof(header));
  cursor += sizeof(header);

  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    std::memcpy(cursor, &index->slots[slot].record, sizeof(AstralMemoryRecord));
    cursor += sizeof(AstralMemoryRecord);
  }
  if (compact_storage(index)) {
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const float scale = index->q8_scales[slot];
      std::memcpy(cursor, &scale, sizeof(float));
      cursor += sizeof(float);
    }
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const float scale = index->compact_score_scales[slot];
      std::memcpy(cursor, &scale, sizeof(float));
      cursor += sizeof(float);
    }
    memory_save_skip_to(&cursor, out_bytes.data + layout.vector_offset);
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const size_t bytes =
          static_cast<size_t>(index->dim) * (i16_storage(index) ? sizeof(int16_t) : sizeof(int8_t));
      std::memcpy(cursor,
                  i16_storage(index) ? static_cast<const void*>(i16_vector_at(index, slot))
                                     : static_cast<const void*>(q8_vector_at(index, slot)),
                  bytes);
      cursor += bytes;
    }
    if (f32_rerank_storage(index)) {
      memory_save_skip_to(&cursor, out_bytes.data + layout.rerank_vector_offset);
      for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
        const uint32_t slot = active_slot_at(index, active_pos);
        std::memcpy(cursor, vector_at(index, slot), sizeof(float) * index->dim);
        cursor += sizeof(float) * index->dim;
      }
    }
  } else {
    memory_save_skip_to(&cursor, out_bytes.data + layout.vector_offset);
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      std::memcpy(cursor, vector_at(index, slot), sizeof(float) * index->dim);
      cursor += sizeof(float) * index->dim;
    }
  }

  memory_save_skip_to(&cursor, out_bytes.data + layout.graph_offset);
  if (graph_enabled(index)) {
    SaveGraphHeader graph_header{};
    graph_header.flags = kSaveGraphTopologyFlag;
    graph_header.neighbor_capacity = index->graph_neighbor_capacity;
    graph_header.base_neighbor_capacity = index->graph_base_neighbor_capacity;
    graph_header.search_capacity = index->graph_search_capacity;
    graph_header.query_search_capacity = index->graph_query_search_capacity;
    graph_header.level_capacity = index->graph_level_capacity;
    graph_header.max_level = index->graph_max_level;
    graph_header.entry_active_pos = index->graph_entry_slot != kU32Max
                                        ? index->slots[index->graph_entry_slot].active_pos
                                        : kU32Max;
    std::memcpy(cursor, &graph_header, sizeof(graph_header));
    cursor += sizeof(graph_header);

    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      *cursor = index->graph_levels[slot];
      ++cursor;
    }
    for (uint32_t level = 0; level < index->graph_level_capacity; ++level) {
      for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
        const uint32_t slot = active_slot_at(index, active_pos);
        const uint8_t count =
            static_cast<uint8_t>(graph_neighbor_count_at_level(index, slot, level));
        std::memcpy(cursor, &count, sizeof(count));
        cursor += sizeof(count);
      }
    }
    for (uint32_t level = 0; level < index->graph_level_capacity; ++level) {
      for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
        const uint32_t slot = active_slot_at(index, active_pos);
        const uint32_t* neighbors = graph_neighbors_at_level(index, slot, level);
        const uint32_t capacity = graph_neighbor_capacity_at_level(index, level);
        for (uint32_t i = 0; i < capacity; ++i) {
          const uint32_t neighbor = neighbors[i];
          const uint32_t neighbor_pos = neighbor != kU32Max && neighbor < index->capacity &&
                                                index->slots[neighbor].occupied != 0
                                            ? index->slots[neighbor].active_pos
                                            : kU32Max;
          std::memcpy(cursor, &neighbor_pos, sizeof(neighbor_pos));
          cursor += sizeof(neighbor_pos);
        }
      }
    }
  }
  return ASTRAL_OK;
}

namespace {

struct SnapshotBytes {
  const uint8_t* data;
  uint64_t len;
};

AstralErr memory_snapshot_info_bytes(SnapshotBytes bytes, AstralMemorySnapshotInfo* out_info) {
  if (bytes.data == nullptr || bytes.len < sizeof(SaveHeader) || out_info == nullptr ||
      out_info->size != sizeof(AstralMemorySnapshotInfo)) {
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
  AstralMemoryStorageKind storage = static_cast<AstralMemoryStorageKind>(ASTRAL_MEMORY_STORAGE_F32);
  if (header.version >= kSaveVersionCompactStorage) {
    storage = static_cast<AstralMemoryStorageKind>(header._reserved0);
  }
  if (header.magic != kSaveMagic || !version_valid || header.dim == 0 || header.dim > kMaxDim ||
      !metric_valid(static_cast<AstralMemoryMetric>(header.metric)) ||
      !index_kind_valid(static_cast<AstralMemoryIndexKind>(header.index_kind)) ||
      !storage_kind_valid(storage)) {
    return ASTRAL_E_INVALID;
  }

  SaveLayout layout{};
  if (!memory_save_layout(header.version, header.dim, header.count, storage, header.index_kind, 0,
                          0, 0, &layout) ||
      bytes.len < layout.graph_offset) {
    return ASTRAL_E_INVALID;
  }

  if (header.index_kind == ASTRAL_MEMORY_INDEX_GRAPH &&
      header._reserved1 == kSaveGraphTopologyFlag) {
    const uint32_t graph_header_bytes = save_graph_header_bytes(header.version);
    if (bytes.len < layout.graph_offset + graph_header_bytes) {
      return ASTRAL_E_INVALID;
    }
    SaveGraphHeader graph_header{};
    read_save_graph_header(header.version, bytes.data + layout.graph_offset, &graph_header);
    if (graph_header.flags != kSaveGraphTopologyFlag || graph_header.neighbor_capacity == 0 ||
        graph_header.neighbor_capacity > kGraphMaxNeighbors ||
        graph_header.base_neighbor_capacity == 0 ||
        graph_header.base_neighbor_capacity > kGraphMaxBaseNeighbors ||
        graph_header.base_neighbor_capacity < graph_header.neighbor_capacity ||
        graph_header.query_search_capacity == 0 || graph_header.level_capacity == 0 ||
        graph_header.level_capacity > kGraphMaxLevels) {
      return ASTRAL_E_INVALID;
    }
    if (!memory_save_layout(header.version, header.dim, header.count, storage, header.index_kind,
                            graph_header.base_neighbor_capacity, graph_header.neighbor_capacity,
                            graph_header.level_capacity, &layout) ||
        bytes.len < layout.total_bytes) {
      return ASTRAL_E_INVALID;
    }
  } else if (bytes.len < layout.total_bytes) {
    return ASTRAL_E_INVALID;
  }

  out_info->version = header.version;
  out_info->dim = header.dim;
  out_info->count = header.count;
  out_info->metric = static_cast<AstralMemoryMetric>(header.metric);
  out_info->index_kind = static_cast<AstralMemoryIndexKind>(header.index_kind);
  out_info->storage_kind = storage;
  out_info->flags = header._reserved1;
  out_info->record_offset = layout.record_offset;
  out_info->record_stride = layout.record_stride;
  out_info->vector_offset = layout.vector_offset;
  out_info->vector_stride = layout.vector_stride;
  out_info->scale_offset = layout.scale_offset;
  out_info->scale_stride = layout.scale_stride;
  out_info->graph_offset = layout.graph_offset;
  out_info->graph_bytes = layout.graph_bytes;
  out_info->total_bytes = layout.total_bytes;
  return ASTRAL_OK;
}

bool snapshot_graph_layout(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                           SnapshotGraphLayout* out_graph) {
  if (bytes.data == nullptr || info == nullptr || out_graph == nullptr ||
      info->index_kind != ASTRAL_MEMORY_INDEX_GRAPH || info->flags != kSaveGraphTopologyFlag ||
      info->graph_bytes == 0) {
    return false;
  }
  const uint32_t graph_header_bytes = save_graph_header_bytes(info->version);
  if (bytes.len < info->graph_offset + graph_header_bytes) {
    return false;
  }

  SaveGraphHeader graph_header{};
  read_save_graph_header(info->version, bytes.data + info->graph_offset, &graph_header);
  if (graph_header.flags != kSaveGraphTopologyFlag || graph_header.neighbor_capacity == 0 ||
      graph_header.neighbor_capacity > kGraphMaxNeighbors ||
      graph_header.base_neighbor_capacity == 0 ||
      graph_header.base_neighbor_capacity > kGraphMaxBaseNeighbors ||
      graph_header.base_neighbor_capacity < graph_header.neighbor_capacity ||
      graph_header.query_search_capacity == 0 || graph_header.level_capacity == 0 ||
      graph_header.level_capacity > kGraphMaxLevels ||
      graph_header.max_level >= graph_header.level_capacity ||
      graph_header.entry_active_pos >= info->count) {
    return false;
  }

  const uint64_t level_count = static_cast<uint64_t>(info->count) * graph_header.level_capacity;
  const uint64_t neighbor_slots =
      static_cast<uint64_t>(info->count) * graph_header.base_neighbor_capacity +
      static_cast<uint64_t>(info->count) * (graph_header.level_capacity - 1u) *
          graph_header.neighbor_capacity;
  const uint32_t graph_count_bytes = save_graph_count_bytes(info->version);
  const uint64_t graph_bytes = graph_header_bytes + static_cast<uint64_t>(info->count) +
                               level_count * graph_count_bytes + neighbor_slots * sizeof(uint32_t);
  if (bytes.len < info->graph_offset + graph_bytes) {
    return false;
  }

  SnapshotGraphLayout graph{};
  graph.header = graph_header;
  graph.levels_offset = info->graph_offset + graph_header_bytes;
  graph.counts_offset = graph.levels_offset + static_cast<uint64_t>(info->count);
  graph.neighbors_offset = graph.counts_offset + level_count * graph_count_bytes;
  for (uint32_t active_pos = 0; active_pos < info->count; ++active_pos) {
    if (bytes.data[graph.levels_offset + active_pos] >= graph_header.level_capacity) {
      return false;
    }
  }
  for (uint32_t level = 0; level < graph_header.level_capacity; ++level) {
    const uint32_t level_capacity =
        level == 0 ? graph_header.base_neighbor_capacity : graph_header.neighbor_capacity;
    for (uint32_t active_pos = 0; active_pos < info->count; ++active_pos) {
      uint32_t count = 0;
      const uint8_t* count_ptr =
          bytes.data + graph.counts_offset +
          (static_cast<uint64_t>(level) * info->count + active_pos) * graph_count_bytes;
      if (graph_count_bytes == sizeof(uint8_t)) {
        count = *count_ptr;
      } else {
        std::memcpy(&count, count_ptr, sizeof(count));
      }
      if (count > level_capacity) {
        return false;
      }
      const uint64_t neighbor_base =
          level == 0
              ? graph.neighbors_offset + static_cast<uint64_t>(active_pos) *
                                             graph_header.base_neighbor_capacity * sizeof(uint32_t)
              : graph.neighbors_offset +
                    (static_cast<uint64_t>(info->count) * graph_header.base_neighbor_capacity +
                     (static_cast<uint64_t>(level - 1u) * info->count + active_pos) *
                         graph_header.neighbor_capacity) *
                        sizeof(uint32_t);
      for (uint32_t neighbor_i = 0; neighbor_i < level_capacity; ++neighbor_i) {
        uint32_t neighbor = kU32Max;
        std::memcpy(&neighbor,
                    bytes.data + neighbor_base +
                        static_cast<uint64_t>(neighbor_i) * sizeof(uint32_t),
                    sizeof(neighbor));
        if (neighbor != kU32Max && neighbor >= info->count) {
          return false;
        }
      }
    }
  }
  uint32_t scratch_capacity = graph_header.query_search_capacity > graph_header.search_capacity
                                  ? graph_header.query_search_capacity
                                  : graph_header.search_capacity;
  if (scratch_capacity < kGraphMinSearch) {
    scratch_capacity = kGraphMinSearch;
  }
  if (scratch_capacity > info->count) {
    scratch_capacity = info->count;
  }
  uint32_t candidate_capacity = scratch_capacity;
  if (scratch_capacity <= kU32Max / kGraphCandidateReserveMultiplier) {
    candidate_capacity = scratch_capacity * kGraphCandidateReserveMultiplier;
  }
  if (candidate_capacity > info->count) {
    candidate_capacity = info->count;
  }
  graph.scratch_capacity = scratch_capacity;
  graph.candidate_capacity = candidate_capacity;
  *out_graph = graph;
  return true;
}

inline uint64_t snapshot_graph_level_offset(const AstralMemorySnapshotInfo* info,
                                            const SnapshotGraphLayout* graph, uint32_t active_pos,
                                            uint32_t level) {
  if (level == 0) {
    return graph->neighbors_offset + static_cast<uint64_t>(active_pos) *
                                         graph->header.base_neighbor_capacity * sizeof(uint32_t);
  }
  return graph->neighbors_offset +
         (static_cast<uint64_t>(info->count) * graph->header.base_neighbor_capacity +
          (static_cast<uint64_t>(level - 1u) * info->count + active_pos) *
              graph->header.neighbor_capacity) *
             sizeof(uint32_t);
}

inline uint8_t snapshot_graph_level(const uint8_t* bytes, const SnapshotGraphLayout* graph,
                                    uint32_t active_pos) {
  return bytes[graph->levels_offset + active_pos];
}

template <bool CompactCounts>
inline uint32_t
snapshot_graph_neighbor_count(const uint8_t* bytes, const AstralMemorySnapshotInfo* info,
                              const SnapshotGraphLayout* graph, uint32_t slot, uint32_t level) {
  const uint64_t ordinal = static_cast<uint64_t>(level) * info->count + slot;
  if constexpr (CompactCounts) {
    return bytes[graph->counts_offset + ordinal];
  } else {
    uint32_t count = 0;
    std::memcpy(&count, bytes + graph->counts_offset + ordinal * sizeof(uint32_t), sizeof(count));
    return count;
  }
}

inline uint32_t snapshot_graph_neighbor_at_base(const uint8_t* bytes, uint64_t neighbor_base,
                                                uint32_t neighbor_i) {
  uint32_t neighbor = kU32Max;
  std::memcpy(&neighbor,
              bytes + neighbor_base + static_cast<uint64_t>(neighbor_i) * sizeof(uint32_t),
              sizeof(neighbor));
  return neighbor;
}

struct SnapshotPreparedQuery {
  alignas(kVectorStorageAlign) float query[kMaxDim];
  alignas(kVectorStorageAlign) int8_t compact_query[kMaxDim];
  alignas(kVectorStorageAlign) int16_t compact_query_i16[kMaxDim];
  uint64_t score_vector_offset;
  uint64_t score_vector_stride;
  uint64_t compact_score_scale_offset;
  uint64_t compact_score_scale_stride;
  const E5m2Kernels* e5m2_kernels;
  float query_scale;
  float compact_query_scale;
  int32_t compact_query_sum;
  uint8_t compact;
  uint8_t use_f6;
  uint8_t use_f6_i16;
  uint8_t use_f8;
  uint8_t compact_i8_vectors_aligned;
  uint8_t compact_i16_vectors_aligned;
  uint8_t normalized_f32_cosine;
  uint8_t e5m2_clamp_non_finite;
};

void snapshot_prepare_query(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                            const float* query, uint8_t e5m2_clamp_non_finite,
                            SnapshotPreparedQuery* prepared) {
  *prepared = {};
  prepared->e5m2_kernels = select_e5m2_kernels();
  prepared->e5m2_clamp_non_finite = e5m2_clamp_non_finite;
  prepared->score_vector_offset = info->vector_offset;
  prepared->score_vector_stride = info->vector_stride;
  if (f32_rerank_storage_kind(info->storage_kind)) {
    SaveLayout layout{};
    if (memory_save_layout(info->version, info->dim, info->count, info->storage_kind,
                           info->index_kind, 0, 0, 0, &layout)) {
      prepared->score_vector_offset = layout.rerank_vector_offset;
      prepared->score_vector_stride = layout.rerank_vector_stride;
    }
  }
  if (info->version >= kSaveVersionCompactScoreScales && compact_storage_kind(info->storage_kind)) {
    SaveLayout layout{};
    if (memory_save_layout(info->version, info->dim, info->count, info->storage_kind,
                           info->index_kind, 0, 0, 0, &layout)) {
      prepared->compact_score_scale_offset = layout.compact_score_scale_offset;
      prepared->compact_score_scale_stride = layout.compact_score_scale_stride;
    }
  }
  prepared->compact = compact_storage_kind(info->storage_kind) ? 1u : 0u;
  prepared->normalized_f32_cosine = !prepared->compact &&
                                            info->metric == ASTRAL_MEMORY_METRIC_COSINE &&
                                            info->version >= kSaveVersionNormalizedF32Cosine
                                        ? 1u
                                        : 0u;
  if (prepared->normalized_f32_cosine != 0) {
    normalize_f32_vector(prepared->query, query, info->dim);
  } else {
    std::memcpy(prepared->query, query, sizeof(float) * info->dim);
  }
  prepared->query_scale =
      info->metric == ASTRAL_MEMORY_METRIC_COSINE && prepared->normalized_f32_cosine == 0
          ? cosine_scale(prepared->query, info->dim)
          : 1.0f;
  prepared->compact_query_scale = 1.0f;
  prepared->use_f6 = (info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 ||
                      info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3_F32_RERANK)
                         ? 1u
                         : 0u;
  prepared->use_f6_i16 = (info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E3M2 ||
                          info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK)
                             ? 1u
                             : 0u;
  prepared->use_f8 = (info->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2 ||
                      info->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK)
                         ? 1u
                         : 0u;
  if (!i16_storage_kind(info->storage_kind) && compact_storage_kind(info->storage_kind) &&
      info->version >= kSaveVersionAlignedVectorData &&
      (info->vector_stride & static_cast<uint64_t>(kVectorStorageAlign - 1u)) == 0) {
    const uintptr_t vector_address = reinterpret_cast<uintptr_t>(bytes.data + info->vector_offset);
    prepared->compact_i8_vectors_aligned =
        (vector_address & static_cast<uintptr_t>(kVectorStorageAlign - 1u)) == 0 ? 1u : 0u;
  }
  if (i16_storage_kind(info->storage_kind) && info->version >= kSaveVersionAlignedVectorData &&
      (info->vector_stride & static_cast<uint64_t>(kVectorStorageAlign - 1u)) == 0) {
    const uintptr_t vector_address = reinterpret_cast<uintptr_t>(bytes.data + info->vector_offset);
    prepared->compact_i16_vectors_aligned =
        (vector_address & static_cast<uintptr_t>(kVectorStorageAlign - 1u)) == 0 ? 1u : 0u;
  }
  if (info->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK) {
    quantize_q8_vector(prepared->compact_query, &prepared->compact_query_scale, prepared->query,
                       info->dim);
    prepared->compact_query_sum = sum_i8(prepared->compact_query, info->dim);
  } else if (prepared->use_f6 != 0) {
    quantize_e2m3_vector(prepared->compact_query, &prepared->compact_query_scale, prepared->query,
                         info->dim);
    prepared->compact_query_scale *= kE2M3InvScale;
    prepared->compact_query_sum = sum_i8(prepared->compact_query, info->dim);
  } else if (prepared->use_f6_i16 != 0) {
    quantize_e3m2_vector(prepared->compact_query_i16, &prepared->compact_query_scale,
                         prepared->query, info->dim);
    prepared->compact_query_scale *= kE3M2InvScale;
  } else if (prepared->use_f8 != 0) {
    quantize_e5m2_vector(prepared->compact_query, &prepared->compact_query_scale, prepared->query,
                         info->dim);
  }
}

inline const uint8_t* snapshot_record_ptr(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                          uint32_t active_pos) {
  return bytes.data + info->record_offset + static_cast<uint64_t>(active_pos) * info->record_stride;
}

inline const uint8_t* snapshot_vector_ptr(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                          uint32_t active_pos) {
  return bytes.data + info->vector_offset + static_cast<uint64_t>(active_pos) * info->vector_stride;
}

bool snapshot_e5m2_requires_clamp(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info) {
  if (info->storage_kind != ASTRAL_MEMORY_STORAGE_F8_E5M2 &&
      info->storage_kind != ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK) {
    return false;
  }

  constexpr uint64_t kExponentMask = 0x7c7c7c7c7c7c7c7cull;
  constexpr uint64_t kByteOnes = 0x0101010101010101ull;
  constexpr uint64_t kByteHighBits = 0x8080808080808080ull;
  for (uint32_t active_pos = 0; active_pos < info->count; ++active_pos) {
    const uint8_t* vector = snapshot_vector_ptr(bytes, info, active_pos);
    uint32_t dim_i = 0;
    for (; dim_i + sizeof(uint64_t) <= info->dim; dim_i += sizeof(uint64_t)) {
      uint64_t packed = 0;
      std::memcpy(&packed, vector + dim_i, sizeof(packed));
      const uint64_t exponent_delta = (packed & kExponentMask) ^ kExponentMask;
      if (((exponent_delta - kByteOnes) & ~exponent_delta & kByteHighBits) != 0) {
        return true;
      }
    }
    for (; dim_i < info->dim; ++dim_i) {
      if ((vector[dim_i] & 0x7cu) == 0x7cu) {
        return true;
      }
    }
  }
  return false;
}

inline const uint8_t* snapshot_score_vector_ptr(SnapshotBytes bytes,
                                                const SnapshotPreparedQuery* prepared,
                                                uint32_t active_pos) {
  return bytes.data + prepared->score_vector_offset +
         static_cast<uint64_t>(active_pos) * prepared->score_vector_stride;
}

inline float snapshot_stored_scale(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                   uint32_t active_pos) {
  float stored_scale = 1.0f;
  std::memcpy(&stored_scale,
              bytes.data + info->scale_offset +
                  static_cast<uint64_t>(active_pos) * info->scale_stride,
              sizeof(stored_scale));
  return stored_scale;
}

inline float snapshot_compact_score_scale(SnapshotBytes bytes,
                                          const SnapshotPreparedQuery* prepared,
                                          uint32_t active_pos) {
  float scale = 1.0f;
  std::memcpy(&scale,
              bytes.data + prepared->compact_score_scale_offset +
                  static_cast<uint64_t>(active_pos) * prepared->compact_score_scale_stride,
              sizeof(scale));
  return scale;
}

float snapshot_score_active(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                            const SnapshotPreparedQuery* prepared, uint32_t active_pos);

float snapshot_graph_score_active(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                  const SnapshotPreparedQuery* prepared, uint32_t active_pos) {
  if (info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK) {
    const float stored_scale = snapshot_stored_scale(bytes, info, active_pos);
    const int16_t* vector =
        reinterpret_cast<const int16_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    const float scale = prepared->compact_score_scale_stride != 0
                            ? snapshot_compact_score_scale(bytes, prepared, active_pos)
                            : compact_value_scale_kind(info->storage_kind, stored_scale);
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      const float dot = prepared->compact_i16_vectors_aligned != 0
                            ? dot_i16_i16_aligned(vector, prepared->compact_query_i16, info->dim)
                            : dot_i16_i16(vector, prepared->compact_query_i16, info->dim);
      return dot * scale * prepared->compact_query_scale;
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      const float dot = prepared->compact_i16_vectors_aligned != 0
                            ? dot_i16_i16_aligned(vector, prepared->compact_query_i16, info->dim)
                            : dot_i16_i16(vector, prepared->compact_query_i16, info->dim);
      return dot * scale * prepared->compact_query_scale * prepared->query_scale;
    }
    return l2_score_i16_i16(vector, scale, prepared->compact_query_i16,
                            prepared->compact_query_scale, info->dim);
  }
  if (info->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK) {
    const float stored_scale = snapshot_stored_scale(bytes, info, active_pos);
    const int8_t* vector =
        reinterpret_cast<const int8_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    const float scale = prepared->compact_score_scale_stride != 0
                            ? snapshot_compact_score_scale(bytes, prepared, active_pos)
                            : compact_value_scale_kind(info->storage_kind, stored_scale);
    const float dot = prepared->compact_i8_vectors_aligned != 0
                          ? dot_q8_q8_query_aligned(vector, prepared->compact_query, info->dim,
                                                    prepared->compact_query_sum)
                          : dot_q8_q8_query(vector, prepared->compact_query, info->dim,
                                            prepared->compact_query_sum);
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return dot * scale * prepared->compact_query_scale;
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      const float vector_scale = prepared->compact_score_scale_stride == 0
                                     ? cosine_scale_q8(vector, stored_scale, info->dim)
                                     : 1.0f;
      return dot * scale * prepared->compact_query_scale * prepared->query_scale * vector_scale;
    }
    return l2_score_q8_q8(vector, scale, prepared->compact_query, prepared->compact_query_scale,
                          info->dim);
  }
  if (info->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK) {
    const float stored_scale = snapshot_stored_scale(bytes, info, active_pos);
    const int8_t* vector =
        reinterpret_cast<const int8_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    const float scale = prepared->compact_score_scale_stride != 0
                            ? snapshot_compact_score_scale(bytes, prepared, active_pos)
                            : compact_value_scale_kind(info->storage_kind, stored_scale);
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      const float dot =
          prepared->e5m2_clamp_non_finite != 0
              ? prepared->e5m2_kernels->dot_e5m2_clamped_a(vector, prepared->compact_query,
                                                           info->dim)
              : prepared->e5m2_kernels->dot_e5m2(vector, prepared->compact_query, info->dim);
      return dot * scale * prepared->compact_query_scale;
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      const float dot =
          prepared->e5m2_clamp_non_finite != 0
              ? prepared->e5m2_kernels->dot_e5m2_clamped_a(vector, prepared->compact_query,
                                                           info->dim)
              : prepared->e5m2_kernels->dot_e5m2(vector, prepared->compact_query, info->dim);
      return dot * scale * prepared->compact_query_scale * prepared->query_scale;
    }
    return prepared->e5m2_clamp_non_finite != 0
               ? prepared->e5m2_kernels->l2_e5m2_clamped_a(vector, scale, prepared->compact_query,
                                                           prepared->compact_query_scale, info->dim)
               : prepared->e5m2_kernels->l2_e5m2(vector, scale, prepared->compact_query,
                                                 prepared->compact_query_scale, info->dim);
  }
  return snapshot_score_active(bytes, info, prepared, active_pos);
}

float snapshot_score_active(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                            const SnapshotPreparedQuery* prepared, uint32_t active_pos) {
  if (f32_rerank_storage_kind(info->storage_kind)) {
    const float* vector =
        reinterpret_cast<const float*>(snapshot_score_vector_ptr(bytes, prepared, active_pos));
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return dot_f32(prepared->query, vector, info->dim);
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return dot_f32(prepared->query, vector, info->dim) * prepared->query_scale;
    }
    return l2_score_f32(prepared->query, vector, info->dim);
  }
  if (prepared->compact != 0) {
    const float stored_scale = snapshot_stored_scale(bytes, info, active_pos);
    const float scale = compact_value_scale_kind(info->storage_kind, stored_scale);
    const int8_t* vector =
        reinterpret_cast<const int8_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    const int16_t* vector_i16 =
        reinterpret_cast<const int16_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return prepared->use_f6 != 0
                 ? (prepared->compact_i8_vectors_aligned != 0
                        ? dot_q8_q8_query_aligned(vector, prepared->compact_query, info->dim,
                                                  prepared->compact_query_sum)
                        : dot_q8_q8_query(vector, prepared->compact_query, info->dim,
                                          prepared->compact_query_sum)) *
                       scale * prepared->compact_query_scale
             : prepared->use_f6_i16 != 0
                 ? (prepared->compact_i16_vectors_aligned != 0
                        ? dot_i16_i16_aligned(vector_i16, prepared->compact_query_i16, info->dim)
                        : dot_i16_i16(vector_i16, prepared->compact_query_i16, info->dim)) *
                       scale * prepared->compact_query_scale
             : prepared->use_f8 != 0 ? (prepared->e5m2_clamp_non_finite != 0
                                            ? prepared->e5m2_kernels->dot_e5m2_clamped_a(
                                                  vector, prepared->compact_query, info->dim)
                                            : prepared->e5m2_kernels->dot_e5m2(
                                                  vector, prepared->compact_query, info->dim)) *
                                           scale * prepared->compact_query_scale
                                     : dot_q8_f32(vector, prepared->query, info->dim) * scale;
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      if (prepared->use_f6 != 0) {
        const float dot = prepared->compact_i8_vectors_aligned != 0
                              ? dot_q8_q8_query_aligned(vector, prepared->compact_query, info->dim,
                                                        prepared->compact_query_sum)
                              : dot_q8_q8_query(vector, prepared->compact_query, info->dim,
                                                prepared->compact_query_sum);
        return dot * scale * prepared->compact_query_scale * prepared->query_scale;
      }
      if (prepared->use_f6_i16 != 0) {
        const float dot =
            prepared->compact_i16_vectors_aligned != 0
                ? dot_i16_i16_aligned(vector_i16, prepared->compact_query_i16, info->dim)
                : dot_i16_i16(vector_i16, prepared->compact_query_i16, info->dim);
        return dot * scale * prepared->compact_query_scale * prepared->query_scale;
      }
      if (prepared->use_f8 != 0) {
        const float dot =
            prepared->e5m2_clamp_non_finite != 0
                ? prepared->e5m2_kernels->dot_e5m2_clamped_a(vector, prepared->compact_query,
                                                             info->dim)
                : prepared->e5m2_kernels->dot_e5m2(vector, prepared->compact_query, info->dim);
        return dot * scale * prepared->compact_query_scale * prepared->query_scale;
      }
      const float vector_scale = cosine_scale_q8(vector, stored_scale, info->dim);
      return dot_q8_f32(vector, prepared->query, info->dim) * scale * prepared->query_scale *
             vector_scale;
    }
    return prepared->use_f6 != 0 ? l2_score_q8_q8(vector, scale, prepared->compact_query,
                                                  prepared->compact_query_scale, info->dim)
           : prepared->use_f6_i16 != 0
               ? l2_score_i16_i16(vector_i16, scale, prepared->compact_query_i16,
                                  prepared->compact_query_scale, info->dim)
           : prepared->use_f8 != 0
               ? (prepared->e5m2_clamp_non_finite != 0
                      ? prepared->e5m2_kernels->l2_e5m2_clamped_a(
                            vector, scale, prepared->compact_query, prepared->compact_query_scale,
                            info->dim)
                      : prepared->e5m2_kernels->l2_e5m2(vector, scale, prepared->compact_query,
                                                        prepared->compact_query_scale, info->dim))
               : l2_score_q8_f32(vector, scale, prepared->query, info->dim);
  }

  const float* vector =
      reinterpret_cast<const float*>(snapshot_vector_ptr(bytes, info, active_pos));
  if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
    return dot_f32(prepared->query, vector, info->dim);
  }
  if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return prepared->normalized_f32_cosine != 0
               ? dot_f32(prepared->query, vector, info->dim)
               : dot_f32(prepared->query, vector, info->dim) * prepared->query_scale *
                     cosine_scale(vector, info->dim);
  }
  return l2_score_f32(prepared->query, vector, info->dim);
}

bool snapshot_graph_better(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                           float candidate_score, uint32_t candidate_slot, float existing_score,
                           uint32_t existing_slot) {
  if (candidate_score != existing_score) {
    return candidate_score > existing_score;
  }
  AstralMemoryRecord candidate{};
  AstralMemoryRecord existing{};
  std::memcpy(&candidate, snapshot_record_ptr(bytes, info, candidate_slot), sizeof(candidate));
  std::memcpy(&existing, snapshot_record_ptr(bytes, info, existing_slot), sizeof(existing));
  return candidate.key < existing.key;
}

inline bool snapshot_graph_worse(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                 float candidate_score, uint32_t candidate_slot,
                                 float existing_score, uint32_t existing_slot) {
  return snapshot_graph_better(bytes, info, existing_score, existing_slot, candidate_score,
                               candidate_slot);
}

bool snapshot_graph_insert_top(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                               GraphSearchScratch* scratch, uint32_t capacity, uint32_t* filled,
                               uint32_t slot, float score) {
  uint32_t count = *filled;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!snapshot_graph_better(bytes, info, scratch->top_scores[parent],
                                 scratch->top_slots[parent], score, slot)) {
        break;
      }
      scratch->top_scores[pos] = scratch->top_scores[parent];
      scratch->top_slots[pos] = scratch->top_slots[parent];
      pos = parent;
    }
    scratch->top_scores[pos] = score;
    scratch->top_slots[pos] = slot;
    *filled = count + 1u;
    return true;
  }

  if (!snapshot_graph_better(bytes, info, score, slot, scratch->top_scores[0],
                             scratch->top_slots[0])) {
    return false;
  }

  uint32_t pos = 0;
  for (;;) {
    const uint32_t left = (pos << 1u) + 1u;
    if (left >= count) {
      break;
    }
    const uint32_t right = left + 1u;
    uint32_t child = left;
    if (right < count &&
        snapshot_graph_worse(bytes, info, scratch->top_scores[right], scratch->top_slots[right],
                             scratch->top_scores[left], scratch->top_slots[left])) {
      child = right;
    }
    if (!snapshot_graph_worse(bytes, info, scratch->top_scores[child], scratch->top_slots[child],
                              score, slot)) {
      break;
    }
    scratch->top_scores[pos] = scratch->top_scores[child];
    scratch->top_slots[pos] = scratch->top_slots[child];
    pos = child;
  }
  scratch->top_scores[pos] = score;
  scratch->top_slots[pos] = slot;
  return true;
}

void snapshot_graph_add_candidate(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                  GraphSearchScratch* scratch, uint32_t capacity, uint32_t slot,
                                  float score, uint32_t* candidate_count) {
  uint32_t count = *candidate_count;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!snapshot_graph_better(bytes, info, score, slot, scratch->candidate_scores[parent],
                                 scratch->candidates[parent])) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[parent];
      scratch->candidate_scores[pos] = scratch->candidate_scores[parent];
      pos = parent;
    }
    scratch->candidates[pos] = slot;
    scratch->candidate_scores[pos] = score;
    *candidate_count = count + 1u;
    return;
  }

  if (scratch->candidate_worst_valid == 0) {
    uint32_t worst_pos = 0;
    float worst_score = scratch->candidate_scores[0];
    for (uint32_t i = 1; i < count; ++i) {
      if (snapshot_graph_worse(bytes, info, scratch->candidate_scores[i], scratch->candidates[i],
                               worst_score, scratch->candidates[worst_pos])) {
        worst_score = scratch->candidate_scores[i];
        worst_pos = i;
      }
    }
    scratch->candidate_worst_pos = worst_pos;
    scratch->candidate_worst_slot = scratch->candidates[worst_pos];
    scratch->candidate_worst_score = worst_score;
    scratch->candidate_worst_valid = 1;
  }
  if (!snapshot_graph_better(bytes, info, score, slot, scratch->candidate_worst_score,
                             scratch->candidate_worst_slot)) {
    return;
  }

  uint32_t pos = scratch->candidate_worst_pos;
  bool moved_up = false;
  while (pos > 0) {
    const uint32_t parent = (pos - 1u) >> 1u;
    if (!snapshot_graph_better(bytes, info, score, slot, scratch->candidate_scores[parent],
                               scratch->candidates[parent])) {
      break;
    }
    scratch->candidates[pos] = scratch->candidates[parent];
    scratch->candidate_scores[pos] = scratch->candidate_scores[parent];
    pos = parent;
    moved_up = true;
  }
  if (!moved_up) {
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count &&
          snapshot_graph_better(bytes, info, scratch->candidate_scores[right],
                                scratch->candidates[right], scratch->candidate_scores[left],
                                scratch->candidates[left])) {
        child = right;
      }
      if (!snapshot_graph_better(bytes, info, scratch->candidate_scores[child],
                                 scratch->candidates[child], score, slot)) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[child];
      scratch->candidate_scores[pos] = scratch->candidate_scores[child];
      pos = child;
    }
  }
  scratch->candidates[pos] = slot;
  scratch->candidate_scores[pos] = score;
  scratch->candidate_worst_valid = 0;
}

bool snapshot_graph_pop_candidate(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                  GraphSearchScratch* scratch, uint32_t* candidate_count,
                                  uint32_t* out_slot, float* out_score) {
  uint32_t count = *candidate_count;
  if (count == 0) {
    return false;
  }

  *out_slot = scratch->candidates[0];
  *out_score = scratch->candidate_scores[0];
  --count;
  if (count != 0) {
    const uint32_t slot = scratch->candidates[count];
    const float score = scratch->candidate_scores[count];
    uint32_t pos = 0;
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count &&
          snapshot_graph_better(bytes, info, scratch->candidate_scores[right],
                                scratch->candidates[right], scratch->candidate_scores[left],
                                scratch->candidates[left])) {
        child = right;
      }
      if (!snapshot_graph_better(bytes, info, scratch->candidate_scores[child],
                                 scratch->candidates[child], score, slot)) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[child];
      scratch->candidate_scores[pos] = scratch->candidate_scores[child];
      pos = child;
    }
    scratch->candidates[pos] = slot;
    scratch->candidate_scores[pos] = score;
  }
  *candidate_count = count;
  scratch->candidate_worst_valid = 0;
  return true;
}

void remove_snapshot_graph_worst_top(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                     GraphSearchScratch* scratch, uint32_t* top_count) {
  uint32_t count = *top_count;
  if (count == 0) {
    return;
  }

  --count;
  if (count != 0) {
    const uint32_t slot = scratch->top_slots[count];
    const float score = scratch->top_scores[count];
    uint32_t pos = 0;
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count &&
          snapshot_graph_worse(bytes, info, scratch->top_scores[right], scratch->top_slots[right],
                               scratch->top_scores[left], scratch->top_slots[left])) {
        child = right;
      }
      if (!snapshot_graph_worse(bytes, info, scratch->top_scores[child], scratch->top_slots[child],
                                score, slot)) {
        break;
      }
      scratch->top_scores[pos] = scratch->top_scores[child];
      scratch->top_slots[pos] = scratch->top_slots[child];
      pos = child;
    }
    scratch->top_scores[pos] = score;
    scratch->top_slots[pos] = slot;
  }
  *top_count = count;
}

void trim_snapshot_graph_top_candidates(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                        GraphSearchScratch* scratch, uint32_t* top_count,
                                        uint32_t target_count) {
  while (*top_count > target_count) {
    remove_snapshot_graph_worst_top(bytes, info, scratch, top_count);
  }
}

void snapshot_graph_begin_visit(const AstralMemorySnapshotInfo* info, GraphSearchScratch* scratch) {
  ++scratch->visit_generation;
  scratch->candidate_worst_valid = 0;
  if (scratch->visit_generation == 0) {
    std::memset(scratch->visited, 0, sizeof(uint16_t) * info->count);
    scratch->visit_generation = kGraphVisitGenerationStart;
  }
}

inline void snapshot_graph_mark_visited(GraphSearchScratch* scratch, uint32_t active_pos) {
  scratch->visited[active_pos] = scratch->visit_generation;
}

inline bool snapshot_graph_was_visited(const GraphSearchScratch* scratch, uint32_t active_pos) {
  return scratch->visited[active_pos] == scratch->visit_generation;
}

inline uint32_t snapshot_graph_search_capacity(const MemorySnapshotView* view,
                                               const AstralMemorySearchDesc* desc) {
  uint32_t requested =
      desc->graph_search != 0 ? desc->graph_search : view->graph.header.query_search_capacity;
  if (requested < kGraphMinSearch) {
    requested = kGraphMinSearch;
  }
  if (requested > view->graph.scratch_capacity) {
    requested = view->graph.scratch_capacity;
  }
  return requested;
}

inline uint32_t snapshot_graph_candidate_search_capacity(const SnapshotGraphLayout* graph,
                                                         uint32_t search_capacity) {
  uint32_t requested = search_capacity;
  if (search_capacity <= kU32Max / kGraphCandidateReserveMultiplier) {
    requested = search_capacity * kGraphCandidateReserveMultiplier;
  }
  return requested < graph->candidate_capacity ? requested : graph->candidate_capacity;
}

template <bool CompactCounts>
uint32_t snapshot_graph_greedy_closest(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                       const SnapshotGraphLayout* graph,
                                       const SnapshotPreparedQuery* prepared, uint32_t entry,
                                       uint32_t begin_level, uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score = snapshot_graph_score_active(bytes, info, prepared, closest);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t neighbor_count =
          snapshot_graph_neighbor_count<CompactCounts>(bytes.data, info, graph, closest, level);
      const uint64_t neighbor_base = snapshot_graph_level_offset(info, graph, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = snapshot_graph_neighbor_at_base(bytes.data, neighbor_base, i);
        if (candidate == kU32Max || candidate >= info->count ||
            snapshot_graph_level(bytes.data, graph, candidate) < level) {
          continue;
        }
        const float score = snapshot_graph_score_active(bytes, info, prepared, candidate);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

template <bool CompactCounts>
void snapshot_graph_search_layer(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                 const SnapshotGraphLayout* graph,
                                 const SnapshotPreparedQuery* prepared, GraphSearchScratch* scratch,
                                 uint32_t entry, uint32_t search_capacity,
                                 uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity =
      snapshot_graph_candidate_search_capacity(graph, search_capacity);
  snapshot_graph_mark_visited(scratch, entry);
  const float entry_score = snapshot_graph_score_active(bytes, info, prepared, entry);
  snapshot_graph_insert_top(bytes, info, scratch, search_capacity, &top_count, entry, entry_score);
  snapshot_graph_add_candidate(bytes, info, scratch, candidate_capacity, entry, entry_score,
                               &candidate_count);

  while (candidate_count != 0) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!snapshot_graph_pop_candidate(bytes, info, scratch, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == search_capacity &&
        !snapshot_graph_better(bytes, info, slot_score, slot, scratch->top_scores[0],
                               scratch->top_slots[0])) {
      break;
    }
    const uint32_t neighbor_count =
        snapshot_graph_neighbor_count<CompactCounts>(bytes.data, info, graph, slot, 0);
    const uint64_t neighbor_base = snapshot_graph_level_offset(info, graph, slot, 0);
    const uint32_t prefetch_limit = neighbor_count > kGraphNeighborPrefetchDistance
                                        ? neighbor_count - kGraphNeighborPrefetchDistance
                                        : 0u;
    uint32_t i = 0;
    for (; i < prefetch_limit; ++i) {
      const uint32_t neighbor = snapshot_graph_neighbor_at_base(bytes.data, neighbor_base, i);
#if defined(__GNUC__) || defined(__clang__)
      const uint32_t prefetch_neighbor = snapshot_graph_neighbor_at_base(
          bytes.data, neighbor_base, i + kGraphNeighborPrefetchDistance);
      __builtin_prefetch(info->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK
                             ? snapshot_vector_ptr(bytes, info, prefetch_neighbor)
                             : snapshot_score_vector_ptr(bytes, prepared, prefetch_neighbor),
                         0, 1);
#endif
      if (snapshot_graph_was_visited(scratch, neighbor)) {
        continue;
      }
      snapshot_graph_mark_visited(scratch, neighbor);
      const float score = snapshot_graph_score_active(bytes, info, prepared, neighbor);
      if (snapshot_graph_insert_top(bytes, info, scratch, search_capacity, &top_count, neighbor,
                                    score)) {
        snapshot_graph_add_candidate(bytes, info, scratch, candidate_capacity, neighbor, score,
                                     &candidate_count);
      }
    }
    for (; i < neighbor_count; ++i) {
      const uint32_t neighbor = snapshot_graph_neighbor_at_base(bytes.data, neighbor_base, i);
      if (snapshot_graph_was_visited(scratch, neighbor)) {
        continue;
      }
      snapshot_graph_mark_visited(scratch, neighbor);
      const float score = snapshot_graph_score_active(bytes, info, prepared, neighbor);
      if (snapshot_graph_insert_top(bytes, info, scratch, search_capacity, &top_count, neighbor,
                                    score)) {
        snapshot_graph_add_candidate(bytes, info, scratch, candidate_capacity, neighbor, score,
                                     &candidate_count);
      }
    }
  }
  *out_top_count = top_count;
}

AstralErr memory_snapshot_search_with_info(SnapshotBytes bytes,
                                           const AstralMemorySnapshotInfo* info,
                                           const AstralMemorySearchDesc* desc, const float* query,
                                           AstralMemorySearchResult* out_results,
                                           uint32_t max_results, uint32_t* out_count,
                                           uint8_t e5m2_clamp_non_finite);

AstralErr memory_snapshot_view_search_graph(MemorySnapshotView* view,
                                            const AstralMemorySearchDesc* desc, const float* query,
                                            AstralMemorySearchResult* out_results,
                                            uint32_t max_results, uint32_t* out_count) {
  if (desc == nullptr || desc->size != sizeof(AstralMemorySearchDesc) || query == nullptr ||
      out_count == nullptr || desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }
  if (view->graph_ready == 0 || desc->group_id != ASTRAL_MEMORY_GROUP_ANY) {
    return memory_snapshot_search_with_info(SnapshotBytes{view->map.data, view->map.size},
                                            &view->info, desc, query, out_results, max_results,
                                            out_count, view->e5m2_clamp_non_finite);
  }
  if (snapshot_compact_graph_exact_search_preferred(&view->info)) {
    return memory_snapshot_search_with_info(SnapshotBytes{view->map.data, view->map.size},
                                            &view->info, desc, query, out_results, max_results,
                                            out_count, view->e5m2_clamp_non_finite);
  }
  if (out_results == nullptr || max_results < desc->top_k) {
    return ASTRAL_E_NOMEM;
  }
  if (!f32_values_finite(query, view->info.dim)) {
    return ASTRAL_E_INVALID;
  }
  SnapshotBytes bytes{view->map.data, view->map.size};
  SnapshotPreparedQuery prepared{};
  snapshot_prepare_query(bytes, &view->info, query, view->e5m2_clamp_non_finite, &prepared);
  GraphSearchScratch* scratch = &view->graph_scratch;
  snapshot_graph_begin_visit(&view->info, scratch);
  const uint32_t search_capacity = snapshot_graph_search_capacity(view, desc);
  uint32_t top_count = 0;
  if (save_graph_count_bytes(view->info.version) == sizeof(uint8_t)) {
    const uint32_t entry =
        view->graph.header.max_level != 0
            ? snapshot_graph_greedy_closest<true>(bytes, &view->info, &view->graph, &prepared,
                                                  view->graph.header.entry_active_pos,
                                                  view->graph.header.max_level, 0)
            : view->graph.header.entry_active_pos;
    snapshot_graph_search_layer<true>(bytes, &view->info, &view->graph, &prepared, scratch, entry,
                                      search_capacity, &top_count);
  } else {
    const uint32_t entry =
        view->graph.header.max_level != 0
            ? snapshot_graph_greedy_closest<false>(bytes, &view->info, &view->graph, &prepared,
                                                   view->graph.header.entry_active_pos,
                                                   view->graph.header.max_level, 0)
            : view->graph.header.entry_active_pos;
    snapshot_graph_search_layer<false>(bytes, &view->info, &view->graph, &prepared, scratch, entry,
                                       search_capacity, &top_count);
  }
  if (f32_rerank_storage_kind(view->info.storage_kind)) {
    const uint32_t rerank_capacity = graph_f32_rerank_capacity(desc->top_k, top_count);
    trim_snapshot_graph_top_candidates(bytes, &view->info, scratch, &top_count, rerank_capacity);
  }

  uint32_t filled = 0;
  for (uint32_t i = 0; i < top_count; ++i) {
    AstralMemoryRecord record{};
    std::memcpy(&record, snapshot_record_ptr(bytes, &view->info, scratch->top_slots[i]),
                sizeof(record));
    if (record.size != sizeof(AstralMemoryRecord) || record.key == 0) {
      return ASTRAL_E_INVALID;
    }
    MemorySlot slot{};
    slot.record = record;
    AstralMemorySearchResult candidate{};
    const float score =
        f32_rerank_storage_kind(view->info.storage_kind)
            ? snapshot_score_active(bytes, &view->info, &prepared, scratch->top_slots[i])
            : scratch->top_scores[i];
    fill_result(&candidate, slot, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  *out_count = filled;
  return ASTRAL_OK;
}

AstralErr memory_snapshot_search_with_info(SnapshotBytes bytes,
                                           const AstralMemorySnapshotInfo* info,
                                           const AstralMemorySearchDesc* desc, const float* query,
                                           AstralMemorySearchResult* out_results,
                                           uint32_t max_results, uint32_t* out_count,
                                           uint8_t e5m2_clamp_non_finite) {
  if (bytes.data == nullptr || info == nullptr || desc == nullptr ||
      desc->size != sizeof(AstralMemorySearchDesc) || query == nullptr || out_count == nullptr ||
      desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }
  if (out_results == nullptr || max_results < desc->top_k) {
    return ASTRAL_E_NOMEM;
  }
  if (!f32_values_finite(query, info->dim)) {
    return ASTRAL_E_INVALID;
  }

  SnapshotPreparedQuery prepared{};
  snapshot_prepare_query(bytes, info, query, e5m2_clamp_non_finite, &prepared);
  uint32_t filled = 0;
  for (uint32_t i = 0; i < info->count; ++i) {
#if defined(__GNUC__) || defined(__clang__)
    if (prepared.compact != 0 && i + kFlatQ8PrefetchDistance < info->count) {
      const uint64_t prefetch_i = static_cast<uint64_t>(i + kFlatQ8PrefetchDistance);
      __builtin_prefetch(bytes.data + info->scale_offset + prefetch_i * info->scale_stride, 0, 1);
      __builtin_prefetch(
          snapshot_score_vector_ptr(bytes, &prepared, static_cast<uint32_t>(prefetch_i)), 0, 1);
      __builtin_prefetch(bytes.data + info->record_offset + prefetch_i * info->record_stride, 0, 1);
    }
#endif
    AstralMemoryRecord record{};
    std::memcpy(&record, snapshot_record_ptr(bytes, info, i), sizeof(record));
    if (record.size != sizeof(AstralMemoryRecord) || record.key == 0) {
      return ASTRAL_E_INVALID;
    }
    if (desc->group_id != ASTRAL_MEMORY_GROUP_ANY && record.group_id != desc->group_id) {
      continue;
    }

    MemorySlot slot{};
    slot.record = record;
    AstralMemorySearchResult candidate{};
    fill_result(&candidate, slot, snapshot_score_active(bytes, info, &prepared, i));
    insert_result(out_results, desc->top_k, &filled, candidate);
  }

  *out_count = filled;
  return ASTRAL_OK;
}

AstralErr memory_snapshot_search_bytes(SnapshotBytes bytes, const AstralMemorySearchDesc* desc,
                                       const float* query, AstralMemorySearchResult* out_results,
                                       uint32_t max_results, uint32_t* out_count) {
  AstralMemorySnapshotInfo info{};
  info.size = sizeof(AstralMemorySnapshotInfo);
  AstralErr err = memory_snapshot_info_bytes(bytes, &info);
  if (err != ASTRAL_OK) {
    return err;
  }

  return memory_snapshot_search_with_info(bytes, &info, desc, query, out_results, max_results,
                                          out_count, 1u);
}

} // namespace

AstralErr memory_snapshot_info(AstralSpanU8 bytes, AstralMemorySnapshotInfo* out_info) {
  return memory_snapshot_info_bytes(SnapshotBytes{bytes.data, bytes.len}, out_info);
}

AstralErr memory_snapshot_search(AstralSpanU8 bytes, const AstralMemorySearchDesc* desc,
                                 const float* query, AstralMemorySearchResult* out_results,
                                 uint32_t max_results, uint32_t* out_count) {
  return memory_snapshot_search_bytes(SnapshotBytes{bytes.data, bytes.len}, desc, query,
                                      out_results, max_results, out_count);
}

AstralErr memory_snapshot_map(AstralSpanU8 path, MemorySnapshotView** out_view,
                              AstralMemorySnapshotInfo* out_info) {
  if (path.data == nullptr || path.len == 0 || out_view == nullptr || out_info == nullptr ||
      out_info->size != sizeof(AstralMemorySnapshotInfo)) {
    return ASTRAL_E_INVALID;
  }
  *out_view = nullptr;

  std::string path_string(reinterpret_cast<const char*>(path.data), path.len);
  auto* view = core::runtime_new<MemorySnapshotView>();
  if (view == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  *view = {};
  if (!platform::file_map_readonly(path_string.c_str(), &view->map)) {
    core::runtime_delete(view);
    return ASTRAL_E_NOT_FOUND;
  }

  AstralMemorySnapshotInfo info{};
  info.size = sizeof(AstralMemorySnapshotInfo);
  AstralErr err = memory_snapshot_info_bytes(SnapshotBytes{view->map.data, view->map.size}, &info);
  if (err != ASTRAL_OK) {
    platform::file_unmap_readonly(&view->map);
    core::runtime_delete(view);
    return err;
  }
  SnapshotBytes view_bytes{view->map.data, view->map.size};
  view->graph_ready = 0;
  view->e5m2_clamp_non_finite = snapshot_e5m2_requires_clamp(view_bytes, &info) ? 1u : 0u;
  if (snapshot_graph_layout(view_bytes, &info, &view->graph) &&
      snapshot_graph_scratch_alloc(&info, &view->graph, &view->graph_scratch)) {
    view->graph_ready = 1;
  }

  const AstralHandle handle = core::register_handle(core::HandleKind::MemorySnapshot, view);
  if (handle == 0) {
    if (view->graph_ready != 0) {
      snapshot_graph_scratch_free(&info, &view->graph, &view->graph_scratch);
    }
    platform::file_unmap_readonly(&view->map);
    core::runtime_delete(view);
    return ASTRAL_E_NOMEM;
  }
  view->handle = handle;
  view->info = info;
  *out_info = info;
  *out_view = view;
  return ASTRAL_OK;
}

void memory_snapshot_unmap(MemorySnapshotView* view) {
  if (view == nullptr) {
    return;
  }
  core::unregister_handle(view->handle, core::HandleKind::MemorySnapshot);
  if (view->graph_ready != 0) {
    snapshot_graph_scratch_free(&view->info, &view->graph, &view->graph_scratch);
  }
  platform::file_unmap_readonly(&view->map);
  core::runtime_delete(view);
}

AstralErr memory_snapshot_view_info(MemorySnapshotView* view, AstralMemorySnapshotInfo* out_info) {
  if (view == nullptr || out_info == nullptr ||
      out_info->size != sizeof(AstralMemorySnapshotInfo)) {
    return ASTRAL_E_INVALID;
  }
  *out_info = view->info;
  return ASTRAL_OK;
}

AstralErr memory_snapshot_view_search(MemorySnapshotView* view, const AstralMemorySearchDesc* desc,
                                      const float* query, AstralMemorySearchResult* out_results,
                                      uint32_t max_results, uint32_t* out_count) {
  if (view == nullptr) {
    return ASTRAL_E_INVALID;
  }
  if (view->info.index_kind == ASTRAL_MEMORY_INDEX_GRAPH) {
    return memory_snapshot_view_search_graph(view, desc, query, out_results, max_results,
                                             out_count);
  }
  return memory_snapshot_search_with_info(SnapshotBytes{view->map.data, view->map.size},
                                          &view->info, desc, query, out_results, max_results,
                                          out_count, view->e5m2_clamp_non_finite);
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
