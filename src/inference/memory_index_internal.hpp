#pragma once

#include "memory_index.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace astral::inference {

inline constexpr uint32_t kMaxDim = 8192;
inline constexpr uint32_t kU32Max = 0xFFFFFFFFu;
inline constexpr float kE2M3InvScale = 1.0f / 16.0f;
inline constexpr float kE3M2InvScale = 1.0f / 16.0f;
inline constexpr float kWorstScore = -3.4028234663852886e38f;
inline constexpr uint32_t kMemorySearchBatchParallelMaxWorkers = 16;
inline constexpr uint32_t kSaveMagic = 0x414D454Du;
inline constexpr uint32_t kSaveVersionF32 = 1;
inline constexpr uint32_t kSaveVersionCompactStorage = 2;
inline constexpr uint32_t kSaveVersionGraphTopology = 3;
inline constexpr uint32_t kSaveVersionLegacyLayout = 4;
inline constexpr uint32_t kSaveVersionModernLayout = 5;
inline constexpr uint32_t kSaveVersionNormalizedF32Cosine = 6;
inline constexpr uint32_t kSaveVersionGraphQuerySearch = 7;
inline constexpr uint32_t kSaveVersionCompactScoreScales = 8;
inline constexpr uint32_t kSaveVersionCompactGraphCounts = 9;
inline constexpr uint32_t kSaveVersionAlignedVectorData = 10;
inline constexpr uint32_t kSaveVersion = kSaveVersionAlignedVectorData;
inline constexpr uint32_t kSaveGraphTopologyFlag = 1;

#if defined(__AVX2__)
inline constexpr size_t kVectorStorageAlign = 32;
#elif defined(__aarch64__) && defined(__ARM_NEON)
inline constexpr size_t kVectorStorageAlign = 16;
#else
inline constexpr size_t kVectorStorageAlign = alignof(float);
#endif

using DotE5m2F32Fn = float (*)(const int8_t*, const float*, uint32_t);
using DotE5m2E5m2Fn = float (*)(const int8_t*, const int8_t*, uint32_t);
using L2E5m2F32Fn = float (*)(const int8_t*, float, const float*, uint32_t);
using L2E5m2E5m2Fn = float (*)(const int8_t*, float, const int8_t*, float, uint32_t);

struct E5m2Kernels {
  DotE5m2F32Fn dot_f32;
  DotE5m2E5m2Fn dot_e5m2;
  DotE5m2E5m2Fn dot_e5m2_clamped_a;
  L2E5m2F32Fn l2_f32;
  L2E5m2E5m2Fn l2_e5m2;
  L2E5m2E5m2Fn l2_e5m2_clamped_a;
};

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

float dot_f32(const float* a, const float* b, uint32_t dim);
float l2_score_f32(const float* a, const float* b, uint32_t dim);
float dot_q8_f32(const int8_t* a, const float* b, uint32_t dim);
float dot_i16_f32(const int16_t* a, const float* b, uint32_t dim);
int32_t sum_i8(const int8_t* v, uint32_t dim);
float dot_q8_q8_query(const int8_t* a, const int8_t* b, uint32_t dim, int32_t b_sum);
float dot_q8_q8_query_aligned(const int8_t* a, const int8_t* b, uint32_t dim, int32_t b_sum);
float dot_i16_i16(const int16_t* a, const int16_t* b, uint32_t dim);
float dot_i16_i16_aligned(const int16_t* a, const int16_t* b, uint32_t dim);

float e5m2_to_f32(uint8_t raw);
void copy_finite_e5m2(int8_t* dst, const int8_t* src, uint32_t dim);
bool f32_values_finite(const float* values, size_t count);
float dot_e5m2_f32_384(const int8_t* a, const float* b);
const E5m2Kernels* select_e5m2_kernels();

float l2_score_q8_f32(const int8_t* a, float scale, const float* b, uint32_t dim);
float l2_score_i16_f32(const int16_t* a, float scale, const float* b, uint32_t dim);
float l2_score_q8_q8(const int8_t* a, float a_scale, const int8_t* b, float b_scale,
                     uint32_t dim);
float l2_score_i16_i16(const int16_t* a, float a_scale, const int16_t* b, float b_scale,
                       uint32_t dim);

void quantize_q8_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim);
void quantize_e2m3_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim);
void quantize_e3m2_vector(int16_t* dst, float* out_scale, const float* src, uint32_t dim);
void quantize_e3m2_cosine_vector(int16_t* dst, float* out_scale, const float* src, uint32_t dim);
void quantize_e2m3_cosine_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim);
void quantize_e5m2_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim);
void quantize_e5m2_cosine_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim);

float cosine_scale(const float* v, uint32_t dim);
void normalize_f32_vector(float* dst, const float* src, uint32_t dim);
float cosine_scale_q8(const int8_t* v, float scale, uint32_t dim);

inline float e2m3_scaled_to_f32(int8_t scaled, float scale) {
  return static_cast<float>(scaled) * scale * kE2M3InvScale;
}

inline float e3m2_scaled_to_f32(int16_t scaled, float scale) {
  return static_cast<float>(scaled) * scale * kE3M2InvScale;
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

inline void quantize_compact_query(const MemoryIndex* index, int8_t* dst, float* out_scale,
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
inline void insert_result(AstralMemorySearchResult* results, uint32_t top_k, uint32_t* filled,
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

void graph_connect_slot(MemoryIndex* index, uint32_t slot);
void graph_rebuild(MemoryIndex* index);
void memory_search_graph(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                         const float* query, AstralMemorySearchResult* out_results,
                         uint32_t* out_count);
void graph_search_scratch_free(const MemoryIndex* index, GraphSearchScratch* scratch);
bool graph_search_scratch_alloc(const MemoryIndex* index, GraphSearchScratch* scratch);
bool memory_search_graph_batch_parallel(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                        const float* queries, uint32_t query_count,
                                        AstralMemorySearchResult* out_results,
                                        uint32_t* out_counts);
void memory_search_flat_fallback(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                 const float* query, AstralMemorySearchResult* out_results,
                                 uint32_t* out_count);
uint64_t memory_save_byte_count(const MemoryIndex* index);
uint32_t save_graph_count_bytes(uint32_t version);
bool memory_save_layout(uint32_t version, uint32_t dim, uint32_t count,
                        AstralMemoryStorageKind storage, uint32_t index_kind,
                        uint32_t graph_base_neighbors, uint32_t graph_neighbors,
                        uint32_t graph_levels, SaveLayout* out_layout);
uint32_t save_graph_header_bytes(uint32_t version);
void read_save_graph_header(uint32_t version, const uint8_t* bytes,
                            SaveGraphHeader* out_header);

} // namespace astral::inference
