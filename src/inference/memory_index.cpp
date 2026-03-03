#include "memory_index.hpp"

#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"

#include <atomic>
#include <cstddef>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace astral::inference {

namespace {

constexpr uint32_t kMaxDim = 8192;
constexpr uint32_t kSaveMagic = 0x414D454Du;
constexpr uint32_t kSaveVersion = 1;
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
constexpr uint32_t kGraphDefaultNeighbors = 16;
constexpr uint32_t kGraphMaxNeighbors = 64;
constexpr uint32_t kGraphDefaultSearch = 64;
constexpr uint32_t kGraphMinSearch = 4;
constexpr uint32_t kGraphLongLinkCount = 4;
constexpr float kWorstScore = -3.4028234663852886e38f;
#if defined(__AVX2__)
constexpr uint32_t kAvx2F32Lanes = 8;
constexpr size_t kVectorStorageAlign = 32;
constexpr uint32_t kAvx2UnrollVectors = 4;
constexpr uint32_t kAvx2UnrollF32 = kAvx2F32Lanes * kAvx2UnrollVectors;
constexpr uint32_t kAvx2Offset1 = kAvx2F32Lanes;
constexpr uint32_t kAvx2Offset2 = kAvx2F32Lanes * 2u;
constexpr uint32_t kAvx2Offset3 = kAvx2F32Lanes * 3u;
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
constexpr uint32_t kNeonF32Lanes = 4;
constexpr size_t kVectorStorageAlign = 16;
constexpr uint32_t kNeonUnrollVectors = 4;
constexpr uint32_t kNeonUnrollF32 = kNeonF32Lanes * kNeonUnrollVectors;
constexpr uint32_t kNeonOffset1 = kNeonF32Lanes;
constexpr uint32_t kNeonOffset2 = kNeonF32Lanes * 2u;
constexpr uint32_t kNeonOffset3 = kNeonF32Lanes * 3u;
#endif
#if !defined(__AVX2__) && !(defined(__aarch64__) && defined(__ARM_NEON))
constexpr size_t kVectorStorageAlign = alignof(float);
#endif

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

inline bool metric_valid(AstralMemoryMetric metric) {
  return metric == ASTRAL_MEMORY_METRIC_DOT || metric == ASTRAL_MEMORY_METRIC_COSINE ||
         metric == ASTRAL_MEMORY_METRIC_L2;
}

inline bool index_kind_valid(AstralMemoryIndexKind kind) {
  return kind == ASTRAL_MEMORY_INDEX_FLAT || kind == ASTRAL_MEMORY_INDEX_GRAPH;
}

inline bool desc_valid(const AstralMemoryIndexDesc* desc) {
  return desc != nullptr && desc->size == sizeof(AstralMemoryIndexDesc) && desc->dim != 0 &&
         desc->dim <= kMaxDim && desc->capacity != 0 && desc->_reserved0 == 0 &&
         index_kind_valid(desc->index_kind) && metric_valid(desc->metric);
}

inline uint32_t graph_neighbors_from_desc(const AstralMemoryIndexDesc* desc) {
  if (desc->index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    return 0;
  }
  const uint32_t requested = desc->graph_neighbors != 0 ? desc->graph_neighbors : kGraphDefaultNeighbors;
  return requested < kGraphMaxNeighbors ? requested : kGraphMaxNeighbors;
}

inline uint32_t graph_search_from_desc(const AstralMemoryIndexDesc* desc) {
  if (desc->index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    return 0;
  }
  const uint32_t requested = desc->graph_search != 0 ? desc->graph_search : kGraphDefaultSearch;
  return requested < kGraphMinSearch ? kGraphMinSearch : requested;
}

inline float* alloc_vector_storage(uint32_t capacity, uint32_t dim) {
  const size_t count = static_cast<size_t>(capacity) * dim;
  return static_cast<float*>(core::runtime_alloc(count * sizeof(float), kVectorStorageAlign));
}

inline void free_vector_storage(float* ptr, uint32_t capacity, uint32_t dim) {
  core::runtime_free(ptr, static_cast<size_t>(capacity) * dim * sizeof(float), kVectorStorageAlign);
}

#if defined(__AVX2__)
inline float reduce_avx2_f32(__m256 acc) {
  const __m128 lo = _mm256_castps256_ps128(acc);
  const __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}
#endif

float dot_f32(const float* a, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + kAvx2UnrollF32 <= dim; i += kAvx2UnrollF32) {
    const __m256 a0 = _mm256_loadu_ps(a + i);
    const __m256 b0 = _mm256_loadu_ps(b + i);
    const __m256 a1 = _mm256_loadu_ps(a + i + kAvx2Offset1);
    const __m256 b1 = _mm256_loadu_ps(b + i + kAvx2Offset1);
    const __m256 a2 = _mm256_loadu_ps(a + i + kAvx2Offset2);
    const __m256 b2 = _mm256_loadu_ps(b + i + kAvx2Offset2);
    const __m256 a3 = _mm256_loadu_ps(a + i + kAvx2Offset3);
    const __m256 b3 = _mm256_loadu_ps(b + i + kAvx2Offset3);
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    acc2 = _mm256_fmadd_ps(a2, b2, acc2);
    acc3 = _mm256_fmadd_ps(a3, b3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(a2, b2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(a3, b3));
#endif
  }
  acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  for (; i + kAvx2F32Lanes <= dim; i += kAvx2F32Lanes) {
    const __m256 av = _mm256_loadu_ps(a + i);
    const __m256 bv = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(av, bv, acc0);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av, bv));
#endif
  }
  float sum = reduce_avx2_f32(acc0);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#elif defined(__aarch64__) && defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + kNeonUnrollF32 <= dim; i += kNeonUnrollF32) {
    const float32x4_t a0 = vld1q_f32(a + i);
    const float32x4_t b0 = vld1q_f32(b + i);
    const float32x4_t a1 = vld1q_f32(a + i + kNeonOffset1);
    const float32x4_t b1 = vld1q_f32(b + i + kNeonOffset1);
    const float32x4_t a2 = vld1q_f32(a + i + kNeonOffset2);
    const float32x4_t b2 = vld1q_f32(b + i + kNeonOffset2);
    const float32x4_t a3 = vld1q_f32(a + i + kNeonOffset3);
    const float32x4_t b3 = vld1q_f32(b + i + kNeonOffset3);
    acc0 = vfmaq_f32(acc0, a0, b0);
    acc1 = vfmaq_f32(acc1, a1, b1);
    acc2 = vfmaq_f32(acc2, a2, b2);
    acc3 = vfmaq_f32(acc3, a3, b3);
  }
  acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  for (; i + kNeonF32Lanes <= dim; i += kNeonF32Lanes) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  float sum = vaddvq_f32(acc0);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#else
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    sum0 += a[i] * b[i];
    sum1 += a[i + 1u] * b[i + 1u];
    sum2 += a[i + 2u] * b[i + 2u];
    sum3 += a[i + 3u] * b[i + 3u];
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

float l2_score_f32(const float* a, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + kAvx2UnrollF32 <= dim; i += kAvx2UnrollF32) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + kAvx2Offset1),
                                    _mm256_loadu_ps(b + i + kAvx2Offset1));
    const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + kAvx2Offset2),
                                    _mm256_loadu_ps(b + i + kAvx2Offset2));
    const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + kAvx2Offset3),
                                    _mm256_loadu_ps(b + i + kAvx2Offset3));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    acc2 = _mm256_fmadd_ps(d2, d2, acc2);
    acc3 = _mm256_fmadd_ps(d3, d3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
#endif
  }
  acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  for (; i + kAvx2F32Lanes <= dim; i += kAvx2F32Lanes) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(d, d, acc0);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d, d));
#endif
  }
  float sum = reduce_avx2_f32(acc0);
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return -sum;
#elif defined(__aarch64__) && defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + kNeonUnrollF32 <= dim; i += kNeonUnrollF32) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + kNeonOffset1), vld1q_f32(b + i + kNeonOffset1));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + kNeonOffset2), vld1q_f32(b + i + kNeonOffset2));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + kNeonOffset3), vld1q_f32(b + i + kNeonOffset3));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
    acc2 = vfmaq_f32(acc2, d2, d2);
    acc3 = vfmaq_f32(acc3, d3, d3);
  }
  acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  for (; i + kNeonF32Lanes <= dim; i += kNeonF32Lanes) {
    const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    acc0 = vfmaq_f32(acc0, d, d);
  }
  float sum = vaddvq_f32(acc0);
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return -sum;
#else
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return -sum;
#endif
}

inline float vector_norm(const float* v, uint32_t dim) {
  const float sq = dot_f32(v, v, dim);
  return sq > 0.0f ? std::sqrt(sq) : 0.0f;
}

inline float cosine_scale(const float* v, uint32_t dim) {
  const float norm = vector_norm(v, dim);
  return norm > 0.0f ? 1.0f / norm : 0.0f;
}

} // namespace

struct MemoryIndex {
  AstralHandle handle;
  uint32_t dim;
  uint32_t capacity;
  uint32_t count;
  AstralMemoryMetric metric;
  AstralMemoryIndexKind index_kind;
  MemorySlot* slots;
  float* vectors;
  uint32_t* active_slots;
  uint32_t* key_table;
  uint32_t* graph_neighbors;
  uint32_t* graph_neighbor_counts;
  uint32_t* graph_candidates;
  float* graph_candidate_scores;
  uint32_t* graph_build_slots;
  float* graph_build_scores;
  uint32_t* graph_visited;
  uint32_t key_table_capacity;
  uint32_t key_table_mask;
  uint32_t graph_neighbor_capacity;
  uint32_t graph_search_capacity;
  uint32_t graph_entry_slot;
  uint32_t graph_visit_generation;
  uint32_t free_slot_hint;
  uint8_t dense_active;
};

struct MemorySearchCursor {
  AstralHandle handle;
  uint32_t capacity;
  uint32_t count;
  uint32_t offset;
  std::atomic<uint32_t> canceled;
  AstralMemorySearchResult* results;
};

AstralHandle memory_handle(MemoryIndex* index) {
  return index != nullptr ? index->handle : 0;
}

AstralHandle memory_search_cursor_handle(MemorySearchCursor* cursor) {
  return cursor != nullptr ? cursor->handle : 0;
}

namespace {

inline float* vector_at(MemoryIndex* index, uint32_t slot) {
  return index->vectors + static_cast<size_t>(slot) * index->dim;
}

inline uint32_t active_slot_at(const MemoryIndex* index, uint32_t active_pos) {
  return index->dense_active != 0 ? active_pos : index->active_slots[active_pos];
}

inline bool graph_enabled(const MemoryIndex* index) {
  return index->index_kind == ASTRAL_MEMORY_INDEX_GRAPH && index->graph_neighbor_capacity != 0;
}

inline uint32_t* graph_neighbors_at(MemoryIndex* index, uint32_t slot) {
  return index->graph_neighbors + static_cast<size_t>(slot) * index->graph_neighbor_capacity;
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
  const uint32_t target = capacity < kKeyTableLoadFactorDen
      ? kKeyTableMinCapacity
      : capacity * kKeyTableLoadFactorDen;
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

AstralErr key_table_insert_new_hashed(MemoryIndex* index, uint64_t hash, uint32_t slot) {
  uint32_t table_pos = static_cast<uint32_t>(hash) & index->key_table_mask;
  uint32_t tombstone_pos = kU32Max;
  for (uint32_t probe = 0; probe < index->key_table_capacity; ++probe) {
    const uint32_t ref = index->key_table[table_pos];
    if (ref == kKeyTableEmpty) {
      index->key_table[tombstone_pos != kU32Max ? tombstone_pos : table_pos] = slot_to_key_ref(slot);
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

inline void fill_result(AstralMemorySearchResult* out_result,
                        const MemorySlot& slot,
                        float score) {
  out_result->size = sizeof(AstralMemorySearchResult);
  out_result->group_id = slot.record.group_id;
  out_result->key = slot.record.key;
  out_result->document_id = slot.record.document_id;
  out_result->chunk_id = slot.record.chunk_id;
  out_result->flags = slot.record.flags;
  out_result->score = score;
}

inline float score_slot(MemoryIndex* index, const float* query, uint32_t slot, float query_scale) {
  if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    return dot_f32(query, vector_at(index, slot), index->dim);
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return dot_f32(query, vector_at(index, slot), index->dim) * query_scale *
           index->slots[slot].score_scale;
  }
  return l2_score_f32(query, vector_at(index, slot), index->dim);
}

inline float score_pair(MemoryIndex* index, uint32_t a, uint32_t b) {
  const float* va = vector_at(index, a);
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return dot_f32(va, vector_at(index, b), index->dim) *
           index->slots[a].score_scale * index->slots[b].score_scale;
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
    return l2_score_f32(va, vector_at(index, b), index->dim);
  }
  return dot_f32(va, vector_at(index, b), index->dim);
}

void memory_search_flat(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                        AstralMemorySearchResult* out_results, uint32_t* out_count);
void graph_begin_visit(MemoryIndex* index);
void graph_add_candidate(MemoryIndex* index, uint32_t slot, float score, uint32_t* candidate_count);
bool graph_pop_candidate(MemoryIndex* index, uint32_t* candidate_count, uint32_t* out_slot, float* out_score);
inline void graph_mark_visited(MemoryIndex* index, uint32_t slot);
inline bool graph_was_visited(const MemoryIndex* index, uint32_t slot);

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

void insert_graph_neighbor(MemoryIndex* index, uint32_t owner_slot, uint32_t neighbor_slot) {
  if (owner_slot == neighbor_slot) {
    return;
  }

  uint32_t* neighbors = graph_neighbors_at(index, owner_slot);
  uint32_t& count = index->graph_neighbor_counts[owner_slot];
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == neighbor_slot) {
      return;
    }
  }

  if (count < index->graph_neighbor_capacity) {
    neighbors[count] = neighbor_slot;
    ++count;
    return;
  }

  uint32_t worst_pos = 0;
  float worst_score = score_pair(index, owner_slot, neighbors[0]);
  for (uint32_t i = 1; i < count; ++i) {
    const float score = score_pair(index, owner_slot, neighbors[i]);
    if (score < worst_score) {
      worst_score = score;
      worst_pos = i;
    }
  }

  const float candidate_score = score_pair(index, owner_slot, neighbor_slot);
  if (candidate_score > worst_score) {
    neighbors[worst_pos] = neighbor_slot;
  }
}

void force_graph_neighbor(MemoryIndex* index, uint32_t owner_slot, uint32_t neighbor_slot, uint32_t position) {
  if (owner_slot == neighbor_slot || position >= index->graph_neighbor_capacity) {
    return;
  }
  uint32_t* neighbors = graph_neighbors_at(index, owner_slot);
  uint32_t& count = index->graph_neighbor_counts[owner_slot];
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

void insert_graph_build_candidate(MemoryIndex* index, uint32_t* filled, uint32_t slot, float score) {
  uint32_t pos = *filled;
  if (pos < index->graph_neighbor_capacity) {
    ++(*filled);
  } else if (score <= index->graph_build_scores[index->graph_neighbor_capacity - 1u]) {
    return;
  } else {
    pos = index->graph_neighbor_capacity - 1u;
  }

  while (pos > 0 && score > index->graph_build_scores[pos - 1u]) {
    index->graph_build_scores[pos] = index->graph_build_scores[pos - 1u];
    index->graph_build_slots[pos] = index->graph_build_slots[pos - 1u];
    --pos;
  }
  index->graph_build_scores[pos] = score;
  index->graph_build_slots[pos] = slot;
}

void graph_collect_neighbors_exact(MemoryIndex* index, uint32_t slot, uint32_t* filled) {
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t other = active_slot_at(index, active_pos);
    if (other == slot || index->slots[other].occupied == 0) {
      continue;
    }
    insert_graph_build_candidate(index, filled, other, score_pair(index, slot, other));
  }
}

void graph_collect_neighbors_bounded(MemoryIndex* index, uint32_t slot, uint32_t* filled) {
  uint32_t entry = index->graph_entry_slot;
  if (entry == kU32Max || index->slots[entry].occupied == 0 || entry == slot) {
    entry = active_slot_at(index, 0);
  }
  if (entry == slot || index->slots[entry].occupied == 0) {
    return;
  }

  graph_begin_visit(index);
  uint32_t candidate_count = 0;
  uint32_t expanded_count = 0;
  graph_mark_visited(index, entry);
  graph_add_candidate(index, entry, score_pair(index, slot, entry), &candidate_count);

  while (candidate_count != 0 && expanded_count < index->graph_search_capacity) {
    uint32_t current = kU32Max;
    float current_score = kWorstScore;
    if (!graph_pop_candidate(index, &candidate_count, &current, &current_score)) {
      break;
    }

    ++expanded_count;
    insert_graph_build_candidate(index, filled, current, current_score);

    const uint32_t* neighbors = graph_neighbors_at(index, current);
    const uint32_t neighbor_count = index->graph_neighbor_counts[current];
    for (uint32_t i = 0; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (neighbor == slot || graph_was_visited(index, neighbor) || index->slots[neighbor].occupied == 0) {
        continue;
      }
      graph_mark_visited(index, neighbor);
      graph_add_candidate(index, neighbor, score_pair(index, slot, neighbor), &candidate_count);
    }
  }
}

void graph_connect_slot(MemoryIndex* index, uint32_t slot) {
  if (!graph_enabled(index) || index->count <= 1u) {
    index->graph_entry_slot = slot;
    return;
  }

  uint32_t filled = 0;
  if (index->count <= index->graph_search_capacity) {
    graph_collect_neighbors_exact(index, slot, &filled);
  } else {
    graph_collect_neighbors_bounded(index, slot, &filled);
  }

  uint32_t* neighbors = graph_neighbors_at(index, slot);
  index->graph_neighbor_counts[slot] = filled;
  for (uint32_t i = 0; i < filled; ++i) {
    neighbors[i] = index->graph_build_slots[i];
    insert_graph_neighbor(index, index->graph_build_slots[i], slot);
  }
  if (index->count > index->graph_neighbor_capacity && index->graph_neighbor_capacity > kGraphLongLinkCount) {
    const uint32_t long_links = kGraphLongLinkCount < index->graph_neighbor_capacity ? kGraphLongLinkCount
                                                                                     : index->graph_neighbor_capacity;
    const uint32_t stride = index->count / long_links;
    const uint32_t first_long_pos = index->graph_neighbor_capacity - long_links;
    for (uint32_t i = 0; i < long_links; ++i) {
      const uint32_t linked = active_slot_at(index, i * stride);
      force_graph_neighbor(index, slot, linked, first_long_pos + i);
      force_graph_neighbor(index, linked, slot, first_long_pos + i);
    }
  }
  index->graph_entry_slot = slot;
}

void graph_rebuild(MemoryIndex* index) {
  if (!graph_enabled(index)) {
    return;
  }
  std::memset(index->graph_neighbor_counts, 0, sizeof(uint32_t) * index->capacity);
  index->graph_entry_slot = kU32Max;
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    graph_connect_slot(index, active_slot_at(index, active_pos));
  }
}

inline void graph_mark_visited(MemoryIndex* index, uint32_t slot) {
  index->graph_visited[slot] = index->graph_visit_generation;
}

inline bool graph_was_visited(const MemoryIndex* index, uint32_t slot) {
  return index->graph_visited[slot] == index->graph_visit_generation;
}

void graph_begin_visit(MemoryIndex* index) {
  ++index->graph_visit_generation;
  if (index->graph_visit_generation == 0) {
    std::memset(index->graph_visited, 0, sizeof(uint32_t) * index->capacity);
    index->graph_visit_generation = 1;
  }
}

void graph_add_candidate(MemoryIndex* index, uint32_t slot, float score, uint32_t* candidate_count) {
  uint32_t count = *candidate_count;
  if (count < index->graph_search_capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (index->graph_candidate_scores[parent] >= score) {
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

  uint32_t worst_pos = 0;
  float worst_score = index->graph_candidate_scores[0];
  for (uint32_t i = 1; i < count; ++i) {
    if (index->graph_candidate_scores[i] < worst_score) {
      worst_score = index->graph_candidate_scores[i];
      worst_pos = i;
    }
  }
  if (score > worst_score) {
    uint32_t pos = worst_pos;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (index->graph_candidate_scores[parent] >= score) {
        break;
      }
      index->graph_candidates[pos] = index->graph_candidates[parent];
      index->graph_candidate_scores[pos] = index->graph_candidate_scores[parent];
      pos = parent;
    }
    index->graph_candidates[pos] = slot;
    index->graph_candidate_scores[pos] = score;
  }
}

bool graph_pop_candidate(MemoryIndex* index, uint32_t* candidate_count, uint32_t* out_slot, float* out_score) {
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
      if (right < count && index->graph_candidate_scores[right] > index->graph_candidate_scores[left]) {
        child = right;
      }
      if (index->graph_candidate_scores[child] <= score) {
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
  return true;
}

void memory_search_graph(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                         AstralMemorySearchResult* out_results, uint32_t* out_count) {
  if (!graph_enabled(index) || index->count == 0 || desc->group_id != ASTRAL_MEMORY_GROUP_ANY ||
      index->graph_entry_slot == kU32Max || index->slots[index->graph_entry_slot].occupied == 0) {
    memory_search_flat(index, desc, query, out_results, out_count);
    return;
  }

  const float query_scale =
      index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(query, index->dim) : 1.0f;
  graph_begin_visit(index);

  uint32_t filled = 0;
  uint32_t candidate_count = 0;
  uint32_t expanded_count = 0;
  const uint32_t entry = index->graph_entry_slot;
  graph_mark_visited(index, entry);
  const float entry_score = score_slot(index, query, entry, query_scale);
  graph_add_candidate(index, entry, entry_score, &candidate_count);
  AstralMemorySearchResult entry_result{};
  fill_result(&entry_result, index->slots[entry], entry_score);
  insert_result(out_results, desc->top_k, &filled, entry_result);

  while (candidate_count != 0 && expanded_count < index->graph_search_capacity) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!graph_pop_candidate(index, &candidate_count, &slot, &slot_score)) {
      break;
    }
    (void)slot_score;

    ++expanded_count;
    const uint32_t* neighbors = graph_neighbors_at(index, slot);
    const uint32_t neighbor_count = index->graph_neighbor_counts[slot];
    for (uint32_t i = 0; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (graph_was_visited(index, neighbor) || index->slots[neighbor].occupied == 0) {
        continue;
      }
      graph_mark_visited(index, neighbor);
      const MemorySlot& s = index->slots[neighbor];
      const float score = score_slot(index, query, neighbor, query_scale);
      graph_add_candidate(index, neighbor, score, &candidate_count);
      if (filled == desc->top_k && !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
        continue;
      }
      AstralMemorySearchResult candidate{};
      fill_result(&candidate, s, score);
      insert_result(out_results, desc->top_k, &filled, candidate);
    }
  }

  *out_count = filled;
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
    if (filled == desc->top_k && !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
      continue;
    }

    AstralMemorySearchResult candidate{};
    fill_result(&candidate, s, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  *out_count = filled;
}

void memory_search_cosine(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                          AstralMemorySearchResult* out_results, uint32_t* out_count) {
  uint32_t filled = 0;
  const float query_scale = cosine_scale(query, index->dim);
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score = dot_f32(query, vector_at(index, slot), index->dim) * query_scale * s.score_scale;
    if (filled == desc->top_k && !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
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
    if (filled == desc->top_k && !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
      continue;
    }

    AstralMemorySearchResult candidate{};
    fill_result(&candidate, s, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  *out_count = filled;
}

void memory_search_dot_top1(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                            AstralMemorySearchResult* out_results, uint32_t* out_count) {
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
    if (best_slot == nullptr || score > best_score || (score == best_score && s.record.key < best_key)) {
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

void memory_search_cosine_top1(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                               AstralMemorySearchResult* out_results, uint32_t* out_count) {
  const float query_scale = cosine_scale(query, index->dim);
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
      float best_score = dot_f32(query, vectors, dim) * query_scale * best_slot->score_scale;
      uint64_t best_key = best_slot->record.key;
      for (uint32_t slot = 1; slot < index->count; ++slot) {
        MemorySlot& s = slots[slot];
        const float score = dot_f32(query, vectors + static_cast<size_t>(slot) * dim, dim) * query_scale * s.score_scale;
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
    float best_score = dot_f32(query, vector_at(index, first_slot), index->dim) * query_scale * best_slot->score_scale;
    uint64_t best_key = best_slot->record.key;
    for (uint32_t active_pos = 1; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const MemorySlot& s = index->slots[slot];
      const float score = dot_f32(query, vector_at(index, slot), index->dim) * query_scale * s.score_scale;
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

    const float score = dot_f32(query, vector_at(index, slot), index->dim) * query_scale * s.score_scale;
    if (best_slot == nullptr || score > best_score || (score == best_score && s.record.key < best_key)) {
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

void memory_search_l2_top1(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                           AstralMemorySearchResult* out_results, uint32_t* out_count) {
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
    if (best_slot == nullptr || score > best_score || (score == best_score && s.record.key < best_key)) {
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

void destroy_allocations(MemoryIndex* index) {
  if (index->slots != nullptr) {
    core::runtime_free_array(index->slots, index->capacity);
    index->slots = nullptr;
  }
  if (index->vectors != nullptr) {
    free_vector_storage(index->vectors, index->capacity, index->dim);
    index->vectors = nullptr;
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
                             index->capacity * index->graph_neighbor_capacity);
    index->graph_neighbors = nullptr;
  }
  if (index->graph_neighbor_counts != nullptr) {
    core::runtime_free_array(index->graph_neighbor_counts, index->capacity);
    index->graph_neighbor_counts = nullptr;
  }
  if (index->graph_candidates != nullptr) {
    core::runtime_free_array(index->graph_candidates, index->graph_search_capacity);
    index->graph_candidates = nullptr;
  }
  if (index->graph_candidate_scores != nullptr) {
    core::runtime_free_array(index->graph_candidate_scores, index->graph_search_capacity);
    index->graph_candidate_scores = nullptr;
  }
  if (index->graph_build_slots != nullptr) {
    core::runtime_free_array(index->graph_build_slots, index->graph_neighbor_capacity);
    index->graph_build_slots = nullptr;
  }
  if (index->graph_build_scores != nullptr) {
    core::runtime_free_array(index->graph_build_scores, index->graph_neighbor_capacity);
    index->graph_build_scores = nullptr;
  }
  if (index->graph_visited != nullptr) {
    core::runtime_free_array(index->graph_visited, index->capacity);
    index->graph_visited = nullptr;
  }
}

void destroy_search_cursor(MemorySearchCursor* cursor) {
  if (cursor == nullptr) {
    return;
  }
  if (cursor->results != nullptr) {
    core::runtime_free_array(cursor->results, cursor->capacity);
    cursor->results = nullptr;
  }
  core::runtime_delete(cursor);
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
  const uint32_t requested_graph_search = graph_search_from_desc(desc);
  const uint32_t graph_search_capacity =
      requested_graph_search > desc->capacity ? desc->capacity : requested_graph_search;

  MemoryIndex* index = core::runtime_new<MemoryIndex>();
  if (index == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  index->handle = 0;
  index->dim = desc->dim;
  index->capacity = desc->capacity;
  index->count = 0;
  index->metric = desc->metric;
  index->index_kind = desc->index_kind;
  index->slots = core::runtime_alloc_array<MemorySlot>(desc->capacity);
  index->vectors = alloc_vector_storage(desc->capacity, desc->dim);
  index->active_slots = core::runtime_alloc_array<uint32_t>(desc->capacity);
  index->key_table = core::runtime_alloc_array<uint32_t>(key_table_capacity);
  index->graph_neighbors = nullptr;
  index->graph_neighbor_counts = nullptr;
  index->graph_candidates = nullptr;
  index->graph_candidate_scores = nullptr;
  index->graph_build_slots = nullptr;
  index->graph_build_scores = nullptr;
  index->graph_visited = nullptr;
  index->key_table_capacity = key_table_capacity;
  index->key_table_mask = key_table_capacity - 1u;
  index->graph_neighbor_capacity = graph_neighbor_capacity;
  index->graph_search_capacity = graph_search_capacity;
  index->graph_entry_slot = kU32Max;
  index->graph_visit_generation = 0;
  index->dense_active = 1u;
  if (graph_neighbor_capacity != 0) {
    if (desc->capacity > kU32Max / graph_neighbor_capacity) {
      destroy_allocations(index);
      core::runtime_delete(index);
      return ASTRAL_E_NOMEM;
    }
    index->graph_neighbors =
        core::runtime_alloc_array<uint32_t>(desc->capacity * graph_neighbor_capacity);
    index->graph_neighbor_counts = core::runtime_alloc_array<uint32_t>(desc->capacity);
    index->graph_candidates = core::runtime_alloc_array<uint32_t>(graph_search_capacity);
    index->graph_candidate_scores = core::runtime_alloc_array<float>(graph_search_capacity);
    index->graph_build_slots = core::runtime_alloc_array<uint32_t>(graph_neighbor_capacity);
    index->graph_build_scores = core::runtime_alloc_array<float>(graph_neighbor_capacity);
    index->graph_visited = core::runtime_alloc_array<uint32_t>(desc->capacity);
  }
  if (index->slots == nullptr || index->vectors == nullptr || index->active_slots == nullptr ||
      index->key_table == nullptr ||
      (graph_neighbor_capacity != 0 &&
       (index->graph_neighbors == nullptr || index->graph_neighbor_counts == nullptr ||
        index->graph_candidates == nullptr || index->graph_candidate_scores == nullptr ||
        index->graph_build_slots == nullptr || index->graph_build_scores == nullptr ||
        index->graph_visited == nullptr))) {
    destroy_allocations(index);
    core::runtime_delete(index);
    return ASTRAL_E_NOMEM;
  }

  std::memset(index->slots, 0, sizeof(MemorySlot) * desc->capacity);
  std::memset(index->active_slots, 0, sizeof(uint32_t) * desc->capacity);
  std::memset(index->key_table, 0, sizeof(uint32_t) * key_table_capacity);
  if (graph_neighbor_capacity != 0) {
    std::memset(index->graph_neighbors, 0,
                sizeof(uint32_t) * desc->capacity * graph_neighbor_capacity);
    std::memset(index->graph_neighbor_counts, 0, sizeof(uint32_t) * desc->capacity);
    std::memset(index->graph_candidates, 0, sizeof(uint32_t) * graph_search_capacity);
    std::memset(index->graph_candidate_scores, 0, sizeof(float) * graph_search_capacity);
    std::memset(index->graph_build_slots, 0, sizeof(uint32_t) * graph_neighbor_capacity);
    std::memset(index->graph_build_scores, 0, sizeof(float) * graph_neighbor_capacity);
    std::memset(index->graph_visited, 0, sizeof(uint32_t) * desc->capacity);
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

AstralErr memory_clear(MemoryIndex* index) {
  if (index == nullptr) {
    return ASTRAL_E_INVALID;
  }
  std::memset(index->slots, 0, sizeof(MemorySlot) * index->capacity);
  std::memset(index->active_slots, 0, sizeof(uint32_t) * index->capacity);
  std::memset(index->key_table, 0, sizeof(uint32_t) * index->key_table_capacity);
  if (graph_enabled(index)) {
    std::memset(index->graph_neighbor_counts, 0, sizeof(uint32_t) * index->capacity);
    std::memset(index->graph_visited, 0, sizeof(uint32_t) * index->capacity);
    index->graph_entry_slot = kU32Max;
    index->graph_visit_generation = 0;
  }
  index->count = 0;
  index->free_slot_hint = 0;
  index->dense_active = 1u;
  return ASTRAL_OK;
}

AstralErr memory_add_batch(MemoryIndex* index, const AstralMemoryRecord* records,
                           const float* vectors, uint32_t count) {
  if (index == nullptr || records == nullptr || vectors == nullptr || count == 0) {
    return ASTRAL_E_INVALID;
  }

  bool graph_rebuild_needed = false;
  for (uint32_t i = 0; i < count; ++i) {
    if (records[i].size != sizeof(AstralMemoryRecord) || records[i].key == 0) {
      return ASTRAL_E_INVALID;
    }
    const uint64_t key_hash = key_hash_mix(records[i].key);
    uint32_t slot = find_slot_by_key_hashed(index, records[i].key, key_hash);
    const bool is_update = slot != kU32Max;
    if (slot == kU32Max) {
      slot = find_free_slot(index);
      if (slot == kU32Max) {
        return ASTRAL_E_NOMEM;
      }
      if (index->dense_active != 0 && slot != index->count) {
        index->dense_active = 0;
      }
      index->slots[slot].active_pos = index->count;
      index->active_slots[index->count] = slot;
      index->slots[slot].occupied = 1;
      ++index->count;
      index->free_slot_hint = slot + kSlotAdvance;
      if (index->free_slot_hint == index->capacity) {
        index->free_slot_hint = 0;
      }
      const AstralErr key_err = key_table_insert_new_hashed(index, key_hash, slot);
      if (key_err != ASTRAL_OK) {
        --index->count;
        index->slots[slot] = MemorySlot{};
        index->free_slot_hint = slot;
        return key_err;
      }
    }
    index->slots[slot].record = records[i];
    float* dst = vector_at(index, slot);
    const float* src = vectors + static_cast<size_t>(i) * index->dim;
    std::memcpy(dst, src, sizeof(float) * index->dim);
    index->slots[slot].score_scale =
        index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(dst, index->dim) : 0.0f;
    if (graph_enabled(index)) {
      if (is_update) {
        graph_rebuild_needed = true;
      } else {
        graph_connect_slot(index, slot);
      }
    }
  }
  if (graph_rebuild_needed) {
    graph_rebuild(index);
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

  if (index->index_kind == ASTRAL_MEMORY_INDEX_GRAPH && desc->top_k != kTopOne) {
    memory_search_graph(index, desc, query, out_results, out_count);
  } else {
    memory_search_flat(index, desc, query, out_results, out_count);
  }
  return ASTRAL_OK;
}

AstralErr memory_search_begin(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* query, MemorySearchCursor** out_cursor) {
  if (index == nullptr || desc == nullptr || desc->size != sizeof(AstralMemorySearchDesc) ||
      query == nullptr || out_cursor == nullptr || desc->top_k == 0) {
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
  cursor->results = core::runtime_alloc_array<AstralMemorySearchResult>(capacity);
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
  *out_bytes = sizeof(SaveHeader) +
               static_cast<uint64_t>(index->count) * sizeof(AstralMemoryRecord) +
               static_cast<uint64_t>(index->count) * index->dim * sizeof(float);
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

  SaveHeader header{};
  header.magic = kSaveMagic;
  header.version = kSaveVersion;
  header.dim = index->dim;
  header.count = index->count;
  header.metric = index->metric;
  header.index_kind = index->index_kind;

  uint8_t* cursor = out_bytes.data;
  std::memcpy(cursor, &header, sizeof(header));
  cursor += sizeof(header);

  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    std::memcpy(cursor, &index->slots[slot].record, sizeof(AstralMemoryRecord));
    cursor += sizeof(AstralMemoryRecord);
    std::memcpy(cursor, vector_at(index, slot), sizeof(float) * index->dim);
    cursor += sizeof(float) * index->dim;
  }
  return ASTRAL_OK;
}

AstralErr memory_load(const AstralMemoryIndexDesc* desc, AstralSpanU8 bytes,
                      MemoryIndex** out_index) {
  if (!desc_valid(desc) || bytes.data == nullptr || bytes.len < sizeof(SaveHeader) ||
      out_index == nullptr) {
    return ASTRAL_E_INVALID;
  }

  SaveHeader header{};
  std::memcpy(&header, bytes.data, sizeof(header));
  if (header.magic != kSaveMagic || header.version != kSaveVersion || header.dim != desc->dim ||
      header.metric != desc->metric || header.index_kind != desc->index_kind ||
      header.count > desc->capacity) {
    return ASTRAL_E_INVALID;
  }

  const uint64_t need = sizeof(SaveHeader) +
                        static_cast<uint64_t>(header.count) * sizeof(AstralMemoryRecord) +
                        static_cast<uint64_t>(header.count) * header.dim * sizeof(float);
  if (bytes.len < need) {
    return ASTRAL_E_INVALID;
  }

  MemoryIndex* index = nullptr;
  AstralErr err = memory_create(desc, &index);
  if (err != ASTRAL_OK) {
    return err;
  }

  const uint8_t* cursor = bytes.data + sizeof(SaveHeader);
  for (uint32_t i = 0; i < header.count; ++i) {
    AstralMemoryRecord record{};
    std::memcpy(&record, cursor, sizeof(record));
    cursor += sizeof(record);
    err = memory_add_batch(index, &record, reinterpret_cast<const float*>(cursor), 1);
    if (err != ASTRAL_OK) {
      memory_destroy(index);
      return err;
    }
    cursor += sizeof(float) * header.dim;
  }

  *out_index = index;
  return ASTRAL_OK;
}

} // namespace astral::inference
