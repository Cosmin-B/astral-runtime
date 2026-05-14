#include "memory_index.hpp"

#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
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
constexpr uint32_t kSaveVersionF32 = 1;
constexpr uint32_t kSaveVersionCompactStorage = 2;
constexpr uint32_t kSaveVersionGraphTopology = 3;
constexpr uint32_t kSaveVersion = 4;
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
constexpr uint32_t kGraphDefaultEfConstruction = 68;
constexpr uint32_t kGraphMinSearch = 4;
constexpr uint32_t kGraphQueryReserveMultiplier = 16;
constexpr uint32_t kGraphLongLinkCount = 0;
constexpr uint32_t kGraphNeighborPrefetchDistance = 2;
constexpr uint32_t kFlatQ8PrefetchDistance = 4;
constexpr uint32_t kFlatQ8PrefetchMinCount = 32768;
constexpr uint32_t kGraphMaxLevels = 16;
constexpr uint32_t kMemoryBatchStackQueries = 16;
constexpr int32_t kQ8MinValue = -127;
constexpr int32_t kQ8MaxValue = 127;
constexpr float kQ8MaxFloat = 127.0f;
constexpr float kL2CrossTermScale = 2.0f;
constexpr float kWorstScore = -3.4028234663852886e38f;
#if defined(__AVX2__)
constexpr uint32_t kAvx2F32Lanes = 8;
constexpr uint32_t kAvx2I8Lanes = 16;
constexpr size_t kVectorStorageAlign = 32;
constexpr uint32_t kAvx2UnrollVectors = 4;
constexpr uint32_t kAvx2UnrollF32 = kAvx2F32Lanes * kAvx2UnrollVectors;
constexpr uint32_t kAvx2UnrollI8 = kAvx2I8Lanes * kAvx2UnrollVectors;
constexpr uint32_t kAvx2Offset1 = kAvx2F32Lanes;
constexpr uint32_t kAvx2Offset2 = kAvx2F32Lanes * 2u;
constexpr uint32_t kAvx2Offset3 = kAvx2F32Lanes * 3u;
constexpr uint32_t kAvx2I8Offset1 = kAvx2I8Lanes;
constexpr uint32_t kAvx2I8Offset2 = kAvx2I8Lanes * 2u;
constexpr uint32_t kAvx2I8Offset3 = kAvx2I8Lanes * 3u;
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

struct SaveGraphHeader {
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

inline void prefetch_dense_q8_slot(const int8_t* vectors, const float* scales,
                                   const MemorySlot* slots, uint32_t dim, uint32_t slot) {
  __builtin_prefetch(vectors + static_cast<size_t>(slot) * dim, 0, 1);
  __builtin_prefetch(scales + slot, 0, 1);
  __builtin_prefetch(slots + slot, 0, 1);
}

inline bool metric_valid(AstralMemoryMetric metric) {
  return metric == ASTRAL_MEMORY_METRIC_DOT || metric == ASTRAL_MEMORY_METRIC_COSINE ||
         metric == ASTRAL_MEMORY_METRIC_L2;
}

inline bool index_kind_valid(AstralMemoryIndexKind kind) {
  return kind == ASTRAL_MEMORY_INDEX_FLAT || kind == ASTRAL_MEMORY_INDEX_GRAPH;
}

inline bool storage_kind_valid(AstralMemoryStorageKind kind) {
  return kind == ASTRAL_MEMORY_STORAGE_F32 || kind == ASTRAL_MEMORY_STORAGE_Q8;
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
  return doubled < kGraphMaxNeighbors ? doubled : kGraphMaxNeighbors;
}

inline uint32_t graph_search_from_desc(const AstralMemoryIndexDesc* desc) {
  if (desc->index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    return 0;
  }
  const uint32_t requested =
      desc->graph_search != 0 ? desc->graph_search : kGraphDefaultEfConstruction;
  return requested < kGraphMinSearch ? kGraphMinSearch : requested;
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

inline void free_vector_storage(float* ptr, uint32_t capacity, uint32_t dim) {
  core::runtime_free(ptr, static_cast<size_t>(capacity) * dim * sizeof(float), kVectorStorageAlign);
}

inline void free_q8_vector_storage(int8_t* ptr, uint32_t capacity, uint32_t dim) {
  core::runtime_free(ptr, static_cast<size_t>(capacity) * dim * sizeof(int8_t),
                     kVectorStorageAlign);
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

inline int32_t reduce_avx2_i32(__m256i acc) {
  const __m128i lo = _mm256_castsi256_si128(acc);
  const __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
  return _mm_cvtsi128_si32(sum);
}

inline __m256i dot_i8x16_avx2(const int8_t* a, const int8_t* b, __m256i ones) {
  const __m128i av = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
  const __m128i bv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
  const __m256i a16 = _mm256_cvtepi8_epi16(av);
  const __m256i b16 = _mm256_cvtepi8_epi16(bv);
  return _mm256_madd_epi16(_mm256_mullo_epi16(a16, b16), ones);
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
    const __m256 d1 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + kAvx2Offset1), _mm256_loadu_ps(b + i + kAvx2Offset1));
    const __m256 d2 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + kAvx2Offset2), _mm256_loadu_ps(b + i + kAvx2Offset2));
    const __m256 d3 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + kAvx2Offset3), _mm256_loadu_ps(b + i + kAvx2Offset3));
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
    const float32x4_t d1 =
        vsubq_f32(vld1q_f32(a + i + kNeonOffset1), vld1q_f32(b + i + kNeonOffset1));
    const float32x4_t d2 =
        vsubq_f32(vld1q_f32(a + i + kNeonOffset2), vld1q_f32(b + i + kNeonOffset2));
    const float32x4_t d3 =
        vsubq_f32(vld1q_f32(a + i + kNeonOffset3), vld1q_f32(b + i + kNeonOffset3));
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

float dot_q8_f32(const int8_t* a, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + kAvx2UnrollF32 <= dim; i += kAvx2UnrollF32) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
    const __m256 av0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
    const __m128i bytes_hi = _mm_srli_si128(bytes, 8);
    const __m256 av1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_hi));
    const __m128i bytes2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + kAvx2I8Lanes));
    const __m256 av2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes2));
    const __m128i bytes2_hi = _mm_srli_si128(bytes2, 8);
    const __m256 av3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes2_hi));
    const __m256 bv0 = _mm256_loadu_ps(b + i);
    const __m256 bv1 = _mm256_loadu_ps(b + i + kAvx2F32Lanes);
    const __m256 bv2 = _mm256_loadu_ps(b + i + kAvx2Offset2);
    const __m256 bv3 = _mm256_loadu_ps(b + i + kAvx2Offset3);
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
    acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
    acc2 = _mm256_fmadd_ps(av2, bv2, acc2);
    acc3 = _mm256_fmadd_ps(av3, bv3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av0, bv0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(av1, bv1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(av2, bv2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(av3, bv3));
#endif
  }
  acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
    const __m256 av0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
    const __m128i bytes_hi = _mm_srli_si128(bytes, 8);
    const __m256 av1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_hi));
    const __m256 bv0 = _mm256_loadu_ps(b + i);
    const __m256 bv1 = _mm256_loadu_ps(b + i + kAvx2F32Lanes);
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
    acc0 = _mm256_fmadd_ps(av1, bv1, acc0);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av0, bv0));
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av1, bv1));
#endif
  }
  float sum = reduce_avx2_f32(acc0);
  for (; i < dim; ++i) {
    sum += static_cast<float>(a[i]) * b[i];
  }
  return sum;
#else
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    sum0 += static_cast<float>(a[i]) * b[i];
    sum1 += static_cast<float>(a[i + 1u]) * b[i + 1u];
    sum2 += static_cast<float>(a[i + 2u]) * b[i + 2u];
    sum3 += static_cast<float>(a[i + 3u]) * b[i + 3u];
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    sum += static_cast<float>(a[i]) * b[i];
  }
  return sum;
#endif
}

float dot_q8_q8(const int8_t* a, const int8_t* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256();
  __m256i acc3 = _mm256_setzero_si256();
  const __m256i ones = _mm256_set1_epi16(1);
  uint32_t i = 0;
  for (; i + kAvx2UnrollI8 <= dim; i += kAvx2UnrollI8) {
    acc0 = _mm256_add_epi32(acc0, dot_i8x16_avx2(a + i, b + i, ones));
    acc1 = _mm256_add_epi32(acc1,
                            dot_i8x16_avx2(a + i + kAvx2I8Offset1, b + i + kAvx2I8Offset1, ones));
    acc2 = _mm256_add_epi32(acc2,
                            dot_i8x16_avx2(a + i + kAvx2I8Offset2, b + i + kAvx2I8Offset2, ones));
    acc3 = _mm256_add_epi32(acc3,
                            dot_i8x16_avx2(a + i + kAvx2I8Offset3, b + i + kAvx2I8Offset3, ones));
  }
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    acc0 = _mm256_add_epi32(acc0, dot_i8x16_avx2(a + i, b + i, ones));
  }
  acc0 = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
  int32_t sum = reduce_avx2_i32(acc0);
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
#else
  int32_t sum0 = 0;
  int32_t sum1 = 0;
  int32_t sum2 = 0;
  int32_t sum3 = 0;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    sum0 += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    sum1 += static_cast<int32_t>(a[i + 1u]) * static_cast<int32_t>(b[i + 1u]);
    sum2 += static_cast<int32_t>(a[i + 2u]) * static_cast<int32_t>(b[i + 2u]);
    sum3 += static_cast<int32_t>(a[i + 3u]) * static_cast<int32_t>(b[i + 3u]);
  }
  int32_t sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
#endif
}

float l2_score_q8_f32(const int8_t* a, float scale, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  const __m256 scale_v = _mm256_set1_ps(scale);
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + kAvx2UnrollF32 <= dim; i += kAvx2UnrollF32) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
    const __m256 av0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)), scale_v);
    const __m128i bytes_hi = _mm_srli_si128(bytes, 8);
    const __m256 av1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_hi)), scale_v);
    const __m128i bytes2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + kAvx2I8Lanes));
    const __m256 av2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes2)), scale_v);
    const __m128i bytes2_hi = _mm_srli_si128(bytes2, 8);
    const __m256 av3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes2_hi)), scale_v);
    const __m256 d0 = _mm256_sub_ps(av0, _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(av1, _mm256_loadu_ps(b + i + kAvx2F32Lanes));
    const __m256 d2 = _mm256_sub_ps(av2, _mm256_loadu_ps(b + i + kAvx2Offset2));
    const __m256 d3 = _mm256_sub_ps(av3, _mm256_loadu_ps(b + i + kAvx2Offset3));
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
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
    const __m256 av0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)), scale_v);
    const __m128i bytes_hi = _mm_srli_si128(bytes, 8);
    const __m256 av1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_hi)), scale_v);
    const __m256 d0 = _mm256_sub_ps(av0, _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(av1, _mm256_loadu_ps(b + i + kAvx2F32Lanes));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc0 = _mm256_fmadd_ps(d1, d1, acc0);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d1, d1));
#endif
  }
  float sum = reduce_avx2_f32(acc0);
  for (; i < dim; ++i) {
    const float d = static_cast<float>(a[i]) * scale - b[i];
    sum += d * d;
  }
  return -sum;
#else
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    const float d0 = static_cast<float>(a[i]) * scale - b[i];
    const float d1 = static_cast<float>(a[i + 1u]) * scale - b[i + 1u];
    const float d2 = static_cast<float>(a[i + 2u]) * scale - b[i + 2u];
    const float d3 = static_cast<float>(a[i + 3u]) * scale - b[i + 3u];
    sum0 += d0 * d0;
    sum1 += d1 * d1;
    sum2 += d2 * d2;
    sum3 += d3 * d3;
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    const float d = static_cast<float>(a[i]) * scale - b[i];
    sum += d * d;
  }
  return -sum;
#endif
}

float l2_score_q8_q8(const int8_t* a, float a_scale, const int8_t* b, float b_scale, uint32_t dim) {
#if defined(__AVX2__)
  __m256i acc_a = _mm256_setzero_si256();
  __m256i acc_b = _mm256_setzero_si256();
  __m256i acc_ab = _mm256_setzero_si256();
  const __m256i ones = _mm256_set1_epi16(1);
  uint32_t i = 0;
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    const __m128i av = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
    const __m128i bv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
    const __m256i a16 = _mm256_cvtepi8_epi16(av);
    const __m256i b16 = _mm256_cvtepi8_epi16(bv);
    acc_a = _mm256_add_epi32(acc_a, _mm256_madd_epi16(_mm256_mullo_epi16(a16, a16), ones));
    acc_b = _mm256_add_epi32(acc_b, _mm256_madd_epi16(_mm256_mullo_epi16(b16, b16), ones));
    acc_ab = _mm256_add_epi32(acc_ab, _mm256_madd_epi16(_mm256_mullo_epi16(a16, b16), ones));
  }

  int32_t sum_a = reduce_avx2_i32(acc_a);
  int32_t sum_b = reduce_avx2_i32(acc_b);
  int32_t sum_ab = reduce_avx2_i32(acc_ab);
  for (; i < dim; ++i) {
    const int32_t ai = static_cast<int32_t>(a[i]);
    const int32_t bi = static_cast<int32_t>(b[i]);
    sum_a += ai * ai;
    sum_b += bi * bi;
    sum_ab += ai * bi;
  }

  const float scaled_a = a_scale * a_scale * static_cast<float>(sum_a);
  const float scaled_b = b_scale * b_scale * static_cast<float>(sum_b);
  const float scaled_cross = kL2CrossTermScale * a_scale * b_scale * static_cast<float>(sum_ab);
  return -(scaled_a + scaled_b - scaled_cross);
#else
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    const float d0 = static_cast<float>(a[i]) * a_scale - static_cast<float>(b[i]) * b_scale;
    const float d1 =
        static_cast<float>(a[i + 1u]) * a_scale - static_cast<float>(b[i + 1u]) * b_scale;
    const float d2 =
        static_cast<float>(a[i + 2u]) * a_scale - static_cast<float>(b[i + 2u]) * b_scale;
    const float d3 =
        static_cast<float>(a[i + 3u]) * a_scale - static_cast<float>(b[i + 3u]) * b_scale;
    sum0 += d0 * d0;
    sum1 += d1 * d1;
    sum2 += d2 * d2;
    sum3 += d3 * d3;
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    const float d = static_cast<float>(a[i]) * a_scale - static_cast<float>(b[i]) * b_scale;
    sum += d * d;
  }
  return -sum;
#endif
}

inline int32_t round_scaled_q8(float value) {
  const float biased = value >= 0.0f ? value + 0.5f : value - 0.5f;
  return static_cast<int32_t>(biased);
}

void quantize_q8_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim) {
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float abs_value = std::fabs(src[i]);
    if (abs_value > max_abs) {
      max_abs = abs_value;
    }
  }
  const float scale = max_abs > 0.0f ? max_abs / kQ8MaxFloat : 1.0f;
  const float inv_scale = 1.0f / scale;
  for (uint32_t i = 0; i < dim; ++i) {
    int32_t v = round_scaled_q8(src[i] * inv_scale);
    if (v < kQ8MinValue) {
      v = kQ8MinValue;
    } else if (v > kQ8MaxValue) {
      v = kQ8MaxValue;
    }
    dst[i] = static_cast<int8_t>(v);
  }
  *out_scale = scale;
}

inline float vector_norm(const float* v, uint32_t dim) {
  const float sq = dot_f32(v, v, dim);
  return sq > 0.0f ? std::sqrt(sq) : 0.0f;
}

inline float cosine_scale(const float* v, uint32_t dim) {
  const float norm = vector_norm(v, dim);
  return norm > 0.0f ? 1.0f / norm : 0.0f;
}

inline float cosine_scale_q8(const int8_t* v, float scale, uint32_t dim) {
  const float sq = dot_q8_q8(v, v, dim) * scale * scale;
  return sq > 0.0f ? 1.0f / std::sqrt(sq) : 0.0f;
}

} // namespace

struct MemoryIndex {
  AstralHandle handle;
  uint32_t dim;
  uint32_t capacity;
  uint32_t count;
  AstralMemoryMetric metric;
  AstralMemoryIndexKind index_kind;
  AstralMemoryStorageKind storage_kind;
  MemorySlot* slots;
  float* vectors;
  int8_t* q8_vectors;
  float* q8_scales;
  uint32_t* active_slots;
  uint32_t* key_table;
  uint32_t* graph_neighbors;
  uint32_t* graph_neighbor_counts;
  uint8_t* graph_levels;
  uint32_t* graph_candidates;
  float* graph_candidate_scores;
  uint32_t* graph_scratch_slots;
  float* graph_scratch_scores;
  uint32_t* graph_visited;
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
  uint32_t graph_visit_generation;
  uint64_t graph_build_score_evals;
  uint64_t graph_build_candidate_visits;
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

namespace {

inline uint64_t memory_save_byte_count(const MemoryIndex* index) {
  const uint64_t vector_bytes =
      index->storage_kind == ASTRAL_MEMORY_STORAGE_Q8
          ? static_cast<uint64_t>(index->count) *
                (sizeof(float) + static_cast<uint64_t>(index->dim) * sizeof(int8_t))
          : static_cast<uint64_t>(index->count) * index->dim * sizeof(float);
  uint64_t bytes = sizeof(SaveHeader) +
                   static_cast<uint64_t>(index->count) * sizeof(AstralMemoryRecord) + vector_bytes;
  if (index->index_kind == ASTRAL_MEMORY_INDEX_GRAPH && index->graph_neighbor_capacity != 0) {
    const uint64_t count = index->count;
    const uint64_t neighbor_slots =
        count * index->graph_base_neighbor_capacity +
        count * (index->graph_level_capacity - 1u) * index->graph_neighbor_capacity;
    bytes += sizeof(SaveGraphHeader);
    bytes += static_cast<uint64_t>(index->count) * sizeof(uint8_t);
    bytes += static_cast<uint64_t>(index->count) * index->graph_level_capacity * sizeof(uint32_t);
    bytes += neighbor_slots * sizeof(uint32_t);
  }
  return bytes;
}

} // namespace

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

inline const float* vector_at(const MemoryIndex* index, uint32_t slot) {
  return index->vectors + static_cast<size_t>(slot) * index->dim;
}

inline int8_t* q8_vector_at(MemoryIndex* index, uint32_t slot) {
  return index->q8_vectors + static_cast<size_t>(slot) * index->dim;
}

inline const int8_t* q8_vector_at(const MemoryIndex* index, uint32_t slot) {
  return index->q8_vectors + static_cast<size_t>(slot) * index->dim;
}

inline bool q8_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_Q8;
}

inline void prefetch_slot_vector(const MemoryIndex* index, uint32_t slot) {
#if defined(__GNUC__) || defined(__clang__)
  if (q8_storage(index)) {
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
  uint32_t requested = desc->graph_search != 0 ? desc->graph_search : index->graph_search_capacity;
  if (requested < kGraphMinSearch) {
    requested = kGraphMinSearch;
  }
  return requested < index->graph_query_search_capacity ? requested
                                                        : index->graph_query_search_capacity;
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

inline uint32_t& graph_neighbor_count_ref(MemoryIndex* index, uint32_t slot, uint32_t level) {
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
  if (q8_storage(index)) {
    const int8_t* q8 = q8_vector_at(index, slot);
    const float scale = index->q8_scales[slot];
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return dot_q8_f32(q8, query, index->dim) * scale;
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return dot_q8_f32(q8, query, index->dim) * scale * query_scale *
             index->slots[slot].score_scale;
    }
    return l2_score_q8_f32(q8, scale, query, index->dim);
  }
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
  ++index->graph_build_score_evals;
  if (q8_storage(index)) {
    const int8_t* va = q8_vector_at(index, a);
    const int8_t* vb = q8_vector_at(index, b);
    const float scale_a = index->q8_scales[a];
    const float scale_b = index->q8_scales[b];
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return dot_q8_q8(va, vb, index->dim) * scale_a * scale_b * index->slots[a].score_scale *
             index->slots[b].score_scale;
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
      return l2_score_q8_q8(va, scale_a, vb, scale_b, index->dim);
    }
    return dot_q8_q8(va, vb, index->dim) * scale_a * scale_b;
  }
  const float* va = vector_at(index, a);
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return dot_f32(va, vector_at(index, b), index->dim) * index->slots[a].score_scale *
           index->slots[b].score_scale;
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
void graph_begin_visit(MemoryIndex* index);
void graph_add_candidate(MemoryIndex* index, uint32_t capacity, uint32_t slot, float score,
                         uint32_t* candidate_count);
bool graph_pop_candidate(MemoryIndex* index, uint32_t* candidate_count, uint32_t* out_slot,
                         float* out_score);
inline void graph_mark_visited(MemoryIndex* index, uint32_t slot);
inline bool graph_was_visited(const MemoryIndex* index, uint32_t slot);
void insert_graph_build_candidate(MemoryIndex* index, uint32_t capacity, uint32_t* filled,
                                  uint32_t slot, float score);
bool graph_neighbor_diverse(MemoryIndex* index, uint32_t candidate_slot, float candidate_score,
                            const uint32_t* neighbors, uint32_t count);
void graph_select_neighbors(MemoryIndex* index, uint32_t owner_slot, uint32_t level,
                            uint32_t candidate_count, uint32_t* neighbors, uint32_t* out_count);

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
                                uint32_t level) {
  if (owner_slot == candidate_slot) {
    return;
  }

  uint32_t* neighbors = graph_neighbors_at_level(index, owner_slot, level);
  uint32_t& count = graph_neighbor_count_ref(index, owner_slot, level);
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

  uint32_t filled = 0;
  const uint32_t refine_capacity = capacity + 1u;
  const float candidate_score = score_pair(index, owner_slot, candidate_slot);
  float weakest_score = kWorstScore;
  for (uint32_t i = 0; i < count; ++i) {
    const float score = score_pair(index, owner_slot, neighbors[i]);
    insert_graph_build_candidate(index, refine_capacity, &filled, neighbors[i], score);
    if (score < weakest_score) {
      weakest_score = score;
    }
  }
  if (candidate_score <= weakest_score) {
    return;
  }

  insert_graph_build_candidate(index, refine_capacity, &filled, candidate_slot, candidate_score);

  uint32_t selected = 0;
  graph_select_neighbors(index, owner_slot, level, filled, neighbors, &selected);
  count = selected;
}

void force_graph_neighbor(MemoryIndex* index, uint32_t owner_slot, uint32_t neighbor_slot,
                          uint32_t position, uint32_t level) {
  if (owner_slot == neighbor_slot || position >= graph_neighbor_capacity_at_level(index, level)) {
    return;
  }
  uint32_t* neighbors = graph_neighbors_at_level(index, owner_slot, level);
  uint32_t& count = graph_neighbor_count_ref(index, owner_slot, level);
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

void graph_select_neighbors(MemoryIndex* index, uint32_t owner_slot, uint32_t level,
                            uint32_t candidate_count, uint32_t* neighbors, uint32_t* out_count) {
  uint32_t selected = 0;
  const uint32_t capacity = graph_neighbor_capacity_at_level(index, level);
  for (uint32_t i = 0; i < candidate_count && selected < capacity; ++i) {
    const uint32_t candidate = index->graph_scratch_slots[i];
    const float candidate_score = index->graph_scratch_scores[i];
    if (candidate == owner_slot || graph_neighbor_selected(neighbors, selected, candidate)) {
      continue;
    }
    if (graph_neighbor_diverse(index, candidate, candidate_score, neighbors, selected)) {
      neighbors[selected] = candidate;
      ++selected;
    }
  }

  for (uint32_t i = 0; i < candidate_count && selected < capacity; ++i) {
    const uint32_t candidate = index->graph_scratch_slots[i];
    if (candidate == owner_slot || graph_neighbor_selected(neighbors, selected, candidate)) {
      continue;
    }
    neighbors[selected] = candidate;
    ++selected;
  }
  *out_count = selected;
}

bool insert_graph_top_candidate(MemoryIndex* index, uint32_t capacity, uint32_t* filled,
                                uint32_t slot, float score) {
  uint32_t count = *filled;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, index->graph_scratch_scores[parent],
                                  index->graph_scratch_slots[parent], score, slot)) {
        break;
      }
      index->graph_scratch_scores[pos] = index->graph_scratch_scores[parent];
      index->graph_scratch_slots[pos] = index->graph_scratch_slots[parent];
      pos = parent;
    }
    index->graph_scratch_scores[pos] = score;
    index->graph_scratch_slots[pos] = slot;
    *filled = count + 1u;
    return true;
  }

  if (!graph_candidate_better(index, score, slot, index->graph_scratch_scores[0],
                              index->graph_scratch_slots[0])) {
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
        graph_candidate_worse(index, index->graph_scratch_scores[right],
                              index->graph_scratch_slots[right], index->graph_scratch_scores[left],
                              index->graph_scratch_slots[left])) {
      child = right;
    }
    if (!graph_candidate_worse(index, index->graph_scratch_scores[child],
                               index->graph_scratch_slots[child], score, slot)) {
      break;
    }
    index->graph_scratch_scores[pos] = index->graph_scratch_scores[child];
    index->graph_scratch_slots[pos] = index->graph_scratch_slots[child];
    pos = child;
  }
  index->graph_scratch_scores[pos] = score;
  index->graph_scratch_slots[pos] = slot;
  return true;
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
  graph_mark_visited(index, entry);
  const float entry_score = score_pair(index, slot, entry);
  insert_graph_build_candidate(index, capacity, filled, entry, entry_score);
  graph_add_candidate(index, capacity, entry, entry_score, &candidate_count);

  while (candidate_count != 0) {
    uint32_t current = kU32Max;
    float current_score = kWorstScore;
    if (!graph_pop_candidate(index, &candidate_count, &current, &current_score)) {
      break;
    }
    if (*filled == capacity && current_score < index->graph_scratch_scores[capacity - 1u]) {
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
      if (*filled < capacity || score > index->graph_scratch_scores[capacity - 1u]) {
        graph_add_candidate(index, capacity, neighbor, score, &candidate_count);
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
    graph_select_neighbors(index, slot, level, candidate_count, neighbors, &filled);
    graph_neighbor_count_ref(index, slot, level) = filled;
    for (uint32_t i = 0; i < filled; ++i) {
      refine_graph_neighbor_list(index, neighbors[i], slot, level);
    }
    const uint32_t level_capacity = graph_neighbor_capacity_at_level(index, level);
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
              sizeof(uint32_t) * index->capacity * index->graph_level_capacity);
  index->graph_entry_slot = kU32Max;
  index->graph_max_level = 0;
  index->graph_build_score_evals = 0;
  index->graph_build_candidate_visits = 0;
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

  uint32_t worst_pos = 0;
  float worst_score = index->graph_candidate_scores[0];
  for (uint32_t i = 1; i < count; ++i) {
    if (graph_candidate_worse(index, index->graph_candidate_scores[i], index->graph_candidates[i],
                              worst_score, index->graph_candidates[worst_pos])) {
      worst_score = index->graph_candidate_scores[i];
      worst_pos = i;
    }
  }
  if (graph_candidate_better(index, score, slot, worst_score, index->graph_candidates[worst_pos])) {
    uint32_t pos = worst_pos;
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
  const uint32_t search_capacity = graph_search_for_query(index, desc);

  uint32_t filled = 0;
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  uint32_t expanded_count = 0;
  const uint32_t entry =
      index->graph_max_level != 0
          ? graph_greedy_closest_query(index, query, query_scale, index->graph_entry_slot,
                                       index->graph_max_level, 0)
          : index->graph_entry_slot;
  graph_mark_visited(index, entry);
  const float entry_score = score_slot(index, query, entry, query_scale);
  insert_graph_top_candidate(index, search_capacity, &top_count, entry, entry_score);
  graph_add_candidate(index, search_capacity, entry, entry_score, &candidate_count);

  while (candidate_count != 0 && expanded_count < search_capacity) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!graph_pop_candidate(index, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == search_capacity && slot_score < index->graph_scratch_scores[0]) {
      break;
    }

    ++expanded_count;
    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
    for (uint32_t i = 0; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (i + kGraphNeighborPrefetchDistance < neighbor_count) {
        prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      }
      if (graph_was_visited(index, neighbor) || index->slots[neighbor].occupied == 0) {
        continue;
      }
      graph_mark_visited(index, neighbor);
      const float score = score_slot(index, query, neighbor, query_scale);
      if (insert_graph_top_candidate(index, search_capacity, &top_count, neighbor, score)) {
        graph_add_candidate(index, search_capacity, neighbor, score, &candidate_count);
      }
    }
  }

  for (uint32_t i = 0; i < top_count; ++i) {
    const uint32_t slot = index->graph_scratch_slots[i];
    const MemorySlot& s = index->slots[slot];
    AstralMemorySearchResult candidate{};
    fill_result(&candidate, s, index->graph_scratch_scores[i]);
    insert_result(out_results, desc->top_k, &filled, candidate);
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

void memory_search_q8(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                      AstralMemorySearchResult* out_results, uint32_t* out_count) {
  const float query_scale =
      index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(query, index->dim) : 0.0f;
  if (desc->group_id == ASTRAL_MEMORY_GROUP_ANY && index->dense_active != 0) {
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
        const float score =
            dot_q8_f32(vectors + static_cast<size_t>(slot) * dim, query, dim) * scales[slot];
        if (filled == desc->top_k &&
            !result_better_values(score, s.record.key, out_results[desc->top_k - 1u])) {
          continue;
        }
        AstralMemorySearchResult candidate{};
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
                            scales[slot] * query_scale * s.score_scale;
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
        const float score =
            l2_score_q8_f32(vectors + static_cast<size_t>(slot) * dim, scales[slot], query, dim);
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

    if (index->dense_active != 0) {
      MemorySlot* slots = index->slots;
      const uint32_t dim = index->dim;
      const int8_t* vectors = index->q8_vectors;
      const float* scales = index->q8_scales;
      const bool use_prefetch = index->count >= kFlatQ8PrefetchMinCount;
      MemorySlot* best_slot = &slots[0];
      float best_score = 0.0f;
      if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
        best_score = dot_q8_f32(vectors, query, dim) * scales[0];
        for (uint32_t slot = 1; slot < index->count; ++slot) {
          if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
            prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
          }
          MemorySlot& s = slots[slot];
          const float score =
              dot_q8_f32(vectors + static_cast<size_t>(slot) * dim, query, dim) * scales[slot];
          if (score > best_score || (score == best_score && s.record.key < best_slot->record.key)) {
            best_slot = &s;
            best_score = score;
          }
        }
      } else if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        best_score =
            dot_q8_f32(vectors, query, dim) * scales[0] * query_scale * best_slot->score_scale;
        for (uint32_t slot = 1; slot < index->count; ++slot) {
          if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
            prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
          }
          MemorySlot& s = slots[slot];
          const float score = dot_q8_f32(vectors + static_cast<size_t>(slot) * dim, query, dim) *
                              scales[slot] * query_scale * s.score_scale;
          if (score > best_score || (score == best_score && s.record.key < best_slot->record.key)) {
            best_slot = &s;
            best_score = score;
          }
        }
      } else {
        best_score = l2_score_q8_f32(vectors, scales[0], query, dim);
        for (uint32_t slot = 1; slot < index->count; ++slot) {
          if (use_prefetch && slot + kFlatQ8PrefetchDistance < index->count) {
            prefetch_dense_q8_slot(vectors, scales, slots, dim, slot + kFlatQ8PrefetchDistance);
          }
          MemorySlot& s = slots[slot];
          const float score =
              l2_score_q8_f32(vectors + static_cast<size_t>(slot) * dim, scales[slot], query, dim);
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
  const float query_scale = cosine_scale(query, index->dim);
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    const MemorySlot& s = index->slots[slot];
    if (!record_matches_group(desc, s)) {
      continue;
    }

    const float score =
        dot_f32(query, vector_at(index, slot), index->dim) * query_scale * s.score_scale;
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
        const float score = dot_f32(query, vectors + static_cast<size_t>(slot) * dim, dim) *
                            query_scale * s.score_scale;
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
    float best_score = dot_f32(query, vector_at(index, first_slot), index->dim) * query_scale *
                       best_slot->score_scale;
    uint64_t best_key = best_slot->record.key;
    for (uint32_t active_pos = 1; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const MemorySlot& s = index->slots[slot];
      const float score =
          dot_f32(query, vector_at(index, slot), index->dim) * query_scale * s.score_scale;
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

    const float score =
        dot_f32(query, vector_at(index, slot), index->dim) * query_scale * s.score_scale;
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
  if (q8_storage(index)) {
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

  if (q8_storage(index) && desc->group_id == ASTRAL_MEMORY_GROUP_ANY && index->dense_active != 0) {
    MemorySlot* slots = index->slots;
    const uint32_t dim = index->dim;
    const int8_t* vectors = index->q8_vectors;
    const float* scales = index->q8_scales;
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
      for (uint32_t slot = 0; slot < index->count; ++slot) {
        MemorySlot& s = slots[slot];
        const int8_t* vector = vectors + static_cast<size_t>(slot) * dim;
        const float scale = scales[slot];
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
        const float scale = scales[slot] * s.score_scale;
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
        const float scale = scales[slot];
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
  if (index->q8_scales != nullptr) {
    core::runtime_free_array(index->q8_scales, index->capacity);
    index->q8_scales = nullptr;
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
  const uint32_t graph_base_neighbor_capacity =
      graph_neighbor_capacity != 0
          ? graph_base_neighbors_from_graph_neighbors(graph_neighbor_capacity)
          : 0;
  const uint32_t graph_level_capacity =
      graph_level_capacity_for(desc->capacity, graph_neighbor_capacity);
  const uint32_t requested_graph_search = graph_search_from_desc(desc);
  const uint32_t graph_search_capacity =
      requested_graph_search > desc->capacity ? desc->capacity : requested_graph_search;
  uint32_t requested_query_search = kGraphDefaultEfConstruction > graph_search_capacity
                                        ? kGraphDefaultEfConstruction
                                        : graph_search_capacity;
  if (graph_search_capacity <= kU32Max / kGraphQueryReserveMultiplier) {
    const uint32_t reserve_query_search = graph_search_capacity * kGraphQueryReserveMultiplier;
    if (reserve_query_search > requested_query_search) {
      requested_query_search = reserve_query_search;
    }
  }
  const uint32_t graph_query_search_capacity =
      requested_query_search > desc->capacity ? desc->capacity : requested_query_search;
  const uint32_t graph_candidate_capacity = graph_query_search_capacity > graph_search_capacity
                                                ? graph_query_search_capacity
                                                : graph_search_capacity;
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
  index->dim = desc->dim;
  index->capacity = desc->capacity;
  index->count = 0;
  index->metric = desc->metric;
  index->index_kind = desc->index_kind;
  index->storage_kind = desc->storage_kind;
  index->slots = core::runtime_alloc_array<MemorySlot>(desc->capacity);
  index->vectors = desc->storage_kind == ASTRAL_MEMORY_STORAGE_F32
                       ? alloc_vector_storage(desc->capacity, desc->dim)
                       : nullptr;
  index->q8_vectors = desc->storage_kind == ASTRAL_MEMORY_STORAGE_Q8
                          ? alloc_q8_vector_storage(desc->capacity, desc->dim)
                          : nullptr;
  index->q8_scales = desc->storage_kind == ASTRAL_MEMORY_STORAGE_Q8
                         ? core::runtime_alloc_array<float>(desc->capacity)
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
  index->graph_build_score_evals = 0;
  index->graph_build_candidate_visits = 0;
  index->dense_active = 1u;
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
        core::runtime_alloc_array<uint32_t>(desc->capacity * graph_level_capacity);
    index->graph_levels = core::runtime_alloc_array<uint8_t>(desc->capacity);
    index->graph_candidates = core::runtime_alloc_array<uint32_t>(graph_candidate_capacity);
    index->graph_candidate_scores = core::runtime_alloc_array<float>(graph_candidate_capacity);
    index->graph_scratch_slots = core::runtime_alloc_array<uint32_t>(graph_scratch_capacity);
    index->graph_scratch_scores = core::runtime_alloc_array<float>(graph_scratch_capacity);
    index->graph_visited = core::runtime_alloc_array<uint32_t>(desc->capacity);
  }
  if (index->slots == nullptr || index->active_slots == nullptr || index->key_table == nullptr ||
      (desc->storage_kind == ASTRAL_MEMORY_STORAGE_F32 && index->vectors == nullptr) ||
      (desc->storage_kind == ASTRAL_MEMORY_STORAGE_Q8 &&
       (index->q8_vectors == nullptr || index->q8_scales == nullptr)) ||
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
  if (desc->storage_kind == ASTRAL_MEMORY_STORAGE_Q8) {
    std::memset(index->q8_vectors, 0,
                static_cast<size_t>(desc->capacity) * desc->dim * sizeof(int8_t));
    std::memset(index->q8_scales, 0, sizeof(float) * desc->capacity);
  }
  if (graph_neighbor_capacity != 0) {
    std::memset(index->graph_neighbors, 0, sizeof(uint32_t) * graph_neighbor_storage_count(index));
    std::memset(index->graph_neighbor_counts, 0,
                sizeof(uint32_t) * desc->capacity * graph_level_capacity);
    std::memset(index->graph_levels, 0, sizeof(uint8_t) * desc->capacity);
    std::memset(index->graph_candidates, 0, sizeof(uint32_t) * graph_candidate_capacity);
    std::memset(index->graph_candidate_scores, 0, sizeof(float) * graph_candidate_capacity);
    std::memset(index->graph_scratch_slots, 0, sizeof(uint32_t) * graph_scratch_capacity);
    std::memset(index->graph_scratch_scores, 0, sizeof(float) * graph_scratch_capacity);
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

AstralErr memory_stats(MemoryIndex* index, AstralMemoryStats* out_stats) {
  if (index == nullptr || out_stats == nullptr || out_stats->size != sizeof(AstralMemoryStats)) {
    return ASTRAL_E_INVALID;
  }

  const uint64_t vector_bytes =
      q8_storage(index) ? static_cast<uint64_t>(index->capacity) * index->dim * sizeof(int8_t) +
                              static_cast<uint64_t>(index->capacity) * sizeof(float)
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
        static_cast<uint64_t>(index->capacity) * index->graph_level_capacity * sizeof(uint32_t);
    graph_bytes += static_cast<uint64_t>(index->capacity) * sizeof(uint8_t);
    graph_bytes += static_cast<uint64_t>(index->graph_candidate_capacity) * sizeof(uint32_t);
    graph_bytes += static_cast<uint64_t>(index->graph_candidate_capacity) * sizeof(float);
    graph_bytes += static_cast<uint64_t>(index->graph_scratch_capacity) * sizeof(uint32_t);
    graph_bytes += static_cast<uint64_t>(index->graph_scratch_capacity) * sizeof(float);
    graph_bytes += static_cast<uint64_t>(index->capacity) * sizeof(uint32_t);
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
                sizeof(uint32_t) * index->capacity * index->graph_level_capacity);
    std::memset(index->graph_levels, 0, sizeof(uint8_t) * index->capacity);
    std::memset(index->graph_visited, 0, sizeof(uint32_t) * index->capacity);
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
      if (graph_enabled(index)) {
        index->graph_levels[slot] =
            static_cast<uint8_t>(graph_level_for_key(index, records[i].key));
      }
    }
    index->slots[slot].record = records[i];
    const float* src = vectors + static_cast<size_t>(i) * index->dim;
    if (q8_storage(index)) {
      quantize_q8_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src, index->dim);
    } else {
      float* dst = vector_at(index, slot);
      std::memcpy(dst, src, sizeof(float) * index->dim);
    }
    index->slots[slot].score_scale =
        index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(src, index->dim) : 0.0f;
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
  const uint32_t result_capacity = query_count * desc->top_k;
  if (max_results < result_capacity) {
    return ASTRAL_E_NOMEM;
  }

  if (index->index_kind == ASTRAL_MEMORY_INDEX_FLAT) {
    for (uint32_t query_base = 0; query_base < query_count;
         query_base += kMemoryBatchStackQueries) {
      const uint32_t remaining = query_count - query_base;
      const uint32_t chunk =
          remaining < kMemoryBatchStackQueries ? remaining : kMemoryBatchStackQueries;
      memory_search_flat_batch(index, desc, queries + static_cast<size_t>(query_base) * index->dim,
                               chunk, out_results + static_cast<size_t>(query_base) * desc->top_k,
                               out_counts + query_base);
    }
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
    if (q8_storage(index)) {
      const float scale = index->q8_scales[slot];
      std::memcpy(cursor, &scale, sizeof(float));
      cursor += sizeof(float);
      std::memcpy(cursor, q8_vector_at(index, slot),
                  static_cast<size_t>(index->dim) * sizeof(int8_t));
      cursor += static_cast<size_t>(index->dim) * sizeof(int8_t);
    } else {
      std::memcpy(cursor, vector_at(index, slot), sizeof(float) * index->dim);
      cursor += sizeof(float) * index->dim;
    }
  }

  if (graph_enabled(index)) {
    SaveGraphHeader graph_header{};
    graph_header.flags = kSaveGraphTopologyFlag;
    graph_header.neighbor_capacity = index->graph_neighbor_capacity;
    graph_header.base_neighbor_capacity = index->graph_base_neighbor_capacity;
    graph_header.search_capacity = index->graph_search_capacity;
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
        const uint32_t count = graph_neighbor_count_at_level(index, slot, level);
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
      header.version == kSaveVersionGraphTopology || header.version == kSaveVersion;
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

  const uint64_t saved_vector_bytes =
      saved_storage == ASTRAL_MEMORY_STORAGE_Q8
          ? static_cast<uint64_t>(header.count) *
                (sizeof(float) + static_cast<uint64_t>(header.dim) * sizeof(int8_t))
          : static_cast<uint64_t>(header.count) * header.dim * sizeof(float);
  const uint64_t need = sizeof(SaveHeader) +
                        static_cast<uint64_t>(header.count) * sizeof(AstralMemoryRecord) +
                        saved_vector_bytes;
  if (bytes.len < need) {
    return ASTRAL_E_INVALID;
  }

  MemoryIndex* index = nullptr;
  AstralErr err = memory_create(desc, &index);
  if (err != ASTRAL_OK) {
    return err;
  }

  const uint8_t* cursor = bytes.data + sizeof(SaveHeader);
  float vector[kMaxDim];
  for (uint32_t i = 0; i < header.count; ++i) {
    AstralMemoryRecord record{};
    std::memcpy(&record, cursor, sizeof(record));
    cursor += sizeof(record);
    if (record.size != sizeof(AstralMemoryRecord) || record.key == 0) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    const bool saved_q8 = saved_storage == ASTRAL_MEMORY_STORAGE_Q8;
    const bool dst_q8 = q8_storage(index);
    float saved_q8_scale = 1.0f;
    const int8_t* saved_q8_vector = nullptr;
    if (saved_q8) {
      float scale = 1.0f;
      std::memcpy(&scale, cursor, sizeof(float));
      cursor += sizeof(float);
      const int8_t* q8 = reinterpret_cast<const int8_t*>(cursor);
      if (dst_q8) {
        saved_q8_scale = scale;
        saved_q8_vector = q8;
      } else {
        for (uint32_t dim_i = 0; dim_i < header.dim; ++dim_i) {
          vector[dim_i] = static_cast<float>(q8[dim_i]) * scale;
        }
      }
      cursor += static_cast<size_t>(header.dim) * sizeof(int8_t);
    } else {
      std::memcpy(vector, cursor, sizeof(float) * header.dim);
      cursor += sizeof(float) * header.dim;
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
    if (dst_q8) {
      if (saved_q8) {
        index->q8_scales[slot] = saved_q8_scale;
        std::memcpy(q8_vector_at(index, slot), saved_q8_vector,
                    static_cast<size_t>(index->dim) * sizeof(int8_t));
        index->slots[slot].score_scale =
            index->metric == ASTRAL_MEMORY_METRIC_COSINE
                ? cosine_scale_q8(q8_vector_at(index, slot), saved_q8_scale, index->dim)
                : 0.0f;
      } else {
        index->slots[slot].score_scale =
            index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(vector, index->dim) : 0.0f;
        quantize_q8_vector(q8_vector_at(index, slot), &index->q8_scales[slot], vector, index->dim);
      }
    } else {
      index->slots[slot].score_scale =
          index->metric == ASTRAL_MEMORY_METRIC_COSINE ? cosine_scale(vector, index->dim) : 0.0f;
      std::memcpy(vector_at(index, slot), vector, sizeof(float) * index->dim);
    }
    if (graph_enabled(index)) {
      index->graph_levels[slot] = static_cast<uint8_t>(graph_level_for_key(index, record.key));
    }
  }
  index->count = header.count;
  index->free_slot_hint = header.count < index->capacity ? header.count : 0;
  index->dense_active = 1u;

  bool graph_loaded = false;
  if (graph_enabled(index) && header.version >= kSaveVersionGraphTopology &&
      header._reserved1 == kSaveGraphTopologyFlag) {
    const uint32_t saved_graph_header_bytes =
        header.version >= kSaveVersion ? sizeof(SaveGraphHeader) : sizeof(SaveGraphHeaderV3);
    if (bytes.len < static_cast<uint64_t>(cursor - bytes.data) + saved_graph_header_bytes) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    SaveGraphHeader graph_header{};
    if (header.version >= kSaveVersion) {
      std::memcpy(&graph_header, cursor, sizeof(graph_header));
      cursor += sizeof(graph_header);
    } else {
      SaveGraphHeaderV3 graph_header_v3{};
      std::memcpy(&graph_header_v3, cursor, sizeof(graph_header_v3));
      cursor += sizeof(graph_header_v3);
      graph_header.flags = graph_header_v3.flags;
      graph_header.neighbor_capacity = graph_header_v3.neighbor_capacity;
      graph_header.base_neighbor_capacity = graph_header_v3.neighbor_capacity;
      graph_header.search_capacity = graph_header_v3.search_capacity;
      graph_header.level_capacity = graph_header_v3.level_capacity;
      graph_header.max_level = graph_header_v3.max_level;
      graph_header.entry_active_pos = graph_header_v3.entry_active_pos;
    }
    if (graph_header.neighbor_capacity == 0 ||
        graph_header.neighbor_capacity > kGraphMaxNeighbors ||
        graph_header.base_neighbor_capacity == 0 ||
        graph_header.base_neighbor_capacity > kGraphMaxNeighbors ||
        graph_header.base_neighbor_capacity < graph_header.neighbor_capacity ||
        graph_header.level_capacity == 0 || graph_header.level_capacity > kGraphMaxLevels ||
        graph_header.search_capacity > desc->capacity) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    const uint64_t level_count = static_cast<uint64_t>(header.count) * graph_header.level_capacity;
    const uint64_t neighbor_slots =
        static_cast<uint64_t>(header.count) * graph_header.base_neighbor_capacity +
        static_cast<uint64_t>(header.count) * (graph_header.level_capacity - 1u) *
            graph_header.neighbor_capacity;
    const uint64_t graph_bytes = saved_graph_header_bytes +
                                 static_cast<uint64_t>(header.count) * sizeof(uint8_t) +
                                 level_count * sizeof(uint32_t) + neighbor_slots * sizeof(uint32_t);
    const uint64_t graph_begin = need;
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
          std::memcpy(&count, cursor, sizeof(count));
          cursor += sizeof(count);
          if (count > graph_neighbor_capacity_at_level(index, level)) {
            memory_destroy(index);
            return ASTRAL_E_INVALID;
          }
          graph_neighbor_count_ref(index, active_pos, level) = count;
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
