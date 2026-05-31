#include "memory_index.hpp"

#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"
#include "../core/runtime_state.hpp"
#include "../core/work_queue.hpp"
#include "../platform/atomics.h"
#include "../platform/file_map.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>

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
constexpr uint32_t kSaveVersionLegacyLayout = 4;
constexpr uint32_t kSaveVersionModernLayout = 5;
constexpr uint32_t kSaveVersionNormalizedF32Cosine = 6;
constexpr uint32_t kSaveVersion = kSaveVersionNormalizedF32Cosine;
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
constexpr uint32_t kGraphDefaultEfConstruction = 68;
constexpr uint32_t kGraphMinSearch = 4;
constexpr uint32_t kGraphCandidateReserveMultiplier = 8;
constexpr uint32_t kGraphLongLinkCount = 0;
constexpr uint32_t kGraphNeighborPrefetchDistance = 2;
constexpr uint64_t kBytesPerKiB = 1024;
constexpr uint64_t kBytesPerMiB = kBytesPerKiB * kBytesPerKiB;
constexpr uint64_t kGraphCompactExactSearchMaxBytes = 16 * kBytesPerMiB;
constexpr uint32_t kFlatQ8PrefetchDistance = 4;
constexpr uint32_t kFlatQ8PrefetchMinCount = 32768;
constexpr uint32_t kGraphMaxLevels = 16;
constexpr uint32_t kMemoryBatchStackQueries = 16;
constexpr uint32_t kMemoryAddStackSlots = 256;
constexpr uint32_t kMemoryAddParallelMinCount = 1024;
constexpr uint32_t kMemoryAddParallelMaxWorkers = 8;
constexpr uint32_t kMemorySearchBatchParallelMinQueries = 8;
constexpr uint32_t kMemorySearchBatchParallelMaxRecords = 32768;
constexpr uint32_t kMemorySearchBatchParallelMaxWorkers = 8;
constexpr uint32_t kMemorySearchBatchParallelMaxTopK = 8;
constexpr int32_t kQ8MinValue = -127;
constexpr int32_t kQ8MaxValue = 127;
constexpr float kQ8MaxFloat = 127.0f;
constexpr int32_t kE2M3Scale = 16;
constexpr int32_t kE2M3MagnitudeMask = 0x1F;
constexpr uint32_t kE2M3CodeCount = 32;
constexpr float kE2M3MaxFloat = 7.5f;
constexpr float kE2M3InvScale = 1.0f / static_cast<float>(kE2M3Scale);
constexpr float kE2M3LinearMinThreshold = 1.0f;
constexpr float kE2M3LinearMaxThreshold = 31.0f;
constexpr float kE2M3MidMinThreshold = 34.0f;
constexpr float kE2M3MidMaxThreshold = 62.0f;
constexpr float kE2M3HighMinThreshold = 68.0f;
constexpr float kE2M3HighMaxThreshold = 116.0f;
constexpr uint32_t kE2M3LinearStep = 2;
constexpr uint32_t kE2M3MidStep = 4;
constexpr uint32_t kE2M3HighStep = 8;
constexpr uint32_t kE2M3LinearBase = 0;
constexpr uint32_t kE2M3MidBase = 32;
constexpr uint32_t kE2M3HighBase = 64;
constexpr uint32_t kE2M3MaxScaled = 120;
constexpr float kE2M3LinearStepInv = 1.0f / static_cast<float>(kE2M3LinearStep);
constexpr float kE2M3MidStepInv = 1.0f / static_cast<float>(kE2M3MidStep);
constexpr float kE2M3HighStepInv = 1.0f / static_cast<float>(kE2M3HighStep);
constexpr uint32_t kF32SignShift = 31;
constexpr uint32_t kF32ExponentShift = 23;
constexpr uint32_t kF32ExponentMask = 0xFFu;
constexpr uint32_t kF32MantissaMask = 0x7FFFFFu;
constexpr int32_t kF32ExponentBias = 127;
constexpr uint32_t kF32InfBits = 0x7F800000u;
constexpr uint32_t kF32AbsMask = 0x7FFFFFFFu;
constexpr uint32_t kE5M2SignShift = 7;
constexpr uint32_t kE5M2ExponentShift = 2;
constexpr uint32_t kE5M2ExponentMask = 0x1Fu;
constexpr uint32_t kE5M2MantissaMask = 0x03u;
constexpr int32_t kE5M2ExponentBias = 15;
constexpr uint32_t kE5M2MantissaRoundBit = 1u << 20u;
constexpr uint32_t kE5M2MantissaShift = 21;
constexpr uint32_t kE5M2MantissaOverflow = 4;
constexpr uint32_t kE5M2MaxFinite = 0x7Bu;
constexpr float kE5M2MaxFloat = 57344.0f;
constexpr float kE5M2SubnormalUnit = 1.0f / 65536.0f;
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
constexpr uint32_t kNeonI8Lanes = 16;
constexpr uint32_t kNeonI8HalfLanes = 8;
constexpr size_t kVectorStorageAlign = 16;
constexpr uint32_t kNeonUnrollVectors = 4;
constexpr uint32_t kNeonUnrollF32 = kNeonF32Lanes * kNeonUnrollVectors;
constexpr uint32_t kNeonOffset1 = kNeonF32Lanes;
constexpr uint32_t kNeonOffset2 = kNeonF32Lanes * 2u;
constexpr uint32_t kNeonOffset3 = kNeonF32Lanes * 3u;
constexpr uint32_t kNeonI8Offset1 = kNeonI8HalfLanes;
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

struct SaveLayout {
  uint64_t record_offset;
  uint64_t record_stride;
  uint64_t scale_offset;
  uint64_t scale_stride;
  uint64_t vector_offset;
  uint64_t vector_stride;
  uint64_t graph_offset;
  uint64_t graph_bytes;
  uint64_t total_bytes;
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
  return kind == ASTRAL_MEMORY_STORAGE_F32 || kind == ASTRAL_MEMORY_STORAGE_Q8 ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 || kind == ASTRAL_MEMORY_STORAGE_F8_E5M2;
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

inline uint32_t graph_query_search_from_desc(const AstralMemoryIndexDesc* desc,
                                             uint32_t graph_search) {
  if (desc->index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    return 0;
  }
  uint32_t requested = desc->graph_query_search != 0 ? desc->graph_query_search : graph_search;
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
#if defined(__aarch64__) && defined(__ARM_NEON)
inline float32x4_t neon_i8_low_to_f32(int8x8_t bytes) {
  return vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(bytes))));
}

inline float32x4_t neon_i8_high_to_f32(int8x8_t bytes) {
  return vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(bytes))));
}

inline int32x4_t dot_i8x8_neon(const int8_t* a, const int8_t* b) {
  return vpaddlq_s16(vmull_s8(vld1_s8(a), vld1_s8(b)));
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
#elif defined(__aarch64__) && defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + kNeonI8Lanes <= dim; i += kNeonI8Lanes) {
    const int8x8_t a0 = vld1_s8(a + i);
    const int8x8_t a1 = vld1_s8(a + i + kNeonI8Offset1);
    acc0 = vfmaq_f32(acc0, neon_i8_low_to_f32(a0), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, neon_i8_high_to_f32(a0), vld1q_f32(b + i + kNeonF32Lanes));
    acc2 = vfmaq_f32(acc2, neon_i8_low_to_f32(a1), vld1q_f32(b + i + kNeonI8Offset1));
    acc3 =
        vfmaq_f32(acc3, neon_i8_high_to_f32(a1), vld1q_f32(b + i + kNeonI8Offset1 + kNeonF32Lanes));
  }
  float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  for (; i + kNeonF32Lanes <= dim; i += kNeonF32Lanes) {
    const int8x8_t av = vld1_s8(a + i);
    acc = vfmaq_f32(acc, neon_i8_low_to_f32(av), vld1q_f32(b + i));
  }
  float sum = vaddvq_f32(acc);
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
#elif defined(__aarch64__) && defined(__ARM_NEON)
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  uint32_t i = 0;
  for (; i + kNeonI8Lanes <= dim; i += kNeonI8Lanes) {
    acc0 = vaddq_s32(acc0, dot_i8x8_neon(a + i, b + i));
    acc1 = vaddq_s32(acc1, dot_i8x8_neon(a + i + kNeonI8HalfLanes, b + i + kNeonI8HalfLanes));
  }
  int32_t sum = vaddvq_s32(vaddq_s32(acc0, acc1));
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

inline uint32_t f32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline const float* e5m2_to_f32_table() {
  static constexpr float kE5M2ToF32[256] = {0.0f,
                                            1.52587891e-05f,
                                            3.05175781e-05f,
                                            4.57763672e-05f,
                                            6.10351562e-05f,
                                            7.62939453e-05f,
                                            9.15527344e-05f,
                                            0.000106811523f,
                                            0.000122070312f,
                                            0.000152587891f,
                                            0.000183105469f,
                                            0.000213623047f,
                                            0.000244140625f,
                                            0.000305175781f,
                                            0.000366210938f,
                                            0.000427246094f,
                                            0.00048828125f,
                                            0.000610351562f,
                                            0.000732421875f,
                                            0.000854492188f,
                                            0.0009765625f,
                                            0.00122070312f,
                                            0.00146484375f,
                                            0.00170898438f,
                                            0.001953125f,
                                            0.00244140625f,
                                            0.0029296875f,
                                            0.00341796875f,
                                            0.00390625f,
                                            0.0048828125f,
                                            0.005859375f,
                                            0.0068359375f,
                                            0.0078125f,
                                            0.009765625f,
                                            0.01171875f,
                                            0.013671875f,
                                            0.015625f,
                                            0.01953125f,
                                            0.0234375f,
                                            0.02734375f,
                                            0.03125f,
                                            0.0390625f,
                                            0.046875f,
                                            0.0546875f,
                                            0.0625f,
                                            0.078125f,
                                            0.09375f,
                                            0.109375f,
                                            0.125f,
                                            0.15625f,
                                            0.1875f,
                                            0.21875f,
                                            0.25f,
                                            0.3125f,
                                            0.375f,
                                            0.4375f,
                                            0.5f,
                                            0.625f,
                                            0.75f,
                                            0.875f,
                                            1.0f,
                                            1.25f,
                                            1.5f,
                                            1.75f,
                                            2.0f,
                                            2.5f,
                                            3.0f,
                                            3.5f,
                                            4.0f,
                                            5.0f,
                                            6.0f,
                                            7.0f,
                                            8.0f,
                                            10.0f,
                                            12.0f,
                                            14.0f,
                                            16.0f,
                                            20.0f,
                                            24.0f,
                                            28.0f,
                                            32.0f,
                                            40.0f,
                                            48.0f,
                                            56.0f,
                                            64.0f,
                                            80.0f,
                                            96.0f,
                                            112.0f,
                                            128.0f,
                                            160.0f,
                                            192.0f,
                                            224.0f,
                                            256.0f,
                                            320.0f,
                                            384.0f,
                                            448.0f,
                                            512.0f,
                                            640.0f,
                                            768.0f,
                                            896.0f,
                                            1024.0f,
                                            1280.0f,
                                            1536.0f,
                                            1792.0f,
                                            2048.0f,
                                            2560.0f,
                                            3072.0f,
                                            3584.0f,
                                            4096.0f,
                                            5120.0f,
                                            6144.0f,
                                            7168.0f,
                                            8192.0f,
                                            10240.0f,
                                            12288.0f,
                                            14336.0f,
                                            16384.0f,
                                            20480.0f,
                                            24576.0f,
                                            28672.0f,
                                            32768.0f,
                                            40960.0f,
                                            49152.0f,
                                            57344.0f,
                                            57344.0f,
                                            57344.0f,
                                            57344.0f,
                                            57344.0f,
                                            -0.0f,
                                            -1.52587891e-05f,
                                            -3.05175781e-05f,
                                            -4.57763672e-05f,
                                            -6.10351562e-05f,
                                            -7.62939453e-05f,
                                            -9.15527344e-05f,
                                            -0.000106811523f,
                                            -0.000122070312f,
                                            -0.000152587891f,
                                            -0.000183105469f,
                                            -0.000213623047f,
                                            -0.000244140625f,
                                            -0.000305175781f,
                                            -0.000366210938f,
                                            -0.000427246094f,
                                            -0.00048828125f,
                                            -0.000610351562f,
                                            -0.000732421875f,
                                            -0.000854492188f,
                                            -0.0009765625f,
                                            -0.00122070312f,
                                            -0.00146484375f,
                                            -0.00170898438f,
                                            -0.001953125f,
                                            -0.00244140625f,
                                            -0.0029296875f,
                                            -0.00341796875f,
                                            -0.00390625f,
                                            -0.0048828125f,
                                            -0.005859375f,
                                            -0.0068359375f,
                                            -0.0078125f,
                                            -0.009765625f,
                                            -0.01171875f,
                                            -0.013671875f,
                                            -0.015625f,
                                            -0.01953125f,
                                            -0.0234375f,
                                            -0.02734375f,
                                            -0.03125f,
                                            -0.0390625f,
                                            -0.046875f,
                                            -0.0546875f,
                                            -0.0625f,
                                            -0.078125f,
                                            -0.09375f,
                                            -0.109375f,
                                            -0.125f,
                                            -0.15625f,
                                            -0.1875f,
                                            -0.21875f,
                                            -0.25f,
                                            -0.3125f,
                                            -0.375f,
                                            -0.4375f,
                                            -0.5f,
                                            -0.625f,
                                            -0.75f,
                                            -0.875f,
                                            -1.0f,
                                            -1.25f,
                                            -1.5f,
                                            -1.75f,
                                            -2.0f,
                                            -2.5f,
                                            -3.0f,
                                            -3.5f,
                                            -4.0f,
                                            -5.0f,
                                            -6.0f,
                                            -7.0f,
                                            -8.0f,
                                            -10.0f,
                                            -12.0f,
                                            -14.0f,
                                            -16.0f,
                                            -20.0f,
                                            -24.0f,
                                            -28.0f,
                                            -32.0f,
                                            -40.0f,
                                            -48.0f,
                                            -56.0f,
                                            -64.0f,
                                            -80.0f,
                                            -96.0f,
                                            -112.0f,
                                            -128.0f,
                                            -160.0f,
                                            -192.0f,
                                            -224.0f,
                                            -256.0f,
                                            -320.0f,
                                            -384.0f,
                                            -448.0f,
                                            -512.0f,
                                            -640.0f,
                                            -768.0f,
                                            -896.0f,
                                            -1024.0f,
                                            -1280.0f,
                                            -1536.0f,
                                            -1792.0f,
                                            -2048.0f,
                                            -2560.0f,
                                            -3072.0f,
                                            -3584.0f,
                                            -4096.0f,
                                            -5120.0f,
                                            -6144.0f,
                                            -7168.0f,
                                            -8192.0f,
                                            -10240.0f,
                                            -12288.0f,
                                            -14336.0f,
                                            -16384.0f,
                                            -20480.0f,
                                            -24576.0f,
                                            -28672.0f,
                                            -32768.0f,
                                            -40960.0f,
                                            -49152.0f,
                                            -57344.0f,
                                            -57344.0f,
                                            -57344.0f,
                                            -57344.0f,
                                            -57344.0f};
  return kE5M2ToF32;
}

inline float e5m2_to_f32(uint8_t raw) {
  return e5m2_to_f32_table()[raw];
}

uint8_t f32_to_e5m2(float value) {
  const uint32_t bits = f32_bits(value);
  const uint32_t sign = bits >> kF32SignShift;
  const uint32_t abs_bits = bits & kF32AbsMask;
  if (abs_bits == 0) {
    return static_cast<uint8_t>(sign << kE5M2SignShift);
  }
  if (abs_bits >= kF32InfBits) {
    return static_cast<uint8_t>((sign << kE5M2SignShift) | kE5M2MaxFinite);
  }

  const uint32_t exponent = (abs_bits >> kF32ExponentShift) & kF32ExponentMask;
  const uint32_t mantissa = abs_bits & kF32MantissaMask;
  int32_t e5_exponent = static_cast<int32_t>(exponent) - kF32ExponentBias + kE5M2ExponentBias;
  if (e5_exponent <= 0) {
    const float abs_value = std::fabs(value);
    const uint32_t subnormal = static_cast<uint32_t>(abs_value / kE5M2SubnormalUnit + 0.5f);
    const uint32_t clamped = subnormal < kE5M2MantissaOverflow ? subnormal : kE5M2MantissaMask;
    return static_cast<uint8_t>((sign << kE5M2SignShift) | clamped);
  }

  uint32_t e5_mantissa = (mantissa + kE5M2MantissaRoundBit) >> kE5M2MantissaShift;
  if (e5_mantissa == kE5M2MantissaOverflow) {
    e5_mantissa = 0;
    ++e5_exponent;
  }
  if (e5_exponent >= static_cast<int32_t>(kE5M2ExponentMask)) {
    return static_cast<uint8_t>((sign << kE5M2SignShift) | kE5M2MaxFinite);
  }
  return static_cast<uint8_t>((sign << kE5M2SignShift) |
                              (static_cast<uint32_t>(e5_exponent) << kE5M2ExponentShift) |
                              e5_mantissa);
}

float dot_e5m2_f32(const int8_t* a, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  const float* table = e5m2_to_f32_table();
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + kAvx2F32Lanes * 2u <= dim; i += kAvx2F32Lanes * 2u) {
    const __m128i bytes0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
    const __m128i bytes1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i + kAvx2F32Lanes));
    const __m256 av0 = _mm256_i32gather_ps(table, _mm256_cvtepu8_epi32(bytes0), sizeof(float));
    const __m256 av1 = _mm256_i32gather_ps(table, _mm256_cvtepu8_epi32(bytes1), sizeof(float));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(b + i + kAvx2F32Lanes), acc1);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av0, _mm256_loadu_ps(b + i)));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(av1, _mm256_loadu_ps(b + i + kAvx2F32Lanes)));
#endif
  }
  float sum = reduce_avx2_f32(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) {
    sum += e5m2_to_f32(static_cast<uint8_t>(a[i])) * b[i];
  }
  return sum;
#else
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    sum0 += e5m2_to_f32(static_cast<uint8_t>(a[i])) * b[i];
    sum1 += e5m2_to_f32(static_cast<uint8_t>(a[i + 1u])) * b[i + 1u];
    sum2 += e5m2_to_f32(static_cast<uint8_t>(a[i + 2u])) * b[i + 2u];
    sum3 += e5m2_to_f32(static_cast<uint8_t>(a[i + 3u])) * b[i + 3u];
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    sum += e5m2_to_f32(static_cast<uint8_t>(a[i])) * b[i];
  }
  return sum;
#endif
}

float dot_e5m2_e5m2(const int8_t* a, const int8_t* b, uint32_t dim) {
#if defined(__AVX2__)
  const float* table = e5m2_to_f32_table();
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + kAvx2F32Lanes * 2u <= dim; i += kAvx2F32Lanes * 2u) {
    const __m128i a_bytes0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
    const __m128i a_bytes1 =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i + kAvx2F32Lanes));
    const __m128i b_bytes0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
    const __m128i b_bytes1 =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i + kAvx2F32Lanes));
    const __m256 av0 = _mm256_i32gather_ps(table, _mm256_cvtepu8_epi32(a_bytes0), sizeof(float));
    const __m256 av1 = _mm256_i32gather_ps(table, _mm256_cvtepu8_epi32(a_bytes1), sizeof(float));
    const __m256 bv0 = _mm256_i32gather_ps(table, _mm256_cvtepu8_epi32(b_bytes0), sizeof(float));
    const __m256 bv1 = _mm256_i32gather_ps(table, _mm256_cvtepu8_epi32(b_bytes1), sizeof(float));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
    acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av0, bv0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(av1, bv1));
#endif
  }
  float sum = reduce_avx2_f32(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) {
    sum += e5m2_to_f32(static_cast<uint8_t>(a[i])) * e5m2_to_f32(static_cast<uint8_t>(b[i]));
  }
  return sum;
#else
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    sum0 += e5m2_to_f32(static_cast<uint8_t>(a[i])) * e5m2_to_f32(static_cast<uint8_t>(b[i]));
    sum1 +=
        e5m2_to_f32(static_cast<uint8_t>(a[i + 1u])) * e5m2_to_f32(static_cast<uint8_t>(b[i + 1u]));
    sum2 +=
        e5m2_to_f32(static_cast<uint8_t>(a[i + 2u])) * e5m2_to_f32(static_cast<uint8_t>(b[i + 2u]));
    sum3 +=
        e5m2_to_f32(static_cast<uint8_t>(a[i + 3u])) * e5m2_to_f32(static_cast<uint8_t>(b[i + 3u]));
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    sum += e5m2_to_f32(static_cast<uint8_t>(a[i])) * e5m2_to_f32(static_cast<uint8_t>(b[i]));
  }
  return sum;
#endif
}

float l2_score_e5m2_f32(const int8_t* a, float scale, const float* b, uint32_t dim) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    const float d0 = e5m2_to_f32(static_cast<uint8_t>(a[i])) * scale - b[i];
    const float d1 = e5m2_to_f32(static_cast<uint8_t>(a[i + 1u])) * scale - b[i + 1u];
    const float d2 = e5m2_to_f32(static_cast<uint8_t>(a[i + 2u])) * scale - b[i + 2u];
    const float d3 = e5m2_to_f32(static_cast<uint8_t>(a[i + 3u])) * scale - b[i + 3u];
    sum0 += d0 * d0;
    sum1 += d1 * d1;
    sum2 += d2 * d2;
    sum3 += d3 * d3;
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    const float d = e5m2_to_f32(static_cast<uint8_t>(a[i])) * scale - b[i];
    sum += d * d;
  }
  return -sum;
}

float l2_score_e5m2_e5m2(const int8_t* a, float a_scale, const int8_t* b, float b_scale,
                         uint32_t dim) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    const float d0 = e5m2_to_f32(static_cast<uint8_t>(a[i])) * a_scale -
                     e5m2_to_f32(static_cast<uint8_t>(b[i])) * b_scale;
    const float d1 = e5m2_to_f32(static_cast<uint8_t>(a[i + 1u])) * a_scale -
                     e5m2_to_f32(static_cast<uint8_t>(b[i + 1u])) * b_scale;
    const float d2 = e5m2_to_f32(static_cast<uint8_t>(a[i + 2u])) * a_scale -
                     e5m2_to_f32(static_cast<uint8_t>(b[i + 2u])) * b_scale;
    const float d3 = e5m2_to_f32(static_cast<uint8_t>(a[i + 3u])) * a_scale -
                     e5m2_to_f32(static_cast<uint8_t>(b[i + 3u])) * b_scale;
    sum0 += d0 * d0;
    sum1 += d1 * d1;
    sum2 += d2 * d2;
    sum3 += d3 * d3;
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    const float d = e5m2_to_f32(static_cast<uint8_t>(a[i])) * a_scale -
                    e5m2_to_f32(static_cast<uint8_t>(b[i])) * b_scale;
    sum += d * d;
  }
  return -sum;
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
#elif defined(__aarch64__) && defined(__ARM_NEON)
  const float32x4_t scale_v = vdupq_n_f32(scale);
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + kNeonI8Lanes <= dim; i += kNeonI8Lanes) {
    const int8x8_t a0 = vld1_s8(a + i);
    const int8x8_t a1 = vld1_s8(a + i + kNeonI8Offset1);
    const float32x4_t d0 = vsubq_f32(vmulq_f32(neon_i8_low_to_f32(a0), scale_v), vld1q_f32(b + i));
    const float32x4_t d1 =
        vsubq_f32(vmulq_f32(neon_i8_high_to_f32(a0), scale_v), vld1q_f32(b + i + kNeonF32Lanes));
    const float32x4_t d2 =
        vsubq_f32(vmulq_f32(neon_i8_low_to_f32(a1), scale_v), vld1q_f32(b + i + kNeonI8Offset1));
    const float32x4_t d3 = vsubq_f32(vmulq_f32(neon_i8_high_to_f32(a1), scale_v),
                                     vld1q_f32(b + i + kNeonI8Offset1 + kNeonF32Lanes));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
    acc2 = vfmaq_f32(acc2, d2, d2);
    acc3 = vfmaq_f32(acc3, d3, d3);
  }
  float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  for (; i + kNeonF32Lanes <= dim; i += kNeonF32Lanes) {
    const int8x8_t av = vld1_s8(a + i);
    const float32x4_t d = vsubq_f32(vmulq_f32(neon_i8_low_to_f32(av), scale_v), vld1q_f32(b + i));
    acc = vfmaq_f32(acc, d, d);
  }
  float sum = vaddvq_f32(acc);
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
#elif defined(__aarch64__) && defined(__ARM_NEON)
  int32x4_t acc_a = vdupq_n_s32(0);
  int32x4_t acc_b = vdupq_n_s32(0);
  int32x4_t acc_ab = vdupq_n_s32(0);
  uint32_t i = 0;
  for (; i + kNeonI8HalfLanes <= dim; i += kNeonI8HalfLanes) {
    acc_a = vaddq_s32(acc_a, dot_i8x8_neon(a + i, a + i));
    acc_b = vaddq_s32(acc_b, dot_i8x8_neon(b + i, b + i));
    acc_ab = vaddq_s32(acc_ab, dot_i8x8_neon(a + i, b + i));
  }
  int32_t sum_a = vaddvq_s32(acc_a);
  int32_t sum_b = vaddvq_s32(acc_b);
  int32_t sum_ab = vaddvq_s32(acc_ab);
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

inline int8_t e2m3_magnitude_scaled(uint32_t magnitude) {
  static constexpr int8_t kScaledMagnitudes[kE2M3CodeCount] = {
      0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,
      32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120};
  return kScaledMagnitudes[magnitude & kE2M3MagnitudeMask];
}

inline float e2m3_scaled_to_f32(int8_t scaled, float scale) {
  return static_cast<float>(scaled) * scale * kE2M3InvScale;
}

inline uint32_t ceil_positive_to_u32(float value) {
  const uint32_t truncated = static_cast<uint32_t>(value);
  return truncated + (value > static_cast<float>(truncated) ? 1u : 0u);
}

int8_t round_scaled_e2m3(float value) {
  const float abs_value = std::fabs(value);
  uint32_t best = kE2M3MaxScaled;
  if (abs_value <= kE2M3LinearMinThreshold) {
    best = 0;
  } else if (abs_value <= kE2M3LinearMaxThreshold) {
    best = kE2M3LinearBase +
           kE2M3LinearStep *
               ceil_positive_to_u32((abs_value - kE2M3LinearMinThreshold) * kE2M3LinearStepInv);
  } else if (abs_value <= kE2M3MidMinThreshold) {
    best = kE2M3MidBase;
  } else if (abs_value <= kE2M3MidMaxThreshold) {
    best = kE2M3MidBase + kE2M3MidStep * ceil_positive_to_u32((abs_value - kE2M3MidMinThreshold) *
                                                              kE2M3MidStepInv);
  } else if (abs_value <= kE2M3HighMinThreshold) {
    best = kE2M3HighBase;
  } else if (abs_value <= kE2M3HighMaxThreshold) {
    best = kE2M3HighBase +
           kE2M3HighStep *
               ceil_positive_to_u32((abs_value - kE2M3HighMinThreshold) * kE2M3HighStepInv);
  }
  return value < 0.0f ? static_cast<int8_t>(-best) : static_cast<int8_t>(best);
}

void quantize_e2m3_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim) {
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float abs_value = std::fabs(src[i]);
    if (abs_value > max_abs) {
      max_abs = abs_value;
    }
  }
  const float scale = max_abs > 0.0f ? max_abs / kE2M3MaxFloat : 1.0f;
  const float inv_scale = static_cast<float>(kE2M3Scale) / scale;
  for (uint32_t i = 0; i < dim; ++i) {
    dst[i] = round_scaled_e2m3(src[i] * inv_scale);
  }
  *out_scale = scale;
}

void quantize_e2m3_cosine_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim) {
  const float sq = dot_f32(src, src, dim);
  const float norm = sq > 0.0f ? std::sqrt(sq) : 0.0f;
  if (norm == 0.0f) {
    quantize_e2m3_vector(dst, out_scale, src, dim);
    return;
  }
  float normalized[kMaxDim];
  const float inv_norm = 1.0f / norm;
  for (uint32_t i = 0; i < dim; ++i) {
    normalized[i] = src[i] * inv_norm;
  }
  quantize_e2m3_vector(dst, out_scale, normalized, dim);
}

void quantize_e5m2_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim) {
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float abs_value = std::fabs(src[i]);
    if (abs_value > max_abs) {
      max_abs = abs_value;
    }
  }
  const float scale = max_abs > 0.0f ? max_abs / kE5M2MaxFloat : 1.0f;
  const float inv_scale = 1.0f / scale;
  for (uint32_t i = 0; i < dim; ++i) {
    dst[i] = static_cast<int8_t>(f32_to_e5m2(src[i] * inv_scale));
  }
  *out_scale = scale;
}

void quantize_e5m2_cosine_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim) {
  const float sq = dot_f32(src, src, dim);
  const float norm = sq > 0.0f ? std::sqrt(sq) : 0.0f;
  if (norm == 0.0f) {
    quantize_e5m2_vector(dst, out_scale, src, dim);
    return;
  }
  float normalized[kMaxDim];
  const float inv_norm = 1.0f / norm;
  for (uint32_t i = 0; i < dim; ++i) {
    normalized[i] = src[i] * inv_norm;
  }
  quantize_e5m2_vector(dst, out_scale, normalized, dim);
}

inline float vector_norm(const float* v, uint32_t dim) {
  const float sq = dot_f32(v, v, dim);
  return sq > 0.0f ? std::sqrt(sq) : 0.0f;
}

inline float cosine_scale(const float* v, uint32_t dim) {
  const float norm = vector_norm(v, dim);
  return norm > 0.0f ? 1.0f / norm : 0.0f;
}

void normalize_f32_vector(float* dst, const float* src, uint32_t dim) {
  const float norm = vector_norm(src, dim);
  const float inv_norm = norm > 0.0f ? 1.0f / norm : 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    dst[i] = src[i] * inv_norm;
  }
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

struct MemorySnapshotView {
  AstralHandle handle;
  platform::ReadOnlyFileMap map;
  AstralMemorySnapshotInfo info;
};

namespace {

struct MemoryAddPreprocessJob {
  MemoryIndex* index;
  const float* vectors;
  const uint32_t* slots;
  uint32_t begin;
  uint32_t end;
  std::atomic<uint32_t>* remaining;
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

struct GraphSearchScratch {
  uint32_t* candidates;
  float* candidate_scores;
  uint32_t* top_slots;
  float* top_scores;
  uint32_t* visited;
  uint32_t visit_generation;
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
  GraphSearchScratch scratch;
};

inline uint64_t memory_save_byte_count(const MemoryIndex* index) {
  const uint64_t vector_bytes =
      (index->storage_kind == ASTRAL_MEMORY_STORAGE_Q8 ||
       index->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 ||
       index->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2)
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

bool memory_save_layout(uint32_t version, uint32_t dim, uint32_t count,
                        AstralMemoryStorageKind storage, uint32_t index_kind,
                        uint32_t graph_base_neighbors, uint32_t graph_neighbors,
                        uint32_t graph_levels, SaveLayout* out_layout) {
  if (out_layout == nullptr || !storage_kind_valid(storage)) {
    return false;
  }
  SaveLayout layout{};
  const bool compact = storage == ASTRAL_MEMORY_STORAGE_Q8 ||
                       storage == ASTRAL_MEMORY_STORAGE_F6_E2M3 ||
                       storage == ASTRAL_MEMORY_STORAGE_F8_E5M2;
  layout.record_offset = sizeof(SaveHeader);
  layout.record_stride = sizeof(AstralMemoryRecord);
  if (version >= kSaveVersionModernLayout) {
    uint64_t cursor =
        layout.record_offset + static_cast<uint64_t>(count) * sizeof(AstralMemoryRecord);
    if (compact) {
      layout.scale_offset = cursor;
      layout.scale_stride = sizeof(float);
      cursor += static_cast<uint64_t>(count) * sizeof(float);
      layout.vector_offset = cursor;
      layout.vector_stride = static_cast<uint64_t>(dim) * sizeof(int8_t);
      cursor += static_cast<uint64_t>(count) * layout.vector_stride;
    } else {
      layout.vector_offset = cursor;
      layout.vector_stride = static_cast<uint64_t>(dim) * sizeof(float);
      cursor += static_cast<uint64_t>(count) * layout.vector_stride;
    }
    layout.graph_offset = cursor;
  } else {
    layout.scale_offset = compact ? layout.record_offset + sizeof(AstralMemoryRecord) : 0;
    layout.scale_stride =
        compact ? sizeof(AstralMemoryRecord) + sizeof(float) + static_cast<uint64_t>(dim) : 0;
    layout.vector_offset =
        layout.record_offset + sizeof(AstralMemoryRecord) + (compact ? sizeof(float) : 0);
    layout.vector_stride =
        sizeof(AstralMemoryRecord) + (compact ? sizeof(float) + static_cast<uint64_t>(dim)
                                              : static_cast<uint64_t>(dim) * sizeof(float));
    layout.graph_offset =
        layout.record_offset + static_cast<uint64_t>(count) * layout.vector_stride;
  }
  if (index_kind == ASTRAL_MEMORY_INDEX_GRAPH && graph_neighbors != 0 && graph_levels != 0) {
    const uint64_t graph_header_bytes =
        version >= kSaveVersionLegacyLayout ? sizeof(SaveGraphHeader) : sizeof(SaveGraphHeaderV3);
    const uint64_t neighbor_slots =
        static_cast<uint64_t>(count) * graph_base_neighbors +
        static_cast<uint64_t>(count) * (graph_levels - 1u) * graph_neighbors;
    layout.graph_bytes = graph_header_bytes + static_cast<uint64_t>(count) * sizeof(uint8_t) +
                         static_cast<uint64_t>(count) * graph_levels * sizeof(uint32_t) +
                         neighbor_slots * sizeof(uint32_t);
  }
  layout.total_bytes = layout.graph_offset + layout.graph_bytes;
  *out_layout = layout;
  return true;
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

inline bool q8_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_Q8;
}

inline bool e2m3_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3;
}

inline bool e5m2_storage(const MemoryIndex* index) {
  return index->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2;
}

inline bool compact_storage_kind(AstralMemoryStorageKind kind) {
  return kind == ASTRAL_MEMORY_STORAGE_Q8 || kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 ||
         kind == ASTRAL_MEMORY_STORAGE_F8_E5M2;
}

inline bool compact_storage(const MemoryIndex* index) {
  return compact_storage_kind(index->storage_kind);
}

inline float compact_value_scale(const MemoryIndex* index, float scale) {
  return e2m3_storage(index) ? scale * kE2M3InvScale : scale;
}

inline float compact_value_scale_kind(AstralMemoryStorageKind kind, float scale) {
  return kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 ? scale * kE2M3InvScale : scale;
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
  if (compact_storage(index)) {
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
  if (compact_storage(index)) {
    const int8_t* q8 = q8_vector_at(index, slot);
    const float scale = compact_value_scale(index, index->q8_scales[slot]);
    if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return (e5m2_storage(index) ? dot_e5m2_f32(q8, query, index->dim)
                                  : dot_q8_f32(q8, query, index->dim)) *
             scale;
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return (e5m2_storage(index) ? dot_e5m2_f32(q8, query, index->dim)
                                  : dot_q8_f32(q8, query, index->dim)) *
             scale * query_scale * index->slots[slot].score_scale;
    }
    return e5m2_storage(index) ? l2_score_e5m2_f32(q8, scale, query, index->dim)
                               : l2_score_q8_f32(q8, scale, query, index->dim);
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    return dot_f32(query, vector_at(index, slot), index->dim);
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return dot_f32(query, vector_at(index, slot), index->dim) * query_scale;
  }
  return l2_score_f32(query, vector_at(index, slot), index->dim);
}

inline float score_slot_compact_query(MemoryIndex* index, const int8_t* query, float query_scale,
                                      uint32_t slot, float cosine_query_scale) {
  const int8_t* q8 = q8_vector_at(index, slot);
  const float scale = compact_value_scale(index, index->q8_scales[slot]);
  if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    return (e5m2_storage(index) ? dot_e5m2_e5m2(q8, query, index->dim)
                                : dot_q8_q8(q8, query, index->dim)) *
           scale * query_scale;
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return (e5m2_storage(index) ? dot_e5m2_e5m2(q8, query, index->dim)
                                : dot_q8_q8(q8, query, index->dim)) *
           scale * query_scale * cosine_query_scale * index->slots[slot].score_scale;
  }
  return e5m2_storage(index) ? l2_score_e5m2_e5m2(q8, scale, query, query_scale, index->dim)
                             : l2_score_q8_q8(q8, scale, query, query_scale, index->dim);
}

inline float score_pair(MemoryIndex* index, uint32_t a, uint32_t b) {
  ++index->graph_build_score_evals;
  if (compact_storage(index)) {
    const int8_t* va = q8_vector_at(index, a);
    const int8_t* vb = q8_vector_at(index, b);
    const float scale_a = compact_value_scale(index, index->q8_scales[a]);
    const float scale_b = compact_value_scale(index, index->q8_scales[b]);
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return (e5m2_storage(index) ? dot_e5m2_e5m2(va, vb, index->dim)
                                  : dot_q8_q8(va, vb, index->dim)) *
             scale_a * scale_b * index->slots[a].score_scale * index->slots[b].score_scale;
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
      return e5m2_storage(index) ? l2_score_e5m2_e5m2(va, scale_a, vb, scale_b, index->dim)
                                 : l2_score_q8_q8(va, scale_a, vb, scale_b, index->dim);
    }
    return (e5m2_storage(index) ? dot_e5m2_e5m2(va, vb, index->dim)
                                : dot_q8_q8(va, vb, index->dim)) *
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

  const float candidate_score = score_pair(index, owner_slot, candidate_slot);
  uint32_t weakest_slot = neighbors[0];
  float weakest_score = score_pair(index, owner_slot, weakest_slot);
  for (uint32_t i = 0; i < count; ++i) {
    if (i == 0) {
      continue;
    }
    const float score = score_pair(index, owner_slot, neighbors[i]);
    if (graph_candidate_worse(index, score, neighbors[i], weakest_score, weakest_slot)) {
      weakest_score = score;
      weakest_slot = neighbors[i];
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
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == weakest_slot) {
      neighbors[i] = candidate_slot;
      return;
    }
  }
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
                            uint32_t candidate_count, uint32_t* neighbors, uint32_t* out_count,
                            uint32_t selection_capacity) {
  uint32_t selected = 0;
  const uint32_t capacity = selection_capacity;
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

void graph_search_layer_query(MemoryIndex* index, GraphSearchScratch* scratch, const float* query,
                              float query_scale, uint32_t entry, uint32_t level, uint32_t capacity,
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

    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, level);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, level);
    for (uint32_t i = 0; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (i + kGraphNeighborPrefetchDistance < neighbor_count) {
        prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      }
      if (graph_query_was_visited(scratch, neighbor) || index->slots[neighbor].occupied == 0 ||
          index->graph_levels[neighbor] < level) {
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

void graph_search_layer_compact_query(MemoryIndex* index, GraphSearchScratch* scratch,
                                      const int8_t* query, float query_scale,
                                      float cosine_query_scale, uint32_t entry, uint32_t level,
                                      uint32_t capacity, uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_query_mark_visited(scratch, entry);
  const float entry_score =
      score_slot_compact_query(index, query, query_scale, entry, cosine_query_scale);
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

    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, level);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, level);
    for (uint32_t i = 0; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (i + kGraphNeighborPrefetchDistance < neighbor_count) {
        prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      }
      if (graph_query_was_visited(scratch, neighbor) || index->slots[neighbor].occupied == 0 ||
          index->graph_levels[neighbor] < level) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score =
          score_slot_compact_query(index, query, query_scale, neighbor, cosine_query_scale);
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

uint32_t graph_greedy_closest_compact_query(MemoryIndex* index, const int8_t* query,
                                            float query_scale, float cosine_query_scale,
                                            uint32_t entry, uint32_t begin_level,
                                            uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score =
      score_slot_compact_query(index, query, query_scale, closest, cosine_query_scale);
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
            score_slot_compact_query(index, query, query_scale, candidate, cosine_query_scale);
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
    graph_select_neighbors(index, slot, level, candidate_count, neighbors, &filled, level_capacity);
    graph_neighbor_count_ref(index, slot, level) = filled;
    for (uint32_t i = 0; i < filled; ++i) {
      refine_graph_neighbor_list(index, neighbors[i], slot, level);
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
              sizeof(uint32_t) * index->capacity * index->graph_level_capacity);
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

inline void graph_query_mark_visited(GraphSearchScratch* scratch, uint32_t slot) {
  scratch->visited[slot] = scratch->visit_generation;
}

inline bool graph_query_was_visited(const GraphSearchScratch* scratch, uint32_t slot) {
  return scratch->visited[slot] == scratch->visit_generation;
}

void graph_query_begin_visit(const MemoryIndex* index, GraphSearchScratch* scratch) {
  ++scratch->visit_generation;
  if (scratch->visit_generation == 0) {
    std::memset(scratch->visited, 0, sizeof(uint32_t) * index->capacity);
    scratch->visit_generation = 1;
  }
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

  uint32_t worst_pos = 0;
  float worst_score = scratch->candidate_scores[0];
  for (uint32_t i = 1; i < count; ++i) {
    if (graph_candidate_worse(index, scratch->candidate_scores[i], scratch->candidates[i],
                              worst_score, scratch->candidates[worst_pos])) {
      worst_score = scratch->candidate_scores[i];
      worst_pos = i;
    }
  }
  if (graph_candidate_better(index, score, slot, worst_score, scratch->candidates[worst_pos])) {
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

  float normalized_query[kMaxDim];
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
  if (compact_storage(index)) {
    int8_t compact_query[kMaxDim];
    float compact_query_scale = 1.0f;
    float compact_cosine_query_scale = query_scale;
    quantize_compact_query(index, compact_query, &compact_query_scale, query);
    const uint32_t entry =
        index->graph_max_level != 0
            ? graph_greedy_closest_compact_query(index, compact_query, compact_query_scale,
                                                 compact_cosine_query_scale,
                                                 index->graph_entry_slot, index->graph_max_level, 0)
            : index->graph_entry_slot;
    graph_search_layer_compact_query(index, scratch, compact_query, compact_query_scale,
                                     compact_cosine_query_scale, entry, 0, search_capacity,
                                     &top_count);
  } else {
    const uint32_t entry =
        index->graph_max_level != 0
            ? graph_greedy_closest_query(index, query, query_scale, index->graph_entry_slot,
                                         index->graph_max_level, 0)
            : index->graph_entry_slot;
    graph_search_layer_query(index, scratch, query, query_scale, entry, 0, search_capacity,
                             &top_count);
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
  if (compact_storage(index)) {
    for (uint32_t i = 0; i < top_count; ++i) {
      const uint32_t slot = scratch->top_slots[i];
      const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
      for (uint32_t neighbor_i = 0; neighbor_i < neighbor_count; ++neighbor_i) {
        const uint32_t neighbor = neighbors[neighbor_i];
        if (graph_query_was_visited(scratch, neighbor) || index->slots[neighbor].occupied == 0) {
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
  GraphSearchScratch scratch{index->graph_candidates,    index->graph_candidate_scores,
                             index->graph_scratch_slots, index->graph_scratch_scores,
                             index->graph_visited,       index->graph_visit_generation};
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
  scratch->visited = core::runtime_alloc_array<uint32_t>(index->capacity);
  if (scratch->candidates == nullptr || scratch->candidate_scores == nullptr ||
      scratch->top_slots == nullptr || scratch->top_scores == nullptr ||
      scratch->visited == nullptr) {
    graph_search_scratch_free(index, scratch);
    return false;
  }
  std::memset(scratch->visited, 0, sizeof(uint32_t) * index->capacity);
  return true;
}

void memory_search_graph_batch_work(void* user) {
  MemoryGraphSearchBatchJob* job = static_cast<MemoryGraphSearchBatchJob*>(user);
  for (uint32_t i = job->begin; i < job->end; ++i) {
    uint32_t count = 0;
    const float* query = job->queries + static_cast<size_t>(i) * job->index->dim;
    AstralMemorySearchResult* results =
        job->out_results + static_cast<size_t>(i) * job->desc->top_k;
    memory_search_graph_with_scratch(job->index, job->desc, query, results, &count, &job->scratch);
    job->out_counts[i] = count;
  }
  job->remaining->fetch_sub(1, std::memory_order_release);
  astral::platform::cpu_signal_event();
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

  MemoryGraphSearchBatchJob jobs[kMemorySearchBatchParallelMaxWorkers]{};
  uint32_t allocated = 0;
  for (; allocated < worker_count; ++allocated) {
    if (!graph_search_scratch_alloc(index, &jobs[allocated].scratch)) {
      for (uint32_t i = 0; i < allocated; ++i) {
        graph_search_scratch_free(index, &jobs[i].scratch);
      }
      return false;
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
    graph_search_scratch_free(index, &jobs[worker_i].scratch);
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
  if (desc->group_id == ASTRAL_MEMORY_GROUP_ANY && index->dense_active != 0 &&
      !e5m2_storage(index)) {
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

    if (index->dense_active != 0 && !e5m2_storage(index)) {
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
  float normalized_query[kMaxDim];
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
  float normalized_query[kMaxDim];
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
  if (compact_storage(index)) {
    if (desc->top_k == kTopOne) {
      memory_search_q8_top1(index, desc, query, out_results, out_count);
    } else {
      memory_search_q8(index, desc, query, out_results, out_count);
    }
    return;
  }
  if (memory_search_flat_batch_record_parallel(index, desc, query, 1, out_results, out_count)) {
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
      index->dense_active != 0 && !e5m2_storage(index)) {
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
  job->remaining->fetch_sub(1, std::memory_order_release);
  astral::platform::cpu_signal_event();
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
  job->remaining->fetch_sub(1, std::memory_order_release);
  astral::platform::cpu_signal_event();
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

void memory_add_preprocess_range(MemoryIndex* index, const float* vectors, const uint32_t* slots,
                                 uint32_t begin, uint32_t end) {
  for (uint32_t i = begin; i < end; ++i) {
    const uint32_t slot = slots[i];
    const float* src = vectors + static_cast<size_t>(i) * index->dim;
    if (q8_storage(index)) {
      quantize_q8_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src, index->dim);
    } else if (e2m3_storage(index)) {
      if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        quantize_e2m3_cosine_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src,
                                    index->dim);
      } else {
        quantize_e2m3_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src, index->dim);
      }
    } else if (e5m2_storage(index)) {
      if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        quantize_e5m2_cosine_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src,
                                    index->dim);
      } else {
        quantize_e5m2_vector(q8_vector_at(index, slot), &index->q8_scales[slot], src, index->dim);
      }
    } else {
      store_f32_vector(index, slot, src);
    }
    if (compact_storage(index)) {
      index->slots[slot].score_scale =
          index->metric == ASTRAL_MEMORY_METRIC_COSINE
              ? (q8_storage(index) ? cosine_scale(src, index->dim) : 1.0f)
              : 0.0f;
    }
  }
}

void memory_add_preprocess_work(void* user) {
  MemoryAddPreprocessJob* job = static_cast<MemoryAddPreprocessJob*>(user);
  memory_add_preprocess_range(job->index, job->vectors, job->slots, job->begin, job->end);
  job->remaining->fetch_sub(1, std::memory_order_release);
  astral::platform::cpu_signal_event();
}

bool memory_add_preprocess_parallel(MemoryIndex* index, const float* vectors, const uint32_t* slots,
                                    uint32_t count) {
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
    jobs[worker_i].remaining = &remaining;

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
  return true;
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
  const uint32_t requested_query_search = graph_query_search_from_desc(desc, graph_search_capacity);
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
  index->q8_vectors = compact_storage_kind(desc->storage_kind)
                          ? alloc_q8_vector_storage(desc->capacity, desc->dim)
                          : nullptr;
  index->q8_scales = compact_storage_kind(desc->storage_kind)
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
      (compact_storage_kind(desc->storage_kind) &&
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
  if (compact_storage_kind(desc->storage_kind)) {
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
      compact_storage(index)
          ? static_cast<uint64_t>(index->capacity) * index->dim * sizeof(int8_t) +
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

  uint32_t stack_slots[kMemoryAddStackSlots];
  const bool use_stack_slots = count <= kMemoryAddStackSlots;
  uint32_t* batch_slots =
      use_stack_slots ? stack_slots : core::runtime_alloc_array<uint32_t>(count);
  if (batch_slots == nullptr) {
    return ASTRAL_E_NOMEM;
  }

  const uint32_t initial_count = index->count;
  bool graph_rebuild_needed = false;
  bool graph_has_new_slots = false;
  for (uint32_t i = 0; i < count; ++i) {
    if (records[i].size != sizeof(AstralMemoryRecord) || records[i].key == 0) {
      if (!use_stack_slots) {
        core::runtime_free_array(batch_slots, count);
      }
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
        if (!use_stack_slots) {
          core::runtime_free_array(batch_slots, count);
        }
        return key_err;
      }
      if (graph_enabled(index)) {
        index->graph_levels[slot] =
            static_cast<uint8_t>(graph_level_for_key(index, records[i].key));
        graph_has_new_slots = true;
      }
    }
    index->slots[slot].record = records[i];
    batch_slots[i] = slot;
    if (graph_enabled(index) && is_update) {
      graph_rebuild_needed = true;
    }
  }

  if (!memory_add_preprocess_parallel(index, vectors, batch_slots, count)) {
    memory_add_preprocess_range(index, vectors, batch_slots, 0, count);
  }

  if (graph_rebuild_needed) {
    graph_rebuild(index);
  } else if (graph_has_new_slots) {
    const uint32_t final_count = index->count;
    index->count = initial_count;
    for (uint32_t i = 0; i < count; ++i) {
      ++index->count;
      graph_connect_slot(index, batch_slots[i]);
    }
    index->count = final_count;
  }
  if (!use_stack_slots) {
    core::runtime_free_array(batch_slots, count);
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
      std::memcpy(cursor, q8_vector_at(index, slot),
                  static_cast<size_t>(index->dim) * sizeof(int8_t));
      cursor += static_cast<size_t>(index->dim) * sizeof(int8_t);
    }
  } else {
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
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
      header.version == kSaveVersionModernLayout || header.version == kSaveVersion;
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
    const uint32_t graph_header_bytes = header.version >= kSaveVersionLegacyLayout
                                            ? sizeof(SaveGraphHeader)
                                            : sizeof(SaveGraphHeaderV3);
    if (bytes.len < layout.graph_offset + graph_header_bytes) {
      return ASTRAL_E_INVALID;
    }
    SaveGraphHeader graph_header{};
    if (header.version >= kSaveVersionLegacyLayout) {
      std::memcpy(&graph_header, bytes.data + layout.graph_offset, sizeof(graph_header));
    } else {
      SaveGraphHeaderV3 graph_header_v3{};
      std::memcpy(&graph_header_v3, bytes.data + layout.graph_offset, sizeof(graph_header_v3));
      graph_header.flags = graph_header_v3.flags;
      graph_header.neighbor_capacity = graph_header_v3.neighbor_capacity;
      graph_header.base_neighbor_capacity = graph_header_v3.neighbor_capacity;
      graph_header.search_capacity = graph_header_v3.search_capacity;
      graph_header.level_capacity = graph_header_v3.level_capacity;
      graph_header.max_level = graph_header_v3.max_level;
      graph_header.entry_active_pos = graph_header_v3.entry_active_pos;
    }
    if (graph_header.flags != kSaveGraphTopologyFlag || graph_header.neighbor_capacity == 0 ||
        graph_header.neighbor_capacity > kGraphMaxNeighbors ||
        graph_header.base_neighbor_capacity == 0 ||
        graph_header.base_neighbor_capacity > kGraphMaxBaseNeighbors ||
        graph_header.base_neighbor_capacity < graph_header.neighbor_capacity ||
        graph_header.level_capacity == 0 || graph_header.level_capacity > kGraphMaxLevels) {
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

AstralErr memory_snapshot_search_with_info(SnapshotBytes bytes,
                                           const AstralMemorySnapshotInfo* info,
                                           const AstralMemorySearchDesc* desc, const float* query,
                                           AstralMemorySearchResult* out_results,
                                           uint32_t max_results, uint32_t* out_count) {
  if (bytes.data == nullptr || info == nullptr || desc == nullptr ||
      desc->size != sizeof(AstralMemorySearchDesc) || query == nullptr || out_count == nullptr ||
      desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }
  if (out_results == nullptr || max_results < desc->top_k) {
    return ASTRAL_E_NOMEM;
  }

  const bool compact = compact_storage_kind(info->storage_kind);
  const bool normalized_f32_cosine = !compact && info->metric == ASTRAL_MEMORY_METRIC_COSINE &&
                                     info->version >= kSaveVersionNormalizedF32Cosine;
  float normalized_query[kMaxDim];
  if (normalized_f32_cosine) {
    normalize_f32_vector(normalized_query, query, info->dim);
    query = normalized_query;
  }
  const float query_scale = info->metric == ASTRAL_MEMORY_METRIC_COSINE && !normalized_f32_cosine
                                ? cosine_scale(query, info->dim)
                                : 1.0f;
  int8_t compact_query[kMaxDim];
  float compact_query_scale = 1.0f;
  const bool use_f6_compact_query = info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3;
  const bool use_f8_compact_query = info->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2;
  if (use_f6_compact_query) {
    quantize_e2m3_vector(compact_query, &compact_query_scale, query, info->dim);
    compact_query_scale *= kE2M3InvScale;
  } else if (use_f8_compact_query) {
    quantize_e5m2_vector(compact_query, &compact_query_scale, query, info->dim);
  }
  uint32_t filled = 0;
  for (uint32_t i = 0; i < info->count; ++i) {
#if defined(__GNUC__) || defined(__clang__)
    if (compact && i + kFlatQ8PrefetchDistance < info->count) {
      const uint64_t prefetch_i = static_cast<uint64_t>(i + kFlatQ8PrefetchDistance);
      __builtin_prefetch(bytes.data + info->scale_offset + prefetch_i * info->scale_stride, 0, 1);
      __builtin_prefetch(bytes.data + info->vector_offset + prefetch_i * info->vector_stride, 0, 1);
      __builtin_prefetch(bytes.data + info->record_offset + prefetch_i * info->record_stride, 0, 1);
    }
#endif
    const uint8_t* record_src =
        bytes.data + info->record_offset + static_cast<uint64_t>(i) * info->record_stride;
    AstralMemoryRecord record{};
    std::memcpy(&record, record_src, sizeof(record));
    if (record.size != sizeof(AstralMemoryRecord) || record.key == 0) {
      return ASTRAL_E_INVALID;
    }
    if (desc->group_id != ASTRAL_MEMORY_GROUP_ANY && record.group_id != desc->group_id) {
      continue;
    }

    float score = 0.0f;
    if (compact) {
      float stored_scale = 1.0f;
      const uint8_t* scale_src =
          bytes.data + info->scale_offset + static_cast<uint64_t>(i) * info->scale_stride;
      std::memcpy(&stored_scale, scale_src, sizeof(stored_scale));
      const float scale = compact_value_scale_kind(info->storage_kind, stored_scale);
      const int8_t* vector = reinterpret_cast<const int8_t*>(
          bytes.data + info->vector_offset + static_cast<uint64_t>(i) * info->vector_stride);
      if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
        score = use_f6_compact_query
                    ? dot_q8_q8(vector, compact_query, info->dim) * scale * compact_query_scale
                : use_f8_compact_query
                    ? dot_e5m2_e5m2(vector, compact_query, info->dim) * scale * compact_query_scale
                    : dot_q8_f32(vector, query, info->dim) * scale;
      } else if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        if (use_f6_compact_query) {
          score = dot_q8_q8(vector, compact_query, info->dim) * scale * compact_query_scale *
                  query_scale;
        } else if (use_f8_compact_query) {
          score = dot_e5m2_e5m2(vector, compact_query, info->dim) * scale * compact_query_scale *
                  query_scale;
        } else {
          const float vector_scale = cosine_scale_q8(vector, stored_scale, info->dim);
          score = dot_q8_f32(vector, query, info->dim) * scale * query_scale * vector_scale;
        }
      } else {
        score = use_f6_compact_query
                    ? l2_score_q8_q8(vector, scale, compact_query, compact_query_scale, info->dim)
                : use_f8_compact_query ? l2_score_e5m2_e5m2(vector, scale, compact_query,
                                                            compact_query_scale, info->dim)
                                       : l2_score_q8_f32(vector, scale, query, info->dim);
      }
    } else {
      const float* vector = reinterpret_cast<const float*>(
          bytes.data + info->vector_offset + static_cast<uint64_t>(i) * info->vector_stride);
      if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
        score = dot_f32(query, vector, info->dim);
      } else if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        score = normalized_f32_cosine ? dot_f32(query, vector, info->dim)
                                      : dot_f32(query, vector, info->dim) * query_scale *
                                            cosine_scale(vector, info->dim);
      } else {
        score = l2_score_f32(query, vector, info->dim);
      }
    }

    MemorySlot slot{};
    slot.record = record;
    AstralMemorySearchResult candidate{};
    fill_result(&candidate, slot, score);
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
                                          out_count);
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

  const AstralHandle handle = core::register_handle(core::HandleKind::MemorySnapshot, view);
  if (handle == 0) {
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
  return memory_snapshot_search_with_info(SnapshotBytes{view->map.data, view->map.size},
                                          &view->info, desc, query, out_results, max_results,
                                          out_count);
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
      header.version == kSaveVersionModernLayout || header.version == kSaveVersion;
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
    if (saved_compact) {
      float scale = 1.0f;
      const uint8_t* scale_src =
          bytes.data + layout.scale_offset + static_cast<uint64_t>(i) * layout.scale_stride;
      std::memcpy(&scale, scale_src, sizeof(float));
      const int8_t* compact = reinterpret_cast<const int8_t*>(
          bytes.data + layout.vector_offset + static_cast<uint64_t>(i) * layout.vector_stride);
      if (dst_compact && saved_storage == desc->storage_kind) {
        saved_compact_scale = scale;
        saved_compact_vector = compact;
      } else {
        for (uint32_t dim_i = 0; dim_i < header.dim; ++dim_i) {
          if (saved_storage == ASTRAL_MEMORY_STORAGE_F6_E2M3) {
            vector[dim_i] = e2m3_scaled_to_f32(compact[dim_i], scale);
          } else if (saved_storage == ASTRAL_MEMORY_STORAGE_F8_E5M2) {
            vector[dim_i] = e5m2_to_f32(static_cast<uint8_t>(compact[dim_i])) * scale;
          } else {
            vector[dim_i] = static_cast<float>(compact[dim_i]) * scale;
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
      if (saved_compact_vector != nullptr) {
        index->q8_scales[slot] = saved_compact_scale;
        std::memcpy(q8_vector_at(index, slot), saved_compact_vector,
                    static_cast<size_t>(index->dim) * sizeof(int8_t));
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
        } else {
          index->slots[slot].score_scale =
              index->metric == ASTRAL_MEMORY_METRIC_COSINE ? 1.0f : 0.0f;
          if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
            quantize_e5m2_cosine_vector(q8_vector_at(index, slot), &index->q8_scales[slot], vector,
                                        index->dim);
          } else {
            quantize_e5m2_vector(q8_vector_at(index, slot), &index->q8_scales[slot], vector,
                                 index->dim);
          }
        }
      }
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
    const uint32_t saved_graph_header_bytes = header.version >= kSaveVersionLegacyLayout
                                                  ? sizeof(SaveGraphHeader)
                                                  : sizeof(SaveGraphHeaderV3);
    if (bytes.len < static_cast<uint64_t>(cursor - bytes.data) + saved_graph_header_bytes) {
      memory_destroy(index);
      return ASTRAL_E_INVALID;
    }
    SaveGraphHeader graph_header{};
    if (header.version >= kSaveVersionLegacyLayout) {
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
        graph_header.base_neighbor_capacity > kGraphMaxBaseNeighbors ||
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
