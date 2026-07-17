#include "memory_index_internal.hpp"

#include "../platform/cpu_features.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#if defined(__F16C__) || defined(_MSC_VER)
#define ASTRAL_X86_F16C 1
#endif
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace astral::inference {

namespace {
constexpr int32_t kQ8MinValue = -127;
constexpr int32_t kQ8MaxValue = 127;
constexpr float kQ8MaxFloat = 127.0f;
constexpr int32_t kE2M3Scale = 16;
constexpr float kE2M3MaxFloat = 7.5f;
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
constexpr float kE3M2MaxFloat = 28.0f;
constexpr int32_t kE3M2Scale = 16;
constexpr uint32_t kE3M2MaxScaled = 448;
constexpr float kE3M2TieBias = 0.5f;
constexpr float kE3M2UnitBandMaxThreshold = 7.5f;
constexpr uint32_t kE3M2StepBandCount = 6;
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
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr uint32_t kE5M2F32ExponentOffset = kF32ExponentBias - kE5M2ExponentBias;
#endif
constexpr uint32_t kE5M2MantissaRoundBit = 1u << 20u;
constexpr uint32_t kE5M2MantissaShift = 21;
constexpr uint32_t kE5M2MantissaOverflow = 4;
constexpr uint32_t kE5M2MaxFinite = 0x7Bu;
constexpr float kE5M2MaxFloat = 57344.0f;
constexpr float kE5M2SubnormalUnit = 1.0f / 65536.0f;
constexpr float kL2CrossTermScale = 2.0f;
#if defined(__AVX2__)
constexpr uint32_t kAvx2F32Lanes = 8;
constexpr uint32_t kAvx2I8Lanes = 16;
constexpr uint32_t kAvx2UnrollVectors = 4;
constexpr uint32_t kAvx2UnrollF32 = kAvx2F32Lanes * kAvx2UnrollVectors;
constexpr uint32_t kAvx2UnrollI8 = kAvx2I8Lanes * kAvx2UnrollVectors;
constexpr uint32_t kAvx2Offset1 = kAvx2F32Lanes;
constexpr uint32_t kAvx2Offset2 = kAvx2F32Lanes * 2u;
constexpr uint32_t kAvx2Offset3 = kAvx2F32Lanes * 3u;
constexpr uint32_t kAvx2I8Offset1 = kAvx2I8Lanes;
constexpr uint32_t kAvx2I8Offset2 = kAvx2I8Lanes * 2u;
constexpr uint32_t kAvx2I8Offset3 = kAvx2I8Lanes * 3u;
constexpr uint32_t kAvxVnniI8Lanes = 32;
constexpr uint32_t kAvxVnniUnrollI8 = kAvxVnniI8Lanes * kAvx2UnrollVectors;
constexpr uint32_t kAvxVnniI8Offset1 = kAvxVnniI8Lanes;
constexpr uint32_t kAvxVnniI8Offset2 = kAvxVnniI8Lanes * 2u;
constexpr uint32_t kAvxVnniI8Offset3 = kAvxVnniI8Lanes * 3u;
constexpr int32_t kAvxVnniSignOffset = 128;
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
constexpr uint32_t kNeonF32Lanes = 4;
constexpr uint32_t kNeonI8Lanes = 16;
constexpr uint32_t kNeonI8HalfLanes = 8;
constexpr uint32_t kNeonUnrollVectors = 4;
constexpr uint32_t kNeonUnrollF32 = kNeonF32Lanes * kNeonUnrollVectors;
constexpr uint32_t kNeonOffset1 = kNeonF32Lanes;
constexpr uint32_t kNeonOffset2 = kNeonF32Lanes * 2u;
constexpr uint32_t kNeonOffset3 = kNeonF32Lanes * 3u;
constexpr uint32_t kNeonI8Offset1 = kNeonI8HalfLanes;
#endif
} // namespace

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

#if (defined(__GNUC__) || defined(__clang__)) &&                                                   \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
inline bool cpu_supports_avx_vnni() {
  static const bool supported = __builtin_cpu_supports("avxvnni") != 0;
  return supported;
}

__attribute__((target("avxvnni"))) int32_t dot_q8_q8_avx_vnni_i32(const int8_t* a, const int8_t* b,
                                                                  uint32_t dim, int32_t b_sum) {
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256();
  __m256i acc3 = _mm256_setzero_si256();
  const __m256i sign_offset = _mm256_set1_epi8(static_cast<char>(0x80));
  uint32_t i = 0;
  for (; i + kAvxVnniUnrollI8 <= dim; i += kAvxVnniUnrollI8) {
    const __m256i a0 =
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)), sign_offset);
    const __m256i a1 = _mm256_xor_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + kAvxVnniI8Offset1)),
        sign_offset);
    const __m256i a2 = _mm256_xor_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + kAvxVnniI8Offset2)),
        sign_offset);
    const __m256i a3 = _mm256_xor_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + kAvxVnniI8Offset3)),
        sign_offset);
    acc0 =
        _mm256_dpbusd_epi32(acc0, a0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
    acc1 = _mm256_dpbusd_epi32(
        acc1, a1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + kAvxVnniI8Offset1)));
    acc2 = _mm256_dpbusd_epi32(
        acc2, a2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + kAvxVnniI8Offset2)));
    acc3 = _mm256_dpbusd_epi32(
        acc3, a3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + kAvxVnniI8Offset3)));
  }
  for (; i + kAvxVnniI8Lanes <= dim; i += kAvxVnniI8Lanes) {
    const __m256i av =
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)), sign_offset);
    acc0 =
        _mm256_dpbusd_epi32(acc0, av, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
  }
  acc0 = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
  return reduce_avx2_i32(acc0) - b_sum * kAvxVnniSignOffset;
}

__attribute__((target("avxvnni"))) int32_t dot_q8_q8_avx_vnni_aligned_i32(const int8_t* a,
                                                                          const int8_t* b,
                                                                          uint32_t dim,
                                                                          int32_t b_sum) {
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256();
  __m256i acc3 = _mm256_setzero_si256();
  const __m256i sign_offset = _mm256_set1_epi8(static_cast<char>(0x80));
  uint32_t i = 0;
  for (; i + kAvxVnniUnrollI8 <= dim; i += kAvxVnniUnrollI8) {
    const __m256i a0 =
        _mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i*>(a + i)), sign_offset);
    const __m256i a1 = _mm256_xor_si256(
        _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i + kAvxVnniI8Offset1)),
        sign_offset);
    const __m256i a2 = _mm256_xor_si256(
        _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i + kAvxVnniI8Offset2)),
        sign_offset);
    const __m256i a3 = _mm256_xor_si256(
        _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i + kAvxVnniI8Offset3)),
        sign_offset);
    acc0 =
        _mm256_dpbusd_epi32(acc0, a0, _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i)));
    acc1 = _mm256_dpbusd_epi32(
        acc1, a1, _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i + kAvxVnniI8Offset1)));
    acc2 = _mm256_dpbusd_epi32(
        acc2, a2, _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i + kAvxVnniI8Offset2)));
    acc3 = _mm256_dpbusd_epi32(
        acc3, a3, _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i + kAvxVnniI8Offset3)));
  }
  for (; i + kAvxVnniI8Lanes <= dim; i += kAvxVnniI8Lanes) {
    const __m256i av =
        _mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i*>(a + i)), sign_offset);
    acc0 =
        _mm256_dpbusd_epi32(acc0, av, _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i)));
  }
  acc0 = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
  return reduce_avx2_i32(acc0) - b_sum * kAvxVnniSignOffset;
}
#endif

inline void quantize_q8_store8_avx2(int8_t* dst, const float* src, __m256 inv_scale,
                                    __m256 positive_bias, __m256 negative_bias, __m256 zero,
                                    __m256i min_value, __m256i max_value) {
  const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(src), inv_scale);
  const __m256 bias =
      _mm256_blendv_ps(negative_bias, positive_bias, _mm256_cmp_ps(scaled, zero, _CMP_GE_OQ));
  __m256i rounded = _mm256_cvttps_epi32(_mm256_add_ps(scaled, bias));
  rounded = _mm256_max_epi32(min_value, _mm256_min_epi32(max_value, rounded));
  const __m128i lo = _mm256_castsi256_si128(rounded);
  const __m128i hi = _mm256_extracti128_si256(rounded, 1);
  const __m128i packed16 = _mm_packs_epi32(lo, hi);
  const __m128i packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128());
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), packed8);
}

inline void quantize_e5m2_store8_avx2(int8_t* dst, const float* src, __m256 inv_scale) {
  const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(src), inv_scale);
  const __m256i bits = _mm256_castps_si256(scaled);
  const __m256i sign = _mm256_and_si256(_mm256_srli_epi32(bits, 24), _mm256_set1_epi32(0x80));
  const __m256i abs_bits = _mm256_and_si256(bits, _mm256_set1_epi32(static_cast<int>(kF32AbsMask)));
  const __m256i exponent = _mm256_and_si256(_mm256_srli_epi32(abs_bits, kF32ExponentShift),
                                            _mm256_set1_epi32(kF32ExponentMask));
  __m256i e5_exponent =
      _mm256_sub_epi32(exponent, _mm256_set1_epi32(kF32ExponentBias - kE5M2ExponentBias));
  __m256i e5_mantissa = _mm256_srli_epi32(
      _mm256_add_epi32(_mm256_and_si256(abs_bits, _mm256_set1_epi32(kF32MantissaMask)),
                       _mm256_set1_epi32(kE5M2MantissaRoundBit)),
      kE5M2MantissaShift);
  const __m256i mantissa_overflow =
      _mm256_cmpeq_epi32(e5_mantissa, _mm256_set1_epi32(kE5M2MantissaOverflow));
  e5_mantissa = _mm256_blendv_epi8(e5_mantissa, _mm256_setzero_si256(), mantissa_overflow);
  e5_exponent =
      _mm256_add_epi32(e5_exponent, _mm256_and_si256(mantissa_overflow, _mm256_set1_epi32(1)));

  const __m256 abs_scaled = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), scaled);
  __m256i subnormal = _mm256_cvttps_epi32(_mm256_add_ps(
      _mm256_mul_ps(abs_scaled, _mm256_set1_ps(1.0f / kE5M2SubnormalUnit)), _mm256_set1_ps(0.5f)));
  subnormal = _mm256_min_epi32(subnormal, _mm256_set1_epi32(kE5M2MantissaMask));
  const __m256i subnormal_raw = _mm256_or_si256(sign, subnormal);
  const __m256i normal_raw = _mm256_or_si256(
      sign, _mm256_or_si256(_mm256_slli_epi32(e5_exponent, kE5M2ExponentShift), e5_mantissa));
  const __m256i saturated_raw = _mm256_or_si256(sign, _mm256_set1_epi32(kE5M2MaxFinite));
  const __m256i zero_raw = sign;
  const __m256i subnormal_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e5_exponent);
  const __m256i saturated_mask =
      _mm256_or_si256(_mm256_cmpgt_epi32(e5_exponent, _mm256_set1_epi32(kE5M2ExponentMask - 1u)),
                      _mm256_cmpgt_epi32(abs_bits, _mm256_set1_epi32(kF32InfBits - 1u)));
  const __m256i zero_mask = _mm256_cmpeq_epi32(abs_bits, _mm256_setzero_si256());
  __m256i raw = _mm256_blendv_epi8(normal_raw, subnormal_raw, subnormal_mask);
  raw = _mm256_blendv_epi8(raw, saturated_raw, saturated_mask);
  raw = _mm256_blendv_epi8(raw, zero_raw, zero_mask);
  const __m128i lo = _mm256_castsi256_si128(raw);
  const __m128i hi = _mm256_extracti128_si256(raw, 1);
  const __m128i packed16 = _mm_packus_epi32(lo, hi);
  const __m128i packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), packed8);
}

#if defined(ASTRAL_X86_F16C)
template <bool ClampNonFinite> inline __m256 e5m2_load8_f32_avx2(const int8_t* src) {
  __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
  if constexpr (ClampNonFinite) {
    const __m128i sign = _mm_and_si128(bytes, _mm_set1_epi8(static_cast<char>(0x80)));
    const __m128i magnitude = _mm_min_epu8(_mm_and_si128(bytes, _mm_set1_epi8(0x7f)),
                                           _mm_set1_epi8(static_cast<char>(kE5M2MaxFinite)));
    bytes = _mm_or_si128(sign, magnitude);
  }
  const __m256i half_u32 = _mm256_slli_epi32(_mm256_cvtepu8_epi32(bytes), 8);
  const __m128i half_lo = _mm256_castsi256_si128(half_u32);
  const __m128i half_hi = _mm256_extracti128_si256(half_u32, 1);
  const __m128i half = _mm_packus_epi32(half_lo, half_hi);
  return _mm256_cvtph_ps(half);
}
#endif
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

inline float32x4_t e5m2_u32_to_f32_neon(uint32x4_t raw) {
  const uint32x4_t sign = vshlq_n_u32(vandq_u32(raw, vdupq_n_u32(1u << kE5M2SignShift)), 24);
  const uint32x4_t exponent =
      vandq_u32(vshrq_n_u32(raw, kE5M2ExponentShift), vdupq_n_u32(kE5M2ExponentMask));
  const uint32x4_t mantissa = vandq_u32(raw, vdupq_n_u32(kE5M2MantissaMask));
  const uint32x4_t normal_bits = vorrq_u32(
      sign, vorrq_u32(vshlq_n_u32(vaddq_u32(exponent, vdupq_n_u32(kE5M2F32ExponentOffset)),
                                  kF32ExponentShift),
                      vshlq_n_u32(mantissa, kE5M2MantissaShift)));
  const float32x4_t subnormal = vmulq_n_f32(vcvtq_f32_u32(mantissa), kE5M2SubnormalUnit);
  const float32x4_t signed_subnormal =
      vreinterpretq_f32_u32(vorrq_u32(sign, vreinterpretq_u32_f32(subnormal)));
  const float32x4_t saturated =
      vreinterpretq_f32_u32(vorrq_u32(sign, vreinterpretq_u32_f32(vdupq_n_f32(kE5M2MaxFloat))));
  const float32x4_t normal = vreinterpretq_f32_u32(normal_bits);
  const float32x4_t finite =
      vbslq_f32(vceqq_u32(exponent, vdupq_n_u32(0u)), signed_subnormal, normal);
  return vbslq_f32(vceqq_u32(exponent, vdupq_n_u32(kE5M2ExponentMask)), saturated, finite);
}

inline void e5m2_load8_f32_neon(const int8_t* src, float32x4_t* out_low, float32x4_t* out_high) {
  const uint16x8_t raw16 = vmovl_u8(vld1_u8(reinterpret_cast<const uint8_t*>(src)));
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  // E5M2 is the high byte of binary16. Native FP16 conversion needs only this lane shift; ARMv8
  // builds without vector FP16 keep the integer bit-construction path below.
  const float16x8_t half = vreinterpretq_f16_u16(vshlq_n_u16(raw16, 8));
  *out_low = vcvt_f32_f16(vget_low_f16(half));
  *out_high = vcvt_f32_f16(vget_high_f16(half));
#else
  *out_low = e5m2_u32_to_f32_neon(vmovl_u16(vget_low_u16(raw16)));
  *out_high = e5m2_u32_to_f32_neon(vmovl_u16(vget_high_u16(raw16)));
#endif
}
#endif

float dot_f32(const float* a, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  if (dim == kAvx2UnrollF32) {
    const __m256 product0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    const __m256 product1 =
        _mm256_mul_ps(_mm256_loadu_ps(a + kAvx2Offset1), _mm256_loadu_ps(b + kAvx2Offset1));
    const __m256 product2 =
        _mm256_mul_ps(_mm256_loadu_ps(a + kAvx2Offset2), _mm256_loadu_ps(b + kAvx2Offset2));
    const __m256 product3 =
        _mm256_mul_ps(_mm256_loadu_ps(a + kAvx2Offset3), _mm256_loadu_ps(b + kAvx2Offset3));
    return reduce_avx2_f32(
        _mm256_add_ps(_mm256_add_ps(product0, product1), _mm256_add_ps(product2, product3)));
  }
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

float dot_i16_f32(const int16_t* a, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256 av0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(bytes)));
    const __m256 av1 =
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bytes, 1)));
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
    sum += static_cast<float>(a[i]) * b[i];
  }
  return sum;
#elif defined(__aarch64__) && defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + kNeonI8HalfLanes <= dim; i += kNeonI8HalfLanes) {
    const int16x8_t av = vld1q_s16(a + i);
    acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(av))), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(av))),
                     vld1q_f32(b + i + kNeonF32Lanes));
  }
  float32x4_t acc = vaddq_f32(acc0, acc1);
  for (; i + kNeonF32Lanes <= dim; i += kNeonF32Lanes) {
    const int16x4_t av = vld1_s16(a + i);
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(av)), vld1q_f32(b + i));
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

int32_t sum_i8(const int8_t* v, uint32_t dim) {
  int32_t sum0 = 0;
  int32_t sum1 = 0;
  int32_t sum2 = 0;
  int32_t sum3 = 0;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    sum0 += static_cast<int32_t>(v[i]);
    sum1 += static_cast<int32_t>(v[i + 1u]);
    sum2 += static_cast<int32_t>(v[i + 2u]);
    sum3 += static_cast<int32_t>(v[i + 3u]);
  }
  int32_t sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(v[i]);
  }
  return sum;
}

float dot_q8_q8_query(const int8_t* a, const int8_t* b, uint32_t dim, int32_t b_sum) {
#if defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__)) &&                              \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
  if ((dim & 31u) == 0 && cpu_supports_avx_vnni()) {
    return static_cast<float>(dot_q8_q8_avx_vnni_i32(a, b, dim, b_sum));
  }
#else
  (void)b_sum;
#endif
  return dot_q8_q8(a, b, dim);
}

float dot_q8_q8_query_aligned(const int8_t* a, const int8_t* b, uint32_t dim, int32_t b_sum) {
#if defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__)) &&                              \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
  if ((dim & 31u) == 0 && cpu_supports_avx_vnni()) {
    return static_cast<float>(dot_q8_q8_avx_vnni_aligned_i32(a, b, dim, b_sum));
  }
#else
  (void)b_sum;
#endif
  return dot_q8_q8(a, b, dim);
}

float dot_i16_i16(const int16_t* a, const int16_t* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256();
  __m256i acc3 = _mm256_setzero_si256();
  uint32_t i = 0;
  for (; i + kAvx2UnrollI8 <= dim; i += kAvx2UnrollI8) {
    acc0 = _mm256_add_epi32(
        acc0, _mm256_madd_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)),
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i))));
    acc1 = _mm256_add_epi32(
        acc1, _mm256_madd_epi16(
                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + kAvx2I8Offset1)),
                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + kAvx2I8Offset1))));
    acc2 = _mm256_add_epi32(
        acc2, _mm256_madd_epi16(
                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + kAvx2I8Offset2)),
                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + kAvx2I8Offset2))));
    acc3 = _mm256_add_epi32(
        acc3, _mm256_madd_epi16(
                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + kAvx2I8Offset3)),
                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + kAvx2I8Offset3))));
  }
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    acc0 = _mm256_add_epi32(
        acc0, _mm256_madd_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)),
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i))));
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
    acc0 = vmlal_s16(acc0, vget_low_s16(vld1q_s16(a + i)), vget_low_s16(vld1q_s16(b + i)));
    acc1 = vmlal_s16(acc1, vget_high_s16(vld1q_s16(a + i)), vget_high_s16(vld1q_s16(b + i)));
  }
  int32_t sum = vaddvq_s32(vaddq_s32(acc0, acc1));
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
#else
  int64_t sum0 = 0;
  int64_t sum1 = 0;
  int64_t sum2 = 0;
  int64_t sum3 = 0;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    sum0 += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    sum1 += static_cast<int32_t>(a[i + 1u]) * static_cast<int32_t>(b[i + 1u]);
    sum2 += static_cast<int32_t>(a[i + 2u]) * static_cast<int32_t>(b[i + 2u]);
    sum3 += static_cast<int32_t>(a[i + 3u]) * static_cast<int32_t>(b[i + 3u]);
  }
  int64_t sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
#endif
}

float dot_i16_i16_aligned(const int16_t* a, const int16_t* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256();
  __m256i acc3 = _mm256_setzero_si256();
  uint32_t i = 0;
  for (; i + kAvx2UnrollI8 <= dim; i += kAvx2UnrollI8) {
    acc0 = _mm256_add_epi32(
        acc0, _mm256_madd_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(a + i)),
                                _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i))));
    acc1 = _mm256_add_epi32(
        acc1, _mm256_madd_epi16(
                  _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i + kAvx2I8Offset1)),
                  _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i + kAvx2I8Offset1))));
    acc2 = _mm256_add_epi32(
        acc2, _mm256_madd_epi16(
                  _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i + kAvx2I8Offset2)),
                  _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i + kAvx2I8Offset2))));
    acc3 = _mm256_add_epi32(
        acc3, _mm256_madd_epi16(
                  _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i + kAvx2I8Offset3)),
                  _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i + kAvx2I8Offset3))));
  }
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    acc0 = _mm256_add_epi32(
        acc0, _mm256_madd_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(a + i)),
                                _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i))));
  }
  acc0 = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
  int32_t sum = reduce_avx2_i32(acc0);
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
#else
  return dot_i16_i16(a, b, dim);
#endif
}

float max_abs_f32(const float* v, uint32_t dim) {
#if defined(__AVX2__)
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + kAvx2UnrollF32 <= dim; i += kAvx2UnrollF32) {
    acc0 = _mm256_max_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(v + i)));
    acc1 = _mm256_max_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(v + i + kAvx2Offset1)));
    acc2 = _mm256_max_ps(acc2, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(v + i + kAvx2Offset2)));
    acc3 = _mm256_max_ps(acc3, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(v + i + kAvx2Offset3)));
  }
  acc0 = _mm256_max_ps(_mm256_max_ps(acc0, acc1), _mm256_max_ps(acc2, acc3));
  for (; i + kAvx2F32Lanes <= dim; i += kAvx2F32Lanes) {
    acc0 = _mm256_max_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(v + i)));
  }
  alignas(kVectorStorageAlign) float lanes[kAvx2F32Lanes];
  _mm256_store_ps(lanes, acc0);
  float max_abs = lanes[0];
  for (uint32_t lane = 1; lane < kAvx2F32Lanes; ++lane) {
    if (lanes[lane] > max_abs) {
      max_abs = lanes[lane];
    }
  }
  for (; i < dim; ++i) {
    const float abs_value = std::fabs(v[i]);
    if (abs_value > max_abs) {
      max_abs = abs_value;
    }
  }
  return max_abs;
#elif defined(__aarch64__) && defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + kNeonUnrollF32 <= dim; i += kNeonUnrollF32) {
    acc0 = vmaxq_f32(acc0, vabsq_f32(vld1q_f32(v + i)));
    acc1 = vmaxq_f32(acc1, vabsq_f32(vld1q_f32(v + i + kNeonOffset1)));
    acc2 = vmaxq_f32(acc2, vabsq_f32(vld1q_f32(v + i + kNeonOffset2)));
    acc3 = vmaxq_f32(acc3, vabsq_f32(vld1q_f32(v + i + kNeonOffset3)));
  }
  acc0 = vmaxq_f32(vmaxq_f32(acc0, acc1), vmaxq_f32(acc2, acc3));
  for (; i + kNeonF32Lanes <= dim; i += kNeonF32Lanes) {
    acc0 = vmaxq_f32(acc0, vabsq_f32(vld1q_f32(v + i)));
  }
  float max_abs = vmaxvq_f32(acc0);
  for (; i < dim; ++i) {
    const float abs_value = std::fabs(v[i]);
    if (abs_value > max_abs) {
      max_abs = abs_value;
    }
  }
  return max_abs;
#else
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float abs_value = std::fabs(v[i]);
    if (abs_value > max_abs) {
      max_abs = abs_value;
    }
  }
  return max_abs;
#endif
}

void scale_f32_vector(float* dst, const float* src, float scale, uint32_t dim) {
#if defined(__AVX2__)
  const __m256 scale_v = _mm256_set1_ps(scale);
  uint32_t i = 0;
  for (; i + kAvx2UnrollF32 <= dim; i += kAvx2UnrollF32) {
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), scale_v));
    _mm256_storeu_ps(dst + i + kAvx2Offset1,
                     _mm256_mul_ps(_mm256_loadu_ps(src + i + kAvx2Offset1), scale_v));
    _mm256_storeu_ps(dst + i + kAvx2Offset2,
                     _mm256_mul_ps(_mm256_loadu_ps(src + i + kAvx2Offset2), scale_v));
    _mm256_storeu_ps(dst + i + kAvx2Offset3,
                     _mm256_mul_ps(_mm256_loadu_ps(src + i + kAvx2Offset3), scale_v));
  }
  for (; i + kAvx2F32Lanes <= dim; i += kAvx2F32Lanes) {
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), scale_v));
  }
  for (; i < dim; ++i) {
    dst[i] = src[i] * scale;
  }
#elif defined(__aarch64__) && defined(__ARM_NEON)
  const float32x4_t scale_v = vdupq_n_f32(scale);
  uint32_t i = 0;
  for (; i + kNeonUnrollF32 <= dim; i += kNeonUnrollF32) {
    vst1q_f32(dst + i, vmulq_f32(vld1q_f32(src + i), scale_v));
    vst1q_f32(dst + i + kNeonOffset1, vmulq_f32(vld1q_f32(src + i + kNeonOffset1), scale_v));
    vst1q_f32(dst + i + kNeonOffset2, vmulq_f32(vld1q_f32(src + i + kNeonOffset2), scale_v));
    vst1q_f32(dst + i + kNeonOffset3, vmulq_f32(vld1q_f32(src + i + kNeonOffset3), scale_v));
  }
  for (; i + kNeonF32Lanes <= dim; i += kNeonF32Lanes) {
    vst1q_f32(dst + i, vmulq_f32(vld1q_f32(src + i), scale_v));
  }
  for (; i < dim; ++i) {
    dst[i] = src[i] * scale;
  }
#else
  for (uint32_t i = 0; i < dim; ++i) {
    dst[i] = src[i] * scale;
  }
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

float e5m2_to_f32(uint8_t raw) {
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

void copy_finite_e5m2(int8_t* dst, const int8_t* src, uint32_t dim) {
  for (uint32_t i = 0; i < dim; ++i) {
    const uint8_t value = static_cast<uint8_t>(src[i]);
    const uint8_t magnitude = value & 0x7fu;
    dst[i] = static_cast<int8_t>((value & 0x80u) |
                                 (magnitude < kE5M2MaxFinite ? magnitude : kE5M2MaxFinite));
  }
}

bool f32_values_finite(const float* values, size_t count) {
  size_t i = 0;
#if defined(__AVX2__)
  const __m256i exponent_mask = _mm256_set1_epi32(static_cast<int32_t>(kF32InfBits));
  for (; i + kAvx2F32Lanes <= count; i += kAvx2F32Lanes) {
    const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + i));
    const __m256i exponent = _mm256_and_si256(bits, exponent_mask);
    const __m256i non_finite = _mm256_cmpeq_epi32(exponent, exponent_mask);
    if (!_mm256_testz_si256(non_finite, non_finite)) {
      return false;
    }
  }
#elif defined(__aarch64__) && defined(__ARM_NEON)
  const uint32x4_t exponent_mask = vdupq_n_u32(kF32InfBits);
  for (; i + 4u <= count; i += 4u) {
    const uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(values + i));
    const uint32x4_t non_finite = vceqq_u32(vandq_u32(bits, exponent_mask), exponent_mask);
    if (vmaxvq_u32(non_finite) != 0) {
      return false;
    }
  }
#endif
  for (; i < count; ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, values + i, sizeof(bits));
    if ((bits & kF32InfBits) == kF32InfBits) {
      return false;
    }
  }
  return true;
}

#if defined(ASTRAL_X86_F16C)
float dot_e5m2_f32_scalar(const int8_t* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    sum += e5m2_to_f32(static_cast<uint8_t>(a[i])) * b[i];
  }
  return sum;
}

float dot_e5m2_e5m2_scalar(const int8_t* a, const int8_t* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    sum += e5m2_to_f32(static_cast<uint8_t>(a[i])) * e5m2_to_f32(static_cast<uint8_t>(b[i]));
  }
  return sum;
}

float l2_score_e5m2_f32_scalar(const int8_t* a, float scale, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float d = e5m2_to_f32(static_cast<uint8_t>(a[i])) * scale - b[i];
    sum += d * d;
  }
  return -sum;
}

float l2_score_e5m2_e5m2_scalar(const int8_t* a, float a_scale, const int8_t* b, float b_scale,
                                uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float d = e5m2_to_f32(static_cast<uint8_t>(a[i])) * a_scale -
                    e5m2_to_f32(static_cast<uint8_t>(b[i])) * b_scale;
    sum += d * d;
  }
  return -sum;
}
#endif

template <bool UseF16c> float dot_e5m2_f32_impl(const int8_t* a, const float* b, uint32_t dim) {
#if defined(__AVX2__)
#if defined(ASTRAL_X86_F16C)
  if constexpr (!UseF16c) {
    return dot_e5m2_f32_scalar(a, b, dim);
  } else {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + kAvx2UnrollF32 <= dim; i += kAvx2UnrollF32) {
#if defined(__FMA__)
      acc0 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<false>(a + i), _mm256_loadu_ps(b + i), acc0);
      acc1 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2F32Lanes),
                             _mm256_loadu_ps(b + i + kAvx2F32Lanes), acc1);
      acc2 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2Offset2),
                             _mm256_loadu_ps(b + i + kAvx2Offset2), acc2);
      acc3 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2Offset3),
                             _mm256_loadu_ps(b + i + kAvx2Offset3), acc3);
#else
      acc0 = _mm256_add_ps(
          acc0, _mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i), _mm256_loadu_ps(b + i)));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2F32Lanes),
                                               _mm256_loadu_ps(b + i + kAvx2F32Lanes)));
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2Offset2),
                                               _mm256_loadu_ps(b + i + kAvx2Offset2)));
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2Offset3),
                                               _mm256_loadu_ps(b + i + kAvx2Offset3)));
#endif
    }
    float sum =
        reduce_avx2_f32(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < dim; ++i) {
      sum += e5m2_to_f32(static_cast<uint8_t>(a[i])) * b[i];
    }
    return sum;
  }
#else
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
#endif
#elif defined(__aarch64__) && defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + kNeonUnrollF32 <= dim; i += kNeonUnrollF32) {
    float32x4_t e0;
    float32x4_t e1;
    float32x4_t e2;
    float32x4_t e3;
    e5m2_load8_f32_neon(a + i, &e0, &e1);
    e5m2_load8_f32_neon(a + i + kNeonI8HalfLanes, &e2, &e3);
    acc0 = vfmaq_f32(acc0, e0, vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, e1, vld1q_f32(b + i + kNeonF32Lanes));
    acc2 = vfmaq_f32(acc2, e2, vld1q_f32(b + i + kNeonOffset2));
    acc3 = vfmaq_f32(acc3, e3, vld1q_f32(b + i + kNeonOffset3));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
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

float dot_e5m2_f32_384(const int8_t* a, const float* b) {
#if defined(__AVX2__) && defined(ASTRAL_X86_F16C)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  // The target dimension is exactly twelve 32-lane blocks. Expanding them removes loop and tail
  // control from every one of the 100,000 vector scores while retaining four live accumulators.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 12
#endif
  for (uint32_t i = 0; i < 384u; i += kAvx2UnrollF32) {
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<false>(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2F32Lanes),
                           _mm256_loadu_ps(b + i + kAvx2F32Lanes), acc1);
    acc2 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2Offset2),
                           _mm256_loadu_ps(b + i + kAvx2Offset2), acc2);
    acc3 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2Offset3),
                           _mm256_loadu_ps(b + i + kAvx2Offset3), acc3);
#else
    acc0 = _mm256_add_ps(acc0,
                         _mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i), _mm256_loadu_ps(b + i)));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2F32Lanes),
                                             _mm256_loadu_ps(b + i + kAvx2F32Lanes)));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2Offset2),
                                             _mm256_loadu_ps(b + i + kAvx2Offset2)));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2Offset3),
                                             _mm256_loadu_ps(b + i + kAvx2Offset3)));
#endif
  }
  return reduce_avx2_f32(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
#else
  return dot_e5m2_f32_impl<false>(a, b, 384u);
#endif
}

template <bool UseF16c, bool ClampA>
float dot_e5m2_e5m2_impl(const int8_t* a, const int8_t* b, uint32_t dim) {
#if defined(__AVX2__)
#if defined(ASTRAL_X86_F16C)
  if constexpr (!UseF16c) {
    return dot_e5m2_e5m2_scalar(a, b, dim);
  } else {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + kAvx2UnrollF32 <= dim; i += kAvx2UnrollF32) {
#if defined(__FMA__)
      acc0 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<ClampA>(a + i), e5m2_load8_f32_avx2<false>(b + i),
                             acc0);
      acc1 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<ClampA>(a + i + kAvx2F32Lanes),
                             e5m2_load8_f32_avx2<false>(b + i + kAvx2F32Lanes), acc1);
      acc2 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<ClampA>(a + i + kAvx2Offset2),
                             e5m2_load8_f32_avx2<false>(b + i + kAvx2Offset2), acc2);
      acc3 = _mm256_fmadd_ps(e5m2_load8_f32_avx2<ClampA>(a + i + kAvx2Offset3),
                             e5m2_load8_f32_avx2<false>(b + i + kAvx2Offset3), acc3);
#else
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(e5m2_load8_f32_avx2<ClampA>(a + i),
                                               e5m2_load8_f32_avx2<false>(b + i)));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(e5m2_load8_f32_avx2<ClampA>(a + i + kAvx2F32Lanes),
                                               e5m2_load8_f32_avx2<false>(b + i + kAvx2F32Lanes)));
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(e5m2_load8_f32_avx2<ClampA>(a + i + kAvx2Offset2),
                                               e5m2_load8_f32_avx2<false>(b + i + kAvx2Offset2)));
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(e5m2_load8_f32_avx2<ClampA>(a + i + kAvx2Offset3),
                                               e5m2_load8_f32_avx2<false>(b + i + kAvx2Offset3)));
#endif
    }
    float sum =
        reduce_avx2_f32(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < dim; ++i) {
      sum += e5m2_to_f32(static_cast<uint8_t>(a[i])) * e5m2_to_f32(static_cast<uint8_t>(b[i]));
    }
    return sum;
  }
#else
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
#endif
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

template <bool UseF16c>
float l2_score_e5m2_f32_impl(const int8_t* a, float scale, const float* b, uint32_t dim) {
#if defined(__AVX2__) && defined(ASTRAL_X86_F16C)
  if constexpr (!UseF16c) {
    return l2_score_e5m2_f32_scalar(a, scale, b, dim);
  } else {
    const __m256 scale_v = _mm256_set1_ps(scale);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + kAvx2F32Lanes * 2u <= dim; i += kAvx2F32Lanes * 2u) {
      const __m256 d0 = _mm256_sub_ps(_mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i), scale_v),
                                      _mm256_loadu_ps(b + i));
      const __m256 d1 =
          _mm256_sub_ps(_mm256_mul_ps(e5m2_load8_f32_avx2<false>(a + i + kAvx2F32Lanes), scale_v),
                        _mm256_loadu_ps(b + i + kAvx2F32Lanes));
#if defined(__FMA__)
      acc0 = _mm256_fmadd_ps(d0, d0, acc0);
      acc1 = _mm256_fmadd_ps(d1, d1, acc1);
#else
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
#endif
    }
    float sum = reduce_avx2_f32(_mm256_add_ps(acc0, acc1));
    for (; i < dim; ++i) {
      const float d = e5m2_to_f32(static_cast<uint8_t>(a[i])) * scale - b[i];
      sum += d * d;
    }
    return -sum;
  }
#else
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
#endif
}

template <bool UseF16c, bool ClampA>
float l2_score_e5m2_e5m2_impl(const int8_t* a, float a_scale, const int8_t* b, float b_scale,
                              uint32_t dim) {
#if defined(__AVX2__) && defined(ASTRAL_X86_F16C)
  if constexpr (!UseF16c) {
    return l2_score_e5m2_e5m2_scalar(a, a_scale, b, b_scale, dim);
  } else {
    const __m256 a_scale_v = _mm256_set1_ps(a_scale);
    const __m256 b_scale_v = _mm256_set1_ps(b_scale);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + kAvx2F32Lanes * 2u <= dim; i += kAvx2F32Lanes * 2u) {
      const __m256 d0 = _mm256_sub_ps(_mm256_mul_ps(e5m2_load8_f32_avx2<ClampA>(a + i), a_scale_v),
                                      _mm256_mul_ps(e5m2_load8_f32_avx2<false>(b + i), b_scale_v));
      const __m256 d1 = _mm256_sub_ps(
          _mm256_mul_ps(e5m2_load8_f32_avx2<ClampA>(a + i + kAvx2F32Lanes), a_scale_v),
          _mm256_mul_ps(e5m2_load8_f32_avx2<false>(b + i + kAvx2F32Lanes), b_scale_v));
#if defined(__FMA__)
      acc0 = _mm256_fmadd_ps(d0, d0, acc0);
      acc1 = _mm256_fmadd_ps(d1, d1, acc1);
#else
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
#endif
    }
    float sum = reduce_avx2_f32(_mm256_add_ps(acc0, acc1));
    for (; i < dim; ++i) {
      const float d = e5m2_to_f32(static_cast<uint8_t>(a[i])) * a_scale -
                      e5m2_to_f32(static_cast<uint8_t>(b[i])) * b_scale;
      sum += d * d;
    }
    return -sum;
  }
#else
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
#endif
}


const E5m2Kernels* select_e5m2_kernels() {
  static constexpr E5m2Kernels kFallback = {
      dot_e5m2_f32_impl<false>,
      dot_e5m2_e5m2_impl<false, false>,
      dot_e5m2_e5m2_impl<false, true>,
      l2_score_e5m2_f32_impl<false>,
      l2_score_e5m2_e5m2_impl<false, false>,
      l2_score_e5m2_e5m2_impl<false, true>,
  };
#if defined(ASTRAL_X86_F16C)
  static constexpr E5m2Kernels kF16c = {
      dot_e5m2_f32_impl<true>,
      dot_e5m2_e5m2_impl<true, false>,
      dot_e5m2_e5m2_impl<true, true>,
      l2_score_e5m2_f32_impl<true>,
      l2_score_e5m2_e5m2_impl<true, false>,
      l2_score_e5m2_e5m2_impl<true, true>,
  };
  return platform::cpu_supports_avx2() && platform::cpu_supports_f16c() ? &kF16c : &kFallback;
#else
  return &kFallback;
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

float l2_score_i16_f32(const int16_t* a, float scale, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  const __m256 scale_v = _mm256_set1_ps(scale);
  uint32_t i = 0;
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256 av0 = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(bytes))), scale_v);
    const __m256 av1 = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bytes, 1))), scale_v);
    const __m256 d0 = _mm256_sub_ps(av0, _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(av1, _mm256_loadu_ps(b + i + kAvx2F32Lanes));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
#endif
  }
  float sum = reduce_avx2_f32(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) {
    const float d = static_cast<float>(a[i]) * scale - b[i];
    sum += d * d;
  }
  return -sum;
#elif defined(__aarch64__) && defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  const float32x4_t scale_v = vdupq_n_f32(scale);
  uint32_t i = 0;
  for (; i + kNeonI8HalfLanes <= dim; i += kNeonI8HalfLanes) {
    const int16x8_t av = vld1q_s16(a + i);
    const float32x4_t d0 =
        vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(av))), scale_v), vld1q_f32(b + i));
    const float32x4_t d1 =
        vsubq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(av))), scale_v),
                  vld1q_f32(b + i + kNeonF32Lanes));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
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

float l2_score_i16_i16(const int16_t* a, float a_scale, const int16_t* b, float b_scale,
                       uint32_t dim) {
#if defined(__AVX2__)
  __m256i acc_a = _mm256_setzero_si256();
  __m256i acc_b = _mm256_setzero_si256();
  __m256i acc_ab = _mm256_setzero_si256();
  uint32_t i = 0;
  for (; i + kAvx2I8Lanes <= dim; i += kAvx2I8Lanes) {
    const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    acc_a = _mm256_add_epi32(acc_a, _mm256_madd_epi16(av, av));
    acc_b = _mm256_add_epi32(acc_b, _mm256_madd_epi16(bv, bv));
    acc_ab = _mm256_add_epi32(acc_ab, _mm256_madd_epi16(av, bv));
  }
  int32_t sum_a = reduce_avx2_i32(acc_a);
  int32_t sum_b = reduce_avx2_i32(acc_b);
  int32_t sum_ab = reduce_avx2_i32(acc_ab);
  for (; i < dim; ++i) {
    const int32_t av = a[i];
    const int32_t bv = b[i];
    sum_a += av * av;
    sum_b += bv * bv;
    sum_ab += av * bv;
  }
  return -(static_cast<float>(sum_a) * a_scale * a_scale +
           static_cast<float>(sum_b) * b_scale * b_scale -
           static_cast<float>(sum_ab) * kL2CrossTermScale * a_scale * b_scale);
#elif defined(__aarch64__) && defined(__ARM_NEON)
  int32x4_t acc_a = vdupq_n_s32(0);
  int32x4_t acc_b = vdupq_n_s32(0);
  int32x4_t acc_ab = vdupq_n_s32(0);
  uint32_t i = 0;
  for (; i + kNeonI8Lanes <= dim; i += kNeonI8Lanes) {
    const int16x8_t av = vld1q_s16(a + i);
    const int16x8_t bv = vld1q_s16(b + i);
    const int16x4_t av_lo = vget_low_s16(av);
    const int16x4_t av_hi = vget_high_s16(av);
    const int16x4_t bv_lo = vget_low_s16(bv);
    const int16x4_t bv_hi = vget_high_s16(bv);
    acc_a = vmlal_s16(acc_a, av_lo, av_lo);
    acc_a = vmlal_s16(acc_a, av_hi, av_hi);
    acc_b = vmlal_s16(acc_b, bv_lo, bv_lo);
    acc_b = vmlal_s16(acc_b, bv_hi, bv_hi);
    acc_ab = vmlal_s16(acc_ab, av_lo, bv_lo);
    acc_ab = vmlal_s16(acc_ab, av_hi, bv_hi);
  }
  int32_t sum_a = vaddvq_s32(acc_a);
  int32_t sum_b = vaddvq_s32(acc_b);
  int32_t sum_ab = vaddvq_s32(acc_ab);
  for (; i < dim; ++i) {
    const int32_t av = a[i];
    const int32_t bv = b[i];
    sum_a += av * av;
    sum_b += bv * bv;
    sum_ab += av * bv;
  }
  return -(static_cast<float>(sum_a) * a_scale * a_scale +
           static_cast<float>(sum_b) * b_scale * b_scale -
           static_cast<float>(sum_ab) * kL2CrossTermScale * a_scale * b_scale);
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
  const float max_abs = max_abs_f32(src, dim);
  const float scale = max_abs > 0.0f ? max_abs / kQ8MaxFloat : 1.0f;
  const float inv_scale = 1.0f / scale;
  uint32_t i = 0;
#if defined(__AVX2__)
  const __m256 inv_scale_v = _mm256_set1_ps(inv_scale);
  const __m256 positive_bias = _mm256_set1_ps(0.5f);
  const __m256 negative_bias = _mm256_set1_ps(-0.5f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256i min_value = _mm256_set1_epi32(kQ8MinValue);
  const __m256i max_value = _mm256_set1_epi32(kQ8MaxValue);
  for (; i + kAvx2F32Lanes <= dim; i += kAvx2F32Lanes) {
    quantize_q8_store8_avx2(dst + i, src + i, inv_scale_v, positive_bias, negative_bias, zero,
                            min_value, max_value);
  }
#endif
  for (; i < dim; ++i) {
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
  return value < 0.0f ? static_cast<int8_t>(-static_cast<int32_t>(best))
                      : static_cast<int8_t>(best);
}

void quantize_e2m3_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim) {
  const float max_abs = max_abs_f32(src, dim);
  const float scale = max_abs > 0.0f ? max_abs / kE2M3MaxFloat : 1.0f;
  const float inv_scale = static_cast<float>(kE2M3Scale) / scale;
  for (uint32_t i = 0; i < dim; ++i) {
    dst[i] = round_scaled_e2m3(src[i] * inv_scale);
  }
  *out_scale = scale;
}

inline int16_t e3m2_nearest_scaled(float value) {
  static constexpr uint32_t kBandBase[kE3M2StepBandCount] = {8, 16, 32, 64, 128, 256};
  static constexpr float kBandHoldThreshold[kE3M2StepBandCount] = {9.0f,  18.0f,  36.0f,
                                                                   72.0f, 144.0f, 288.0f};
  static constexpr float kBandStepEndThreshold[kE3M2StepBandCount] = {15.0f,  30.0f,  60.0f,
                                                                      120.0f, 240.0f, 416.0f};
  static constexpr uint32_t kBandStep[kE3M2StepBandCount] = {2, 4, 8, 16, 32, 64};
  static constexpr float kBandInvStep[kE3M2StepBandCount] = {0.5f,    0.25f,    0.125f,
                                                             0.0625f, 0.03125f, 0.015625f};
  const float abs_value = std::fabs(value);
  uint32_t scaled = kE3M2MaxScaled;
  if (abs_value <= kE3M2UnitBandMaxThreshold) {
    scaled = abs_value <= kE3M2TieBias ? 0u : ceil_positive_to_u32(abs_value - kE3M2TieBias);
  } else {
    for (uint32_t band = 0; band < kE3M2StepBandCount; ++band) {
      if (abs_value <= kBandHoldThreshold[band]) {
        scaled = kBandBase[band];
        break;
      }
      if (abs_value <= kBandStepEndThreshold[band]) {
        scaled = kBandBase[band] +
                 kBandStep[band] * ceil_positive_to_u32((abs_value - kBandHoldThreshold[band]) *
                                                        kBandInvStep[band]);
        break;
      }
    }
  }
  return value < 0.0f ? static_cast<int16_t>(-static_cast<int16_t>(scaled))
                      : static_cast<int16_t>(scaled);
}


void quantize_e3m2_vector(int16_t* dst, float* out_scale, const float* src, uint32_t dim) {
  const float max_abs = max_abs_f32(src, dim);
  const float scale = max_abs > 0.0f ? max_abs / kE3M2MaxFloat : 1.0f;
  const float inv_scale = static_cast<float>(kE3M2Scale) / scale;
  for (uint32_t i = 0; i < dim; ++i) {
    dst[i] = e3m2_nearest_scaled(src[i] * inv_scale);
  }
  *out_scale = scale;
}

void quantize_e3m2_cosine_vector(int16_t* dst, float* out_scale, const float* src, uint32_t dim) {
  const float sq = dot_f32(src, src, dim);
  const float norm = sq > 0.0f ? std::sqrt(sq) : 0.0f;
  if (norm == 0.0f) {
    quantize_e3m2_vector(dst, out_scale, src, dim);
    return;
  }
  float normalized[kMaxDim];
  const float inv_norm = 1.0f / norm;
  scale_f32_vector(normalized, src, inv_norm, dim);
  quantize_e3m2_vector(dst, out_scale, normalized, dim);
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
  scale_f32_vector(normalized, src, inv_norm, dim);
  quantize_e2m3_vector(dst, out_scale, normalized, dim);
}

void quantize_e5m2_vector(int8_t* dst, float* out_scale, const float* src, uint32_t dim) {
  const float max_abs = max_abs_f32(src, dim);
  const float scale = max_abs > 0.0f ? max_abs / kE5M2MaxFloat : 1.0f;
  const float inv_scale = 1.0f / scale;
  uint32_t i = 0;
#if defined(__AVX2__)
  const __m256 inv_scale_v = _mm256_set1_ps(inv_scale);
  for (; i + kAvx2F32Lanes <= dim; i += kAvx2F32Lanes) {
    quantize_e5m2_store8_avx2(dst + i, src + i, inv_scale_v);
  }
#endif
  for (; i < dim; ++i) {
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
  scale_f32_vector(normalized, src, inv_norm, dim);
  quantize_e5m2_vector(dst, out_scale, normalized, dim);
}

inline float vector_norm(const float* v, uint32_t dim) {
  const float sq = dot_f32(v, v, dim);
  return sq > 0.0f ? std::sqrt(sq) : 0.0f;
}

float cosine_scale(const float* v, uint32_t dim) {
  const float norm = vector_norm(v, dim);
  return norm > 0.0f ? 1.0f / norm : 0.0f;
}

void normalize_f32_vector(float* dst, const float* src, uint32_t dim) {
  const float norm = vector_norm(src, dim);
  const float inv_norm = norm > 0.0f ? 1.0f / norm : 0.0f;
  scale_f32_vector(dst, src, inv_norm, dim);
}

float cosine_scale_q8(const int8_t* v, float scale, uint32_t dim) {
  const float sq = dot_q8_q8(v, v, dim) * scale * scale;
  return sq > 0.0f ? 1.0f / std::sqrt(sq) : 0.0f;
}


} // namespace astral::inference
