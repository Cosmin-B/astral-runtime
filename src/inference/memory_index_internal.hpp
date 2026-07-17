#pragma once

#include "memory_index.hpp"

#include <cstddef>
#include <cstdint>

namespace astral::inference {

inline constexpr uint32_t kMaxDim = 8192;
inline constexpr float kE2M3InvScale = 1.0f / 16.0f;
inline constexpr float kE3M2InvScale = 1.0f / 16.0f;
inline constexpr float kWorstScore = -3.4028234663852886e38f;

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

} // namespace astral::inference
