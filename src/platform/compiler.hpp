#pragma once

// Private compiler and branch-hint helpers for Astral internals.
//
// Keep this header small and dependency-free. It is used by hot-path private
// primitives, but it is not part of the stable C ABI.

#if defined(_MSC_VER)
  #define ASTRAL_FORCE_INLINE __forceinline
  #define ASTRAL_NOINLINE __declspec(noinline)
  #define ASTRAL_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define ASTRAL_FORCE_INLINE inline __attribute__((always_inline))
  #define ASTRAL_NOINLINE __attribute__((noinline))
  #define ASTRAL_RESTRICT __restrict__
#else
  #define ASTRAL_FORCE_INLINE inline
  #define ASTRAL_NOINLINE
  #define ASTRAL_RESTRICT
#endif

#if defined(__has_cpp_attribute)
  #if __cplusplus >= 202002L && __has_cpp_attribute(likely)
    #define ASTRAL_LIKELY [[likely]]
  #else
    #define ASTRAL_LIKELY
  #endif
  #if __cplusplus >= 202002L && __has_cpp_attribute(unlikely)
    #define ASTRAL_UNLIKELY [[unlikely]]
  #else
    #define ASTRAL_UNLIKELY
  #endif
#else
  #define ASTRAL_LIKELY
  #define ASTRAL_UNLIKELY
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define ASTRAL_PREDICT_TRUE(expr) (__builtin_expect(!!(expr), 1))
  #define ASTRAL_PREDICT_FALSE(expr) (__builtin_expect(!!(expr), 0))
#else
  #define ASTRAL_PREDICT_TRUE(expr) (expr)
  #define ASTRAL_PREDICT_FALSE(expr) (expr)
#endif
