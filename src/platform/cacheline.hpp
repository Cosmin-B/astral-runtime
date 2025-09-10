#pragma once

#include <cstddef>

// Compile-time cache line alignment used for `alignas(...)`.
//
// Notes:
// - Runtime detection (`cache_line_size()`) cannot be used with `alignas`.
// - Many ARM64 systems (notably Apple Silicon) use 128B L1D cache lines; defaulting to 64B
//   can increase false sharing for hot atomics.
// - Users can override this via a compile definition: `-DASTRAL_CACHELINE_ALIGN=<bytes>`.
#ifndef ASTRAL_CACHELINE_ALIGN
  #if defined(__aarch64__) || defined(_M_ARM64)
    #define ASTRAL_CACHELINE_ALIGN 128
  #else
    #define ASTRAL_CACHELINE_ALIGN 64
  #endif
#endif

namespace astral::platform {

inline constexpr size_t kCacheLineAlign = static_cast<size_t>(ASTRAL_CACHELINE_ALIGN);

} // namespace astral::platform

