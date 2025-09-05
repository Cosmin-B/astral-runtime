#pragma once

#include <cstdint>

#if defined(_WIN32) || defined(_WIN64)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <time.h>
#endif

namespace astral::platform {

// Monotonic time in nanoseconds.
// Intended for timeouts/stats; not for wall-clock timestamps.
inline uint64_t monotonic_time_ns() {
#if defined(_WIN32) || defined(_WIN64)
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    if (!::QueryPerformanceFrequency(&freq) || freq.QuadPart <= 0) {
        // Fallback: milliseconds to nanoseconds (coarse).
        return static_cast<uint64_t>(::GetTickCount64()) * 1000000ull;
    }
    ::QueryPerformanceCounter(&counter);
    const uint64_t ticks = static_cast<uint64_t>(counter.QuadPart);
    const uint64_t hz = static_cast<uint64_t>(freq.QuadPart);
    // Convert to ns with integer math.
    return (ticks * 1000000000ull) / hz;
#else
    timespec ts{};
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
#endif
}

} // namespace astral::platform

