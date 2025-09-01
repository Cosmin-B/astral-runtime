#pragma once

#include <cstdint>
#include <ctime>

#if defined(__APPLE__)
  #include <mach/mach_time.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__GNUC__) || defined(__clang__)
    #include <x86intrin.h>
  #elif defined(_MSC_VER)
    #include <intrin.h>
    #include <emmintrin.h>
  #endif
#endif

namespace astral::bench {

enum class ClockKind : uint8_t {
    Tsc,
    MachAbsolute,
    MonotonicNs,
};

struct ClockInfo {
    ClockKind kind;
    const char* name;
    double tick_to_ns; // Multiply ticks delta by this to get ns (0 when unknown).
};

inline uint64_t ns_now() {
    timespec ts{};
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

inline ClockInfo clock_info() {
#if defined(__APPLE__)
    static ClockInfo info = []() {
        mach_timebase_info_data_t tb{};
        mach_timebase_info(&tb);
        const double tick_to_ns = (tb.denom != 0) ? (static_cast<double>(tb.numer) / static_cast<double>(tb.denom))
                                                  : 0.0;
        return ClockInfo{ClockKind::MachAbsolute, "mach_absolute_time", tick_to_ns};
    }();
    return info;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    return ClockInfo{ClockKind::Tsc, "rdtsc", 0.0};
#else
    return ClockInfo{ClockKind::MonotonicNs, "clock_gettime_ns", 1.0};
#endif
}

inline uint64_t ticks_now() {
#if defined(__APPLE__)
    return mach_absolute_time();
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__GNUC__) || defined(__clang__)
    _mm_lfence();
    const uint64_t t = static_cast<uint64_t>(__rdtsc());
    _mm_lfence();
    return t;
  #elif defined(_MSC_VER)
    _mm_lfence();
    const uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
  #else
    return ns_now();
  #endif
#else
    return ns_now();
#endif
}

} // namespace astral::bench

