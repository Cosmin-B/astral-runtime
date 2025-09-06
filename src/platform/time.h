#pragma once

#include <cstdint>

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

#if defined(_WIN32) || defined(_WIN64)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <time.h>
#endif

namespace astral::platform {

struct TickClock {
    enum class Kind : uint8_t {
        Tsc,
        MachAbsolute,
        ArmVct,
        MonotonicNs,
    };

    Kind kind;
    double tick_to_ns; // Multiply tick deltas by this to get ns.
};

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
    return monotonic_time_ns();
  #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  #if defined(__GNUC__) || defined(__clang__)
    uint64_t t = 0;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(t));
    return t;
  #elif defined(_MSC_VER)
    return monotonic_time_ns();
  #else
    return monotonic_time_ns();
  #endif
#else
    return monotonic_time_ns();
#endif
}

inline TickClock tick_clock() {
#if defined(__APPLE__)
    static TickClock c = []() {
        mach_timebase_info_data_t tb{};
        mach_timebase_info(&tb);
        const double tick_to_ns =
            (tb.denom != 0) ? (static_cast<double>(tb.numer) / static_cast<double>(tb.denom)) : 1.0;
        return TickClock{TickClock::Kind::MachAbsolute, tick_to_ns};
    }();
    return c;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static TickClock c = []() {
        // Calibrate TSC -> ns using the monotonic clock at init time.
        // This avoids clock syscalls in hot/wait paths while keeping timeouts meaningful.
        const uint64_t start_ns = monotonic_time_ns();
        const uint64_t start_t = ticks_now();
        uint64_t now_ns = start_ns;
        // Busy-wait until we have a reasonable interval.
        while (now_ns - start_ns < 25'000'000ull) { // 25ms
            now_ns = monotonic_time_ns();
        }
        const uint64_t end_ns = now_ns;
        const uint64_t end_t = ticks_now();

        const uint64_t dt_t = end_t - start_t;
        const uint64_t dt_ns = end_ns - start_ns;
        const double tick_to_ns = (dt_t != 0 && dt_ns != 0) ? (static_cast<double>(dt_ns) / static_cast<double>(dt_t))
                                                            : 1.0;
        return TickClock{TickClock::Kind::Tsc, tick_to_ns};
    }();
    return c;
#elif defined(__aarch64__) || defined(_M_ARM64)
    static TickClock c = []() {
  #if defined(__GNUC__) || defined(__clang__)
        uint64_t freq = 0;
        __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(freq));
        const double tick_to_ns = (freq > 0) ? (1000000000.0 / static_cast<double>(freq)) : 1.0;
        return TickClock{TickClock::Kind::ArmVct, tick_to_ns};
  #else
        return TickClock{TickClock::Kind::MonotonicNs, 1.0};
  #endif
    }();
    return c;
#else
    return TickClock{TickClock::Kind::MonotonicNs, 1.0};
#endif
}

inline uint64_t ticks_to_ns(uint64_t dt_ticks) {
    const TickClock c = tick_clock();
    return static_cast<uint64_t>(static_cast<double>(dt_ticks) * c.tick_to_ns);
}

inline uint64_t ticks_from_ns(uint64_t ns) {
    const TickClock c = tick_clock();
    const double ns_to_tick = (c.tick_to_ns > 0.0) ? (1.0 / c.tick_to_ns) : 1.0;
    return static_cast<uint64_t>(static_cast<double>(ns) * ns_to_tick);
}

} // namespace astral::platform
