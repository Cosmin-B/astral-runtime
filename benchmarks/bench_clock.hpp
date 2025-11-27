#pragma once

#include "platform/time.h"

#include <cstdint>

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
    return astral::platform::monotonic_time_ns();
}

inline ClockInfo clock_info() {
    const astral::platform::TickClock clock = astral::platform::tick_clock();
    switch (clock.kind) {
        case astral::platform::TickClock::Kind::Tsc:
            return ClockInfo{ClockKind::Tsc, "rdtsc", clock.tick_to_ns};
        case astral::platform::TickClock::Kind::MachAbsolute:
            return ClockInfo{ClockKind::MachAbsolute, "mach_absolute_time", clock.tick_to_ns};
        case astral::platform::TickClock::Kind::ArmVct:
            return ClockInfo{ClockKind::MonotonicNs, "arm_cntvct", clock.tick_to_ns};
        case astral::platform::TickClock::Kind::MonotonicNs:
        default:
            return ClockInfo{ClockKind::MonotonicNs, "monotonic_ns", clock.tick_to_ns};
    }
}

inline uint64_t ticks_now() {
    return astral::platform::ticks_now();
}

} // namespace astral::bench
