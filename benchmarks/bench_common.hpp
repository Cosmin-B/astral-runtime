#pragma once

#include <cstdint>
#include <cstdio>

namespace astral::bench {

struct BenchResult {
    const char* name;
    uint64_t ticks;
    uint64_t ns;
    uint64_t ops;
    const char* extra_label;
    double extra_value;
};

struct LatencyResult {
    const char* name;
    uint64_t p50_ticks;
    uint64_t p95_ticks;
    uint64_t p99_ticks;
    uint64_t max_ticks;
    double tick_to_ns;
};

inline void print_result(const BenchResult& r, const char* clock_name) {
    const double seconds = static_cast<double>(r.ns) * 1e-9;
    const double mops = (seconds > 0.0) ? (static_cast<double>(r.ops) / seconds) / 1e6 : 0.0;
    const double ns_per_op = (r.ops > 0) ? (static_cast<double>(r.ns) / static_cast<double>(r.ops)) : 0.0;
    const double ticks_per_op = (r.ops > 0) ? (static_cast<double>(r.ticks) / static_cast<double>(r.ops)) : 0.0;

    std::printf("%-28s  %8.2f Mops/s  %8.2f ns/op  %10.2f ticks/op  (%s)",
                r.name, mops, ns_per_op, ticks_per_op, clock_name);
    if (r.extra_label != nullptr) {
        std::printf("  %s=%8.2f", r.extra_label, r.extra_value);
    }
    std::printf("\n");
}

inline void print_latency_result(const LatencyResult& r, const char* clock_name) {
    const double s = r.tick_to_ns;
    std::printf("%-28s  p50=%8.2f ns  p95=%8.2f ns  p99=%8.2f ns  max=%8.2f ns  (%s)\n",
                r.name,
                static_cast<double>(r.p50_ticks) * s,
                static_cast<double>(r.p95_ticks) * s,
                static_cast<double>(r.p99_ticks) * s,
                static_cast<double>(r.max_ticks) * s,
                clock_name);
}

} // namespace astral::bench
