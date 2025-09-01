#pragma once

#include <cstdint>
#include <cstdio>

namespace astral::bench {

struct BenchResult {
    const char* name;
    uint64_t ticks;
    uint64_t ns;
    uint64_t ops;
};

inline void print_result(const BenchResult& r, const char* clock_name) {
    const double seconds = static_cast<double>(r.ns) * 1e-9;
    const double mops = (seconds > 0.0) ? (static_cast<double>(r.ops) / seconds) / 1e6 : 0.0;
    const double ns_per_op = (r.ops > 0) ? (static_cast<double>(r.ns) / static_cast<double>(r.ops)) : 0.0;
    const double ticks_per_op = (r.ops > 0) ? (static_cast<double>(r.ticks) / static_cast<double>(r.ops)) : 0.0;

    std::printf("%-28s  %8.2f Mops/s  %8.2f ns/op  %10.2f ticks/op  (%s)\n",
                r.name, mops, ns_per_op, ticks_per_op, clock_name);
}

} // namespace astral::bench

