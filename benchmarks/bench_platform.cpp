#include "bench_common.hpp"
#include "bench_clock.hpp"

#include "platform/atomics.h"
#include "platform/cpu_features.hpp"

#include <cstdint>
#include <cstdio>

namespace astral::bench {
namespace {

inline void do_not_optimize(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::"r"(v) : "memory");
#else
    (void)v;
#endif
}

} // namespace

BenchResult bench_ticks_now(uint64_t iters) {
    uint64_t acc = 0;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        acc += ticks_now();
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    do_not_optimize(acc);

    return BenchResult{
        "ticks_now",
        t1 - t0,
        n1 - n0,
        iters,
    };
}

BenchResult bench_cpu_pause(uint64_t iters) {
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        astral::platform::cpu_pause();
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    return BenchResult{
        "cpu_pause",
        t1 - t0,
        n1 - n0,
        iters,
    };
}

void bench_platform_print(uint64_t iters) {
    const astral::platform::CpuFeatures& features = astral::platform::cpu_features();
    std::printf("CPU dispatch: arch=%s tier=%s avx2=%u neon=%u\n",
                astral::platform::cpu_arch_name(features.arch),
                astral::platform::cpu_dispatch_tier_name(),
                features.x86_avx2 ? 1u : 0u,
                features.arm_neon ? 1u : 0u);

    const ClockInfo clk = clock_info();
    print_result(bench_ticks_now(iters), clk.name);
    print_result(bench_cpu_pause(iters), clk.name);
}

} // namespace astral::bench
