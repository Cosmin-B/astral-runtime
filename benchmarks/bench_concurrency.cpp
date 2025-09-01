#include "bench_common.hpp"
#include "bench_clock.hpp"

#include "concurrency/mpmc_queue.hpp"
#include "concurrency/spsc_ring.hpp"
#include "platform/atomics.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

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

BenchResult bench_spsc_ring(uint64_t items) {
    constexpr size_t kCapacity = 4096;
    astral::concurrency::SpscRing<uint64_t, kCapacity> ring;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};

    uint64_t sum = 0;

    std::thread producer([&]() {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
            astral::platform::cpu_pause();
        }

        for (uint64_t i = 0; i < items; ++i) {
            while (!ring.push(i)) {
                astral::platform::cpu_wait_for_event();
            }
        }
    });

    std::thread consumer([&]() {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
            astral::platform::cpu_pause();
        }

        uint64_t v = 0;
        for (uint64_t i = 0; i < items; ++i) {
            while (!ring.pop(&v)) {
                astral::platform::cpu_wait_for_event();
            }
            sum += v;
        }
    });

    while (ready.load(std::memory_order_acquire) != 2) {
        astral::platform::cpu_pause();
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    start.store(true, std::memory_order_release);

    producer.join();
    consumer.join();

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    do_not_optimize(sum);

    return BenchResult{
        "SPSC ring (4096)",
        t1 - t0,
        n1 - n0,
        items * 2, // push + pop
    };
}

BenchResult bench_mpmc_queue(uint32_t producers, uint32_t consumers, uint64_t items_per_producer) {
    constexpr size_t kCapacity = 16384;
    astral::concurrency::MpmcQueue<uint64_t, kCapacity> queue;

    if (producers == 0) {
        producers = 1;
    }
    if (consumers == 0) {
        consumers = 1;
    }

    const uint64_t total_items = static_cast<uint64_t>(producers) * items_per_producer;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};

    constexpr uint64_t kStop = UINT64_MAX;

    std::vector<std::thread> prod_threads;
    std::vector<std::thread> cons_threads;
    prod_threads.reserve(producers);
    cons_threads.reserve(consumers);

    std::vector<uint64_t> consumer_sums(consumers, 0);

    for (uint32_t p = 0; p < producers; ++p) {
        prod_threads.emplace_back([&, p]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            const uint64_t base = static_cast<uint64_t>(p) * items_per_producer;
            for (uint64_t i = 0; i < items_per_producer; ++i) {
                queue.enqueue_wait(base + i);
            }
        });
    }

    for (uint32_t c = 0; c < consumers; ++c) {
        cons_threads.emplace_back([&, c]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            uint64_t local = 0;
            uint64_t v = 0;
            for (;;) {
                queue.dequeue_wait(&v);
                if (v == kStop) {
                    break;
                }
                local += v;
            }
            consumer_sums[c] = local;
        });
    }

    while (ready.load(std::memory_order_acquire) != producers + consumers) {
        astral::platform::cpu_pause();
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    start.store(true, std::memory_order_release);

    for (auto& t : prod_threads) {
        t.join();
    }

    // Wait for the queue to drain (no more producers at this point).
    while (!queue.empty()) {
        astral::platform::cpu_pause();
    }

    // Wake consumers to exit.
    for (uint32_t i = 0; i < consumers; ++i) {
        queue.enqueue_wait(kStop);
    }

    for (auto& t : cons_threads) {
        t.join();
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    uint64_t checksum = 0;
    for (uint64_t s : consumer_sums) {
        checksum += s;
    }
    do_not_optimize(checksum);

    return BenchResult{
        "MPMC queue (16384)",
        t1 - t0,
        n1 - n0,
        total_items * 2, // enqueue + dequeue (stop tokens excluded)
    };
}

} // namespace astral::bench
