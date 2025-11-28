#include "bench_common.hpp"
#include "bench_clock.hpp"

#include "concurrency/mpmc_queue.hpp"
#include "concurrency/mpsc_ring.hpp"
#include "concurrency/spsc_fan_in.hpp"
#include "concurrency/spsc_ring.hpp"
#include "platform/atomics.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#if defined(__linux__)
  #include <pthread.h>
  #include <sched.h>
#endif

namespace astral::bench {

namespace {

inline void do_not_optimize(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::"r"(v) : "memory");
#else
    (void)v;
#endif
}

uint64_t percentile(const std::vector<uint64_t>& values, uint32_t pct) {
    if (values.empty()) {
        return 0;
    }
    const uint64_t scaled = (static_cast<uint64_t>(values.size() - 1) * pct) / 100;
    return values[static_cast<size_t>(scaled)];
}

bool pin_threads_enabled() {
    const char* v = std::getenv("ASTRAL_BENCH_PIN_THREADS");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

void pin_current_thread(uint32_t core_id) {
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    uint32_t seen = 0;
    for (uint32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        if (seen == core_id) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
            return;
        }
        ++seen;
    }
#else
    (void)core_id;
#endif
}

void maybe_pin_current_thread(uint32_t core_id) {
    if (pin_threads_enabled()) {
        pin_current_thread(core_id);
    }
}

void print_named_result(const char* name, BenchResult r, const char* clock_name) {
    r.name = name;
    print_result(r, clock_name);
}

void print_named_latency_result(const char* name, LatencyResult r, const char* clock_name) {
    r.name = name;
    print_latency_result(r, clock_name);
}

} // namespace

BenchResult bench_spsc_ring_local(uint64_t items) {
    constexpr size_t kCapacity = 4096;
    astral::concurrency::SpscRing<uint64_t, kCapacity> ring;

    uint64_t sum = 0;
    uint64_t v = 0;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    uint64_t remaining = items;
    while (remaining != 0) {
        size_t pushed = 0;
        while (pushed < kCapacity && remaining != 0) {
            const uint64_t value = items - remaining;
            if (ring.push(value)) {
                ++pushed;
                --remaining;
            }
        }
        for (size_t i = 0; i < pushed; ++i) {
            while (!ring.pop(v)) {
                astral::platform::cpu_pause();
            }
            sum += v;
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    do_not_optimize(sum);

    return BenchResult{
        "SPSC local cached",
        t1 - t0,
        n1 - n0,
        items * 2,
    };
}

BenchResult bench_spsc_ring_batch(uint64_t items, uint32_t batch_size) {
    constexpr size_t kCapacity = 4096;
    constexpr size_t kMaxBatch = 1024;
    astral::concurrency::SpscRing<uint64_t, kCapacity> ring;

    if (batch_size == 0) {
        batch_size = 1;
    }
    if (batch_size > kMaxBatch) {
        batch_size = kMaxBatch;
    }

    uint64_t input[kMaxBatch];
    uint64_t output[kMaxBatch];
    uint64_t sum = 0;
    uint64_t produced = 0;
    uint64_t consumed = 0;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    while (consumed < items) {
        while (produced < items) {
            uint32_t n = batch_size;
            const uint64_t remaining = items - produced;
            if (remaining < n) {
                n = static_cast<uint32_t>(remaining);
            }
            for (uint32_t i = 0; i < n; ++i) {
                input[i] = produced + i;
            }
            const size_t pushed = ring.push_batch(input, n);
            if (pushed == 0) {
                break;
            }
            produced += pushed;
        }

        const size_t popped = ring.pop_batch(output, batch_size);
        if (popped == 0) {
            astral::platform::cpu_pause();
            continue;
        }
        for (size_t i = 0; i < popped; ++i) {
            sum += output[i];
        }
        consumed += popped;
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    do_not_optimize(sum);

    return BenchResult{
        "SPSC local batch",
        t1 - t0,
        n1 - n0,
        items * 2,
    };
}

BenchResult bench_spsc_ring(uint64_t items) {
    constexpr size_t kCapacity = 4096;
    astral::concurrency::SpscRing<uint64_t, kCapacity> ring;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};

    uint64_t sum = 0;

    std::thread producer([&]() {
        maybe_pin_current_thread(0);
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
        maybe_pin_current_thread(1);
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

BenchResult bench_spsc_fan_in(uint32_t producers, uint64_t items_per_producer) {
    constexpr size_t kCapacity = 4096;
    constexpr uint32_t kMaxProducers = 16;

    if (producers == 0) {
        producers = 1;
    }
    if (producers > kMaxProducers) {
        producers = kMaxProducers;
    }

    astral::concurrency::SpscFanIn<uint64_t, kMaxProducers, kCapacity> fan_in;
    const uint64_t total_items = static_cast<uint64_t>(producers) * items_per_producer;
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<uint64_t> consumed{0};

    std::vector<std::thread> prod_threads;
    prod_threads.reserve(producers);

    for (uint32_t p = 0; p < producers; ++p) {
        prod_threads.emplace_back([&, p]() {
            maybe_pin_current_thread(p);
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            const uint64_t base = static_cast<uint64_t>(p) * items_per_producer;
            for (uint64_t i = 0; i < items_per_producer; ++i) {
                while (!fan_in.try_push(p, base + i)) {
                    astral::platform::cpu_pause();
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != producers) {
        astral::platform::cpu_pause();
    }

    uint64_t sum = 0;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    start.store(true, std::memory_order_release);

    uint64_t v = 0;
    while (consumed.load(std::memory_order_relaxed) < total_items) {
        if (fan_in.pop(v)) {
            sum += v;
            consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
            astral::platform::cpu_pause();
        }
    }

    for (auto& t : prod_threads) {
        t.join();
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    do_not_optimize(sum);

    return BenchResult{
        "SPSC fan-in lanes",
        t1 - t0,
        n1 - n0,
        total_items * 2,
    };
}

LatencyResult bench_spsc_latency(uint64_t samples) {
    constexpr size_t kCapacity = 4096;
    astral::concurrency::SpscRing<uint64_t, kCapacity> ring;
    std::vector<uint64_t> deltas;
    deltas.resize(static_cast<size_t>(samples));

    uint64_t timer_overhead = UINT64_MAX;
    for (uint32_t i = 0; i < 1024; ++i) {
        const uint64_t t0 = ticks_now();
        const uint64_t t1 = ticks_now();
        const uint64_t delta = t1 - t0;
        if (delta < timer_overhead) {
            timer_overhead = delta;
        }
    }

    uint64_t out = 0;
    for (uint64_t i = 0; i < samples; ++i) {
        const uint64_t t0 = ticks_now();
        (void)ring.push(i);
        (void)ring.pop(out);
        const uint64_t t1 = ticks_now();
        const uint64_t measured = t1 - t0;
        deltas[static_cast<size_t>(i)] = measured > timer_overhead ? measured - timer_overhead : 0;
    }
    do_not_optimize(out);

    std::sort(deltas.begin(), deltas.end());
    const ClockInfo clk = clock_info();
    return LatencyResult{
        "SPSC push+pop local",
        percentile(deltas, 50),
        percentile(deltas, 95),
        percentile(deltas, 99),
        deltas.empty() ? 0 : deltas.back(),
        clk.tick_to_ns,
    };
}

BenchResult bench_mpsc_ring(uint32_t producers, uint64_t items_per_producer) {
    constexpr size_t kCapacity = 4096;
    astral::concurrency::MpscRing<uint64_t, kCapacity> ring;

    if (producers == 0) {
        producers = 1;
    }

    const uint64_t total_items = static_cast<uint64_t>(producers) * items_per_producer;
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<uint64_t> consumed{0};

    std::vector<std::thread> prod_threads;
    prod_threads.reserve(producers);

    for (uint32_t p = 0; p < producers; ++p) {
        prod_threads.emplace_back([&, p]() {
            maybe_pin_current_thread(p);
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            const uint64_t base = static_cast<uint64_t>(p) * items_per_producer;
            for (uint64_t i = 0; i < items_per_producer; ++i) {
                while (!ring.try_push(base + i)) {
                    astral::platform::cpu_pause();
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != producers) {
        astral::platform::cpu_pause();
    }

    uint64_t sum = 0;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    start.store(true, std::memory_order_release);

    uint64_t v = 0;
    while (consumed.load(std::memory_order_relaxed) < total_items) {
        if (ring.pop(v)) {
            sum += v;
            consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
            astral::platform::cpu_pause();
        }
    }

    for (auto& t : prod_threads) {
        t.join();
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    do_not_optimize(sum);

    return BenchResult{
        "MPSC ring (4096)",
        t1 - t0,
        n1 - n0,
        total_items * 2, // push + pop
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
            maybe_pin_current_thread(p);
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
            maybe_pin_current_thread(producers + c);
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

BenchResult bench_mpmc_queue_dense(uint32_t producers, uint32_t consumers, uint64_t items_per_producer) {
    constexpr size_t kCapacity = 64;
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
            maybe_pin_current_thread(p);
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
            maybe_pin_current_thread(producers + c);
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

    while (!queue.empty()) {
        astral::platform::cpu_pause();
    }

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
        "MPMC dense (64)",
        t1 - t0,
        n1 - n0,
        total_items * 2,
    };
}

void bench_concurrency_matrix_print(uint64_t items_per_producer) {
    const ClockInfo clk = clock_info();

    std::printf("\nSPSC coverage\n");
    print_named_result("SPSC batch", bench_spsc_ring_batch(items_per_producer, 64), clk.name);
    print_named_result("SPSC cached local", bench_spsc_ring_local(items_per_producer), clk.name);
    print_named_latency_result("SPSC local pcts", bench_spsc_latency(100000), clk.name);
    print_named_result("SPSC 1P/1C transit", bench_spsc_ring(items_per_producer), clk.name);

    std::printf("\nMPSC coverage\n");
    print_named_result("MPSC 1P/1C", bench_mpsc_ring(1, items_per_producer), clk.name);
    print_named_result("SPSC fan-in 4P/1C", bench_spsc_fan_in(4, items_per_producer / 4), clk.name);
    print_named_result("MPSC 2P/1C", bench_mpsc_ring(2, items_per_producer / 2), clk.name);
    print_named_result("MPSC 4P/1C", bench_mpsc_ring(4, items_per_producer / 4), clk.name);
    print_named_result("MPSC 8P/1C", bench_mpsc_ring(8, items_per_producer / 8), clk.name);

    std::printf("\nMPMC coverage\n");
    print_named_result("MPMC 1P/1C", bench_mpmc_queue(1, 1, items_per_producer), clk.name);
    print_named_result("MPMC 4P/4C spaced", bench_mpmc_queue(4, 4, items_per_producer / 4), clk.name);
    print_named_result("MPMC 4P/4C dense", bench_mpmc_queue_dense(4, 4, items_per_producer / 4), clk.name);
    print_named_result("MPMC 8P/8C", bench_mpmc_queue(8, 8, items_per_producer / 8), clk.name);
    std::printf("%-28s  %s\n", "MPMC ticket-batched", "not implemented");
}

} // namespace astral::bench
