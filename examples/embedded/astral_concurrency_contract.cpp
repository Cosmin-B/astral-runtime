#include "platform/atomics.h"
#include "concurrency/mpmc_queue.hpp"
#include "concurrency/spsc_ring.hpp"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

static void cpu_event_smoke(uint32_t iters) {
    std::atomic<uint32_t> seq{0};
    std::atomic<bool> start{false};

    std::thread waiter([&]() {
        while (!start.load(std::memory_order_acquire)) {
            astral::platform::cpu_pause();
        }

        uint32_t expected = 0;
        for (uint32_t i = 0; i < iters; ++i) {
            while (seq.load(std::memory_order_acquire) == expected) {
                astral::platform::cpu_wait_for_event();
            }
            expected += 1;
        }
    });

    std::thread signaler([&]() {
        start.store(true, std::memory_order_release);
        for (uint32_t i = 0; i < iters; ++i) {
            seq.store(i + 1, std::memory_order_release);
            astral::platform::cpu_signal_event();
        }
    });

    signaler.join();
    waiter.join();
}

static void spsc_stress(uint64_t items) {
    astral::concurrency::SpscRing<uint64_t, 4096> ring;

    std::atomic<bool> start{false};
    std::atomic<uint32_t> ready{0};

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
    start.store(true, std::memory_order_release);

    producer.join();
    consumer.join();

    const uint64_t expected = (items == 0) ? 0 : ((items - 1) * items) / 2;
    if (sum != expected) {
        std::fprintf(stderr, "SPSC checksum mismatch: got=%llu expected=%llu\n",
                     static_cast<unsigned long long>(sum),
                     static_cast<unsigned long long>(expected));
        std::abort();
    }
}

static void mpmc_stress(uint32_t producers, uint32_t consumers, uint32_t items_per_producer) {
    if (producers == 0) {
        producers = 1;
    }
    if (consumers == 0) {
        consumers = 1;
    }

    constexpr uint64_t kStop = UINT64_MAX;
    constexpr size_t kCapacity = 1024;
    astral::concurrency::MpmcQueue<uint64_t, kCapacity> q;

    std::atomic<bool> start{false};
    std::atomic<uint32_t> ready{0};

    std::vector<std::thread> prod;
    std::vector<std::thread> cons;
    prod.reserve(producers);
    cons.reserve(consumers);

    std::vector<uint64_t> sums(consumers, 0);
    std::atomic<uint32_t> produced{0};

    for (uint32_t p = 0; p < producers; ++p) {
        prod.emplace_back([&, p]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            const uint64_t base = static_cast<uint64_t>(p) * items_per_producer;
            for (uint32_t i = 0; i < items_per_producer; ++i) {
                q.enqueue_wait(base + i);
                produced.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (uint32_t c = 0; c < consumers; ++c) {
        cons.emplace_back([&, c]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            uint64_t local = 0;
            uint64_t v = 0;
            for (;;) {
                q.dequeue_wait(&v);
                if (v == kStop) {
                    break;
                }
                local += v;
            }
            sums[c] = local;
        });
    }

    while (ready.load(std::memory_order_acquire) != producers + consumers) {
        astral::platform::cpu_pause();
    }
    start.store(true, std::memory_order_release);

    for (auto& t : prod) {
        t.join();
    }

    while (!q.empty()) {
        astral::platform::cpu_pause();
    }

    for (uint32_t i = 0; i < consumers; ++i) {
        q.enqueue_wait(kStop);
    }

    for (auto& t : cons) {
        t.join();
    }

    if (produced.load(std::memory_order_relaxed) != producers * items_per_producer) {
        std::fprintf(stderr, "MPMC produced mismatch\n");
        std::abort();
    }

    // Expected sum of 0..items_per_producer-1 repeated for each producer, offset by base.
    const uint64_t per = (items_per_producer == 0) ? 0 : (static_cast<uint64_t>(items_per_producer - 1) * items_per_producer) / 2;
    uint64_t expected = 0;
    for (uint32_t p = 0; p < producers; ++p) {
        expected += per + static_cast<uint64_t>(p) * items_per_producer * items_per_producer;
    }

    uint64_t got = 0;
    for (uint64_t s : sums) {
        got += s;
    }
    if (got != expected) {
        std::fprintf(stderr, "MPMC checksum mismatch: got=%llu expected=%llu\n",
                     static_cast<unsigned long long>(got),
                     static_cast<unsigned long long>(expected));
        std::abort();
    }
}

} // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    cpu_event_smoke(20000);
    spsc_stress(50000);
    mpmc_stress(2, 2, 5000);

    std::fprintf(stderr, "OK\n");
    return 0;
}
