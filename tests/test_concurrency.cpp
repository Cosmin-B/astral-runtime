/**
 * test_concurrency.cpp - Concurrency primitives tests
 *
 * Tests for MPMC queue, SPSC ring, and epoch manager.
 * Validates: enqueue/dequeue, push/pop, full/empty, thread-safety, memory ordering.
 */

#include "test_framework.hpp"
#include "../src/concurrency/mpmc_queue.hpp"
#include "../src/concurrency/mpsc_ring.hpp"
#include "../src/concurrency/spsc_ring.hpp"
#include "../src/platform/atomics.h"

#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <chrono>

using namespace astral::concurrency;

// Simple test data structure
struct TestData {
    uint64_t value;
    uint32_t thread_id;
    uint32_t sequence;
};

static constexpr uint64_t kStopValue = UINT64_MAX;
static constexpr size_t kSpscBackpressureCapacity = 4;
static constexpr uint32_t kSpscBackpressureTotal = 512;

struct SpscBackpressureProducerContext {
    SpscRing<TestData, kSpscBackpressureCapacity>* ring;
    std::atomic<uint32_t>* blocked_pushes;
};

static void run_spsc_backpressure_producer(SpscBackpressureProducerContext* ctx) {
    for (uint32_t i = static_cast<uint32_t>(kSpscBackpressureCapacity);
         i < kSpscBackpressureTotal;
         ++i) {
        TestData data = {i, 0, i};
        while (!ctx->ring->push(data)) {
            ctx->blocked_pushes->fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    }
}

//
// MPMC Queue Tests
//

TEST(mpmc_enqueue_dequeue_basic) {
    MpmcQueue<TestData, 16> queue;

    TestData data = {42, 0, 0};
    queue.enqueue_wait(data);

    TestData out;
    queue.dequeue_wait(&out);
    ASSERT_EQ(out.value, 42);
}

TEST(mpmc_queue_full) {
    constexpr size_t kCapacity = 8;
    MpmcQueue<TestData, kCapacity> queue;

    // Fill queue
    for (size_t i = 0; i < kCapacity; ++i) {
        TestData data = {i, 0, 0};
        queue.enqueue_wait(data);
    }

    // Next enqueue should block until a dequeue happens.
    std::atomic<bool> enqueued{false};
    std::thread producer([&]() {
        TestData data = {99, 0, 0};
        queue.enqueue_wait(data);
        enqueued.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ASSERT_FALSE(enqueued.load(std::memory_order_acquire));

    // Free one slot
    TestData out;
    queue.dequeue_wait(&out);

    producer.join();
    ASSERT_TRUE(enqueued.load(std::memory_order_acquire));
}

TEST(mpmc_queue_empty) {
    MpmcQueue<TestData, 16> queue;

    std::atomic<bool> dequeued{false};
    TestData out{};

    std::thread consumer([&]() {
        queue.dequeue_wait(&out);
        dequeued.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ASSERT_FALSE(dequeued.load(std::memory_order_acquire));

    TestData data = {123, 0, 0};
    queue.enqueue_wait(data);

    consumer.join();
    ASSERT_TRUE(dequeued.load(std::memory_order_acquire));
    ASSERT_EQ(out.value, 123);
}

TEST(mpmc_multi_producer) {
    // Capacity must hold all produced items because this test has no concurrent consumer.
    constexpr size_t kCapacity = 4096;
    constexpr size_t kThreads = 4;
    constexpr size_t kItemsPerThread = 1000;

    MpmcQueue<TestData, kCapacity> queue;
    std::atomic<size_t> successful_enqueues{0};

    auto producer = [&queue, &successful_enqueues](uint32_t thread_id) {
        for (size_t i = 0; i < kItemsPerThread; ++i) {
            TestData data = {i, thread_id, static_cast<uint32_t>(i)};
            queue.enqueue_wait(data);
            successful_enqueues.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < kThreads; ++i) {
        threads.emplace_back(producer, static_cast<uint32_t>(i));
    }

    for (auto& t : threads) {
        t.join();
    }

    ASSERT_EQ(successful_enqueues.load(), kThreads * kItemsPerThread);

    // Drain queue
    for (size_t i = 0; i < kThreads * kItemsPerThread; ++i) {
        TestData out;
        queue.dequeue_wait(&out);
    }
}

TEST(mpmc_multi_consumer) {
    // Capacity must hold all pre-filled items because all enqueues happen up-front.
    constexpr size_t kCapacity = 4096;
    constexpr size_t kThreads = 4;
    constexpr size_t kTotalItems = 4000;

    MpmcQueue<TestData, kCapacity> queue;

    // Pre-fill queue
    for (size_t i = 0; i < kTotalItems; ++i) {
        TestData data = {i, 0, 0};
        queue.enqueue_wait(data);
    }

    std::atomic<size_t> items_consumed{0};
    std::atomic<size_t> next_item{0};

    auto consumer = [&queue, &items_consumed, &next_item]() {
        for (;;) {
            const size_t idx = next_item.fetch_add(1, std::memory_order_relaxed);
            if (idx >= kTotalItems) {
                break;
            }
            TestData out;
            queue.dequeue_wait(&out);
            items_consumed.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < kThreads; ++i) {
        threads.emplace_back(consumer);
    }

    for (auto& t : threads) {
        t.join();
    }

    ASSERT_EQ(items_consumed.load(), kTotalItems);
}

TEST(mpmc_producer_consumer_concurrent) {
    constexpr size_t kCapacity = 256;
    constexpr size_t kProducers = 2;
    constexpr size_t kConsumers = 2;
    constexpr size_t kItemsPerProducer = 5000;

    MpmcQueue<TestData, kCapacity> queue;
    std::atomic<size_t> items_produced{0};
    std::atomic<size_t> items_consumed{0};

    auto producer = [&](uint32_t thread_id) {
        for (size_t i = 0; i < kItemsPerProducer; ++i) {
            TestData data = {i, thread_id, static_cast<uint32_t>(i)};
            queue.enqueue_wait(data);
            items_produced.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto consumer = [&]() {
        for (;;) {
            TestData out;
            queue.dequeue_wait(&out);
            if (out.value == kStopValue) {
                break;
            }
            items_consumed.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < kProducers; ++i) {
        threads.emplace_back(producer, static_cast<uint32_t>(i));
    }
    for (size_t i = 0; i < kConsumers; ++i) {
        threads.emplace_back(consumer);
    }

    // Wait for producers
    for (size_t i = 0; i < kProducers; ++i) {
        threads[i].join();
    }

    // Stop consumers (one sentinel per consumer)
    for (size_t i = 0; i < kConsumers; ++i) {
        TestData stop = {kStopValue, 0, 0};
        queue.enqueue_wait(stop);
    }

    // Wait for consumers
    for (size_t i = kProducers; i < threads.size(); ++i) {
        threads[i].join();
    }

    ASSERT_EQ(items_produced.load(), kProducers * kItemsPerProducer);
    ASSERT_EQ(items_consumed.load(), kProducers * kItemsPerProducer);
}

TEST(mpmc_stress_checksum) {
    constexpr size_t kCapacity = 1024;
    constexpr size_t kProducers = 4;
    constexpr size_t kConsumers = 4;
    constexpr uint64_t kItemsPerProducer = 20000;

    MpmcQueue<uint64_t, kCapacity> queue;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};

    constexpr uint64_t kStop = UINT64_MAX;

    std::vector<std::thread> prod_threads;
    std::vector<std::thread> cons_threads;
    prod_threads.reserve(kProducers);
    cons_threads.reserve(kConsumers);

    std::vector<uint64_t> consumer_sums(kConsumers, 0);

    for (size_t p = 0; p < kProducers; ++p) {
        prod_threads.emplace_back([&, p]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            const uint64_t base = static_cast<uint64_t>(p) * kItemsPerProducer;
            for (uint64_t i = 0; i < kItemsPerProducer; ++i) {
                queue.enqueue_wait(base + i);
            }
        });
    }

    for (size_t c = 0; c < kConsumers; ++c) {
        cons_threads.emplace_back([&, c]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            uint64_t local = 0;
            for (;;) {
                uint64_t v = 0;
                queue.dequeue_wait(&v);
                if (v == kStop) {
                    break;
                }
                local += v;
            }
            consumer_sums[c] = local;
        });
    }

    while (ready.load(std::memory_order_acquire) != (kProducers + kConsumers)) {
        std::this_thread::yield();
    }

    start.store(true, std::memory_order_release);

    for (auto& t : prod_threads) {
        t.join();
    }

    // Wait for the queue to drain (no more producers at this point).
    while (!queue.empty()) {
        std::this_thread::yield();
    }

    // Wake consumers to exit.
    for (size_t i = 0; i < kConsumers; ++i) {
        queue.enqueue_wait(kStop);
    }

    for (auto& t : cons_threads) {
        t.join();
    }

    uint64_t sum = 0;
    for (uint64_t s : consumer_sums) {
        sum += s;
    }

    // Expected sum of 0..(N-1) where N = kProducers*kItemsPerProducer.
    const uint64_t n = static_cast<uint64_t>(kProducers) * kItemsPerProducer;
    const uint64_t expected = (n - 1) * n / 2;
    ASSERT_EQ(sum, expected);
}

//
// MPSC Ring Tests
//

TEST(mpsc_try_push_pop_basic) {
    MpscRing<TestData, 8> ring;

    TestData data = {42, 7, 1};
    ASSERT_TRUE(ring.try_push(data));

    TestData out{};
    ASSERT_TRUE(ring.pop(out));
    ASSERT_EQ(out.value, 42);
    ASSERT_EQ(out.thread_id, 7u);
    ASSERT_EQ(out.sequence, 1u);
    ASSERT_TRUE(ring.empty());
}

TEST(mpsc_full_returns_busy_signal) {
    MpscRing<TestData, 4> ring;

    for (uint32_t i = 0; i < 4; ++i) {
        TestData data = {i, 0, i};
        ASSERT_TRUE(ring.try_push(data));
    }

    TestData overflow = {99, 0, 99};
    ASSERT_FALSE(ring.try_push(overflow));

    TestData out{};
    ASSERT_TRUE(ring.pop(out));
    ASSERT_EQ(out.value, 0u);
    ASSERT_TRUE(ring.try_push(overflow));
}

TEST(mpsc_multi_producer_single_consumer_checksum) {
    constexpr size_t kCapacity = 256;
    constexpr uint32_t kProducers = 4;
    constexpr uint64_t kItemsPerProducer = 4000;
    constexpr uint64_t kTotalItems = static_cast<uint64_t>(kProducers) * kItemsPerProducer;

    MpscRing<uint64_t, kCapacity> ring;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<uint64_t> produced{0};
    std::atomic<uint64_t> consumed{0};

    uint64_t sum = 0;
    std::vector<std::thread> producers;
    producers.reserve(kProducers);

    for (uint32_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            const uint64_t base = static_cast<uint64_t>(p) * kItemsPerProducer;
            for (uint64_t i = 0; i < kItemsPerProducer; ++i) {
                const uint64_t value = base + i;
                while (!ring.try_push(value)) {
                    astral::platform::cpu_pause();
                }
                produced.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kProducers) {
        astral::platform::cpu_pause();
    }
    start.store(true, std::memory_order_release);

    while (consumed.load(std::memory_order_relaxed) < kTotalItems) {
        uint64_t value = 0;
        if (ring.pop(value)) {
            sum += value;
            consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
            astral::platform::cpu_pause();
        }
    }

    for (auto& t : producers) {
        t.join();
    }

    const uint64_t expected = (kTotalItems - 1) * kTotalItems / 2;
    ASSERT_EQ(produced.load(), kTotalItems);
    ASSERT_EQ(consumed.load(), kTotalItems);
    ASSERT_EQ(sum, expected);
    ASSERT_TRUE(ring.empty());
}

//
// SPSC Ring Tests
//

TEST(spsc_push_pop_basic) {
    SpscRing<TestData, 16> ring;

    TestData data = {123, 0, 0};
    bool ok = ring.push(data);
    ASSERT_TRUE(ok);

    TestData out;
    ok = ring.pop(&out);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out.value, 123);
}

TEST(spsc_pop_reference_basic) {
    SpscRing<TestData, 16> ring;

    TestData data = {777, 0, 1};
    ASSERT_TRUE(ring.push(data));

    TestData out{};
    ASSERT_TRUE(ring.pop(out));
    ASSERT_EQ(out.value, 777);
    ASSERT_TRUE(ring.empty());
}

TEST(spsc_ring_full) {
    constexpr size_t kCapacity = 8;
    SpscRing<TestData, kCapacity> ring;

    // Fill ring
    for (size_t i = 0; i < kCapacity; ++i) {
        TestData data = {i, 0, 0};
        bool ok = ring.push(data);
        if (!ok) break; // Full
    }

    // Should be full now
    TestData data = {99, 0, 0};
    bool ok = ring.push(data);
    ASSERT_FALSE(ok);
}

TEST(spsc_backpressure_wraparound) {
    SpscRing<TestData, kSpscBackpressureCapacity> ring;

    for (uint32_t i = 0; i < kSpscBackpressureCapacity; ++i) {
        TestData data = {i, 0, i};
        ASSERT_TRUE(ring.push(data));
    }

    std::atomic<uint32_t> blocked_pushes{0};
    SpscBackpressureProducerContext ctx{&ring, &blocked_pushes};
    std::thread producer(run_spsc_backpressure_producer, &ctx);

    for (uint32_t attempt = 0; attempt < 1000; ++attempt) {
        if (blocked_pushes.load(std::memory_order_relaxed) > 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    ASSERT_TRUE(blocked_pushes.load(std::memory_order_relaxed) > 0);

    TestData out{};
    for (uint32_t expected = 0; expected < kSpscBackpressureTotal; ++expected) {
        while (!ring.pop(&out)) {
            std::this_thread::yield();
        }
        ASSERT_EQ(out.value, expected);
        ASSERT_EQ(out.sequence, expected);
    }

    producer.join();
    ASSERT_TRUE(ring.empty());
}

TEST(spsc_ring_empty) {
    SpscRing<TestData, 16> ring;

    TestData out;
    bool ok = ring.pop(&out);
    ASSERT_FALSE(ok);
}

TEST(spsc_producer_consumer_pattern) {
    constexpr size_t kCapacity = 128;
    constexpr size_t kItemCount = 10000;

    SpscRing<TestData, kCapacity> ring;
    std::atomic<size_t> items_consumed{0};

    auto producer = [&ring]() {
        for (size_t i = 0; i < kItemCount; ++i) {
            TestData data = {i, 0, static_cast<uint32_t>(i)};
            while (!ring.push(data)) {
                // Spin wait
            }
        }
    };

    auto consumer = [&ring, &items_consumed]() {
        TestData out;
        size_t count = 0;
        while (count < kItemCount) {
            if (ring.pop(&out)) {
                ASSERT_EQ(out.value, count);
                ASSERT_EQ(out.sequence, count);
                ++count;
                items_consumed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::thread producer_thread(producer);
    std::thread consumer_thread(consumer);

    producer_thread.join();
    consumer_thread.join();

    ASSERT_EQ(items_consumed.load(), kItemCount);
}

TEST(spsc_pop_nullptr) {
    SpscRing<TestData, 16> ring;

    // pop with nullptr should return false
    bool ok = ring.pop(nullptr);
    ASSERT_FALSE(ok);
}

TEST(spsc_ordering_preservation) {
    constexpr size_t kCapacity = 64;
    constexpr size_t kItemCount = 1000;

    SpscRing<TestData, kCapacity> ring;

    auto producer = [&ring]() {
        for (size_t i = 0; i < kItemCount; ++i) {
            TestData data = {i * 2, 0, static_cast<uint32_t>(i)};
            while (!ring.push(data)) {}
        }
    };

    auto consumer = [&ring]() {
        TestData out;
        for (size_t i = 0; i < kItemCount; ++i) {
            while (!ring.pop(&out)) {}
            // Verify ordering preserved
            ASSERT_EQ(out.sequence, i);
            ASSERT_EQ(out.value, i * 2);
        }
    };

    std::thread producer_thread(producer);
    std::thread consumer_thread(consumer);

    producer_thread.join();
    consumer_thread.join();
}

TEST(spsc_reset_reuse) {
    SpscRing<TestData, 64> ring;

    // Push a few items.
    for (uint32_t i = 0; i < 16; ++i) {
        TestData data = {i, 0, i};
        ASSERT_TRUE(ring.push(data));
    }

    // Pop some items.
    TestData out{};
    for (uint32_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(ring.pop(&out));
        ASSERT_EQ(out.sequence, i);
    }

    // Reset and ensure we can reuse from an empty state.
    ring.reset();
    ASSERT_TRUE(ring.empty());

    for (uint32_t i = 0; i < 8; ++i) {
        TestData data = {100 + i, 0, i};
        ASSERT_TRUE(ring.push(data));
    }

    for (uint32_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(ring.pop(&out));
        ASSERT_EQ(out.value, 100 + i);
    }
}
