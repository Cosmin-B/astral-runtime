/**
 * test_concurrency.cpp - Concurrency primitives tests
 *
 * Tests for MPMC queue, SPSC ring, and epoch manager.
 * Validates: enqueue/dequeue, push/pop, full/empty, thread-safety, memory ordering.
 */

#include "../src/concurrency/epoch.hpp"
#include "../src/concurrency/event_spin_lock.hpp"
#include "../src/concurrency/mpmc_queue.hpp"
#include "../src/concurrency/mpsc_ring.hpp"
#include "../src/concurrency/mpsc_ticket_ring.hpp"
#include "../src/concurrency/spsc_fan_in.hpp"
#include "../src/concurrency/spsc_ring.hpp"
#include "../src/platform/atomics.h"
#include "test_framework.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

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

struct EpochRetireProbe {
  std::atomic<uint32_t>* destroyed = nullptr;
  std::atomic<uint32_t>* duplicates = nullptr;
  std::atomic<uint8_t>* seen = nullptr;
  uint32_t id = 0;

  ~EpochRetireProbe() {
    if (seen != nullptr && seen[id].fetch_add(1, std::memory_order_relaxed) != 0) {
      duplicates->fetch_add(1, std::memory_order_relaxed);
    }
    destroyed->fetch_add(1, std::memory_order_release);
  }
};

static void delete_epoch_retire_probe(void* ptr) noexcept {
  delete static_cast<EpochRetireProbe*>(ptr);
}

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

TEST(event_spin_lock_serializes_access) {
  EventSpinLock lock;
  ASSERT_TRUE(lock.try_lock());
  ASSERT_FALSE(lock.try_lock());
  lock.unlock();

  constexpr uint32_t kThreadCount = 4;
  constexpr uint32_t kIterations = 10000;
  uint32_t counter = 0;
  std::thread threads[kThreadCount];

  for (uint32_t thread_i = 0; thread_i < kThreadCount; ++thread_i) {
    threads[thread_i] = std::thread([&]() {
      for (uint32_t i = 0; i < kIterations; ++i) {
        lock.lock();
        ++counter;
        lock.unlock();
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  ASSERT_EQ(counter, kThreadCount * kIterations);
}

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
    static constexpr size_t kItemsPerThread = 1000;

    MpmcQueue<TestData, kCapacity> queue;
    std::atomic<size_t> successful_enqueues{0};

    auto producer = [&queue, &successful_enqueues, kItemsPerThread](uint32_t thread_id) {
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
    static constexpr size_t kTotalItems = 4000;

    MpmcQueue<TestData, kCapacity> queue;

    // Pre-fill queue
    for (size_t i = 0; i < kTotalItems; ++i) {
        TestData data = {i, 0, 0};
        queue.enqueue_wait(data);
    }

    std::atomic<size_t> items_consumed{0};
    std::atomic<size_t> next_item{0};

    auto consumer = [&queue, &items_consumed, &next_item, kTotalItems]() {
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
    static constexpr size_t kItemsPerProducer = 5000;

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
    static constexpr uint64_t kItemsPerProducer = 4000;
    static constexpr uint64_t kTotalItems = static_cast<uint64_t>(kProducers) * kItemsPerProducer;

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

TEST(mpsc_ticket_multi_producer_single_consumer_checksum) {
    constexpr size_t kCapacity = 256;
    constexpr uint32_t kProducers = 4;
    static constexpr uint64_t kItemsPerProducer = 4000;
    static constexpr uint64_t kTotalItems = static_cast<uint64_t>(kProducers) * kItemsPerProducer;

    MpscTicketRing<uint64_t, kCapacity> ring;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<uint64_t> consumed{0};
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
                ring.push_wait(base + i);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kProducers) {
        astral::platform::cpu_pause();
    }
    start.store(true, std::memory_order_release);

    uint64_t sum = 0;
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
    ASSERT_EQ(consumed.load(), kTotalItems);
    ASSERT_EQ(sum, expected);
}

TEST(mpsc_ticket_pop_wait_blocks_until_producer_publishes) {
  MpscTicketRing<TestData, 4> ring;

  std::atomic<bool> popped{false};
  TestData out{};

  std::thread consumer([&]() {
    ring.pop_wait(out);
    popped.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_FALSE(popped.load(std::memory_order_acquire));

  ring.push_wait(TestData{77, 3, 9});

  consumer.join();
  ASSERT_TRUE(popped.load(std::memory_order_acquire));
  ASSERT_EQ(out.value, 77u);
  ASSERT_EQ(out.thread_id, 3u);
  ASSERT_EQ(out.sequence, 9u);
}

TEST(mpsc_ticket_batch_multi_producer_single_consumer_checksum) {
    constexpr size_t kCapacity = 256;
    constexpr uint32_t kProducers = 4;
    static constexpr uint64_t kItemsPerProducer = 4096;
    static constexpr uint64_t kBatch = 16;
    static constexpr uint64_t kTotalItems = static_cast<uint64_t>(kProducers) * kItemsPerProducer;

    MpscTicketRing<uint64_t, kCapacity> ring;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<uint64_t> consumed{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducers);

    for (uint32_t p = 0; p < kProducers; ++p) {
      producers.emplace_back([&, p]() {
        uint64_t batch[16];
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
          astral::platform::cpu_pause();
        }

        const uint64_t base = static_cast<uint64_t>(p) * kItemsPerProducer;
        for (uint64_t i = 0; i < kItemsPerProducer; i += kBatch) {
          for (uint64_t j = 0; j < kBatch; ++j) {
            batch[j] = base + i + j;
          }
          ring.push_batch_wait(batch, kBatch);
        }
      });
    }

    while (ready.load(std::memory_order_acquire) != kProducers) {
        astral::platform::cpu_pause();
    }
    start.store(true, std::memory_order_release);

    uint64_t sum = 0;
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
    ASSERT_EQ(consumed.load(), kTotalItems);
    ASSERT_EQ(sum, expected);
}

TEST(spsc_fan_in_multi_producer_single_consumer_checksum) {
    constexpr size_t kProducers = 4;
    constexpr size_t kCapacity = 64;
    static constexpr uint64_t kItemsPerProducer = 1000;
    static constexpr uint64_t kTotalItems = static_cast<uint64_t>(kProducers) * kItemsPerProducer;

    SpscFanIn<uint64_t, kProducers, kCapacity> fan_in;

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<uint64_t> consumed{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducers);

    for (size_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                astral::platform::cpu_pause();
            }

            const uint64_t base = static_cast<uint64_t>(p) * kItemsPerProducer;
            for (uint64_t i = 0; i < kItemsPerProducer; ++i) {
                while (!fan_in.try_push(p, base + i)) {
                    astral::platform::cpu_pause();
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kProducers) {
        astral::platform::cpu_pause();
    }
    start.store(true, std::memory_order_release);

    uint64_t sum = 0;
    while (consumed.load(std::memory_order_relaxed) < kTotalItems) {
        uint64_t value = 0;
        size_t producer_index = 0;
        if (fan_in.pop(value, &producer_index)) {
            ASSERT_LT(producer_index, kProducers);
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
    ASSERT_EQ(consumed.load(), kTotalItems);
    ASSERT_EQ(sum, expected);
}

TEST(spsc_fan_in_rejects_invalid_producer_lane) {
  constexpr size_t kProducers = 2;
  SpscFanIn<uint64_t, kProducers, 8> fan_in;

  const uint64_t item = 42;
  ASSERT_FALSE(fan_in.try_push(kProducers, item));
  ASSERT_EQ(fan_in.push_batch(kProducers, &item, 1), 0u);

  ASSERT_TRUE(fan_in.try_push(0, item));
  uint64_t out = 0;
  ASSERT_TRUE(fan_in.pop(out));
  ASSERT_EQ(out, item);
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

TEST(spsc_batch_push_pop_wraparound) {
    constexpr size_t kCapacity = 8;
    SpscRing<TestData, kCapacity> ring;

    TestData input[12]{};
    for (uint32_t i = 0; i < 12; ++i) {
        input[i] = TestData{i, 0, i};
    }

    ASSERT_EQ(ring.push_batch(input, 6), 6u);

    TestData first[4]{};
    ASSERT_EQ(ring.pop_batch(first, 4), 4u);
    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(first[i].value, i);
        ASSERT_EQ(first[i].sequence, i);
    }

    ASSERT_EQ(ring.push_batch(input + 6, 6), 6u);
    ASSERT_EQ(ring.push_batch(input, 1), 0u);

    TestData out[8]{};
    ASSERT_EQ(ring.pop_batch(out, 8), 8u);
    for (uint32_t i = 0; i < 8; ++i) {
        ASSERT_EQ(out[i].value, i + 4);
        ASSERT_EQ(out[i].sequence, i + 4);
    }
    ASSERT_EQ(ring.pop_batch(out, 8), 0u);
    ASSERT_TRUE(ring.empty());
}

TEST(spsc_consume_batch_wraparound) {
  constexpr size_t kCapacity = 8;
  SpscRing<TestData, kCapacity> ring;

  TestData input[12]{};
  for (uint32_t i = 0; i < 12; ++i) {
    input[i] = TestData{i, 0, i};
  }

  ASSERT_EQ(ring.push_batch(input, 6), 6u);

  TestData first[4]{};
  ASSERT_EQ(ring.pop_batch(first, 4), 4u);
  ASSERT_EQ(ring.push_batch(input + 6, 6), 6u);

  uint32_t seen = 0;
  const size_t consumed = ring.consume_batch(8, [&](const TestData& current) {
    ASSERT_EQ(current.value, seen + 4u);
    ASSERT_EQ(current.sequence, seen + 4u);
    ++seen;
  });

  ASSERT_EQ(consumed, 8u);
  ASSERT_EQ(seen, 8u);
  ASSERT_TRUE(ring.empty());
}

TEST(spsc_batch_partial_capacity) {
    SpscRing<TestData, 4> ring;

    TestData input[8]{};
    for (uint32_t i = 0; i < 8; ++i) {
        input[i] = TestData{i + 10, 0, i};
    }

    ASSERT_EQ(ring.push_batch(input, 8), 4u);
    ASSERT_EQ(ring.push_batch(input, 1), 0u);

    TestData out[8]{};
    ASSERT_EQ(ring.pop_batch(out, 8), 4u);
    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(out[i].value, i + 10);
        ASSERT_EQ(out[i].sequence, i);
    }

    ASSERT_EQ(ring.push_batch(input + 4, 4), 4u);
    ASSERT_EQ(ring.pop_batch(out, 2), 2u);
    ASSERT_EQ(ring.pop_batch(out + 2, 2), 2u);
    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(out[i].value, i + 14);
        ASSERT_EQ(out[i].sequence, i + 4);
    }
}

TEST(spsc_producer_consumer_pattern) {
    constexpr size_t kCapacity = 128;
    static constexpr size_t kItemCount = 10000;

    SpscRing<TestData, kCapacity> ring;
    std::atomic<size_t> items_consumed{0};

    auto producer = [&ring, kItemCount]() {
      for (size_t i = 0; i < kItemCount; ++i) {
        TestData data = {i, 0, static_cast<uint32_t>(i)};
        while (!ring.push(data)) {
          // Spin wait
        }
      }
    };

    auto consumer = [&ring, &items_consumed, kItemCount]() {
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
    static constexpr size_t kItemCount = 1000;

    SpscRing<TestData, kCapacity> ring;

    auto producer = [&ring, kItemCount]() {
      for (size_t i = 0; i < kItemCount; ++i) {
        TestData data = {i * 2, 0, static_cast<uint32_t>(i)};
        while (!ring.push(data)) {
        }
      }
    };

    auto consumer = [&ring, kItemCount]() {
      TestData out;
      for (size_t i = 0; i < kItemCount; ++i) {
        while (!ring.pop(&out)) {
        }
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

TEST(epoch_registration_slots_are_reusable) {
  EpochManager<> manager;

  for (size_t i = 0; i < EpochManager<>::kMaxThreads * 3; ++i) {
    const int32_t participant = manager.register_thread();
    ASSERT_GE(participant, 0);
    manager.unregister_thread(participant);
  }
}

TEST(epoch_retire_overflow_never_frees_a_pinned_object) {
  EpochManager<> manager;
  const int32_t retire_participant = manager.register_thread();
  const int32_t reader_participant = manager.register_thread();
  ASSERT_GE(retire_participant, 0);
  ASSERT_GE(reader_participant, 0);

  std::atomic<uint32_t> destroyed{0};
  std::atomic<uint32_t> duplicates{0};
  manager.enter(reader_participant);

  for (size_t i = 0; i < EpochManager<>::kMaxRetiredPerParticipant; ++i) {
    auto* probe = new EpochRetireProbe{&destroyed, &duplicates, nullptr, 0};
    ASSERT_TRUE(manager.defer_delete(retire_participant, probe, delete_epoch_retire_probe));
  }

  auto* overflow = new EpochRetireProbe{&destroyed, &duplicates, nullptr, 0};
  ASSERT_FALSE(manager.defer_delete(retire_participant, overflow, delete_epoch_retire_probe));

  for (uint32_t i = 0; i < 8; ++i) {
    manager.collect();
  }
  ASSERT_EQ(destroyed.load(std::memory_order_acquire), 0u);

  manager.leave(reader_participant);
  for (uint32_t i = 0; i < 4; ++i) {
    manager.collect();
  }
  ASSERT_EQ(destroyed.load(std::memory_order_acquire),
            static_cast<uint32_t>(EpochManager<>::kMaxRetiredPerParticipant));

  ASSERT_TRUE(manager.defer_delete(retire_participant, overflow, delete_epoch_retire_probe));
  for (uint32_t i = 0; i < 4; ++i) {
    manager.collect();
  }
  ASSERT_EQ(destroyed.load(std::memory_order_acquire),
            static_cast<uint32_t>(EpochManager<>::kMaxRetiredPerParticipant + 1));
  ASSERT_EQ(duplicates.load(std::memory_order_relaxed), 0u);

  manager.unregister_thread(reader_participant);
  manager.unregister_thread(retire_participant);
}

TEST(epoch_concurrent_retire_and_collect_reclaims_once) {
  constexpr uint32_t kProducerCount = 8;
  constexpr uint32_t kRetiresPerProducer = 2000;
  constexpr uint32_t kTotalRetires = kProducerCount * kRetiresPerProducer;

  EpochManager<> manager;
  std::atomic<uint32_t> destroyed{0};
  std::atomic<uint32_t> duplicates{0};
  std::atomic<uint32_t> producers_ready{0};
  std::atomic<uint32_t> producers_done{0};
  std::atomic<bool> start{false};
  std::atomic<bool> registration_failed{false};
  std::atomic<bool> timed_out{false};
  auto seen = std::make_unique<std::atomic<uint8_t>[]>(kTotalRetires);
  for (uint32_t i = 0; i < kTotalRetires; ++i) {
    seen[i].store(0, std::memory_order_relaxed);
  }

  std::vector<std::thread> producers;
  producers.reserve(kProducerCount);
  for (uint32_t producer = 0; producer < kProducerCount; ++producer) {
    producers.emplace_back([&, producer]() {
      const int32_t participant = manager.register_thread();
      if (participant < 0) {
        registration_failed.store(true, std::memory_order_release);
        producers_ready.fetch_add(1, std::memory_order_release);
        producers_done.fetch_add(1, std::memory_order_release);
        return;
      }

      producers_ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (uint32_t item = 0; item < kRetiresPerProducer; ++item) {
        const uint32_t id = producer * kRetiresPerProducer + item;
        auto* probe = new EpochRetireProbe{&destroyed, &duplicates, seen.get(), id};
        while (!manager.defer_delete(participant, probe, delete_epoch_retire_probe)) {
          std::this_thread::yield();
        }
      }

      manager.unregister_thread(participant);
      producers_done.fetch_add(1, std::memory_order_release);
    });
  }

  while (producers_ready.load(std::memory_order_acquire) != kProducerCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (producers_done.load(std::memory_order_acquire) != kProducerCount ||
         destroyed.load(std::memory_order_acquire) != kTotalRetires) {
    manager.collect();
    if (std::chrono::steady_clock::now() >= deadline) {
      timed_out.store(true, std::memory_order_release);
      break;
    }
  }

  for (std::thread& producer : producers) {
    producer.join();
  }
  for (uint32_t i = 0; i < 4; ++i) {
    manager.collect();
  }

  ASSERT_FALSE(registration_failed.load(std::memory_order_acquire));
  ASSERT_FALSE(timed_out.load(std::memory_order_acquire));
  ASSERT_EQ(destroyed.load(std::memory_order_acquire), kTotalRetires);
  ASSERT_EQ(duplicates.load(std::memory_order_relaxed), 0u);
  for (uint32_t i = 0; i < kTotalRetires; ++i) {
    ASSERT_EQ(seen[i].load(std::memory_order_relaxed), static_cast<uint8_t>(1));
  }
}
