/**
 * test_memory.cpp - Memory subsystem tests
 *
 * Tests for FrameAllocator, ObjectPool, and MemoryStats.
 * Validates: allocation, reset, alignment, capacity, thread-safety, out-of-memory.
 */

#include "test_framework.hpp"
#include "../src/memory/frame_allocator.hpp"
#include "../src/memory/object_pool.hpp"
#include "../src/memory/stats.hpp"
#include "../src/platform/vm.h"

#include <cstring>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>

using namespace astral::memory;
using namespace astral::platform;

namespace {

static void* alloc_backing(size_t size) {
#if ASTRAL_ENABLE_VIRTUAL_MEMORY
  void* memory = vm_reserve(size);
  if (memory == nullptr || !vm_commit(memory, size)) {
    vm_release(memory, size);
    return nullptr;
  }
  return memory;
#else
  return std::malloc(size);
#endif
}

static void free_backing(void* memory, size_t size) {
#if ASTRAL_ENABLE_VIRTUAL_MEMORY
    vm_release(memory, size);
#else
    (void)size;
    std::free(memory);
#endif
}

} // namespace

//
// FrameAllocator Tests
//

TEST(frame_allocator_basic) {
    constexpr size_t kCapacity = 1024 * 1024; // 1 MB
    void* memory = alloc_backing(kCapacity);

    FrameAllocator alloc(memory, kCapacity);
    ASSERT_EQ(alloc.capacity(), kCapacity);
    ASSERT_EQ(alloc.used(), 0);
    ASSERT_EQ(alloc.available(), kCapacity);

    void* p1 = alloc.alloc(64);
    ASSERT_NOT_NULL(p1);
    ASSERT_EQ(alloc.used(), 64);

    void* p2 = alloc.alloc(128);
    ASSERT_NOT_NULL(p2);
    ASSERT_LE(alloc.used(), 64 + 128 + 16); // May have alignment padding

    free_backing(memory, kCapacity);
}

TEST(frame_allocator_reset) {
    constexpr size_t kCapacity = 4096;
    void* memory = alloc_backing(kCapacity);

    FrameAllocator alloc(memory, kCapacity);

    void* p1 = alloc.alloc(64, 16);
    ASSERT_NOT_NULL(p1);

    alloc.reset();
    ASSERT_EQ(alloc.used(), 0);

    void* p2 = alloc.alloc(64, 16);
    ASSERT_NOT_NULL(p2);

    // After reset, same address should be returned (bump pointer reset)
    ASSERT_EQ(p1, p2);

    free_backing(memory, kCapacity);
}

TEST(frame_allocator_alignment) {
    constexpr size_t kCapacity = 8192;
    void* memory = alloc_backing(kCapacity);

    FrameAllocator alloc(memory, kCapacity);

    // Allocate with 64-byte alignment
    void* p1 = alloc.alloc(1, 64);
    ASSERT_NOT_NULL(p1);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p1) % 64, 0);

    // Allocate with 256-byte alignment
    void* p2 = alloc.alloc(1, 256);
    ASSERT_NOT_NULL(p2);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p2) % 256, 0);

    // Allocate with default alignment (16 bytes)
    void* p3 = alloc.alloc(1);
    ASSERT_NOT_NULL(p3);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p3) % 16, 0);

    free_backing(memory, kCapacity);
}

TEST(frame_allocator_alignment_unaligned_base) {
    constexpr size_t kCapacity = 8192;
    void* raw = std::malloc(kCapacity + 256);
    ASSERT_NOT_NULL(raw);

    void* memory = static_cast<void*>(static_cast<uint8_t*>(raw) + 1);
    FrameAllocator alloc(memory, kCapacity);

    void* p1 = alloc.alloc(1, 64);
    ASSERT_NOT_NULL(p1);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p1) % 64, 0);

    void* p2 = alloc.alloc(1, 256);
    ASSERT_NOT_NULL(p2);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p2) % 256, 0);

    std::free(raw);
}

TEST(frame_allocator_out_of_memory) {
    constexpr size_t kCapacity = 1024;
    void* memory = alloc_backing(kCapacity);

    FrameAllocator alloc(memory, kCapacity);

    // Allocate entire capacity
    void* p1 = alloc.alloc(kCapacity);
    ASSERT_NOT_NULL(p1);

    // Next allocation should fail
    void* p2 = alloc.alloc(1);
    ASSERT_NULL(p2);

    free_backing(memory, kCapacity);
}

TEST(frame_allocator_available) {
    constexpr size_t kCapacity = 2048;
    void* memory = alloc_backing(kCapacity);

    FrameAllocator alloc(memory, kCapacity);

    ASSERT_EQ(alloc.available(), kCapacity);

    alloc.alloc(512);
    ASSERT_LE(alloc.available(), kCapacity - 512);

    alloc.reset();
    ASSERT_EQ(alloc.available(), kCapacity);

    free_backing(memory, kCapacity);
}

TEST(frame_allocator_multiple_allocs) {
    constexpr size_t kCapacity = 16384;
    void* memory = alloc_backing(kCapacity);

    FrameAllocator alloc(memory, kCapacity);

    // Allocate multiple blocks
    constexpr size_t kBlockSize = 64;
    constexpr size_t kBlockCount = 100;

    for (size_t i = 0; i < kBlockCount; ++i) {
        void* p = alloc.alloc(kBlockSize, 16);
        ASSERT_NOT_NULL(p);

        // Write to verify accessibility
        memset(p, static_cast<int>(i & 0xFF), kBlockSize);
    }

    ASSERT_LE(alloc.used(), kBlockCount * (kBlockSize + 16)); // With alignment padding

    free_backing(memory, kCapacity);
}

//
// ObjectPool Tests
//

struct Token {
    uint64_t id;
    uint64_t timestamp;
    char data[48]; // Padding to make realistic size
};

TEST(object_pool_basic) {
    ObjectPool<Token, 16> pool;

    // Acquire object
    Token* tok = pool.acquire();
    ASSERT_NOT_NULL(tok);

    // Initialize and use
    tok->id = 42;
    tok->timestamp = 12345;

    // Release back to pool
    pool.release(tok);

    // Acquire again - should get same object
    Token* tok2 = pool.acquire();
    ASSERT_NOT_NULL(tok2);

    // Same pointer returned
    ASSERT_EQ(tok, tok2);

    pool.release(tok2);
}

TEST(object_pool_capacity) {
    constexpr size_t kCapacity = 8;
    ObjectPool<Token, kCapacity> pool;

    std::vector<Token*> tokens;

    // Acquire all objects
    for (size_t i = 0; i < kCapacity; ++i) {
        Token* tok = pool.acquire();
        ASSERT_NOT_NULL(tok);
        tokens.push_back(tok);
    }

    // Pool should be empty now
    Token* tok = pool.acquire();
    ASSERT_NULL(tok);

    // Release one
    pool.release(tokens.back());
    tokens.pop_back();

    // Should be able to acquire again
    tok = pool.acquire();
    ASSERT_NOT_NULL(tok);

    // Cleanup
    pool.release(tok);
    for (Token* t : tokens) {
        pool.release(t);
    }
}

TEST(object_pool_thread_safety) {
    constexpr size_t kPoolSize = 256;
    constexpr size_t kThreads = 4;
    static constexpr size_t kIterations = 10000;

    ObjectPool<Token, kPoolSize> pool;
    std::atomic<size_t> successful_acquires{0};

    auto worker = [&pool, &successful_acquires](int thread_id) {
      for (size_t i = 0; i < kIterations; ++i) {
        // Acquire
        Token* tok = pool.acquire();
        if (tok) {
          successful_acquires.fetch_add(1, std::memory_order_relaxed);

          // Simulate work
          tok->id = thread_id * 1000000 + i;
          tok->timestamp = i;

          // Release
          pool.release(tok);
        }
      }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < kThreads; ++i) {
        threads.emplace_back(worker, static_cast<int>(i));
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify successful acquires
    ASSERT_GT(successful_acquires.load(), 0);

    // All objects should be back in pool
    for (size_t i = 0; i < kPoolSize; ++i) {
        Token* tok = pool.acquire();
        ASSERT_NOT_NULL(tok);
    }

    // Pool should be empty
    Token* tok = pool.acquire();
    ASSERT_NULL(tok);
}

TEST(object_pool_release_nullptr) {
    ObjectPool<Token, 16> pool;

    // Releasing nullptr should be safe (no-op)
    pool.release(nullptr);

    // Pool should still have all objects
    for (size_t i = 0; i < pool.capacity(); ++i) {
        Token* tok = pool.acquire();
        ASSERT_NOT_NULL(tok);
    }
}

TEST(object_pool_stress_test) {
    constexpr size_t kPoolSize = 128;
    constexpr size_t kThreads = 8;
    static constexpr size_t kIterations = 5000;

    ObjectPool<Token, kPoolSize> pool;
    std::atomic<size_t> total_ops{0};

    auto worker = [&pool, &total_ops]() {
      for (size_t i = 0; i < kIterations; ++i) {
        Token* tok = pool.acquire();
        if (tok) {
          total_ops.fetch_add(1, std::memory_order_relaxed);
          // Brief work
          tok->id = i;
          pool.release(tok);
        }
      }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < kThreads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    ASSERT_GT(total_ops.load(), 0);
}

TEST(local_object_pool_basic) {
    constexpr size_t kCapacity = 16;
    constexpr uint64_t kTokenId = 42;
    constexpr uint64_t kTimestamp = 12345;
    LocalObjectPool<Token, kCapacity> pool;

    Token* tok = pool.acquire();
    ASSERT_NOT_NULL(tok);

    tok->id = kTokenId;
    tok->timestamp = kTimestamp;

    pool.release(tok);

    Token* tok2 = pool.acquire();
    ASSERT_NOT_NULL(tok2);
    ASSERT_EQ(tok, tok2);

    pool.release(tok2);
}

TEST(local_object_pool_capacity) {
    constexpr size_t kCapacity = 8;
    constexpr size_t kLastSlot = kCapacity - 1;
    LocalObjectPool<Token, kCapacity> pool;
    Token* tokens[kCapacity]{};

    for (size_t i = 0; i < kCapacity; ++i) {
        tokens[i] = pool.acquire();
        ASSERT_NOT_NULL(tokens[i]);
    }

    Token* tok = pool.acquire();
    ASSERT_NULL(tok);

    pool.release(tokens[kLastSlot]);
    tokens[kLastSlot] = nullptr;

    tok = pool.acquire();
    ASSERT_NOT_NULL(tok);

    pool.release(tok);
    for (Token* token : tokens) {
        if (token != nullptr) {
            pool.release(token);
        }
    }
}

TEST(local_object_pool_reuse_order) {
    constexpr size_t kCapacity = 4;
    constexpr size_t kFirstSlot = 0;
    constexpr size_t kSecondSlot = 1;
    LocalObjectPool<Token, kCapacity> pool;

    Token* first = pool.acquire();
    Token* second = pool.acquire();
    ASSERT_NOT_NULL(first);
    ASSERT_NOT_NULL(second);
    ASSERT_NE(first, second);

    pool.release(first);
    ASSERT_EQ(pool.acquire(), first);

    pool.release(second);
    Token* remaining[kCapacity]{};
    remaining[kFirstSlot] = pool.acquire();
    remaining[kSecondSlot] = pool.acquire();
    ASSERT_NOT_NULL(remaining[kFirstSlot]);
    ASSERT_NOT_NULL(remaining[kSecondSlot]);
}

//
// MemoryStats Tests
//

TEST(memory_stats_size) {
    MemoryStats stats = {};
    ASSERT_EQ(sizeof(stats), 40);
}

TEST(memory_stats_alignment) {
    MemoryStats stats = {};
    ASSERT_EQ(alignof(MemoryStats), 8);
}

TEST(memory_stats_basic) {
    MemoryStats stats = {};
    stats.bytes_reserved = 2ULL * 1024 * 1024 * 1024; // 2GB
    stats.bytes_committed = 4ULL * 1024 * 1024;        // 4MB
    stats.bytes_used = 1ULL * 1024 * 1024;             // 1MB
    stats.alloc_count = 100;
    stats.free_count = 50;

    ASSERT_EQ(stats.bytes_reserved, 2ULL * 1024 * 1024 * 1024);
    ASSERT_EQ(stats.bytes_committed, 4ULL * 1024 * 1024);
    ASSERT_EQ(stats.bytes_used, 1ULL * 1024 * 1024);
    ASSERT_EQ(stats.alloc_count, 100);
    ASSERT_EQ(stats.free_count, 50);

    // Compute live allocations
    uint64_t live = stats.alloc_count - stats.free_count;
    ASSERT_EQ(live, 50);
}

TEST(memory_stats_invariants) {
    MemoryStats stats = {};
    stats.bytes_reserved = 1024 * 1024 * 1024; // 1GB
    stats.bytes_committed = 512 * 1024 * 1024;  // 512MB
    stats.bytes_used = 256 * 1024 * 1024;       // 256MB

    // Invariants: used <= committed <= reserved
    ASSERT_LE(stats.bytes_used, stats.bytes_committed);
    ASSERT_LE(stats.bytes_committed, stats.bytes_reserved);
}

TEST(memory_stats_trivially_copyable) {
    MemoryStats stats1 = {};
    stats1.bytes_reserved = 1024;
    stats1.alloc_count = 10;

    // Verify trivially copyable (memcpy should work)
    MemoryStats stats2;
    memcpy(&stats2, &stats1, sizeof(MemoryStats));

    ASSERT_EQ(stats2.bytes_reserved, stats1.bytes_reserved);
    ASSERT_EQ(stats2.alloc_count, stats1.alloc_count);
}
