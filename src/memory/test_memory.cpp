/// Simple validation tests for memory subsystem
/// Compile: g++ -std=c++20 -Wall -Wextra -O2 test_memory.cpp -o test_memory
/// Run: ./test_memory

#include "frame_allocator.hpp"
#include "object_pool.hpp"
#include "stats.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace astral::memory;

// Simple test framework
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    static void run_test_##name() { \
        printf("Running test: %s\n", #name); \
        test_##name(); \
        printf("  PASSED\n"); \
        g_tests_passed++; \
    } \
    static void test_##name()

#define ASSERT(expr) \
    do { \
        if (!(expr)) { \
            printf("  ASSERTION FAILED: %s (line %d)\n", #expr, __LINE__); \
            g_tests_failed++; \
            return; \
        } \
    } while (0)

// FrameAllocator Tests
TEST(frame_allocator_basic) {
    constexpr size_t kCapacity = 1024 * 1024; // 1 MB
    void* memory = malloc(kCapacity);
    ASSERT(memory != nullptr);

    FrameAllocator alloc(memory, kCapacity);
    ASSERT(alloc.capacity() == kCapacity);
    ASSERT(alloc.used() == 0);
    ASSERT(alloc.available() == kCapacity);

    void* p1 = alloc.alloc(64);
    ASSERT(p1 != nullptr);
    ASSERT(alloc.used() == 64);

    void* p2 = alloc.alloc(128);
    ASSERT(p2 != nullptr);
    ASSERT(alloc.used() <= 64 + 128 + 16); // May have alignment padding

    free(memory);
}

TEST(frame_allocator_reset) {
    constexpr size_t kCapacity = 4096;
    void* memory = malloc(kCapacity);
    ASSERT(memory != nullptr);

    FrameAllocator alloc(memory, kCapacity);

    void* p1 = alloc.alloc(64, 16);
    ASSERT(p1 != nullptr);

    alloc.reset();
    ASSERT(alloc.used() == 0);

    void* p2 = alloc.alloc(64, 16);
    ASSERT(p2 != nullptr);

    // After reset, same address should be returned (bump pointer reset)
    ASSERT(p1 == p2);

    free(memory);
}

TEST(frame_allocator_alignment) {
    constexpr size_t kCapacity = 8192;
    // Use aligned_alloc to ensure base is properly aligned
    void* memory = aligned_alloc(256, kCapacity);
    ASSERT(memory != nullptr);

    FrameAllocator alloc(memory, kCapacity);

    // Allocate with 64-byte alignment
    void* p1 = alloc.alloc(1, 64);
    ASSERT(p1 != nullptr);
    ASSERT(reinterpret_cast<uintptr_t>(p1) % 64 == 0);

    // Allocate with 256-byte alignment
    void* p2 = alloc.alloc(1, 256);
    ASSERT(p2 != nullptr);
    ASSERT(reinterpret_cast<uintptr_t>(p2) % 256 == 0);

    free(memory);
}

TEST(frame_allocator_out_of_memory) {
    constexpr size_t kCapacity = 1024;
    void* memory = malloc(kCapacity);
    ASSERT(memory != nullptr);

    FrameAllocator alloc(memory, kCapacity);

    // Allocate entire capacity
    void* p1 = alloc.alloc(kCapacity);
    ASSERT(p1 != nullptr);

    // Next allocation should fail
    void* p2 = alloc.alloc(1);
    ASSERT(p2 == nullptr);

    free(memory);
}

// ObjectPool Tests
struct Token {
    uint64_t id;
    uint64_t timestamp;
    char data[48]; // Padding to make it realistic size
};

TEST(object_pool_basic) {
    ObjectPool<Token, 16> pool;

    // Acquire object
    Token* tok = pool.acquire();
    ASSERT(tok != nullptr);

    // Initialize and use
    tok->id = 42;
    tok->timestamp = 12345;

    // Release back to pool
    pool.release(tok);

    // Acquire again - should get same object
    Token* tok2 = pool.acquire();
    ASSERT(tok2 != nullptr);

    // Note: Data is NOT preserved after release (intrusive freelist overwrites first 8 bytes)
    // This is expected behavior for object pools
    ASSERT(tok == tok2); // Same pointer returned
}

TEST(object_pool_capacity) {
    constexpr size_t kCapacity = 8;
    ObjectPool<Token, kCapacity> pool;

    std::vector<Token*> tokens;

    // Acquire all objects
    for (size_t i = 0; i < kCapacity; ++i) {
        Token* tok = pool.acquire();
        ASSERT(tok != nullptr);
        tokens.push_back(tok);
    }

    // Pool should be empty now
    Token* tok = pool.acquire();
    ASSERT(tok == nullptr);

    // Release one
    pool.release(tokens.back());
    tokens.pop_back();

    // Should be able to acquire again
    tok = pool.acquire();
    ASSERT(tok != nullptr);

    // Cleanup
    pool.release(tok);
    for (Token* t : tokens) {
        pool.release(t);
    }
}

TEST(object_pool_thread_safety) {
    constexpr size_t kPoolSize = 256;
    constexpr size_t kThreads = 4;
    constexpr size_t kIterations = 10000;

    ObjectPool<Token, kPoolSize> pool;

    auto worker = [&pool](int thread_id) {
        for (size_t i = 0; i < kIterations; ++i) {
            // Acquire
            Token* tok = pool.acquire();
            if (tok) {
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
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    // All objects should be back in pool
    for (size_t i = 0; i < kPoolSize; ++i) {
        Token* tok = pool.acquire();
        ASSERT(tok != nullptr);
    }

    // Pool should be empty
    Token* tok = pool.acquire();
    ASSERT(tok == nullptr);
}

// MemoryStats Tests
TEST(memory_stats_size) {
    MemoryStats stats = {};
    ASSERT(sizeof(stats) == 40);
}

TEST(memory_stats_basic) {
    MemoryStats stats = {};
    stats.bytes_reserved = 2ULL * 1024 * 1024 * 1024; // 2GB
    stats.bytes_committed = 4ULL * 1024 * 1024;        // 4MB
    stats.bytes_used = 1ULL * 1024 * 1024;             // 1MB
    stats.alloc_count = 100;
    stats.free_count = 50;

    ASSERT(stats.bytes_reserved == 2ULL * 1024 * 1024 * 1024);
    ASSERT(stats.bytes_committed == 4ULL * 1024 * 1024);
    ASSERT(stats.bytes_used == 1ULL * 1024 * 1024);
    ASSERT(stats.alloc_count == 100);
    ASSERT(stats.free_count == 50);

    // Compute live allocations
    uint64_t live = stats.alloc_count - stats.free_count;
    ASSERT(live == 50);
}

// Main test runner
int main() {
    printf("=== Astral Memory Subsystem Tests ===\n\n");

    // Run all tests
    run_test_frame_allocator_basic();
    run_test_frame_allocator_reset();
    run_test_frame_allocator_alignment();
    run_test_frame_allocator_out_of_memory();
    run_test_object_pool_basic();
    run_test_object_pool_capacity();
    run_test_object_pool_thread_safety();
    run_test_memory_stats_size();
    run_test_memory_stats_basic();

    printf("\n=== Test Results ===\n");
    printf("Passed: %d\n", g_tests_passed);
    printf("Failed: %d\n", g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
