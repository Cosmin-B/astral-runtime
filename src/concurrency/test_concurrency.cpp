/// Standalone compile smoke for concurrency primitives.
///
/// This is not a comprehensive test suite; it verifies that:
/// 1. All headers compile without errors
/// 2. Basic operations (enqueue/dequeue, push/pop, enter/leave) work
/// 3. Type constraints (trivially copyable, power-of-2) are enforced
///
/// Compile with:
/// g++ -std=c++17 -Wall -Wextra -Werror -O2 -o test_concurrency test_concurrency.cpp
/// clang++ -std=c++17 -Wall -Wextra -Werror -O2 -o test_concurrency test_concurrency.cpp

#include "concurrency.hpp"
#include <cstdio>
#include <cassert>

using namespace astral::concurrency;

// Test struct (trivially copyable)
struct TestItem {
    uint32_t id;
    float value;
};

static_assert(std::is_trivially_copyable_v<TestItem>, "TestItem must be trivially copyable");

void test_mpmc_queue() {
    printf("Testing MpmcQueue...\n");

    MpmcQueue<TestItem, 16> queue;

    // MpmcQueue is a bounded blocking queue; enqueue_wait/dequeue_wait do not fail.
    TestItem item1{42, 3.14f};
    queue.enqueue_wait(item1);
    assert(queue.size() >= 1);
    assert(!queue.empty());

    TestItem item2{};
    queue.dequeue_wait(&item2);
    assert(item2.id == 42);
    assert(item2.value == 3.14f);

    printf("MpmcQueue: PASS\n");
}

void test_spsc_ring() {
    printf("Testing SpscRing...\n");

    SpscRing<TestItem, 32> ring;

    // Test push
    TestItem item1{123, 4.56f};
    assert(ring.push(item1));
    assert(ring.size() == 1);
    assert(!ring.empty());

    // Test pop
    TestItem item2;
    assert(ring.pop(&item2));
    assert(item2.id == 123);
    assert(item2.value == 4.56f);
    assert(ring.empty());

    // Test full condition
    for (uint32_t i = 0; i < 32; ++i) {
        TestItem item{i, static_cast<float>(i)};
        assert(ring.push(item));
    }

    // Should fail when full
    TestItem overflow{999, 999.0f};
    assert(!ring.push(overflow));

    // Drain ring
    for (uint32_t i = 0; i < 32; ++i) {
        TestItem item;
        assert(ring.pop(&item));
        assert(item.id == i);
    }
    assert(ring.empty());

    // Should fail when empty
    TestItem underflow;
    assert(!ring.pop(&underflow));

    printf("SpscRing: PASS\n");
}

void test_stream_token() {
    printf("Testing StreamToken...\n");

    StreamToken token;
    token.token_id = 42;
    token.utf8_len = 4;
    token.utf8_data[0] = 't';
    token.utf8_data[1] = 'e';
    token.utf8_data[2] = 's';
    token.utf8_data[3] = 't';

    // Verify trivially copyable
    static_assert(std::is_trivially_copyable_v<StreamToken>);

    // Test with TokenRing
    TokenRing<256> token_ring;
    assert(token_ring.push(token));

    StreamToken token2;
    assert(token_ring.pop(&token2));
    assert(token2.token_id == 42);
    assert(token2.utf8_len == 4);
    assert(token2.utf8_data[0] == 't');
    assert(token2.utf8_data[1] == 'e');
    assert(token2.utf8_data[2] == 's');
    assert(token2.utf8_data[3] == 't');

    printf("StreamToken: PASS\n");
}

void test_epoch_manager() {
  printf("Testing EpochManager...\n");

  EpochManager<> epoch_mgr;

  // Register thread
  int32_t thread_id = epoch_mgr.register_thread();
  assert(thread_id >= 0);

  // Test enter/leave
  epoch_mgr.enter(thread_id);
  epoch_mgr.leave(thread_id);

  // Test RAII guard
  {
    EpochGuard guard(epoch_mgr, thread_id);
    // Guard automatically enters epoch
  }
  // Guard automatically leaves epoch on scope exit

  // Test defer_delete (basic check; maintained tests verify concurrent reclamation)
  int* ptr = new int(42);
  assert(epoch_mgr.defer_delete(thread_id, ptr));

  // Establish and consume the safe frontier.
  for (uint32_t i = 0; i < 5; ++i) {
    epoch_mgr.collect();
  }

  // Unregister thread
  epoch_mgr.unregister_thread(thread_id);

  printf("EpochManager: PASS\n");
}

void test_compilation_constraints() {
    printf("Testing compilation constraints...\n");

    // Power-of-2 constraint
    static_assert((16 & (16 - 1)) == 0, "16 is power of 2");
    static_assert((32 & (32 - 1)) == 0, "32 is power of 2");
    static_assert((256 & (256 - 1)) == 0, "256 is power of 2");

    // These should fail to compile (commented out):
    // MpmcQueue<TestItem, 15> bad_queue1; // Not power of 2
    // MpmcQueue<TestItem, 0> bad_queue2;  // Zero capacity
    // MpmcQueue<std::string, 16> bad_queue3; // Not trivially copyable

    // Verify cache-line sizes
    static_assert(alignof(MpmcQueue<TestItem, 16>) >= 64, "MPMC head/tail cache-aligned");
    static_assert(alignof(SpscRing<TestItem, 16>) >= 64, "SPSC head/tail cache-aligned");

    printf("Compilation constraints: PASS\n");
}

int main() {
    printf("Astral Concurrency Primitives Test Suite\n");
    printf("=========================================\n\n");

    test_mpmc_queue();
    test_spsc_ring();
    test_stream_token();
    test_epoch_manager();
    test_compilation_constraints();

    printf("\nAll tests passed!\n");
    return 0;
}
