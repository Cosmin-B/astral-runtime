Astral Lock-Free Concurrency Primitives
========================================

This directory contains lock-free concurrency primitives for the Astral runtime.

FILES
-----

concurrency.hpp       - Main header that includes all primitives
mpmc_queue.hpp        - Multi-producer multi-consumer bounded queue
spsc_ring.hpp         - Single-producer single-consumer ring buffer
epoch.hpp             - Epoch-based memory reclamation
test_concurrency.cpp  - Test suite and usage examples

COMPONENTS
----------

1. MpmcQueue<T, Capacity>
   - Multi-producer multi-consumer bounded queue
   - Lock-free ring buffer with ticket-based ordering
   - Cache-line aligned head/tail atomics (64 bytes)
   - Exponential backoff on contention
   - Target: 20M+ ops/s on modern hardware

2. SpscRing<T, Capacity>
   - Single-producer single-consumer ring buffer
   - Zero contention (faster than MPMC)
   - Cache-line aligned head/tail atomics (64 bytes)
   - Target: 40M+ ops/s

3. EpochManager
   - Epoch-based memory reclamation
   - Safe deferred deletion without hazard pointers
   - Fixed-size thread registration (128 threads max)
   - Fixed-size deferred deletion ring (4096 items per epoch)

4. StreamToken
   - Fixed-size token struct for streaming (40 bytes)
   - Contains token_id, utf8_len, and utf8_data
   - Trivially copyable for efficient ring buffer operations

DESIGN PRINCIPLES
-----------------

- Zero allocations in hot paths (all structures are fixed-size)
- Explicit memory ordering (never seq_cst unless justified)
- Cache-line alignment to prevent false sharing (64 bytes)
- Power-of-2 sizes for fast modulo operations (bitwise AND)
- ARM weak memory model correctness (validated)

MEMORY ORDERING
---------------

MpmcQueue:
- Enqueue: fetch_add ticket (relaxed) + per-slot seq store (release) after writing data
- Dequeue: per-slot seq load (acquire) before reading data + per-slot seq store (release) to free slot
- Size: memory_order_relaxed (approximation only; based on tickets, not exact occupancy)

SpscRing:
- Push: memory_order_release on head store (publish data)
- Pop: memory_order_acquire on head load (synchronize-with push)
- Size: memory_order_relaxed (approximation only)

EpochManager:
- Enter: memory_order_acquire on global epoch load
- Leave: memory_order_release on thread epoch store
- Collect: memory_order_seq_cst on global epoch increment

CRITICAL FIXES
--------------

Current queue requirements:
- MPMC: Use per-slot sequence + acquire/release for ARM correctness
- MPMC: Add backoff in wait loops (reduces cache thrashing)
- Both: Cache-line align head/tail atomics to prevent false sharing

USAGE EXAMPLES
--------------

// MPMC Queue for work scheduling
MpmcQueue<WorkItem, 1024> work_queue;

// Producer thread
WorkItem work = {...};
work_queue.enqueue_wait(work);  // blocks until space is available

// Consumer thread
WorkItem work;
work_queue.dequeue_wait(&work); // blocks until an item is available

// SPSC Ring for token streaming
SpscRing<StreamToken, 4096> token_ring;

// Producer (decode thread)
StreamToken token = {...};
if (!token_ring.push(token)) {
    // Ring full, apply backpressure
}

// Consumer (callback thread)
StreamToken token;
if (token_ring.pop(&token)) {
    // Process token
}

// Epoch-based reclamation
EpochManager epoch_mgr;
int32_t thread_id = epoch_mgr.register_thread();

{
    EpochGuard guard(epoch_mgr, thread_id);
    // Access lock-free data structures here
}

// Defer deletion
Node* node = allocate_node();
epoch_mgr.defer_delete(node);

// Periodically collect garbage (dedicated thread)
epoch_mgr.collect();

BUILDING
--------

Compile test suite:
    g++ -std=c++17 -Wall -Wextra -Werror -O2 -o test_concurrency test_concurrency.cpp
    clang++ -std=c++17 -Wall -Wextra -Werror -O2 -o test_concurrency test_concurrency.cpp

Run tests:
    ./test_concurrency

REFERENCES
----------

- docs/architecture/CONCURRENCY_MODEL.md
- docs/rules/CODING_STANDARDS.md
- Custom MPMC design optimized for game engines
- Influenced by research on lock-free data structures:
  * "Simple, Fast, and Practical Non-Blocking..." (Michael & Scott, 1996)
  * Facebook Folly's ProducerConsumerQueue design patterns

NOTES
-----

- All structures require power-of-2 capacity (compile-time enforced)
- All data types must be trivially copyable (compile-time enforced)
- Cache-line size is assumed to be 64 bytes (standard for x86/ARM)
- EpochManager has fixed limits (128 threads, 4096 deferred deletions per epoch)
- SPSC Ring reserves one slot to distinguish full from empty state
