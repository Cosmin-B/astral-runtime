Astral Concurrency Primitives
=============================

This directory contains bounded queues, reclamation, and short internal locks.

FILES
-----

concurrency.hpp       - Main header that includes all primitives
backoff_spin_lock.hpp - Bounded-backoff lock for short internal critical sections
mpmc_queue.hpp        - Multi-producer multi-consumer bounded queue
mpsc_ring.hpp         - Non-blocking multi-producer single-consumer ring
mpsc_ticket_ring.hpp  - Backpressured multi-producer single-consumer ring
spsc_fan_in.hpp       - Fixed-owner SPSC lanes with one consumer
spsc_ring.hpp         - Single-producer single-consumer ring buffer
epoch.hpp             - Epoch-based memory reclamation
tests/test_concurrency.cpp - Maintained unit and concurrent stress tests

COMPONENTS
----------

1. MpmcQueue<T, Capacity>
   - Multi-producer multi-consumer bounded queue
   - Blocking ring buffer with ticket-based ordering
   - Capacity must be a power of 2 and at least 2
   - Cache-line aligned head/tail atomics (64 bytes)
   - Short active wait followed by the platform wait hint

2. BackoffSpinLock
   - Reads before attempting an atomic exchange
   - Uses bounded exponential pause backoff
   - Yields after the pause budget
   - Unlock is one release store with no wake broadcast

3. SpscRing<T, Capacity>
   - Single-producer single-consumer ring buffer
   - Zero contention (faster than MPMC)
   - Cache-line aligned head/tail atomics (64 bytes)

4. EpochManager
   - Epoch-based memory reclamation
   - Safe deferred deletion without hazard pointers
   - Reusable fixed-size participant registration (128 concurrent threads max)
   - One fixed 256-entry SPSC retirement queue per participant
   - Explicit deleters; queue overflow leaves ownership with the caller

5. StreamToken
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
- Enter: seq_cst epoch publication and validation handshake
- Leave: memory_order_release on thread epoch store
- Collect: drain the previously safe frontier, then seq_cst global epoch increment

QUEUE REQUIREMENTS
------------------

- Sequence-based MPSC and MPMC queues require Capacity >= 2
- Published and reusable sequence generations must remain distinct
- Per-slot acquire and release operations publish and consume payload data
- Shared producer and consumer cursors stay on separate cache lines

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
EpochManager<> epoch_mgr;
int32_t thread_id = epoch_mgr.register_thread();

{
    EpochGuard guard(epoch_mgr, thread_id);
    // Access lock-free data structures here
}

// Defer deletion
Node* node = allocate_node();
while (!epoch_mgr.defer_delete(thread_id, node)) {
    // The queue is fixed-capacity. Keep ownership and retry after collection.
    pause_or_wait();
}

// Periodically collect garbage from exactly one collector thread.
epoch_mgr.collect();

BUILDING
--------

Compile test suite:
    g++ -std=c++17 -Wall -Wextra -Werror -O2 -o test_concurrency test_concurrency.cpp
    clang++ -std=c++17 -Wall -Wextra -Werror -O2 -o test_concurrency test_concurrency.cpp

Run tests:
    ./test_concurrency

NOTES
-----

- Ring capacities that use bit masks must be powers of two
- Queue payloads must be trivially copyable
- Cache-line size is assumed to be 64 bytes (standard for x86/ARM)
- EpochManager defaults to 128 participants and 256 pending retirements per participant;
  both fixed capacities are configurable template arguments
- A failed defer_delete does not invoke the deleter; the caller retains ownership
- SPSC Ring reserves one slot to distinguish full from empty state
