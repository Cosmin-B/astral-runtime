#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../platform/atomics.h"
#include "../platform/cacheline.hpp"
#include "../platform/compiler.hpp"
#include "../utils/trace.hpp"

namespace astral::concurrency {

/// Single-producer single-consumer ring buffer for token streaming.
///
/// Design:
/// - Ring buffer with atomic head (producer) and tail (consumer) indices
/// - Producer: decode thread writes tokens
/// - Consumer: engine callback thread reads tokens
/// - Power-of-2 capacity for fast modulo via bitwise AND
/// - Cache-line aligned head/tail to prevent false sharing (64 bytes each)
///
/// Memory Ordering:
/// - Producer (push): memory_order_release on head write (publishes data)
/// - Consumer (pop): memory_order_acquire on head read (synchronizes-with producer)
/// - No CAS needed (single producer + single consumer = zero contention)
///
/// Performance:
/// - Target: 40M+ ops/s (faster than MPMC due to zero contention)
/// - No retry loops needed (only one writer, one reader)
///
/// Thread-safety: Single producer, single consumer only.
/// Calling push from multiple threads or pop from multiple threads is undefined behavior.
template<typename T, size_t Capacity>
class SpscRing {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    SpscRing()
        : head_(0),
          tail_(0),
          producer_head_cache_(0),
          producer_tail_cache_(0),
          consumer_head_cache_(0),
          consumer_tail_cache_(0) {
        // Initialize buffer to zero
        std::memset(buffer_, 0, sizeof(buffer_));
    }

    ~SpscRing() = default;

    // Non-copyable, non-movable (contains atomics)
    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    SpscRing(SpscRing&&) = delete;
    SpscRing& operator=(SpscRing&&) = delete;

    /// Push an item to the ring (producer only).
    ///
    /// @param item Item to push (copied into ring buffer).
    /// @return true if pushed successfully, false if ring is full.
    ///
    /// Memory ordering: memory_order_release on head store ensures item is visible
    /// to consumer after head is updated.
    ///
    /// IMPORTANT: Must be called from single producer thread only.
    bool push(const T& item) {
        ASTRAL_ZONE_MICRO_N("astral.spsc.push");
        // Producer owns head and caches the remote tail. The expensive remote
        // acquire load is needed only when the cached view says the ring may be full.
        uint64_t head = producer_head_cache_;
        uint64_t tail = producer_tail_cache_;
        const uint64_t next_head = head + 1;
        if (next_head - tail > Capacity) ASTRAL_UNLIKELY {
            tail = tail_.load(std::memory_order_acquire);
            producer_tail_cache_ = tail;
            if (next_head - tail > Capacity) ASTRAL_UNLIKELY {
                return false;
            }
        }

        const bool was_empty = (head == tail);
        const size_t index = head & kIndexMask;
        buffer_[index] = item;
        producer_head_cache_ = next_head;
        head_.store(next_head, std::memory_order_release);

        if (was_empty) {
            astral::platform::cpu_signal_event();
        }
        return true;
    }

    /// Pop an item from the ring (consumer only).
    ///
    /// @param out Output parameter to store popped item.
    /// @return true if popped successfully, false if ring is empty.
    ///
    /// Memory ordering: memory_order_acquire on head load ensures item data
    /// is synchronized-with the producer's push operation.
    ///
    /// IMPORTANT: Must be called from single consumer thread only.
    bool pop(T* out) {
        ASTRAL_ZONE_MICRO_N("astral.spsc.pop");
        if (out == nullptr) ASTRAL_UNLIKELY {
            return false;
        }
        return pop(*out);
    }

    /// Pop an item from the ring into a prevalidated reference.
    ///
    /// This overload avoids a null check in callers that already own the output
    /// storage and want the leanest hot path.
    bool pop(T& out) {
        ASTRAL_ZONE_MICRO_N("astral.spsc.pop_ref");

        // Consumer owns tail and caches the remote head. The expensive remote
        // acquire load is needed only when the cached view says the ring may be empty.
        uint64_t tail = consumer_tail_cache_;
        uint64_t head = consumer_head_cache_;
        if (tail >= head) ASTRAL_UNLIKELY {
            head = head_.load(std::memory_order_acquire);
            consumer_head_cache_ = head;
            if (tail >= head) ASTRAL_UNLIKELY {
                return false;
            }
        }

        const bool was_full = ((head - tail) >= Capacity);
        const size_t index = tail & kIndexMask;
        out = buffer_[index];
        const uint64_t next_tail = tail + 1;
        consumer_tail_cache_ = next_tail;
        // Reviewed release-store contract: tail_.store(tail + 1, std::memory_order_release)
        tail_.store(next_tail, std::memory_order_release);

        if (was_full) {
            astral::platform::cpu_signal_event();
        }
        return true;
    }

    /// Push as many items as currently fit, up to `count`.
    ///
    /// Preconditions:
    /// - `items` points to at least `count` elements.
    /// - Called only by the single producer.
    ///
    /// Returns the number of items written. A partial write means the ring was full.
    size_t push_batch(const T* items, size_t count) {
        ASTRAL_ZONE_MICRO_N("astral.spsc.push_batch");
        if (count == 0) ASTRAL_UNLIKELY {
            return 0;
        }

        uint64_t head = producer_head_cache_;
        uint64_t tail = producer_tail_cache_;
        size_t available = Capacity - static_cast<size_t>(head - tail);
        if (available < count) ASTRAL_UNLIKELY {
            tail = tail_.load(std::memory_order_acquire);
            producer_tail_cache_ = tail;
            available = Capacity - static_cast<size_t>(head - tail);
            if (available == 0) ASTRAL_UNLIKELY {
                return 0;
            }
        }

        const bool was_empty = (head == tail);
        const size_t n = count < available ? count : available;
        size_t first = Capacity - static_cast<size_t>(head & kIndexMask);
        if (first > n) {
            first = n;
        }

        std::memcpy(&buffer_[head & kIndexMask], items, first * sizeof(T));
        if (first < n) {
            std::memcpy(buffer_, items + first, (n - first) * sizeof(T));
        }

        const uint64_t next_head = head + n;
        producer_head_cache_ = next_head;
        head_.store(next_head, std::memory_order_release);
        if (was_empty) {
            astral::platform::cpu_signal_event();
        }
        return n;
    }

    /// Pop as many available items as possible, up to `count`.
    ///
    /// Preconditions:
    /// - `out` points to at least `count` elements.
    /// - Called only by the single consumer.
    ///
    /// Returns the number of items read. A zero return means the ring was empty.
    size_t pop_batch(T* out, size_t count) {
        ASTRAL_ZONE_MICRO_N("astral.spsc.pop_batch");
        if (count == 0) ASTRAL_UNLIKELY {
            return 0;
        }

        uint64_t tail = consumer_tail_cache_;
        uint64_t head = consumer_head_cache_;
        size_t available = static_cast<size_t>(head - tail);
        if (available < count) ASTRAL_UNLIKELY {
            head = head_.load(std::memory_order_acquire);
            consumer_head_cache_ = head;
            available = static_cast<size_t>(head - tail);
            if (available == 0) ASTRAL_UNLIKELY {
                return 0;
            }
        }

        const bool was_full = (available >= Capacity);
        const size_t n = count < available ? count : available;
        size_t first = Capacity - static_cast<size_t>(tail & kIndexMask);
        if (first > n) {
            first = n;
        }

        std::memcpy(out, &buffer_[tail & kIndexMask], first * sizeof(T));
        if (first < n) {
            std::memcpy(out + first, buffer_, (n - first) * sizeof(T));
        }

        const uint64_t next_tail = tail + n;
        consumer_tail_cache_ = next_tail;
        tail_.store(next_tail, std::memory_order_release);
        if (was_full) {
            astral::platform::cpu_signal_event();
        }
        return n;
    }

    /// Get approximate size of ring.
    ///
    /// @return Approximate number of items in ring (may be stale).
    ///
    /// Memory ordering: memory_order_relaxed is sufficient for size approximation;
    /// no synchronization needed for statistics.
    size_t size() const {
        uint64_t head = head_.load(std::memory_order_relaxed);
        uint64_t tail = tail_.load(std::memory_order_relaxed);
        // Handle wrap-around by casting to signed difference
        int64_t diff = static_cast<int64_t>(head - tail);
        return diff >= 0 ? static_cast<size_t>(diff) : 0;
    }

    /// Check if ring is approximately empty.
    ///
    /// @return true if ring appears empty (may be stale).
    bool empty() const {
        return size() == 0;
    }

    /// Get capacity of ring.
    ///
    /// @return Maximum number of items that can be stored.
    constexpr size_t capacity() const {
        return Capacity;
    }

    /// Reset ring to empty state.
    ///
    /// Preconditions:
    /// - Must not be called concurrently with push/pop.
    ///
    /// This does not clear the underlying buffer contents; it only resets
    /// head/tail indices.
    void reset() {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        producer_head_cache_ = 0;
        producer_tail_cache_ = 0;
        consumer_head_cache_ = 0;
        consumer_tail_cache_ = 0;
    }

private:
    static constexpr size_t kCacheLineSize = astral::platform::kCacheLineAlign;
    static constexpr size_t kIndexMask = Capacity - 1;

    // Cache-line aligned atomics keep the producer and consumer cursors apart.
    // Producer writes head, consumer writes tail - keep them on separate cache lines
    alignas(kCacheLineSize) std::atomic<uint64_t> head_; // Producer writes
    alignas(kCacheLineSize) std::atomic<uint64_t> tail_; // Consumer writes
    alignas(kCacheLineSize) uint64_t producer_head_cache_;
    alignas(kCacheLineSize) uint64_t producer_tail_cache_;
    alignas(kCacheLineSize) uint64_t consumer_head_cache_;
    alignas(kCacheLineSize) uint64_t consumer_tail_cache_;

    // Ring buffer storage (not atomic; protected by head/tail indices)
    T buffer_[Capacity];
};

/// Token format for streaming decoded tokens to engine.
///
/// Design:
/// - Fixed-size struct (trivially copyable) for zero-copy transfer
/// - token_id: Vocabulary index (0 to vocab_size-1)
/// - utf8_len: Number of valid bytes in utf8_data (0 to MAX_TOKEN_BYTES)
/// - utf8_data: UTF-8 encoded text for this token
///
/// Memory layout: Trivially copyable for efficient ring buffer operations.
static constexpr uint32_t kStreamTokenUtf8Capacity = 32;

struct StreamToken {
    uint32_t token_id;
    uint16_t utf8_len;
    uint8_t utf8_data[kStreamTokenUtf8Capacity]; // Max bytes per token (typical: 1-4)
};

static_assert(std::is_trivially_copyable_v<StreamToken>,
              "StreamToken must be trivially copyable for SPSC ring");

/// Type alias for token streaming ring.
/// Common size: 4096 tokens = ~128KB ring buffer (suitable for real-time streaming).
template<size_t Capacity = 4096>
using TokenRing = SpscRing<StreamToken, Capacity>;

} // namespace astral::concurrency
