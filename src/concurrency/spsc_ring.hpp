#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../platform/atomics.h"
#include "../platform/cacheline.hpp"
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

    SpscRing() : head_(0), tail_(0) {
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
        // Load tail with acquire semantics to see consumer's progress
        // This ensures we don't overwrite data the consumer is reading
        uint64_t tail = tail_.load(std::memory_order_acquire);

        // Load head with relaxed semantics (we're the only writer)
        uint64_t head = head_.load(std::memory_order_relaxed);

        const bool was_empty = (head == tail);

        // Check if ring is full (bounded by Capacity elements).
        uint64_t next_head = head + 1;
        if (next_head - tail > Capacity) [[unlikely]] {
            return false;
        }

        // Write item to buffer
        size_t index = head & kIndexMask;
        buffer_[index] = item;

        // Publish new head with release semantics
        // This ensures item write is visible to consumer before head update
        head_.store(next_head, std::memory_order_release);

        // Wake a consumer potentially waiting for data.
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
        if (out == nullptr) [[unlikely]] {
            return false;
        }

        // Load head with acquire semantics to see producer's progress
        // This ensures we see the latest item writes from the producer
        uint64_t head = head_.load(std::memory_order_acquire);

        // Load tail with relaxed semantics (we're the only reader)
        uint64_t tail = tail_.load(std::memory_order_relaxed);

        // Check if ring is empty
        if (tail >= head) [[unlikely]] {
            return false;
        }

        const bool was_full = ((head - tail) >= Capacity);

        // Read item from buffer
        size_t index = tail & kIndexMask;
        *out = buffer_[index];

        // Publish new tail with release semantics
        // This ensures the producer sees that we've consumed the item
        tail_.store(tail + 1, std::memory_order_release);

        // Wake a producer potentially waiting for space.
        if (was_full) {
            astral::platform::cpu_signal_event();
        }
        return true;
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
    }

private:
    static constexpr size_t kCacheLineSize = astral::platform::kCacheLineAlign;
    static constexpr size_t kIndexMask = Capacity - 1;

    // Cache-line aligned atomics keep the producer and consumer cursors apart.
    // Producer writes head, consumer writes tail - keep them on separate cache lines
    alignas(kCacheLineSize) std::atomic<uint64_t> head_; // Producer writes
    alignas(kCacheLineSize) std::atomic<uint64_t> tail_; // Consumer writes

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
struct StreamToken {
    uint32_t token_id;
    uint16_t utf8_len;
    uint8_t utf8_data[32]; // Max 32 bytes per token (typical: 1-4 bytes)
};

static_assert(std::is_trivially_copyable_v<StreamToken>,
              "StreamToken must be trivially copyable for SPSC ring");

/// Type alias for token streaming ring.
/// Common size: 4096 tokens = ~128KB ring buffer (suitable for real-time streaming).
template<size_t Capacity = 4096>
using TokenRing = SpscRing<StreamToken, Capacity>;

} // namespace astral::concurrency
