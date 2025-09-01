#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

#include "../platform/atomics.h"

namespace astral::concurrency {

/// Multi-producer multi-consumer bounded queue (fixed-capacity ring).
///
/// Implementation: per-slot sequence (Vyukov-style) with explicit memory ordering.
///
/// Goals:
/// - Correct on weak memory models (ARM).
/// - No dynamic allocation.
/// - No compare-exchange in the hot path:
///   - `enqueue_wait`/`dequeue_wait` are blocking primitives built from fetch_add + per-slot sequence + WFE/SEV.
template<typename T, size_t Capacity>
class MpmcQueue {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    MpmcQueue() : enqueue_pos_(0), dequeue_pos_(0) {
        for (uint64_t i = 0; i < Capacity; ++i) {
            buffer_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    ~MpmcQueue() = default;

    // Non-copyable, non-movable (contains atomics)
    MpmcQueue(const MpmcQueue&) = delete;
    MpmcQueue& operator=(const MpmcQueue&) = delete;
    MpmcQueue(MpmcQueue&&) = delete;
    MpmcQueue& operator=(MpmcQueue&&) = delete;

    /// Enqueue an item, waiting until space is available.
    ///
    /// Hot path has no CAS (fetch_add + per-slot sequence + WFE/SEV).
    void enqueue_wait(const T& item) {
        const uint64_t pos = enqueue_pos_.fetch_add(1, std::memory_order_relaxed);
        Slot& slot = buffer_[pos & kIndexMask];

        uint32_t spins = 0;
        // Spin with relaxed loads and only pay the acquire barrier once, right before writing data.
        while (slot.seq.load(std::memory_order_relaxed) != pos) {
            wait_hint(spins);
        }

        // Synchronize with the consumer's release-store of seq (which follows reading slot.data).
        (void)slot.seq.load(std::memory_order_acquire);
        slot.data = item;
        slot.seq.store(pos + 1, std::memory_order_release);
        astral::platform::cpu_signal_event();
    }

    /// Dequeue an item, waiting until one is available.
    ///
    /// Hot path has no CAS (fetch_add + per-slot sequence + WFE/SEV).
    void dequeue_wait(T* out) {
        if (out == nullptr) {
            return;
        }

        const uint64_t pos = dequeue_pos_.fetch_add(1, std::memory_order_relaxed);
        Slot& slot = buffer_[pos & kIndexMask];

        uint32_t spins = 0;
        // Spin with relaxed loads and only pay the acquire barrier once, right before reading data.
        while (slot.seq.load(std::memory_order_relaxed) != pos + 1) {
            wait_hint(spins);
        }

        // Synchronize with the producer's release-store of seq (which follows writing slot.data).
        (void)slot.seq.load(std::memory_order_acquire);
        *out = slot.data;
        slot.seq.store(pos + Capacity, std::memory_order_release);
        astral::platform::cpu_signal_event();
    }

    /// Get approximate size of queue.
    ///
    /// @return Approximate number of items in queue (may be stale).
    ///
    /// Memory ordering: memory_order_relaxed is sufficient for size approximation;
    /// no synchronization needed for statistics.
    size_t size() const {
        uint64_t head = enqueue_pos_.load(std::memory_order_relaxed);
        uint64_t tail = dequeue_pos_.load(std::memory_order_relaxed);
        const int64_t diff = static_cast<int64_t>(head - tail);
        return diff >= 0 ? static_cast<size_t>(diff) : 0;
    }

    /// Check if queue is approximately empty.
    ///
    /// @return true if queue appears empty (may be stale).
    bool empty() const {
        return size() == 0;
    }

private:
    static constexpr size_t kCacheLineSize = 64;
    static constexpr size_t kIndexMask = Capacity - 1;

    struct Slot {
        std::atomic<uint64_t> seq;
        T data;
    };

    static inline void wait_hint(uint32_t& spins) {
        // Spin a bit (cheap), then use WFE on ARM for lower power.
        if (spins < 64) {
            astral::platform::cpu_pause();
        } else {
            astral::platform::cpu_wait_for_event();
        }

        if (spins < 1024) {
            ++spins;
        }
    }

    alignas(kCacheLineSize) std::atomic<uint64_t> enqueue_pos_;
    alignas(kCacheLineSize) std::atomic<uint64_t> dequeue_pos_;

    Slot buffer_[Capacity];
};

} // namespace astral::concurrency
