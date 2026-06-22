#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../platform/atomics.h"
#include "../platform/cacheline.hpp"
#include "../utils/trace.hpp"

namespace astral::concurrency {

/// Bounded multi-producer single-consumer ring.
///
/// Use this when multiple runtime threads publish events or work to one owner:
/// async completion, tool-call fan-in, ingest queues, and progress events.
///
/// The queue is fixed-capacity and allocation-free. `try_push` returns false
/// when the ring is full; public C ABI callers should map that case to
/// `ASTRAL_E_BUSY`.
///
/// Thread-safety:
/// - `try_push` may be called by multiple producer threads.
/// - `pop` must be called by one consumer thread.
template<typename T, size_t Capacity>
class MpscRing {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

    MpscRing() : enqueue_pos_(0), dequeue_pos_(0) {
        for (uint64_t i = 0; i < Capacity; ++i) {
            buffer_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    MpscRing(const MpscRing&) = delete;
    MpscRing& operator=(const MpscRing&) = delete;
    MpscRing(MpscRing&&) = delete;
    MpscRing& operator=(MpscRing&&) = delete;

    bool try_push(const T& item) {
        ASTRAL_ZONE_MICRO_N("astral.mpsc.try_push");

        uint64_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = buffer_[pos & kIndexMask];
            const uint64_t seq = slot.seq.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
                    slot.data = item;
                    slot.seq.store(pos + 1, std::memory_order_release);
                    astral::platform::cpu_signal_event();
                    return true;
                }
                continue;
            }

            if (diff < 0) {
                return false;
            }

            pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
    }

    bool pop(T& out) {
        ASTRAL_ZONE_MICRO_N("astral.mpsc.pop");

        const uint64_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        Slot& slot = buffer_[pos & kIndexMask];
        const uint64_t seq = slot.seq.load(std::memory_order_acquire);
        const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

        if (diff != 0) {
            return false;
        }

        out = slot.data;
        slot.seq.store(pos + Capacity, std::memory_order_release);
        dequeue_pos_.store(pos + 1, std::memory_order_release);
        astral::platform::cpu_signal_event();
        return true;
    }

    size_t size() const {
        const uint64_t head = enqueue_pos_.load(std::memory_order_relaxed);
        const uint64_t tail = dequeue_pos_.load(std::memory_order_relaxed);
        const int64_t diff = static_cast<int64_t>(head - tail);
        if (diff <= 0) {
            return 0;
        }
        const size_t n = static_cast<size_t>(diff);
        return n > Capacity ? Capacity : n;
    }

    bool empty() const {
        return size() == 0;
    }

    constexpr size_t capacity() const {
        return Capacity;
    }

    void reset() {
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
        for (uint64_t i = 0; i < Capacity; ++i) {
            buffer_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

private:
    static constexpr size_t kCacheLineSize = astral::platform::kCacheLineAlign;
    static constexpr size_t kIndexMask = Capacity - 1;

    struct Slot {
        std::atomic<uint64_t> seq;
        T data;
    };

    alignas(kCacheLineSize) std::atomic<uint64_t> enqueue_pos_;
    alignas(kCacheLineSize) std::atomic<uint64_t> dequeue_pos_;

    Slot buffer_[Capacity];
};

} // namespace astral::concurrency
