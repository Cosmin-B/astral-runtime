#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../platform/atomics.h"
#include "../platform/cacheline.hpp"
#include "../utils/trace.hpp"

namespace astral::concurrency {

/// Bounded multi-producer single-consumer ticket ring.
///
/// Producers reserve slots with one unconditional `fetch_add`. On x86 this
/// avoids a CAS retry branch and maps to one locked RMW. Producers wait for
/// their reserved slot to become reusable, so this primitive is for internal
/// backpressured fan-in, not public `ASTRAL_E_BUSY` API boundaries.
template<typename T, size_t Capacity>
class MpscTicketRing {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

    MpscTicketRing() : enqueue_pos_(0), dequeue_pos_(0) {
        for (uint64_t i = 0; i < Capacity; ++i) {
            buffer_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    MpscTicketRing(const MpscTicketRing&) = delete;
    MpscTicketRing& operator=(const MpscTicketRing&) = delete;
    MpscTicketRing(MpscTicketRing&&) = delete;
    MpscTicketRing& operator=(MpscTicketRing&&) = delete;

    void push_wait(const T& item) {
        ASTRAL_ZONE_MICRO_N("astral.mpsc_ticket.push_wait");

        const uint64_t pos = enqueue_pos_.fetch_add(1, std::memory_order_relaxed);
        Slot& slot = buffer_[pos & kIndexMask];

        uint32_t spins = 0;
        for (;;) {
            if (slot.seq.load(std::memory_order_acquire) == pos) {
                break;
            }
            wait_hint(spins);
        }

        slot.data = item;
        slot.seq.store(pos + 1, std::memory_order_release);
        astral::platform::cpu_signal_event();
    }

    bool pop(T& out) {
        ASTRAL_ZONE_MICRO_N("astral.mpsc_ticket.pop");

        const uint64_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        Slot& slot = buffer_[pos & kIndexMask];
        if (slot.seq.load(std::memory_order_acquire) != pos + 1) {
            return false;
        }

        out = slot.data;
        slot.seq.store(pos + Capacity, std::memory_order_release);
        dequeue_pos_.store(pos + 1, std::memory_order_release);
        astral::platform::cpu_signal_event();
        return true;
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

    static inline void wait_hint(uint32_t& spins) {
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
