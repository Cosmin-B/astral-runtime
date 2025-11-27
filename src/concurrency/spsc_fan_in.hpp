#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "spsc_ring.hpp"

namespace astral::concurrency {

/// Per-producer SPSC fan-in lanes.
///
/// This is the preferred fan-in topology when producer ownership is known:
/// every producer writes only its own SPSC lane and the owner consumer polls
/// lanes in round-robin order. Producers never contend with each other, so the
/// hot producer path avoids the shared CAS used by a general MPSC queue.
template<typename T, size_t Producers, size_t LaneCapacity>
class SpscFanIn {
public:
    static_assert(Producers > 0, "Producers must be greater than 0");
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

    SpscFanIn() = default;

    SpscFanIn(const SpscFanIn&) = delete;
    SpscFanIn& operator=(const SpscFanIn&) = delete;
    SpscFanIn(SpscFanIn&&) = delete;
    SpscFanIn& operator=(SpscFanIn&&) = delete;

    bool try_push(size_t producer_index, const T& item) {
        return lanes_[producer_index].push(item);
    }

    size_t push_batch(size_t producer_index, const T* items, size_t count) {
        return lanes_[producer_index].push_batch(items, count);
    }

    bool pop(T& out, size_t* producer_index = nullptr) {
        for (size_t attempt = 0; attempt < Producers; ++attempt) {
            const size_t index = (next_consumer_lane_ + attempt) % Producers;
            if (lanes_[index].pop(out)) {
                next_consumer_lane_ = (index + 1) % Producers;
                if (producer_index != nullptr) {
                    *producer_index = index;
                }
                return true;
            }
        }
        return false;
    }

    size_t pop_batch(T* out, size_t count, size_t* producer_index = nullptr) {
        for (size_t attempt = 0; attempt < Producers; ++attempt) {
            const size_t index = (next_consumer_lane_ + attempt) % Producers;
            const size_t n = lanes_[index].pop_batch(out, count);
            if (n != 0) {
                next_consumer_lane_ = (index + 1) % Producers;
                if (producer_index != nullptr) {
                    *producer_index = index;
                }
                return n;
            }
        }
        return 0;
    }

    void reset() {
        for (size_t i = 0; i < Producers; ++i) {
            lanes_[i].reset();
        }
        next_consumer_lane_ = 0;
    }

    static constexpr size_t producer_count() {
        return Producers;
    }

    static constexpr size_t lane_capacity() {
        return LaneCapacity;
    }

private:
    SpscRing<T, LaneCapacity> lanes_[Producers];
    size_t next_consumer_lane_ = 0;
};

} // namespace astral::concurrency
