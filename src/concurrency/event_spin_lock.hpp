#pragma once

#include "../platform/atomics.h"

#include <atomic>
#include <cstdint>

namespace astral::concurrency {

class EventSpinLock {
public:
  EventSpinLock() noexcept = default;

  EventSpinLock(const EventSpinLock&) = delete;
  EventSpinLock& operator=(const EventSpinLock&) = delete;

  void lock() noexcept {
    uint32_t spins = 0;
    for (;;) {
      if (state_.exchange(1u, std::memory_order_acquire) == 0u) {
        return;
      }

      while (state_.load(std::memory_order_relaxed) != 0u) {
        if (spins < kActiveSpinCount) {
          astral::platform::cpu_pause();
        } else {
          astral::platform::cpu_wait_for_event();
        }
        if (spins < kMaxSpinCount) {
          ++spins;
        }
      }
    }
  }

  bool try_lock() noexcept { return state_.exchange(1u, std::memory_order_acquire) == 0u; }

  void unlock() noexcept {
    state_.store(0u, std::memory_order_release);
    astral::platform::cpu_signal_event();
  }

  void reset() noexcept { state_.store(0u, std::memory_order_relaxed); }

private:
  static constexpr uint32_t kActiveSpinCount = 64;
  static constexpr uint32_t kMaxSpinCount = 1024;

  std::atomic<uint32_t> state_{0};
};

class EventSpinLockGuard {
public:
  explicit EventSpinLockGuard(EventSpinLock& lock) noexcept : lock_(lock) { lock_.lock(); }

  ~EventSpinLockGuard() { lock_.unlock(); }

  EventSpinLockGuard(const EventSpinLockGuard&) = delete;
  EventSpinLockGuard& operator=(const EventSpinLockGuard&) = delete;

private:
  EventSpinLock& lock_;
};

} // namespace astral::concurrency
