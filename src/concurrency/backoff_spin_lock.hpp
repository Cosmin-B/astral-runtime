#pragma once

#include "../platform/atomics.h"

#include <atomic>
#include <cstdint>
#include <thread>

namespace astral::concurrency {

class BackoffSpinLock {
public:
  BackoffSpinLock() noexcept = default;

  BackoffSpinLock(const BackoffSpinLock&) = delete;
  BackoffSpinLock& operator=(const BackoffSpinLock&) = delete;

  void lock() noexcept {
    if (state_.exchange(1u, std::memory_order_acquire) == 0u) {
      return;
    }

    // Observe before attempting ownership so waiters do not issue a locked
    // operation against a line that is visibly held.
    uint32_t pause_count = 1;
    for (;;) {
      for (uint32_t probe = 0; probe < kProbeCount; ++probe) {
        if (state_.load(std::memory_order_relaxed) != 0u) {
          continue;
        }

        if (state_.exchange(1u, std::memory_order_acquire) == 0u) {
          return;
        }
        break;
      }

      if (pause_count <= kMaxPauseCount) {
        for (uint32_t pause = 0; pause < pause_count; ++pause) {
          astral::platform::cpu_pause();
        }
        pause_count <<= 1u;
      } else {
        std::this_thread::yield();
      }
    }
  }

  bool try_lock() noexcept {
    return state_.exchange(1u, std::memory_order_acquire) == 0u;
  }

  void unlock() noexcept { state_.store(0u, std::memory_order_release); }

  void reset() noexcept { state_.store(0u, std::memory_order_relaxed); }

private:
  static constexpr uint32_t kProbeCount = 8;
  static constexpr uint32_t kMaxPauseCount = 1024;

  std::atomic<uint32_t> state_{0};
};

class BackoffSpinLockGuard {
public:
  explicit BackoffSpinLockGuard(BackoffSpinLock& lock) noexcept : lock_(lock) { lock_.lock(); }

  ~BackoffSpinLockGuard() { lock_.unlock(); }

  BackoffSpinLockGuard(const BackoffSpinLockGuard&) = delete;
  BackoffSpinLockGuard& operator=(const BackoffSpinLockGuard&) = delete;

private:
  BackoffSpinLock& lock_;
};

} // namespace astral::concurrency
