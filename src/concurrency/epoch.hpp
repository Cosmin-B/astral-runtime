#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../platform/atomics.h"
#include "../platform/cacheline.hpp"
#include "../platform/compiler.hpp"

namespace astral::concurrency {

/// Epoch-based reclamation with one fixed retirement queue per participant.
///
/// Reader entry writes only the caller's cache-line-isolated participant slot.
/// Retirement is SPSC: the registered participant is the sole producer for its
/// queue and collect() is the sole consumer. A full queue returns false and
/// leaves ownership with the caller; protected memory is never freed as an
/// overflow fallback.
///
/// Exactly one thread may call collect(). Calls that overlap an active collector
/// return without doing work. Registration is serialized because it is a cold
/// lifecycle operation; enter(), leave(), and defer_delete() take no locks.
template <size_t MaxThreads = 128, size_t MaxRetiredPerParticipant = 256> class EpochManager {
public:
  using Deleter = void (*)(void*) noexcept;

  static constexpr size_t kMaxThreads = MaxThreads;
  static constexpr size_t kMaxRetiredPerParticipant = MaxRetiredPerParticipant;
  static constexpr uint64_t kGraceEpochs = 3;

  EpochManager() noexcept : global_epoch_(0) {
    for (Participant& participant : participants_) {
      participant.epoch.store(kInactiveEpoch, std::memory_order_relaxed);
      participant.registered.store(false, std::memory_order_relaxed);
      participant.retired.head.store(0, std::memory_order_relaxed);
      participant.retired.tail.store(0, std::memory_order_relaxed);
    }
  }

  ~EpochManager() {
    // Shutdown requires all participants and the collector to have stopped.
    for (Participant& participant : participants_) {
      drain_all(participant.retired);
    }
  }

  EpochManager(const EpochManager&) = delete;
  EpochManager& operator=(const EpochManager&) = delete;
  EpochManager(EpochManager&&) = delete;
  EpochManager& operator=(EpochManager&&) = delete;

  /// Claim a reusable participant slot. Registration is a cold-path operation.
  int32_t register_thread() noexcept {
    lock_registration();
    int32_t result = -1;
    for (size_t i = 0; i < kMaxThreads; ++i) {
      Participant& participant = participants_[i];
      if (participant.registered.load(std::memory_order_relaxed)) {
        continue;
      }

      participant.epoch.store(kInactiveEpoch, std::memory_order_relaxed);
      participant.registered.store(true, std::memory_order_release);
      result = static_cast<int32_t>(i);
      break;
    }
    unlock_registration();
    return result;
  }

  /// Release a participant slot after its final leave() and defer_delete().
  void unregister_thread(int32_t thread_id) noexcept {
    Participant* participant = get_participant(thread_id);
    if (participant == nullptr)
      ASTRAL_UNLIKELY {
        return;
      }

    participant->epoch.store(kInactiveEpoch, std::memory_order_seq_cst);
    participant->registered.store(false, std::memory_order_release);
  }

  /// Enter a read-side critical section before loading a shared pointer.
  void enter(int32_t thread_id) noexcept {
    Participant* participant = get_registered_participant(thread_id);
    if (participant == nullptr)
      ASTRAL_UNLIKELY {
        return;
      }

    // The second global load closes the race with collect()'s seq_cst epoch
    // publication. If publication won, retry before the caller can load a
    // protected pointer. Otherwise the collector's later scan observes this
    // participant epoch or conservatively delays reclamation.
    uint64_t epoch = 0;
    do {
      epoch = global_epoch_.load(std::memory_order_seq_cst);
      participant->epoch.store(epoch, std::memory_order_seq_cst);
    } while (global_epoch_.load(std::memory_order_seq_cst) != epoch);
  }

  /// Leave the current read-side critical section.
  void leave(int32_t thread_id) noexcept {
    Participant* participant = get_participant(thread_id);
    if (participant == nullptr)
      ASTRAL_UNLIKELY {
        return;
      }
    participant->epoch.store(kInactiveEpoch, std::memory_order_release);
  }

  /// Queue a pointer for deferred destruction by the single collector.
  ///
  /// Returns false without invoking deleter when the participant is invalid,
  /// unregistered, or its fixed queue is full. On failure, the caller still
  /// owns ptr and may retry after collection advances.
  bool defer_delete(int32_t thread_id, void* ptr, Deleter deleter) noexcept {
    if (ptr == nullptr || deleter == nullptr)
      ASTRAL_UNLIKELY {
        return false;
      }

    Participant* participant = get_registered_participant(thread_id);
    if (participant == nullptr)
      ASTRAL_UNLIKELY {
        return false;
      }

    RetireQueue& queue = participant->retired;
    const uint64_t head = queue.head.load(std::memory_order_relaxed);
    const uint64_t tail = queue.tail.load(std::memory_order_acquire);
    if (head - tail >= kMaxRetiredPerParticipant)
      ASTRAL_UNLIKELY {
        return false;
      }

    RetiredEntry& entry = queue.entries[head & kRetiredMask];
    entry.ptr = ptr;
    entry.deleter = deleter;
    entry.epoch = global_epoch_.load(std::memory_order_acquire);
    queue.head.store(head + 1, std::memory_order_release);
    return true;
  }

  template <typename T> bool defer_delete(int32_t thread_id, T* ptr) noexcept {
    static_assert(!std::is_void_v<T>);
    return defer_delete(thread_id, ptr, &delete_typed<T>);
  }

  /// Reclaim the previously proven safe frontier, then establish the next one.
  ///
  /// The safe frontier is intentionally consumed before the global epoch is
  /// advanced. The seq_cst increment and participant scan prove a frontier for
  /// the next collect() call, so a retirement concurrent with publication can
  /// never be consumed by the drain already in progress.
  size_t collect() noexcept {
    if (collecting_.test_and_set(std::memory_order_acquire))
      ASTRAL_UNLIKELY {
        return 0;
      }

    size_t reclaimed = 0;
    if (has_safe_epoch_) {
      for (Participant& participant : participants_) {
        reclaimed += drain_through(participant.retired, safe_epoch_);
      }
    }

    const uint64_t new_epoch = global_epoch_.fetch_add(1, std::memory_order_seq_cst) + 1;
    update_safe_frontier(new_epoch);

    collecting_.clear(std::memory_order_release);
    return reclaimed;
  }

  uint64_t current_epoch() const noexcept { return global_epoch_.load(std::memory_order_acquire); }

private:
  static constexpr size_t kCacheLineSize = astral::platform::kCacheLineAlign;
  static constexpr size_t kRetiredMask = kMaxRetiredPerParticipant - 1;
  static constexpr uint64_t kInactiveEpoch = UINT64_MAX;

  static_assert(kMaxThreads > 0, "epoch manager requires at least one participant");
  static_assert(kMaxRetiredPerParticipant > 0,
                "epoch manager requires a non-empty retirement queue");
  static_assert((kMaxRetiredPerParticipant & kRetiredMask) == 0,
                "retirement queue capacity must be a power of two");

  struct RetiredEntry {
    void* ptr = nullptr;
    Deleter deleter = nullptr;
    uint64_t epoch = 0;
  };

  struct RetireQueue {
    alignas(kCacheLineSize) std::atomic<uint64_t> head{0};
    alignas(kCacheLineSize) std::atomic<uint64_t> tail{0};
    RetiredEntry entries[kMaxRetiredPerParticipant]{};
  };

  struct alignas(kCacheLineSize) Participant {
    std::atomic<uint64_t> epoch{kInactiveEpoch};
    std::atomic<bool> registered{false};
    RetireQueue retired;
  };

  static_assert(alignof(Participant) >= kCacheLineSize);
  static_assert(sizeof(Participant) % kCacheLineSize == 0);

  template <typename T> static void delete_typed(void* value) noexcept {
    delete static_cast<T*>(value);
  }

  Participant* get_participant(int32_t thread_id) noexcept {
    if (thread_id < 0 || thread_id >= static_cast<int32_t>(kMaxThreads))
      ASTRAL_UNLIKELY {
        return nullptr;
      }
    return &participants_[static_cast<size_t>(thread_id)];
  }

  Participant* get_registered_participant(int32_t thread_id) noexcept {
    Participant* participant = get_participant(thread_id);
    if (participant == nullptr || !participant->registered.load(std::memory_order_acquire))
      ASTRAL_UNLIKELY {
        return nullptr;
      }
    return participant;
  }

  void lock_registration() noexcept {
    while (registration_lock_.test_and_set(std::memory_order_acquire)) {
      astral::platform::cpu_pause();
    }
  }

  void unlock_registration() noexcept {
    registration_lock_.clear(std::memory_order_release);
    astral::platform::cpu_signal_event();
  }

  static size_t drain_through(RetireQueue& queue, uint64_t safe_epoch) noexcept {
    const uint64_t head = queue.head.load(std::memory_order_acquire);
    uint64_t tail = queue.tail.load(std::memory_order_relaxed);
    size_t reclaimed = 0;

    while (tail != head) {
      RetiredEntry& entry = queue.entries[tail & kRetiredMask];
      if (entry.epoch > safe_epoch) {
        break;
      }

      void* ptr = entry.ptr;
      Deleter deleter = entry.deleter;
      entry = RetiredEntry{};
      ++tail;
      queue.tail.store(tail, std::memory_order_release);
      deleter(ptr);
      ++reclaimed;
    }
    return reclaimed;
  }

  static void drain_all(RetireQueue& queue) noexcept {
    const uint64_t head = queue.head.load(std::memory_order_acquire);
    uint64_t tail = queue.tail.load(std::memory_order_relaxed);
    while (tail != head) {
      RetiredEntry& entry = queue.entries[tail & kRetiredMask];
      void* ptr = entry.ptr;
      Deleter deleter = entry.deleter;
      entry = RetiredEntry{};
      ++tail;
      if (ptr != nullptr && deleter != nullptr) {
        deleter(ptr);
      }
    }
    queue.tail.store(head, std::memory_order_relaxed);
  }

  void update_safe_frontier(uint64_t new_epoch) noexcept {
    if (new_epoch < kGraceEpochs) {
      return;
    }

    uint64_t candidate = new_epoch - kGraceEpochs;
    for (Participant& participant : participants_) {
      if (!participant.registered.load(std::memory_order_acquire)) {
        continue;
      }

      const uint64_t participant_epoch = participant.epoch.load(std::memory_order_acquire);
      if (participant_epoch == kInactiveEpoch || participant_epoch > candidate) {
        continue;
      }
      if (participant_epoch == 0) {
        return;
      }
      candidate = participant_epoch - 1;
    }

    if (!has_safe_epoch_ || candidate > safe_epoch_) {
      safe_epoch_ = candidate;
      has_safe_epoch_ = true;
    }
  }

  alignas(kCacheLineSize) std::atomic<uint64_t> global_epoch_;
  alignas(kCacheLineSize) std::atomic_flag registration_lock_ = ATOMIC_FLAG_INIT;
  alignas(kCacheLineSize) std::atomic_flag collecting_ = ATOMIC_FLAG_INIT;
  Participant participants_[kMaxThreads];

  // Single-collector state; protected by collecting_.
  uint64_t safe_epoch_ = 0;
  bool has_safe_epoch_ = false;
};

/// RAII guard for one epoch-protected read-side critical section.
template <typename Manager> class EpochGuard {
public:
  EpochGuard(Manager& manager, int32_t thread_id) noexcept
      : manager_(manager), thread_id_(thread_id) {
    manager_.enter(thread_id_);
  }

  ~EpochGuard() { manager_.leave(thread_id_); }

  EpochGuard(const EpochGuard&) = delete;
  EpochGuard& operator=(const EpochGuard&) = delete;
  EpochGuard(EpochGuard&&) = delete;
  EpochGuard& operator=(EpochGuard&&) = delete;

private:
  Manager& manager_;
  int32_t thread_id_;
};

template <typename Manager> EpochGuard(Manager&, int32_t) -> EpochGuard<Manager>;

} // namespace astral::concurrency
