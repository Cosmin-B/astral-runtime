#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../platform/cacheline.hpp"

namespace astral::concurrency {

/// Epoch-based memory reclamation for lock-free data structures.
///
/// Design:
/// - Global epoch counter incremented periodically by reclamation thread
/// - Each thread tracks its local epoch when accessing shared data structures
/// - Defer deletion of nodes until all threads have advanced past node's epoch
/// - No per-object hazard pointers (lower overhead for small objects)
///
/// Use Cases:
/// - Free transient nodes (callback entries, work items) without hazard pointers
/// - Safe deferred deletion in MPMC/SPSC rings
/// - Avoid use-after-free when nodes are retired but may still be accessed
///
/// Memory Ordering:
/// - Epoch loads: memory_order_acquire (synchronize-with epoch updates)
/// - Epoch stores: memory_order_release (publish local epoch to reclamation thread)
/// - Thread registration: memory_order_seq_cst (ensure visibility across all threads)
///
/// Performance:
/// - Low overhead: Single atomic load per critical section entry
/// - Bounded latency: Nodes freed within 3 epochs (~3 reclamation cycles)
///
/// Thread-safety: Thread-safe; safe to call from multiple threads concurrently.
class EpochManager {
public:
    /// Maximum number of threads that can register with the epoch manager.
    /// This is a fixed-size limit to avoid dynamic allocation.
    static constexpr size_t kMaxThreads = 128;

    /// Maximum number of deferred deletions per epoch.
    /// Fixed ring buffer to avoid allocation in defer_delete().
    static constexpr size_t kMaxDeferredPerEpoch = 4096;

    /// Number of epochs to keep before reclamation.
    /// Nodes are safe to free after all threads have advanced 3 epochs.
    static constexpr size_t kEpochHistorySize = 3;

    EpochManager() : global_epoch_(0), thread_count_(0) {
        // Initialize thread-local epochs to 0
        for (size_t i = 0; i < kMaxThreads; ++i) {
            thread_epochs_[i].store(0, std::memory_order_relaxed);
            thread_active_[i].store(false, std::memory_order_relaxed);
        }

        // Initialize deferred deletion rings
        for (size_t i = 0; i < kEpochHistorySize; ++i) {
            epoch_rings_[i].head = 0;
            epoch_rings_[i].tail = 0;
            std::memset(epoch_rings_[i].ptrs, 0, sizeof(epoch_rings_[i].ptrs));
        }
    }

    ~EpochManager() {
        // Reclaim all remaining deferred deletions on shutdown
        // WARNING: This is not thread-safe; must be called after all threads exit
        for (size_t i = 0; i < kEpochHistorySize; ++i) {
            DeferredRing& ring = epoch_rings_[i];
            for (size_t j = ring.tail; j < ring.head; ++j) {
                size_t index = j & kDeferredMask;
                void* ptr = ring.ptrs[index];
                if (ptr != nullptr) {
                    ::operator delete(ptr);
                }
            }
        }
    }

    // Non-copyable, non-movable
    EpochManager(const EpochManager&) = delete;
    EpochManager& operator=(const EpochManager&) = delete;
    EpochManager(EpochManager&&) = delete;
    EpochManager& operator=(EpochManager&&) = delete;

    /// Register the current thread with the epoch manager.
    ///
    /// @return Thread-local ID (0 to kMaxThreads-1), or -1 if registration failed.
    ///
    /// IMPORTANT: Must be called once per thread before calling enter/leave.
    /// The returned ID must be passed to enter/leave/unregister.
    int32_t register_thread() {
        uint32_t id = thread_count_.fetch_add(1, std::memory_order_seq_cst);
        if (id >= kMaxThreads) [[unlikely]] {
            // Registration failed: too many threads
            thread_count_.fetch_sub(1, std::memory_order_seq_cst);
            return -1;
        }

        thread_active_[id].store(true, std::memory_order_release);
        thread_epochs_[id].store(global_epoch_.load(std::memory_order_acquire),
                                 std::memory_order_release);
        return static_cast<int32_t>(id);
    }

    /// Unregister the current thread from the epoch manager.
    ///
    /// @param thread_id Thread-local ID returned by register_thread().
    void unregister_thread(int32_t thread_id) {
        if (thread_id < 0 || thread_id >= static_cast<int32_t>(kMaxThreads)) [[unlikely]] {
            return;
        }

        thread_active_[thread_id].store(false, std::memory_order_release);
    }

    /// Enter an epoch-protected critical section.
    ///
    /// @param thread_id Thread-local ID returned by register_thread().
    ///
    /// Call this before accessing shared lock-free data structures.
    /// Must be paired with leave() at the end of the critical section.
    void enter(int32_t thread_id) {
        if (thread_id < 0 || thread_id >= static_cast<int32_t>(kMaxThreads)) [[unlikely]] {
            return;
        }

        // Load global epoch with acquire semantics to see latest reclamation state
        uint64_t current_epoch = global_epoch_.load(std::memory_order_acquire);

        // Update thread-local epoch with release semantics to publish our entry
        thread_epochs_[thread_id].store(current_epoch, std::memory_order_release);
    }

    /// Leave an epoch-protected critical section.
    ///
    /// @param thread_id Thread-local ID returned by register_thread().
    ///
    /// Must be paired with enter() at the start of the critical section.
    void leave(int32_t thread_id) {
        if (thread_id < 0 || thread_id >= static_cast<int32_t>(kMaxThreads)) [[unlikely]] {
            return;
        }

        // Mark thread as not in critical section by advancing to a future epoch
        // Use UINT64_MAX as sentinel value for "not in critical section"
        thread_epochs_[thread_id].store(UINT64_MAX, std::memory_order_release);
    }

    /// Defer deletion of a pointer until all threads have advanced past current epoch.
    ///
    /// @param ptr Pointer to delete (must be allocated with ::operator new).
    ///
    /// The pointer will be freed after all threads have advanced 3 epochs.
    /// If the deferred deletion ring is full, the pointer is freed immediately.
    void defer_delete(void* ptr) {
        if (ptr == nullptr) [[unlikely]] {
            return;
        }

        // Get current epoch ring index
        uint64_t epoch = global_epoch_.load(std::memory_order_acquire);
        size_t ring_index = epoch % kEpochHistorySize;
        DeferredRing& ring = epoch_rings_[ring_index];

        // Try to add to deferred deletion ring
        uint64_t head = ring.head.load(std::memory_order_relaxed);
        uint64_t tail = ring.tail.load(std::memory_order_acquire);

        if (head - tail >= kMaxDeferredPerEpoch) [[unlikely]] {
            // Ring is full; free immediately (fallback)
            ::operator delete(ptr);
            return;
        }

        // Add to ring
        size_t index = head & kDeferredMask;
        ring.ptrs[index] = ptr;
        ring.head.store(head + 1, std::memory_order_release);
    }

    /// Advance the global epoch and collect garbage from old epochs.
    ///
    /// This should be called periodically by a dedicated reclamation thread.
    /// Typical frequency: 1-10ms (balance between latency and overhead).
    ///
    /// Memory ordering: memory_order_seq_cst to ensure all threads see epoch advance.
    void collect() {
        // Advance global epoch
        uint64_t old_epoch = global_epoch_.fetch_add(1, std::memory_order_seq_cst);
        uint64_t new_epoch = old_epoch + 1;

        // Find minimum thread-local epoch (oldest thread still in critical section)
        uint64_t min_epoch = new_epoch;
        uint32_t count = thread_count_.load(std::memory_order_acquire);

        for (uint32_t i = 0; i < count && i < kMaxThreads; ++i) {
            if (!thread_active_[i].load(std::memory_order_acquire)) {
                continue;
            }

            uint64_t thread_epoch = thread_epochs_[i].load(std::memory_order_acquire);
            if (thread_epoch != UINT64_MAX && thread_epoch < min_epoch) {
                min_epoch = thread_epoch;
            }
        }

        // Reclaim deferred deletions from epochs that are at least 3 epochs old
        if (new_epoch >= kEpochHistorySize) {
            uint64_t safe_epoch = new_epoch - kEpochHistorySize;
            if (min_epoch > safe_epoch) {
                // All threads have advanced past safe_epoch; safe to reclaim
                size_t ring_index = safe_epoch % kEpochHistorySize;
                DeferredRing& ring = epoch_rings_[ring_index];

                // Free all deferred deletions in this ring
                uint64_t head = ring.head.load(std::memory_order_acquire);
                uint64_t tail = ring.tail.load(std::memory_order_relaxed);

                for (uint64_t i = tail; i < head; ++i) {
                    size_t index = i & kDeferredMask;
                    void* ptr = ring.ptrs[index];
                    if (ptr != nullptr) {
                        ::operator delete(ptr);
                        ring.ptrs[index] = nullptr;
                    }
                }

                // Reset ring
                ring.tail.store(head, std::memory_order_release);
            }
        }
    }

    /// Get current global epoch.
    ///
    /// @return Current epoch value (monotonically increasing).
    uint64_t current_epoch() const {
        return global_epoch_.load(std::memory_order_acquire);
    }

private:
    static constexpr size_t kCacheLineSize = astral::platform::kCacheLineAlign;
    static constexpr size_t kDeferredMask = kMaxDeferredPerEpoch - 1;

    /// Ring buffer for deferred deletions per epoch.
    struct DeferredRing {
        alignas(kCacheLineSize) std::atomic<uint64_t> head;
        alignas(kCacheLineSize) std::atomic<uint64_t> tail;
        void* ptrs[kMaxDeferredPerEpoch];
    };

    // Global epoch counter (incremented by reclamation thread)
    alignas(kCacheLineSize) std::atomic<uint64_t> global_epoch_;

    // Thread registration counter
    alignas(kCacheLineSize) std::atomic<uint32_t> thread_count_;

    // Per-thread epoch tracking (cache-line aligned to prevent false sharing)
    alignas(kCacheLineSize) std::atomic<uint64_t> thread_epochs_[kMaxThreads];
    alignas(kCacheLineSize) std::atomic<bool> thread_active_[kMaxThreads];

    // Deferred deletion rings (one per epoch history slot)
    DeferredRing epoch_rings_[kEpochHistorySize];
};

/// RAII guard for epoch-protected critical sections.
///
/// Usage:
/// ```cpp
/// EpochManager epoch_mgr;
/// int32_t thread_id = epoch_mgr.register_thread();
///
/// {
///     EpochGuard guard(epoch_mgr, thread_id);
///     // Access lock-free data structures here
/// }
/// // Automatically calls leave() on scope exit
///
/// epoch_mgr.unregister_thread(thread_id);
/// ```
class EpochGuard {
public:
    EpochGuard(EpochManager& mgr, int32_t thread_id)
        : mgr_(mgr), thread_id_(thread_id) {
        mgr_.enter(thread_id_);
    }

    ~EpochGuard() {
        mgr_.leave(thread_id_);
    }

    // Non-copyable, non-movable
    EpochGuard(const EpochGuard&) = delete;
    EpochGuard& operator=(const EpochGuard&) = delete;
    EpochGuard(EpochGuard&&) = delete;
    EpochGuard& operator=(EpochGuard&&) = delete;

private:
    EpochManager& mgr_;
    int32_t thread_id_;
};

} // namespace astral::concurrency
