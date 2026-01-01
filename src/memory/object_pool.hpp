#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "../platform/atomics.h"
#include "../platform/compiler.hpp"

namespace astral::memory {

inline constexpr size_t kObjectPoolCacheLineBytes = 64;
inline constexpr size_t kObjectPoolPointerBitsForCounter = 24;
inline constexpr size_t kObjectPoolMaxObjects = size_t{1} << kObjectPoolPointerBitsForCounter;

/// ObjectPool - Thread-safe object pool using intrusive freelist
///
/// Design:
/// - Thread-safe acquire/release protected by a tiny spinlock (no CAS usage)
/// - Intrusive freelist: next pointer stored in freed object's first 8 bytes
/// - Fixed capacity determined at compile time
/// - Pre-allocated backing storage (no dynamic allocation)
///
/// Thread-safety: Thread-safe
/// - Multiple threads can acquire/release concurrently
///
/// Usage:
///   ObjectPool<Token, 1024> pool;
///   Token* tok = pool.acquire();
///   if (tok) {
///       // Use token
///       pool.release(tok);
///   }
///
/// @tparam T Object type (must be at least sizeof(void*) bytes)
/// @tparam MaxObjects Maximum number of objects in pool
template<typename T, size_t MaxObjects>
class ObjectPool {
public:
    // Ensure T is large enough to store next pointer in freed object
    static_assert(sizeof(T) >= sizeof(void*),
                  "Object type must be at least pointer-sized for intrusive freelist");

    static_assert(alignof(T) >= alignof(void*),
                  "Object type alignment must be at least pointer alignment");

    static_assert(MaxObjects > 0,
                  "Pool must have at least one object");

    static_assert(MaxObjects <= kObjectPoolMaxObjects,
                  "Pool size limited to 16M objects (ABA counter uses upper bits)");

    /// Constructor - initializes pool with all objects in freelist
    ObjectPool() {
        // Initialize freelist: all objects linked together
        // Last object points to nullptr
        for (size_t i = 0; i < MaxObjects; ++i) {
            T* obj = &storage_[i];
            void** next_ptr = reinterpret_cast<void**>(obj);

            if (i < MaxObjects - 1) {
                *next_ptr = &storage_[i + 1]; // Link to next
            } else {
                *next_ptr = nullptr; // End of list
            }
        }

        // Head points to first free object.
        head_ = &storage_[0];
    }

    /// Acquire an object from the pool.
    ///
    /// @return Pointer to object, or nullptr if pool is empty
    T* acquire() {
        lock_();

        T* obj = head_;
        if (obj != nullptr) {
            void** next_ptr = reinterpret_cast<void**>(obj);
            head_ = static_cast<T*>(*next_ptr);
        }

        unlock_();
        return obj;
    }

    /// Release an object back to the pool.
    ///
    /// @param obj Pointer to object previously acquired from this pool
    ///
    ///  obj must have been acquired from this pool
    /// Releasing foreign pointers causes undefined behavior
    void release(T* obj) {
        if (obj == nullptr) ASTRAL_UNLIKELY {
            return; // Null release is no-op
        }

        lock_();

        void** next_ptr = reinterpret_cast<void**>(obj);
        *next_ptr = head_;
        head_ = obj;

        unlock_();
    }

    /// Query capacity
    static constexpr size_t capacity() {
        return MaxObjects;
    }

private:
    // Backing storage for objects (cache-line aligned)
    alignas(kObjectPoolCacheLineBytes) T storage_[MaxObjects];

    // Freelist head (protected by table_lock_)
    T* head_ = nullptr;

    std::atomic_flag table_lock_ = ATOMIC_FLAG_INIT;

    void lock_() {
        uint32_t spins = 0;
        while (table_lock_.test_and_set(std::memory_order_acquire)) {
            if (spins < 64) {
                astral::platform::cpu_pause();
            } else {
                astral::platform::cpu_wait_for_event();
            }
            if (spins < 1024) {
                ++spins;
            }
        }
    }

    void unlock_() {
        table_lock_.clear(std::memory_order_release);
        astral::platform::cpu_signal_event();
    }
};

template<typename T, size_t MaxObjects>
class LocalObjectPool {
public:
    static_assert(sizeof(T) >= sizeof(void*),
                  "Object type must be at least pointer-sized for intrusive freelist");

    static_assert(alignof(T) >= alignof(void*),
                  "Object type alignment must be at least pointer alignment");

    static_assert(MaxObjects > 0,
                  "Pool must have at least one object");

    LocalObjectPool() noexcept {
        for (size_t i = 0; i < MaxObjects; ++i) {
            T* obj = &storage_[i];
            void** next_ptr = reinterpret_cast<void**>(obj);

            if (i + kNextObjectStep < MaxObjects) {
                *next_ptr = &storage_[i + kNextObjectStep];
            } else {
                *next_ptr = nullptr;
            }
        }

        head_ = &storage_[0];
    }

    T* acquire() noexcept {
        T* obj = head_;
        if (obj == nullptr) ASTRAL_UNLIKELY {
            return nullptr;
        }

        void** next_ptr = reinterpret_cast<void**>(obj);
        head_ = static_cast<T*>(*next_ptr);
        return obj;
    }

    void release(T* obj) noexcept {
        void** next_ptr = reinterpret_cast<void**>(obj);
        *next_ptr = head_;
        head_ = obj;
    }

    static constexpr size_t capacity() noexcept {
        return MaxObjects;
    }

private:
    static constexpr size_t kNextObjectStep = 1;

    alignas(kObjectPoolCacheLineBytes) T storage_[MaxObjects];
    T* head_ = nullptr;
};

} // namespace astral::memory
