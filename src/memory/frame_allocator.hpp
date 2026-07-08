#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "../platform/compiler.hpp"

namespace astral::memory {

/// FrameAllocator - Linear/bump allocator for per-frame or per-session short-lived objects
///
/// Design:
/// - Bump pointer allocation over pre-committed memory region
/// - Fast O(1) allocation: just increment offset with alignment
/// - Reset invalidates all allocations instantly (no per-object tracking)
/// - NO vm_commit in hot path - all memory must be pre-committed at construction
///
/// Thread-safety: NOT thread-safe
/// - One allocator per thread/session
/// - No synchronization; designed for single-threaded use
///
/// Memory discipline:
/// - Pre-commit strategy: Caller must commit memory before passing to constructor
/// - No dynamic allocation in alloc() hot path
/// - No syscalls during allocation
///
/// Usage:
///   void* memory = vm_reserve(capacity);
///   if (memory == nullptr || !vm_commit(memory, capacity)) return;
///   FrameAllocator alloc(memory, capacity);
///   void* p1 = alloc.alloc(64);
///   void* p2 = alloc.alloc(128, 32); // 32-byte aligned
///   alloc.reset(); // All allocations invalidated
///   void* p3 = alloc.alloc(64); // Same address as p1
class FrameAllocator {
public:
    /// Initialize allocator with pre-committed memory region
    ///
    /// @param base Pre-committed memory base address (must not be null)
    /// @param capacity Total capacity in bytes (must be > 0)
    ///
    /// Memory pointed to by `base` MUST be already committed.
    /// Passing reserved-but-not-committed memory will cause page faults on first alloc().
    /// Use platform vm_commit() before constructing FrameAllocator.
    FrameAllocator(void* base, size_t capacity)
        : base_(static_cast<uint8_t*>(base))
        , capacity_(capacity)
        , used_(0) {
        // Validate inputs (debug builds)
        // Release builds skip validation for performance
#ifndef NDEBUG
        if (base == nullptr || capacity == 0) {
            // In production code, this would trigger an assertion
            // For now, set to invalid state
            base_ = nullptr;
            capacity_ = 0;
        }
#endif
    }

    /// Allocate memory with specified alignment
    ///
    /// @param size Number of bytes to allocate
    /// @param align Alignment requirement (must be power of 2, default 16)
    /// @return Pointer to allocated memory, or nullptr if out of capacity
    ///
    /// Performance: O(1) - just align offset and increment
    /// No syscalls, no locks, no dynamic allocation
    void* alloc(size_t size, size_t align = 16) {
        const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base_);
        const std::uintptr_t current_addr = base_addr + used_;
        const std::uintptr_t aligned_addr = (current_addr + align - 1) & ~(static_cast<std::uintptr_t>(align) - 1);
        size_t aligned_offset = static_cast<size_t>(aligned_addr - base_addr);
        size_t new_used = aligned_offset + size;

        if (new_used > capacity_) ASTRAL_UNLIKELY {
            return nullptr;
        }

        used_ = new_used;
        return base_ + aligned_offset;
    }

    /// Reset allocator to initial state
    ///
    /// All previous allocations are invalidated immediately.
    /// No per-object tracking; just reset the bump pointer.
    ///
    /// Performance: O(1) - single store operation
    void reset() {
        used_ = 0;
    }

    /// Query total capacity
    size_t capacity() const {
        return capacity_;
    }

    /// Query current usage
    size_t used() const {
        return used_;
    }

    /// Query remaining capacity
    size_t available() const {
        return capacity_ - used_;
    }

private:
    uint8_t* base_;      // Base address of committed memory
    size_t capacity_;    // Total capacity in bytes
    size_t used_;        // Current offset (bump pointer)
};

// Static assertions for FrameAllocator properties
static_assert(std::is_trivially_destructible_v<FrameAllocator>,
              "FrameAllocator must be trivially destructible (no cleanup needed)");

static_assert(sizeof(FrameAllocator) <= 32,
              "FrameAllocator should be small (fits in single cache line with padding)");

} // namespace astral::memory
