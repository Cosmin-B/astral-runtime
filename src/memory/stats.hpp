#pragma once

#include <cstdint>

namespace astral::memory {

/// Memory statistics for tracking allocator usage
///
/// Design:
/// - POD struct for zero-cost abstraction
/// - Thread-safe when updated via atomic operations
/// - Used for debugging, profiling, and telemetry
///
/// Usage:
///   MemoryStats stats = {};
///   stats.bytes_reserved = 2 * 1024 * 1024 * 1024; // 2GB
///   stats.bytes_committed = 4 * 1024 * 1024;        // 4MB
///   stats.bytes_used = 1 * 1024 * 1024;             // 1MB
///   stats.alloc_count++;
struct MemoryStats {
    /// Total bytes reserved from OS (virtual address space)
    ///
    /// On platforms with virtual memory:
    /// - Linux: mmap with MAP_NORESERVE
    /// - Windows: VirtualAlloc with MEM_RESERVE
    /// - macOS: vm_allocate
    ///
    /// Reserved memory consumes address space but no physical memory.
    uint64_t bytes_reserved;

    /// Total bytes committed (backed by physical memory or swap)
    ///
    /// Committed memory is accessible without page faults.
    /// bytes_committed <= bytes_reserved
    ///
    /// On platforms with virtual memory:
    /// - Linux: madvise(MADV_WILLNEED) or mmap(MAP_POPULATE)
    /// - Windows: VirtualAlloc with MEM_COMMIT
    /// - macOS: Pages committed on first access
    uint64_t bytes_committed;

    /// Total bytes actually used by allocations
    ///
    /// bytes_used <= bytes_committed
    ///
    /// This tracks application-requested memory, not internal overhead.
    /// Waste = bytes_committed - bytes_used
    uint64_t bytes_used;

    /// Total number of allocations performed
    ///
    /// Incremented on each alloc/acquire operation.
    /// For debugging allocation patterns.
    uint64_t alloc_count;

    /// Total number of deallocations performed
    ///
    /// Incremented on each free/release operation.
    /// alloc_count - free_count = live allocations
    uint64_t free_count;
};

// Static assertions for MemoryStats properties
static_assert(sizeof(MemoryStats) == 40,
              "MemoryStats should be 40 bytes (fits in single cache line)");

static_assert(alignof(MemoryStats) == 8,
              "MemoryStats should be 8-byte aligned for atomic operations");

// Verify MemoryStats is trivially copyable (required for memcpy optimization)
static_assert(std::is_trivially_copyable_v<MemoryStats>,
              "MemoryStats must be trivially copyable for bulk operations");

// Verify MemoryStats is standard layout (required for C ABI compatibility)
static_assert(std::is_standard_layout_v<MemoryStats>,
              "MemoryStats must be standard layout for C ABI compatibility");

} // namespace astral::memory
