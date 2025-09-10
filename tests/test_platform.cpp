/**
 * test_platform.cpp - Platform abstraction layer tests
 *
 * Tests for platform-specific virtual memory and atomics utilities.
 * Validates: VM reserve/commit/decommit/release, huge pages, cache line detection,
 * CPU pause, compiler fence, error handling.
 */

#include "test_framework.hpp"
#include "../src/platform/vm.h"
#include "../src/platform/atomics.h"

#include <cstring>
#include <cstdio>

using namespace astral::platform;

#if ASTRAL_ENABLE_VIRTUAL_MEMORY
// Test basic VM reserve/commit/decommit/release cycle
TEST(vm_reserve_commit_release) {
    constexpr size_t kSize = 4 * 1024 * 1024; // 4MB

    // Reserve address space
    void* addr = vm_reserve(kSize);
    ASSERT_NOT_NULL(addr);

    // Commit first 1MB
    constexpr size_t kCommitSize = 1 * 1024 * 1024;
    vm_commit(addr, kCommitSize);

    // Write to committed region (should not crash)
    memset(addr, 0xAB, kCommitSize);

    // Verify write
    auto* bytes = static_cast<unsigned char*>(addr);
    ASSERT_EQ(bytes[0], 0xAB);
    ASSERT_EQ(bytes[kCommitSize - 1], 0xAB);

    // Decommit
    vm_decommit(addr, kCommitSize);

    // Release entire region
    vm_release(addr, kSize);
}

// Test multiple commit operations on same region
TEST(vm_commit_incremental) {
    constexpr size_t kSize = 8 * 1024 * 1024; // 8MB
    void* addr = vm_reserve(kSize);
    ASSERT_NOT_NULL(addr);

    // Commit in 1MB chunks
    constexpr size_t kChunkSize = 1 * 1024 * 1024;
    for (size_t i = 0; i < 4; ++i) {
        void* chunk = static_cast<char*>(addr) + (i * kChunkSize);
        vm_commit(chunk, kChunkSize);

        // Write to chunk
        memset(chunk, static_cast<int>(i), kChunkSize);
    }

    // Verify all chunks
    auto* bytes = static_cast<unsigned char*>(addr);
    for (size_t i = 0; i < 4; ++i) {
        size_t offset = i * kChunkSize;
        ASSERT_EQ(bytes[offset], static_cast<unsigned char>(i));
    }

    vm_release(addr, kSize);
}

// Test huge pages (best-effort, may fail)
TEST(vm_hugepages_fallback) {
    constexpr size_t kHugePageSize = 2 * 1024 * 1024; // 2MB
    constexpr size_t kSize = 4 * kHugePageSize;       // 8MB

    void* addr = vm_reserve(kSize);
    ASSERT_NOT_NULL(addr);

    // Commit entire region
    vm_commit(addr, kSize);

    // Try to use huge pages (may fail, which is acceptable)
    bool huge_pages = vm_try_hugepages(addr, kSize);
    // No assertion here - this is best-effort

    // Write to region (should work regardless of huge page status)
    memset(addr, 0xCD, kSize);
    auto* bytes = static_cast<unsigned char*>(addr);
    ASSERT_EQ(bytes[0], 0xCD);
    ASSERT_EQ(bytes[kSize - 1], 0xCD);

    vm_release(addr, kSize);
}
#endif // ASTRAL_ENABLE_VIRTUAL_MEMORY

// Test cache line size detection
TEST(cache_line_size_detection) {
    size_t size = cache_line_size();

    // Should be power of 2 between 32 and 256
    ASSERT_GE(size, 32);
    ASSERT_LE(size, 256);

    // Power of 2 check
    ASSERT_EQ((size & (size - 1)), 0);

    // Most common values: 64 (x86/ARM) or 128 (Apple Silicon)
    bool is_common = (size == 64 || size == 128);
    ASSERT_TRUE(is_common);
}

// Test CPU pause (no crashes)
TEST(cpu_pause_no_crash) {
    for (int i = 0; i < 1000; ++i) {
        cpu_pause();
    }
    // If we get here, no crash occurred
    ASSERT_TRUE(true);
}

// Test compiler fence (no crashes, ordering preserved)
TEST(compiler_fence_no_crash) {
    volatile int x = 0;
    x = 1;
    compiler_fence();
    ASSERT_EQ(x, 1);

    x = 2;
    compiler_fence();
    ASSERT_EQ(x, 2);

    x = 3;
    compiler_fence();
    ASSERT_EQ(x, 3);
}

#if ASTRAL_ENABLE_VIRTUAL_MEMORY
// Test error handling: reserve returns nullptr on huge allocation
TEST(vm_reserve_huge_allocation_fails) {
    // Try to reserve 1 PB (petabyte) - should fail
    constexpr size_t kHugeSize = 1ULL << 50; // 1 PB
    void* addr = vm_reserve(kHugeSize);

    // On some systems this might succeed (address space is cheap)
    // but on most it should fail
    if (addr != nullptr) {
        vm_release(addr, kHugeSize);
    }
    // No assertion - behavior is platform-dependent
    // Just verify no crash
}

// Test decommit without prior commit (should be safe no-op)
TEST(vm_decommit_without_commit) {
    constexpr size_t kSize = 2 * 1024 * 1024;
    void* addr = vm_reserve(kSize);
    ASSERT_NOT_NULL(addr);

    // Decommit without committing first (should be safe no-op)
    vm_decommit(addr, kSize);

    vm_release(addr, kSize);
}

// Test commit after decommit (recommit)
TEST(vm_commit_after_decommit) {
    constexpr size_t kSize = 2 * 1024 * 1024;
    void* addr = vm_reserve(kSize);
    ASSERT_NOT_NULL(addr);

    // Commit
    vm_commit(addr, kSize);
    memset(addr, 0xAA, kSize);

    // Decommit
    vm_decommit(addr, kSize);

    // Recommit
    vm_commit(addr, kSize);

    // Write again (should work)
    memset(addr, 0xBB, kSize);
    auto* bytes = static_cast<unsigned char*>(addr);
    ASSERT_EQ(bytes[0], 0xBB);

    vm_release(addr, kSize);
}

// Test alignment of reserved memory (should be page-aligned)
TEST(vm_reserve_alignment) {
    constexpr size_t kSize = 4096; // 1 page
    void* addr = vm_reserve(kSize);
    ASSERT_NOT_NULL(addr);

    // Check alignment (should be at least page-aligned, typically 4KB)
    uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
    ASSERT_EQ(addr_int % 4096, 0);

    vm_release(addr, kSize);
}

// Test multiple independent reservations
TEST(vm_multiple_reservations) {
    constexpr size_t kSize = 1024 * 1024; // 1MB
    constexpr int kCount = 4;
    void* addrs[kCount];

    // Reserve multiple regions
    for (int i = 0; i < kCount; ++i) {
        addrs[i] = vm_reserve(kSize);
        ASSERT_NOT_NULL(addrs[i]);
    }

    // All addresses should be different
    for (int i = 0; i < kCount; ++i) {
        for (int j = i + 1; j < kCount; ++j) {
            ASSERT_NE(addrs[i], addrs[j]);
        }
    }

    // Release all
    for (int i = 0; i < kCount; ++i) {
        vm_release(addrs[i], kSize);
    }
}
#endif // ASTRAL_ENABLE_VIRTUAL_MEMORY
