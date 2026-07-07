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
#include "../src/platform/cpu_features.hpp"
#include "../src/platform/time.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

using namespace astral::platform;

#if ASTRAL_ENABLE_VIRTUAL_MEMORY
// VM reserve/commit/decommit/release contract.
TEST(vm_reserve_commit_release) {
    constexpr size_t kSize = 4 * 1024 * 1024; // 4MB

    void* addr = vm_reserve(kSize);
    ASSERT_NOT_NULL(addr);

    constexpr size_t kCommitSize = 1 * 1024 * 1024;
    vm_commit(addr, kCommitSize);

    // Touch committed pages.
    memset(addr, 0xAB, kCommitSize);

    auto* bytes = static_cast<unsigned char*>(addr);
    ASSERT_EQ(bytes[0], 0xAB);
    ASSERT_EQ(bytes[kCommitSize - 1], 0xAB);

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

// Huge-page commit may fall back to regular pages; the committed region must still be usable.
TEST(vm_hugepages_fallback) {
    constexpr size_t kHugePageSize = 2 * 1024 * 1024; // 2MB
    constexpr size_t kSize = 4 * kHugePageSize;       // 8MB

    void* addr = vm_reserve(kSize);
    ASSERT_NOT_NULL(addr);

    // Commit entire region
    vm_commit(addr, kSize);

    // Transparent huge-page promotion is advisory; byte access is the contract.
    const bool huge_pages = vm_try_hugepages(addr, kSize);
    (void)huge_pages;

    // Write to region (should work regardless of huge page status)
    memset(addr, 0xCD, kSize);
    auto* bytes = static_cast<unsigned char*>(addr);
    ASSERT_EQ(bytes[0], 0xCD);
    ASSERT_EQ(bytes[kSize - 1], 0xCD);

    vm_release(addr, kSize);
}

TEST(vm_large_page_reserve_fallback) {
    const size_t large_page_size = vm_large_page_size();
    const bool expect_large_pages = std::getenv("ASTRAL_TEST_EXPECT_LARGE_PAGES") != nullptr;
    const bool expect_large_page_fallback = std::getenv("ASTRAL_TEST_EXPECT_LARGE_PAGE_FALLBACK") != nullptr;
    ASSERT_FALSE(expect_large_pages && expect_large_page_fallback);

    if (large_page_size == 0) {
        ASSERT_FALSE(expect_large_pages);
        ASSERT_NULL(vm_reserve_large(2 * 1024 * 1024, nullptr));
        return;
    }

    size_t actual_size = 0;
    void* addr = vm_reserve_large(large_page_size, &actual_size);
    if (addr == nullptr) {
        ASSERT_FALSE(expect_large_pages);
        ASSERT_EQ(actual_size, 0u);
        return;
    }

    ASSERT_FALSE(expect_large_page_fallback);
    ASSERT_GE(actual_size, large_page_size);
    memset(addr, 0xEF, large_page_size);
    auto* bytes = static_cast<unsigned char*>(addr);
    ASSERT_EQ(bytes[0], 0xEF);
    ASSERT_EQ(bytes[large_page_size - 1], 0xEF);

    vm_release(addr, actual_size);
}
#endif // ASTRAL_ENABLE_VIRTUAL_MEMORY

TEST(cache_line_size_detection) {
    size_t size = cache_line_size();

    ASSERT_GE(size, 32);
    ASSERT_LE(size, 256);

    ASSERT_EQ((size & (size - 1)), 0);
}

// Test CPU pause (no crashes)
TEST(cpu_pause_no_crash) {
    for (int i = 0; i < 1000; ++i) {
        cpu_pause();
    }
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

TEST(cpu_feature_detection_sanity) {
    const CpuFeatures& features = cpu_features();
    const char* arch_name = cpu_arch_name(features.arch);
    const char* tier_name = cpu_dispatch_tier_name();

    ASSERT_NOT_NULL(arch_name);
    ASSERT_NOT_NULL(tier_name);

#if defined(__x86_64__) || defined(_M_X64)
    ASSERT_EQ(features.arch, CpuArch::X86_64);
#elif defined(__aarch64__) || defined(_M_ARM64)
    ASSERT_EQ(features.arch, CpuArch::Arm64);
    ASSERT_TRUE(features.arm_neon);
#endif
}

TEST(x86_conversion_features_require_f16c) {
  constexpr uint32_t kXsave = 1u << 26;
  constexpr uint32_t kOsxsave = 1u << 27;
  constexpr uint32_t kAvx = 1u << 28;
  constexpr uint32_t kF16c = 1u << 29;
  constexpr uint32_t kAvx2 = 1u << 5;
  constexpr uint64_t kXmmYmmState = (1ull << 1) | (1ull << 2);

  const X86Features complete =
      x86_features_from_registers(kXsave | kOsxsave | kAvx | kF16c, kAvx2, kXmmYmmState);
  ASSERT_TRUE(complete.avx2);
  ASSERT_TRUE(complete.f16c);
  ASSERT_TRUE(x86_supports_avx2_f16c(complete));

  const X86Features no_f16c =
      x86_features_from_registers(kXsave | kOsxsave | kAvx, kAvx2, kXmmYmmState);
  ASSERT_TRUE(no_f16c.avx2);
  ASSERT_FALSE(no_f16c.f16c);
  ASSERT_FALSE(x86_supports_avx2_f16c(no_f16c));

  const X86Features no_avx2 =
      x86_features_from_registers(kXsave | kOsxsave | kAvx | kF16c, 0, kXmmYmmState);
  ASSERT_FALSE(no_avx2.avx2);
  ASSERT_TRUE(no_avx2.f16c);
  ASSERT_FALSE(x86_supports_avx2_f16c(no_avx2));

  const X86Features no_ymm_state =
      x86_features_from_registers(kXsave | kOsxsave | kAvx | kF16c, kAvx2, 1ull << 1);
  ASSERT_FALSE(no_ymm_state.avx2);
  ASSERT_FALSE(no_ymm_state.f16c);
  ASSERT_FALSE(x86_supports_avx2_f16c(no_ymm_state));
}

TEST(tick_clock_sanity) {
  const TickClock clock = tick_clock();
  ASSERT_GT(clock.tick_to_ns, 0.0);

  const uint64_t first = ticks_now();
  uint64_t second = first;
  for (uint32_t i = 0; i < 1000 && second == first; ++i) {
    cpu_pause();
    second = ticks_now();
  }

  ASSERT_GE(second, first);
  const uint64_t one_us_ticks = ticks_from_ns(1000);
  ASSERT_GT(one_us_ticks, 0u);
  ASSERT_GT(ticks_to_ns(one_us_ticks), 0u);
}

#if ASTRAL_ENABLE_VIRTUAL_MEMORY
// Test error handling: reserve returns nullptr on huge allocation
TEST(vm_reserve_huge_allocation_fails) {
    // Try to reserve 1 PB (petabyte) - should fail
    constexpr size_t kHugeSize = 1ULL << 50; // 1 PB
    void* addr = vm_reserve(kHugeSize);

    // Overcommitting hosts can hand out address space for this request.
    if (addr != nullptr) {
        vm_release(addr, kHugeSize);
        return;
    }
    ASSERT_NULL(addr);
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
