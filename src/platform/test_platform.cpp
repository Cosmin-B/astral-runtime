// Standalone smoke for the platform abstraction layer.

#include "vm.h"
#include "atomics.h"
#include "cpu_features.hpp"
#include "time.h"

#include <cstdio>
#include <cstring>
#include <cassert>

using namespace astral::platform;

void test_vm_basic() {
  printf("Testing vm_reserve/commit/decommit/release...\n");

  constexpr size_t kSize = 4 * 1024 * 1024; // 4MB

  void* addr = vm_reserve(kSize);
  assert(addr != nullptr);
  printf("  Reserved %zu bytes at %p\n", kSize, addr);

  constexpr size_t kCommitSize = 1 * 1024 * 1024;
  const bool committed = vm_commit(addr, kCommitSize);
  assert(committed);
  if (!committed) {
    vm_release(addr, kSize);
    return;
  }
  printf("  Committed %zu bytes\n", kCommitSize);

  // Touch committed pages.
  memset(addr, 0xAB, kCommitSize);
  printf("  Wrote to committed region (OK)\n");

  // Verify write
  auto* bytes = static_cast<unsigned char*>(addr);
  assert(bytes[0] == 0xAB);
  assert(bytes[kCommitSize - 1] == 0xAB);
  printf("  Verified write (OK)\n");

  // Decommit
  vm_decommit(addr, kCommitSize);
  printf("  Decommitted %zu bytes\n", kCommitSize);

  // Release entire region
  vm_release(addr, kSize);
  printf("  Released %zu bytes\n", kSize);

  printf("  [PASS] Basic VM operations\n\n");
}

// Test huge pages (may fail, which is acceptable)
void test_vm_hugepages() {
  printf("Testing vm_try_hugepages...\n");

  constexpr size_t kHugePageSize = 2 * 1024 * 1024; // 2MB
  constexpr size_t kSize = 4 * kHugePageSize;        // 8MB

  void* addr = vm_reserve(kSize);
  assert(addr != nullptr);

  // Commit entire region
  const bool committed = vm_commit(addr, kSize);
  assert(committed);
  if (!committed) {
    vm_release(addr, kSize);
    return;
  }

  // Try to use huge pages (may fail, which is OK)
  bool huge_pages = vm_try_hugepages(addr, kSize);
  if (huge_pages) {
    printf("  Huge pages enabled (OK)\n");
  } else {
    printf("  Huge pages NOT enabled (fallback to regular pages, OK)\n");
  }

  // Write to region (should work regardless of huge page status)
  memset(addr, 0xCD, kSize);
  auto* bytes = static_cast<unsigned char*>(addr);
  assert(bytes[0] == 0xCD);

  vm_release(addr, kSize);
  printf("  [PASS] Huge pages test (graceful fallback)\n\n");
}

// Test cache line size detection
void test_cache_line_size() {
  printf("Testing cache_line_size...\n");

  size_t size = cache_line_size();
  printf("  Detected cache line size: %zu bytes\n", size);

  // Sanity check: should be power of 2 between 32 and 256
  assert(size >= 32 && size <= 256);
  assert((size & (size - 1)) == 0); // power of 2

  printf("  [PASS] Cache line size detection\n\n");
}

// Test CPU pause (just check it doesn't crash)
void test_cpu_pause() {
  printf("Testing cpu_pause...\n");

  for (int i = 0; i < 100; ++i) {
    cpu_pause();
  }

  printf("  [PASS] CPU pause (no crash)\n\n");
}

// Test compiler fence (just check it doesn't crash)
void test_compiler_fence() {
  printf("Testing compiler_fence...\n");

  volatile int x = 0;
  x = 1;
  compiler_fence();
  x = 2;
  compiler_fence();
  assert(x == 2);

  printf("  [PASS] Compiler fence (no crash)\n\n");
}

void test_cpu_features() {
  printf("Testing cpu_features...\n");

  const CpuFeatures& features = cpu_features();
  printf("  CPU arch: %s dispatch tier: %s avx2=%u neon=%u\n",
         cpu_arch_name(features.arch),
         cpu_dispatch_tier_name(),
         features.x86_avx2 ? 1u : 0u,
         features.arm_neon ? 1u : 0u);

  assert(cpu_arch_name(features.arch) != nullptr);
  assert(cpu_dispatch_tier_name() != nullptr);

  printf("  [PASS] CPU feature detection\n\n");
}

void test_tick_clock() {
  printf("Testing tick_clock...\n");

  const TickClock clock = tick_clock();
  assert(clock.tick_to_ns > 0.0);

  const uint64_t first = ticks_now();
  uint64_t second = first;
  for (int i = 0; i < 1000 && second == first; ++i) {
    cpu_pause();
    second = ticks_now();
  }

  assert(second >= first);
  const uint64_t one_us_ticks = ticks_from_ns(1000);
  assert(one_us_ticks > 0);
  assert(ticks_to_ns(one_us_ticks) > 0);

  printf("  [PASS] Tick clock\n\n");
}

int main() {
  printf("=== Astral Platform Abstraction Layer Tests ===\n\n");

  test_vm_basic();
  test_vm_hugepages();
  test_cache_line_size();
  test_cpu_pause();
  test_compiler_fence();
  test_cpu_features();
  test_tick_clock();

  printf("=== All tests passed! ===\n");
  return 0;
}
