// Simple validation test for platform abstraction layer
// NOT a unit test framework - just basic smoke tests

#include "vm.h"
#include "atomics.h"

#include <cstdio>
#include <cstring>
#include <cassert>

using namespace astral::platform;

// Test vm_reserve/commit/decommit/release cycle
void test_vm_basic() {
  printf("Testing vm_reserve/commit/decommit/release...\n");

  constexpr size_t kSize = 4 * 1024 * 1024; // 4MB

  // Reserve address space
  void* addr = vm_reserve(kSize);
  assert(addr != nullptr);
  printf("  Reserved %zu bytes at %p\n", kSize, addr);

  // Commit first 1MB
  constexpr size_t kCommitSize = 1 * 1024 * 1024;
  vm_commit(addr, kCommitSize);
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
  vm_commit(addr, kSize);

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

int main() {
  printf("=== Astral Platform Abstraction Layer Tests ===\n\n");

  test_vm_basic();
  test_vm_hugepages();
  test_cache_line_size();
  test_cpu_pause();
  test_compiler_fence();

  printf("=== All tests passed! ===\n");
  return 0;
}
