#include "vm.h"

#ifdef _WIN32

#include <windows.h>
#include <cstdint>

namespace astral::platform {

namespace {

// Cached after the first system query; Windows page size is process-stable.
size_t get_page_size() {
  static size_t page_size = 0;
  if (page_size == 0) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    page_size = static_cast<size_t>(si.dwPageSize);
  }
  return page_size;
}

// Zero means the host does not expose explicit large-page allocation.
size_t get_large_page_minimum() {
  static size_t large_page_size = 0;
  if (large_page_size == 0) {
    large_page_size = GetLargePageMinimum();
  }
  return large_page_size;
}

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

void* vm_reserve(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  // Reserve inaccessible address space; Windows rounds to allocation granularity.
  void* addr = VirtualAlloc(
    nullptr,
    size,
    MEM_RESERVE,
    PAGE_NOACCESS   // no access until committed
  );

  if (addr == nullptr) {
    return nullptr;
  }

  return addr;
}

void* vm_reserve_aligned(size_t size, size_t alignment) {
  if (size == 0) {
    return nullptr;
  }
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return nullptr;
  }

  // Alignment strategy:
  // 1) Reserve a larger region anywhere.
  // 2) Release it.
  // 3) Try reserving exactly `size` at an aligned address inside the released range.
  // Repeat a few times if the aligned address cannot be reserved.
  constexpr uint32_t kMaxAttempts = 32;
  const size_t total = size + alignment;

  for (uint32_t attempt = 0; attempt < kMaxAttempts; ++attempt) {
    void* base = ::VirtualAlloc(nullptr, total, MEM_RESERVE, PAGE_NOACCESS);
    if (base == nullptr) {
      return nullptr;
    }

    const uintptr_t base_int = reinterpret_cast<uintptr_t>(base);
    const uintptr_t aligned_int = (base_int + (alignment - 1)) & ~(static_cast<uintptr_t>(alignment) - 1);
    ::VirtualFree(base, 0, MEM_RELEASE);

    void* aligned = ::VirtualAlloc(reinterpret_cast<void*>(aligned_int), size, MEM_RESERVE, PAGE_NOACCESS);
    if (aligned != nullptr) {
      return aligned;
    }
  }

  // The caller still gets a valid reservation when strict alignment cannot be placed.
  return vm_reserve(size);
}

void vm_commit(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return;
  }

  // MEM_COMMIT is idempotent for pages that are already committed.
  void* result = VirtualAlloc(
    addr,
    size,
    MEM_COMMIT,
    PAGE_READWRITE
  );

  if (result == nullptr) {
    // Common failure reasons:
    // - ERROR_NOT_ENOUGH_MEMORY: system out of memory
    // - ERROR_INVALID_ADDRESS: addr not within reserved region
    // - ERROR_COMMITMENT_LIMIT: commit charge limit exceeded
    // The region remains inaccessible; callers prevalidate page ranges at setup boundaries.
  }
}

void vm_decommit(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return;
  }

  // MEM_DECOMMIT releases backing memory while keeping the address range reserved.
  if (!VirtualFree(addr, size, MEM_DECOMMIT)) {
    // Common failure reasons:
    // - ERROR_INVALID_ADDRESS: addr not page-aligned or not committed
    // - ERROR_INVALID_PARAMETER: size invalid
    // Non-critical failure; pages may remain committed
  }
}

void vm_release(void* addr, size_t size) {
  if (addr == nullptr) {
    return;
  }

  // MEM_RELEASE requires the original reservation base and a zero size.
  (void)size;

  if (!VirtualFree(addr, 0, MEM_RELEASE)) {
    // Common failure reasons:
    // - ERROR_INVALID_ADDRESS: addr not a base address from VirtualAlloc
    // - ERROR_INVALID_PARAMETER: size was non-zero (caller error)
    // VirtualFree failures indicate programmer error or corrupted state
  }
}

bool vm_try_hugepages(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return false;
  }

  size_t large_page_min = get_large_page_minimum();
  if (large_page_min == 0) {
    return false;
  }

  const uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
  if (addr_int % large_page_min != 0) {
    return false;
  }

  if (size % large_page_min != 0) {
    return false;
  }

  // Windows large pages must be requested at reservation time.
  return false;
}

size_t vm_large_page_size() {
  return get_large_page_minimum();
}

void* vm_reserve_large(size_t size, size_t* out_size) {
  if (out_size != nullptr) {
    *out_size = 0;
  }
  if (size == 0) {
    return nullptr;
  }

  const size_t large_page_min = get_large_page_minimum();
  if (large_page_min == 0) {
    return nullptr;
  }

  const size_t alloc_size = align_up(size, large_page_min);
  void* addr = ::VirtualAlloc(
    nullptr,
    alloc_size,
    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
    PAGE_READWRITE
  );
  if (addr == nullptr) {
    return nullptr;
  }

  if (out_size != nullptr) {
    *out_size = alloc_size;
  }
  return addr;
}

bool vm_commit_large(void* addr, size_t size) {
  (void)addr;
  (void)size;
  return false;
}

} // namespace astral::platform

#endif // _WIN32
