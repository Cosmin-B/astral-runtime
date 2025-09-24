#include "vm.h"

#ifdef _WIN32

#include <windows.h>
#include <cstdint>

namespace astral::platform {

namespace {

// Query system page size (typically 4KB on x86/x64)
size_t get_page_size() {
  static size_t page_size = 0;
  if (page_size == 0) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    page_size = static_cast<size_t>(si.dwPageSize);
  }
  return page_size;
}

// Query large page minimum size (typically 2MB on x64)
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

  // Reserve address space without committing physical memory
  // MEM_RESERVE: reserve address space only
  // PAGE_NOACCESS: pages are inaccessible until committed
  //
  // Note: VirtualAlloc automatically rounds size up to page granularity
  // Allocation granularity is typically 64KB (not 4KB page size)
  void* addr = VirtualAlloc(
    nullptr,        // kernel chooses address
    size,           // size to reserve
    MEM_RESERVE,    // reserve only, don't commit
    PAGE_NOACCESS   // no access until committed
  );

  if (addr == nullptr) {
    // Common failure reasons:
    // - ERROR_NOT_ENOUGH_MEMORY: insufficient address space
    // - ERROR_INVALID_PARAMETER: size too large or invalid
    // GetLastError() would provide specific error code
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

  // Best-effort strategy:
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

  // Fallback: reserve without alignment.
  return vm_reserve(size);
}

void vm_commit(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return;
  }

  // Commit physical memory to reserved pages
  // MEM_COMMIT: allocate physical memory
  // PAGE_READWRITE: allow read/write access
  //
  // Safe to call on already-committed pages (no-op, returns same address)
  // VirtualAlloc returns nullptr on failure
  void* result = VirtualAlloc(
    addr,            // specific address within reserved region
    size,            // size to commit
    MEM_COMMIT,      // commit physical memory
    PAGE_READWRITE   // read/write access
  );

  if (result == nullptr) {
    // Common failure reasons:
    // - ERROR_NOT_ENOUGH_MEMORY: system out of memory
    // - ERROR_INVALID_ADDRESS: addr not within reserved region
    // - ERROR_COMMITMENT_LIMIT: commit charge limit exceeded
    // Silently fail; caller should check for access violations if needed
  }
}

void vm_decommit(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return;
  }

  // Decommit physical memory but keep address space reserved
  // MEM_DECOMMIT: release physical memory, keep reservation
  //
  // After decommit:
  // - Physical memory is released back to system
  // - Address space remains reserved
  // - Pages become inaccessible (PAGE_NOACCESS)
  // - Accessing pages causes access violation until re-committed
  //
  // Note: Unlike munmap, this does NOT free the address space
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

  // Release entire virtual address space reservation
  // MEM_RELEASE: free both physical memory and address space
  //
  //  When using MEM_RELEASE, size MUST be 0 and addr MUST be
  // the base address returned by the original VirtualAlloc(MEM_RESERVE) call.
  // VirtualFree releases the entire region in one shot.
  //
  // We ignore the size parameter because Windows requires it to be 0
  (void)size; // Suppress unused parameter warning

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

  // Query large page minimum size
  size_t large_page_min = get_large_page_minimum();
  if (large_page_min == 0) {
    // Large pages not supported or not enabled
    return false;
  }

  // Check alignment requirements
  const uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
  if (addr_int % large_page_min != 0) {
    return false; // Not aligned to large page boundary
  }

  if (size % large_page_min != 0) {
    return false; // Size not multiple of large page size
  }

  return false; // Large pages cannot be applied retroactively on Windows
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
