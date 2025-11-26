#include "vm.h"

#ifdef __APPLE__

#include <sys/mman.h>
#include <unistd.h>
#include <mach/vm_statistics.h>
#include <cerrno>

namespace astral::platform {

namespace {

// Get system page size (typically 16KB on Apple Silicon, 4KB on Intel)
size_t get_page_size() {
  static const size_t page_size = static_cast<size_t>(getpagesize());
  return page_size;
}

} // namespace

void* vm_reserve(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  // Reserve address space without committing physical memory
  // PROT_NONE: pages are inaccessible (no read/write/execute)
  // MAP_PRIVATE: changes are private to this process
  // MAP_ANONYMOUS: not backed by a file
  // MAP_NORESERVE: do not reserve swap space (we'll commit explicitly)
  //
  // macOS-specific: VM_FLAGS_SUPERPAGE_NONE hint tells kernel not to use
  // superpages (huge pages) unless explicitly requested later
  void* addr = mmap(
    nullptr,                                    // kernel chooses address
    size,                                       // size to reserve
    PROT_NONE,                                  // no access until committed
    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, // flags
    VM_MAKE_TAG(VM_MEMORY_APPLICATION_SPECIFIC_1), // mach VM tag (for vmmap)
    0                                           // no offset
  );

  if (addr == MAP_FAILED) {
    // Common failure reasons:
    // - ENOMEM: insufficient address space (unlikely on 64-bit)
    // - EINVAL: invalid size (e.g., too large)
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

  // Over-reserve and trim with munmap to return an aligned base.
  const size_t total = size + alignment;
  void* base = vm_reserve(total);
  if (base == nullptr) {
    return nullptr;
  }

  const uintptr_t base_int = reinterpret_cast<uintptr_t>(base);
  const uintptr_t aligned_int = (base_int + (alignment - 1)) & ~(static_cast<uintptr_t>(alignment) - 1);
  const size_t prefix = static_cast<size_t>(aligned_int - base_int);
  const size_t suffix = total - prefix - size;

  if (prefix != 0) {
    munmap(base, prefix);
  }
  if (suffix != 0) {
    void* tail = reinterpret_cast<void*>(aligned_int + size);
    munmap(tail, suffix);
  }

  return reinterpret_cast<void*>(aligned_int);
}

void vm_commit(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return;
  }

  // Step 1: Make pages accessible (read/write)
  // mprotect changes page protection on already-mapped pages
  // Safe to call on already-committed pages (no-op)
  if (mprotect(addr, size, PROT_READ | PROT_WRITE) != 0) {
    // Failure reasons:
    // - EINVAL: addr not page-aligned, or invalid range
    // - ENOMEM: kernel cannot allocate internal structures
    // - EACCES: permission denied (shouldn't happen for our reserved pages)
    return; // The region remains inaccessible; callers prevalidate page ranges at setup boundaries.
  }

  // Step 2: Advise kernel that we'll use these pages
  // MADV_NORMAL: normal page usage (not sequential, not random)
  // This tells macOS to allocate physical pages and make them resident
  //
  // Note: On macOS, MADV_WILLNEED is less aggressive than on Linux
  // MADV_NORMAL is preferred for general-purpose memory
  if (madvise(addr, size, MADV_NORMAL) != 0) {
    // EINVAL: addr not page-aligned
    // ENOMEM: addr not mapped
    // Non-critical failure; pages will fault in on first access
  }
}

void vm_decommit(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return;
  }

  // Step 1: Tell kernel we don't need these pages anymore
  // macOS-specific: MADV_FREE_REUSABLE is preferred over MADV_FREE
  //
  // MADV_FREE_REUSABLE:
  // - Explicitly marks pages as reusable by other processes
  // - Shows up as "purgeable" in Activity Monitor and vmmap
  // - Kernel can reclaim immediately under memory pressure
  // - Must call MADV_FREE_REUSE before accessing again (we use mprotect instead)
  //
  // Fallback to MADV_FREE if MADV_FREE_REUSABLE not available:
  // - Lazy release; kernel reclaims when needed
  // - Pages are zeroed on next access
  // - More portable (works on BSD/Linux)
  int result = madvise(addr, size, MADV_FREE_REUSABLE);
  if (result != 0) {
    // Fallback to MADV_FREE if MADV_FREE_REUSABLE failed
    result = madvise(addr, size, MADV_FREE);
    if (result != 0) {
      // EINVAL: addr not page-aligned
      // ENOMEM: addr not mapped
      // Non-critical failure; pages may remain resident
    }
  }

  // Step 2: Remove access permissions to catch use-after-decommit bugs
  // This is optional but helpful for debugging
  // Also required when using MADV_FREE_REUSABLE (must call MADV_FREE_REUSE
  // or re-protect before next access)
  mprotect(addr, size, PROT_NONE);
}

void vm_release(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return;
  }

  // Unmap the entire reserved region
  // This releases both physical memory and address space
  // Must pass original base address and size from vm_reserve()
  if (munmap(addr, size) != 0) {
    // Failure reasons:
    // - EINVAL: addr not page-aligned, or not mapped
    // munmap failures are rare and usually indicate programmer error
  }
}

bool vm_try_hugepages(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return false;
  }

  // macOS does not support explicit huge page allocation via madvise
  // or mmap flags like Linux (MADV_HUGEPAGE) or Windows (MEM_LARGE_PAGES).
  //
  // Instead, macOS uses "superpages" which are managed automatically by
  // the virtual memory subsystem. The kernel decides when to promote
  // pages to superpages based on:
  // - Page access patterns
  // - Memory pressure
  // - Alignment and contiguity
  //
  // There is no public API to force superpage usage, though some
  // undocumented VM flags exist (VM_FLAGS_SUPERPAGE_SIZE_2MB) which
  // are not officially supported and may break across OS versions.
  //
  // For portable code, we return false and let the kernel manage
  // superpages automatically via its heuristics.
  //
  // Reference:
  // - XNU source: osfmk/mach/vm_statistics.h (VM_FLAGS_SUPERPAGE_*)
  // - Not exposed in public SDK headers
  // - Used internally by IOKit for DMA buffers

  return false; // No explicit huge page control on macOS
}

size_t vm_large_page_size() {
  return 0;
}

void* vm_reserve_large(size_t size, size_t* out_size) {
  (void)size;
  if (out_size != nullptr) {
    *out_size = 0;
  }
  return nullptr;
}

bool vm_commit_large(void* addr, size_t size) {
  (void)addr;
  (void)size;
  return false;
}

} // namespace astral::platform

#endif // __APPLE__
