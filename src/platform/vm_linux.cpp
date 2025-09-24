#include "vm.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

namespace astral::platform {

namespace {

// Get system page size (typically 4KB)
[[maybe_unused]] size_t get_page_size() {
  static const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return page_size;
}

// Huge page size (2MB on x86-64, aarch64)
constexpr size_t kHugePageSize = 2 * 1024 * 1024;

} // namespace

void* vm_reserve(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  // Reserve address space without committing physical memory
  // PROT_NONE: pages are inaccessible (no read/write/execute)
  // MAP_PRIVATE: changes are private to this process
  // MAP_ANONYMOUS: not backed by a file (no fd)
  // MAP_NORESERVE: do not reserve swap space (we'll commit explicitly)
  void* addr = mmap(
    nullptr,                                    // kernel chooses address
    size,                                       // size to reserve
    PROT_NONE,                                  // no access until committed
    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, // flags
    -1,                                         // no file descriptor
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

  // Best-effort: over-reserve and trim prefix/suffix with munmap so the remaining mapping is
  // exactly `size` and starts at an `alignment` boundary. This preserves the vm_release contract
  // (release with the same base+size).
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
    return; // Silently fail; caller should check errno if needed
  }

  // Step 2: Advise kernel to allocate physical pages
  // MADV_WILLNEED: expect access in near future (pre-fault pages)
  // This is a hint; failure is not critical
  // Note: on some systems, madvise may return EINVAL for MADV_WILLNEED
  // if the region is already resident, which is fine
  madvise(addr, size, MADV_WILLNEED);
}

void vm_decommit(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return;
  }

  // Step 1: Tell kernel we don't need these pages anymore
  // MADV_DONTNEED: pages can be freed immediately
  // On Linux, this zeros the pages and releases physical memory
  // Alternative: MADV_FREE (lazy release, faster but less aggressive)
  //
  // Note: MADV_DONTNEED is destructive on Linux (zeros pages)
  // but non-destructive on BSD/macOS. We accept Linux semantics.
  if (madvise(addr, size, MADV_DONTNEED) != 0) {
    // Non-critical failure; pages may remain resident
    // EINVAL: addr not page-aligned
    // ENOMEM: addr not mapped
  }

  // Step 2: Remove access permissions to catch use-after-decommit bugs
  // This is optional but helpful for debugging
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

  // Try to use transparent huge pages (THP)
  // MADV_HUGEPAGE: advise kernel to use huge pages for this region
  //
  // Requirements:
  // - Kernel support for THP (CONFIG_TRANSPARENT_HUGEPAGE=y)
  // - THP enabled: /sys/kernel/mm/transparent_hugepage/enabled
  // - Sufficient contiguous physical memory
  //
  // This is a hint; kernel may ignore if:
  // - THP disabled globally
  // - Insufficient huge pages available
  // - Memory fragmentation prevents large contiguous allocation
  if (madvise(addr, size, MADV_HUGEPAGE) != 0) {
    // EINVAL: THP not supported, or addr/size invalid
    // This is not fatal; we fall back to regular pages
    return false;
  }

  (void)kHugePageSize;

  // Alternative: use hugetlbfs or MAP_HUGETLB flag in mmap
  // For now, we rely on THP which is more portable and doesn't require
  // pre-allocating huge pages via vm.nr_hugepages sysctl

  // Cannot verify if huge pages were actually used without reading
  // /proc/self/smaps and checking for "AnonHugePages" field
  // We optimistically assume success if madvise didn't fail
  return true;
}

size_t vm_large_page_size() {
  return kHugePageSize;
}

void* vm_reserve_large(size_t size, size_t* out_size) {
  if (out_size != nullptr) {
    *out_size = 0;
  }
  if (size == 0) {
    return nullptr;
  }

  void* addr = vm_reserve_aligned(size, kHugePageSize);
  if (addr == nullptr) {
    return nullptr;
  }

  vm_commit(addr, size);
  if (!vm_try_hugepages(addr, size)) {
    vm_release(addr, size);
    return nullptr;
  }

  if (out_size != nullptr) {
    *out_size = size;
  }
  return addr;
}

bool vm_commit_large(void* addr, size_t size) {
  if (addr == nullptr || size == 0) {
    return false;
  }

  vm_commit(addr, size);
  return vm_try_hugepages(addr, size);
}

} // namespace astral::platform
