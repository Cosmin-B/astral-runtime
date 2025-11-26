#pragma once

#include <cstddef>
#include <cstdint>

namespace astral::platform {

/// Reserve virtual address space without committing physical memory.
///
/// This operation reserves a contiguous region of address space but does NOT
/// allocate physical memory. Pages are inaccessible until committed via vm_commit().
///
/// Platform implementation:
/// - Linux: mmap(PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS)
/// - Windows: VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)
/// - macOS: mmap(PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS)
///
/// @param size Size in bytes to reserve (should be multiple of page size, typically 4KB)
/// @return Pointer to reserved address space, or nullptr on failure
///
/// Alignment: Returned pointer is page-aligned (minimum 4KB)
/// Thread-safety: Safe to call from multiple threads
/// Memory: Does not consume physical memory; only address space
void* vm_reserve(size_t size);

/// Reserve virtual address space with a requested base alignment.
///
/// Notes:
/// - On Linux/macOS, this over-reserves and trims with munmap to return an aligned base.
/// - On Windows, this retries aligned reservations and returns a normal page-aligned reservation if no aligned slot is available.
/// - `alignment` should be a power of two and typically a multiple of the OS page size.
///
/// @param size Size in bytes to reserve
/// @param alignment Requested base alignment in bytes
/// @return Aligned pointer to reserved address space, or nullptr on failure
void* vm_reserve_aligned(size_t size, size_t alignment);

/// Commit pages in reserved address space (allocate physical memory).
///
/// Makes reserved pages accessible by committing physical memory. Can be called
/// multiple times on the same reserved region to incrementally commit pages.
/// Calling on already-committed pages is safe (no-op).
///
/// Platform implementation:
/// - Linux: mprotect(PROT_READ | PROT_WRITE) + madvise(MADV_WILLNEED)
/// - Windows: VirtualAlloc(MEM_COMMIT, PAGE_READWRITE)
/// - macOS: mprotect(PROT_READ | PROT_WRITE) + madvise(MADV_NORMAL)
///
/// @param addr Address within reserved region (must be page-aligned)
/// @param size Size in bytes to commit (must be multiple of page size)
///
///  This function must NOT be called in hot paths (decode/sampling loops).
/// Use pre-commit strategy during initialization to avoid page fault stalls.
///
/// Thread-safety: Safe if different threads commit non-overlapping regions
/// Memory: Allocates physical memory; may fail if system is out of memory
void vm_commit(void* addr, size_t size);

/// Decommit pages (release physical memory but keep address space reserved).
///
/// Releases physical memory backing the pages but keeps the address space reserved.
/// Accessing decommitted pages will cause a page fault until re-committed.
///
/// Platform implementation:
/// - Linux: madvise(MADV_DONTNEED) or madvise(MADV_FREE) + mprotect(PROT_NONE)
/// - Windows: VirtualFree(MEM_DECOMMIT)
/// - macOS: madvise(MADV_FREE_REUSABLE) or madvise(MADV_FREE) + mprotect(PROT_NONE)
///
/// @param addr Address within reserved region (must be page-aligned)
/// @param size Size in bytes to decommit (must be multiple of page size)
///
/// Thread-safety: Safe if different threads decommit non-overlapping regions
/// Memory: Releases physical memory; does not fail
void vm_decommit(void* addr, size_t size);

/// Release entire virtual address space reservation.
///
/// Unmaps the entire reserved region, releasing both physical memory and address space.
/// The pointer becomes invalid and must not be accessed after this call.
///
/// Platform implementation:
/// - Linux: munmap()
/// - Windows: VirtualFree(MEM_RELEASE)
/// - macOS: munmap()
///
/// @param addr Address returned by vm_reserve() (must be original base address)
/// @param size Size passed to vm_reserve() (must match original size)
///
/// Thread-safety: NOT safe; caller must ensure no concurrent access to the region
/// Memory: Releases all physical memory and address space
void vm_release(void* addr, size_t size);

/// Request huge pages (2MB/1GB) for better TLB performance.
///
/// Linux asks the kernel to promote an existing committed region through
/// MADV_HUGEPAGE. Windows cannot promote existing normal pages, so the call
/// returns false there; use vm_reserve_large() for Windows large pages.
/// The function returns false without changing the mapping when:
/// - System doesn't support huge pages
/// - Insufficient contiguous physical memory
/// - Insufficient huge page quota (Linux: vm.nr_hugepages)
/// - Insufficient privilege (Windows: SeLockMemoryPrivilege)
///
/// Platform requirements:
/// - Linux: addr must be 2MB-aligned, size must be multiple of 2MB
///         Requires vm.nr_hugepages > 0 or transparent huge pages enabled
/// - Windows: addr must be 2MB-aligned, process must have SeLockMemoryPrivilege
///            Use GetLargePageMinimum() to query minimum size
/// - macOS: Not supported; returns false
///
/// @param addr Address of committed region (must be huge-page-aligned, typically 2MB)
/// @param size Size in bytes (must be multiple of huge page size)
/// @return true if huge pages were successfully enabled, false otherwise
///
/// Thread-safety: Safe if different threads operate on non-overlapping regions
/// Memory: Does not change memory consumption; may improve TLB performance
bool vm_try_hugepages(void* addr, size_t size);

/// Return the platform large-page size in bytes, or 0 when explicit large pages
/// are unavailable.
///
/// On Windows this wraps GetLargePageMinimum(). On Linux this is the standard
/// 2 MiB huge-page size used by the Linux THP request path. macOS returns 0.
size_t vm_large_page_size();

/// Reserve a large-page region.
///
/// This is an initialization-time primitive for platforms where large pages must
/// be requested before a region exists. On Windows, VirtualAlloc requires
/// MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES in one call, so a successful
/// allocation is already committed and cannot be partially decommitted. The
/// caller must release it with vm_release().
///
/// @param size Requested size in bytes
/// @param out_size Optional output for the actual allocated size after platform
///                 large-page rounding
/// @return Large-page region base, or nullptr when unsupported, unprivileged, or
///         out of memory
void* vm_reserve_large(size_t size, size_t* out_size);

/// Commit a reserved region using the platform large-page policy where possible.
///
/// This exists for platforms that can apply a huge-page policy after reserving
/// normal VM. Windows returns false because large pages cannot be committed
/// retroactively there.
bool vm_commit_large(void* addr, size_t size);

} // namespace astral::platform
