#include "atomics.h"

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
  #elif defined(_MSC_VER)
    #include <intrin.h>
  #endif
#endif

#if defined(__linux__)
  #include <unistd.h>
#elif defined(__APPLE__)
  #include <sys/sysctl.h>
#elif defined(_WIN32)
  #include <windows.h>
#endif

namespace astral::platform {

namespace {

size_t valid_cache_line_size(size_t cache_line) {
  if (cache_line >= 32 && cache_line <= 256 && (cache_line & (cache_line - 1)) == 0) {
    return cache_line;
  }
  return 0;
}

// Detect cache line size at runtime
size_t detect_cache_line_size() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  // x86/x64: Use CPUID instruction to query cache parameters
  // CPUID leaf 0x80000006 (Extended L2 Cache Features)
  // Returns L2 cache line size in ECX bits 7:0

#if defined(__GNUC__) || defined(__clang__)
  uint32_t eax, ebx, ecx, edx;
  // GCC/Clang: use __get_cpuid intrinsic
  if (__get_cpuid(0x80000006, &eax, &ebx, &ecx, &edx)) {
    // Cache line size is in ECX bits 7:0 (in bytes)
    const size_t valid_size = valid_cache_line_size(ecx & 0xFFu);
    if (valid_size != 0) {
      return valid_size;
    }
  }
#elif defined(_MSC_VER)
  // MSVC: use __cpuid intrinsic
  int cpu_info[4];
  __cpuid(cpu_info, 0x80000006);
  const uint32_t ecx = static_cast<uint32_t>(cpu_info[2]);
  const size_t valid_size = valid_cache_line_size(ecx & 0xFFu);
  if (valid_size != 0) {
    return valid_size;
  }
#endif

  // Fallback for x86: 64 bytes (standard for modern Intel/AMD)
  return 64;

#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
  size_t cache_line = 0;
  size_t len = sizeof(cache_line);
  if (sysctlbyname("hw.cachelinesize", &cache_line, &len, nullptr, 0) == 0) {
    const size_t valid_size = valid_cache_line_size(cache_line);
    if (valid_size != 0) {
      return valid_size;
    }
  }
#elif defined(__GNUC__) || defined(__clang__)
  uint64_t ctr;
  __asm__ __volatile__("mrs %0, ctr_el0" : "=r"(ctr));

  uint32_t dmin_line = (ctr >> 16) & 0xF;

  const size_t valid_size = valid_cache_line_size(4ull << dmin_line);
  if (valid_size != 0) {
    return valid_size;
  }
#endif

  return 64;

#elif defined(__arm__) || defined(_M_ARM)
  // ARM32: Read CTR (Cache Type Register)
  // Similar to ARM64 but uses CP15 coprocessor register

#if defined(__GNUC__) || defined(__clang__)
  uint32_t ctr;
  __asm__ __volatile__("mrc p15, 0, %0, c0, c0, 1" : "=r"(ctr));

  uint32_t dmin_line = (ctr >> 16) & 0xF;

  size_t detected_cache_line = 4 << dmin_line;

  if (detected_cache_line >= 32 && detected_cache_line <= 256) {
    return detected_cache_line;
  }
#endif

  // Fallback for ARM32: 64 bytes
  return 64;

#else
  // Unknown architecture: use safe default
  return 64;
#endif

}

} // namespace

size_t cache_line_size() {
  // Cached result (initialized once on first call)
  // Thread-safe in C++11: static local initialization is thread-safe
  static const size_t size = detect_cache_line_size();
  return size;
}

} // namespace astral::platform
