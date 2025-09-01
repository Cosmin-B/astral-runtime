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

// Detect cache line size at runtime
size_t detect_cache_line_size() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  // x86/x64: Use CPUID instruction to query cache parameters
  // CPUID leaf 0x80000006 (Extended L2 Cache Features)
  // Returns L2 cache line size in ECX bits 7:0

  uint32_t eax, ebx, ecx, edx;

#if defined(__GNUC__) || defined(__clang__)
  // GCC/Clang: use __get_cpuid intrinsic
  if (__get_cpuid(0x80000006, &eax, &ebx, &ecx, &edx)) {
    // Cache line size is in ECX bits 7:0 (in bytes)
    uint32_t cache_line = ecx & 0xFF;
    if (cache_line > 0) {
      return static_cast<size_t>(cache_line);
    }
  }
#elif defined(_MSC_VER)
  // MSVC: use __cpuid intrinsic
  int cpu_info[4];
  __cpuid(cpu_info, 0x80000006);
  ecx = static_cast<uint32_t>(cpu_info[2]);
  uint32_t cache_line = ecx & 0xFF;
  if (cache_line > 0) {
    return static_cast<size_t>(cache_line);
  }
#endif

  // Fallback for x86: 64 bytes (standard for modern Intel/AMD)
  return 64;

#elif defined(__aarch64__) || defined(_M_ARM64)
  // ARM64: Read CTR_EL0 register (Cache Type Register)
  // Bits 19:16 (DminLine) = log2(cache line size in words)
  // Cache line size = 4 * 2^DminLine bytes

#if defined(__GNUC__) || defined(__clang__)
  // GCC/Clang: inline assembly to read CTR_EL0
  uint64_t ctr;
  __asm__ __volatile__("mrs %0, ctr_el0" : "=r"(ctr));

  // Extract DminLine field (bits 19:16)
  uint32_t dmin_line = (ctr >> 16) & 0xF;

  // Cache line size = 4 * 2^DminLine
  size_t cache_line = 4 << dmin_line;

  // Sanity check: typical values are 64 or 128 bytes
  if (cache_line >= 32 && cache_line <= 256) {
    return cache_line;
  }
#endif

  // Fallback for ARM64: 64 bytes (common for Cortex-A cores)
  // Note: Apple Silicon (M1/M2) uses 128 bytes, but we can't detect
  // that without reading vendor-specific registers
  return 64;

#elif defined(__arm__) || defined(_M_ARM)
  // ARM32: Read CTR (Cache Type Register)
  // Similar to ARM64 but uses CP15 coprocessor register

#if defined(__GNUC__) || defined(__clang__)
  uint32_t ctr;
  __asm__ __volatile__("mrc p15, 0, %0, c0, c0, 1" : "=r"(ctr));

  // Extract DminLine field (bits 19:16)
  uint32_t dmin_line = (ctr >> 16) & 0xF;

  // Cache line size = 4 * 2^DminLine
  size_t cache_line = 4 << dmin_line;

  if (cache_line >= 32 && cache_line <= 256) {
    return cache_line;
  }
#endif

  // Fallback for ARM32: 64 bytes
  return 64;

#else
  // Unknown architecture: use safe default
  return 64;
#endif

  // Alternative detection methods (OS-specific):
  // These are more portable but may be slower or less accurate

#if defined(__linux__)
  // Linux: sysconf(_SC_LEVEL1_DCACHE_LINESIZE)
  long cache_line = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  if (cache_line > 0) {
    return static_cast<size_t>(cache_line);
  }
#elif defined(__APPLE__)
  // macOS: sysctlbyname("hw.cachelinesize")
  size_t cache_line = 0;
  size_t len = sizeof(cache_line);
  if (sysctlbyname("hw.cachelinesize", &cache_line, &len, nullptr, 0) == 0) {
    if (cache_line > 0) {
      return cache_line;
    }
  }
#elif defined(_WIN32)
  // Windows: GetLogicalProcessorInformation
  DWORD buffer_size = 0;
  GetLogicalProcessorInformation(nullptr, &buffer_size);

  if (buffer_size > 0) {
    auto* buffer = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(
      malloc(buffer_size)
    );

    if (buffer && GetLogicalProcessorInformation(buffer, &buffer_size)) {
      DWORD count = buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      for (DWORD i = 0; i < count; ++i) {
        if (buffer[i].Relationship == RelationCache &&
            buffer[i].Cache.Level == 1) {
          size_t cache_line = buffer[i].Cache.LineSize;
          free(buffer);
          if (cache_line > 0) {
            return cache_line;
          }
        }
      }
    }

    free(buffer);
  }
#endif

  // Final fallback: 64 bytes (safe for all modern CPUs)
  return 64;
}

} // namespace

size_t cache_line_size() {
  // Cached result (initialized once on first call)
  // Thread-safe in C++11: static local initialization is thread-safe
  static const size_t size = detect_cache_line_size();
  return size;
}

} // namespace astral::platform
