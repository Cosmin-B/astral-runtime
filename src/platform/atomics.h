#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace astral::platform {

/// Get cache line size for the current CPU.
///
/// Returns the L1 data cache line size in bytes. This is used for:
/// - Padding shared atomics to prevent false sharing
/// - Aligning hot data structures to cache lines
///
/// Typical values:
/// - x86/x64: 64 bytes
/// - ARM Cortex-A: 64 bytes (Cortex-A53, A57, A72, A73)
/// - Apple Silicon (M1/M2): 128 bytes (larger to reduce false sharing)
///
/// Detection strategy:
/// - x86: CPUID instruction (leaf 0x80000006, ECX bits 7:0)
/// - ARM: Read CTR_EL0 register (DminLine field)
/// - Fallback: 64 bytes (safe default for x86/ARM)
///
/// Thread-safety: Safe to call from multiple threads (result is cached)
/// Performance: First call may use CPUID/sysconf; subsequent calls are fast
///
/// @return Cache line size in bytes (typically 64 or 128)
size_t cache_line_size();

/// Pause instruction for spin-wait loops.
///
/// Inserts a CPU hint that we are in a spin-wait loop. This:
/// - Reduces power consumption during busy-waiting
/// - Improves performance by reducing memory order violations
/// - Prevents speculative execution overhead in tight loops
///
/// Platform implementation:
/// - x86: PAUSE instruction (rep; nop encoding)
/// - ARM: YIELD instruction (hint to switch to another thread)
/// - Other: No-op or compiler barrier
///
/// Usage pattern:
/// ```cpp
/// while (!condition.load(std::memory_order_acquire)) {
///     cpu_pause(); // Reduce contention and power usage
/// }
/// ```
///
/// Thread-safety: Safe (no side effects beyond CPU pipeline hint)
/// Performance: ~10 cycles on x86, ~1 cycle on ARM (very fast)
inline void cpu_pause();

/// Wait for an event (low-power wait).
///
/// On ARM, this maps to `WFE` and can be paired with `cpu_signal_event()` (`SEV`)
/// to build low-overhead wait loops without syscalls.
///
/// On x86 and other platforms, this falls back to a light pause (no true event wait).
inline void cpu_wait_for_event();

/// Signal an event to wake threads waiting in `cpu_wait_for_event()`.
///
/// On ARM, this maps to `SEV`.
/// On x86 and other platforms, this is a no-op.
inline void cpu_signal_event();

/// Compiler fence (prevent instruction reordering by compiler).
///
/// Prevents the compiler from reordering loads/stores across this barrier.
/// Does NOT emit any CPU instructions; only affects compiler optimization.
///
/// Use cases:
/// - Enforce ordering when hardware memory model is strong (x86 TSO)
/// - Debug builds where you want strict ordering without atomic overhead
/// - Single-threaded code where you need to prevent register caching
///
/// NOT a substitute for std::atomic memory ordering:
/// - Does not prevent CPU reordering on weakly-ordered architectures (ARM)
/// - Does not synchronize across threads
/// - Use std::atomic with explicit memory_order for multi-threaded code
///
/// Platform implementation:
/// - GCC/Clang: asm volatile("" ::: "memory")
/// - MSVC: _ReadWriteBarrier()
///
/// Thread-safety: N/A (single-threaded barrier only)
/// Performance: Zero CPU cycles (compiler-only)
inline void compiler_fence();

//
// Inline implementations
//

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

// x86/x64 implementation
inline void cpu_pause() {
  // PAUSE instruction: tells CPU we're in a spin-wait loop
  // Encoded as "rep; nop" for backward compatibility with older CPUs
  // Benefits:
  // - Reduces pipeline flushes (improves performance)
  // - Lowers power consumption (reduces clock speed of executing core)
  // - Prevents memory order machine clear (MOMC) penalty on Intel
#if defined(__GNUC__) || defined(__clang__)
  __builtin_ia32_pause();
#elif defined(_MSC_VER)
  _mm_pause();
#else
  // Fallback: inline assembly
  __asm__ __volatile__("pause");
#endif
}

inline void cpu_wait_for_event() {
  // No architectural user-space "event wait" instruction that is portable across x86 CPUs.
  // Fall back to a pause hint in spin-wait loops.
  cpu_pause();
}

inline void cpu_signal_event() {
  // No-op on x86 fallback path.
}

#elif defined(__aarch64__) || defined(_M_ARM64)

// ARM64 implementation
inline void cpu_pause() {
  // YIELD instruction: hint to CPU that we're in a spin loop
  // The CPU may:
  // - Switch to another hardware thread (if SMT/hyperthreading is supported)
  // - Reduce power consumption
  // - Reorder pending stores to improve memory latency
  //
  // Note: Not all ARM implementations support multi-threading,
  // but YIELD is always safe (no-op if not supported)
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("yield" ::: "memory");
#elif defined(_MSC_VER)
  __yield();
#endif
}

inline void cpu_wait_for_event() {
  // ARM WFE: wait for event.
  // Safe in user space; may also return spuriously (interrupts).
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("wfe" ::: "memory");
#elif defined(_MSC_VER)
  __wfe();
#endif
}

inline void cpu_signal_event() {
  // ARM SEV: send event to wake WFE waiters.
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("sev" ::: "memory");
#elif defined(_MSC_VER)
  __sev();
#endif
}

#elif defined(__arm__) || defined(_M_ARM)

// ARM32 implementation
inline void cpu_pause() {
  // ARMv7+ supports YIELD instruction
  // ARMv6 and earlier: no-op (instruction is treated as NOP)
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("yield" ::: "memory");
#elif defined(_MSC_VER)
  __yield();
#endif
}

inline void cpu_wait_for_event() {
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("wfe" ::: "memory");
#elif defined(_MSC_VER)
  __wfe();
#endif
}

inline void cpu_signal_event() {
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("sev" ::: "memory");
#elif defined(_MSC_VER)
  __sev();
#endif
}

#else

// Fallback: no-op for unknown architectures
inline void cpu_pause() {
  // No CPU-specific pause instruction; use compiler fence
  compiler_fence();
}

inline void cpu_wait_for_event() {
  cpu_pause();
}

inline void cpu_signal_event() {
  // No-op
}

#endif

// Compiler fence implementation (platform-independent)
inline void compiler_fence() {
#if defined(__GNUC__) || defined(__clang__)
  // GCC/Clang: inline assembly with "memory" clobber
  // Prevents compiler from moving loads/stores across this point
  // Does not emit any CPU instructions
  __asm__ __volatile__("" ::: "memory");
#elif defined(_MSC_VER)
  // MSVC: intrinsic to prevent compiler reordering
  // Does not emit CPU instructions (compile-time only)
  _ReadWriteBarrier();
#else
  // Fallback: atomic signal fence with relaxed ordering
  // This is part of C++11 standard and portable
  std::atomic_signal_fence(std::memory_order_acq_rel);
#endif
}

} // namespace astral::platform
