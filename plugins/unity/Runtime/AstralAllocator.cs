// Astral Unity Allocator Bridge
// Integrates Unity's native allocator with Astral runtime for zero-copy memory management
//
// Thread-safety: All callbacks are thread-safe (Unity allocator is thread-safe)
// IL2CPP: Uses MonoPInvokeCallback for IL2CPP/Burst compatibility
// Memory tracking: All allocations tracked by Unity Profiler under "Native Memory"

using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using AOT; // Required for IL2CPP MonoPInvokeCallback

namespace Astral.Runtime
{
    /// <summary>
    /// Bridge between Astral runtime and Unity's native allocator.
    /// Allows Unity to track all Astral memory allocations via Unity Profiler.
    ///
    ///  All callbacks must be thread-safe and non-throwing.
    /// Unity's UnsafeUtility.Malloc/Free are thread-safe and use Unity runtime internally.
    /// </summary>
    public static class AstralAllocatorBridge
    {
        // Callback delegates for Astral C ABI
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr AllocFn(IntPtr user, UIntPtr size, UIntPtr align);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void FreeFn(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align);

        // Keep delegates alive to prevent GC collection
        //  These MUST be static to survive across native calls
        private static AllocFn s_allocDelegate;
        private static FreeFn s_freeDelegate;

        // Allocation tracking for debugging
        private static long s_totalAllocated = 0;
        private static long s_totalFreed = 0;
        private static int s_allocCount = 0;
        private static int s_freeCount = 0;

        /// <summary>
        /// Create AstralAllocator that uses Unity's Allocator.Persistent.
        ///
        /// Memory strategy:
        /// - Uses Unity's UnsafeUtility.Malloc (backed by Unity runtime allocator)
        /// - All allocations tracked by Unity Memory Profiler
        /// - Thread-safe: Unity allocator allocator is thread-safe
        /// - No GC pressure: All allocations are unmanaged
        /// </summary>
        /// <returns>AstralAllocator struct ready to pass to astral_init()</returns>
        public static AstralNative.AstralAllocator CreateUnityAllocator()
        {
            // Initialize delegates (GC roots)
            s_allocDelegate = UnityAlloc;
            s_freeDelegate = UnityFree;

            // Reset tracking counters
            s_totalAllocated = 0;
            s_totalFreed = 0;
            s_allocCount = 0;
            s_freeCount = 0;

            return new AstralNative.AstralAllocator
            {
                alloc = Marshal.GetFunctionPointerForDelegate(s_allocDelegate),
                free = Marshal.GetFunctionPointerForDelegate(s_freeDelegate),
                user = IntPtr.Zero // No user data needed
            };
        }

        /// <summary>
        /// Allocation callback - uses Unity's Allocator.Persistent.
        ///
        /// CRITICAL REQUIREMENTS:
        /// - MUST be thread-safe (Unity allocator is)
        /// - MUST NOT throw exceptions (catch all, return null on failure)
        /// - MUST NOT allocate managed memory (no GC pressure)
        /// - MUST honor alignment requirements (power-of-2, up to 128 bytes)
        ///
        /// MonoPInvokeCallback required for IL2CPP/Burst compatibility.
        /// </summary>
        [MonoPInvokeCallback(typeof(AllocFn))]
        private static IntPtr UnityAlloc(IntPtr user, UIntPtr size, UIntPtr align)
        {
            try
            {
                // Validate alignment (must be power-of-2)
                int alignment = (int)align;
                if (alignment == 0) alignment = 8; // Default alignment

                // Unity UnsafeUtility.Malloc requirements:
                // - size: bytes to allocate
                // - alignment: must be power-of-2 (validated by Unity)
                // - allocator: Allocator.Persistent for long-lived allocations
                unsafe
                {
                    void* ptr = UnsafeUtility.Malloc(
                        (long)size,
                        alignment,
                        Allocator.Persistent
                    );

                    if (ptr == null)
                    {
                        Debug.LogError($"[Astral] Unity allocator failed: size={size}, align={align}");
                        return IntPtr.Zero;
                    }

                    // Track allocations (for debugging and Unity Profiler)
                    System.Threading.Interlocked.Add(ref s_totalAllocated, (long)size);
                    System.Threading.Interlocked.Increment(ref s_allocCount);

                    return new IntPtr(ptr);
                }
            }
            catch (Exception ex)
            {
                //  Never throw across C ABI boundary
                Debug.LogError($"[Astral] UnityAlloc exception: {ex.Message}");
                return IntPtr.Zero;
            }
        }

        /// <summary>
        /// Free callback - uses Unity's Allocator.Persistent.
        ///
        /// CRITICAL REQUIREMENTS:
        /// - MUST be thread-safe (Unity allocator is)
        /// - MUST NOT throw exceptions (catch all, log errors)
        /// - MUST handle null pointers gracefully
        /// - MUST match allocator used in UnityAlloc (Allocator.Persistent)
        ///
        /// MonoPInvokeCallback required for IL2CPP/Burst compatibility.
        /// </summary>
        [MonoPInvokeCallback(typeof(FreeFn))]
        private static void UnityFree(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align)
        {
            try
            {
                if (ptr == IntPtr.Zero)
                {
                    // Null pointer is valid (no-op)
                    return;
                }

                unsafe
                {
                    UnsafeUtility.Free(ptr.ToPointer(), Allocator.Persistent);
                }

                // Track deallocations
                System.Threading.Interlocked.Add(ref s_totalFreed, (long)size);
                System.Threading.Interlocked.Increment(ref s_freeCount);
            }
            catch (Exception ex)
            {
                //  Never throw across C ABI boundary
                Debug.LogError($"[Astral] UnityFree exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Get memory statistics from Unity allocator.
        /// Useful for debugging and Unity Profiler integration.
        /// </summary>
        /// <returns>
        /// (allocated, freed) - Total bytes allocated and freed since CreateUnityAllocator()
        /// </returns>
        public static (long allocated, long freed, int allocCount, int freeCount) GetMemoryStats()
        {
            return (
                System.Threading.Interlocked.Read(ref s_totalAllocated),
                System.Threading.Interlocked.Read(ref s_totalFreed),
                System.Threading.Volatile.Read(ref s_allocCount),
                System.Threading.Volatile.Read(ref s_freeCount)
            );
        }

        /// <summary>
        /// Check for memory leaks (allocated != freed).
        /// Call before AstralRuntime.Shutdown() to detect leaks.
        /// </summary>
        /// <returns>True if no leaks detected, false if allocated != freed</returns>
        public static bool ValidateNoLeaks()
        {
            var stats = GetMemoryStats();
            long delta = stats.allocated - stats.freed;
            int countDelta = stats.allocCount - stats.freeCount;

            if (delta != 0 || countDelta != 0)
            {
                Debug.LogWarning($"[Astral] Potential memory leak: {delta} bytes, {countDelta} allocations not freed");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Log memory statistics to Unity console.
        /// Useful for debugging and profiling.
        /// </summary>
        public static void LogMemoryStats()
        {
            var stats = GetMemoryStats();
            long delta = stats.allocated - stats.freed;
            int countDelta = stats.allocCount - stats.freeCount;

            Debug.Log($"[Astral] Memory Stats:\n" +
                      $"  Total Allocated: {stats.allocated:N0} bytes ({stats.allocCount} allocations)\n" +
                      $"  Total Freed: {stats.freed:N0} bytes ({stats.freeCount} frees)\n" +
                      $"  Current Usage: {delta:N0} bytes ({countDelta} live allocations)");
        }
    }
}
