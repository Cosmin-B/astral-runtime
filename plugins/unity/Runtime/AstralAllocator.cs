using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using AOT; // Required for IL2CPP MonoPInvokeCallback

namespace Astral.Runtime
{
    /// <summary>
    /// Creates the native allocator callbacks used by AstralRuntime.Initialize().
    /// </summary>
    public static class AstralAllocatorBridge
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr AllocFn(IntPtr user, UIntPtr size, UIntPtr align);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void FreeFn(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align);

        private static AllocFn s_allocDelegate;
        private static FreeFn s_freeDelegate;

        private static long s_totalAllocated = 0;
        private static long s_totalFreed = 0;
        private static int s_allocCount = 0;
        private static int s_freeCount = 0;

        /// <summary>
        /// Creates an AstralAllocator backed by Unity's Allocator.Persistent.
        /// </summary>
        public static AstralNative.AstralAllocator CreateUnityAllocator()
        {
            s_allocDelegate = UnityAlloc;
            s_freeDelegate = UnityFree;

            s_totalAllocated = 0;
            s_totalFreed = 0;
            s_allocCount = 0;
            s_freeCount = 0;

            return new AstralNative.AstralAllocator
            {
                alloc = Marshal.GetFunctionPointerForDelegate(s_allocDelegate),
                free = Marshal.GetFunctionPointerForDelegate(s_freeDelegate),
                user = IntPtr.Zero
            };
        }

        /// <summary>
        /// Native allocation callback. Exceptions are logged and converted to allocation failure.
        /// </summary>
        [MonoPInvokeCallback(typeof(AllocFn))]
        private static IntPtr UnityAlloc(IntPtr user, UIntPtr size, UIntPtr align)
        {
            try
            {
                int alignment = (int)align;
                if (alignment == 0) alignment = 8;

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

                    System.Threading.Interlocked.Add(ref s_totalAllocated, (long)size);
                    System.Threading.Interlocked.Increment(ref s_allocCount);

                    return new IntPtr(ptr);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Astral] UnityAlloc exception: {ex.Message}");
                return IntPtr.Zero;
            }
        }

        /// <summary>
        /// Native free callback. Exceptions are logged because the C ABI cannot receive them.
        /// </summary>
        [MonoPInvokeCallback(typeof(FreeFn))]
        private static void UnityFree(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align)
        {
            try
            {
                if (ptr == IntPtr.Zero)
                {
                    return;
                }

                unsafe
                {
                    UnsafeUtility.Free(ptr.ToPointer(), Allocator.Persistent);
                }

                System.Threading.Interlocked.Add(ref s_totalFreed, (long)size);
                System.Threading.Interlocked.Increment(ref s_freeCount);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Astral] UnityFree exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Returns bridge counters since the latest CreateUnityAllocator() call.
        /// </summary>
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
        /// Returns false when bridge counters show live native allocations.
        /// </summary>
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
        /// Logs bridge allocation counters to the Unity console.
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
