// Astral Unity NativeArray Extensions
// Zero-copy conversions between Unity NativeArray and Astral spans
//
// Performance: All conversions are zero-copy (pointer-based)
// Safety: Uses unsafe pointers with proper lifetime management
// Compatibility: Works with Burst compiler and IL2CPP

using System;
using System.Runtime.CompilerServices;
using System.Text;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    /// <summary>
    /// Helpers for zero-copy conversion between NativeArray and Astral spans.
    ///
    /// CRITICAL DESIGN PRINCIPLES:
    /// - All conversions are zero-copy (no memcpy, just pointer casts)
    /// - Lifetime: Span is valid only while NativeArray is alive
    /// - Thread-safety: NativeArray must not be disposed while span is in use
    /// - UTF-8: All string conversions use UTF-8 encoding (no UTF-16)
    /// </summary>
    public static class AstralNativeArray
    {
        // ====== NativeArray<byte> -> Astral Spans ======

        /// <summary>
        /// Convert NativeArray{byte} to AstralSpanU8 (read-only, zero-copy).
        ///
        /// SAFETY:
        /// - NativeArray must remain alive while span is in use
        /// - Caller is responsible for lifetime management
        /// - No bounds checking after conversion (Astral runtime handles this)
        ///
        /// Usage:
        ///   var array = new NativeArray{byte}(1024, Allocator.Temp);
        ///   var span = array.AsSpan();
        ///   AstralNative.astral_session_feed(session, span, finalize: 0);
        ///   array.Dispose();
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe AstralNative.AstralSpanU8 AsSpan(this NativeArray<byte> array)
        {
            return new AstralNative.AstralSpanU8
            {
                data = (IntPtr)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(array),
                len = (uint)array.Length
            };
        }

        /// <summary>
        /// Convert NativeArray{byte} to AstralMutSpanU8 (mutable, zero-copy).
        ///
        /// SAFETY:
        /// - NativeArray must remain alive while span is in use
        /// - NativeArray must be writable (not read-only)
        /// - Caller is responsible for thread-safety (no concurrent writes)
        ///
        /// Usage:
        ///   var buffer = new NativeArray{byte}(1024, Allocator.Temp);
        ///   var span = buffer.AsMutSpan();
        ///   int bytesRead = AstralNative.astral_stream_read(session, span, 1000);
        ///   // Use buffer[0..bytesRead]
        ///   buffer.Dispose();
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe AstralNative.AstralMutSpanU8 AsMutSpan(this NativeArray<byte> array)
        {
            return new AstralNative.AstralMutSpanU8
            {
                data = (IntPtr)NativeArrayUnsafeUtility.GetUnsafePtr(array),
                len = (uint)array.Length
            };
        }

        /// <summary>
        /// Convert NativeSlice{byte} to AstralSpanU8 (read-only, zero-copy).
        ///
        /// SAFETY:
        /// - NativeSlice must remain alive while span is in use
        /// - Underlying NativeArray must remain alive
        ///
        /// Usage:
        ///   var array = new NativeArray{byte}(1024, Allocator.Temp);
        ///   var slice = new NativeSlice{byte}(array, 0, 512);
        ///   var span = slice.AsSpan();
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe AstralNative.AstralSpanU8 AsSpan(this NativeSlice<byte> slice)
        {
            return new AstralNative.AstralSpanU8
            {
                data = (IntPtr)slice.GetUnsafeReadOnlyPtr(),
                len = (uint)slice.Length
            };
        }

        /// <summary>
        /// Convert NativeSlice{byte} to AstralMutSpanU8 (mutable, zero-copy).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe AstralNative.AstralMutSpanU8 AsMutSpan(this NativeSlice<byte> slice)
        {
            return new AstralNative.AstralMutSpanU8
            {
                data = (IntPtr)slice.GetUnsafePtr(),
                len = (uint)slice.Length
            };
        }

        // ====== String -> NativeArray<byte> (UTF-8) ======

        /// <summary>
        /// Convert C# string to NativeArray{byte} (UTF-8 encoding).
        ///
        /// 
        /// - Always UTF-8 encoding (no UTF-16, no Latin-1)
        /// - Caller must dispose NativeArray when done
        /// - No NUL terminator (Astral uses length-based spans)
        ///
        /// Performance:
        /// - Allocates NativeArray (caller must dispose)
        /// - Encoding.UTF8.GetBytes allocates managed array (GC pressure)
        /// - Use Allocator.Temp for short-lived conversions
        /// - Use Allocator.TempJob for job-based conversions
        /// - Use Allocator.Persistent for long-lived conversions
        ///
        /// Usage:
        ///   using (var utf8 = "Hello, AI!".ToUtf8(Allocator.Temp))
        ///   {
        ///       var span = utf8.AsSpan();
        ///       AstralNative.astral_session_feed(session, span, finalize: 1);
        ///   } // utf8 disposed here
        /// </summary>
        public static NativeArray<byte> ToUtf8(this string str, Allocator allocator)
        {
            if (string.IsNullOrEmpty(str))
            {
                return new NativeArray<byte>(0, allocator);
            }

            // Get UTF-8 bytes (managed allocation - GC pressure)
            byte[] bytes = Encoding.UTF8.GetBytes(str);

            // Allocate NativeArray
            var array = new NativeArray<byte>(bytes.Length, allocator, NativeArrayOptions.UninitializedMemory);

            // Copy bytes (unsafe memcpy for performance)
            unsafe
            {
                fixed (byte* src = bytes)
                {
                    UnsafeUtility.MemCpy(
                        NativeArrayUnsafeUtility.GetUnsafePtr(array),
                        src,
                        bytes.Length
                    );
                }
            }

            return array;
        }

        /// <summary>
        /// Convert C# string to NativeArray{byte} with explicit length (UTF-8).
        /// Useful for pre-allocated buffers or string pooling.
        /// </summary>
        public static unsafe int ToUtf8(this string str, NativeArray<byte> dest)
        {
            if (string.IsNullOrEmpty(str))
            {
                return 0;
            }

            // Get UTF-8 bytes
            byte[] bytes = Encoding.UTF8.GetBytes(str);

            if (bytes.Length > dest.Length)
            {
                throw new ArgumentException($"Destination buffer too small: need {bytes.Length}, have {dest.Length}");
            }

            // Copy bytes
            fixed (byte* src = bytes)
            {
                UnsafeUtility.MemCpy(
                    NativeArrayUnsafeUtility.GetUnsafePtr(dest),
                    src,
                    bytes.Length
                );
            }

            return bytes.Length;
        }

        // ====== NativeArray<byte> (UTF-8) -> String ======

        /// <summary>
        /// Convert NativeArray{byte} (UTF-8) to C# string.
        ///
        /// 
        /// - Input MUST be valid UTF-8 (no validation, undefined behavior on invalid UTF-8)
        /// - Allocates managed string (GC pressure)
        /// - Use sparingly in hot paths (prefer working with NativeArray directly)
        ///
        /// Usage:
        ///   var buffer = new NativeArray{byte}(1024, Allocator.Temp);
        ///   int bytesRead = AstralNative.astral_stream_read(session, buffer.AsMutSpan(), 1000);
        ///   var slice = new NativeSlice{byte}(buffer, 0, bytesRead);
        ///   string response = slice.ToUtf8String();
        ///   buffer.Dispose();
        /// </summary>
        public static unsafe string ToUtf8String(this NativeArray<byte> array)
        {
            if (array.Length == 0)
            {
                return string.Empty;
            }

            byte* ptr = (byte*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(array);
            return Encoding.UTF8.GetString(ptr, array.Length);
        }

        /// <summary>
        /// Convert NativeSlice{byte} (UTF-8) to C# string.
        /// Useful after reading partial data from Astral stream.
        /// </summary>
        public static unsafe string ToUtf8String(this NativeSlice<byte> slice)
        {
            if (slice.Length == 0)
            {
                return string.Empty;
            }

            byte* ptr = (byte*)slice.GetUnsafeReadOnlyPtr();
            return Encoding.UTF8.GetString(ptr, slice.Length);
        }

        /// <summary>
        /// Convert AstralSpanU8 to C# string (UTF-8).
        ///  Span must remain valid during conversion.
        /// </summary>
        public static unsafe string ToUtf8String(this AstralNative.AstralSpanU8 span)
        {
            if (span.len == 0 || span.data == IntPtr.Zero)
            {
                return string.Empty;
            }

            byte* ptr = (byte*)span.data;
            return Encoding.UTF8.GetString(ptr, (int)span.len);
        }

        // ====== Utility Methods ======

        /// <summary>
        /// Get maximum UTF-8 byte count for a C# string.
        /// Use for pre-allocating buffers.
        ///
        /// Note: This returns maximum size (each char could be up to 4 bytes in UTF-8).
        /// Actual size may be smaller.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetMaxUtf8ByteCount(string str)
        {
            return Encoding.UTF8.GetMaxByteCount(str.Length);
        }

        /// <summary>
        /// Get exact UTF-8 byte count for a C# string.
        /// Use when exact size is needed.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetUtf8ByteCount(string str)
        {
            return Encoding.UTF8.GetByteCount(str);
        }

        /// <summary>
        /// Validate UTF-8 byte sequence.
        /// Returns true if valid UTF-8, false otherwise.
        ///
        /// PERFORMANCE: This is expensive (O(n) scan). Use sparingly.
        /// </summary>
        public static unsafe bool IsValidUtf8(this NativeArray<byte> array)
        {
            byte* ptr = (byte*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(array);
            try
            {
                // This will throw on invalid UTF-8
                Encoding.UTF8.GetString(ptr, array.Length);
                return true;
            }
            catch (DecoderFallbackException)
            {
                return false;
            }
        }

        /// <summary>
        /// Create a NativeArray view over managed byte array (zero-copy).
        ///  Managed array must be pinned or this is unsafe.
        /// Use with extreme caution.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe NativeArray<byte> AsNativeArray(byte[] managedArray)
        {
            //  This is unsafe if managed array is moved by GC
            // Only use if array is pinned or within fixed block
            var array = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<byte>(
                managedArray,
                managedArray.Length,
                Allocator.None
            );

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            // Set safety handle (debug only)
            var safety = AtomicSafetyHandle.Create();
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref array, safety);
#endif

            return array;
        }
    }
}
