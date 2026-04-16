using System;
using System.Runtime.CompilerServices;
using System.Text;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    /// <summary>
    /// Creates Astral span views over Unity NativeArray and NativeSlice storage.
    /// </summary>
    public static class AstralNativeArray
    {
        /// <summary>
        /// Creates a read-only span view. The NativeArray must stay alive while
        /// native code uses the span.
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
        /// Creates a mutable span view. The NativeArray must stay writable and
        /// alive while native code uses the span.
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
        /// Creates a read-only span view over a NativeSlice.
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
        /// Creates a mutable span view over a NativeSlice.
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

        /// <summary>
        /// Encodes a managed string into a caller-disposed UTF-8 NativeArray.
        /// </summary>
        public static NativeArray<byte> ToUtf8(this string str, Allocator allocator)
        {
            if (string.IsNullOrEmpty(str))
            {
                return new NativeArray<byte>(0, allocator);
            }

            byte[] bytes = Encoding.UTF8.GetBytes(str);

            var array = new NativeArray<byte>(bytes.Length, allocator, NativeArrayOptions.UninitializedMemory);

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

        /// <summary>
        /// Decodes a UTF-8 NativeArray into a managed string.
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
        /// Returns true when the byte array is valid UTF-8.
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
        /// Copy a managed byte array into a temporary NativeArray.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NativeArray<byte> AsNativeArray(byte[] managedArray)
        {
            return new NativeArray<byte>(managedArray, Allocator.Temp);
        }
    }
}
