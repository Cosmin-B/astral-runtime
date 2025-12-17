// AstralChunker.cs - Unity wrapper for native text and token chunk planning.
//
// NativeArray paths keep ownership with the caller and avoid managed string
// materialization in ingest hot paths.

using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    /// <summary>
    /// Allocation-free text and token chunk planning wrapper.
    /// </summary>
    public static class AstralChunker
    {
        public const uint DefaultTextUnitsPerChunk = 512;
        public const uint DefaultTextOverlapUnits = 64;
        public const uint DefaultTokenUnitsPerChunk = 256;
        public const uint DefaultTokenOverlapUnits = 32;
        public const uint DefaultDocumentId = 0;
        public const uint DefaultGroupId = 0;
        private const int FirstByteIndex = 0;
        private const uint EmptyByteCount = 0;

        public static AstralNative.AstralChunkerDesc TextDesc(
            AstralNative.AstralChunkMode mode = AstralNative.AstralChunkMode.Word,
            uint maxUnits = DefaultTextUnitsPerChunk,
            uint overlapUnits = DefaultTextOverlapUnits,
            uint documentId = DefaultDocumentId,
            uint groupId = DefaultGroupId,
            AstralNative.AstralChunkFlags flags = AstralNative.AstralChunkFlags.None)
        {
            return new AstralNative.AstralChunkerDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralChunkerDesc>(),
                mode = mode,
                max_units = maxUnits,
                overlap_units = overlapUnits,
                document_id = documentId,
                group_id = groupId,
                flags = flags,
                delimiters = default
            };
        }

        public static AstralNative.AstralChunkerDesc TokenDesc(
            uint maxUnits = DefaultTokenUnitsPerChunk,
            uint overlapUnits = DefaultTokenOverlapUnits,
            uint documentId = DefaultDocumentId,
            uint groupId = DefaultGroupId,
            AstralNative.AstralChunkFlags flags = AstralNative.AstralChunkFlags.None)
        {
            return new AstralNative.AstralChunkerDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralChunkerDesc>(),
                mode = AstralNative.AstralChunkMode.Token,
                max_units = maxUnits,
                overlap_units = overlapUnits,
                document_id = documentId,
                group_id = groupId,
                flags = flags,
                delimiters = default
            };
        }

        public static uint CountText(ref AstralNative.AstralChunkerDesc desc, NativeArray<byte> utf8Text)
        {
            if (!utf8Text.IsCreated)
            {
                throw new ArgumentException("utf8Text must be created", nameof(utf8Text));
            }

            return CountTextSpan(ref desc, AstralNative.AstralSpanU8.FromNativeArray(utf8Text));
        }

        public static uint CountText(ref AstralNative.AstralChunkerDesc desc, string text)
        {
            NativeArray<byte> textArray;
            var textSpan = AstralNative.AstralSpanU8.FromString(text, out textArray);
            try
            {
                return CountTextSpan(ref desc, textSpan);
            }
            finally
            {
                if (textArray.IsCreated)
                {
                    textArray.Dispose();
                }
            }
        }

        public static unsafe uint Ranges(ref AstralNative.AstralChunkerDesc desc, NativeArray<byte> utf8Text, NativeArray<AstralNative.AstralChunkRange> outRanges)
        {
            if (!utf8Text.IsCreated)
            {
                throw new ArgumentException("utf8Text must be created", nameof(utf8Text));
            }
            if (!outRanges.IsCreated)
            {
                throw new ArgumentException("outRanges must be created", nameof(outRanges));
            }

            EnsureDescSize(ref desc);
            int err = AstralNative.astral_chunk_ranges(
                ref desc,
                AstralNative.AstralSpanU8.FromNativeArray(utf8Text),
                (AstralNative.AstralChunkRange*)outRanges.GetUnsafePtr(),
                (uint)outRanges.Length,
                out uint written);
            ThrowIfError(err, "astral_chunk_ranges");
            return written;
        }

        public static NativeArray<AstralNative.AstralChunkRange> Ranges(ref AstralNative.AstralChunkerDesc desc, NativeArray<byte> utf8Text, Allocator allocator)
        {
            uint count = CountText(ref desc, utf8Text);
            var ranges = new NativeArray<AstralNative.AstralChunkRange>((int)count, allocator, NativeArrayOptions.UninitializedMemory);
            try
            {
                Ranges(ref desc, utf8Text, ranges);
                return ranges;
            }
            catch
            {
                ranges.Dispose();
                throw;
            }
        }

        public static uint CountTokens(ref AstralNative.AstralChunkerDesc desc, uint tokenCount)
        {
            EnsureDescSize(ref desc);
            int err = AstralNative.astral_token_chunk_count(ref desc, tokenCount, out uint count);
            ThrowIfError(err, "astral_token_chunk_count");
            return count;
        }

        public static unsafe uint TokenRanges(ref AstralNative.AstralChunkerDesc desc, uint tokenCount, NativeArray<AstralNative.AstralChunkRange> outRanges)
        {
            if (!outRanges.IsCreated)
            {
                throw new ArgumentException("outRanges must be created", nameof(outRanges));
            }

            EnsureDescSize(ref desc);
            int err = AstralNative.astral_token_chunk_ranges(
                ref desc,
                tokenCount,
                (AstralNative.AstralChunkRange*)outRanges.GetUnsafePtr(),
                (uint)outRanges.Length,
                out uint written);
            ThrowIfError(err, "astral_token_chunk_ranges");
            return written;
        }

        public static NativeArray<AstralNative.AstralChunkRange> TokenRanges(ref AstralNative.AstralChunkerDesc desc, uint tokenCount, Allocator allocator)
        {
            uint count = CountTokens(ref desc, tokenCount);
            var ranges = new NativeArray<AstralNative.AstralChunkRange>((int)count, allocator, NativeArrayOptions.UninitializedMemory);
            try
            {
                TokenRanges(ref desc, tokenCount, ranges);
                return ranges;
            }
            catch
            {
                ranges.Dispose();
                throw;
            }
        }

        public static uint CopyText(NativeArray<byte> utf8Text, ref AstralNative.AstralChunkRange range, NativeArray<byte> outText)
        {
            if (!utf8Text.IsCreated)
            {
                throw new ArgumentException("utf8Text must be created", nameof(utf8Text));
            }
            if (!outText.IsCreated)
            {
                throw new ArgumentException("outText must be created", nameof(outText));
            }

            EnsureRangeSize(ref range);
            int err = AstralNative.astral_chunk_text_copy(
                AstralNative.AstralSpanU8.FromNativeArray(utf8Text),
                ref range,
                AstralNative.AstralMutSpanU8.FromNativeArray(outText),
                out uint written);
            ThrowIfError(err, "astral_chunk_text_copy");
            return written;
        }

        public static string CopyTextToString(NativeArray<byte> utf8Text, ref AstralNative.AstralChunkRange range)
        {
            EnsureRangeSize(ref range);
            uint byteCount = range.byte_end - range.byte_begin;
            if (byteCount == EmptyByteCount)
            {
                return string.Empty;
            }

            using (var bytes = new NativeArray<byte>((int)byteCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory))
            {
                uint written = CopyText(utf8Text, ref range, bytes);
                var slice = new NativeSlice<byte>(bytes, FirstByteIndex, (int)written);
                return slice.ToUtf8String();
            }
        }

        private static void EnsureDescSize(ref AstralNative.AstralChunkerDesc desc)
        {
            desc.size = (uint)Marshal.SizeOf<AstralNative.AstralChunkerDesc>();
        }

        private static uint CountTextSpan(ref AstralNative.AstralChunkerDesc desc, AstralNative.AstralSpanU8 textSpan)
        {
            EnsureDescSize(ref desc);
            int err = AstralNative.astral_chunk_count(ref desc, textSpan, out uint count);
            ThrowIfError(err, "astral_chunk_count");
            return count;
        }

        private static void EnsureRangeSize(ref AstralNative.AstralChunkRange range)
        {
            range.size = (uint)Marshal.SizeOf<AstralNative.AstralChunkRange>();
        }

        private static void ThrowIfError(int err, string call)
        {
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"{call} failed: {AstralRuntime.GetErrorString(err)}", err);
            }
        }
    }
}
