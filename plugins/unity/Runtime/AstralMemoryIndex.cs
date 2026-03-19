// AstralMemoryIndex.cs - Unity wrapper for native embedding retrieval indexes.

using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    /// <summary>
    /// Owned native memory index handle for RAG retrieval.
    /// </summary>
    public sealed class AstralMemoryIndex : IDisposable
    {
        public const uint DefaultCapacity = 4096;
        public const uint DefaultTopK = 8;
        public const uint DefaultRecordGroupId = 0;
        public const uint DefaultRecordFlags = 0;
        public const uint DefaultSearchGroupId = AstralNative.ASTRAL_MEMORY_GROUP_ANY;
        public const uint DefaultSearchFlags = 0;
        private const ulong MaxUnityByteArrayLength = int.MaxValue;
        private const int EmptySnapshotLength = 0;
        private const ulong EmptySnapshotBytes = 0;

        private AstralNative.AstralHandle m_handle;
        private uint m_dim;
        private bool m_disposed;

        public bool IsValid => !m_disposed && m_handle.IsValid;
        public uint Dim => m_dim;
        public AstralNative.AstralHandle Handle => m_handle;

        public static AstralMemoryIndex Create(AstralMemoryIndexConfig config)
        {
            if (config == null)
            {
                throw new ArgumentNullException(nameof(config));
            }

            var desc = config.ToNativeDesc();
            int err = AstralNative.astral_memory_create(ref desc, out var handle);
            ThrowIfError(err, "astral_memory_create");

            return new AstralMemoryIndex
            {
                m_handle = handle,
                m_dim = config.dim,
                m_disposed = false
            };
        }

        public static AstralMemoryIndex Load(AstralMemoryIndexConfig config, byte[] snapshot)
        {
            if (config == null)
            {
                throw new ArgumentNullException(nameof(config));
            }
            if (snapshot == null || snapshot.Length == EmptySnapshotLength)
            {
                throw new ArgumentException("snapshot must not be empty", nameof(snapshot));
            }

            using (var bytes = new NativeArray<byte>(snapshot, Allocator.Temp))
            {
                var desc = config.ToNativeDesc();
                int err = AstralNative.astral_memory_load(
                    ref desc,
                    AstralNative.AstralSpanU8.FromNativeArray(bytes),
                    out var handle);
                ThrowIfError(err, "astral_memory_load");

                return new AstralMemoryIndex
                {
                    m_handle = handle,
                    m_dim = config.dim,
                    m_disposed = false
                };
            }
        }

        public uint Count()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_memory_count(m_handle, out uint count);
            ThrowIfError(err, "astral_memory_count");
            return count;
        }

        public AstralNative.AstralMemoryStats GetStats()
        {
            ThrowIfInvalid();
            var stats = new AstralNative.AstralMemoryStats
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralMemoryStats>()
            };
            int err = AstralNative.astral_memory_stats(m_handle, ref stats);
            ThrowIfError(err, "astral_memory_stats");
            return stats;
        }

        public void Clear()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_memory_clear(m_handle);
            ThrowIfError(err, "astral_memory_clear");
        }

        public unsafe void AddBatch(NativeArray<AstralNative.AstralMemoryRecord> records, NativeArray<float> vectors)
        {
            ThrowIfInvalid();
            ValidateRecords(records);
            ValidateVectorBuffer(records.Length, vectors);

            int err = AstralNative.astral_memory_add_batch(
                m_handle,
                (AstralNative.AstralMemoryRecord*)records.GetUnsafeReadOnlyPtr(),
                (float*)vectors.GetUnsafeReadOnlyPtr(),
                (uint)records.Length);
            ThrowIfError(err, "astral_memory_add_batch");
        }

        public void Remove(ulong key)
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_memory_remove(m_handle, key);
            ThrowIfError(err, "astral_memory_remove");
        }

        public unsafe uint Search(ref AstralNative.AstralMemorySearchDesc desc, NativeArray<float> query, NativeArray<AstralNative.AstralMemorySearchResult> outResults)
        {
            ThrowIfInvalid();
            ValidateQuery(query);
            ValidateResults(outResults);
            EnsureSearchDescSize(ref desc);

            int err = AstralNative.astral_memory_search(
                m_handle,
                ref desc,
                (float*)query.GetUnsafeReadOnlyPtr(),
                (AstralNative.AstralMemorySearchResult*)outResults.GetUnsafePtr(),
                (uint)outResults.Length,
                out uint written);
            ThrowIfError(err, "astral_memory_search");
            return written;
        }

        public unsafe AstralMemorySearchCursor BeginSearch(ref AstralNative.AstralMemorySearchDesc desc, NativeArray<float> query)
        {
            ThrowIfInvalid();
            ValidateQuery(query);
            EnsureSearchDescSize(ref desc);

            int err = AstralNative.astral_memory_search_begin(
                m_handle,
                ref desc,
                (float*)query.GetUnsafeReadOnlyPtr(),
                out var cursor);
            ThrowIfError(err, "astral_memory_search_begin");
            return new AstralMemorySearchCursor(cursor);
        }

        public byte[] Save()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_memory_save_size(m_handle, out ulong byteCount);
            ThrowIfError(err, "astral_memory_save_size");
            if (byteCount == EmptySnapshotBytes)
            {
                return Array.Empty<byte>();
            }
            if (byteCount > MaxUnityByteArrayLength)
            {
                throw new AstralException("Memory index snapshot is too large for a Unity byte array.", AstralNative.ASTRAL_E_NOMEM);
            }

            using (var bytes = new NativeArray<byte>((int)byteCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory))
            {
                err = AstralNative.astral_memory_save(m_handle, AstralNative.AstralMutSpanU8.FromNativeArray(bytes), out ulong written);
                ThrowIfError(err, "astral_memory_save");

                int writtenLength = (int)written;
                var managed = new byte[writtenLength];
                NativeArray<byte>.Copy(bytes, managed, writtenLength);
                return managed;
            }
        }

        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_memory_destroy(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_disposed = true;
        }

        public static AstralNative.AstralMemorySearchDesc SearchDesc(
            uint topK = DefaultTopK,
            uint groupId = DefaultSearchGroupId,
            uint flags = DefaultSearchFlags)
        {
            return new AstralNative.AstralMemorySearchDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralMemorySearchDesc>(),
                top_k = topK,
                group_id = groupId,
                flags = flags
            };
        }

        public static AstralNative.AstralMemoryRecord Record(
            ulong key,
            uint documentId,
            uint chunkId,
            uint groupId = DefaultRecordGroupId,
            uint flags = DefaultRecordFlags)
        {
            return new AstralNative.AstralMemoryRecord
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralMemoryRecord>(),
                group_id = groupId,
                key = key,
                document_id = documentId,
                chunk_id = chunkId,
                flags = flags
            };
        }

        private void ThrowIfInvalid()
        {
            if (!IsValid)
            {
                throw new AstralException("Memory index is not valid (disposed or not created).");
            }
        }

        private static void ThrowIfError(int err, string call)
        {
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"{call} failed: {AstralRuntime.GetErrorString(err)}", err);
            }
        }

        private static void EnsureSearchDescSize(ref AstralNative.AstralMemorySearchDesc desc)
        {
            desc.size = (uint)Marshal.SizeOf<AstralNative.AstralMemorySearchDesc>();
        }

        private static void ValidateRecords(NativeArray<AstralNative.AstralMemoryRecord> records)
        {
            if (!records.IsCreated)
            {
                throw new ArgumentException("records must be created", nameof(records));
            }
        }

        private void ValidateVectorBuffer(int recordCount, NativeArray<float> vectors)
        {
            if (!vectors.IsCreated)
            {
                throw new ArgumentException("vectors must be created", nameof(vectors));
            }

            ulong requiredScalars = (ulong)recordCount * m_dim;
            if ((ulong)vectors.Length < requiredScalars)
            {
                throw new ArgumentException("vectors is smaller than records.Length * Dim", nameof(vectors));
            }
        }

        private void ValidateQuery(NativeArray<float> query)
        {
            if (!query.IsCreated)
            {
                throw new ArgumentException("query must be created", nameof(query));
            }
            if ((ulong)query.Length < m_dim)
            {
                throw new ArgumentException("query is smaller than Dim", nameof(query));
            }
        }

        private static void ValidateResults(NativeArray<AstralNative.AstralMemorySearchResult> results)
        {
            if (!results.IsCreated)
            {
                throw new ArgumentException("outResults must be created", nameof(results));
            }
        }
    }

    /// <summary>
    /// Native memory index creation settings.
    /// </summary>
    [Serializable]
    public sealed class AstralMemoryIndexConfig
    {
        public uint dim;
        public uint capacity = AstralMemoryIndex.DefaultCapacity;
        public AstralNative.AstralMemoryMetric metric = AstralNative.AstralMemoryMetric.Cosine;
        public AstralNative.AstralMemoryIndexKind indexKind = AstralNative.AstralMemoryIndexKind.Flat;
        public AstralNative.AstralMemoryStorageKind storageKind = AstralNative.AstralMemoryStorageKind.F32;
        public uint graphNeighbors;
        public uint graphSearch;

        internal AstralNative.AstralMemoryIndexDesc ToNativeDesc()
        {
            return new AstralNative.AstralMemoryIndexDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralMemoryIndexDesc>(),
                dim = dim,
                capacity = capacity,
                metric = metric,
                index_kind = indexKind,
                graph_neighbors = graphNeighbors,
                graph_search = graphSearch,
                storage_kind = storageKind
            };
        }
    }

    /// <summary>
    /// Owned native memory search cursor.
    /// </summary>
    public sealed class AstralMemorySearchCursor : IDisposable
    {
        private AstralNative.AstralHandle m_handle;
        private bool m_disposed;

        internal AstralMemorySearchCursor(AstralNative.AstralHandle handle)
        {
            m_handle = handle;
            m_disposed = false;
        }

        public bool IsValid => !m_disposed && m_handle.IsValid;
        internal AstralNative.AstralHandle Handle => m_handle;

        public unsafe uint Fetch(NativeArray<AstralNative.AstralMemorySearchResult> outResults)
        {
            if (!IsValid)
            {
                throw new AstralException("Memory search cursor is not valid (disposed or not created).");
            }
            if (!outResults.IsCreated)
            {
                throw new ArgumentException("outResults must be created", nameof(outResults));
            }

            int err = AstralNative.astral_memory_search_fetch(
                m_handle,
                (AstralNative.AstralMemorySearchResult*)outResults.GetUnsafePtr(),
                (uint)outResults.Length,
                out uint written);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_memory_search_fetch failed: {AstralRuntime.GetErrorString(err)}", err);
            }
            return written;
        }

        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_memory_search_end(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_disposed = true;
        }
    }
}
