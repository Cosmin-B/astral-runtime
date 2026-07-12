using System;
using System.IO;
using System.Text;
using Astral.Runtime;
using Unity.Collections;
using UnityEngine;

namespace Astral.Examples
{
    public sealed class LocalKnowledgeExample : MonoBehaviour
    {
        [Header("Embedding model")]
        [SerializeField] private string modelPath = "mock";
        [SerializeField] private string backendName = "mock";

        [Header("Knowledge")]
        [SerializeField] [TextArea(8, 20)] private string documentText =
            "The north gate opens at dawn. The apothecary lives beside the river. " +
            "Silver keys open the observatory. The ferry leaves the west pier at sunset.";
        [SerializeField] private uint chunkWords = 12;
        [SerializeField] private uint overlapWords = 2;
        [SerializeField] private string query = "When does the ferry leave?";
        [SerializeField] private uint topK = 3;
        [SerializeField] private string snapshotFileName = "astral-local-knowledge.index";

        private AstralModel model;
        private AstralEmbedder embedder;
        private AstralMemoryIndex index;
        private NativeArray<byte> documentBytes;
        private NativeArray<AstralNative.AstralChunkRange> chunkRanges;

        private string SnapshotPath => Path.Combine(Application.persistentDataPath, snapshotFileName);

        public void BuildIndex()
        {
            if (!EnsureRuntime())
            {
                return;
            }

            try
            {
                EnsureEmbedder();
                PrepareDocumentMetadata();
                index?.Dispose();
                index = AstralMemoryIndex.Create(NewIndexConfig());

                int dim = checked((int)model.GetEmbeddingDim());
                var records = new NativeArray<AstralNative.AstralMemoryRecord>(
                    chunkRanges.Length,
                    Allocator.Temp,
                    NativeArrayOptions.UninitializedMemory);
                try
                {
                    using (var vectors = new NativeArray<float>(
                        checked(chunkRanges.Length * dim),
                        Allocator.Temp,
                        NativeArrayOptions.UninitializedMemory))
                    using (var vector = new NativeArray<float>(dim, Allocator.Temp, NativeArrayOptions.UninitializedMemory))
                    {
                        for (int chunkIndex = 0; chunkIndex < chunkRanges.Length; ++chunkIndex)
                        {
                            var range = chunkRanges[chunkIndex];
                            string chunkText = AstralChunker.CopyTextToString(documentBytes, ref range);
                            embedder.Embed(chunkText, vector);
                            NativeArray<float>.Copy(vector, 0, vectors, chunkIndex * dim, dim);
                            records[chunkIndex] = AstralMemoryIndex.Record(
                                key: (ulong)(chunkIndex + 1),
                                documentId: range.document_id,
                                chunkId: (uint)chunkIndex,
                                groupId: range.group_id);
                        }
                        index.AddBatch(records, vectors);
                    }
                }
                finally
                {
                    records.Dispose();
                }

                Debug.Log($"[LocalKnowledge] Indexed {index.Count()} chunks.");
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[LocalKnowledge] {ex.Message}");
            }
        }

        public void Search()
        {
            if (index == null || !index.IsValid)
            {
                Debug.LogError("[LocalKnowledge] Build or load the index before searching.");
                return;
            }

            try
            {
                int dim = checked((int)index.Dim);
                int resultCapacity = Math.Max(1, Math.Min((int)topK, chunkRanges.Length));
                using (var queryVector = new NativeArray<float>(dim, Allocator.Temp, NativeArrayOptions.UninitializedMemory))
                using (var results = new NativeArray<AstralNative.AstralMemorySearchResult>(
                    resultCapacity,
                    Allocator.Temp,
                    NativeArrayOptions.UninitializedMemory))
                {
                    embedder.Embed(query, queryVector);
                    var search = AstralMemoryIndex.SearchDesc((uint)resultCapacity);
                    uint written = index.Search(ref search, queryVector, results);
                    for (int resultIndex = 0; resultIndex < (int)written; ++resultIndex)
                    {
                        var result = results[resultIndex];
                        if (result.chunk_id >= (uint)chunkRanges.Length)
                        {
                            continue;
                        }
                        var range = chunkRanges[(int)result.chunk_id];
                        string chunkText = AstralChunker.CopyTextToString(documentBytes, ref range);
                        Debug.Log($"[LocalKnowledge] {result.score:F3}: {chunkText}");
                    }
                }
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[LocalKnowledge] {ex.Message}");
            }
        }

        public void Save()
        {
            if (index == null || !index.IsValid)
            {
                Debug.LogError("[LocalKnowledge] Build or load the index before saving.");
                return;
            }
            File.WriteAllBytes(SnapshotPath, index.Save());
            Debug.Log($"[LocalKnowledge] Saved index to {SnapshotPath}.");
        }

        public void Load()
        {
            if (!EnsureRuntime())
            {
                return;
            }
            if (!File.Exists(SnapshotPath))
            {
                Debug.LogError($"[LocalKnowledge] Snapshot not found: {SnapshotPath}");
                return;
            }

            try
            {
                EnsureEmbedder();
                PrepareDocumentMetadata();
                index?.Dispose();
                index = AstralMemoryIndex.Load(NewIndexConfig(), File.ReadAllBytes(SnapshotPath));
                Debug.Log($"[LocalKnowledge] Loaded {index.Count()} chunks from {SnapshotPath}.");
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[LocalKnowledge] {ex.Message}");
            }
        }

        private bool EnsureRuntime()
        {
            if (AstralRuntime.IsInitialized)
            {
                return true;
            }
            Debug.LogError("[LocalKnowledge] Add AstralRuntimeInitializer before running the sample.");
            return false;
        }

        private void EnsureEmbedder()
        {
            if (embedder != null && embedder.IsValid)
            {
                return;
            }
            model = AstralModel.Load(modelPath, new AstralModelConfig
            {
                backendName = backendName,
                embeddingsOnly = true
            });
            embedder = AstralEmbedder.Create(model);
        }

        private void PrepareDocumentMetadata()
        {
            if (string.IsNullOrWhiteSpace(documentText))
            {
                throw new AstralException("Document text is empty.");
            }

            DisposeDocumentMetadata();
            documentBytes = new NativeArray<byte>(Encoding.UTF8.GetBytes(documentText), Allocator.Persistent);
            var chunker = AstralChunker.TextDesc(
                AstralNative.AstralChunkMode.Word,
                Math.Max(1u, chunkWords),
                Math.Min(overlapWords, Math.Max(1u, chunkWords) - 1u),
                documentId: 1,
                groupId: 1);
            chunkRanges = AstralChunker.Ranges(ref chunker, documentBytes, Allocator.Persistent);
            if (chunkRanges.Length == 0)
            {
                throw new AstralException("The document did not produce any chunks.");
            }
        }

        private AstralMemoryIndexConfig NewIndexConfig()
        {
            return new AstralMemoryIndexConfig
            {
                dim = model.GetEmbeddingDim(),
                capacity = (uint)Math.Max(1, chunkRanges.Length),
                metric = AstralNative.AstralMemoryMetric.Cosine,
                indexKind = AstralNative.AstralMemoryIndexKind.Flat,
                storageKind = AstralNative.AstralMemoryStorageKind.F32
            };
        }

        private void DisposeDocumentMetadata()
        {
            if (chunkRanges.IsCreated)
            {
                chunkRanges.Dispose();
                chunkRanges = default;
            }
            if (documentBytes.IsCreated)
            {
                documentBytes.Dispose();
                documentBytes = default;
            }
        }

        private void OnDestroy()
        {
            index?.Dispose();
            embedder?.Dispose();
            model?.Dispose();
            DisposeDocumentMetadata();
        }
    }
}
