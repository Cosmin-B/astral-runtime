// AstralModel.cs - GGUF model handle with explicit native ownership.
//
//  Dispose releases the native model handle; prefer a using scope.
//  No GC allocations in hot paths.
//  Thread-safety: Safe to load from multiple threads; must not be in use when releasing.

using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;

namespace Astral.Runtime
{
    /// <summary>
    /// GGUF model handle.
    /// Implements IDisposable to release the native model handle deterministically.
    /// Thread-safety: Safe to load from multiple threads; must not be in use when releasing.
    /// </summary>
    public class AstralModel : IDisposable
    {
        private AstralNative.AstralHandle m_handle;
        private bool m_disposed = false;

        /// <summary>
        /// Get native handle (for internal use).
        /// </summary>
        internal AstralNative.AstralHandle Handle => m_handle;

        /// <summary>
        /// Check if model is valid.
        /// </summary>
        public bool IsValid => !m_disposed && m_handle.IsValid;

        /// <summary>
        /// Get the model embedding dimension (number of floats per embedding vector).
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        public uint GetEmbeddingDim()
        {
            if (!IsValid)
            {
                throw new AstralException("Model is not valid (disposed or not loaded).");
            }

            int err = AstralNative.astral_model_embedding_dim(m_handle, out uint dim);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_model_embedding_dim failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return dim;
        }

        /// <summary>
        /// Query model capability bits (provider-agnostic fast path for wrappers).
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        public ulong GetCaps()
        {
            if (!IsValid)
            {
                throw new AstralException("Model is not valid (disposed or not loaded).");
            }

            int err = AstralNative.astral_model_caps(m_handle, out ulong caps);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_model_caps failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return caps;
        }

        /// <summary>
        /// Query model limits (best-effort; fields may be 0 if unknown).
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        public AstralNative.AstralModelLimits GetLimits()
        {
            if (!IsValid)
            {
                throw new AstralException("Model is not valid (disposed or not loaded).");
            }

            int err = AstralNative.astral_model_limits(m_handle, out var limits);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_model_limits failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return limits;
        }

        /// <summary>
        /// Initialize media (vision/audio) support for this model.
        /// </summary>
        public void InitMedia(ref AstralNative.AstralModelMediaDesc desc)
        {
            if (!IsValid)
            {
                throw new AstralException("Model is not valid (disposed or not loaded).");
            }

            desc.size = (uint)Marshal.SizeOf<AstralNative.AstralModelMediaDesc>();
            int err = AstralNative.astral_model_media_init(m_handle, ref desc);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_model_media_init failed: {AstralRuntime.GetErrorString(err)}", err);
            }
        }

        /// <summary>
        /// Initialize media (vision/audio) support using a media/projector GGUF path.
        /// </summary>
        public void InitMediaFromPath(
            string mediaPath,
            AstralNative.AstralMediaFlags flags = AstralNative.AstralMediaFlags.None,
            uint imageMinTokens = 0,
            uint imageMaxTokens = 0,
            AstralNative.AstralGpuRouteFlags gpuRouteFlags = AstralNative.AstralGpuRouteFlags.None,
            int gpuDevice = 0,
            ulong gpuDeviceMask = 0,
            IntPtr gpuStream = default)
        {
            if (!IsValid)
            {
                throw new AstralException("Model is not valid (disposed or not loaded).");
            }
            if (string.IsNullOrEmpty(mediaPath))
            {
                throw new ArgumentNullException(nameof(mediaPath));
            }

            NativeArray<byte> pathArray;
            var pathSpan = AstralNative.AstralSpanU8.FromString(mediaPath, out pathArray);

            try
            {
                var desc = new AstralNative.AstralModelMediaDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralModelMediaDesc>(),
                    source_kind = AstralNative.AstralModelSourceKind.Path,
                    flags = (uint)flags,
                    image_min_tokens = imageMinTokens,
                    image_max_tokens = imageMaxTokens,
                    media_path = pathSpan,
                    gpu_device = gpuDevice,
                    gpu_route_flags = (uint)gpuRouteFlags,
                    gpu_device_mask = gpuDeviceMask,
                    gpu_stream = gpuStream
                };

                int err = AstralNative.astral_model_media_init(m_handle, ref desc);
                if (err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"astral_model_media_init failed: {AstralRuntime.GetErrorString(err)}", err);
                }
            }
            finally
            {
                if (pathArray.IsCreated)
                {
                    pathArray.Dispose();
                }
            }
        }

        /// <summary>
        /// Query media (vision/audio) info for this model.
        /// </summary>
        public AstralNative.AstralMediaInfo GetMediaInfo()
        {
            if (!IsValid)
            {
                throw new AstralException("Model is not valid (disposed or not loaded).");
            }

            var info = new AstralNative.AstralMediaInfo
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralMediaInfo>()
            };
            int err = AstralNative.astral_model_media_info(m_handle, ref info);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_model_media_info failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return info;
        }

        /// <summary>
        /// Load a GGUF model.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        /// <param name="modelPath">Path to GGUF model file</param>
        /// <param name="config">Model configuration (null = default)</param>
        /// <returns>Loaded model (must be disposed)</returns>
        /// <exception cref="AstralException">Thrown if model loading fails</exception>
        public static AstralModel Load(string modelPath, AstralModelConfig config = null)
        {
            if (!AstralRuntime.IsInitialized)
            {
                throw new AstralException("Astral runtime not initialized. Call AstralRuntime.Initialize() first.");
            }

            if (string.IsNullOrEmpty(modelPath))
            {
                throw new ArgumentNullException(nameof(modelPath));
            }

            config = config ?? AstralModelConfig.Default;

            // Convert model path to UTF-8 (no NUL termination required)
            NativeArray<byte> pathArray;
            var pathSpan = AstralNative.AstralSpanU8.FromString(modelPath, out pathArray);

            // Optional backend override
            NativeArray<byte> backendArray = default;
            var backendSpan = string.IsNullOrEmpty(config.backendName)
                ? new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 }
                : AstralNative.AstralSpanU8.FromString(config.backendName, out backendArray);

            try
            {
                var desc = new AstralNative.AstralModelDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralModelDesc>(),
                    source_kind = AstralNative.AstralModelSourceKind.Path,
                    model_path = pathSpan,
                    backend_name = backendSpan,
                    gpu_layers = config.gpuLayers,
                    n_ctx = config.contextSize,
                    n_batch = config.batchSize,
                    n_threads = config.threads,
                    embeddings_only = (byte)(config.embeddingsOnly ? 1 : 0)
                };

                AstralNative.AstralHandle handle;
                int err = AstralNative.astral_model_load(ref desc, out handle);

                if (err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"Failed to load model '{modelPath}': {AstralRuntime.GetErrorString(err)}", err);
                }

                Debug.Log($"[Astral] Model loaded: {modelPath} (ctx={config.contextSize}, batch={config.batchSize}, gpu_layers={config.gpuLayers})");

                return new AstralModel { m_handle = handle };
            }
            finally
            {
                if (pathArray.IsCreated)
                {
                    pathArray.Dispose();
                }
                if (backendArray.IsCreated)
                {
                    backendArray.Dispose();
                }
            }
        }

        /// <summary>
        /// Release the native model handle.
        /// Thread-safety: Not thread-safe; must not be in use by any session.
        /// </summary>
        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_model_release(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_disposed = true;
        }

        ~AstralModel()
        {
            if (!m_disposed)
            {
                Debug.LogWarning("[Astral] Model was not disposed properly. Always use 'using' statement or call Dispose().");
                Dispose();
            }
        }
    }

    /// <summary>
    /// Configuration for model loading.
    /// Immutable after model is loaded.
    /// </summary>
    [Serializable]
    public class AstralModelConfig
    {
        /// <summary>
        /// Number of layers to offload to GPU (0 = CPU only).
        ///  Requires CUDA/Metal backend support.
        /// Recommended: 0 (CPU) for v0.1; 32+ for v0.2 GPU backends.
        /// </summary>
        public uint gpuLayers = 0;

        /// <summary>
        /// Context size in tokens.
        ///  Larger context = more memory usage (KV cache grows quadratically).
        /// Mobile: 512-2048; Desktop: 2048-8192; High-end: 8192-32768
        /// </summary>
        public uint contextSize = 2048;

        /// <summary>
        /// Batch size for prompt processing.
        ///  Larger batch = faster prompt ingestion but more memory.
        /// Recommended: 512 (balanced); 128 (mobile); 1024 (desktop)
        /// </summary>
        public uint batchSize = 512;

        /// <summary>
        /// Threads for backend (0 = auto-detect).
        ///  Should match or be less than AstralConfig.threadCount.
        /// Recommended: 0 (auto); 2-4 (mobile); 4-8 (desktop)
        /// </summary>
        public uint threads = 0;

        /// <summary>
        /// Embeddings-only mode (no text generation).
        /// Set to true for embedding models (e.g., all-MiniLM, instructor-xl).
        /// </summary>
        public bool embeddingsOnly = false;

        /// <summary>
        /// Optional backend override (e.g., "cpu", "mock").
        /// Leave null/empty for auto-selection.
        /// </summary>
        public string backendName = null;

        /// <summary>
        /// Default configuration for desktop platforms.
        /// </summary>
        public static AstralModelConfig Default => new AstralModelConfig();

        /// <summary>
        /// Mobile-optimized configuration.
        /// </summary>
        public static AstralModelConfig Mobile => new AstralModelConfig
        {
            gpuLayers = 0,
            contextSize = 1024,
            batchSize = 128,
            threads = 2,
            embeddingsOnly = false
        };

        /// <summary>
        /// High-performance desktop configuration.
        /// </summary>
        public static AstralModelConfig HighPerformance => new AstralModelConfig
        {
            gpuLayers = 0, // CPU-only for v0.1; set to 32+ for v0.2 GPU support
            contextSize = 8192,
            batchSize = 1024,
            threads = 0, // Auto-detect
            embeddingsOnly = false
        };

        /// <summary>
        /// Embeddings-only configuration.
        /// </summary>
        public static AstralModelConfig Embeddings => new AstralModelConfig
        {
            gpuLayers = 0,
            contextSize = 512,  // Embeddings don't need large context
            batchSize = 512,
            threads = 0,
            embeddingsOnly = true
        };
    }
}
