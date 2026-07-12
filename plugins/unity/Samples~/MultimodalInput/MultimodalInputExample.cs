using System;
using System.Collections;
using Astral.Runtime;
using Unity.Collections;
using UnityEngine;

namespace Astral.Examples
{
    public sealed class MultimodalInputExample : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private string modelPath = "mock";
        [SerializeField] private string backendName = "mock";
        [SerializeField] private string mediaProjectorPath = "mock-media";

        [Header("Inputs")]
        [SerializeField] private Texture2D image;
        [SerializeField] private AudioClip audioClip;
        [SerializeField] [TextArea(2, 6)] private string prompt = "Describe the important gameplay signal.";

        private AstralModel model;
        private AstralSession session;
        private AstralEmbedder embedder;
        private NativeArray<byte> streamBuffer;
        private Coroutine activeRun;
        private ulong activeEmbeddingTicket;

        public void RunImageGeneration()
        {
            if (!ValidateTexture() || !EnsureModel(AstralNative.ASTRAL_CAP_IMAGE, "image input"))
            {
                return;
            }

            StopActive();
            try
            {
                session = AstralSession.Create(model);
                session.Feed(prompt, finalize: false);
                NativeArray<byte> pixels = image.GetRawTextureData<byte>();
                session.FeedImage(
                    pixels,
                    (uint)image.width,
                    (uint)image.height,
                    AstralNative.AstralImageFormat.RGBA8,
                    finalize: true,
                    rowStride: (uint)image.width * 4u);
                StartSessionStream();
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[MultimodalInput] {ex.Message}");
                DisposeActive();
            }
        }

        public void RunAudioGeneration()
        {
            if (audioClip == null)
            {
                Debug.LogError("[MultimodalInput] Assign an AudioClip.");
                return;
            }
            if (!EnsureModel(AstralNative.ASTRAL_CAP_AUDIO, "audio input"))
            {
                return;
            }

            StopActive();
            try
            {
                float[] managedSamples = new float[audioClip.samples * audioClip.channels];
                if (!audioClip.GetData(managedSamples, 0))
                {
                    throw new AstralException("AudioClip sample data is unavailable. Keep the source clip readable.");
                }

                using (var samples = new NativeArray<float>(managedSamples, Allocator.Temp))
                {
                    session = AstralSession.Create(model);
                    session.Feed(prompt, finalize: false);
                    session.FeedAudio(
                        samples,
                        (uint)audioClip.channels,
                        (uint)audioClip.frequency,
                        finalize: true);
                    StartSessionStream();
                }
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[MultimodalInput] {ex.Message}");
                DisposeActive();
            }
        }

        public void EmbedImage()
        {
            if (!ValidateTexture() ||
                !EnsureModel(AstralNative.ASTRAL_CAP_MM_EMBEDDINGS, "multimodal embeddings"))
            {
                return;
            }

            CancelEmbedding();
            embedder?.Dispose();
            embedder = null;
            try
            {
                embedder = AstralEmbedder.Create(model);
                NativeArray<byte> pixels = image.GetRawTextureData<byte>();
                var imageDesc = new AstralNative.AstralImageDesc
                {
                    format = AstralNative.AstralImageFormat.RGBA8,
                    width = (uint)image.width,
                    height = (uint)image.height,
                    row_stride = (uint)image.width * 4u,
                    pixels = AstralNative.AstralSpanU8.FromNativeArray(pixels)
                };

                activeEmbeddingTicket = embedder.EnqueueMultimodal(prompt, ref imageDesc);
                var request = AstralRequest.FromEmbedding(embedder, activeEmbeddingTicket);
                using (var vector = new NativeArray<float>(
                    checked((int)model.GetEmbeddingDim()),
                    Allocator.Temp,
                    NativeArrayOptions.UninitializedMemory))
                {
                    embedder.Collect(activeEmbeddingTicket, vector);
                    activeEmbeddingTicket = 0;
                    var status = AstralRequest.GetStatus(request);
                    Debug.Log($"[MultimodalInput] Embedded image to {vector.Length} values; {AstralRequest.StateName(status.state)}.");
                }
            }
            catch (AstralException ex)
            {
                activeEmbeddingTicket = 0;
                Debug.LogError($"[MultimodalInput] {ex.Message}");
            }
        }

        public void Cancel()
        {
            if (session != null && session.IsValid)
            {
                AstralRequest.TryCancel(AstralRequest.FromSession(session), out _);
            }
            CancelEmbedding();
        }

        private void StartSessionStream()
        {
            session.Decode();
            streamBuffer = new NativeArray<byte>(4096, Allocator.Persistent);
            activeRun = StartCoroutine(StreamSession());
        }

        private IEnumerator StreamSession()
        {
            while (true)
            {
                int bytesRead = session.ReadStream(streamBuffer, timeoutMs: 0);
                if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
                {
                    yield return null;
                    continue;
                }
                if (bytesRead < 0)
                {
                    Debug.LogError($"[MultimodalInput] Stream failed: {AstralRuntime.GetErrorString(bytesRead)}");
                    break;
                }
                if (bytesRead == 0)
                {
                    break;
                }
                Debug.Log($"[MultimodalInput] {new NativeSlice<byte>(streamBuffer, 0, bytesRead).ToUtf8String()}");
                yield return null;
            }
            DisposeSession();
            activeRun = null;
        }

        private bool EnsureModel(ulong requiredCapability, string capabilityName)
        {
            if (!AstralRuntime.IsInitialized)
            {
                Debug.LogError("[MultimodalInput] Add AstralRuntimeInitializer before running the sample.");
                return false;
            }

            try
            {
                if (model == null || !model.IsValid)
                {
                    if (string.IsNullOrWhiteSpace(mediaProjectorPath))
                    {
                        throw new AstralException("Set a media-projector GGUF path.");
                    }
                    model = AstralModel.Load(modelPath, new AstralModelConfig { backendName = backendName });
                    model.InitMediaFromPath(mediaProjectorPath);
                }

                ulong caps = model.GetCaps();
                if ((caps & requiredCapability) == 0)
                {
                    Debug.LogError($"[MultimodalInput] The selected model/provider does not support {capabilityName}.");
                    return false;
                }
                return true;
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[MultimodalInput] {ex.Message}");
                return false;
            }
        }

        private bool ValidateTexture()
        {
            if (image == null)
            {
                Debug.LogError("[MultimodalInput] Assign a Texture2D.");
                return false;
            }
            if (!image.isReadable || image.format != TextureFormat.RGBA32)
            {
                Debug.LogError("[MultimodalInput] Texture must be readable RGBA32 data.");
                return false;
            }
            return true;
        }

        private void CancelEmbedding()
        {
            if (embedder == null || !embedder.IsValid || activeEmbeddingTicket == 0)
            {
                return;
            }
            AstralRequest.TryCancel(
                AstralRequest.FromEmbedding(embedder, activeEmbeddingTicket),
                out _);
            activeEmbeddingTicket = 0;
        }

        private void StopActive()
        {
            Cancel();
            if (activeRun != null)
            {
                StopCoroutine(activeRun);
                activeRun = null;
            }
            DisposeSession();
        }

        private void DisposeSession()
        {
            session?.Dispose();
            session = null;
            if (streamBuffer.IsCreated)
            {
                streamBuffer.Dispose();
                streamBuffer = default;
            }
        }

        private void DisposeActive()
        {
            DisposeSession();
            CancelEmbedding();
        }

        private void OnDestroy()
        {
            StopActive();
            embedder?.Dispose();
            model?.Dispose();
        }
    }
}
