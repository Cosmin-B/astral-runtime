using System;
using System.Collections;
using Unity.Collections;
using UnityEngine;
using Astral.Runtime;

namespace Astral.Examples
{
    /// <summary>
    /// Minimal Unity chat flow using the maintained model/session wrappers.
    /// Native token bytes stay in a caller-owned buffer; this sample converts to
    /// a managed string only at the Debug.Log boundary.
    /// </summary>
    public sealed class BasicChatExample : MonoBehaviour
    {
        [Header("Model")]
        [Tooltip("Absolute path to a GGUF model visible to the Unity process.")]
        [SerializeField] private string modelPath = "";

        [Tooltip("Optional backend override such as cpu, cuda, or mock.")]
        [SerializeField] private string backendName = "cpu";

        [Tooltip("Context size passed to AstralModelConfig.")]
        [SerializeField] private uint contextSize = 2048;

        [Tooltip("GPU layers requested by the backend; 0 keeps CPU-only behavior.")]
        [SerializeField] private uint gpuLayers = 0;

        [Header("Prompt")]
        [SerializeField] [TextArea(3, 10)] private string prompt = "Hello from Unity.";
        [SerializeField] private uint maxTokens = 128;
        [SerializeField] [Range(0.0f, 2.0f)] private float temperature = 0.7f;
        [SerializeField] private bool runOnStart = true;

        [Header("Streaming")]
        [SerializeField] private int streamBufferBytes = 4096;
        [SerializeField] private uint streamTimeoutMs = 0;

        private AstralModel model;
        private Coroutine activeRun;

        private void Start()
        {
            if (runOnStart)
            {
                RunInference();
            }
        }

        private void OnDestroy()
        {
            if (activeRun != null)
            {
                StopCoroutine(activeRun);
                activeRun = null;
            }
            model?.Dispose();
        }

        public void RunInference()
        {
            if (!AstralRuntime.IsInitialized)
            {
                Debug.LogError("[BasicChat] Add AstralRuntimeInitializer before running the sample.");
                return;
            }
            if (string.IsNullOrEmpty(modelPath))
            {
                Debug.LogError("[BasicChat] Set modelPath to a GGUF path or a mock model name.");
                return;
            }

            if (activeRun != null)
            {
                StopCoroutine(activeRun);
            }
            activeRun = StartCoroutine(RunInferenceCoroutine());
        }

        private IEnumerator RunInferenceCoroutine()
        {
            AstralSession session = null;
            NativeArray<byte> streamBuffer = default;

            try
            {
                EnsureModelLoaded();

                var sessionConfig = new AstralSessionConfig
                {
                    maxTokens = maxTokens,
                    temperature = temperature,
                    streamEnabled = true,
                    seed = 1
                };
                session = AstralSession.Create(model, sessionConfig);
                session.Feed(prompt, finalize: true);
                session.Decode();

                int capacity = Math.Max(256, streamBufferBytes);
                streamBuffer = new NativeArray<byte>(capacity, Allocator.Persistent);

                while (true)
                {
                    int bytesRead = session.ReadStream(streamBuffer, streamTimeoutMs);
                    if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
                    {
                        yield return null;
                        continue;
                    }
                    if (bytesRead < 0)
                    {
                        Debug.LogError($"[BasicChat] Stream read failed: {AstralRuntime.GetErrorString(bytesRead)}");
                        yield break;
                    }
                    if (bytesRead == 0)
                    {
                        break;
                    }

                    var chunk = new NativeSlice<byte>(streamBuffer, 0, bytesRead);
                    Debug.Log($"[BasicChat] {chunk.ToUtf8String()}");
                }

                Debug.Log($"[BasicChat] Complete: {session.GetStats()}");
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[BasicChat] Astral call failed: {ex.Message}");
            }
            finally
            {
                if (streamBuffer.IsCreated)
                {
                    streamBuffer.Dispose();
                }
                session?.Dispose();
                activeRun = null;
            }
        }

        private void EnsureModelLoaded()
        {
            if (model != null && model.IsValid)
            {
                return;
            }

            var config = new AstralModelConfig
            {
                backendName = backendName,
                contextSize = contextSize,
                gpuLayers = gpuLayers
            };
            model = AstralModel.Load(modelPath, config);
        }
    }
}
