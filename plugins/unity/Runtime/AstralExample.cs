// AstralExample.cs - Example usage of Astral Unity plugin
//
// USAGE:
// 1. Add AstralRuntimeInitializer component to a GameObject
// 2. Attach this script to a GameObject
// 3. Set modelPath in Inspector
// 4. Call RunInference() from UI button or other script

using System;
using System.Collections;
using UnityEngine;
using Unity.Collections;

namespace Astral.Runtime.Examples
{
    /// <summary>
    /// Example usage of Astral Unity plugin.
    /// Demonstrates model loading, session creation, and streaming inference.
    /// </summary>
    public class AstralExample : MonoBehaviour
    {
        [Header("Model Configuration")]
        [SerializeField]
        [Tooltip("Path to GGUF model file (e.g., /path/to/model.gguf)")]
        private string modelPath = "";

        [SerializeField]
        private bool useMobileConfig = false;

        [Header("Inference Configuration")]
        [SerializeField]
        [Tooltip("Prompt to send to model")]
        private string prompt = "Once upon a time";

        [SerializeField]
        private uint maxTokens = 256;

        [SerializeField]
        [Range(0.0f, 2.0f)]
        private float temperature = 0.7f;

        [SerializeField]
        private bool enableStreaming = true;

        [Header("Output")]
        [SerializeField]
        [TextArea(5, 10)]
        private string output = "";

        private AstralModel m_model;
        private AstralSession m_session;

        private void Start()
        {
            // Ensure runtime is initialized
            if (!AstralRuntime.IsInitialized)
            {
                Debug.LogError("[AstralExample] Runtime not initialized. Add AstralRuntimeInitializer component.");
                return;
            }

            // Load model if path is set
            if (!string.IsNullOrEmpty(modelPath))
            {
                LoadModel();
            }
        }

        private void OnDestroy()
        {
            // Cleanup (RAII pattern)
            m_session?.Dispose();
            m_model?.Dispose();
        }

        /// <summary>
        /// Load GGUF model.
        /// </summary>
        public void LoadModel()
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                Debug.LogError("[AstralExample] Model path is empty");
                return;
            }

            try
            {
                // Dispose existing model if any
                m_model?.Dispose();

                // Load model with appropriate config
                var config = useMobileConfig ? AstralModelConfig.Mobile : AstralModelConfig.Default;
                m_model = AstralModel.Load(modelPath, config);

                Debug.Log($"[AstralExample] Model loaded: {modelPath}");
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[AstralExample] Failed to load model: {ex.Message}");
            }
        }

        /// <summary>
        /// Run inference (blocking).
        /// WARNING: This will block the main thread. Use RunInferenceAsync() instead.
        /// </summary>
        public void RunInference()
        {
            if (m_model == null || !m_model.IsValid)
            {
                Debug.LogError("[AstralExample] Model not loaded. Call LoadModel() first.");
                return;
            }

            try
            {
                // Create session
                var sessionConfig = new AstralSessionConfig
                {
                    maxTokens = maxTokens,
                    temperature = temperature,
                    streamEnabled = false // Blocking mode
                };

                using (var session = AstralSession.Create(m_model, sessionConfig))
                {
                    // Feed prompt
                    session.Feed(prompt, finalize: true);

                    // Start decode
                    session.Decode();

                    // Wait for completion (blocking)
                    // In streaming mode, tokens would be emitted as they are generated
                    // In non-streaming mode, all tokens are buffered

                    Debug.Log("[AstralExample] Inference started (blocking mode)");
                }
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[AstralExample] Inference failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Run inference with streaming (coroutine).
        /// Recommended: Use this method for responsive UI.
        /// </summary>
        public void RunInferenceAsync()
        {
            if (m_model == null || !m_model.IsValid)
            {
                Debug.LogError("[AstralExample] Model not loaded. Call LoadModel() first.");
                return;
            }

            StartCoroutine(RunInferenceCoroutine());
        }

        private IEnumerator RunInferenceCoroutine()
        {
            output = ""; // Clear previous output

            AstralSession session = null;

            try
            {
                // Create session
                var sessionConfig = new AstralSessionConfig
                {
                    maxTokens = maxTokens,
                    temperature = temperature,
                    streamEnabled = enableStreaming
                };

                session = AstralSession.Create(m_model, sessionConfig);

                // Feed prompt
                session.Feed(prompt, finalize: true);

                // Start decode (non-blocking)
                session.Decode();

                Debug.Log("[AstralExample] Inference started (streaming mode)");

                // Stream tokens
                yield return StartCoroutine(session.StreamCoroutine(
                    onToken: (token) =>
                    {
                        output += token;
                        Debug.Log($"[AstralExample] Token: {token}");
                    },
                    onComplete: () =>
                    {
                        Debug.Log("[AstralExample] Inference complete");
                        PrintStats(session);
                    }
                ));
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[AstralExample] Inference failed: {ex.Message}");
            }
            finally
            {
                session?.Dispose();
            }
        }

        /// <summary>
        /// Run inference with zero-allocation streaming.
        ///  No GC allocations during token streaming.
        /// </summary>
        public void RunInferenceZeroAlloc()
        {
            if (m_model == null || !m_model.IsValid)
            {
                Debug.LogError("[AstralExample] Model not loaded. Call LoadModel() first.");
                return;
            }

            StartCoroutine(RunInferenceZeroAllocCoroutine());
        }

        private IEnumerator RunInferenceZeroAllocCoroutine()
        {
            AstralSession session = null;
            NativeArray<byte> buffer = new NativeArray<byte>(4096, Allocator.Persistent);

            try
            {
                // Create session
                var sessionConfig = new AstralSessionConfig
                {
                    maxTokens = maxTokens,
                    temperature = temperature,
                    streamEnabled = true
                };

                session = AstralSession.Create(m_model, sessionConfig);

                // Feed prompt (zero-copy)
                var promptBytes = System.Text.Encoding.UTF8.GetBytes(prompt);
                var promptArray = new NativeArray<byte>(promptBytes, Allocator.Temp);
                session.Feed(promptArray, finalize: true);
                promptArray.Dispose();

                // Start decode
                session.Decode();

                Debug.Log("[AstralExample] Inference started (zero-alloc mode)");

                // Stream tokens (zero GC allocation)
                while (true)
                {
                    int bytesRead = session.ReadStream(buffer, timeoutMs: 0);

                    if (bytesRead > 0)
                    {
                        // Process token bytes (no string allocation)
                        // In production: write to GPU texture, append to TextMesh, etc.
                        unsafe
                        {
                            fixed (byte* ptr = &buffer[0])
                            {
                                // Example: just log (this allocates string for debug)
                                string token = System.Text.Encoding.UTF8.GetString(ptr, bytesRead);
                                Debug.Log($"[AstralExample] Token: {token}");
                            }
                        }
                    }
                    else if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
                    {
                        // No data available; wait one frame
                        yield return null;
                    }
                    else if (bytesRead < 0)
                    {
                        Debug.LogError($"[AstralExample] Stream read failed: {AstralRuntime.GetErrorString(bytesRead)}");
                        break;
                    }
                    else
                    {
                        // End of stream
                        Debug.Log("[AstralExample] Inference complete");
                        PrintStats(session);
                        break;
                    }
                }
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[AstralExample] Inference failed: {ex.Message}");
            }
            finally
            {
                session?.Dispose();
                buffer.Dispose();
            }
        }

        private void PrintStats(AstralSession session)
        {
            try
            {
                var stats = session.GetStats();
                Debug.Log($"[AstralExample] {stats}");
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[AstralExample] Failed to get stats: {ex.Message}");
            }
        }

        // Unity Editor helpers
#if UNITY_EDITOR
        [ContextMenu("Load Model")]
        private void EditorLoadModel()
        {
            LoadModel();
        }

        [ContextMenu("Run Inference")]
        private void EditorRunInference()
        {
            RunInferenceAsync();
        }
#endif
    }
}
