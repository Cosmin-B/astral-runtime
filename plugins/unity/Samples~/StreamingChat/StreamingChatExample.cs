using System;
using System.Collections;
using Astral.Runtime;
using Unity.Collections;
using UnityEngine;

namespace Astral.Examples
{
    public sealed class StreamingChatExample : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private string modelPath = "mock";
        [SerializeField] private string backendName = "mock";
        [SerializeField] private uint contextSize = 2048;
        [SerializeField] private uint gpuLayers;

        [Header("Request")]
        [SerializeField] [TextArea(3, 10)] private string prompt = "Introduce yourself to the player.";
        [SerializeField] private uint maxTokens = 128;
        [SerializeField] [Range(0.0f, 2.0f)] private float temperature = 0.7f;
        [SerializeField] private int streamBufferBytes = 4096;
        [SerializeField] private bool runOnStart;

        private AstralModel model;
        private AstralSession session;
        private NativeArray<byte> streamBuffer;
        private Coroutine activeRun;

        private void Start()
        {
            if (runOnStart)
            {
                Run();
            }
        }

        public void Run()
        {
            if (!AstralRuntime.IsInitialized)
            {
                Debug.LogError("[StreamingChat] Add AstralRuntimeInitializer before running the sample.");
                return;
            }

            StopActiveRequest();
            try
            {
                EnsureModel();
                session = AstralSession.Create(model, new AstralSessionConfig
                {
                    maxTokens = maxTokens,
                    temperature = temperature,
                    streamEnabled = true,
                    seed = 1
                });
                streamBuffer = new NativeArray<byte>(Math.Max(256, streamBufferBytes), Allocator.Persistent);
                session.Feed(prompt, finalize: true);
                session.Decode();
                activeRun = StartCoroutine(StreamCoroutine(AstralRequest.FromSession(session)));
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[StreamingChat] {ex.Message}");
                DisposeRequest();
            }
        }

        public void Cancel()
        {
            if (session == null || !session.IsValid)
            {
                return;
            }

            var request = AstralRequest.FromSession(session);
            if (!AstralRequest.TryCancel(request, out int errorCode) &&
                errorCode != AstralNative.ASTRAL_E_STATE)
            {
                Debug.LogError($"[StreamingChat] Cancel failed: {AstralRuntime.GetErrorString(errorCode)}");
            }
        }

        private IEnumerator StreamCoroutine(AstralNative.AstralRequestRef request)
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
                    Debug.LogError($"[StreamingChat] Read failed: {AstralRuntime.GetErrorString(bytesRead)}");
                    break;
                }
                if (bytesRead == 0)
                {
                    break;
                }

                var bytes = new NativeSlice<byte>(streamBuffer, 0, bytesRead);
                Debug.Log($"[StreamingChat] {bytes.ToUtf8String()}");
            }

            if (AstralRequest.TryGetStatus(request, out var status, out int statusError))
            {
                Debug.Log($"[StreamingChat] {AstralRequest.StateName(status.state)}; {session.GetStats()}");
            }
            else
            {
                Debug.LogError($"[StreamingChat] Status failed: {AstralRuntime.GetErrorString(statusError)}");
            }
            DisposeRequest();
            activeRun = null;
        }

        private void EnsureModel()
        {
            if (model != null && model.IsValid)
            {
                return;
            }

            model = AstralModel.Load(modelPath, new AstralModelConfig
            {
                backendName = backendName,
                contextSize = contextSize,
                gpuLayers = gpuLayers
            });
        }

        private void StopActiveRequest()
        {
            if (activeRun != null)
            {
                Cancel();
                StopCoroutine(activeRun);
                activeRun = null;
            }
            DisposeRequest();
        }

        private void DisposeRequest()
        {
            session?.Dispose();
            session = null;
            if (streamBuffer.IsCreated)
            {
                streamBuffer.Dispose();
                streamBuffer = default;
            }
        }

        private void OnDestroy()
        {
            StopActiveRequest();
            model?.Dispose();
            model = null;
        }
    }
}
