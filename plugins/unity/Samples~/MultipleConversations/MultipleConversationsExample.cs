using System;
using System.Collections;
using System.Collections.Generic;
using Astral.Runtime;
using Unity.Collections;
using UnityEngine;

namespace Astral.Examples
{
    public sealed class MultipleConversationsExample : MonoBehaviour
    {
        private sealed class ActiveConversation : IDisposable
        {
            public AstralConversation conversation;
            public NativeArray<byte> streamBuffer;
            public string label;
            public bool complete;

            public void Dispose()
            {
                conversation?.Dispose();
                conversation = null;
                if (streamBuffer.IsCreated)
                {
                    streamBuffer.Dispose();
                    streamBuffer = default;
                }
            }
        }

        [Header("Model")]
        [SerializeField] private string modelPath = "mock";
        [SerializeField] private string backendName = "mock";
        [SerializeField] private uint maxBatchTokens = 64;

        [Header("NPC prompts")]
        [SerializeField] private string systemPrompt = "Reply as a concise game character.";
        [SerializeField] private string[] prompts =
        {
            "You are the blacksmith. Greet the player.",
            "You are the cartographer. Suggest a destination."
        };
        [SerializeField] private uint maxTokens = 96;
        [SerializeField] private int streamBufferBytes = 2048;

        private readonly List<ActiveConversation> active = new List<ActiveConversation>();
        private AstralModel model;
        private Coroutine activeRun;

        public void Run()
        {
            if (!AstralRuntime.IsInitialized)
            {
                Debug.LogError("[MultipleConversations] Add AstralRuntimeInitializer before running the sample.");
                return;
            }
            if (prompts == null || prompts.Length == 0)
            {
                Debug.LogError("[MultipleConversations] Add at least one NPC prompt.");
                return;
            }

            StopActive();
            try
            {
                model = AstralModel.Load(modelPath, new AstralModelConfig { backendName = backendName });
                model.ConfigureExecutor(new AstralExecutorConfig
                {
                    maxSlots = (uint)prompts.Length,
                    maxBatchTokens = maxBatchTokens,
                    workerHint = AstralExecutorConfig.AutoWorkerHint
                });

                for (int index = 0; index < prompts.Length; ++index)
                {
                    var conversation = AstralConversation.Create(model, new AstralConversationConfig
                    {
                        maxTokens = maxTokens,
                        temperature = 0.0f,
                        topK = 0,
                        topP = 1.0f,
                        streamEnabled = true,
                        seed = (uint)(index + 1)
                    });
                    conversation.SetSystemPrompt(systemPrompt);
                    conversation.Feed(prompts[index], finalize: true);
                    conversation.Decode();
                    active.Add(new ActiveConversation
                    {
                        conversation = conversation,
                        streamBuffer = new NativeArray<byte>(Math.Max(256, streamBufferBytes), Allocator.Persistent),
                        label = $"NPC {index + 1}"
                    });
                }
                activeRun = StartCoroutine(PollCoroutine());
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[MultipleConversations] {ex.Message}");
                DisposeActive();
            }
        }

        public void Cancel()
        {
            foreach (var item in active)
            {
                if (item.conversation == null || !item.conversation.IsValid || item.complete)
                {
                    continue;
                }
                var request = AstralRequest.FromConversation(item.conversation);
                AstralRequest.TryCancel(request, out _);
            }
        }

        private IEnumerator PollCoroutine()
        {
            while (!AllComplete())
            {
                foreach (var item in active)
                {
                    Poll(item);
                }
                yield return null;
            }
            DisposeActive();
            activeRun = null;
        }

        private void Poll(ActiveConversation item)
        {
            if (item.complete)
            {
                return;
            }

            int bytesRead = item.conversation.ReadStream(
                item.streamBuffer,
                AstralConversation.NonBlockingTimeoutMs);
            if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
            {
                return;
            }
            if (bytesRead < 0)
            {
                Debug.LogError($"[MultipleConversations] {item.label}: {AstralRuntime.GetErrorString(bytesRead)}");
                item.complete = true;
                return;
            }
            if (bytesRead > 0)
            {
                var bytes = new NativeSlice<byte>(item.streamBuffer, 0, bytesRead);
                Debug.Log($"[MultipleConversations] {item.label}: {bytes.ToUtf8String()}");
                return;
            }

            var request = AstralRequest.FromConversation(item.conversation);
            if (AstralRequest.TryGetStatus(request, out var status, out int errorCode))
            {
                item.complete = AstralRequest.IsTerminal(status);
                if (item.complete)
                {
                    var stats = item.conversation.GetStats();
                    Debug.Log($"[MultipleConversations] {item.label} finished in slot {stats.slotId}.");
                }
            }
            else
            {
                Debug.LogError($"[MultipleConversations] Status failed: {AstralRuntime.GetErrorString(errorCode)}");
                item.complete = true;
            }
        }

        private bool AllComplete()
        {
            foreach (var item in active)
            {
                if (!item.complete)
                {
                    return false;
                }
            }
            return active.Count > 0;
        }

        private void StopActive()
        {
            Cancel();
            if (activeRun != null)
            {
                StopCoroutine(activeRun);
                activeRun = null;
            }
            DisposeActive();
        }

        private void DisposeActive()
        {
            foreach (var item in active)
            {
                item.Dispose();
            }
            active.Clear();
            model?.Dispose();
            model = null;
        }

        private void OnDestroy()
        {
            StopActive();
        }
    }
}
