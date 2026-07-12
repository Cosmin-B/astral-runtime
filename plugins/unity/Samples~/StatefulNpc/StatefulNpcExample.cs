using System.Collections;
using System.IO;
using Astral.Runtime;
using Unity.Collections;
using UnityEngine;

namespace Astral.Examples
{
    public sealed class StatefulNpcExample : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private string modelPath = "mock";
        [SerializeField] private string backendName = "mock";

        [Header("NPC state")]
        [SerializeField] [TextArea(2, 6)] private string systemPrompt =
            "You are the harbor master. Keep answers short and use move_to when the player chooses a destination.";
        [SerializeField] [TextArea(2, 6)] private string summary = "The player has just arrived at the harbor.";
        [SerializeField] [TextArea(2, 6)] private string memoryContext =
            "The lighthouse is north. The market is east. The ferry leaves at sunset.";
        [SerializeField] private string userMessage = "Where should I go first?";
        [SerializeField] private string historyFileName = "astral-harbor-master.history";
        [SerializeField] private uint maxTokens = 128;

        private AstralModel model;
        private AstralToolset toolset;
        private AstralAgent agent;
        private NativeArray<byte> streamBuffer;
        private Coroutine activeChat;

        public void Ask()
        {
            if (!AstralRuntime.IsInitialized)
            {
                Debug.LogError("[StatefulNpc] Add AstralRuntimeInitializer before running the sample.");
                return;
            }

            StopActiveChat();
            try
            {
                EnsureAgent();
                activeChat = StartCoroutine(ChatCoroutine());
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[StatefulNpc] {ex.Message}");
            }
        }

        public void Cancel()
        {
            if (agent == null || !agent.IsValid)
            {
                return;
            }

            var request = AstralRequest.FromAgentChat(agent);
            AstralRequest.TryCancel(request, out _);
        }

        public void SaveHistory()
        {
            if (agent == null || !agent.IsValid)
            {
                return;
            }
            File.WriteAllBytes(HistoryPath, agent.SaveHistory());
            Debug.Log($"[StatefulNpc] Saved {agent.HistoryCount()} messages to {HistoryPath}.");
        }

        public void LoadHistory()
        {
            if (agent == null || !agent.IsValid || !File.Exists(HistoryPath))
            {
                return;
            }
            agent.LoadHistory(File.ReadAllBytes(HistoryPath));
            Debug.Log($"[StatefulNpc] Restored {agent.HistoryCount()} messages from {HistoryPath}.");
        }

        public void ClearHistory()
        {
            if (agent != null && agent.IsValid)
            {
                agent.ClearHistory();
            }
            if (File.Exists(HistoryPath))
            {
                File.Delete(HistoryPath);
            }
        }

        private string HistoryPath => Path.Combine(Application.persistentDataPath, historyFileName);

        private void EnsureAgent()
        {
            if (agent != null && agent.IsValid)
            {
                agent.SetSummary(summary);
                agent.SetMemoryContext(memoryContext);
                return;
            }

            model = AstralModel.Load(modelPath, new AstralModelConfig { backendName = backendName });
            toolset = AstralToolset.Create(new[]
            {
                new AstralToolDefinition
                {
                    toolId = 1,
                    name = "move_to",
                    description = "Move the NPC or player to a named destination.",
                    jsonSchema = "{\"type\":\"object\",\"properties\":{\"destination\":{\"type\":\"string\"}},\"required\":[\"destination\"]}"
                }
            });

            var config = AstralAgentConfig.Default.WithToolset(
                toolset,
                AstralNative.AstralToolChoiceMode.Auto);
            config.maxTokens = maxTokens;
            config.systemPrompt = systemPrompt;
            config.summary = summary;
            config.memoryContext = memoryContext;
            config.overflowPolicy = AstralNative.AstralAgentOverflowPolicy.TruncateOldest;
            agent = AstralAgent.Create(model, config);
            streamBuffer = new NativeArray<byte>(4096, Allocator.Persistent);
            LoadHistory();
        }

        private IEnumerator ChatCoroutine()
        {
            agent.EnqueueChat(userMessage);
            var request = AstralRequest.FromAgentChat(agent);

            while (true)
            {
                int bytesRead = agent.ReadChat(streamBuffer, timeoutMs: 0);
                if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
                {
                    yield return null;
                    continue;
                }
                if (bytesRead == 0)
                {
                    break;
                }
                if (bytesRead < 0)
                {
                    Debug.Log($"[StatefulNpc] Chat ended: {AstralRuntime.GetErrorString(bytesRead)}");
                    break;
                }

                var bytes = new NativeSlice<byte>(streamBuffer, 0, bytesRead);
                Debug.Log($"[StatefulNpc] {bytes.ToUtf8String()}");
                yield return null;
            }

            if (AstralRequest.TryGetStatus(request, out var status, out int statusError))
            {
                Debug.Log($"[StatefulNpc] Chat {AstralRequest.StateName(status.state)}.");
            }
            else
            {
                Debug.LogError($"[StatefulNpc] Status failed: {AstralRuntime.GetErrorString(statusError)}");
            }

            var result = agent.GetChatResult();
            var toolCall = agent.GetChatToolCallResult();
            if (toolCall.Parsed)
            {
                Debug.Log($"[StatefulNpc] Tool {toolCall.name}: {toolCall.argumentsJson}");
            }
            Debug.Log($"[StatefulNpc] {result.generated_tokens} tokens; {result.history_messages} history messages.");

            SaveHistory();
            agent.ReleaseSlot();
            activeChat = null;
        }

        private void StopActiveChat()
        {
            if (activeChat == null)
            {
                return;
            }
            Cancel();
            StopCoroutine(activeChat);
            activeChat = null;
        }

        private void OnDestroy()
        {
            StopActiveChat();
            agent?.Dispose();
            toolset?.Dispose();
            model?.Dispose();
            if (streamBuffer.IsCreated)
            {
                streamBuffer.Dispose();
            }
        }
    }
}
