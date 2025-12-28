// AstralAgent.cs - Native agent handle with explicit ownership.
//
// The native runtime owns prompt assembly, summary bytes, chat history, and
// streaming state. This wrapper only converts Unity strings/arrays at call
// boundaries and releases the native handle.

using System;
using System.Runtime.InteropServices;
using System.Text;
using Unity.Collections;

namespace Astral.Runtime
{
    /// <summary>
    /// Native agent wrapper. Dispose releases the native agent handle.
    /// </summary>
    public sealed class AstralAgent : IDisposable
    {
        private const uint DefaultStreamReadBytes = 4096;
        internal const uint DefaultMaxTokens = 128;
        internal const uint DefaultMaxMessages = 64;
        internal const uint DefaultMaxPromptBytes = 64u * 1024u;
        internal const float DefaultTemperature = 0.0f;
        internal const uint DefaultTopK = 0;
        internal const float DefaultTopP = 1.0f;
        internal const uint DefaultSeed = 0;

        private AstralNative.AstralHandle m_handle;
        private AstralModel m_model;
        private bool m_disposed;

        private enum AgentTextSlot
        {
            SystemPrompt,
            Summary,
            MemoryContext
        }

        public bool IsValid => !m_disposed && m_handle.IsValid;
        internal AstralNative.AstralHandle Handle => m_handle;

        public static AstralAgent Create(AstralModel model, AstralAgentConfig config = null)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }
            if (!model.IsValid)
            {
                throw new AstralException("Model is not valid (disposed or not loaded).");
            }

            config = config ?? AstralAgentConfig.Default;
            var desc = new AstralNative.AstralAgentDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralAgentDesc>(),
                model = model.Handle,
                prompt_cache = config.promptCache,
                memory_index = config.memoryIndex,
                toolset = config.toolset,
                max_tokens = config.maxTokens,
                temperature = config.temperature,
                top_k = config.topK,
                top_p = config.topP,
                stream_enabled = (byte)(config.streamEnabled ? 1 : 0),
                seed = config.seed,
                tool_choice_mode = (uint)config.toolChoiceMode,
                max_messages = config.maxMessages,
                max_prompt_bytes = config.maxPromptBytes,
                overflow_policy = config.overflowPolicy
            };

            int err = AstralNative.astral_agent_create(ref desc, out var handle);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_agent_create failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return new AstralAgent
            {
                m_handle = handle,
                m_model = model,
                m_disposed = false
            };
        }

        public void SetSystemPrompt(string systemPrompt)
        {
            SetText(AgentTextSlot.SystemPrompt, systemPrompt, nameof(systemPrompt));
        }

        public string GetSystemPrompt()
        {
            return GetText(AgentTextSlot.SystemPrompt);
        }

        public void SetSummary(string summary)
        {
            SetText(AgentTextSlot.Summary, summary, nameof(summary));
        }

        public string GetSummary()
        {
            return GetText(AgentTextSlot.Summary);
        }

        public void SetMemoryContext(string memoryContext)
        {
            SetText(AgentTextSlot.MemoryContext, memoryContext, nameof(memoryContext));
        }

        public string GetMemoryContext()
        {
            return GetText(AgentTextSlot.MemoryContext);
        }

        public void AddMessage(AstralNative.AstralAgentRole role, string text)
        {
            ThrowIfInvalid();
            NativeArray<byte> textBytes;
            var span = AstralNative.AstralSpanU8.FromString(text, out textBytes);
            try
            {
                var message = new AstralNative.AstralAgentMessage
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralAgentMessage>(),
                    role = role,
                    content = span
                };
                int err = AstralNative.astral_agent_message_add(m_handle, ref message);
                ThrowIfError(err, "astral_agent_message_add");
            }
            finally
            {
                if (textBytes.IsCreated)
                {
                    textBytes.Dispose();
                }
            }
        }

        public void ClearHistory()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_agent_history_clear(m_handle);
            ThrowIfError(err, "astral_agent_history_clear");
        }

        public uint HistoryCount()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_agent_history_count(m_handle, out uint count);
            ThrowIfError(err, "astral_agent_history_count");
            return count;
        }

        public byte[] SaveHistory()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_agent_history_save_size(m_handle, out uint byteCount);
            ThrowIfError(err, "astral_agent_history_save_size");
            if (byteCount == 0)
            {
                return Array.Empty<byte>();
            }

            using (var bytes = new NativeArray<byte>((int)byteCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory))
            {
                var span = AstralNative.AstralMutSpanU8.FromNativeArray(bytes);
                err = AstralNative.astral_agent_history_save(m_handle, span, out uint written);
                ThrowIfError(err, "astral_agent_history_save");

                var managed = new byte[written];
                NativeArray<byte>.Copy(bytes, managed, (int)written);
                return managed;
            }
        }

        public void LoadHistory(byte[] bytes)
        {
            ThrowIfInvalid();
            if (bytes == null || bytes.Length == 0)
            {
                throw new ArgumentException("bytes must not be empty", nameof(bytes));
            }

            using (var native = new NativeArray<byte>(bytes, Allocator.Temp))
            {
                var span = AstralNative.AstralSpanU8.FromNativeArray(native);
                int err = AstralNative.astral_agent_history_load(m_handle, span);
                ThrowIfError(err, "astral_agent_history_load");
            }
        }

        public void EnqueueChat(string userMessage, bool warmupOnly = false)
        {
            ThrowIfInvalid();
            NativeArray<byte> userBytes;
            var span = AstralNative.AstralSpanU8.FromString(userMessage, out userBytes);
            try
            {
                var desc = new AstralNative.AstralAgentChatDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralAgentChatDesc>(),
                    flags = warmupOnly ? AstralNative.AstralAgentChatFlags.Warmup : AstralNative.AstralAgentChatFlags.None,
                    user_message = span
                };

                int err = AstralNative.astral_agent_chat_enqueue(m_handle, ref desc);
                ThrowIfError(err, "astral_agent_chat_enqueue");
            }
            finally
            {
                if (userBytes.IsCreated)
                {
                    userBytes.Dispose();
                }
            }
        }

        public void CancelChat()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_agent_chat_cancel(m_handle);
            ThrowIfError(err, "astral_agent_chat_cancel");
        }

        public int ReadChat(NativeArray<byte> buffer, uint timeoutMs = 0)
        {
            ThrowIfInvalid();
            if (!buffer.IsCreated)
            {
                throw new ArgumentException("buffer must be created", nameof(buffer));
            }

            int result = AstralNative.astral_agent_chat_stream_read(m_handle, AstralNative.AstralMutSpanU8.FromNativeArray(buffer), timeoutMs);
            if (result < 0)
            {
                ThrowIfError(result, "astral_agent_chat_stream_read");
            }
            return result;
        }

        public string ReadChatString(uint timeoutMs = 0, uint maxBytes = DefaultStreamReadBytes)
        {
            ThrowIfInvalid();
            using (var buffer = new NativeArray<byte>((int)maxBytes, Allocator.Temp, NativeArrayOptions.UninitializedMemory))
            {
                int bytesRead = ReadChat(buffer, timeoutMs);
                if (bytesRead <= 0)
                {
                    return string.Empty;
                }

                byte[] managed = new byte[bytesRead];
                NativeArray<byte>.Copy(buffer, managed, bytesRead);
                return Encoding.UTF8.GetString(managed);
            }
        }

        public AstralNative.AstralAgentChatResult GetChatResult()
        {
            ThrowIfInvalid();
            var result = new AstralNative.AstralAgentChatResult
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralAgentChatResult>()
            };
            int err = AstralNative.astral_agent_chat_result(m_handle, ref result);
            ThrowIfError(err, "astral_agent_chat_result");
            return result;
        }

        public AstralToolCall ParseToolCall(string generatedText)
        {
            NativeArray<byte> textArray;
            var textSpan = AstralNative.AstralSpanU8.FromString(generatedText, out textArray);
            try
            {
                return ParseToolCall(textSpan);
            }
            finally
            {
                if (textArray.IsCreated)
                {
                    textArray.Dispose();
                }
            }
        }

        public AstralToolCall ParseToolCall(NativeArray<byte> generatedText)
        {
            if (!generatedText.IsCreated)
            {
                throw new ArgumentException("generatedText must be created", nameof(generatedText));
            }

            return ParseToolCall(AstralNative.AstralSpanU8.FromNativeArray(generatedText));
        }

        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_agent_destroy(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_model = null;
            m_disposed = true;
        }

        private void SetText(AgentTextSlot slot, string text, string paramName)
        {
            ThrowIfInvalid();
            NativeArray<byte> bytes;
            var span = AstralNative.AstralSpanU8.FromString(text, out bytes);
            try
            {
                int err = SetTextNative(slot, span);
                ThrowIfError(err, paramName);
            }
            finally
            {
                if (bytes.IsCreated)
                {
                    bytes.Dispose();
                }
            }
        }

        private string GetText(AgentTextSlot slot)
        {
            ThrowIfInvalid();
            uint byteCount = 0;
            int err = GetTextSizeNative(slot, out byteCount);
            ThrowIfError(err, "agent text size");
            if (byteCount == 0)
            {
                return string.Empty;
            }

            using (var bytes = new NativeArray<byte>((int)byteCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory))
            {
                uint written = 0;
                err = GetTextNative(slot, AstralNative.AstralMutSpanU8.FromNativeArray(bytes), out written);
                ThrowIfError(err, "agent text get");
                byte[] managed = new byte[written];
                NativeArray<byte>.Copy(bytes, managed, (int)written);
                return Encoding.UTF8.GetString(managed);
            }
        }

        private AstralToolCall ParseToolCall(AstralNative.AstralSpanU8 generatedText)
        {
            ThrowIfInvalid();
            var result = new AstralNative.AstralToolCallResult
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralToolCallResult>()
            };

            int err = AstralNative.astral_agent_parse_tool_call(m_handle, generatedText, ref result);
            if (err == AstralNative.ASTRAL_E_NOT_FOUND)
            {
                return new AstralToolCall
                {
                    found = false,
                    parseStatus = err
                };
            }
            ThrowIfError(err, "astral_agent_parse_tool_call");

            return new AstralToolCall
            {
                found = true,
                toolId = result.tool_id,
                parseStatus = result.parse_status,
                name = result.name.ToUtf8String(),
                argumentsJson = result.arguments_json.ToUtf8String()
            };
        }

        private int SetTextNative(AgentTextSlot slot, AstralNative.AstralSpanU8 span)
        {
            switch (slot)
            {
                case AgentTextSlot.SystemPrompt:
                    return AstralNative.astral_agent_set_system_prompt(m_handle, span);
                case AgentTextSlot.MemoryContext:
                    return AstralNative.astral_agent_set_memory_context(m_handle, span);
                case AgentTextSlot.Summary:
                default:
                    return AstralNative.astral_agent_set_summary(m_handle, span);
            }
        }

        private int GetTextSizeNative(AgentTextSlot slot, out uint byteCount)
        {
            switch (slot)
            {
                case AgentTextSlot.SystemPrompt:
                    return AstralNative.astral_agent_get_system_prompt_size(m_handle, out byteCount);
                case AgentTextSlot.MemoryContext:
                    return AstralNative.astral_agent_get_memory_context_size(m_handle, out byteCount);
                case AgentTextSlot.Summary:
                default:
                    return AstralNative.astral_agent_get_summary_size(m_handle, out byteCount);
            }
        }

        private int GetTextNative(AgentTextSlot slot, AstralNative.AstralMutSpanU8 outText, out uint written)
        {
            switch (slot)
            {
                case AgentTextSlot.SystemPrompt:
                    return AstralNative.astral_agent_get_system_prompt(m_handle, outText, out written);
                case AgentTextSlot.MemoryContext:
                    return AstralNative.astral_agent_get_memory_context(m_handle, outText, out written);
                case AgentTextSlot.Summary:
                default:
                    return AstralNative.astral_agent_get_summary(m_handle, outText, out written);
            }
        }

        private void ThrowIfInvalid()
        {
            if (!IsValid)
            {
                throw new AstralException("Agent is not valid (disposed or not created).");
            }
        }

        private static void ThrowIfError(int err, string call)
        {
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"{call} failed: {AstralRuntime.GetErrorString(err)}", err);
            }
        }
    }

    /// <summary>
    /// Native agent creation settings.
    /// </summary>
    [Serializable]
    public class AstralAgentConfig
    {
        public uint maxTokens = AstralAgent.DefaultMaxTokens;
        public float temperature = AstralAgent.DefaultTemperature;
        public uint topK = AstralAgent.DefaultTopK;
        public float topP = AstralAgent.DefaultTopP;
        public bool streamEnabled = true;
        public uint seed = AstralAgent.DefaultSeed;
        public uint maxMessages = AstralAgent.DefaultMaxMessages;
        public uint maxPromptBytes = AstralAgent.DefaultMaxPromptBytes;
        public AstralNative.AstralToolChoiceMode toolChoiceMode = AstralNative.AstralToolChoiceMode.Auto;
        public AstralNative.AstralAgentOverflowPolicy overflowPolicy = AstralNative.AstralAgentOverflowPolicy.Reject;
        public AstralNative.AstralHandle promptCache = AstralNative.AstralHandle.Invalid;
        public AstralNative.AstralHandle memoryIndex = AstralNative.AstralHandle.Invalid;
        public AstralNative.AstralHandle toolset = AstralNative.AstralHandle.Invalid;

        public static AstralAgentConfig Default => new AstralAgentConfig();
    }
}
