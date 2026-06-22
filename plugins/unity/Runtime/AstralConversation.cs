// AstralConversation.cs - Unity owner for continuous batching conversation slots.

using System;
using System.Runtime.InteropServices;
using System.Text;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    [Serializable]
    public struct AstralConversationConfig
    {
        public const uint DefaultMaxSlots = 4;
        public const uint DefaultMaxBatchTokens = 64;
        public const uint DefaultMaxTokens = AstralSessionConfig.DefaultMaxTokens;
        public const float DefaultTemperature = AstralSessionConfig.DefaultTemperature;
        public const uint DefaultTopK = AstralSessionConfig.DefaultTopK;
        public const float DefaultTopP = AstralSessionConfig.DefaultTopP;
        public const bool DefaultStreamEnabled = AstralSessionConfig.DefaultStreamEnabled;
        public const uint AutoSeed = AstralSessionConfig.AutoSeed;

        public uint maxTokens;
        public float temperature;
        public uint topK;
        public float topP;
        public bool streamEnabled;
        public uint seed;

        public static AstralConversationConfig Default => new AstralConversationConfig
        {
            maxTokens = DefaultMaxTokens,
            temperature = DefaultTemperature,
            topK = DefaultTopK,
            topP = DefaultTopP,
            streamEnabled = DefaultStreamEnabled,
            seed = AutoSeed
        };
    }

    public struct AstralConversationStats
    {
        public uint slotId;
        public uint promptTokens;
        public uint kvTokens;
        public ulong generatedTokens;
        public double firstTokenTimeMs;
        public double tokensPerSecond;
    }

    /// <summary>
    /// Continuous batching conversation bound to one model executor slot.
    /// </summary>
    public sealed class AstralConversation : IDisposable
    {
        public const int DefaultStreamBufferBytes = 4096;
        public const uint DefaultReadTimeoutMs = 100;
        public const uint NonBlockingTimeoutMs = 0;

        private const byte NativeFalse = 0;
        private const byte NativeTrue = 1;

        private AstralNative.AstralHandle m_handle;
        private AstralConversationConfig m_config;
        private AstralModel m_model;
        private NativeArray<byte> m_streamBuffer;
        private bool m_disposed;

        internal AstralNative.AstralHandle Handle => m_handle;
        public bool IsValid => !m_disposed && m_handle.IsValid;

        public static AstralConversation Create(AstralModel model, AstralConversationConfig? config = null)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }
            if (!model.IsValid)
            {
                throw new ArgumentException("Model is not valid", nameof(model));
            }

            AstralConversationConfig resolved = config ?? AstralConversationConfig.Default;
            var desc = BuildDesc(model, resolved);

            int err = AstralNative.astral_conv_create(ref desc, out var handle);
            ThrowIfError(err, "astral_conv_create");

            return new AstralConversation
            {
                m_handle = handle,
                m_config = resolved,
                m_model = model,
                m_streamBuffer = new NativeArray<byte>(
                    DefaultStreamBufferBytes,
                    Allocator.Persistent,
                    NativeArrayOptions.UninitializedMemory)
            };
        }

        public void Feed(string promptChunk, bool finalize = true)
        {
            ThrowIfDisposed();

            if (string.IsNullOrEmpty(promptChunk))
            {
                if (finalize)
                {
                    int emptyErr = AstralNative.astral_conv_feed(
                        m_handle,
                        new AstralNative.AstralSpanU8(),
                        ToNativeBool(finalize));
                    ThrowIfError(emptyErr, "astral_conv_feed");
                }
                return;
            }

            NativeArray<byte> tempArray;
            var span = AstralNative.AstralSpanU8.FromString(promptChunk, out tempArray);
            try
            {
                int err = AstralNative.astral_conv_feed(m_handle, span, ToNativeBool(finalize));
                ThrowIfError(err, "astral_conv_feed");
            }
            finally
            {
                DisposeIfCreated(ref tempArray);
            }
        }

        public void SetSystemPrompt(string systemPrompt)
        {
            ThrowIfDisposed();

            NativeArray<byte> tempArray;
            var span = AstralNative.AstralSpanU8.FromString(systemPrompt, out tempArray);
            try
            {
                int err = AstralNative.astral_conv_set_system_prompt(m_handle, span);
                ThrowIfError(err, "astral_conv_set_system_prompt");
            }
            finally
            {
                DisposeIfCreated(ref tempArray);
            }
        }

        public void FeedImage(ref AstralNative.AstralImageDesc image, bool finalize = true)
        {
            ThrowIfDisposed();

            image.size = (uint)Marshal.SizeOf<AstralNative.AstralImageDesc>();
            int err = AstralNative.astral_conv_feed_image(m_handle, ref image, ToNativeBool(finalize));
            ThrowIfError(err, "astral_conv_feed_image");
        }

        public void FeedAudio(ref AstralNative.AstralAudioDesc audio, bool finalize = true)
        {
            ThrowIfDisposed();

            audio.size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>();
            int err = AstralNative.astral_conv_feed_audio(m_handle, ref audio, ToNativeBool(finalize));
            ThrowIfError(err, "astral_conv_feed_audio");
        }

        public void Decode()
        {
            ThrowIfDisposed();
            int err = AstralNative.astral_conv_decode(m_handle);
            ThrowIfError(err, "astral_conv_decode");
        }

        public void Cancel()
        {
            ThrowIfDisposed();
            int err = AstralNative.astral_conv_cancel(m_handle);
            ThrowIfError(err, "astral_conv_cancel");
        }

        public AstralSessionState GetState()
        {
            ThrowIfDisposed();

            int err = AstralNative.astral_conv_state(m_handle, out uint state);
            ThrowIfError(err, "astral_conv_state");
            return (AstralSessionState)state;
        }

        public int WaitResult(uint timeoutMs)
        {
            ThrowIfDisposed();
            return AstralNative.astral_conv_wait(m_handle, timeoutMs);
        }

        public bool Wait(uint timeoutMs)
        {
            int err = WaitResult(timeoutMs);
            if (err == AstralNative.ASTRAL_OK || err == AstralNative.ASTRAL_E_CANCELED)
            {
                return true;
            }
            if (err == AstralNative.ASTRAL_E_TIMEOUT)
            {
                return false;
            }
            ThrowIfError(err, "astral_conv_wait");
            return false;
        }

        public void Reset(AstralConversationConfig? config = null)
        {
            ThrowIfDisposed();

            AstralConversationConfig resolved = config ?? m_config;
            var desc = BuildDesc(m_model, resolved);
            int err = AstralNative.astral_conv_reset(m_handle, ref desc);
            ThrowIfError(err, "astral_conv_reset");
            m_config = resolved;
        }

        public void SetSampler(ref AstralNative.AstralSamplerDesc sampler)
        {
            ThrowIfDisposed();

            sampler.size = (uint)Marshal.SizeOf<AstralNative.AstralSamplerDesc>();
            int err = AstralNative.astral_conv_set_sampler(m_handle, ref sampler);
            ThrowIfError(err, "astral_conv_set_sampler");
        }

        public unsafe void SetPenaltyPromptTokens(NativeArray<int> tokens)
        {
            ThrowIfDisposed();
            if (!tokens.IsCreated)
            {
                throw new ArgumentException("tokens must be created", nameof(tokens));
            }

            int err = AstralNative.astral_conv_penalty_prompt_set_tokens(
                m_handle,
                (int*)tokens.GetUnsafeReadOnlyPtr(),
                (uint)tokens.Length);
            ThrowIfError(err, "astral_conv_penalty_prompt_set_tokens");
        }

        public void ClearStops()
        {
            ThrowIfDisposed();
            int err = AstralNative.astral_conv_stop_clear(m_handle);
            ThrowIfError(err, "astral_conv_stop_clear");
        }

        public void AddStop(string stopSequence)
        {
            ThrowIfDisposed();

            NativeArray<byte> tempArray;
            var span = AstralNative.AstralSpanU8.FromString(stopSequence, out tempArray);
            try
            {
                int err = AstralNative.astral_conv_stop_add_utf8(m_handle, span);
                ThrowIfError(err, "astral_conv_stop_add_utf8");
            }
            finally
            {
                DisposeIfCreated(ref tempArray);
            }
        }

        public void SetLogprobs(uint nProbs)
        {
            ThrowIfDisposed();
            int err = AstralNative.astral_conv_set_logprobs(m_handle, nProbs);
            ThrowIfError(err, "astral_conv_set_logprobs");
        }

        public void SetGrammarGbnf(string gbnf, string root = null)
        {
            ThrowIfDisposed();

            NativeArray<byte> gbnfArray;
            NativeArray<byte> rootArray;
            var gbnfSpan = AstralNative.AstralSpanU8.FromString(gbnf, out gbnfArray);
            var rootSpan = AstralNative.AstralSpanU8.FromString(root, out rootArray);
            try
            {
                int err = AstralNative.astral_conv_grammar_set_gbnf(m_handle, gbnfSpan, rootSpan);
                ThrowIfError(err, "astral_conv_grammar_set_gbnf");
            }
            finally
            {
                DisposeIfCreated(ref gbnfArray);
                DisposeIfCreated(ref rootArray);
            }
        }

        public void SetGrammarJsonSchema(string jsonSchema)
        {
            ThrowIfDisposed();

            NativeArray<byte> schemaArray;
            var schemaSpan = AstralNative.AstralSpanU8.FromString(jsonSchema, out schemaArray);
            try
            {
                int err = AstralNative.astral_conv_grammar_set_json_schema(m_handle, schemaSpan);
                ThrowIfError(err, "astral_conv_grammar_set_json_schema");
            }
            finally
            {
                DisposeIfCreated(ref schemaArray);
            }
        }

        public void ClearGrammar()
        {
            ThrowIfDisposed();
            int err = AstralNative.astral_conv_grammar_clear(m_handle);
            ThrowIfError(err, "astral_conv_grammar_clear");
        }

        public void SetToolset(AstralToolset toolset, AstralNative.AstralToolChoiceMode choiceMode)
        {
            ThrowIfDisposed();
            if (toolset == null || !toolset.IsValid)
            {
                throw new ArgumentException("toolset is not valid", nameof(toolset));
            }

            int err = AstralNative.astral_conv_set_toolset(m_handle, toolset.Handle, (uint)choiceMode);
            ThrowIfError(err, "astral_conv_set_toolset");
        }

        public void ClearToolset()
        {
            ThrowIfDisposed();
            int err = AstralNative.astral_conv_clear_toolset(m_handle);
            ThrowIfError(err, "astral_conv_clear_toolset");
        }

        public int ReadStream(NativeArray<byte> outBuffer, uint timeoutMs = DefaultReadTimeoutMs)
        {
            ThrowIfDisposed();
            var span = AstralNative.AstralMutSpanU8.FromNativeArray(outBuffer);
            return AstralNative.astral_conv_stream_read(m_handle, span, timeoutMs);
        }

        public string ReadStreamAsString(uint timeoutMs = DefaultReadTimeoutMs)
        {
            int bytesRead = ReadStream(m_streamBuffer, timeoutMs);
            if (bytesRead <= AstralNative.ASTRAL_OK)
            {
                return null;
            }

            byte[] managed = new byte[bytesRead];
            NativeArray<byte>.Copy(m_streamBuffer, managed, bytesRead);
            return Encoding.UTF8.GetString(managed);
        }

        public unsafe int ReadStreamMeta(
            NativeArray<AstralNative.AstralTokenMeta> outEvents,
            uint timeoutMs = NonBlockingTimeoutMs)
        {
            ThrowIfDisposed();
            if (!outEvents.IsCreated)
            {
                throw new ArgumentException("outEvents must be created", nameof(outEvents));
            }

            return AstralNative.astral_conv_stream_read_meta(
                m_handle,
                (AstralNative.AstralTokenMeta*)outEvents.GetUnsafePtr(),
                (uint)outEvents.Length,
                timeoutMs);
        }

        public AstralConversationStats GetStats()
        {
            ThrowIfDisposed();

            var native = new AstralNative.AstralConvStats();
            int err = AstralNative.astral_conv_stats(m_handle, ref native);
            ThrowIfError(err, "astral_conv_stats");
            return new AstralConversationStats
            {
                slotId = native.slot_id,
                promptTokens = native.prompt_tokens,
                kvTokens = native.kv_tokens,
                generatedTokens = native.generated_tokens,
                firstTokenTimeMs = native.t_first_token_ms,
                tokensPerSecond = native.tok_per_s
            };
        }

        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_conv_destroy(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }
            if (m_streamBuffer.IsCreated)
            {
                m_streamBuffer.Dispose();
            }

            m_disposed = true;
        }

        private static AstralNative.AstralConvDesc BuildDesc(AstralModel model, AstralConversationConfig config)
        {
            return new AstralNative.AstralConvDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralConvDesc>(),
                model = model.Handle,
                max_tokens = config.maxTokens,
                temperature = config.temperature,
                top_k = config.topK,
                top_p = config.topP,
                stream_enabled = ToNativeBool(config.streamEnabled),
                seed = config.seed
            };
        }

        private void ThrowIfDisposed()
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralConversation));
            }
        }

        private static byte ToNativeBool(bool value)
        {
            return value ? NativeTrue : NativeFalse;
        }

        private static void DisposeIfCreated(ref NativeArray<byte> array)
        {
            if (array.IsCreated)
            {
                array.Dispose();
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
}
