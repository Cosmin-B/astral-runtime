// AstralSession.cs - Session wrapper with streaming support
//
//  Always call Dispose() when done (use 'using' statement)
//  No GC allocations in hot paths (streaming, token reading)
//  Single-threaded access per session (not thread-safe)

using System;
using System.Collections;
using System.Text;
using UnityEngine;
using Unity.Collections;

namespace Astral.Runtime
{
    /// <summary>
    /// Inference session handle.
    /// Implements IDisposable for RAII pattern.
    /// Thread-safety: Not thread-safe; single-threaded access per session.
    /// </summary>
    public class AstralSession : IDisposable
    {
        private AstralNative.AstralHandle m_handle;
        private bool m_disposed = false;
        private AstralSessionConfig m_config;
        private AstralModel m_model;

        // Pre-allocated buffers to avoid GC allocations
        private NativeArray<byte> m_streamBuffer;
        private const int STREAM_BUFFER_SIZE = 4096; // 4KB buffer for streaming

        /// <summary>
        /// Get native handle (for internal use).
        /// </summary>
        internal AstralNative.AstralHandle Handle => m_handle;

        /// <summary>
        /// Check if session is valid.
        /// </summary>
        public bool IsValid => !m_disposed && m_handle.IsValid;

        /// <summary>
        /// Create an inference session.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        /// <param name="model">Model to use (must remain valid for session lifetime)</param>
        /// <param name="config">Session configuration (null = default)</param>
        /// <returns>Created session (must be disposed)</returns>
        /// <exception cref="AstralException">Thrown if session creation fails</exception>
        public static AstralSession Create(AstralModel model, AstralSessionConfig config = null)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            if (!model.IsValid)
            {
                throw new ArgumentException("Model is not valid", nameof(model));
            }

            config = config ?? AstralSessionConfig.Default;

            var desc = new AstralNative.AstralSessionDesc
            {
                model = model.Handle,
                max_tokens = config.maxTokens,
                temperature = config.temperature,
                top_k = config.topK,
                top_p = config.topP,
                stream_enabled = (byte)(config.streamEnabled ? 1 : 0),
                seed = config.seed
            };

            AstralNative.AstralHandle handle;
            int err = AstralNative.astral_session_create(ref desc, out handle);

            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"Failed to create session: {AstralRuntime.GetErrorString(err)}", err);
            }

            var session = new AstralSession
            {
                m_handle = handle,
                m_config = config,
                m_model = model,
                m_streamBuffer = new NativeArray<byte>(STREAM_BUFFER_SIZE, Allocator.Persistent)
            };

            Debug.Log($"[Astral] Session created (max_tokens={config.maxTokens}, temp={config.temperature}, streaming={config.streamEnabled})");

            return session;
        }

        /// <summary>
        /// Feed a prompt chunk.
        /// Call multiple times for long prompts, setting finalize=true on the last chunk.
        /// Thread-safety: Not thread-safe; single-threaded access per session.
        /// </summary>
        /// <param name="promptChunk">UTF-8 prompt text</param>
        /// <param name="finalize">True if this is the last chunk</param>
        /// <exception cref="AstralException">Thrown if feed fails</exception>
        public void Feed(string promptChunk, bool finalize = true)
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            if (string.IsNullOrEmpty(promptChunk))
            {
                // Empty chunk is valid if finalize=true (e.g., to signal end of prompt)
                if (finalize)
                {
                    var emptySpan = new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 };
                    int err = AstralNative.astral_session_feed(m_handle, emptySpan, 1);
                    if (err != AstralNative.ASTRAL_OK)
                    {
                        throw new AstralException($"Failed to finalize feed: {AstralRuntime.GetErrorString(err)}", err);
                    }
                }
                return;
            }

            NativeArray<byte> tempArray;
            var span = AstralNative.AstralSpanU8.FromString(promptChunk, out tempArray);

            try
            {
                int err = AstralNative.astral_session_feed(m_handle, span, (byte)(finalize ? 1 : 0));
                if (err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"Failed to feed prompt: {AstralRuntime.GetErrorString(err)}", err);
                }
            }
            finally
            {
                if (tempArray.IsCreated)
                {
                    tempArray.Dispose();
                }
            }
        }

        /// <summary>
        /// Feed a prompt from NativeArray (zero-copy).
        ///  No GC allocations.
        /// </summary>
        /// <param name="promptChunk">UTF-8 prompt data</param>
        /// <param name="finalize">True if this is the last chunk</param>
        public void Feed(NativeArray<byte> promptChunk, bool finalize = true)
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            var span = AstralNative.AstralSpanU8.FromNativeArray(promptChunk);
            int err = AstralNative.astral_session_feed(m_handle, span, (byte)(finalize ? 1 : 0));

            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"Failed to feed prompt: {AstralRuntime.GetErrorString(err)}", err);
            }
        }

        /// <summary>
        /// Start decoding (non-blocking).
        /// Submits work to thread pool; returns immediately.
        /// Thread-safety: Not thread-safe; single-threaded access per session.
        /// </summary>
        /// <exception cref="AstralException">Thrown if decode fails</exception>
        public void Decode()
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            int err = AstralNative.astral_session_decode(m_handle);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"Failed to start decode: {AstralRuntime.GetErrorString(err)}", err);
            }
        }

        /// <summary>
        /// Reset session for reuse (clears prompt/stream state and provider KV/cache).
        /// Not thread-safe; must not be called concurrently with ReadStream.
        /// </summary>
        public void Reset(AstralSessionConfig config = null)
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            config = config ?? m_config ?? AstralSessionConfig.Default;

            var desc = new AstralNative.AstralSessionDesc
            {
                model = m_model.Handle,
                max_tokens = config.maxTokens,
                temperature = config.temperature,
                top_k = config.topK,
                top_p = config.topP,
                stream_enabled = (byte)(config.streamEnabled ? 1 : 0),
                seed = config.seed
            };

            int err = AstralNative.astral_session_reset(m_handle, ref desc);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"Failed to reset session: {AstralRuntime.GetErrorString(err)}", err);
            }

            m_config = config;
        }

        /// <summary>
        /// Request cancellation for an in-flight decode.
        /// Thread-safety: Safe to call from any thread, but this wrapper is not synchronized.
        /// </summary>
        public void Cancel()
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            int err = AstralNative.astral_session_cancel(m_handle);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"Failed to cancel session: {AstralRuntime.GetErrorString(err)}", err);
            }
        }

        /// <summary>
        /// Query current session state.
        /// Thread-safety: Safe to call from any thread.
        /// </summary>
        public AstralSessionState GetState()
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            uint state;
            int err = AstralNative.astral_session_state(m_handle, out state);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"Failed to query session state: {AstralRuntime.GetErrorString(err)}", err);
            }

            return (AstralSessionState)state;
        }

        /// <summary>
        /// Wait for session completion.
        /// Returns true if completed (success or canceled), false if timed out.
        /// Thread-safety: Safe to call from any thread.
        /// </summary>
        public bool Wait(uint timeoutMs)
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            int err = WaitResult(timeoutMs);
            if (err == AstralNative.ASTRAL_OK || err == AstralNative.ASTRAL_E_CANCELED)
            {
                return true;
            }

            if (err == AstralNative.ASTRAL_E_TIMEOUT)
            {
                return false;
            }

            throw new AstralException($"Failed to wait for session: {AstralRuntime.GetErrorString(err)}", err);
        }

        /// <summary>
        /// Wait for session completion and return the native status code.
        /// Returns: ASTRAL_OK (success), ASTRAL_E_CANCELED (canceled), ASTRAL_E_TIMEOUT (deadline), or other error.
        /// </summary>
        public int WaitResult(uint timeoutMs)
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            return AstralNative.astral_session_wait(m_handle, timeoutMs);
        }

        /// <summary>
        /// Read tokens from stream (blocking).
        ///  No GC allocations.
        ///
        /// Thread-safety:
        /// - Safe for a single consumer thread calling this concurrently with the decode worker.
        /// - Not safe to call from multiple consumer threads concurrently.
        /// </summary>
        /// <param name="outBuffer">Output buffer for UTF-8 token data</param>
        /// <param name="timeoutMs">Timeout in milliseconds (0 = non-blocking)</param>
        /// <returns>Bytes written to outBuffer (>= 0), or error code (< 0)</returns>
        public int ReadStream(NativeArray<byte> outBuffer, uint timeoutMs = 100)
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            var span = AstralNative.AstralMutSpanU8.FromNativeArray(outBuffer);
            return AstralNative.astral_stream_read(m_handle, span, timeoutMs);
        }

        /// <summary>
        /// Read tokens from stream into internal buffer (convenience method).
        ///  Allocates managed string (use ReadStream(NativeArray) for zero-alloc).
        /// </summary>
        /// <param name="timeoutMs">Timeout in milliseconds</param>
        /// <returns>UTF-8 decoded string, or null if no data available</returns>
        public string ReadStreamAsString(uint timeoutMs = 100)
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            int bytesRead = ReadStream(m_streamBuffer, timeoutMs);

            if (bytesRead > 0)
            {
                // Convert UTF-8 to string (GC allocation)
                unsafe
                {
                    fixed (byte* ptr = &m_streamBuffer[0])
                    {
                        return Encoding.UTF8.GetString(ptr, bytesRead);
                    }
                }
            }
            else if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
            {
                return null; // No data available (not an error)
            }
            else if (bytesRead < 0)
            {
                throw new AstralException($"Stream read failed: {AstralRuntime.GetErrorString(bytesRead)}", bytesRead);
            }

            return null;
        }

        /// <summary>
        /// Stream all tokens via callback (convenience method).
        /// Blocks until all tokens are received.
        ///  Callback is called from this thread (not worker thread).
        /// </summary>
        /// <param name="onToken">Callback for each token chunk (UTF-8 string)</param>
        /// <param name="timeoutMs">Timeout per read in milliseconds</param>
        public void StreamAll(Action<string> onToken, uint timeoutMs = 100)
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            if (!m_config.streamEnabled)
            {
                throw new InvalidOperationException("Streaming is not enabled for this session");
            }

            while (true)
            {
                int bytesRead = ReadStream(m_streamBuffer, timeoutMs);

                if (bytesRead > 0)
                {
                    unsafe
                    {
                        fixed (byte* ptr = &m_streamBuffer[0])
                        {
                            string token = Encoding.UTF8.GetString(ptr, bytesRead);
                            onToken?.Invoke(token);
                        }
                    }
                    continue;
                }

                if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
                {
                    continue;
                }

                if (bytesRead < 0)
                {
                    throw new AstralException($"Stream read failed: {AstralRuntime.GetErrorString(bytesRead)}", bytesRead);
                }

                // End of stream (0).
                break;
            }
        }

        /// <summary>
        /// Unity coroutine for streaming tokens.
        /// Use with StartCoroutine() in MonoBehaviour.
        /// </summary>
        /// <param name="onToken">Callback for each token chunk</param>
        /// <param name="onComplete">Callback when streaming is complete</param>
        /// <returns>IEnumerator for Unity coroutine</returns>
        public IEnumerator StreamCoroutine(Action<string> onToken, Action onComplete = null)
        {
            if (!m_config.streamEnabled)
            {
                Debug.LogError("[Astral] Streaming is not enabled for this session");
                onComplete?.Invoke();
                yield break;
            }

            while (true)
            {
                int bytesRead = ReadStream(m_streamBuffer, 0); // Non-blocking

                if (bytesRead > 0)
                {
                    // Convert UTF-8 to string
                    unsafe
                    {
                        fixed (byte* ptr = &m_streamBuffer[0])
                        {
                            string token = Encoding.UTF8.GetString(ptr, bytesRead);
                            onToken?.Invoke(token);
                        }
                    }
                }
                else if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
                {
                    // No data available; wait one frame and retry
                    yield return null;
                }
                else if (bytesRead < 0)
                {
                    Debug.LogError($"[Astral] Stream read failed: {AstralRuntime.GetErrorString(bytesRead)}");
                    break;
                }
                else
                {
                    // End of stream
                    break;
                }
            }

            onComplete?.Invoke();
        }

        /// <summary>
        /// Get session statistics.
        /// Thread-safety: Safe to call concurrently with other session operations.
        /// </summary>
        /// <returns>Session statistics</returns>
        public AstralStats GetStats()
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(nameof(AstralSession));
            }

            AstralNative.AstralStats nativeStats;
            int err = AstralNative.astral_session_stats(m_handle, out nativeStats);

            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"Failed to get stats: {AstralRuntime.GetErrorString(err)}", err);
            }

            return new AstralStats
            {
                initTimeMs = nativeStats.t_init_ms,
                firstTokenTimeMs = nativeStats.t_first_token_ms,
                tokensPerSecond = nativeStats.tok_per_s,
                bytesCommitted = nativeStats.bytes_committed,
                bytesReserved = nativeStats.bytes_reserved
            };
        }

        /// <summary>
        /// Dispose session (RAII pattern).
        /// Thread-safety: Not thread-safe; must not be in use.
        /// </summary>
        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_streamBuffer.IsCreated)
            {
                m_streamBuffer.Dispose();
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_session_destroy(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_disposed = true;
        }

        ~AstralSession()
        {
            if (!m_disposed)
            {
                Debug.LogWarning("[Astral] Session was not disposed properly. Always use 'using' statement or call Dispose().");
                Dispose();
            }
        }
    }

    /// <summary>
    /// Configuration for inference session.
    /// Immutable after session is created.
    /// </summary>
    [Serializable]
    public class AstralSessionConfig
    {
        /// <summary>
        /// Maximum tokens to generate.
        ///  Higher values allow longer responses but take more time.
        /// Recommended: 256 (short); 512 (medium); 1024 (long); 2048 (very long)
        /// </summary>
        public uint maxTokens = 512;

        /// <summary>
        /// Sampling temperature (0.0 = greedy, 1.0 = diverse).
        ///  Lower = more deterministic; higher = more creative.
        /// Recommended: 0.7 (balanced); 0.1 (factual); 1.0 (creative)
        /// </summary>
        public float temperature = 0.7f;

        /// <summary>
        /// Top-K sampling (0 = disabled).
        /// Only consider top K tokens by probability.
        /// Recommended: 40 (balanced); 1 (greedy); 80 (diverse)
        /// </summary>
        public uint topK = 40;

        /// <summary>
        /// Top-P (nucleus) sampling (0.0-1.0).
        /// Only consider tokens with cumulative probability > P.
        /// Recommended: 0.9 (balanced); 1.0 (disabled); 0.5 (focused)
        /// </summary>
        public float topP = 0.9f;

        /// <summary>
        /// Enable token streaming.
        /// If true, tokens are emitted as they are generated (use ReadStream/StreamAll).
        /// If false, all tokens are buffered until generation completes.
        /// </summary>
        public bool streamEnabled = true;

        /// <summary>
        /// RNG seed for sampling (0 = auto).
        /// Set to a non-zero value for deterministic generation (given same prompt + model).
        /// </summary>
        public uint seed = 0;

        /// <summary>
        /// Default configuration.
        /// </summary>
        public static AstralSessionConfig Default => new AstralSessionConfig();

        /// <summary>
        /// Greedy (deterministic) configuration.
        /// </summary>
        public static AstralSessionConfig Greedy => new AstralSessionConfig
        {
            maxTokens = 512,
            temperature = 0.1f,
            topK = 1,
            topP = 1.0f,
            streamEnabled = true,
            seed = 1
        };

        /// <summary>
        /// Creative (diverse) configuration.
        /// </summary>
        public static AstralSessionConfig Creative => new AstralSessionConfig
        {
            maxTokens = 1024,
            temperature = 1.0f,
            topK = 80,
            topP = 0.95f,
            streamEnabled = true
        };
    }

    /// <summary>
    /// Session state (mirrors native state machine).
    /// </summary>
    public enum AstralSessionState : uint
    {
        Idle = 0,
        FeedingPrompt = 1,
        Decoding = 2,
        Completed = 3,
        Canceled = 4,
        Failed = 5
    }

    /// <summary>
    /// Session statistics.
    /// </summary>
    [Serializable]
    public struct AstralStats
    {
        public double initTimeMs;
        public double firstTokenTimeMs;
        public double tokensPerSecond;
        public ulong bytesCommitted;
        public ulong bytesReserved;

        public override string ToString()
        {
            return $"AstralStats(init={initTimeMs:F2}ms, first_token={firstTokenTimeMs:F2}ms, tok/s={tokensPerSecond:F2}, " +
                   $"mem={bytesCommitted / (1024 * 1024)}MB/{bytesReserved / (1024 * 1024)}MB)";
        }
    }
}
