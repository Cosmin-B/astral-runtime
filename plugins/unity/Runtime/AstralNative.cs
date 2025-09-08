// AstralNative.cs - Low-level P/Invoke wrapper for Astral native library
// Do not use directly - use AstralSession instead.
//
//  All structs use StructLayout(LayoutKind.Sequential) for stable memory layout
//  All P/Invoke uses CallingConvention.Cdecl
//  UTF-8 strings via Span (pointer + length), never assume NUL termination

using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    /// <summary>
    /// Low-level P/Invoke wrapper for Astral native library.
    /// Do not use directly - use AstralSession instead.
    /// Thread-safety: See individual function documentation.
    /// </summary>
    public static class AstralNative
    {
        // Platform-specific library names
#if UNITY_IOS && !UNITY_EDITOR
        private const string DllName = "__Internal";
#else
        private const string DllName = "astral_rt";
#endif

        // ====================================================================
        // Common Types
        // ====================================================================

        /// <summary>
        /// UTF-8 string span (immutable).
        /// No NUL terminator assumed.
        ///  Explicit padding for 64-bit platforms (16 bytes total).
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct AstralSpanU8
        {
            public IntPtr data;       // const uint8_t*
            public uint len;          // uint32_t
#if UNITY_64 || UNITY_EDITOR_64
            public uint _padding;     // Explicit padding for 64-bit
#endif

            /// <summary>
            /// Create span from NativeArray (UTF-8 encoded).
            /// </summary>
            public static unsafe AstralSpanU8 FromNativeArray(NativeArray<byte> arr)
            {
                return new AstralSpanU8
                {
                    data = (IntPtr)arr.GetUnsafeReadOnlyPtr(),
                    len = (uint)arr.Length
                };
            }

            /// <summary>
            /// Create span from managed string (allocates temporary UTF-8 array).
            /// WARNING: Caller must keep returned NativeArray alive until span is used.
            /// </summary>
            public static AstralSpanU8 FromString(string str, out NativeArray<byte> tempArray)
            {
                if (string.IsNullOrEmpty(str))
                {
                    tempArray = default;
                    return new AstralSpanU8 { data = IntPtr.Zero, len = 0 };
                }

                var utf8 = System.Text.Encoding.UTF8.GetBytes(str);
                tempArray = new NativeArray<byte>(utf8, Allocator.Temp);
                return FromNativeArray(tempArray);
            }
        }

        /// <summary>
        /// UTF-8 string span (mutable).
        /// Used for output buffers.
        ///  Explicit padding for 64-bit platforms (16 bytes total).
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct AstralMutSpanU8
        {
            public IntPtr data;       // uint8_t*
            public uint len;          // uint32_t
#if UNITY_64 || UNITY_EDITOR_64
            public uint _padding;     // Explicit padding for 64-bit
#endif

            /// <summary>
            /// Create mutable span from NativeArray.
            /// </summary>
            public static unsafe AstralMutSpanU8 FromNativeArray(NativeArray<byte> arr)
            {
                return new AstralMutSpanU8
                {
                    data = (IntPtr)arr.GetUnsafePtr(),
                    len = (uint)arr.Length
                };
            }
        }

        /// <summary>
        /// Opaque handle (model, session, embedder).
        /// Never dereference; only pass to Astral functions.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct AstralHandle
        {
            public ulong value;

            public bool IsValid => value != 0;
            public static AstralHandle Invalid => new AstralHandle { value = 0 };
        }

        // ====================================================================
        // Error Codes
        // ====================================================================

        public const int ASTRAL_OK = 0;          // Success
        public const int ASTRAL_E_INVALID = -1;  // Invalid parameter
        public const int ASTRAL_E_NOMEM = -2;    // Out of memory
        public const int ASTRAL_E_BUSY = -3;     // Resource busy (queue full)
        public const int ASTRAL_E_TIMEOUT = -4;  // Operation timed out
        public const int ASTRAL_E_STATE = -5;    // Invalid state
        public const int ASTRAL_E_BACKEND = -6;  // Backend error
        public const int ASTRAL_E_CANCELED = -7; // Canceled
        public const int ASTRAL_E_UNSUPPORTED = -8; // Unsupported

        // ====================================================================
        // Tunables / Limits
        // ====================================================================

        public const int ASTRAL_LOGPROBS_MAX = 16;

        // ====================================================================
        // Allocator
        // ====================================================================

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr AllocFn(IntPtr user, UIntPtr size, UIntPtr align);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void FreeFn(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align);

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAllocator
        {
            public IntPtr alloc;  // AllocFn
            public IntPtr free;   // FreeFn
            public IntPtr user;   // void* user data
        }

        // ====================================================================
        // Logging
        // ====================================================================

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LogFn(IntPtr user, int level, AstralSpanU8 msg);

        public const int ASTRAL_LOG_ERROR = 0;
        public const int ASTRAL_LOG_WARN = 1;
        public const int ASTRAL_LOG_INFO = 2;
        public const int ASTRAL_LOG_DEBUG = 3;
        public const int ASTRAL_LOG_TRACE = 4;

        // ====================================================================
        // Initialization
        // ====================================================================

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralInit
        {
            public AstralAllocator sys_alloc;   // System allocator (optional)
            public IntPtr log_cb;               // LogFn callback
            public IntPtr log_user;             // User data for log callback
            public ulong reserve_bytes;         // Virtual memory to reserve
            public uint thread_count;           // Worker threads (0 = auto)
            public uint numa_node;              // NUMA node (0xFFFFFFFF = any)
            public byte enable_hugepages;       // Try huge pages
        }

        /// <summary>
        /// Initialize Astral runtime.
        /// Must be called once before any other Astral functions.
        /// Thread-safety: Not thread-safe; call from main thread only.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_init(ref AstralInit cfg);

        /// <summary>
        /// Shutdown Astral runtime.
        /// All handles must be released before calling this.
        /// Thread-safety: Not thread-safe; call from main thread only.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_shutdown();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_version(out uint major, out uint minor, out uint patch);

        /// <summary>
        /// Check if a handle is valid.
        /// Returns 1 if valid, 0 if invalid/null.
        /// Thread-safety: Safe to call from any thread.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_handle_valid(AstralHandle handle);

        /// <summary>
        /// Get human-readable error string for an error code.
        /// Returns a static string (no need to free).
        /// Thread-safety: Safe to call from any thread.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr astral_error_string(int err);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr astral_last_error();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_clear_last_error();

        // ====================================================================
        // Model
        // ====================================================================

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralModelDesc
        {
            public AstralSpanU8 model_path;  // Path to GGUF model file
            public AstralSpanU8 backend_name; // Optional override ("cpu", "mock", ...)
            public uint gpu_layers;          // Layers to offload to GPU
            public uint n_ctx;               // Context size (tokens)
            public uint n_batch;             // Batch size for prompt processing
            public uint n_threads;           // Threads for backend (0 = auto)
            public byte embeddings_only;     // Embeddings-only mode
        }

        /// <summary>
        /// Load a GGUF model.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_load(ref AstralModelDesc desc, out AstralHandle out_model);

        /// <summary>
        /// Release a model.
        /// Thread-safety: Not thread-safe; must not be in use by any session.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_model_release(AstralHandle model);

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralModelInfo
        {
            public uint vocab_size;
            public uint ctx_size;
            public int token_bos;
            public int token_eos;
        }

        // Mirrors native typedef: `typedef uint64_t AstralCaps;`
        public const ulong ASTRAL_CAP_NONE = 0;
        public const ulong ASTRAL_CAP_SAMPLER_EXT = 1UL << 0;
        public const ulong ASTRAL_CAP_STOP_SEQS = 1UL << 1;

        public const ulong ASTRAL_CAP_EMBEDDINGS = 1UL << 16;
        public const ulong ASTRAL_CAP_GPU_OFFLOAD = 1UL << 17;
        public const ulong ASTRAL_CAP_LORA = 1UL << 18;
        public const ulong ASTRAL_CAP_GRAMMAR = 1UL << 19;
        public const ulong ASTRAL_CAP_LOGPROBS = 1UL << 20;
        public const ulong ASTRAL_CAP_KV_STATE = 1UL << 21;
        public const ulong ASTRAL_CAP_SLOTS = 1UL << 22;
        public const ulong ASTRAL_CAP_GRAMMAR_GBNF = 1UL << 23;
        public const ulong ASTRAL_CAP_GRAMMAR_JSON_SCHEMA = 1UL << 24;

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralModelLimits
        {
            public uint vocab_size;
            public uint ctx_size;
            public uint max_batch;
            public uint max_slots;
        }

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_info(AstralHandle model, out AstralModelInfo out_info);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_caps(AstralHandle model, out ulong out_caps);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_limits(AstralHandle model, out AstralModelLimits out_limits);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_embedding_dim(AstralHandle model, out uint out_dim);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_tokenize(
            AstralHandle model,
            AstralSpanU8 text,
            int* out_tokens,
            uint max_tokens,
            byte add_special,
            byte parse_special,
            out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_detokenize(
            AstralHandle model,
            int* tokens,
            uint count,
            AstralMutSpanU8 out_text,
            out uint out_len);

        // ====================================================================
        // Session
        // ====================================================================

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralSessionDesc
        {
            public AstralHandle model;       // Model handle
            public uint max_tokens;          // Maximum tokens to generate
            public float temperature;        // Sampling temperature
            public uint top_k;               // Top-K sampling
            public float top_p;              // Top-P (nucleus) sampling
            public byte stream_enabled;      // Enable token streaming
            public uint seed;                // RNG seed (0 = auto)
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralSamplerDesc
        {
            public uint size;
            public float temperature;
            public uint top_k;
            public float top_p;
            public float min_p;
            public float typical_p;
            public float repeat_penalty;
            public int repeat_last_n;
            public byte penalize_nl;
            public byte _padding0;
            public byte _padding1;
            public byte _padding2;
            public float presence_penalty;
            public float frequency_penalty;
            public uint mirostat;
            public float mirostat_tau;
            public float mirostat_eta;

            public static AstralSamplerDesc DefaultFromSession(in AstralSessionDesc session)
            {
                return new AstralSamplerDesc
                {
                    size = (uint)Marshal.SizeOf<AstralSamplerDesc>(),
                    temperature = session.temperature,
                    top_k = session.top_k,
                    top_p = session.top_p,
                    min_p = 0.0f,
                    typical_p = 1.0f,
                    repeat_penalty = 1.0f,
                    repeat_last_n = 0,
                    penalize_nl = 0,
                    _padding0 = 0,
                    _padding1 = 0,
                    _padding2 = 0,
                    presence_penalty = 0.0f,
                    frequency_penalty = 0.0f,
                    mirostat = 0,
                    mirostat_tau = 0.0f,
                    mirostat_eta = 0.0f,
                };
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct AstralTokenMeta
        {
            public uint token_id;
            public uint top_n;
            public float logprob;
            public fixed uint top_token_ids[ASTRAL_LOGPROBS_MAX];
            public fixed float top_logprobs[ASTRAL_LOGPROBS_MAX];
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAdapterDesc
        {
            public uint size;
            public AstralSpanU8 path;
        }

        /// <summary>
        /// Create an inference session.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_create(ref AstralSessionDesc desc, out AstralHandle out_session);

        /// <summary>
        /// Destroy a session.
        /// Thread-safety: Not thread-safe; must not be in use.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_session_destroy(AstralHandle session);

        /// <summary>
        /// Feed a prompt chunk.
        /// Thread-safety: Not thread-safe; single-threaded access per session.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_feed(AstralHandle session, AstralSpanU8 prompt_chunk, byte finalize);

        /// <summary>
        /// Start decoding (non-blocking).
        /// Thread-safety: Not thread-safe; single-threaded access per session.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_decode(AstralHandle session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_cancel(AstralHandle session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_state(AstralHandle session, out uint out_state);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_wait(AstralHandle session, uint timeout_ms);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_reset(AstralHandle session, ref AstralSessionDesc desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_sampler(AstralHandle session, ref AstralSamplerDesc desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_session_penalty_prompt_set_tokens(AstralHandle session, int* tokens, uint count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_stop_clear(AstralHandle session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_stop_add_utf8(AstralHandle session, AstralSpanU8 utf8);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_stop_set_utf8(AstralHandle session, IntPtr seqs, uint count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_logprobs(AstralHandle session, uint n_probs);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_stream_read_meta(AstralHandle session, AstralTokenMeta* out_events, uint capacity, uint timeout_ms);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_state_size(AstralHandle session, out ulong out_bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_state_save(AstralHandle session, AstralMutSpanU8 out_buf, out ulong out_written);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_state_load(AstralHandle session, AstralSpanU8 state_bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_adapter_load(AstralHandle model, ref AstralAdapterDesc desc, out AstralHandle out_adapter);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_model_adapter_release(AstralHandle adapter);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_adapters_clear(AstralHandle session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_adapters_add(AstralHandle session, AstralHandle adapter, float scale);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_grammar_gbnf(AstralHandle session, AstralSpanU8 gbnf, AstralSpanU8 root);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_grammar_json_schema(AstralHandle session, AstralSpanU8 json_schema);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_clear_grammar(AstralHandle session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_slot(AstralHandle session, uint slot_id);

        /// <summary>
        /// Read tokens from stream.
        /// Blocks up to timeout_ms if no data available.
        ///
        /// Thread-safety:
        /// - Safe for a single consumer thread calling this concurrently with the decode worker.
        /// - Not safe to call from multiple consumer threads concurrently.
        ///
        /// Returns: Bytes written (>= 0; 0 = end-of-stream), or error code (< 0).
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_stream_read(AstralHandle session, AstralMutSpanU8 out_buf, uint timeout_ms);

        // ====================================================================
        // Embeddings
        // ====================================================================

        /// <summary>
        /// Create an embeddings handle.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_embed_create(AstralHandle model, out AstralHandle out_embedder);

        /// <summary>
        /// Destroy an embeddings handle.
        /// Thread-safety: Not thread-safe; must not be in use.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_embed_destroy(AstralHandle emb);

        /// <summary>
        /// Enqueue text for embedding.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_embed_enqueue(AstralHandle emb, AstralSpanU8 text, out ulong out_ticket);

        /// <summary>
        /// Collect embedding vector for a ticket.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_embed_collect(AstralHandle emb, ulong ticket, AstralMutSpanU8 out_vector);

        // ====================================================================
        // Statistics
        // ====================================================================

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralStats
        {
            public double t_init_ms;         // Init time (model load) in ms
            public double t_first_token_ms;  // Time to first token in ms
            public double tok_per_s;         // Tokens per second
            public ulong bytes_committed;    // Committed memory in bytes
            public ulong bytes_reserved;     // Reserved virtual memory in bytes
        }

        /// <summary>
        /// Get session statistics.
        /// Thread-safety: Safe to call concurrently with other session operations.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_stats(AstralHandle session, out AstralStats out_stats);
    }
}
