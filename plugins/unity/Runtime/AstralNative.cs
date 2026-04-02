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

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralTokenizeRequest
        {
            public AstralSpanU8 text;
            public byte add_special;
            public byte parse_special;
            public ushort _reserved;
        }

        public enum AstralPromptSectionKind : uint
        {
            System = 1,
            Tools = 2,
            Memory = 3,
            History = 4,
            User = 5,
            Raw = 6
        }

        public enum AstralPromptCacheEvictionPolicy : uint
        {
            Fifo = 0
        }

        public enum AstralToolChoiceMode : uint
        {
            Auto = 0,
            Required = 1,
            TextOrTool = 2
        }

        [Flags]
        public enum AstralPromptCacheFlags : uint
        {
            None = 0,
            TrackStats = 1u << 0
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralPromptCacheDesc
        {
            public uint size;
            public uint max_entries;
            public uint max_tokens;
            public uint max_bytes;
            public AstralPromptCacheEvictionPolicy eviction_policy;
            public AstralPromptCacheFlags flags;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralPromptCacheKey
        {
            public uint size;
            public uint section_kind;
            public AstralHandle model;
            public ulong key;
            public uint generation;
            public uint _reserved0;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralPromptCacheStats
        {
            public uint size;
            public uint entries;
            public uint max_entries;
            public uint tokens;
            public uint max_tokens;
            public uint bytes;
            public uint max_bytes;
            public uint _reserved0;
            public ulong hits;
            public ulong misses;
            public ulong evictions;
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
        // Media Types (Vision / Audio)
        // ====================================================================

        public enum AstralImageFormat : uint
        {
            RGB8 = 0,
            RGBA8 = 1,
            RGB_F32 = 2
        }

        public enum AstralAudioFormat : uint
        {
            F32 = 0,
            I16 = 1
        }

        [Flags]
        public enum AstralMediaFlags : uint
        {
            None = 0,
            UseGpu = 1u << 0,
            Warmup = 1u << 1
        }

        [Flags]
        public enum AstralGpuRouteFlags : uint
        {
            None = 0,
            Device = 1u << 0,
            DeviceMask = 1u << 1,
            Stream = 1u << 2
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralImageDesc
        {
            public uint size;
            public AstralImageFormat format;
            public uint width;
            public uint height;
            public uint row_stride;
            public uint flags;
            public AstralSpanU8 pixels;
            public int gpu_device;
            public uint gpu_route_flags;
            public ulong gpu_device_mask;
            public IntPtr gpu_stream;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAudioDesc
        {
            public uint size;
            public AstralAudioFormat format;
            public uint channels;
            public uint sample_rate;
            public ulong frame_count;
            public AstralSpanU8 samples;
            public uint flags;
            public uint _padding0;
            public int gpu_device;
            public uint gpu_route_flags;
            public ulong gpu_device_mask;
            public IntPtr gpu_stream;
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
        public const int ASTRAL_E_NOT_FOUND = -9; // Not found

        // ====================================================================
        // Tunables / Limits
        // ====================================================================

        public const int ASTRAL_LOGPROBS_MAX = 16;
        public const int ASTRAL_SESSION_ADAPTERS_MAX = 8;
        public const uint ASTRAL_TOOL_CHOICE_AUTO = 0;
        public const uint ASTRAL_TOOL_CHOICE_REQUIRED = 1;
        public const uint ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL = 2;

        public enum AstralRequestKind : uint
        {
            None = 0,
            Session = 1,
            Conversation = 2,
            AgentChat = 3,
            Embedding = 4,
            MemorySearch = 5
        }

        public enum AstralRequestState : uint
        {
            Invalid = 0,
            Queued = 1,
            Running = 2,
            Completed = 3,
            Canceled = 4,
            Failed = 5
        }

        [Flags]
        public enum AstralRequestFlags : uint
        {
            None = 0,
            Stream = 1u << 0,
            Ticket = 1u << 1
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralRequestRef
        {
            public uint size;
            public AstralRequestKind kind;
            public AstralHandle owner;
            public ulong ticket;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralRequestStatus
        {
            public uint size;
            public AstralRequestKind kind;
            public AstralRequestState state;
            public AstralRequestFlags flags;
            public AstralHandle owner;
            public ulong ticket;
            public int result;
            public uint queue_depth;
        }

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

        public enum AstralModelSourceKind : uint
        {
            Path = 0,
            Memory = 1,
            IO = 2
        }

        public enum AstralModelPathRoot : uint
        {
            Raw = 0,
            Content = 1,
            Saved = 2,
            Cache = 3,
            Download = 4
        }

        public enum AstralModelPathResolveFlags : uint
        {
            None = 0
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralModelPathResolveDesc
        {
            public uint size;
            public AstralModelPathRoot root;
            public AstralSpanU8 path;
            public AstralSpanU8 content_root;
            public AstralSpanU8 saved_root;
            public AstralSpanU8 cache_root;
            public AstralSpanU8 download_root;
            public AstralModelPathResolveFlags flags;
            public uint _reserved0;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralModelIO
        {
            public IntPtr user;
            public IntPtr size;    // function pointer
            public IntPtr read_at; // function pointer
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralModelMediaDesc
        {
            public uint size;
            public AstralModelSourceKind source_kind;
            public uint flags;
            public uint image_min_tokens;
            public uint image_max_tokens;
            public uint _padding0;

            public AstralSpanU8 media_path;
            public AstralSpanU8 media_bytes;
            public AstralModelIO media_io;

            public int gpu_device;
            public uint gpu_route_flags;
            public ulong gpu_device_mask;
            public IntPtr gpu_stream;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralMediaInfo
        {
            public uint size;
            public uint supports_image;
            public uint supports_audio;
            public uint audio_sample_rate;
            public uint image_min_tokens;
            public uint image_max_tokens;
            public uint _padding0;
        }

        public enum AstralGpuSplitMode : int
        {
            None = 0,
            Layer = 1,
            Row = 2
        }

        [Flags]
        public enum AstralGpuConfigFlags : uint
        {
            None = 0,
            Main = 1u << 0,
            SplitMode = 1u << 1,
            Devices = 1u << 2,
            DeviceMask = 1u << 3,
            TensorSplit = 1u << 4
        }

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct AstralModelDesc
        {
            public uint size;                // sizeof(AstralModelDesc)
            public AstralModelSourceKind source_kind;
            public uint _padding0;

            // Sources
            public AstralSpanU8 model_path;  // PATH
            public AstralSpanU8 model_bytes; // MEMORY
            public AstralModelIO io;         // IO

            // Common options
            public AstralSpanU8 backend_name; // Optional override ("cpu", "mock", ...)
            public uint gpu_layers;          // Layers to offload to GPU
            public uint n_ctx;               // Context size (tokens)
            public uint n_batch;             // Batch size for prompt processing
            public uint n_threads;           // Threads for backend (0 = auto)
            public byte embeddings_only;     // Embeddings-only mode
#if UNITY_64 || UNITY_EDITOR_64
            public fixed byte _padding1[7];
#else
            public fixed byte _padding1[3];
#endif

            // CUDA multi-GPU routing fields consumed by backends that implement ggml CUDA routing.
            public int gpu_main;
            public int gpu_split_mode; // AstralGpuSplitMode
            public uint gpu_flags;     // AstralGpuConfigFlags
            public uint _padding2;
            public ulong gpu_device_mask;
            public IntPtr gpu_devices;        // int32_t*
            public uint gpu_device_count;
            public uint _padding3;
            public IntPtr gpu_tensor_split;   // float*
            public uint gpu_tensor_split_count;
            public uint _padding4;
        }

        /// <summary>
        /// Load a GGUF model.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_load(ref AstralModelDesc desc, out AstralHandle out_model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_load2(ref AstralModelDesc desc, out AstralHandle out_model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_path_resolve(
            ref AstralModelPathResolveDesc desc,
            AstralMutSpanU8 out_path,
            out uint out_len);

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
        public const ulong ASTRAL_CAP_IMAGE = 1UL << 25;
        public const ulong ASTRAL_CAP_AUDIO = 1UL << 26;
        public const ulong ASTRAL_CAP_MM_EMBEDDINGS = 1UL << 27;

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
        public static extern int astral_model_media_init(AstralHandle model, ref AstralModelMediaDesc desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_media_info(AstralHandle model, ref AstralMediaInfo out_info);

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
        public static extern int astral_tokenize_count(
            AstralHandle model,
            AstralSpanU8 text,
            byte add_special,
            byte parse_special,
            out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_tokenize_batch(
            AstralHandle model,
            AstralTokenizeRequest* requests,
            uint request_count,
            uint* out_offsets,
            int* out_tokens,
            uint max_tokens,
            out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_detokenize(
            AstralHandle model,
            int* tokens,
            uint count,
            AstralMutSpanU8 out_text,
            out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_detokenize_count(
            AstralHandle model,
            int* tokens,
            uint count,
            out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_prompt_cache_create(
            ref AstralPromptCacheDesc desc,
            out AstralHandle out_cache);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_prompt_cache_destroy(AstralHandle cache);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_prompt_cache_clear(AstralHandle cache);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_prompt_cache_stats(
            AstralHandle cache,
            ref AstralPromptCacheStats out_stats);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_prompt_cache_save_size(
            AstralHandle cache,
            out uint out_bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_prompt_cache_save(
            AstralHandle cache,
            AstralMutSpanU8 out_bytes,
            out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_prompt_cache_load(
            ref AstralPromptCacheDesc desc,
            AstralSpanU8 bytes,
            out AstralHandle out_cache);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_prompt_cache_key_from_bytes(
            AstralHandle model,
            AstralPromptSectionKind section_kind,
            uint generation,
            AstralSpanU8 bytes,
            ref AstralPromptCacheKey out_key);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_prompt_cache_put_tokens(
            AstralHandle cache,
            ref AstralPromptCacheKey key,
            int* tokens,
            uint token_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_prompt_cache_get_tokens(
            AstralHandle cache,
            ref AstralPromptCacheKey key,
            int* out_tokens,
            uint max_tokens,
            out uint out_token_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_prompt_cache_get_token_view(
            AstralHandle cache,
            ref AstralPromptCacheKey key,
            out IntPtr out_tokens,
            out uint out_token_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_executor_configure(AstralHandle model, ref AstralExecutorDesc desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_executor_tune(AstralHandle model, ref AstralExecutorTuning tuning);

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
        public struct AstralExecutorDesc
        {
            public uint size;
            public uint max_slots;
            public uint max_batch_tokens;
            public uint worker_hint;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralExecutorTuning
        {
            public uint size;
            public uint max_prompt_tokens_per_slot_tick;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralConvDesc
        {
            public uint size;
            public AstralHandle model;
            public uint max_tokens;
            public float temperature;
            public uint top_k;
            public float top_p;
            public byte stream_enabled;
            public uint seed;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralConvStats
        {
            public uint slot_id;
            public uint prompt_tokens;
            public uint kv_tokens;
            public uint _padding0;
            public ulong generated_tokens;
            public double t_first_token_ms;
            public double tok_per_s;
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

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAdapterInfo
        {
            public uint size;
            public AstralHandle model;
            public uint path_bytes;
            public uint refcount;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralToolDesc
        {
            public uint size;
            public uint tool_id;
            public AstralSpanU8 name;
            public AstralSpanU8 description;
            public AstralSpanU8 json_schema;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralToolsetDesc
        {
            public uint size;
            public uint tool_count;
            public uint choice_mode;
            public uint _reserved0;
            public IntPtr tools;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralToolInfo
        {
            public uint size;
            public uint tool_id;
            public AstralSpanU8 name;
            public AstralSpanU8 description;
            public AstralSpanU8 json_schema;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralToolCallResult
        {
            public uint size;
            public uint tool_id;
            public int parse_status;
            public uint _reserved0;
            public AstralSpanU8 name;
            public AstralSpanU8 arguments_json;
        }

        public enum AstralChunkMode : uint
        {
            None = 0,
            Char = 1,
            Word = 2,
            Sentence = 3,
            Token = 4
        }

        [Flags]
        public enum AstralChunkFlags : uint
        {
            None = 0,
            KeepEmpty = 1u << 0
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralChunkerDesc
        {
            public uint size;
            public AstralChunkMode mode;
            public uint max_units;
            public uint overlap_units;
            public uint document_id;
            public uint group_id;
            public AstralChunkFlags flags;
            public uint _reserved0;
            public AstralSpanU8 delimiters;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralChunkRange
        {
            public uint size;
            public uint document_id;
            public uint chunk_id;
            public uint group_id;
            public uint byte_begin;
            public uint byte_end;
            public uint token_begin;
            public uint token_end;
        }

        public enum AstralMemoryMetric : uint
        {
            Dot = 0,
            Cosine = 1,
            L2 = 2
        }

        public enum AstralMemoryIndexKind : uint
        {
            Flat = 0,
            Graph = 1
        }

        public enum AstralMemoryStorageKind : uint
        {
            F32 = 0,
            Q8 = 1
        }

        public const uint ASTRAL_MEMORY_GROUP_ANY = 0xFFFFFFFFu;

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralMemoryIndexDesc
        {
            public uint size;
            public uint dim;
            public uint capacity;
            public AstralMemoryMetric metric;
            public AstralMemoryIndexKind index_kind;
            public uint graph_neighbors;
            public uint graph_search;
            public AstralMemoryStorageKind storage_kind;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralMemoryRecord
        {
            public uint size;
            public uint group_id;
            public ulong key;
            public uint document_id;
            public uint chunk_id;
            public uint flags;
            public uint _reserved0;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralMemorySearchDesc
        {
            public uint size;
            public uint top_k;
            public uint group_id;
            public uint flags;
            public uint graph_search;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralMemorySearchResult
        {
            public uint size;
            public uint group_id;
            public ulong key;
            public uint document_id;
            public uint chunk_id;
            public float score;
            public uint flags;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralMemoryStats
        {
            public uint size;
            public uint dim;
            public uint capacity;
            public uint count;
            public AstralMemoryMetric metric;
            public AstralMemoryIndexKind index_kind;
            public uint graph_neighbors;
            public uint graph_search;
            public uint graph_levels;
            public AstralMemoryStorageKind storage_kind;
            public ulong vector_bytes;
            public ulong metadata_bytes;
            public ulong graph_bytes;
            public ulong total_bytes;
            public ulong save_bytes;
        }

        public enum AstralAgentRole : uint
        {
            System = 1,
            User = 2,
            Assistant = 3,
            Tool = 4
        }

        [Flags]
        public enum AstralAgentFlags : uint
        {
            None = 0
        }

        [Flags]
        public enum AstralAgentChatFlags : uint
        {
            None = 0,
            Warmup = 1u << 0
        }

        public enum AstralAgentOverflowPolicy : uint
        {
            Reject = 0,
            TruncateOldest = 1
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAgentDesc
        {
            public uint size;
            public AstralAgentFlags flags;
            public AstralHandle model;
            public AstralHandle prompt_cache;
            public AstralHandle memory_index;
            public AstralHandle toolset;
            public uint max_tokens;
            public float temperature;
            public uint top_k;
            public float top_p;
            public byte stream_enabled;
            public byte _padding0;
            public byte _padding1;
            public byte _padding2;
            public uint seed;
            public uint tool_choice_mode;
            public uint max_messages;
            public uint max_prompt_bytes;
            public AstralAgentOverflowPolicy overflow_policy;
            public uint slot_affinity;
            public AstralSpanU8 system_prompt;
            public AstralSpanU8 summary;
            public AstralSpanU8 memory_context;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAgentMessage
        {
            public uint size;
            public AstralAgentRole role;
            public AstralSpanU8 content;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAgentChatDesc
        {
            public uint size;
            public AstralAgentChatFlags flags;
            public AstralSpanU8 user_message;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAgentMemoryContextDesc
        {
            public uint size;
            public uint result_count;
            public uint chunk_count;
            public uint max_bytes;
            public AstralSpanU8 document_text;
            public AstralSpanU8 separator;
            public IntPtr chunks;
            public IntPtr results;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAgentChatResult
        {
            public uint size;
            public uint state;
            public uint prompt_bytes;
            public uint history_messages;
            public uint prompt_tokens;
            public uint prompt_cache_reused_tokens;
            public uint prompt_cache_new_tokens;
            public uint prompt_cache_hits;
            public uint prompt_cache_misses;
            public int last_error;
            public double prompt_build_ms;
            public ulong generated_tokens;
            public double t_first_token_ms;
            public double tok_per_s;
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

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_session_feed_tokens(AstralHandle session, int* tokens, uint token_count, byte finalize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_system_prompt(AstralHandle session, AstralSpanU8 system_prompt);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_feed_image(AstralHandle session, ref AstralImageDesc image, byte finalize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_feed_audio(AstralHandle session, ref AstralAudioDesc audio, byte finalize);

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
        public static extern int astral_model_adapter_info(AstralHandle adapter, ref AstralAdapterInfo out_info);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_model_adapter_path_copy(
            AstralHandle adapter,
            AstralMutSpanU8 out_path,
            out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_model_adapter_release(AstralHandle adapter);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_adapters_clear(AstralHandle session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_adapters_add(AstralHandle session, AstralHandle adapter, float scale);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_adapters_count(AstralHandle session, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_adapters_get(AstralHandle session, uint index, out AstralHandle out_adapter, out float out_scale);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_adapters_set_scale(AstralHandle session, uint index, float scale);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_grammar_gbnf(AstralHandle session, AstralSpanU8 gbnf, AstralSpanU8 root);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_grammar_json_schema(AstralHandle session, AstralSpanU8 json_schema);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_clear_grammar(AstralHandle session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_toolset_create(ref AstralToolsetDesc desc, out AstralHandle out_toolset);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_toolset_destroy(AstralHandle toolset);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_toolset_count(AstralHandle toolset, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_toolset_get(AstralHandle toolset, uint index, ref AstralToolInfo out_info);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_toolset_parse_call(AstralHandle toolset, AstralSpanU8 generated_text, ref AstralToolCallResult out_result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_set_toolset(AstralHandle session, AstralHandle toolset, uint choice_mode);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_session_clear_toolset(AstralHandle session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_create(ref AstralConvDesc desc, out AstralHandle out_conv);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_conv_destroy(AstralHandle conv);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_feed(AstralHandle conv, AstralSpanU8 prompt_chunk, byte finalize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_conv_feed_tokens(AstralHandle conv, int* tokens, uint token_count, byte finalize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_set_system_prompt(AstralHandle conv, AstralSpanU8 system_prompt);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_feed_image(AstralHandle conv, ref AstralImageDesc image, byte finalize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_feed_audio(AstralHandle conv, ref AstralAudioDesc audio, byte finalize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_decode(AstralHandle conv);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_cancel(AstralHandle conv);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_state(AstralHandle conv, out uint out_state);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_wait(AstralHandle conv, uint timeout_ms);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_reset(AstralHandle conv, ref AstralConvDesc desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_set_sampler(AstralHandle conv, ref AstralSamplerDesc desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_conv_penalty_prompt_set_tokens(AstralHandle conv, int* tokens, uint count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_stop_clear(AstralHandle conv);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_stop_add_utf8(AstralHandle conv, AstralSpanU8 utf8);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_stop_set_utf8(AstralHandle conv, IntPtr seqs, uint count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_set_logprobs(AstralHandle conv, uint n_probs);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_grammar_set_gbnf(AstralHandle conv, AstralSpanU8 gbnf, AstralSpanU8 root);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_grammar_set_json_schema(AstralHandle conv, AstralSpanU8 json_schema);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_grammar_clear(AstralHandle conv);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_set_toolset(AstralHandle conv, AstralHandle toolset, uint choice_mode);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_clear_toolset(AstralHandle conv);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_stream_read(AstralHandle conv, AstralMutSpanU8 out_buf, uint timeout_ms);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_conv_stream_read_meta(AstralHandle conv, AstralTokenMeta* out_events, uint capacity, uint timeout_ms);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_conv_stats(AstralHandle conv, ref AstralConvStats out_stats);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_chunk_count(ref AstralChunkerDesc desc, AstralSpanU8 text, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_chunk_ranges(ref AstralChunkerDesc desc, AstralSpanU8 text, AstralChunkRange* out_ranges, uint max_ranges, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_chunk_text_copy(AstralSpanU8 text, ref AstralChunkRange range, AstralMutSpanU8 out_text, out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_token_chunk_count(ref AstralChunkerDesc desc, uint token_count, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_token_chunk_ranges(ref AstralChunkerDesc desc, uint token_count, AstralChunkRange* out_ranges, uint max_ranges, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_create(ref AstralMemoryIndexDesc desc, out AstralHandle out_index);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_memory_destroy(AstralHandle index);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_count(AstralHandle index, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_stats(AstralHandle index, ref AstralMemoryStats out_stats);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_clear(AstralHandle index);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_get_record(AstralHandle index, ulong key, ref AstralMemoryRecord out_record);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_update_record(AstralHandle index, ulong key, ref AstralMemoryRecord record);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_memory_add_batch(AstralHandle index, AstralMemoryRecord* records, float* vectors, uint count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_remove(AstralHandle index, ulong key);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_memory_search(AstralHandle index, ref AstralMemorySearchDesc desc, float* query, AstralMemorySearchResult* out_results, uint max_results, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_memory_search_begin(AstralHandle index, ref AstralMemorySearchDesc desc, float* query, out AstralHandle out_cursor);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int astral_memory_search_fetch(AstralHandle cursor, AstralMemorySearchResult* out_results, uint max_results, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_memory_search_end(AstralHandle cursor);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_create(ref AstralAgentDesc desc, out AstralHandle out_agent);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_agent_destroy(AstralHandle agent);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_assigned_slot(AstralHandle agent, out uint out_slot);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_release_slot(AstralHandle agent);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_set_system_prompt(AstralHandle agent, AstralSpanU8 system_prompt);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_get_system_prompt_size(AstralHandle agent, out uint out_bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_get_system_prompt(AstralHandle agent, AstralMutSpanU8 out_text, out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_set_summary(AstralHandle agent, AstralSpanU8 summary);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_get_summary_size(AstralHandle agent, out uint out_bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_get_summary(AstralHandle agent, AstralMutSpanU8 out_text, out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_set_memory_context(AstralHandle agent, AstralSpanU8 memory_context);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_set_memory_context_from_results(
            AstralHandle agent,
            ref AstralAgentMemoryContextDesc desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_get_memory_context_size(AstralHandle agent, out uint out_bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_get_memory_context(AstralHandle agent, AstralMutSpanU8 out_text, out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_parse_tool_call(AstralHandle agent, AstralSpanU8 generated_text, ref AstralToolCallResult out_result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_chat_tool_call_result(AstralHandle agent, ref AstralToolCallResult out_result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_message_add(AstralHandle agent, ref AstralAgentMessage message);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_history_clear(AstralHandle agent);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_history_count(AstralHandle agent, out uint out_count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_history_save_size(AstralHandle agent, out uint out_bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_history_save(AstralHandle agent, AstralMutSpanU8 out_bytes, out uint out_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_history_load(AstralHandle agent, AstralSpanU8 bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_chat_enqueue(AstralHandle agent, ref AstralAgentChatDesc desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_chat_cancel(AstralHandle agent);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_chat_stream_read(AstralHandle agent, AstralMutSpanU8 out_buf, uint timeout_ms);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_agent_chat_result(AstralHandle agent, ref AstralAgentChatResult out_result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_request_from_session(AstralHandle session, out AstralRequestRef out_request);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_request_from_conversation(AstralHandle conv, out AstralRequestRef out_request);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_request_from_agent_chat(AstralHandle agent, out AstralRequestRef out_request);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_request_from_embedding(AstralHandle emb, ulong ticket, out AstralRequestRef out_request);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_request_from_memory_search(AstralHandle cursor, out AstralRequestRef out_request);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_request_state(ref AstralRequestRef request, ref AstralRequestStatus out_status);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_request_cancel(ref AstralRequestRef request);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_request_wait(ref AstralRequestRef request, uint timeout_ms, ref AstralRequestStatus out_status);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_save_size(AstralHandle index, out ulong out_bytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_save(AstralHandle index, AstralMutSpanU8 out_bytes, out ulong out_written);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_memory_load(ref AstralMemoryIndexDesc desc, AstralSpanU8 bytes, out AstralHandle out_index);

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

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_embed_enqueue_image(AstralHandle emb, ref AstralImageDesc image, out ulong out_ticket);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_embed_enqueue_audio(AstralHandle emb, ref AstralAudioDesc audio, out ulong out_ticket);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_embed_enqueue_multimodal(
            AstralHandle emb,
            AstralSpanU8 text,
            IntPtr image,
            IntPtr audio,
            out ulong out_ticket);

        /// <summary>
        /// Collect embedding vector for a ticket.
        /// Thread-safety: Safe to call from multiple threads.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_embed_collect(AstralHandle emb, ulong ticket, AstralMutSpanU8 out_vector);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_embed_cancel(AstralHandle emb, ulong ticket);

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
