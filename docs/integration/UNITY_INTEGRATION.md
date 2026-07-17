# Unity Integration Specification

Implementation note: this document is a design/spec. For maintained Unity API
usage, prefer `plugins/unity/README.md` and `plugins/unity/Runtime/*`. Release
sign-off still requires a real Unity Editor runner with native binaries present.

## Goals

1. **Caller-owned stream buffers**: Read token bytes into `NativeArray<byte>` buffers without converting every token to a managed string.
2. **Burst-friendly job wrappers**: Keep job data blittable and pass native handles explicitly.
3. **IL2CPP constraints**: Avoid reflection and dynamic code generation in native-handle wrappers.
4. **Platform Evidence**: Windows, macOS, Linux, Android, and iOS need import
   and player evidence before support claims.
5. **Native Allocator Passthrough**: Use Unity-owned allocation paths for
   session-local buffers where the C ABI exposes that ownership.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Unity (C#)                                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ AstralSession (MonoBehaviour or pure C# wrapper)        │ │
│ │   ├─ NativeArray<byte> prompt buffer (persistent)      │ │
│ │   ├─ NativeArray<byte> token buffer (persistent)       │ │
│ │   └─ Job/Thread: Poll astral_stream_read()             │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ AstralNative (static class, P/Invoke)                   │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │ P/Invoke
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Native (C/C++)                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ astral_rt.{dll,so,dylib}                                │ │
│ │   ├─ astral_init()                                      │ │
│ │   ├─ astral_model_load()                                │ │
│ │   ├─ astral_session_create()                            │ │
│ │   ├─ astral_session_feed()                              │ │
│ │   ├─ astral_session_decode()                            │ │
│ │   └─ astral_stream_read()                               │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Plugin Structure

```
astral/plugins/unity/
├── Runtime/
│   ├── AstralNative.cs        # P/Invoke declarations
│   ├── AstralNativeArray.cs   # NativeArray ↔ span helpers
│   ├── AstralRuntime.cs       # Runtime initialization/shutdown + config
│   ├── AstralModel.cs         # Model wrapper
│   ├── AstralSession.cs       # Session wrapper + streaming
│   ├── AstralAllocator.cs     # Unity allocator bridge
│   ├── AstralLogging.cs       # Logging bridge
│   ├── AstralJobSystem.cs     # Jobs integration
│   └── Astral.Runtime.asmdef  # Assembly definition
├── README.md
├── ALLOCATOR_INTEGRATION.md
└── package.json
```

## P/Invoke Layer (`AstralNative.cs`)

```csharp
using System;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral
{
    /// <summary>
    /// Low-level P/Invoke bindings to astral_rt native library.
    /// Do not use directly; use AstralSession instead.
    /// </summary>
    public static class AstralNative
    {
        #if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            private const string DLL_NAME = "astral_rt";
        #elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX || UNITY_IOS
            private const string DLL_NAME = "libastral_rt";
        #elif UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX || UNITY_ANDROID
            private const string DLL_NAME = "libastral_rt";
        #else
            private const string DLL_NAME = "astral_rt";
        #endif

        // ====== Common Types ======

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralSpanU8
        {
            public IntPtr data;
            public uint len;
#if UNITY_64 || UNITY_EDITOR_64
            public uint _padding; // 64-bit explicit padding (16B total)
#endif

            public unsafe AstralSpanU8(byte* ptr, int length)
            {
                data = (IntPtr)ptr;
                len = (uint)length;
            }

            public static unsafe AstralSpanU8 FromNativeArray<T>(Unity.Collections.NativeArray<T> array) where T : unmanaged
            {
                return new AstralSpanU8(
                    (byte*)array.GetUnsafeReadOnlyPtr(),
                    array.Length * UnsafeUtility.SizeOf<T>()
                );
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralMutSpanU8
        {
            public IntPtr data;
            public uint len;
#if UNITY_64 || UNITY_EDITOR_64
            public uint _padding; // 64-bit explicit padding (16B total)
#endif

            public unsafe AstralMutSpanU8(byte* ptr, int length)
            {
                data = (IntPtr)ptr;
                len = (uint)length;
            }

            public static unsafe AstralMutSpanU8 FromNativeArray<T>(Unity.Collections.NativeArray<T> array) where T : unmanaged
            {
                return new AstralMutSpanU8(
                    (byte*)array.GetUnsafePtr(),
                    array.Length * UnsafeUtility.SizeOf<T>()
                );
            }
        }

        // 64-bit tagged handle (0 = invalid)
        [StructLayout(LayoutKind.Sequential)]
        public struct AstralHandle
        {
            public ulong value;
        }

        public enum AstralErr : int
        {
            OK = 0,
            Invalid = -1,
            NoMem = -2,
            Busy = -3,
            Timeout = -4,
            State = -5,
            Backend = -6,
            Canceled = -7,
            Unsupported = -8,
        }

        // ====== Allocator ======

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr AllocFn(IntPtr user, UIntPtr size, UIntPtr align);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void FreeFn(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align);

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralAllocator
        {
            public IntPtr alloc; // AllocFn
            public IntPtr free;  // FreeFn
            public IntPtr user;
        }

        // ====== Logging ======

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LogFn(IntPtr user, int level, AstralSpanU8 msg);

        // ====== Init ======

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralInit
        {
            public AstralAllocator sys_alloc;
            public IntPtr log_cb;    // LogFn
            public IntPtr log_user;
            public ulong reserve_bytes;
            public uint thread_count;
            public uint numa_node;
            public byte enable_hugepages;
        }

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_init(ref AstralInit cfg);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_shutdown();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_version(out uint major, out uint minor, out uint patch);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr astral_error_string(AstralErr err);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr astral_last_error();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_clear_last_error();

        // ====== Model ======

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralModelDesc
        {
            public AstralSpanU8 model_path;
            public AstralSpanU8 backend_name; // optional override ("cpu", "mock", ...)
            public uint gpu_layers;
            public uint n_ctx;
            public uint n_batch;
            public uint n_threads;
            public byte embeddings_only;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralModelInfo
        {
            public uint vocab_size;
            public uint ctx_size;
            public int token_bos;
            public int token_eos;
        }

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_model_load(ref AstralModelDesc desc, out AstralHandle out_model);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_model_release(AstralHandle model);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_model_info(AstralHandle model, out AstralModelInfo out_info);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_model_embedding_dim(AstralHandle model, out uint out_dim);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe AstralErr astral_tokenize(
            AstralHandle model,
            AstralSpanU8 text,
            int* out_tokens,
            uint max_tokens,
            byte add_special,
            byte parse_special,
            out uint out_count);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_tokenize_count(
            AstralHandle model,
            AstralSpanU8 text,
            byte add_special,
            byte parse_special,
            out uint out_count);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe AstralErr astral_detokenize(
            AstralHandle model,
            int* tokens,
            uint count,
            AstralMutSpanU8 out_text,
            out uint out_len);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe AstralErr astral_detokenize_count(
            AstralHandle model,
            int* tokens,
            uint count,
            out uint out_len);

        // ====== Session ======

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralSessionDesc
        {
            public AstralHandle model;
            public uint max_tokens;
            public float temperature;
            public uint top_k;
            public float top_p;
            public byte stream_enabled;
            public uint seed; // 0 = auto
        }

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_session_create(ref AstralSessionDesc desc, out AstralHandle out_session);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_session_destroy(AstralHandle session);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_session_feed(AstralHandle session, AstralSpanU8 prompt_chunk, byte finalize);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_session_decode(AstralHandle session);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_session_cancel(AstralHandle session);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_session_state(AstralHandle session, out uint out_state);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_session_wait(AstralHandle session, uint timeout_ms);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_session_reset(AstralHandle session, ref AstralSessionDesc desc);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int astral_stream_read(AstralHandle session, AstralMutSpanU8 out_buf, uint timeout_ms);

        // ====== Embeddings ======
        // Load the model with `embeddings_only = 1` and size `out_vector` to `out_dim * sizeof(float)`.

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_embed_create(AstralHandle model, out AstralHandle out_embedder);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void astral_embed_destroy(AstralHandle emb);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_embed_enqueue(AstralHandle emb, AstralSpanU8 text, out ulong out_ticket);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_embed_collect(AstralHandle emb, ulong ticket, AstralMutSpanU8 out_vector);

        // ====== Stats ======

        [StructLayout(LayoutKind.Sequential)]
        public struct AstralStats
        {
            public double t_init_ms;
            public double t_first_token_ms;
            public double tok_per_s;
            public ulong bytes_committed;
            public ulong bytes_reserved;
        }

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern AstralErr astral_session_stats(AstralHandle session, out AstralStats out_stats);
    }
}
```

## Stream and lifecycle semantics

- `astral_stream_read(session, out_buf, timeout_ms)` returns:
  - `> 0`: UTF-8 bytes for one token piece (written into `out_buf`)
  - `0`: end-of-stream (session is terminal and no buffered data remains)
  - `< 0`: error code (`ASTRAL_E_TIMEOUT` means “no data yet”)
- Always call `astral_session_wait(session, timeout_ms)` to determine the final outcome:
  - `ASTRAL_OK`: completed successfully
  - `ASTRAL_E_CANCELED`: completed due to cancellation
  - other `< 0`: decode failed
- Reset/reuse: call `astral_session_reset(session, &desc)` only when the session is not decoding (cancel + wait first).

## High-Level C# Wrapper (`AstralSession.cs`)

```csharp
using System;
using System.Text;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral
{
    /// <summary>
    /// High-level C# wrapper for Astral inference session.
    /// Manages native resources via IDisposable pattern.
    /// </summary>
    public class AstralSession : IDisposable
    {
        private AstralNative.AstralHandle sessionHandle;
        private NativeArray<byte> tokenBuffer;
        private bool disposed;

        public bool IsValid => sessionHandle.value != 0;

        public AstralSession(
            AstralNative.AstralHandle modelHandle,
            uint maxTokens = 512,
            float temperature = 0.7f,
            uint topK = 40,
            float topP = 0.95f,
            bool streamEnabled = true)
        {
            var desc = new AstralNative.AstralSessionDesc
            {
                model = modelHandle,
                max_tokens = maxTokens,
                temperature = temperature,
                top_k = topK,
                top_p = topP,
                stream_enabled = (byte)(streamEnabled ? 1 : 0)
            };

            var err = AstralNative.astral_session_create(ref desc, out sessionHandle);
            if (err != AstralNative.AstralErr.OK)
            {
                throw new AstralException($"Failed to create session: {err}");
            }

            // Allocate the token buffer before polling the stream.
            tokenBuffer = new NativeArray<byte>(4096, Allocator.Persistent);
        }

        /// <summary>
        /// Feed a prompt chunk. Call with finalize=true on the last chunk.
        /// </summary>
        public unsafe void Feed(string prompt, bool finalize = true)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(prompt);
            fixed (byte* ptr = utf8)
            {
                var span = new AstralNative.AstralSpanU8(ptr, utf8.Length);
                var err = AstralNative.astral_session_feed(sessionHandle, span, (byte)(finalize ? 1 : 0));
                if (err != AstralNative.AstralErr.OK)
                {
                    throw new AstralException($"Feed failed: {err}");
                }
            }
        }

        /// <summary>
        /// Feed a prompt from a caller-owned NativeArray.
        /// </summary>
        public unsafe void FeedNative(NativeArray<byte> utf8Prompt, bool finalize = true)
        {
            var span = AstralNative.AstralSpanU8.FromNativeArray(utf8Prompt);
            var err = AstralNative.astral_session_feed(sessionHandle, span, (byte)(finalize ? 1 : 0));
            if (err != AstralNative.AstralErr.OK)
            {
                throw new AstralException($"Feed failed: {err}");
            }
        }

        /// <summary>
        /// Start decoding (non-blocking).
        /// </summary>
        public void Decode()
        {
            var err = AstralNative.astral_session_decode(sessionHandle);
            if (err != AstralNative.AstralErr.OK)
            {
                throw new AstralException($"Decode failed: {err}");
            }
        }

        /// <summary>
        /// Read tokens from stream. Returns number of bytes read (0 if none available).
        /// Timeout in milliseconds.
        /// </summary>
        public unsafe int StreamRead(NativeArray<byte> outBuffer, uint timeoutMs = 100)
        {
            var span = AstralNative.AstralMutSpanU8.FromNativeArray(outBuffer);
            int bytesRead = AstralNative.astral_stream_read(sessionHandle, span, timeoutMs);
            if (bytesRead < 0)
            {
                var err = (AstralNative.AstralErr)bytesRead;
                if (err == AstralNative.AstralErr.Timeout)
                {
                    return 0; // No data yet
                }
                throw new AstralException($"StreamRead failed: {err}");
            }
            return bytesRead;
        }

        /// <summary>
        /// Helper: Read tokens as UTF-8 string (allocates managed string).
        /// </summary>
        public string StreamReadString(uint timeoutMs = 100)
        {
            int bytesRead = StreamRead(tokenBuffer, timeoutMs);
            if (bytesRead == 0) return string.Empty;

            unsafe
            {
                return Encoding.UTF8.GetString((byte*)tokenBuffer.GetUnsafeReadOnlyPtr(), bytesRead);
            }
        }

        /// <summary>
        /// Get session statistics.
        /// </summary>
        public AstralNative.AstralStats GetStats()
        {
            var err = AstralNative.astral_session_stats(sessionHandle, out var stats);
            if (err != AstralNative.AstralErr.OK)
            {
                throw new AstralException($"GetStats failed: {err}");
            }
            return stats;
        }

        public void Dispose()
        {
            if (disposed) return;

            if (sessionHandle.value != 0)
            {
                AstralNative.astral_session_destroy(sessionHandle);
                sessionHandle.value = 0;
            }

            if (tokenBuffer.IsCreated)
            {
                tokenBuffer.Dispose();
            }

            disposed = true;
        }
    }

    public class AstralException : Exception
    {
        public AstralException(string message) : base(message) { }
    }
}
```

## Unity Allocator Bridge

`AstralRuntime.Initialize()` uses `AstralAllocatorBridge.CreateUnityAllocator()`
when `AstralConfig.useUnityAllocator` is enabled. The bridge passes
`UnsafeUtility.Malloc()` / `UnsafeUtility.Free()` callbacks into
`AstralInit.sys_alloc`, roots the delegates for native lifetime safety, and
keeps simple allocation counters for debugging.

The allocator bridge is a runtime ownership setting. It does not change the
caller-owned span model for prompt bytes, stream buffers, embedding vectors, or
memory-search results. See `plugins/unity/ALLOCATOR_INTEGRATION.md` for the
current contract and validation notes.

## Workflow Samples

The package ships six focused samples. Install them through Package Manager or
open their source directly:

| Workflow | Sample |
| --- | --- |
| Streaming chat | [`StreamingChat`](https://github.com/Cosmin-B/astral-runtime/blob/main/plugins/unity/Samples~/StreamingChat/README.md) |
| Multiple conversations | [`MultipleConversations`](https://github.com/Cosmin-B/astral-runtime/blob/main/plugins/unity/Samples~/MultipleConversations/README.md) |
| Stateful NPC | [`StatefulNpc`](https://github.com/Cosmin-B/astral-runtime/blob/main/plugins/unity/Samples~/StatefulNpc/README.md) |
| Local knowledge | [`LocalKnowledge`](https://github.com/Cosmin-B/astral-runtime/blob/main/plugins/unity/Samples~/LocalKnowledge/README.md) |
| Character variants | [`CharacterVariants`](https://github.com/Cosmin-B/astral-runtime/blob/main/plugins/unity/Samples~/CharacterVariants/README.md) |
| Multimodal input | [`MultimodalInput`](https://github.com/Cosmin-B/astral-runtime/blob/main/plugins/unity/Samples~/MultimodalInput/README.md) |

## Minimal Session Example

```csharp
using UnityEngine;
using Astral.Runtime;
using Unity.Collections;

public class BasicChatExample : MonoBehaviour
{
    private AstralNative.AstralHandle modelHandle;
    private AstralSession session;

    void Start()
    {
        var config = AstralConfig.Default;
        config.useUnityAllocator = true;

        if (!AstralRuntime.Initialize(config, out var err))
        {
            Debug.LogError(AstralRuntime.GetErrorString(err));
            return;
        }

        string modelPath = Application.streamingAssetsPath + "/models/llama-7b-q4.gguf";
        byte[] pathBytes = System.Text.Encoding.UTF8.GetBytes(modelPath);

        unsafe
        {
            fixed (byte* pathPtr = pathBytes)
            {
                var modelDesc = new AstralNative.AstralModelDesc
                {
                    model_path = new AstralNative.AstralSpanU8(pathPtr, pathBytes.Length),
                    gpu_layers = 0,
                    n_ctx = 2048,
                    n_batch = 512,
                    n_threads = 0,
                    embeddings_only = 0
                };

                err = AstralNative.astral_model_load(ref modelDesc, out modelHandle);
                if (err != AstralNative.AstralErr.OK)
                {
                    Debug.LogError($"Model load failed: {err}");
                    return;
                }
            }
        }

        session = new AstralSession(modelHandle, maxTokens: 512, temperature: 0.7f);

        session.Feed("Once upon a time", finalize: true);
        session.Decode();

        Debug.Log("Inference started");
    }

    void Update()
    {
        if (session == null || !session.IsValid) return;

        // Convenience path: converts token bytes to a managed string.
        string token = session.StreamReadString(timeoutMs: 10);
        if (!string.IsNullOrEmpty(token))
        {
            Debug.Log(token);
        }
    }

    void OnDestroy()
    {
        session?.Dispose();
        if (modelHandle.value != 0)
        {
            AstralNative.astral_model_release(modelHandle);
        }
        AstralRuntime.Shutdown();
    }
}
```

## Burst-Compatible Job (Advanced)

```csharp
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

[BurstCompile]
public struct StreamReadJob : IJob
{
    public AstralNative.AstralHandle sessionHandle;
    public NativeArray<byte> outBuffer;
    public uint timeoutMs;

    [WriteOnly]
    public NativeArray<int> bytesRead;

    public unsafe void Execute()
    {
        var span = AstralNative.AstralMutSpanU8.FromNativeArray(outBuffer);
        int result = AstralNative.astral_stream_read(sessionHandle, span, timeoutMs);
        bytesRead[0] = result;
    }
}

// Usage
var job = new StreamReadJob
{
    sessionHandle = session.Handle, // Expose handle in AstralSession
    outBuffer = tokenBuffer,
    timeoutMs = 10,
    bytesRead = new NativeArray<int>(1, Allocator.TempJob)
};

var handle = job.Schedule();
handle.Complete();
int bytes = job.bytesRead[0];
job.bytesRead.Dispose();
```

## Platform-Specific Notes

### Android

- **Scripting backend**: Use IL2CPP for ARM64 player validation.
- **Native library**: Stage `libastral_rt.so` under `Plugins/Android/arm64-v8a/`.
- **Threading**: Keep Unity API calls on Unity-owned threads.
- **Memory**: Tune reserve size on the target device.
- **Models**: Start with `gemma3-270m-q4km` for text smoke checks and
  `qwen3-embed-0.6b-q8` for embedding checks. Use
  `AstralModelPath.StreamingAssets(...)` for packaged GGUF files and
  `AstralModelPath.PersistentData(...)` for first-run downloads after checksum
  validation.

### iOS

- **Scripting backend**: Use IL2CPP for device validation.
- **Native library**: Stage the iOS static library through the Unity plugin
  layout.
- **Memory**: Tune model size, context length, and reserve size on the target
  device.
- **Models**: Keep downloaded GGUF files under `Application.persistentDataPath`
  and keep packaged models under `StreamingAssets/Models`. Validate the selected
  preset on device before increasing context length or batch size.
- **Entitlements**: Inference does not require Astral-specific entitlements.

### WebGL

- **Not Supported**: WebAssembly lacks threading (SharedArrayBuffer optional); defer to server-side inference

## Performance Best Practices

1. **Pre-allocate Buffers**: Create all `NativeArray<byte>` at init, not per-frame
2. **Prefer `StreamRead()` in polling loops**: Use caller-owned `NativeArray` buffers when managed strings are not needed.
3. **Burst Compile Jobs**: Use `StreamReadJob` where job scheduling fits the caller's frame model.
4. **Profile Allocations**: Use Unity Profiler Memory views to check whether token polling allocates.
5. **Threading**: Keep Unity object access on Unity-owned threads and measure
   native worker settings per device.

## Troubleshooting

### DLL Not Found

- **Windows**: Ensure `astral_rt.dll` is in `Plugins/x86_64/`
- **macOS**: Disable Gatekeeper for unsigned libs: `sudo spctl --master-disable`
- **Linux**: Check `libastral_rt.so` has execute permissions

### GC Allocations

- Use `StreamRead()` with persistent `NativeArray`, not `StreamReadString()`
- Avoid `string.Format()` in hot path; use `StringBuilder` or `UnsafeUtility.WriteArrayElement`

### Player Build Failures

- Verify the native plugin is staged for the selected Unity target and CPU
  architecture.
- Check that model files are staged or downloaded into the runtime path the
  wrapper resolves.
- Run the native benchmark suite on the target hardware before changing context
  length, batch size, or worker-thread settings.
