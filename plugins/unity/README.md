# Astral runtime Unity package

Unity bindings and package layout for Astral Runtime. The package exposes native
runtime initialization, model/session handles, streaming reads, embeddings, and
mock media tests through the public C ABI.

The package keeps native handles behind `IDisposable` wrappers and reads stream
bytes into caller-owned `NativeArray` storage. Jobs and main-thread code can
choose where those bytes become managed text.

## Features

- **NativeArray streaming path**: `ReadStream(NativeArray<byte>)` reads UTF-8 bytes into caller-owned buffers with native backpressure.
- **Burst-friendly job wrappers**: Job structs use blittable fields and `NativeArray` buffers.
- **Explicit P/Invoke ABI**: Declarations use the public C ABI and EditMode tests check key struct layouts.
- **Deterministic ownership**: Native handles are released through `IDisposable`.
- **LoRA adapter ownership**: `AstralAdapter` owns model-scoped adapter handles and sessions attach them between requests.
- **Structured output tools**: `AstralToolset` owns native tool definitions and sessions can bind toolsets or grammar.
- **Continuous batching conversations**: `AstralConversation` wraps model-scoped executor slots for multi-stream generation.
- **Thread ownership**: Native buffers are owned by `NativeArray`. Callers
  choose the job or main-thread boundary that consumes them.
- **Platform package surface**: Desktop and mobile plugin layouts are checked
  in. Editor and player validation is platform-specific.

## Requirements

- Unity 6000.0 or later
- Unity Collections package 1.4.0
- Native plugin for target platform (see Runtime/Plugins/)

## Installation

### Unity package manager

1. Open Unity Package Manager (Window > Package Manager)
2. Click '+' > Add package from git URL
3. Enter: `https://github.com/Cosmin-B/astral-runtime.git?path=/plugins/unity`

### Manual installation

1. Copy `plugins/unity` to your Unity project's `Packages/` directory
2. Rename to `com.astral.runtime`
3. Unity will automatically import the package

## Quick start

### 1. Initialize the runtime

Add `AstralRuntimeInitializer` component to a persistent GameObject:

```csharp
using Astral.Runtime;

public class GameManager : MonoBehaviour
{
    void Awake()
    {
        // Option A: Use initializer component (recommended)
        gameObject.AddComponent<AstralRuntimeInitializer>();

        // Option B: Manual initialization
        var cfg = AstralConfig.Default;
        cfg.useUnityAllocator = true;              // default: true
        cfg.enableLogging = true;                  // default: true
        cfg.maxLogLevel = AstralNative.ASTRAL_LOG_INFO;
        AstralRuntime.Initialize(cfg);
    }

    void OnApplicationQuit()
    {
        AstralRuntime.Shutdown();
    }
}
```

### 2. Load a model

```csharp
using Astral.Runtime;

// Load model with default config
using var model = AstralModel.Load("/path/to/model.gguf");

// Or use mobile-optimized config
using var model = AstralModel.Load("/path/to/model.gguf", AstralModelConfig.Mobile);

// Resolve packaged or downloaded model paths before loading
using var packaged = AstralModel.Load(AstralModelPath.StreamingAssets("Models/model.gguf"));
using var cached = AstralModel.Load(AstralModelPath.PersistentData("Models/model.gguf"), AstralModelConfig.Mobile);

var remoteConfig = AstralModelConfig.Default;
remoteConfig.backendName = "remote";
remoteConfig.remoteApiKey = "";
using var remote = AstralModel.Load("http://127.0.0.1:8080", remoteConfig);
```

### 3. Run streaming inference

```csharp
using Astral.Runtime;

IEnumerator RunInference(AstralModel model, string prompt)
{
    using var session = AstralSession.Create(model, AstralSessionConfig.Default);

    // Feed prompt
    session.Feed(prompt, finalize: true);

    // Start decode (non-blocking)
    session.Decode();

    // Stream tokens
    yield return StartCoroutine(session.StreamCoroutine(
        onToken: (token) => Debug.Log($"Token: {token}"),
        onComplete: () => Debug.Log("Done!")
    ));
}
```

### 4. Stream into a NativeArray

```csharp
using Astral.Runtime;
using Unity.Collections;

IEnumerator RunInferenceNativeArray(AstralModel model, string prompt)
{
    using var session = AstralSession.Create(model);
    const int streamBufferBytes = AstralConversation.DefaultStreamBufferBytes;
    const uint pollTimeoutMs = AstralConversation.NonBlockingTimeoutMs;
    using var buffer = new NativeArray<byte>(streamBufferBytes, Allocator.Persistent);

    // Feed prompt
    session.Feed(prompt, finalize: true);
    session.Decode();

    // Stream tokens into a caller-owned byte buffer.
    while (true)
    {
        int bytesRead = session.ReadStream(buffer, timeoutMs: pollTimeoutMs);

        if (bytesRead > 0)
        {
            // Process UTF-8 bytes directly (no string allocation)
            ProcessTokenBytes(buffer, bytesRead);
        }
        else if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
        {
            yield return null; // Wait one frame
        }
        else
        {
            break; // End of stream
        }
    }
}
```

## API reference

### AstralRuntime

Global runtime initialization and shutdown.

```csharp
// Initialize runtime
AstralRuntime.Initialize(AstralConfig.Default);

// Or: no-throw init for embedded-style, exception-free gameplay loops
if (!AstralRuntime.TryInitialize(AstralConfig.Default, out int err))
{
    Debug.LogError($"Astral init failed: {AstralRuntime.GetErrorString(err)}");
}

// Check if initialized
bool isReady = AstralRuntime.IsInitialized;

// Get error string
string errorMsg = AstralRuntime.GetErrorString(errorCode);

// Shutdown runtime
AstralRuntime.Shutdown();
```

### AstralModel

GGUF model handle with deterministic native ownership.

```csharp
// Load model
using var model = AstralModel.Load("/path/to/model.gguf", AstralModelConfig.Default);

// Check validity
bool isValid = model.IsValid;
```

**Configs**: `Default`, `Mobile`, `HighPerformance`, `Embeddings`

Note: the `Embeddings` preset selects `embeddingsOnly=1` and related defaults.

For local smoke tests, use the shared preset manifest instead of hard-coding
model URLs:

```bash
./tests/model_downloader.sh --preset qwen3-0.6b-q8 --dry-run
./tests/model_downloader.sh --preset qwen3-embed-0.6b-q8 --dry-run
```

The manifest pins filenames, byte sizes, SHA-256 checksums, model type, context
length, and embedding dimensions under `scripts/model_presets.json`.

### AstralSession

Inference session with streaming support.

```csharp
// Create session
using var session = AstralSession.Create(model, AstralSessionConfig.Default);

// Feed prompt
session.Feed("Once upon a time", finalize: true);

// Start decode
session.Decode();

// Read stream (blocking)
const uint readTimeoutMs = AstralConversation.DefaultReadTimeoutMs;
string token = session.ReadStreamAsString(timeoutMs: readTimeoutMs);

// Stream all tokens (convenience)
session.StreamAll(token => Debug.Log(token));

// Get statistics
var stats = session.GetStats();
Debug.Log($"Tokens/sec: {stats.tokensPerSecond:F2}");
```

**Configs**: `Default`, `Greedy`, `Creative`

### AstralConversation

Continuous batching conversation slot.

```csharp
model.ConfigureExecutor(AstralExecutorConfig.Default);

using var conv = AstralConversation.Create(model, AstralConversationConfig.Default);
using var buffer = new NativeArray<byte>(
    AstralConversation.DefaultStreamBufferBytes,
    Allocator.Persistent,
    NativeArrayOptions.UninitializedMemory);

conv.SetSystemPrompt("Answer as an in-game navigator.");
conv.Feed("Where should I go next?", finalize: true);
conv.Decode();

int bytes = conv.ReadStream(buffer, AstralConversation.NonBlockingTimeoutMs);
var stats = conv.GetStats();
```

Conversations support grammar, toolsets, stop sequences, logprob metadata,
media chunks, cancellation, reset, and stats. Prefer
`ReadStream(NativeArray<byte>)` for frame-polled gameplay paths.

### Prompt cache

`AstralPromptCache` owns native token storage and uses `NativeArray<int>` for
direct token copy calls. Use `KeyFromBytes()` for section-aware cache keys and
`GetTokenView()` only when the cache lifetime stays local to the read.

### LoRA adapters

`AstralAdapter.GetInfo()` and `GetPath()` expose native adapter diagnostics.
Attach adapters to sessions between requests with `AddAdapter()` and update
their fixed-slot scale with `SetAdapterScale()`.

### Structured output

`AstralToolset` owns native tool definitions. `AstralToolCall.Parsed`,
`Missing`, and `Malformed` expose native parse status after `ParseCall()`
without scanning generated text in Unity. Use
`AstralAgentConfig.WithToolset(toolset, choiceMode)` to bind a toolset to a
native agent at creation time.

### Tokenization

`AstralModel.CountTokensBatch()` and `TokenizeBatch()` route many UTF-8 spans
through one native tokenizer call. Keep text spans and output offsets in
caller-owned `NativeArray` buffers for ingest paths.

### Chunking

`AstralChunker` plans text and token ranges into caller-owned `NativeArray`
buffers. Use `CountTextBytes()` before materializing selected text when a chunk
will cross back into Unity strings.

### Memory search

`AstralMemoryIndex` owns native vector storage. Cursor searches can be wrapped
with `AstralRequest.FromMemorySearch(cursor)` for polling remaining results
before fetching batches.

### Vision and audio

Media support requires a model projector/encoder GGUF and an Astral build compiled with `ASTRAL_ENABLE_MTMD=ON`. Initialize media once per model before creating sessions or embedders that will consume images or audio:

```csharp
model.InitMediaFromPath("/path/to/media.gguf");
```

Feed media into a session prompt. The backing `NativeArray` must stay alive for the duration of the feed call. Astral copies or consumes the data before the method returns.

```csharp
// Image (RGB8)
session.FeedImage(pixels, width: 224, height: 224, AstralNative.AstralImageFormat.RGB8);

// Audio (PCM f32 or i16)
session.FeedAudio(audioF32, channels: 1, sampleRate: 16000);
```

### Multimodal embeddings

Load embedding models with `embeddingsOnly = true`, initialize media first when image/audio input is used, and size the output vector from `embedder.Dimension`.

```csharp
using var embedder = AstralEmbedder.Create(model);
using var outVec = new NativeArray<float>((int)embedder.Dimension, Allocator.Temp);

// Image-only embedding
ulong ticket = embedder.EnqueueImage(pixels, 224, 224, AstralNative.AstralImageFormat.RGB8);
embedder.Collect(ticket, outVec);

// Multimodal embedding (text + image)
var imageDesc = new AstralNative.AstralImageDesc
{
    format = AstralNative.AstralImageFormat.RGB8,
    width = 224,
    height = 224,
    pixels = AstralNative.AstralSpanU8.FromNativeArray(pixels)
};
ulong mmTicket = embedder.EnqueueMultimodal("describe", ref imageDesc);
embedder.Collect(mmTicket, outVec);
```

`Cancel(ticket)` releases queued embedding work that no longer needs to be
collected.

### Request status

`AstralRequest` wraps the native request lifecycle for sessions, conversations,
agent chat, embedding tickets, and memory search cursors. The wrapper returns
the same native status fields Unity jobs or main-thread dispatchers need without
owning prompt assembly, vector storage, or stream buffers.

```csharp
ulong ticket = embedder.Enqueue(textBytes);
var request = AstralRequest.FromEmbedding(embedder, ticket);

if (AstralRequest.TryGetStatus(request, out var status, out int err))
{
    bool queued = AstralRequest.IsQueued(status);
    bool ticketed = AstralRequest.HasTicket(status);
    string label = AstralRequest.StateName(status.state);
}
```

## Configuration

### Runtime configuration

```csharp
var config = new AstralConfig
{
    reserveBytes = 2UL << 30,  // 2GB virtual memory
    threadCount = 0,           // Auto-detect (physical cores - 1)
    enableHugePages = false    // Requires OS support
};

AstralRuntime.Initialize(config);
```

**Presets**: `Default`, `Mobile`, `HighPerformance`

### Model configuration

```csharp
var config = new AstralModelConfig
{
    gpuLayers = 0,          // Use a positive value with a GPU-enabled backend
    contextSize = 2048,     // Context window in tokens
    batchSize = 512,        // Prompt processing batch size
    threads = 0,            // Auto-detect
    embeddingsOnly = false  // Set true for embeddings-only models (enables `astral_embed_*` fast paths)
};

var model = AstralModel.Load("/path/to/model.gguf", config);
```

**Presets**: `Default`, `Mobile`, `HighPerformance`, `Embeddings`

Use `AstralModelPath.StreamingAssets(...)` for files staged with the player and
`AstralModelPath.PersistentData(...)` for first-run downloads or user-managed
cache files. Absolute paths and `AstralModelPath.Raw(...)` are passed through
unchanged.

### Mobile model setup

Use the smallest preset that proves the target workflow before raising context
length or model size:

| Use | Preset | Notes |
|-----|--------|-------|
| Text smoke | `gemma3-270m-q4km` | Smallest documented text preset for player-load checks. |
| Text quality pass | `qwen3-0.6b-q8` | Use when device memory and first-token latency are acceptable. |
| Embeddings | `qwen3-embed-0.6b-q8` | Pair with `AstralModelConfig.Embeddings`. |

Ship packaged models under `StreamingAssets/Models` and resolve them with
`AstralModelPath.StreamingAssets(...)`. For first-run downloads, write the GGUF
to `Application.persistentDataPath`, verify it with the shared preset manifest
or downloader tooling, then resolve it with `AstralModelPath.PersistentData(...)`.
Keep partial downloads and GGUF files out of source control.

Start mobile players with `AstralConfig.Mobile` and `AstralModelConfig.Mobile`.
Measure target devices before increasing `contextSize`, `batchSize`, or worker
thread count. The wrapper does not pin Unity threads or select big/little cores.
those settings must be validated on the actual device runner.

### Session configuration

```csharp
var config = new AstralSessionConfig
{
    maxTokens = 512,        // Maximum tokens to generate
    temperature = 0.7f,     // 0.0 = greedy, 1.0 = diverse
    topK = 40,              // Top-K sampling
    topP = 0.9f,            // Top-P (nucleus) sampling
    streamEnabled = true    // Enable token streaming
};

var session = AstralSession.Create(model, config);
```

**Presets**: `Default`, `Greedy`, `Creative`

## Platform support

| Platform | Arch | Status | Notes |
|----------|------|--------|-------|
| Windows | x86_64 | Package surface | Native player library required |
| Linux | x86_64 | Package surface | Native preload path included |
| macOS | ARM64 | Package surface | Native player library required |
| Android | ARM64 | Package surface | Device library and import settings required |
| iOS | ARM64 | Static-link surface | Xcode static library integration required |
| WebGL | WASM | Unsupported | No maintained native runtime target |

## Troubleshooting

### DllNotFoundException

**Cause**: Native library not found or incorrect platform.

**Fix**:
1. Ensure native library is in `Runtime/Plugins/{platform}/`
2. Check Unity import settings (Platform, CPU, Load on startup)
3. Verify library architecture matches Unity build target

### EntryPointNotFoundException

**Cause**: P/Invoke signature mismatch.

**Fix**:
1. Check function name, calling convention, and parameter types
2. Verify native library exports match C# declarations
3. Use `nm -D libastral_rt.so` (Linux) or `dumpbin /EXPORTS astral_rt.dll` (Windows)

### IL2CPP crashes

**Cause**: Incorrect P/Invoke marshaling or unsafe code.

**Fix**:
1. Ensure all P/Invoke uses `CallingConvention.Cdecl`
2. Verify struct layouts with `StructLayout(LayoutKind.Sequential)`
3. Check `allowUnsafeCode: true` in asmdef

### GC allocations during streaming

**Cause**: Using managed strings in hot paths.

**Fix**:
1. Use `ReadStream(NativeArray<byte>)` instead of `ReadStreamAsString()`
2. Process UTF-8 bytes directly without string conversion
3. Validate with Unity Profiler (Deep Profile enabled)

## Examples

Install the package samples from Package Manager or open them under `Samples~`:

| Workflow | Sample | Focus |
| --- | --- | --- |
| Streaming chat | [StreamingChat](Samples~/StreamingChat/README.md) | Frame-polled UTF-8 streaming, cancellation, and stats |
| Multiple conversations | [MultipleConversations](Samples~/MultipleConversations/README.md) | One executor serving independent conversation slots |
| Stateful NPC | [StatefulNpc](Samples~/StatefulNpc/README.md) | Agent history, summary, memory context, and tool calls |
| Local knowledge | [LocalKnowledge](Samples~/LocalKnowledge/README.md) | Chunking, embeddings, native indexing, search, and persistence |
| Character variants | [CharacterVariants](Samples~/CharacterVariants/README.md) | Prompt caches, structured output, stop sequences, and adapters |
| Multimodal input | [MultimodalInput](Samples~/MultimodalInput/README.md) | Texture, audio, and multimodal embedding requests |

## Building native plugins

See `Runtime/Plugins/README.md` for build instructions for each platform.

## Plugin tests

The package includes small, focused Unity EditMode tests under `Tests/Editor/`:
- ABI layout assertions for key C ABI structs (`AstralSpanU8`, `AstralModelDesc`, `AstralSessionDesc`, …).
- Mock-backend smoke (init → model → session → decode → stream → reset → repeat).
- Mock media feed and multimodal embedding smoke with no GGUF required.

Release and CI runs must provide a native `astral_rt` library for the Editor.
`scripts/run_unity_ci_tests.sh` fails before Unity starts when the platform
binary is missing or empty, then validates the EditMode XML result after Unity
exits.

For Linux container validation, use the GameCI wrapper:

```bash
./scripts/run_unity_gameci_tests.sh
```

The wrapper follows the current GameCI v4 Docker documentation, reads the Unity
version from the CI project, defaults to
`unityci/editor:ubuntu-6000.0.57f1-base-3.2.2`, builds the native Unity plugin
on the host, then runs the same EditMode ABI lane inside the container. License
environment variables are forwarded by name only when already set. License files
and activation responses are not read or written by the wrapper.

## License

See [LICENSE](../../LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

## Support

- [Report a bug in GitHub Issues](https://github.com/Cosmin-B/astral-runtime/issues)
- Documentation: [Astral documentation](../../docs/README.md)

## Changelog

See [CHANGELOG.md](../../CHANGELOG.md) for version history.
