# Astral Runtime - Unity Plugin

Unity bindings and package layout for Astral Runtime. The package exposes native
runtime initialization, model/session handles, streaming reads, embeddings, and
mock media tests through the public C ABI.

## Features

- **NativeArray streaming path**: `ReadStream(NativeArray<byte>)` reads UTF-8 bytes into caller-owned buffers.
- **Burst-friendly job wrappers**: Job structs use blittable fields and `NativeArray` buffers.
- **Explicit P/Invoke ABI**: Declarations use the public C ABI and EditMode tests check key struct layouts.
- **Streaming Support**: Real-time token streaming with backpressure control
- **Deterministic ownership**: Native handles are released through `IDisposable`.
- **Thread ownership**: Native buffers are owned by `NativeArray`; session concurrency still needs real Unity runner evidence.
- **Portable**: armv7, armv8, arm64, x86-64 support

## Requirements

- Unity 2021.3 or later
- Unity Collections package 1.4.0
- Native plugin for target platform (see Runtime/Plugins/)

## Installation

### Unity Package Manager (Recommended)

1. Open Unity Package Manager (Window > Package Manager)
2. Click '+' > Add package from git URL
3. Enter: `https://github.com/astral-runtime/astral.git?path=/plugins/unity`

### Manual Installation

1. Copy `plugins/unity` to your Unity project's `Packages/` directory
2. Rename to `com.astral.runtime`
3. Unity will automatically import the package

## Quick Start

### 1. Initialize Runtime

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

### 2. Load Model

```csharp
using Astral.Runtime;

// Load model with default config
using var model = AstralModel.Load("/path/to/model.gguf");

// Or use mobile-optimized config
using var model = AstralModel.Load("/path/to/model.gguf", AstralModelConfig.Mobile);
```

### 3. Run Inference (Streaming)

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

### 4. NativeArray Streaming

```csharp
using Astral.Runtime;
using Unity.Collections;

IEnumerator RunInferenceNativeArray(AstralModel model, string prompt)
{
    using var session = AstralSession.Create(model);
    using var buffer = new NativeArray<byte>(4096, Allocator.Persistent);

    // Feed prompt
    session.Feed(prompt, finalize: true);
    session.Decode();

    // Stream tokens into a caller-owned byte buffer.
    while (true)
    {
        int bytesRead = session.ReadStream(buffer, timeoutMs: 0);

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

## API Reference

### AstralRuntime

Global runtime initialization and shutdown.

```csharp
// Initialize runtime
AstralRuntime.Initialize(AstralConfig.Default);

// Or: no-throw init (recommended for embedded-style, exception-free gameplay loops)
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

Note: embeddings are supported; `Embeddings` preset selects `embeddingsOnly=1` and related defaults.

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
string token = session.ReadStreamAsString(timeoutMs: 100);

// Stream all tokens (convenience)
session.StreamAll(token => Debug.Log(token));

// Get statistics
var stats = session.GetStats();
Debug.Log($"Tokens/sec: {stats.tokensPerSecond:F2}");
```

**Configs**: `Default`, `Greedy`, `Creative`

### Vision / Audio (Media)

Media support requires a model projector/encoder GGUF and an Astral build compiled with `ASTRAL_ENABLE_MTMD=ON`. Initialize media once per model before creating sessions or embedders that will consume images or audio:

```csharp
model.InitMediaFromPath("/path/to/media.gguf");
```

Feed media into a session prompt. The backing `NativeArray` must stay alive for the duration of the feed call; Astral copies or consumes the data before the method returns.

```csharp
// Image (RGB8)
session.FeedImage(pixels, width: 224, height: 224, AstralNative.AstralImageFormat.RGB8);

// Audio (PCM f32 or i16)
session.FeedAudio(audioF32, channels: 1, sampleRate: 16000);
```

### Multimodal Embeddings

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

## Configuration

### Runtime Configuration

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

### Model Configuration

```csharp
var config = new AstralModelConfig
{
    gpuLayers = 0,          // CPU-only (v0.1); 32+ for GPU (v0.2)
    contextSize = 2048,     // Context window in tokens
    batchSize = 512,        // Prompt processing batch size
    threads = 0,            // Auto-detect
    embeddingsOnly = false  // Set true for embeddings-only models (enables `astral_embed_*` fast paths)
};

var model = AstralModel.Load("/path/to/model.gguf", config);
```

**Presets**: `Default`, `Mobile`, `HighPerformance`, `Embeddings`

### Session Configuration

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

## Platform Support

| Platform | Arch | Status | Notes |
|----------|------|--------|-------|
| Windows | x86_64 | Tested | Visual Studio 2022 |
| Linux | x86_64 | Tested | GCC 11+ |
| macOS | ARM64 | Tested | Apple Silicon (M1/M2) |
| Android | ARM64 | Planned (v0.1.1) | API Level 21+ |
| iOS | ARM64 | Tested | iOS 12.0+ |
| WebGL | WASM | Planned | v0.3 |

## Performance

Measured on Ryzen 7 5800X (desktop) and iPhone 13 Pro (mobile):

| Model | Platform | Tokens/sec | First Token (ms) | Memory (MB) |
|-------|----------|------------|------------------|-------------|
| TinyLlama-1.1B-Q4_0 | Desktop | 45-60 | 150-200 | 800 |
| TinyLlama-1.1B-Q4_0 | Mobile | 12-18 | 400-600 | 600 |
| LLaMA-2-7B-Q4_K_M | Desktop | 18-25 | 300-400 | 4500 |

**Note**: Performance varies based on model size, quantization, and hardware.

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

### IL2CPP Crashes

**Cause**: Incorrect P/Invoke marshaling or unsafe code.

**Fix**:
1. Ensure all P/Invoke uses `CallingConvention.Cdecl`
2. Verify struct layouts with `StructLayout(LayoutKind.Sequential)`
3. Check `allowUnsafeCode: true` in asmdef

### GC Allocations During Streaming

**Cause**: Using managed strings in hot paths.

**Fix**:
1. Use `ReadStream(NativeArray<byte>)` instead of `ReadStreamAsString()`
2. Process UTF-8 bytes directly without string conversion
3. Validate with Unity Profiler (Deep Profile enabled)

## Examples

See `Runtime/AstralExample.cs` for comprehensive usage examples:

- **Basic Inference**: Simple blocking inference
- **Streaming Inference**: Real-time token streaming with coroutines
- **NativeArray Streaming**: Read token bytes into caller-owned buffers.

## Building Native Plugins

See `Runtime/Plugins/README.md` for build instructions for each platform.

## Plugin Tests

The package includes small, focused Unity EditMode tests under `Tests/Editor/`:
- ABI layout assertions for key C ABI structs (`AstralSpanU8`, `AstralModelDesc`, `AstralSessionDesc`, …).
- Mock-backend smoke (init → model → session → decode → stream → reset → repeat).
- Mock media feed + multimodal embedding smoke (mock backend; no GGUF required).

Release and CI runs must provide a native `astral_rt` library for the Editor.
`scripts/run_unity_ci_tests.sh` fails before Unity starts when the platform
binary is missing or empty, then validates the EditMode XML result after Unity
exits.

## License

See [LICENSE](../../LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

## Support

- GitHub Issues: https://github.com/astral-runtime/astral/issues
- Documentation: https://astral-runtime.github.io/astral

## Changelog

See [CHANGELOG.md](../../CHANGELOG.md) for version history.
