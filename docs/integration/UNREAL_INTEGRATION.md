# Unreal Engine 5 Integration Specification

## Goals

1. **Bytes-first streaming**: Avoid per-chunk `FString` allocations in hot paths; stream UTF-8 via `TConstArrayView<uint8>` / `TArray<uint8>`
2. **FMemory Integration**: Bridge UE's memory allocator to Astral's `AstralAllocator`
3. **Thread Safety**: Integrate with UE task graph or use library's own workers
4. **Platform Coverage**: Windows, macOS, Linux, Android, iOS (consoles explicitly out of scope for now)
5. **Shipping-Ready**: Static linking, symbol stripping, minimal binary size impact

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Unreal Engine (C++)                                          │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ UAstralSession (UObject)                                │ │
│ │   ├─ FeedPromptRaw(TConstArrayView<uint8>)             │ │
│ │   ├─ TokenBuffer (pre-sized)                           │ │
│ │   ├─ OnStreamBytesNative(): TConstArrayView<uint8>     │ │
│ │   ├─ OnBytesReceived: TArray<uint8> (Blueprint)        │ │
│ │   └─ FTicker: non-blocking poll + stop on EOS          │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ FAstralRuntimeModule (IModuleInterface)                 │ │
│ │   ├─ Init: astral_init()                                │ │
│ │   ├─ Shutdown: astral_shutdown()                        │ │
│ │   └─ Allocator: FMemory → AstralAllocator              │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │ C ABI
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ThirdParty/AstralCore (static lib)                          │
│   ├─ astral_rt.h (C ABI)                                    │
│   ├─ libastral_rt.a (Linux/macOS)                           │
│   └─ astral_rt.lib (Windows; may be `astral_rt_static.lib` when building both static+shared) │
└─────────────────────────────────────────────────────────────┘
```

## Plugin Structure

```
Plugins/
└── AstralRT/
    ├── AstralRT.uplugin
    ├── Source/
    │   ├── AstralRT/
    │   │   ├── AstralRT.Build.cs
    │   │   ├── Public/
    │   │   │   ├── IAstralRT.h             # Module interface
    │   │   │   ├── AstralSession.h          # High-level C++ API
    │   │   │   └── AstralTypes.h            # Structs, enums
    │   │   └── Private/
    │   │       ├── AstralRuntimeModule.cpp  # Module lifecycle
    │   │       ├── AstralModel.cpp          # Model wrapper
    │   │       ├── AstralSession.cpp        # Implementation
    │   └── ThirdParty/
│       └── AstralCore/
│           ├── include/
│           │   ├── astral_rt.h
│           └── lib/
    │               ├── Win64/
    │               │   └── astral_rt.lib (or `astral_rt_static.lib`)
    │               ├── Linux/
    │               │   └── libastral_rt.a
    │               ├── Mac/
    │               │   └── libastral_rt.a
    │               ├── Android/
    │               │   └── arm64-v8a/libastral_rt.a
    │               └── IOS/
    │                   └── libastral_rt.a
    ├── Content/
    │   └── (Blueprint assets, example widgets)
    └── README.md
```

## Module Definition (`AstralRT.uplugin`)

```json
{
  "FileVersion": 3,
  "Version": 1,
  "VersionName": "0.1.0",
  "FriendlyName": "Astral Runtime",
  "Description": "Native LLM runtime bindings for Unreal Engine",
  "Category": "AI",
  "CreatedBy": "Astral Team",
  "CreatedByURL": "",
  "DocsURL": "",
  "MarketplaceURL": "",
  "SupportURL": "",
  "CanContainContent": true,
  "IsBetaVersion": true,
  "IsExperimentalVersion": false,
  "Installed": false,
  "Modules": [
    {
      "Name": "AstralRT",
      "Type": "Runtime",
      "LoadingPhase": "PreDefault",
      "PlatformAllowList": [
        "Win64",
        "Linux",
        "Mac",
        "Android",
        "IOS"
      ]
    }
  ]
}
```

## Build Rules (`AstralRT.Build.cs`)

```csharp
using UnrealBuildTool;
using System.IO;

public class AstralRT : ModuleRules
{
    public AstralRT(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine"
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            // Add any private dependencies here
        });

        // ThirdParty: Astral Core
        string ThirdPartyPath = Path.Combine(ModuleDirectory, "../../ThirdParty/AstralCore");
        string IncludePath = Path.Combine(ThirdPartyPath, "include");
        string LibPath = Path.Combine(ThirdPartyPath, "lib");

        PublicIncludePaths.Add(IncludePath);

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string StaticLib = Path.Combine(LibPath, "Win64", "astral_rt.lib");
            string StaticLibAlt = Path.Combine(LibPath, "Win64", "astral_rt_static.lib");
            PublicAdditionalLibraries.Add(File.Exists(StaticLib) ? StaticLib : StaticLibAlt);
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            string LibFilePath = Path.Combine(LibPath, "Linux", "libastral_rt.a");
            PublicAdditionalLibraries.Add(LibFilePath);
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            string LibFilePath = Path.Combine(LibPath, "Mac", "libastral_rt.a");
            PublicAdditionalLibraries.Add(LibFilePath);
        }
        else if (Target.Platform == UnrealTargetPlatform.Android)
        {
            string Arm64LibPath = Path.Combine(LibPath, "Android", "arm64-v8a", "libastral_rt.a");
            PublicAdditionalLibraries.Add(Arm64LibPath);
        }
        else if (Target.Platform == UnrealTargetPlatform.IOS)
        {
            string LibFilePath = Path.Combine(LibPath, "IOS", "libastral_rt.a");
            PublicAdditionalLibraries.Add(LibFilePath);
        }

        // Disable exceptions across C ABI
        bEnableExceptions = false;
        bUseRTTI = false;
    }
}
```

## Module Interface (`IAstralRT.h`)

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

/**
 * Astral Runtime module interface.
 * Handles init/shutdown of the Astral inference library.
 */
class ASTRALRT_API IAstralRT : public IModuleInterface
{
public:
    /**
     * Singleton-like access to this module's interface.
     * Beware: not thread-safe, use only from game thread.
     */
    static inline IAstralRT& Get()
    {
        return FModuleManager::LoadModuleChecked<IAstralRT>("AstralRT");
    }

    /**
     * Checks if the module is loaded and ready.
     */
    static inline bool IsAvailable()
    {
        return FModuleManager::Get().IsModuleLoaded("AstralRT");
    }

    /**
     * Called when module is loaded (PreDefault phase).
     */
    virtual void StartupModule() override = 0;

    /**
     * Called when module is unloaded (shutdown).
     */
    virtual void ShutdownModule() override = 0;

    /**
     * Check if Astral runtime is initialized.
     */
    virtual bool IsInitialized() const = 0;
};
```

## Types (`AstralTypes.h`)

```cpp
#pragma once

#include "CoreMinimal.h"
#include "astral_rt.h" // C ABI header

#include "AstralTypes.generated.h"

/** Error codes returned by Astral C API */
UENUM(BlueprintType)
enum class EAstralError : int32
{
    OK = 0,
    Invalid = -1,
    NoMem = -2,
    Busy = -3,
    Timeout = -4,
    State = -5,
    Backend = -6,
    Canceled = -7,
    Unsupported = -8
};

/** Model configuration */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelDesc
{
    GENERATED_BODY()

    /** Path to GGUF model file */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString ModelPath;

    /** Optional backend override (e.g., "cpu", "mock") */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString BackendName;

    /** Number of layers to offload to GPU (0 = CPU only) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 GpuLayers = 0;

    /** Context size (tokens) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 ContextSize = 2048;

    /** Batch size for prompt processing */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 BatchSize = 512;

    /** Number of threads (0 = auto) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 NumThreads = 0;

    /** Embeddings-only mode */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bEmbeddingsOnly = false;
};

/** Session configuration */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralSessionDesc
{
    GENERATED_BODY()

    /** Maximum tokens to generate */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 MaxTokens = 512;

    /** Sampling temperature (0.0 = greedy, 1.0 = diverse) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float Temperature = 0.7f;

    /** Top-K sampling */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 TopK = 40;

    /** Top-P (nucleus) sampling */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TopP = 0.95f;

    /** Enable token streaming */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bStreamEnabled = true;

    /** RNG seed for deterministic sampling (0 = auto) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 Seed = 0;
};

/** Session statistics */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralStats
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double InitTimeMs = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double FirstTokenTimeMs = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double TokensPerSecond = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint64 BytesCommitted = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint64 BytesReserved = 0;
};
```

## Session API (`AstralSession.h`)

```cpp
#pragma once

#include "CoreMinimal.h"
#include "AstralTypes.h"
#include "astral_rt.h"

#include "AstralSession.generated.h"

/**
 * High-level C++ wrapper for Astral inference session.
 * Manages native resources; safe to use from game thread or async tasks.
 */
UCLASS(BlueprintType)
class ASTRALRT_API UAstralSession : public UObject
{
    GENERATED_BODY()

public:
    UAstralSession();

    /**
     * Create a session from a loaded model handle.
     * @param ModelHandle Opaque model handle (from FAstralRuntimeModule::LoadModel)
     * @param Desc Session configuration
     * @return true if creation succeeded
     */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Create(AstralHandle ModelHandle, const FAstralSessionDesc& Desc);

    /**
     * Feed a prompt chunk (UTF-8). Call with bFinalize=true on last chunk.
     * @param Prompt UTF-8 encoded text
     * @param bFinalize True if this is the last chunk
     * @return true if feed succeeded
     */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool FeedPrompt(const FString& Prompt, bool bFinalize = true);

    /**
     * Feed a prompt from raw UTF-8 bytes (zero-copy, no FString conversion).
     * @param Utf8Data UTF-8 byte array
     * @param bFinalize True if this is the last chunk
     * @return true if feed succeeded
     */
    bool FeedPromptRaw(TConstArrayView<uint8> Utf8Data, bool bFinalize = true);

    /**
     * Start decoding (non-blocking). Tokens will be emitted to OnTokenReceived.
     * @return true if decode started
     */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Decode();

    /**
     * Read tokens from stream into output buffer.
     * @param OutBuffer Output buffer (must be pre-sized)
     * @param TimeoutMs Timeout in milliseconds
     * @return Number of bytes read (0 if none available)
     */
    int32 StreamRead(TArray<uint8>& OutBuffer, uint32 TimeoutMs = 100);

    /**
     * Read tokens as FString (allocates; prefer StreamRead for hot paths).
     * @param TimeoutMs Timeout in milliseconds
     * @return Decoded UTF-8 string
     */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    FString StreamReadString(uint32 TimeoutMs = 100);

    /**
     * Get session statistics.
     */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralStats GetStats() const;

    /**
     * Check if session is valid (handle != 0).
     */
    UFUNCTION(BlueprintPure, Category = "Astral")
    bool IsValid() const { return SessionHandle != 0; }

    // UObject overrides
    virtual void BeginDestroy() override;

    /**
     * Blueprint event: called when a token is received.
     * WARNING: This allocates FString; prefer C++ StreamRead for performance.
     */
    UPROPERTY(BlueprintAssignable, Category = "Astral")
    FOnTokenReceived OnTokenReceived;

    DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTokenReceived, const FString&, Token);

private:
    AstralHandle SessionHandle = 0;
    TArray<uint8> TokenBuffer; // Pre-allocated, reused per-frame

    // Ticker handle for polling stream
    FTSTicker::FDelegateHandle TickerHandle;
    bool TickStream(float DeltaTime);
};
```

## Implementation (`AstralSession.cpp`)

```cpp
#include "AstralSession.h"
#include "Containers/UnrealString.h"
#include "HAL/PlatformFileManager.h"

UAstralSession::UAstralSession()
{
    // Pre-allocate token buffer (4KB; no per-frame allocations)
    TokenBuffer.SetNumUninitialized(4096);
}

bool UAstralSession::Create(AstralHandle ModelHandle, const FAstralSessionDesc& Desc)
{
    if (SessionHandle != 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Session already created"));
        return false;
    }

    AstralSessionDesc NativeDesc = {};
    NativeDesc.model = ModelHandle;
    NativeDesc.max_tokens = Desc.MaxTokens;
    NativeDesc.temperature = Desc.Temperature;
    NativeDesc.top_k = Desc.TopK;
    NativeDesc.top_p = Desc.TopP;
    NativeDesc.stream_enabled = Desc.bStreamEnabled ? 1 : 0;
    NativeDesc.seed = Desc.Seed;

    AstralErr Err = astral_session_create(&NativeDesc, &SessionHandle);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogTemp, Error, TEXT("astral_session_create failed: %d"), static_cast<int32>(Err));
        return false;
    }

    // Start ticker for stream polling
    if (Desc.bStreamEnabled)
    {
        TickerHandle = FTSTicker::GetCoreTicker().AddTicker(
            FTickerDelegate::CreateUObject(this, &UAstralSession::TickStream),
            0.0f // Poll every frame
        );
    }

    return true;
}

bool UAstralSession::FeedPrompt(const FString& Prompt, bool bFinalize)
{
    // Convert FString to UTF-8 (allocates; prefer FeedPromptRaw)
    FTCHARToUTF8 Utf8(*Prompt);
    return FeedPromptRaw(TConstArrayView<uint8>(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length()), bFinalize);
}

bool UAstralSession::FeedPromptRaw(TConstArrayView<uint8> Utf8Data, bool bFinalize)
{
    if (SessionHandle == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Session not created"));
        return false;
    }

    AstralSpanU8 Span = {};
    Span.data = Utf8Data.GetData();
    Span.len = static_cast<uint32>(Utf8Data.Num());

    AstralErr Err = astral_session_feed(SessionHandle, Span, bFinalize ? 1 : 0);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogTemp, Error, TEXT("astral_session_feed failed: %d"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::Decode()
{
    if (SessionHandle == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Session not created"));
        return false;
    }

    AstralErr Err = astral_session_decode(SessionHandle);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogTemp, Error, TEXT("astral_session_decode failed: %d"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

int32 UAstralSession::StreamRead(TArray<uint8>& OutBuffer, uint32 TimeoutMs)
{
    if (SessionHandle == 0 || OutBuffer.Num() == 0)
    {
        return 0;
    }

    AstralMutSpanU8 Span = {};
    Span.data = OutBuffer.GetData();
    Span.len = static_cast<uint32>(OutBuffer.Num());

    int32 BytesRead = astral_stream_read(SessionHandle, Span, TimeoutMs);
    if (BytesRead < 0)
    {
        AstralErr Err = static_cast<AstralErr>(BytesRead);
        if (Err == ASTRAL_E_TIMEOUT)
        {
            return 0; // No data yet
        }
        UE_LOG(LogTemp, Error, TEXT("astral_stream_read failed: %d"), BytesRead);
        return 0;
    }

    return BytesRead;
}

FString UAstralSession::StreamReadString(uint32 TimeoutMs)
{
    int32 BytesRead = StreamRead(TokenBuffer, TimeoutMs);
    if (BytesRead == 0)
    {
        return FString();
    }

    // Convert UTF-8 to FString (allocates)
    FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(TokenBuffer.GetData()), BytesRead);
    return FString(Converter.Length(), Converter.Get());
}

FAstralStats UAstralSession::GetStats() const
{
    FAstralStats Result;
    if (SessionHandle == 0)
    {
        return Result;
    }

    AstralStats NativeStats = {};
    AstralErr Err = astral_session_stats(SessionHandle, &NativeStats);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogTemp, Error, TEXT("astral_session_stats failed: %d"), static_cast<int32>(Err));
        return Result;
    }

    Result.InitTimeMs = NativeStats.t_init_ms;
    Result.FirstTokenTimeMs = NativeStats.t_first_token_ms;
    Result.TokensPerSecond = NativeStats.tok_per_s;
    Result.BytesCommitted = NativeStats.bytes_committed;
    Result.BytesReserved = NativeStats.bytes_reserved;

    return Result;
}

bool UAstralSession::TickStream(float DeltaTime)
{
    if (SessionHandle == 0)
    {
        return false; // Stop ticker
    }

    // Poll for tokens (no GC allocation)
    int32 BytesRead = StreamRead(TokenBuffer, 10 /*ms*/);
    if (BytesRead > 0)
    {
        // Fire Blueprint event (allocates FString; prefer C++ callback for perf)
        if (OnTokenReceived.IsBound())
        {
            FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(TokenBuffer.GetData()), BytesRead);
            FString Token(Converter.Length(), Converter.Get());
            OnTokenReceived.Broadcast(Token);
        }
    }

    return true; // Continue ticking
}

void UAstralSession::BeginDestroy()
{
    if (SessionHandle != 0)
    {
        // Stop ticker
        if (TickerHandle.IsValid())
        {
            FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
        }

        astral_session_destroy(SessionHandle);
        SessionHandle = 0;
    }

    Super::BeginDestroy();
}
```

## Module Lifecycle (`AstralRuntimeModule.cpp`)

```cpp
#include "IAstralRT.h"
#include "Modules/ModuleManager.h"
#include "HAL/PlatformProcess.h"
#include "astral_rt.h"

class FAstralRuntimeModule : public IAstralRT
{
public:
    virtual void StartupModule() override
    {
        UE_LOG(LogTemp, Log, TEXT("AstralRT: Module startup"));

        // Create FMemory-backed allocator
        AstralAllocator Allocator = {};
        Allocator.alloc = &UEAlloc;
        Allocator.free = &UEFree;
        Allocator.user = nullptr;

        // Initialize Astral runtime
        AstralInit InitCfg = {};
        InitCfg.sys_alloc = Allocator;
        InitCfg.log_cb = &UELog; // Forwards Astral UTF-8 logs to UE_LOG
        InitCfg.log_user = nullptr;
        InitCfg.reserve_bytes = 2ULL << 30; // 2 GB
        InitCfg.thread_count = 0; // Auto
        InitCfg.numa_node = 0xFFFFFFFF; // Any
        InitCfg.enable_hugepages = 0;

        AstralErr Err = astral_init(&InitCfg);
        if (Err != ASTRAL_OK)
        {
            UE_LOG(LogTemp, Error, TEXT("AstralRT: Init failed: %d"), static_cast<int32>(Err));
            bInitialized = false;
            return;
        }

        bInitialized = true;
        UE_LOG(LogTemp, Log, TEXT("AstralRT: Initialized"));
    }

    virtual void ShutdownModule() override
    {
        if (bInitialized)
        {
            astral_shutdown();
            bInitialized = false;
            UE_LOG(LogTemp, Log, TEXT("AstralRT: Shutdown"));
        }
    }

    virtual bool IsInitialized() const override
    {
        return bInitialized;
    }

private:
    bool bInitialized = false;

    // FMemory allocator bridge
    static void* UEAlloc(void* User, size_t Size, size_t Align)
    {
        return FMemory::Malloc(Size, static_cast<uint32>(Align));
    }

    static void UEFree(void* User, void* Ptr, size_t Size, size_t Align)
    {
        FMemory::Free(Ptr);
    }
};

IMPLEMENT_MODULE(FAstralRuntimeModule, AstralRT)
```

## Blueprint Usage Example

```cpp
// In a Blueprint or C++ Actor:

UCLASS()
class AMyAIActor : public AActor
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI")
    UAstralSession* Session;

    UPROPERTY()
    UAstralModel* Model;

    virtual void BeginPlay() override
    {
        Super::BeginPlay();

        // Load model (mock backend example; real GGUF uses ModelPath instead).
        Model = NewObject<UAstralModel>(this);
        FAstralModelDesc ModelDesc;
        ModelDesc.BackendName = TEXT("mock");
        if (!Model->Load(ModelDesc))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to load model"));
            return;
        }

        // Create session
        Session = NewObject<UAstralSession>(this);
        FAstralSessionDesc Desc;
        Desc.MaxTokens = 512;
        Desc.Temperature = 0.7f;
        if (!Session->Create(Model, Desc))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to create session"));
            return;
        }

        // Bind bytes-first stream handler (no per-chunk FString allocation).
        Session->OnStreamBytesNative().AddUObject(this, &AMyAIActor::OnStreamBytes);

        // Feed prompt
        Session->FeedPrompt(TEXT("Once upon a time"), true);

        // Start decode
        Session->Decode();
    }

    void OnStreamBytes(TConstArrayView<uint8> Bytes)
    {
        // Consume bytes directly (UTF-8). This sample just logs decoded text.
        FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
        UE_LOG(LogTemp, Log, TEXT("Chunk: %.*s"), Text.Length(), Text.Get());
    }
};
```

## Platform-Specific Notes

### Android

- **Build**: Use NDK r21+ for arm64/armv7
- **Threading**: Use `pthread`; avoid JNI from native threads
- **Memory**: Reduce reserve size to 512MB for low-end devices

### iOS

- **Bitcode**: Disable or provide bitcode lib
- **Memory**: Use `mmap` (POSIX); avoid `vm_allocate`

### Consoles

Consoles are intentionally out of scope for this iteration of Astral’s Unreal integration docs.

## Performance Best Practices

1. **Use `FeedPromptRaw()`**: Avoid FString→UTF-8 conversion in hot paths
2. **Pre-size `TArray<uint8>`**: No per-frame allocations
3. **Async Tasks**: Optionally move `StreamRead()` to `UE::Tasks::Launch`
4. **Prefer bytes/text views**: Use `OnStreamBytesNative()` / `OnStreamTextNative()` and keep `OnTokenReceived` for convenience only
5. **Shipping Builds**: Link statically, strip symbols (`-s` flag)

## Troubleshooting

### Link Errors

- Ensure `.lib`/`.a` files are in correct platform directories
- Check `AstralRT.Build.cs` paths

### Crashes on Startup

- Verify `astral_init()` succeeds; check logs
- Ensure FMemory allocator is set correctly

### Slow Inference

- Increase `BatchSize` (512 → 2048)
- Profile with Unreal Insights → CPU profiler

## Future Enhancements

- **UE Task Graph Integration**: Move decode to async tasks
- **Niagara Plugin**: Emit tokens as Niagara events for VFX
- **MetaHuman Integration**: Drive lip-sync from token stream
- **Replicated Sessions**: Network-replicated inference for multiplayer
