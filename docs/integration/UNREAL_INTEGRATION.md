# Unreal Integration

This page describes the Unreal plugin that lives in
`plugins/unreal/AstralRT`. It is a current integration guide, not a release
sign-off. Real UE 5.7 container runs and UE 5.4+ compatibility editor runs are
still required before the plugin can be called production-ready.

For the UE 5.7 container path, use
[`UNREAL_57_QUICKSTART.md`](UNREAL_57_QUICKSTART.md).

## Supported Shape

- UE 5.7 is the lead target.
- UE 5.4, 5.5, 5.6, and 5.7 are compatibility targets.
- The plugin is a runtime module named `AstralRT`.
- The native runtime is linked from the plugin `Source/ThirdParty/AstralCore`
  directory populated by the CMake `unreal-plugin` preset.
- C++ code can consume streaming bytes without converting each chunk to
  `FString`.
- Blueprint delegates are available for convenience and may copy or marshal
  streamed data on the game thread.

## Blueprint Surface

`UAstralBlueprintLibrary` provides Blueprint-callable entry points for common
setup and diagnostics:

- `CreateAstralModel`, `CreateAstralSession`, and `CreateAstralEmbedder`
  allocate the existing UObject wrappers with a caller-provided owner.
- `GetLastAstralError` returns the native thread-local error text after a
  failed Astral call.
- `ErrorCodeName` maps native integer error codes to stable symbolic names such
  as `ASTRAL_E_TIMEOUT`.
- `FAstralAsyncResult` reports `bSuccess`, native `ErrorCode`, ticket, and
  queue-state flags for ticketed Blueprint operations.
- `FAstralOperationResult` reports native handle, count, and polling state for
  tool, memory, and agent helpers that need more detail than a bool return.
- `HasEmbeddings`, `HasSamplerControls`, `HasStopSequences`,
  `HasGpuOffload`, `HasLora`, `HasImageInput`, `HasAudioInput`,
  `HasMultimodalEmbeddings`, `HasGrammar`, `HasLogprobs`, `HasKvState`,
  `HasSlots`, `HasGbnfGrammar`, and `HasJsonSchemaGrammar` decode the int64
  capability mask returned by `UAstralModel::GetCaps`.

The factories do not load models or create native sessions by themselves. They
only create wrapper objects; ownership, model lifetime, and session creation
still follow the contracts on `UAstralModel`, `UAstralSession`, and
`UAstralEmbedder`.

For Blueprint graphs that manage native handles directly, prefer the
result-returning helpers:

- `CreateToolsetResult` and `ParseToolCallResult`
- `CreateMemoryIndexResult`, `LoadMemoryIndexResult`,
  `AddMemoryBatchResult`, `RemoveMemoryRecordResult`,
  `ClearMemoryIndexResult`, `SaveMemoryIndexResult`,
  `SearchMemoryIndexResult`, `BeginMemorySearchResult`, and
  `FetchMemorySearchResult`
- `CreateMemorySearchRequestResult`, `GetRequestStatusResult`,
  `WaitRequestResult`, and `CancelRequestResult`
- `CreatePromptCacheResult`, `LoadPromptCacheResult`,
  `ClearPromptCacheResult`, `GetPromptCacheStatsResult`,
  `PutPromptCacheTokensResult`, `GetPromptCacheTokensResult`, and
  `SavePromptCacheResult`
- `CreateAgentResult`, `SetAgentSystemPromptResult`,
  `AddAgentMessageResult`, `ClearAgentHistoryResult`,
  `EnqueueAgentChatResult`, `CancelAgentChatResult`, `ReadAgentChatResult`,
  and `GetAgentChatStatusResult`

The older bool helpers remain compatibility wrappers over the same native calls.

Prompt cache helpers are setup-time APIs for tokenized system prompts, tool
prefixes, memory sections, and history fragments. `FAstralPromptCacheDesc`
controls entry and token budgets, `FAstralPromptCacheKey` identifies one
model-scoped section, and `FAstralPromptCacheStats` reports occupancy plus
optional hit/miss counters. Agents can consume the resulting native cache
handle through `FAstralAgentDesc::PromptCacheHandle`; Blueprint token arrays are
copied only when `GetPromptCacheTokensResult` is called. Agent chat status also
reports prompt-cache reused/new token counts and per-request hit/miss markers.

Memory helpers expose the native flat vector index lifecycle, including
snapshot save/load for staged RAG data. `SaveMemoryIndexResult` writes an
engine-owned byte array, `LoadMemoryIndexResult` restores it with the same
descriptor shape, and remove/clear helpers update the native index without
rebuilding it from Blueprint arrays.

Request status helpers expose the native `AstralRequestRef` /
`AstralRequestStatus` shape to Blueprint. Memory search cursors can be wrapped
with `CreateMemorySearchRequestResult`, then polled with
`GetRequestStatusResult` or `WaitRequestResult`; `QueueDepth` reports remaining
cursor results. Cursor cancellation returns unsupported, and storage is still
released with `EndMemorySearch`.

## Layout

```text
plugins/unreal/AstralRT/
  AstralRT.uplugin
  README.md
  Source/
    AstralRT/
      AstralRT.Build.cs
      Public/
        AstralBlueprintLibrary.h
        AstralEmbedder.h
        AstralLog.h
        AstralMediaLibrary.h
        AstralModel.h
        AstralSession.h
        AstralTypes.h
        IAstralRT.h
      Private/
        AstralBlueprintLibrary.cpp
        AstralEmbedder.cpp
        AstralMediaLibrary.cpp
        AstralModel.cpp
        AstralRuntimeModule.cpp
        AstralSession.cpp
        AstralSessionStreamPump.cpp
        AstralSessionStreamPump.h
        Tests/AstralRTTests.cpp
    ThirdParty/AstralCore/
      README.md
      include/astral_rt.h
      lib/<Platform>/...
```

The descriptor has `CanContainContent: false`; samples live in docs and tests,
not as plugin content assets.

## Descriptor

The committed descriptor is:

```json
{
  "FileVersion": 3,
  "Version": 1,
  "VersionName": "0.1.0",
  "FriendlyName": "Astral Runtime",
  "Description": "Native LLM runtime bindings for Unreal Engine",
  "Category": "AI",
  "CreatedBy": "Astral",
  "CreatedByURL": "https://github.com/Cosmin-B/astral",
  "DocsURL": "https://github.com/Cosmin-B/astral/tree/main/docs/integration",
  "MarketplaceURL": "",
  "SupportURL": "https://github.com/Cosmin-B/astral/issues",
  "CanContainContent": false,
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

`docs/release/dependency-pins.tsv` pins the descriptor version and URL fields.

## Build The Native Package

Run this before copying the plugin into an Unreal project or running
Automation:

```bash
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

The build stages:

- `Source/ThirdParty/AstralCore/include/astral_rt.h`
- `Source/ThirdParty/AstralCore/lib/Linux/libastral_rt.a`
- the corresponding static library path for other configured platform presets

The package target hashes the staged header and native library after copy and
prints `Unreal ThirdParty provenance OK` when they match the source header and
the built native target.

## Build Rules

`AstralRT.Build.cs` adds `Core`, `CoreUObject`, and `Engine`, then links the
static Astral runtime from `Source/ThirdParty/AstralCore/lib/<Platform>`.
It fails during UnrealBuildTool module evaluation if the staged C ABI header or
selected static library is missing, and prints the `unreal-plugin` rebuild
command in the error.

The current platform library names are:

| Unreal platform | Library path |
|---|---|
| Win64 | `Win64/astral_rt.lib`, falling back to `Win64/astral_rt_static.lib` |
| Linux | `Linux/libastral_rt.a` |
| Mac | `Mac/libastral_rt.a` |
| Android | `Android/arm64-v8a/libastral_rt.a` |
| IOS | `IOS/libastral_rt.a` |

The module builds with `bEnableExceptions = false` and `bUseRTTI = false`.

## Runtime Module

`IAstralRT` exposes:

- `IAstralRT::Get()`
- `IAstralRT::IsAvailable()`
- `IsInitialized()`
- `ResetAllocatorStats()`
- `GetAllocatorStats()`

`FAstralRuntimeModule` initializes the native runtime during module startup and
shuts it down during module unload. It also bridges native logs into
`LogAstralRT`. The runtime module does not expose direct model/session creation
methods; use the UObject wrappers below.

Module startup passes Unreal's `FMemory::Malloc` and `FMemory::Free` through the
native `AstralAllocator` hook before models or sessions are created.
`ResetAllocatorStats()` and `GetAllocatorStats()` expose test/debug counters so
Automation can prove model/session work used the engine-owned allocation path.
They are validation counters, not gameplay telemetry.

## Public Wrappers

### Models

`UAstralModel` owns a native model handle.

```cpp
UAstralModel* Model = NewObject<UAstralModel>(this);

FAstralModelDesc ModelDesc;
ModelDesc.BackendName = TEXT("mock");

if (!Model->Load(ModelDesc))
{
    UE_LOG(LogAstralRT, Error, TEXT("AstralRT: model load failed"));
    return;
}
```

`Load` releases any previous handle before opening the new model. Keep the
`UAstralModel` object alive for sessions and embedders created from it. Call
`Release` before destroying dependent sessions or embedders when the lifetime is
managed manually.

Model source behavior:

- `SourceKind = Path` passes a real filesystem path to the native backend.
- `PathRoot = ProjectContent`, `ProjectSaved`, or `ProjectPersistentDownload`
  resolves relative paths under Unreal's content, saved, or persistent download
  directories before the C ABI call. Absolute paths bypass root resolution.
- `SourceKind = Memory` passes `ModelBytes` directly to native code. Use this
  for cooked pak/IoStore payloads or any model loaded through Unreal asset/bulk
  data APIs.
- The CPU llama backend may materialize `Memory` input to a temporary file in
  desktop presets so llama.cpp can use its file-backed loader. Use `Saved` or a
  project-managed cache for production-sized models when startup time and disk
  lifetime matter.
- `SourceKind = IO` is a native ABI option, but the Unreal UObject wrapper does
  not expose callback-backed IO yet.

Useful queries:

- `GetEmbeddingDim`
- `GetCaps`
- `GetLimits`
- `InitMedia`
- `GetMediaInfo`

### Sessions

`UAstralSession` owns a native session handle and keeps the model handle value
used at creation so `Reset` can rebuild native state for the same model.

```cpp
UAstralSession* Session = NewObject<UAstralSession>(this);

FAstralSessionDesc SessionDesc;
SessionDesc.MaxTokens = 64;
SessionDesc.Temperature = 0.0f;
SessionDesc.Seed = 1;

if (!Session->Create(Model, SessionDesc))
{
    UE_LOG(LogAstralRT, Error, TEXT("AstralRT: session create failed"));
    return;
}
```

Text input paths:

- `FeedPrompt` converts `FString` to UTF-8 before calling native code.
- `FeedPromptRaw` takes caller-owned UTF-8 bytes and avoids the `FString`
  conversion in C++ callers.

Decode and cancellation:

- `Decode` advances generation once.
- `Cancel` requests cancellation.
- `Wait` returns the native code: `0` for OK, `-7` for canceled, `-4` for
  timeout, or another native error value.
- `Reset` replaces native session state with new session settings.

Stop sequences:

- `StopClear` removes all stop sequences.
- `StopAddUtf8Bytes` adds caller-owned UTF-8 bytes.
- `StopAddString` converts `FString` to UTF-8 first.

Streaming:

- `StreamRead` writes UTF-8 bytes into a caller-sized buffer and returns a byte
  count or a negative native error.
- `StreamReadString` returns a decoded `FString`; empty string also represents
  timeout or no data.
- `OnStreamBytesNative` is the low-overhead C++ delegate.
- `OnStreamTextNative` exposes decoded text as an `FStringView` for C++.
- `OnBytesReceived` and `OnTokenReceived` are Blueprint delegates and may
  allocate or marshal on each tick.

## Bytes-First Streaming Example

```cpp
#include "AstralLog.h"
#include "AstralModel.h"
#include "AstralSession.h"

UCLASS()
class AMyAIActor : public AActor
{
    GENERATED_BODY()

public:
    UPROPERTY()
    UAstralModel* Model = nullptr;

    UPROPERTY()
    UAstralSession* Session = nullptr;

    virtual void BeginPlay() override
    {
        Super::BeginPlay();

        Model = NewObject<UAstralModel>(this);

        FAstralModelDesc ModelDesc;
        ModelDesc.BackendName = TEXT("mock");
        if (!Model->Load(ModelDesc))
        {
            UE_LOG(LogAstralRT, Error, TEXT("AstralRT: model load failed"));
            return;
        }

        Session = NewObject<UAstralSession>(this);

        FAstralSessionDesc SessionDesc;
        SessionDesc.MaxTokens = 64;
        SessionDesc.Temperature = 0.0f;
        SessionDesc.Seed = 1;
        if (!Session->Create(Model, SessionDesc))
        {
            UE_LOG(LogAstralRT, Error, TEXT("AstralRT: session create failed"));
            return;
        }

        Session->OnStreamBytesNative().AddUObject(this, &AMyAIActor::OnStreamBytes);
        Session->FeedPrompt(TEXT("hi"), true);
        Session->Decode();
    }

    void OnStreamBytes(TConstArrayView<uint8> Bytes)
    {
        FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
        UE_LOG(LogAstralRT, Log, TEXT("%.*s"), Text.Length(), Text.Get());
    }
};
```

## Media

Media support requires a media projector or encoder GGUF and a native build with
`ASTRAL_ENABLE_MTMD=ON`. Initialize media on the model before feeding image or
audio payloads into sessions or embedders.

```cpp
FAstralModelMediaDesc MediaDesc;
MediaDesc.SourceKind = EAstralModelSourceKind::Path;
MediaDesc.MediaPath = TEXT("/path/to/media.gguf");
MediaDesc.MediaPathRoot = EAstralUnrealPathRoot::Raw;

if (!Model->InitMedia(MediaDesc))
{
    UE_LOG(LogAstralRT, Error, TEXT("AstralRT: media init failed"));
    return;
}
```

`MediaPathRoot` follows the same relative-path policy as `FAstralModelDesc`:
absolute projector paths pass through unchanged, `Raw` relative paths become
full process-relative paths, and packaged projects can resolve projector files
under `ProjectContent`, `ProjectSaved`, or `ProjectPersistentDownload`.

`FAstralImageDesc::Pixels` and `FAstralAudioDesc::Samples` are passed to native
code for the duration of the call. `FAstralAudioDesc::FrameCount` may be set to
0 so the wrapper infers frame count from format, channel count, and sample
bytes.

`UAstralMediaLibrary` provides boundary helpers for common engine payloads:

- `MakeRGBA8ImageFromBytes` copies tightly packed RGBA8 bytes into an image
  descriptor.
- `MakeRGBA8ImageFromTexture` copies the first mip of a CPU-readable
  `PF_B8G8R8A8` `UTexture2D` into an RGBA8 descriptor. It returns `false` for
  compressed, GPU-only, stripped, or non-readable texture data.
- `MakePCM16AudioFromBytes` copies interleaved signed 16-bit PCM into an audio
  descriptor and computes frame count.

```cpp
FAstralImageDesc Image;
TArray<uint8> RgbaBytes;
RgbaBytes.SetNumZeroed(224 * 224 * 4);
UAstralMediaLibrary::MakeRGBA8ImageFromBytes(RgbaBytes, 224, 224, Image);
Session->FeedImage(Image, true);

FAstralAudioDesc Audio;
TArray<uint8> Pcm16Bytes;
Pcm16Bytes.SetNumZeroed(16000 * sizeof(int16));
UAstralMediaLibrary::MakePCM16AudioFromBytes(Pcm16Bytes, 1, 16000, Audio);
Session->FeedAudio(Audio, true);
```

## Embeddings

Load the model with `FAstralModelDesc::bEmbeddingsOnly = true` when the model is
used only for embeddings. Create a `UAstralEmbedder` from the loaded model.

```cpp
UAstralEmbedder* Embedder = NewObject<UAstralEmbedder>(this);
if (!Embedder->Create(Model))
{
    UE_LOG(LogAstralRT, Error, TEXT("AstralRT: embedder create failed"));
    return;
}

TArray<uint8> Utf8Bytes;
constexpr ANSICHAR SampleText[] = "hello";
Utf8Bytes.Append(reinterpret_cast<const uint8*>(SampleText), UE_ARRAY_COUNT(SampleText) - 1);

FAstralAsyncResult Enqueue = Embedder->EnqueueUtf8BytesResult(Utf8Bytes);
if (Enqueue.bSuccess)
{
    TArray<float> Vector;
    FAstralAsyncResult Collect = Embedder->CollectResult(Enqueue.Ticket, Vector);
    if (!Collect.bSuccess)
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: collect failed %d"), Collect.ErrorCode);
    }
}
else if (Enqueue.bBackpressure)
{
    UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: embedding queue is full"));
}
```

`EmbedText` is the `FString` convenience path. `EmbedUtf8Bytes` keeps callers on
the UTF-8 byte path. The result-returning enqueue, collect, and cancel helpers
are preferred for Blueprint logic because they distinguish `ASTRAL_E_BUSY`,
timeout, canceled, invalid-input, and unsupported-backend states without
scraping log text. The older bool helpers remain as simple compatibility paths.
`Cancel` releases a queued ticket that has not started collecting yet, which
lets batching callers shed work under queue pressure.
Image, audio, and multimodal embedding calls require media initialization when
the backend needs a projector.

## Diagnostics And Profiling

Use `LogAstralRT` for plugin diagnostics. Runtime code also contains Unreal CPU
profiler scopes around the session tick path, stream pump, and embedder enqueue
and collect paths. Native embedding paths use Astral's Tracy wrapper macros.

Release and CI checks currently gate:

- plugin diagnostics use `LogAstralRT` instead of Unreal's temporary log category
- Unreal docs keep examples on the same plugin log category
- Unreal stream-pump and embedder profiler scopes
- ThirdParty header/library provenance
- Automation result artifact parsing
- UE 5.7 container runner preflight

## Automation

Editor-only Automation tests live under
`plugins/unreal/AstralRT/Source/AstralRT/Private/Tests/`:

- `AstralRT.Module.Init`
- `AstralRT.Mock.E2E`
- `AstralRT.Mock.Embeddings`
- `AstralRT.Mock.EmbedderQueuePressure`
- `AstralRT.Mock.MediaFeed`
- `AstralRT.Mock.MultimodalEmbed`

Run the local editor lane with:

```bash
UNREAL_EDITOR=/path/to/UnrealEditor-Cmd ./scripts/run_unreal_ci_tests.sh
```

Run the UE 5.7 container lanes with:

```bash
./scripts/run_unreal_container_ci.sh --variant slim
./scripts/run_unreal_container_ci.sh --variant full
```

Run the UE 5.4+ compatibility matrix with:

```bash
UNREAL_54_EDITOR=/path/to/5.4/UnrealEditor-Cmd \
UNREAL_55_EDITOR=/path/to/5.5/UnrealEditor-Cmd \
UNREAL_56_EDITOR=/path/to/5.6/UnrealEditor-Cmd \
UNREAL_57_EDITOR=/path/to/5.7/UnrealEditor-Cmd \
./scripts/run_unreal_compatibility_matrix.sh
```

The local CMake gates validate runner behavior and result parsing, but they do
not replace real editor or container execution.

## Current Limits

- Real UE 5.7 container runs require authenticated access to Epic's GHCR
  Unreal images.
- UE 5.4, 5.5, 5.6, and 5.7 editor compatibility must be run on hosts with the
  matching editors installed.
- CUDA and MTMD support need real GPU/projector fixture evidence before release
  sign-off.
- Android and iOS are build-smoked through native artifacts; runtime device
  validation is still required.
