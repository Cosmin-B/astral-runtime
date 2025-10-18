# AstralRT (Unreal Engine Plugin)

This is a UE5 plugin scaffold that wraps Astral's C ABI (`astral_rt.h`) with engine-friendly types and a bytes-first streaming path (UTF-8 chunks via `TConstArrayView<uint8>` / `TArray<uint8>`).

## Build and package (native)

This repo includes a CMake preset that packages the native static library + headers into the plugin's `ThirdParty/` layout:

```bash
cd astral
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

After this, the plugin will contain:
- `AstralRT/Source/ThirdParty/AstralCore/include/astral_rt.h`
- `AstralRT/Source/ThirdParty/AstralCore/lib/<Platform>/*`

The package target hashes the staged header and native library after copy, then
fails if either one differs from the current source header or built `astral_rt`
target.

For the full UE 5.7 path, including container commands and release evidence,
see [UNREAL_57_QUICKSTART.md](../../../docs/integration/UNREAL_57_QUICKSTART.md).

## Use in a UE project

Copy `astral/plugins/unreal/AstralRT/` into your Unreal project:

```
<YourProject>/Plugins/AstralRT/
```

Enable the plugin and build the project.

## Minimal example (mock backend)

This uses the mock backend (no GGUF needed) and streams UTF-8 bytes via `UAstralSession::OnStreamBytesNative()`:

```cpp
// AMyAIActor.h
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
            UE_LOG(LogAstralRT, Error, TEXT("AstralRT: Model load failed"));
            return;
        }

        Session = NewObject<UAstralSession>(this);
        FAstralSessionDesc SessionDesc;
        SessionDesc.MaxTokens = 64;
        SessionDesc.Temperature = 0.0f;
        SessionDesc.Seed = 1;

        if (!Session->Create(Model, SessionDesc))
        {
            UE_LOG(LogAstralRT, Error, TEXT("AstralRT: Session create failed"));
            return;
        }

        Session->OnStreamBytesNative().AddUObject(this, &AMyAIActor::OnStreamBytes);
        Session->FeedPrompt(TEXT("hi"), true);
        Session->Decode();
    }

    void OnStreamBytes(TConstArrayView<uint8> Bytes)
    {
        // Low-overhead path: keep bytes as UTF-8.
        // This sample just logs the bytes as text.
        FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
        UE_LOG(LogAstralRT, Log, TEXT("%.*s"), Text.Length(), Text.Get());
    }
};
```

For Blueprint convenience, `UAstralSession` also exposes:
- `OnBytesReceived` (UTF-8 bytes, per tick)
- `OnTokenReceived` (decoded text, per tick; allocates an `FString` only when bound)

## Vision / Audio (Media)

Media support requires a projector/encoder GGUF and a native Astral build compiled with `ASTRAL_ENABLE_MTMD=ON`. Initialize media once per model before creating sessions or embedders that will consume images or audio:

```cpp
FAstralModelMediaDesc MediaDesc;
MediaDesc.MediaPath = TEXT("/path/to/media.gguf");
Model->InitMedia(MediaDesc);
```

Feed media into a session prompt. `FAstralImageDesc::Pixels` and `FAstralAudioDesc::Samples` must remain valid until the feed call returns.

```cpp
FAstralImageDesc Image;
Image.Format = EAstralImageFormat::RGB8;
Image.Width = 224;
Image.Height = 224;
Image.Pixels.SetNumZeroed(224 * 224 * 3);
Session->FeedImage(Image, true);

FAstralAudioDesc Audio;
Audio.Format = EAstralAudioFormat::I16;
Audio.Channels = 1;
Audio.SampleRate = 16000;
Audio.FrameCount = 16000;
Audio.Samples.SetNumZeroed(16000 * 2);
Session->FeedAudio(Audio, true);
```

## Multimodal Embeddings

Load the model with `FAstralModelDesc::bEmbeddingsOnly = true`; call `InitMedia` first when image or audio embeddings are used.

```cpp
UAstralEmbedder* Embedder = NewObject<UAstralEmbedder>(this);
Embedder->Create(Model);

int64 Ticket = 0;
Embedder->EnqueueMultimodal(TEXT("describe"), Image, Audio, /*bUseImage=*/true, /*bUseAudio=*/false, Ticket);

TArray<float> Vec;
Embedder->Collect(Ticket, Vec);
```

## Notes

- The module initializes Astral at startup and shuts it down on module unload.
- Streaming is pull-based via `astral_stream_read()` into a pre-sized `TArray<uint8>`.

## Automation tests

Editor-only Automation tests live under `Source/AstralRT/Private/Tests/`:
- `AstralRT.Module.Init`
- `AstralRT.Mock.E2E`
- `AstralRT.Mock.MediaFeed`
- `AstralRT.Mock.MultimodalEmbed`

Run from Unreal's Automation window or via console:
`Automation RunTests AstralRT.*`
