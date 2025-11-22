#pragma once

#include "CoreMinimal.h"

#include "AstralTypes.generated.h"

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

/** Pixel layouts accepted by Astral image media calls. */
UENUM(BlueprintType)
enum class EAstralImageFormat : uint8
{
    RGB8 = 0,
    RGBA8 = 1,
    RGB_F32 = 2
};

/** Packed sample layouts accepted by Astral audio media calls. */
UENUM(BlueprintType)
enum class EAstralAudioFormat : uint8
{
    F32 = 0,
    I16 = 1
};

/** Media initialization flags mirrored from the native C ABI. */
UENUM(BlueprintType)
enum class EAstralMediaFlags : uint8
{
    None = 0,
    UseGPU = 1u << 0,
    Warmup = 1u << 1
};

/** GPU routing fields that should be honored by media-aware backends. */
UENUM(BlueprintType)
enum class EAstralGpuRouteFlags : uint8
{
    None = 0,
    Device = 1u << 0,
    DeviceMask = 1u << 1,
    Stream = 1u << 2
};

/** Backing source for a model or media projector. Unreal currently exposes path and bytes. */
UENUM(BlueprintType)
enum class EAstralModelSourceKind : uint8
{
    Path = 0,
    Memory = 1,
    IO = 2
};

/** Root used to resolve relative filesystem paths before they cross the C ABI. */
UENUM(BlueprintType)
enum class EAstralUnrealPathRoot : uint8
{
    Raw = 0,
    ProjectContent = 1,
    ProjectSaved = 2,
    ProjectPersistentDownload = 3
};

/** Load settings for a native Astral model. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelDesc
{
    GENERATED_BODY()

    /** Backing source used for the model payload. Path and Memory are supported by this wrapper. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralModelSourceKind SourceKind = EAstralModelSourceKind::Path;

    /** Root for relative ModelPath values. Absolute paths are passed through unchanged. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralUnrealPathRoot PathRoot = EAstralUnrealPathRoot::Raw;

    /** Filesystem path to a GGUF model when SourceKind is Path. Empty is valid only for providers that do not need a file. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString ModelPath;

    /** Model bytes when SourceKind is Memory, usually from cooked bulk data or a staged cache file. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> ModelBytes;

    /** Provider name such as "cpu" or "mock". Empty lets the runtime pick. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString BackendName;

    /** Number of layers requested on GPU-capable backends; 0 keeps execution on CPU. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuLayers = 0;

    /** Token context window requested for the model. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 ContextSize = 2048;

    /** Native batch size hint for prompt/decode work. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 BatchSize = 512;

    /** Worker count requested by the caller; 0 uses the runtime default. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 NumThreads = 0;

    /** Load the model for embeddings without preparing generation state where supported. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bEmbeddingsOnly = false;
};

/** Image payload passed to session feed and embedding calls. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralImageDesc
{
    GENERATED_BODY()

    /** Pixel format of Pixels. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralImageFormat Format = EAstralImageFormat::RGB8;

    /** Image width in pixels. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Width = 0;

    /** Image height in pixels. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Height = 0;

    /** Bytes per row. Use 0 when rows are tightly packed for the selected format. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 RowStride = 0;

    /** Native image flags for backend-specific media behavior. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Flags = 0;

    /** Pixel bytes. The wrapper passes this storage to native code for the call duration. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> Pixels;

    /** CUDA device index requested when Device routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    /** Bitmask of EAstralGpuRouteFlags values. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuRouteFlags = 0;

    /** Allowed CUDA device mask requested when DeviceMask routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuDeviceMask = 0;

    /** Backend-specific stream pointer encoded as an integer for Blueprint transport. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuStream = 0;
};

/** Audio payload passed to session feed and embedding calls. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralAudioDesc
{
    GENERATED_BODY()

    /** Sample format of Samples. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralAudioFormat Format = EAstralAudioFormat::F32;

    /** Number of interleaved channels in Samples. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Channels = 1;

    /** Sample rate in hertz. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 SampleRate = 16000;

    /** Frame count. Use 0 to infer from Samples, Format, and Channels. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 FrameCount = 0;

    /** Interleaved sample bytes. The wrapper passes this storage to native code for the call duration. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> Samples;

    /** Native audio flags for backend-specific media behavior. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Flags = 0;

    /** CUDA device index requested when Device routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    /** Bitmask of EAstralGpuRouteFlags values. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuRouteFlags = 0;

    /** Allowed CUDA device mask requested when DeviceMask routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuDeviceMask = 0;

    /** Backend-specific stream pointer encoded as an integer for Blueprint transport. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuStream = 0;
};

/** Projector or encoder settings for model media initialization. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelMediaDesc
{
    GENERATED_BODY()

    /** Source kind for the media projector. Path and Memory are exposed by the wrapper. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralModelSourceKind SourceKind = EAstralModelSourceKind::Path;

    /** Filesystem path used when SourceKind is Path. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString MediaPath;

    /** Root for relative MediaPath values. Absolute paths are passed through unchanged. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralUnrealPathRoot MediaPathRoot = EAstralUnrealPathRoot::Raw;

    /** Projector bytes used when SourceKind is Memory. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> MediaBytes;

    /** Native media init flags such as GPU use and warmup. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Flags = 0;

    /** Minimum image token budget requested from the media projector. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 ImageMinTokens = 0;

    /** Maximum image token budget requested from the media projector. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 ImageMaxTokens = 0;

    /** CUDA device index requested when Device routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    /** Bitmask of EAstralGpuRouteFlags values. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuRouteFlags = 0;

    /** Allowed CUDA device mask requested when DeviceMask routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuDeviceMask = 0;

    /** Backend-specific stream pointer encoded as an integer for Blueprint transport. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuStream = 0;
};

/** Media capabilities reported after InitMedia succeeds. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralMediaInfo
{
    GENERATED_BODY()

    /** Nonzero when the initialized projector accepts image inputs. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 SupportsImage = 0;

    /** Nonzero when the initialized projector accepts audio inputs. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 SupportsAudio = 0;

    /** Preferred audio sample rate reported by the projector. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 AudioSampleRate = 0;

    /** Minimum image-token budget reported by the projector. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ImageMinTokens = 0;

    /** Maximum image-token budget reported by the projector. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ImageMaxTokens = 0;
};

/** Session creation and reset settings for generation. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralSessionDesc
{
    GENERATED_BODY()

    /** Maximum tokens to generate before the session stops. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 MaxTokens = 512;

    /** Sampling temperature. Use 0 for greedy-style deterministic sampling. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float Temperature = 0.7f;

    /** Top-k candidate limit. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 TopK = 40;

    /** Nucleus sampling probability cutoff. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TopP = 0.95f;

    /** Enable the ticker that drains native stream bytes onto Unreal delegates. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bStreamEnabled = true;

    /** Random seed forwarded to native sampling. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Seed = 0;
};

/** Runtime sampler settings that can be changed after session creation. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralSamplerDesc
{
    GENERATED_BODY()

    /** Sampling temperature. Use 0 for greedy-style deterministic sampling. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float Temperature = 0.7f;

    /** Top-k candidate limit. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 TopK = 40;

    /** Nucleus sampling probability cutoff. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TopP = 0.95f;

    /** Minimum probability cutoff relative to the most likely token. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float MinP = 0.0f;

    /** Typical sampling cutoff. Values >= 1 leave typical sampling disabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TypicalP = 1.0f;

    /** Repetition penalty multiplier. 1 leaves logits unchanged. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float RepeatPenalty = 1.0f;

    /** Repetition window: 0 disables it, -1 uses the model context size. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 RepeatLastN = 0;

    /** Include newline tokens in repetition-penalty accounting. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bPenalizeNewline = false;

    /** Presence penalty added for tokens that appeared in the recent window. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float PresencePenalty = 0.0f;

    /** Frequency penalty scaled by token count in the recent window. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float FrequencyPenalty = 0.0f;
};

/** Static limits reported by a loaded model. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelLimits
{
    GENERATED_BODY()

    /** Vocabulary size reported by the backend. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 VocabSize = 0;

    /** Context size used by the loaded model. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ContextSize = 0;

    /** Maximum native batch size reported by the backend. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 MaxBatch = 0;

    /** Maximum concurrent slots reported by continuous-batching backends. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 MaxSlots = 0;
};

/** Runtime counters exposed for profiling and production diagnostics. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralStats
{
    GENERATED_BODY()

    /** Model/session initialization time in milliseconds when reported. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double InitTimeMs = 0.0;

    /** Milliseconds from decode start to first token when reported. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double FirstTokenTimeMs = 0.0;

    /** Throughput reported by the native backend. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double TokensPerSecond = 0.0;

    /** Bytes committed by the runtime arena. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 BytesCommitted = 0;

    /** Bytes reserved by the runtime arena. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 BytesReserved = 0;
};
