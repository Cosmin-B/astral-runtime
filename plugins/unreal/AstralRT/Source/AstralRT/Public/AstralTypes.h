#pragma once

#include "CoreMinimal.h"

#include "AstralTypes.generated.h"

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

UENUM(BlueprintType)
enum class EAstralImageFormat : uint32
{
    RGB8 = 0,
    RGBA8 = 1,
    RGB_F32 = 2
};

UENUM(BlueprintType)
enum class EAstralAudioFormat : uint32
{
    F32 = 0,
    I16 = 1
};

UENUM(BlueprintType)
enum class EAstralMediaFlags : uint32
{
    None = 0,
    UseGPU = 1u << 0,
    Warmup = 1u << 1
};

UENUM(BlueprintType)
enum class EAstralGpuRouteFlags : uint32
{
    None = 0,
    Device = 1u << 0,
    DeviceMask = 1u << 1,
    Stream = 1u << 2
};

UENUM(BlueprintType)
enum class EAstralModelSourceKind : uint32
{
    Path = 0,
    Memory = 1,
    IO = 2
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString ModelPath;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString BackendName;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 GpuLayers = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 ContextSize = 2048;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 BatchSize = 512;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 NumThreads = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bEmbeddingsOnly = false;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralImageDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralImageFormat Format = EAstralImageFormat::RGB8;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 Width = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 Height = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 RowStride = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 Flags = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> Pixels;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 GpuRouteFlags = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint64 GpuDeviceMask = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint64 GpuStream = 0;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralAudioDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralAudioFormat Format = EAstralAudioFormat::F32;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 Channels = 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 SampleRate = 16000;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint64 FrameCount = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> Samples;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 Flags = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 GpuRouteFlags = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint64 GpuDeviceMask = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint64 GpuStream = 0;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelMediaDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralModelSourceKind SourceKind = EAstralModelSourceKind::Path;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString MediaPath;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> MediaBytes;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 Flags = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 ImageMinTokens = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 ImageMaxTokens = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 GpuRouteFlags = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint64 GpuDeviceMask = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint64 GpuStream = 0;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralMediaInfo
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 SupportsImage = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 SupportsAudio = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 AudioSampleRate = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 ImageMinTokens = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 ImageMaxTokens = 0;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralSessionDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 MaxTokens = 512;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float Temperature = 0.7f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 TopK = 40;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TopP = 0.95f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bStreamEnabled = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 Seed = 0;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralSamplerDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float Temperature = 0.7f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    uint32 TopK = 40;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TopP = 0.95f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float MinP = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TypicalP = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float RepeatPenalty = 1.0f;

    // 0 = disabled, -1 = use ctx size.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 RepeatLastN = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bPenalizeNewline = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float PresencePenalty = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float FrequencyPenalty = 0.0f;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelLimits
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 VocabSize = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 ContextSize = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 MaxBatch = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    uint32 MaxSlots = 0;
};

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
