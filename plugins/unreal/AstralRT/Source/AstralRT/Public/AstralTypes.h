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
