#pragma once

#include "CoreMinimal.h"
#include "AstralTypes.h"

#include "AstralEmbedder.generated.h"

class UAstralModel;

/**
 * Embeddings wrapper that owns an Astral embedder handle.
 *
 * Notes:
 * - This is a pull-based API (enqueue + collect) with a convenience synchronous Embed* path.
 * - For low-overhead code, prefer `EmbedUtf8Bytes` to avoid FString conversions.
 */
UCLASS(BlueprintType)
class ASTRALRT_API UAstralEmbedder : public UObject
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Create(UAstralModel* Model);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    void Destroy();

    UFUNCTION(BlueprintPure, Category = "Astral")
    bool IsValid() const { return EmbedderHandle != 0; }

    UFUNCTION(BlueprintPure, Category = "Astral")
    int32 GetDim() const { return EmbeddingDim; }

    /** Enqueue UTF-8 bytes (no FString conversion). */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EnqueueUtf8Bytes(const TArray<uint8>& Utf8Bytes, int64& OutTicket);

    /** Enqueue image for embedding. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EnqueueImage(const FAstralImageDesc& Image, int64& OutTicket);

    /** Enqueue audio for embedding. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EnqueueAudio(const FAstralAudioDesc& Audio, int64& OutTicket);

    /** Enqueue multimodal input (text + optional image/audio) for embedding. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EnqueueMultimodal(const FString& Text,
                           const FAstralImageDesc& Image,
                           const FAstralAudioDesc& Audio,
                           bool bUseImage,
                           bool bUseAudio,
                           int64& OutTicket);

    /** Collect embedding for a previously enqueued ticket. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Collect(int64 Ticket, TArray<float>& OutVector);

    /** Synchronous: enqueue + collect using UTF-8 bytes. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EmbedUtf8Bytes(const TArray<uint8>& Utf8Bytes, TArray<float>& OutVector);

    /** Synchronous: enqueue + collect (convenience; converts FString to UTF-8). */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EmbedText(const FString& Text, TArray<float>& OutVector);

    virtual void BeginDestroy() override;

private:
    uint64 EmbedderHandle = 0;
    uint64 ModelHandle = 0;
    int32 EmbeddingDim = 0;
};
