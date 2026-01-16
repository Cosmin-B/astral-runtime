#pragma once

#include "CoreMinimal.h"
#include "AstralTypes.h"

#include "AstralEmbedder.generated.h"

class UAstralModel;

/**
 * UObject owner for a native Astral embedder handle.
 *
 * The enqueue/collect API exposes native ticket flow for callers that batch
 * work. The synchronous Embed* helpers are convenience wrappers around that
 * path. For low-overhead C++ code, prefer EmbedUtf8Bytes to avoid FString
 * conversion and temporary byte arrays.
 */
UCLASS(BlueprintType)
class ASTRALRT_API UAstralEmbedder : public UObject
{
    GENERATED_BODY()

public:
    /** Create an embedder from a loaded embeddings-capable model. Replaces any existing handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Create(UAstralModel* Model);

    /** Destroy the native embedder handle and clear cached dimension state. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    void Destroy();

    /** True while this object owns a native embedder from the current runtime generation. */
    UFUNCTION(BlueprintPure, Category = "Astral")
    bool IsValid() const;

    /** Vector dimension cached at Create time. */
    UFUNCTION(BlueprintPure, Category = "Astral")
    int32 GetDim() const;

    uint64 GetHandle() const { return IsValid() ? EmbedderHandle : 0; }

    /** Enqueue caller-owned UTF-8 bytes and return a native ticket. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EnqueueUtf8Bytes(const TArray<uint8>& Utf8Bytes, int64& OutTicket);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralAsyncResult EnqueueUtf8BytesResult(const TArray<uint8>& Utf8Bytes);

    /** Enqueue image bytes for embedding. Pixel data must be valid for the call duration. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EnqueueImage(const FAstralImageDesc& Image, int64& OutTicket);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralAsyncResult EnqueueImageResult(const FAstralImageDesc& Image);

    /** Enqueue audio bytes for embedding. FrameCount may be inferred from sample data. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EnqueueAudio(const FAstralAudioDesc& Audio, int64& OutTicket);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralAsyncResult EnqueueAudioResult(const FAstralAudioDesc& Audio);

    /** Enqueue text with image and/or audio payloads when the model media projector supports them. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EnqueueMultimodal(const FString& Text,
                           const FAstralImageDesc& Image,
                           const FAstralAudioDesc& Audio,
                           bool bUseImage,
                           bool bUseAudio,
                           int64& OutTicket);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralAsyncResult EnqueueMultimodalResult(const FString& Text,
                                               const FAstralImageDesc& Image,
                                               const FAstralAudioDesc& Audio,
                                               bool bUseImage,
                                               bool bUseAudio);

    /** Collect a vector for a previously enqueued ticket. OutVector is resized to GetDim(). */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Collect(int64 Ticket, TArray<float>& OutVector);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralAsyncResult CollectResult(int64 Ticket, TArray<float>& OutVector);

    /** Cancel a queued ticket that has not started collecting. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Cancel(int64 Ticket);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralAsyncResult CancelResult(int64 Ticket);

    /** Enqueue and collect in one call using caller-owned UTF-8 bytes. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EmbedUtf8Bytes(const TArray<uint8>& Utf8Bytes, TArray<float>& OutVector);

    /** Enqueue and collect in one call after converting FString to UTF-8. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool EmbedText(const FString& Text, TArray<float>& OutVector);

    virtual void BeginDestroy() override;

private:
    bool IsCurrentRuntimeGeneration() const;

    uint64 EmbedderHandle = 0;
    uint64 ModelHandle = 0;
    uint64 RuntimeGeneration = 0;
    int32 EmbeddingDim = 0;
};
