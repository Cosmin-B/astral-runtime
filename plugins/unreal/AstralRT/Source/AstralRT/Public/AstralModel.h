#pragma once

#include "CoreMinimal.h"
#include "AstralTypes.h"

#include "AstralModel.generated.h"

/**
 * UObject owner for a native Astral model handle.
 *
 * Load releases any previous handle before opening the requested model. The
 * native handle is released from BeginDestroy, so callers should keep this
 * object alive for every session or embedder created from it.
 */
UCLASS(BlueprintType)
class ASTRALRT_API UAstralModel : public UObject
{
    GENERATED_BODY()

public:
    /** Load a model from FAstralModelDesc. Empty BackendName lets the native runtime choose. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Load(const FAstralModelDesc& Desc);

    /** Release the native model handle. Sessions and embedders that use it must be destroyed first. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    void Release();

    /** True while this object owns a live native model handle. */
    UFUNCTION(BlueprintPure, Category = "Astral")
    bool IsValid() const { return ModelHandle != 0; }

    /** Query the embedding vector dimension reported by the loaded model. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool GetEmbeddingDim(int32& OutDim) const;

    /** Query the native capability bitmask for generation, embeddings, and media support. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool GetCaps(uint64& OutCaps) const;

    /** Query model limits such as context size, vocabulary size, batch size, and slot count. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool GetLimits(FAstralModelLimits& OutLimits) const;

    /** Attach a media projector/encoder to the loaded model before feeding images or audio. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool InitMedia(const FAstralModelMediaDesc& Desc);

    /** Query image/audio support reported by the initialized media projector. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool GetMediaInfo(FAstralMediaInfo& OutInfo) const;

    /** Native handle for internal wrapper handoff. Do not cache it across Release. */
    uint64 GetHandle() const { return ModelHandle; }

    virtual void BeginDestroy() override;

private:
    uint64 ModelHandle = 0;
};
