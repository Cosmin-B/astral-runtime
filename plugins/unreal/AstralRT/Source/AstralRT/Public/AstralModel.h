#pragma once

#include "CoreMinimal.h"
#include "AstralTypes.h"

#include "AstralModel.generated.h"

/**
 * Model wrapper that owns an Astral model handle.
 */
UCLASS(BlueprintType)
class ASTRALRT_API UAstralModel : public UObject
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Load(const FAstralModelDesc& Desc);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    void Release();

    UFUNCTION(BlueprintPure, Category = "Astral")
    bool IsValid() const { return ModelHandle != 0; }

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool GetEmbeddingDim(int32& OutDim) const;

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool GetCaps(uint64& OutCaps) const;

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool GetLimits(FAstralModelLimits& OutLimits) const;

    uint64 GetHandle() const { return ModelHandle; }

    virtual void BeginDestroy() override;

private:
    uint64 ModelHandle = 0;
};
