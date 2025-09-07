#pragma once

#include "CoreMinimal.h"
#include "AstralTypes.h"
#include "Containers/ArrayView.h"
#include "Containers/Ticker.h"
#include "Containers/StringView.h"

#include "AstralSession.generated.h"

class UAstralModel;

DECLARE_MULTICAST_DELEGATE_OneParam(FAstralStreamBytesNative, TConstArrayView<uint8>);
DECLARE_MULTICAST_DELEGATE_OneParam(FAstralStreamTextNative, FStringView);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FAstralStreamBytesReceived, const TArray<uint8>&, Bytes);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FAstralTokenReceived, const FString&, Token);

/**
 * Session wrapper that owns an Astral session handle and provides pull-based streaming.
 */
UCLASS(BlueprintType)
class ASTRALRT_API UAstralSession : public UObject
{
    GENERATED_BODY()

public:
    UAstralSession();

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Create(UAstralModel* Model, const FAstralSessionDesc& Desc);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool FeedPrompt(const FString& Prompt, bool bFinalize = true);

    bool FeedPromptRaw(TConstArrayView<uint8> Utf8Data, bool bFinalize = true);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Decode();

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Cancel();

    /** Returns: 0 (OK), -7 (canceled), -4 (timeout), or other error code. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    int32 Wait(uint32 TimeoutMs = 0);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Reset(const FAstralSessionDesc& Desc);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool SetSampler(const FAstralSamplerDesc& Desc);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool StopClear();

    bool StopAddUtf8Bytes(TConstArrayView<uint8> Utf8Data);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool StopAddString(const FString& Utf8Text);

    int32 StreamRead(TArray<uint8>& OutBuffer, uint32 TimeoutMs = 0);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    FString StreamReadString(uint32 TimeoutMs = 0);

    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralStats GetStats() const;

    UFUNCTION(BlueprintPure, Category = "Astral")
    bool IsValid() const { return SessionHandle != 0; }

    /** Raw bytes streaming (UTF-8). Recommended for low-overhead C++ consumers. */
    FAstralStreamBytesNative& OnStreamBytesNative() { return StreamBytesNative; }

    /** Text streaming view (UTF-8 decoded). Avoids FString allocation for C++ consumers. */
    FAstralStreamTextNative& OnStreamTextNative() { return StreamTextNative; }

    /** Raw bytes streaming (UTF-8). Convenience for Blueprints; may copy/marshal data. */
    UPROPERTY(BlueprintAssignable, Category = "Astral")
    FAstralStreamBytesReceived OnBytesReceived;

    /** Text streaming (decoded). Convenience for Blueprints; allocates an FString per tick when bound. */
    UPROPERTY(BlueprintAssignable, Category = "Astral")
    FAstralTokenReceived OnTokenReceived;

    virtual void BeginDestroy() override;

private:
    uint64 SessionHandle = 0;
    uint64 ModelHandle = 0;

    TArray<uint8> TokenBuffer;
    TArray<uint8> TickUtf8Buffer;
    FString TickTextScratch;

    FAstralStreamBytesNative StreamBytesNative;
    FAstralStreamTextNative StreamTextNative;

    FTSTicker::FDelegateHandle TickerHandle;
    bool TickStream(float DeltaTime);
    void UpdateTicker(bool bEnable);
};
