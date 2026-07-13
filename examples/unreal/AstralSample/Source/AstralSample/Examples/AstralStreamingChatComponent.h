#pragma once

#include "AstralTypes.h"
#include "Components/ActorComponent.h"
#include "CoreMinimal.h"

#include "AstralStreamingChatComponent.generated.h"

class UAstralModel;
class UAstralSession;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FAstralStreamingChatText, const FString&, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FAstralStreamingChatFinished, bool, bSuccess, int32,
                                             ErrorCode);

/** One model and session wired as a reusable streaming-chat gameplay component. */
UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))
class ASTRALSAMPLE_API UAstralStreamingChatComponent : public UActorComponent {
  GENERATED_BODY()

public:
  UAstralStreamingChatComponent();

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Streaming Chat")
  FAstralModelDesc ModelDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Streaming Chat")
  FAstralSessionDesc SessionDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Streaming Chat",
            meta = (MultiLine = true))
  FString SystemPrompt = TEXT("Answer as a concise in-game character.");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Streaming Chat",
            meta = (MultiLine = true))
  FString Prompt = TEXT("Say hello from Astral.");

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Streaming Chat")
  FString Output;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Streaming Chat")
  bool bRunning = false;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Streaming Chat")
  int32 LastErrorCode = static_cast<int32>(EAstralError::OK);

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Streaming Chat")
  FAstralStats LastStats;

  UPROPERTY(BlueprintAssignable, Category = "Astral|Streaming Chat")
  FAstralStreamingChatText OnText;

  UPROPERTY(BlueprintAssignable, Category = "Astral|Streaming Chat")
  FAstralStreamingChatFinished OnFinished;

  UFUNCTION(BlueprintCallable, Category = "Astral|Streaming Chat")
  bool Run();

  UFUNCTION(BlueprintCallable, Category = "Astral|Streaming Chat")
  bool Cancel();

protected:
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                             FActorComponentTickFunction* ThisTickFunction) override;

private:
  UPROPERTY(Transient)
  TObjectPtr<UAstralModel> Model;

  UPROPERTY(Transient)
  TObjectPtr<UAstralSession> Session;

  UPROPERTY(Transient)
  FAstralRequestRef ActiveRequest;

  bool EnsureSession();
  void OnStreamBytes(TConstArrayView<uint8> Bytes);
  void Finish(bool bSuccess, int32 ErrorCode);
};
