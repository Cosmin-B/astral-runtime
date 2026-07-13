#pragma once

#include "AstralTypes.h"
#include "Components/ActorComponent.h"
#include "CoreMinimal.h"

#include "AstralStatefulNpcComponent.generated.h"

class UAstralModel;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FAstralNpcText, const FString&, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FAstralNpcFinished, bool, bSuccess, int32, ErrorCode);

/** Persistent agent state, streamed chat, and one structured movement tool. */
UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))
class ASTRALSAMPLE_API UAstralStatefulNpcComponent : public UActorComponent {
  GENERATED_BODY()

public:
  UAstralStatefulNpcComponent();

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Stateful NPC")
  FAstralModelDesc ModelDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Stateful NPC")
  FAstralAgentDesc AgentDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Stateful NPC",
            meta = (MultiLine = true))
  FString SystemPrompt = TEXT("You are the harbor master. Keep answers concise.");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Stateful NPC",
            meta = (MultiLine = true))
  FString Summary = TEXT("The player has just arrived at the harbor.");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Stateful NPC",
            meta = (MultiLine = true))
  FString MemoryContext = TEXT("The lighthouse is north. The ferry leaves at sunset.");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Stateful NPC")
  FString UserMessage = TEXT("Where should I go first?");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Stateful NPC")
  FString HistoryFileName = TEXT("Astral/stateful-npc.history");

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Stateful NPC")
  FString Output;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Stateful NPC")
  FAstralToolCallResult LastToolCall;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Stateful NPC")
  FAstralAgentChatResult LastChat;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Stateful NPC")
  FAstralOperationResult LastOperation;

  UPROPERTY(BlueprintAssignable, Category = "Astral|Stateful NPC")
  FAstralNpcText OnText;

  UPROPERTY(BlueprintAssignable, Category = "Astral|Stateful NPC")
  FAstralNpcFinished OnFinished;

  UFUNCTION(BlueprintCallable, Category = "Astral|Stateful NPC")
  bool Ask();

  UFUNCTION(BlueprintCallable, Category = "Astral|Stateful NPC")
  bool Cancel();

  UFUNCTION(BlueprintCallable, Category = "Astral|Stateful NPC")
  bool SaveHistory();

  UFUNCTION(BlueprintCallable, Category = "Astral|Stateful NPC")
  bool LoadHistory();

  UFUNCTION(BlueprintCallable, Category = "Astral|Stateful NPC")
  bool ClearHistory();

protected:
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                             FActorComponentTickFunction* ThisTickFunction) override;

private:
  UPROPERTY(Transient)
  TObjectPtr<UAstralModel> Model;

  int64 AgentHandle = 0;
  int64 ToolsetHandle = 0;
  FAstralRequestRef ActiveRequest;
  bool bRunning = false;

  bool EnsureAgent();
  FString HistoryPath() const;
  void Finish(bool bSuccess, int32 ErrorCode);
  void ReleaseOwnedResources();
};
