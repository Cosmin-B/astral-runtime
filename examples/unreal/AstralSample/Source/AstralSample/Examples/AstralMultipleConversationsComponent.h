#pragma once

#include "AstralTypes.h"
#include "Components/ActorComponent.h"
#include "CoreMinimal.h"

#include "AstralMultipleConversationsComponent.generated.h"

class UAstralConversation;
class UAstralModel;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FAstralConversationText, int32, ConversationIndex,
                                             const FString&, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FAstralConversationFinished, int32,
                                               ConversationIndex, bool, bSuccess, int32, ErrorCode);

/** Independent nonblocking conversation slots sharing one configured model executor. */
UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))
class ASTRALSAMPLE_API UAstralMultipleConversationsComponent : public UActorComponent {
  GENERATED_BODY()

public:
  UAstralMultipleConversationsComponent();

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multiple Conversations")
  FAstralModelDesc ModelDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multiple Conversations")
  FAstralExecutorDesc ExecutorDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multiple Conversations")
  FAstralConversationDesc ConversationDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multiple Conversations",
            meta = (MultiLine = true))
  FString SystemPrompt = TEXT("Answer as a distinct, concise NPC.");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multiple Conversations")
  TArray<FString> Prompts;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Multiple Conversations")
  TArray<FString> Outputs;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Multiple Conversations")
  TArray<FAstralConversationStats> Stats;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Multiple Conversations")
  TArray<int32> LastErrorCodes;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Multiple Conversations")
  bool bRunning = false;

  UPROPERTY(BlueprintAssignable, Category = "Astral|Multiple Conversations")
  FAstralConversationText OnText;

  UPROPERTY(BlueprintAssignable, Category = "Astral|Multiple Conversations")
  FAstralConversationFinished OnConversationFinished;

  UFUNCTION(BlueprintCallable, Category = "Astral|Multiple Conversations")
  bool Run();

  UFUNCTION(BlueprintCallable, Category = "Astral|Multiple Conversations")
  void Cancel();

  UFUNCTION(BlueprintCallable, Category = "Astral|Multiple Conversations")
  bool CancelConversation(int32 Index);

  UFUNCTION(BlueprintCallable, Category = "Astral|Multiple Conversations")
  bool ResetConversation(int32 Index, const FString& NewPrompt);

protected:
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                             FActorComponentTickFunction* ThisTickFunction) override;

private:
  UPROPERTY(Transient)
  TObjectPtr<UAstralModel> Model;

  UPROPERTY(Transient)
  TArray<TObjectPtr<UAstralConversation>> Conversations;

  UPROPERTY(Transient)
  TArray<FAstralRequestRef> Requests;

  TArray<TArray<uint8>> StreamBuffers;
  TArray<bool> Terminal;
  int32 ActiveConversationCount = 0;

  bool EnsureModel();
  bool StartConversation(int32 Index, const FString& UserPrompt, bool bReset);
  void AbortPartialRun(int32 StartedCount);
  void FinishConversation(int32 Index, bool bSuccess, int32 ErrorCode);
};
