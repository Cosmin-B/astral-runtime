#pragma once

#include "AstralSession.h"
#include "AstralTypes.h"
#include "Containers/Ticker.h"
#include "CoreMinimal.h"

#include "AstralConversation.generated.h"

class UAstralModel;

/** UObject owner for one continuous-batching conversation slot. */
UCLASS(BlueprintType)
class ASTRALRT_API UAstralConversation : public UObject {
  GENERATED_BODY()

public:
  UAstralConversation();

  /** Create a conversation from a model whose executor is already configured. */
  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool Create(UAstralModel* Model, const FAstralConversationDesc& Desc);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool SetSystemPrompt(const FString& Prompt);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool FeedPrompt(const FString& Prompt, bool bFinalize = true);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool FeedImage(const FAstralImageDesc& Image, bool bFinalize = true);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool FeedAudio(const FAstralAudioDesc& Audio, bool bFinalize = true);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool Decode();

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool Cancel();

  /** Return ASTRAL_OK, ASTRAL_E_CANCELED, ASTRAL_E_TIMEOUT, or another native error. */
  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  int32 Wait(int32 TimeoutMs = 0);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool Reset(const FAstralConversationDesc& Desc);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool SetSampler(const FAstralSamplerDesc& Desc);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool StopClear();

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool StopAddString(const FString& Utf8Text);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool SetGrammarGbnf(const FString& Grammar, const FString& RootSymbol);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool SetGrammarJsonSchema(const FString& JsonSchema);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool ClearGrammar();

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool SetToolset(int64 ToolsetHandle,
                  EAstralToolChoiceMode ChoiceMode = EAstralToolChoiceMode::Auto);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  bool ClearToolset();

  /** Read UTF-8 bytes into a caller-sized buffer. Negative values are native errors. */
  int32 StreamRead(TArray<uint8>& OutBuffer, uint32 TimeoutMs = 0);

  /** Read one byte chunk as FString. Empty may mean timeout, end of stream, or error. */
  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  FString StreamReadString(int32 TimeoutMs = 0);

  UFUNCTION(BlueprintCallable, Category = "Astral|Conversations")
  FAstralConversationStats GetStats() const;

  UFUNCTION(BlueprintPure, Category = "Astral|Conversations")
  bool IsValid() const;

  uint64 GetHandle() const { return IsValid() ? ConversationHandle : 0; }

  FAstralStreamBytesNative& OnStreamBytesNative() { return StreamBytesNative; }
  FAstralStreamTextNative& OnStreamTextNative() { return StreamTextNative; }

  UPROPERTY(BlueprintAssignable, Category = "Astral|Conversations")
  FAstralStreamBytesReceived OnBytesReceived;

  UPROPERTY(BlueprintAssignable, Category = "Astral|Conversations")
  FAstralTokenReceived OnTokenReceived;

  virtual void BeginDestroy() override;

private:
  bool IsCurrentRuntimeGeneration() const;
  bool TickStream(float DeltaTime);
  void UpdateTicker(bool bEnable);

  uint64 ConversationHandle = 0;
  uint64 ModelHandle = 0;
  uint64 RuntimeGeneration = 0;

  TArray<uint8> TokenBuffer;
  TArray<uint8> TickUtf8Buffer;
  FString TickTextScratch;

  FAstralStreamBytesNative StreamBytesNative;
  FAstralStreamTextNative StreamTextNative;
  FTSTicker::FDelegateHandle TickerHandle;
};
