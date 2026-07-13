#pragma once

#include "AstralTypes.h"
#include "Components/ActorComponent.h"
#include "CoreMinimal.h"

#include "AstralCharacterVariantsComponent.generated.h"

class UAstralModel;
class UAstralSession;

/** Prompt caching, adapter variants, grammar, stops, and token diagnostics. */
UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))
class ASTRALSAMPLE_API UAstralCharacterVariantsComponent : public UActorComponent {
  GENERATED_BODY()

public:
  UAstralCharacterVariantsComponent();

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants")
  FAstralModelDesc ModelDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants")
  FAstralSessionDesc SessionDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants")
  FAstralPromptCacheDesc CacheDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants")
  FAstralAdapterDesc AdapterDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants")
  float AdapterScale = 1.0f;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants",
            meta = (MultiLine = true))
  FString SystemPrompt = TEXT("You are a cautious scout. Return one JSON object.");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants",
            meta = (MultiLine = true))
  FString UserPrompt = TEXT("Report what is beyond the ridge.");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants")
  FString StopSequence = TEXT("</turn>");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants",
            meta = (MultiLine = true))
  FString JsonSchema = TEXT("{\"type\":\"object\",\"properties\":{")
      TEXT("\"line\":{\"type\":\"string\"}},\"required\":[\"line\"]}");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Character Variants")
  FString CacheFileName = TEXT("Astral/character-variants.cache");

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Character Variants")
  FString Output;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Character Variants")
  FString TokenRoundTrip;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Character Variants")
  int32 PromptTokenCount = 0;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Character Variants")
  FAstralOperationResult LastOperation;

  UFUNCTION(BlueprintCallable, Category = "Astral|Character Variants")
  bool PreparePromptCache();

  UFUNCTION(BlueprintCallable, Category = "Astral|Character Variants")
  bool LoadPromptCache();

  UFUNCTION(BlueprintCallable, Category = "Astral|Character Variants")
  bool Run();

  UFUNCTION(BlueprintCallable, Category = "Astral|Character Variants")
  bool SetVariantScale(float NewScale);

  UFUNCTION(BlueprintCallable, Category = "Astral|Character Variants")
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

  int64 CacheHandle = 0;
  int64 AdapterHandle = 0;
  FAstralRequestRef ActiveRequest;
  bool bRunning = false;

  bool EnsureModel();
  FString CachePath() const;
  void OnStreamBytes(TConstArrayView<uint8> Bytes);
  void ReleaseSession();
  void ReleaseOwnedResources();
};
