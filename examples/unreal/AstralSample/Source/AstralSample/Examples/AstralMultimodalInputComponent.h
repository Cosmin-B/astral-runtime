#pragma once

#include "AstralTypes.h"
#include "Components/ActorComponent.h"
#include "CoreMinimal.h"

#include "AstralMultimodalInputComponent.generated.h"

class UAstralEmbedder;
class UAstralModel;
class UAstralSession;
class UTexture2D;

/** Texture/audio generation inputs and multimodal embedding controls. */
UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))
class ASTRALSAMPLE_API UAstralMultimodalInputComponent : public UActorComponent {
  GENERATED_BODY()

public:
  UAstralMultimodalInputComponent();

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multimodal")
  FAstralModelDesc ModelDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multimodal")
  FAstralModelMediaDesc MediaDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multimodal")
  FAstralSessionDesc SessionDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multimodal")
  TObjectPtr<UTexture2D> Image;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multimodal")
  TArray<uint8> Pcm16Bytes;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multimodal")
  int32 AudioChannels = 1;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multimodal")
  int32 AudioSampleRate = 16000;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Multimodal")
  FString Prompt = TEXT("Describe the important gameplay signal.");

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Multimodal")
  FAstralMediaInfo MediaInfo;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Multimodal")
  TArray<float> Embedding;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Multimodal")
  FString Output;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Multimodal")
  FAstralOperationResult LastOperation;

  UFUNCTION(BlueprintCallable, Category = "Astral|Multimodal")
  bool RunImageGeneration();

  UFUNCTION(BlueprintCallable, Category = "Astral|Multimodal")
  bool RunAudioGeneration();

  UFUNCTION(BlueprintCallable, Category = "Astral|Multimodal")
  bool EmbedImage();

  UFUNCTION(BlueprintCallable, Category = "Astral|Multimodal")
  bool Cancel();

protected:
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
  UPROPERTY(Transient)
  TObjectPtr<UAstralModel> Model;

  UPROPERTY(Transient)
  TObjectPtr<UAstralSession> Session;

  UPROPERTY(Transient)
  TObjectPtr<UAstralEmbedder> Embedder;

  FAstralRequestRef ActiveRequest;
  int64 ActiveEmbeddingTicket = 0;

  bool EnsureModel();
  bool StartGeneration(const FAstralImageDesc* ImageDesc, const FAstralAudioDesc* AudioDesc);
  void OnStreamBytes(TConstArrayView<uint8> Bytes);
  void ReleaseSession();
  void ReleaseOwnedResources();
};
