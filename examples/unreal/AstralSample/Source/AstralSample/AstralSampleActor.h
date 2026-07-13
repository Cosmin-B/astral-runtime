#pragma once

#include "AstralTypes.h"
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

#include "AstralSampleActor.generated.h"

class UAstralEmbedder;
class UAstralModel;
class UAstralMultipleConversationsComponent;
class UAstralLocalKnowledgeComponent;
class UAstralStatefulNpcComponent;
class UAstralSession;
class UAstralStreamingChatComponent;

UCLASS()
class ASTRALSAMPLE_API AAstralSampleActor : public AActor {
  GENERATED_BODY()

public:
  AAstralSampleActor();

  UPROPERTY(EditAnywhere, Category = "Astral")
  FString BackendName = TEXT("mock");

  UPROPERTY(EditAnywhere, Category = "Astral")
  FString MemoryBackendName = TEXT("mock");

  UPROPERTY(EditAnywhere, Category = "Astral")
  FString MediaBackendName = TEXT("mock");

  UPROPERTY(EditAnywhere, Category = "Astral")
  FString ModelPath;

  UPROPERTY(EditAnywhere, Category = "Astral")
  FString EmbeddingModelPath;

  UPROPERTY(EditAnywhere, Category = "Astral")
  FString MediaPath;

  UPROPERTY(EditAnywhere, Category = "Astral")
  EAstralUnrealPathRoot MediaPathRoot = EAstralUnrealPathRoot::Raw;

  UPROPERTY(EditAnywhere, Category = "Astral")
  FString Prompt = TEXT("Say hello from Astral.");

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Astral|Examples")
  TObjectPtr<UAstralStreamingChatComponent> StreamingChat;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Astral|Examples")
  TObjectPtr<UAstralMultipleConversationsComponent> MultipleConversations;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Astral|Examples")
  TObjectPtr<UAstralStatefulNpcComponent> StatefulNpc;

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Astral|Examples")
  TObjectPtr<UAstralLocalKnowledgeComponent> LocalKnowledge;

  UFUNCTION(BlueprintCallable, Category = "Astral")
  void RunGenerationDemo();

  UFUNCTION(BlueprintCallable, Category = "Astral")
  void CancelStreamingDemo();

  UFUNCTION(BlueprintCallable, Category = "Astral")
  void RunEmbeddingDemo();

  UFUNCTION(BlueprintCallable, Category = "Astral")
  void RunMediaFeedDemo();

  UFUNCTION(BlueprintCallable, Category = "Astral")
  void RunPackagedMemorySourceDemo();

  UFUNCTION(BlueprintCallable, Category = "Astral")
  void RunSavedCacheDemo();

  UFUNCTION(BlueprintCallable, Category = "Astral")
  void RunMemorySearchDemo();

  UFUNCTION(BlueprintCallable, Category = "Astral")
  void RunErrorDemo();

protected:
  virtual void BeginPlay() override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
  UPROPERTY()
  TObjectPtr<UAstralModel> GenerationModel;

  UPROPERTY()
  TObjectPtr<UAstralSession> Session;

  UPROPERTY()
  TObjectPtr<UAstralModel> EmbeddingModel;

  UPROPERTY()
  TObjectPtr<UAstralEmbedder> Embedder;

  UPROPERTY()
  TObjectPtr<UAstralModel> MediaModel;

  UPROPERTY()
  TObjectPtr<UAstralSession> MediaSession;

  UPROPERTY()
  TObjectPtr<UAstralModel> ContentMemoryModel;

  UPROPERTY()
  TObjectPtr<UAstralModel> SavedCacheModel;

  void ApplyCommandLineOverrides();
  bool LoadGenerationModel();
  bool CreateSession();
  bool LoadMemoryModel(TObjectPtr<UAstralModel>& Model, const TArray<uint8>& ModelBytes,
                       const TCHAR* Label);
  void OnStreamBytes(TConstArrayView<uint8> Bytes);
};
