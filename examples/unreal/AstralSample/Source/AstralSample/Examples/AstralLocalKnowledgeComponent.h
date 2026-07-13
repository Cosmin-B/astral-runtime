#pragma once

#include "AstralTypes.h"
#include "Components/ActorComponent.h"
#include "CoreMinimal.h"

#include "AstralLocalKnowledgeComponent.generated.h"

class UAstralEmbedder;
class UAstralModel;

/** Chunk, embed, search, and persist a small local gameplay knowledge index. */
UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))
class ASTRALSAMPLE_API UAstralLocalKnowledgeComponent : public UActorComponent {
  GENERATED_BODY()

public:
  UAstralLocalKnowledgeComponent();

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Local Knowledge")
  FAstralModelDesc ModelDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Local Knowledge",
            meta = (MultiLine = true))
  FString DocumentText = TEXT("The north gate opens at dawn. The ferry leaves at sunset.");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Local Knowledge")
  FString Query = TEXT("When does the ferry leave?");

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Local Knowledge")
  FAstralChunkerDesc ChunkerDesc;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Local Knowledge")
  int32 TopK = 3;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Local Knowledge")
  FString SnapshotFileName = TEXT("Astral/local-knowledge.index");

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Local Knowledge")
  TArray<FAstralChunkRange> Chunks;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Local Knowledge")
  TArray<FAstralMemorySearchResult> SearchResults;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Local Knowledge")
  TArray<FString> MatchedText;

  UPROPERTY(BlueprintReadOnly, Category = "Astral|Local Knowledge")
  FAstralOperationResult LastOperation;

  UFUNCTION(BlueprintCallable, Category = "Astral|Local Knowledge")
  bool BuildIndex();

  UFUNCTION(BlueprintCallable, Category = "Astral|Local Knowledge")
  bool Search();

  UFUNCTION(BlueprintCallable, Category = "Astral|Local Knowledge")
  bool SaveIndex();

  UFUNCTION(BlueprintCallable, Category = "Astral|Local Knowledge")
  bool LoadIndex();

protected:
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
  UPROPERTY(Transient)
  TObjectPtr<UAstralModel> Model;

  UPROPERTY(Transient)
  TObjectPtr<UAstralEmbedder> Embedder;

  int64 MemoryHandle = 0;
  FAstralMemoryIndexDesc IndexDesc;

  bool EnsureEmbedder();
  FString SnapshotPath() const;
  void ReleaseIndex();
  void ReleaseOwnedResources();
};
