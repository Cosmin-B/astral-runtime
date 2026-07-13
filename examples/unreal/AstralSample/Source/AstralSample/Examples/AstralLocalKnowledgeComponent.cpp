#include "AstralLocalKnowledgeComponent.h"

#include "AstralBlueprintLibrary.h"
#include "AstralEmbedder.h"
#include "AstralModel.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

UAstralLocalKnowledgeComponent::UAstralLocalKnowledgeComponent() {
  PrimaryComponentTick.bCanEverTick = false;
  ModelDesc.BackendName = TEXT("mock");
  ModelDesc.bEmbeddingsOnly = true;
  ChunkerDesc.Mode = EAstralChunkMode::Word;
  ChunkerDesc.MaxUnits = 12;
  ChunkerDesc.OverlapUnits = 2;
  ChunkerDesc.DocumentId = 1;
  ChunkerDesc.GroupId = 1;
}

FString UAstralLocalKnowledgeComponent::SnapshotPath() const {
  return FPaths::Combine(FPaths::ProjectSavedDir(), SnapshotFileName);
}

bool UAstralLocalKnowledgeComponent::EnsureEmbedder() {
  if (Embedder != nullptr && Embedder->IsValid()) {
    return true;
  }
  Model = NewObject<UAstralModel>(this);
  if (!Model->Load(ModelDesc)) {
    return false;
  }
  int64 Caps = 0;
  if (!Model->GetCaps(Caps) || !UAstralBlueprintLibrary::HasEmbeddings(Caps)) {
    LastOperation.ErrorCode = static_cast<int32>(EAstralError::Unsupported);
    return false; // Semantic retrieval requires an embeddings-capable model.
  }
  Embedder = NewObject<UAstralEmbedder>(this);
  return Embedder->Create(Model);
}

bool UAstralLocalKnowledgeComponent::BuildIndex() {
  if (DocumentText.IsEmpty() || !EnsureEmbedder()) {
    return false;
  }
  int32 ErrorCode = 0;
  if (!UAstralBlueprintLibrary::ChunkText(DocumentText, ChunkerDesc, Chunks, ErrorCode) ||
      Chunks.IsEmpty()) {
    LastOperation.ErrorCode = ErrorCode;
    return false;
  }

  ReleaseIndex();
  IndexDesc.Dimension = Embedder->GetDim();
  IndexDesc.Capacity = Chunks.Num();
  IndexDesc.Metric = EAstralMemoryMetric::Cosine;
  IndexDesc.IndexKind = EAstralMemoryIndexKind::Flat;
  IndexDesc.StorageKind = EAstralMemoryStorageKind::F32;
  LastOperation = UAstralBlueprintLibrary::CreateMemoryIndexResult(IndexDesc);
  if (!LastOperation.bSuccess) {
    return false;
  }
  MemoryHandle = LastOperation.Handle;

  TArray<FAstralMemoryRecord> Records;
  TArray<float> Vectors;
  Records.Reserve(Chunks.Num());
  Vectors.Reserve(Chunks.Num() * IndexDesc.Dimension);
  for (int32 Index = 0; Index < Chunks.Num(); ++Index) {
    FString ChunkText;
    LastOperation =
        UAstralBlueprintLibrary::CopyChunkTextResult(DocumentText, Chunks[Index], ChunkText);
    if (!LastOperation.bSuccess) {
      ReleaseIndex();
      return false;
    }
    TArray<float> Vector;
    if (!Embedder->EmbedText(ChunkText, Vector)) {
      ReleaseIndex();
      return false;
    }
    FAstralMemoryRecord Record{};
    LastOperation = UAstralBlueprintLibrary::MakeMemoryRecordFromChunkResult(Chunks[Index],
                                                                             Index + 1, 0, Record);
    if (!LastOperation.bSuccess) {
      ReleaseIndex();
      return false;
    }
    Records.Add(Record);
    Vectors.Append(Vector);
  }
  LastOperation = UAstralBlueprintLibrary::AddMemoryBatchResult(MemoryHandle, Records, Vectors,
                                                                IndexDesc.Dimension);
  return LastOperation.bSuccess;
}

bool UAstralLocalKnowledgeComponent::Search() {
  if (MemoryHandle == 0 || !EnsureEmbedder()) {
    return false;
  }
  TArray<float> QueryVector;
  if (!Embedder->EmbedText(Query, QueryVector)) {
    return false;
  }
  SearchResults.Reset();
  MatchedText.Reset();
  LastOperation = UAstralBlueprintLibrary::SearchMemoryIndexResult(
      MemoryHandle, QueryVector, FMath::Clamp(TopK, 1, Chunks.Num()), ChunkerDesc.GroupId,
      SearchResults);
  if (!LastOperation.bSuccess) {
    return false;
  }
  for (const FAstralMemorySearchResult& Result : SearchResults) {
    if (!Chunks.IsValidIndex(Result.ChunkId)) {
      continue;
    }
    FString Text;
    LastOperation =
        UAstralBlueprintLibrary::CopyChunkTextResult(DocumentText, Chunks[Result.ChunkId], Text);
    if (LastOperation.bSuccess) {
      MatchedText.Add(Text);
    }
  }
  return LastOperation.bSuccess;
}

bool UAstralLocalKnowledgeComponent::SaveIndex() {
  if (MemoryHandle == 0) {
    return false;
  }
  TArray<uint8> Bytes;
  LastOperation = UAstralBlueprintLibrary::SaveMemoryIndexResult(MemoryHandle, Bytes);
  if (!LastOperation.bSuccess) {
    return false;
  }
  IFileManager::Get().MakeDirectory(*FPaths::GetPath(SnapshotPath()), true);
  return FFileHelper::SaveArrayToFile(Bytes, *SnapshotPath());
}

bool UAstralLocalKnowledgeComponent::LoadIndex() {
  if (!EnsureEmbedder()) {
    return false;
  }
  int32 ErrorCode = 0;
  if (!UAstralBlueprintLibrary::ChunkText(DocumentText, ChunkerDesc, Chunks, ErrorCode)) {
    return false;
  }
  TArray<uint8> Bytes;
  if (!FFileHelper::LoadFileToArray(Bytes, *SnapshotPath())) {
    return false;
  }
  ReleaseIndex();
  IndexDesc.Dimension = Embedder->GetDim();
  IndexDesc.Capacity = FMath::Max(1, Chunks.Num());
  LastOperation = UAstralBlueprintLibrary::LoadMemoryIndexResult(IndexDesc, Bytes);
  MemoryHandle = LastOperation.bSuccess ? LastOperation.Handle : 0;
  return LastOperation.bSuccess;
}

void UAstralLocalKnowledgeComponent::ReleaseIndex() {
  if (MemoryHandle != 0) {
    UAstralBlueprintLibrary::DestroyMemoryIndex(MemoryHandle);
    MemoryHandle = 0;
  }
}

void UAstralLocalKnowledgeComponent::ReleaseOwnedResources() {
  ReleaseIndex();
  if (Embedder != nullptr) {
    Embedder->Destroy();
    Embedder = nullptr;
  }
  if (Model != nullptr) {
    Model->Release();
    Model = nullptr;
  }
}

void UAstralLocalKnowledgeComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  ReleaseOwnedResources();
  Super::EndPlay(EndPlayReason);
}
