#include "AstralSampleActor.h"

#include "AstralBlueprintLibrary.h"
#include "AstralEmbedder.h"
#include "AstralMediaLibrary.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "AstralTypes.h"
#include "Examples/AstralMultipleConversationsComponent.h"
#include "Examples/AstralStreamingChatComponent.h"
#include "astral_rt.h"

#include "Containers/ArrayView.h"
#include "Containers/StringConv.h"
#include "Engine/Texture2D.h"
#include "GenericPlatform/GenericPlatformMisc.h"
#include "HAL/FileManager.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "PixelFormat.h"

DEFINE_LOG_CATEGORY_STATIC(LogAstralSample, Log, All);

static EAstralUnrealPathRoot AstralSampleParsePathRoot(const FString& Value,
                                                       EAstralUnrealPathRoot Fallback) {
  if (Value.Equals(TEXT("Raw"), ESearchCase::IgnoreCase)) {
    return EAstralUnrealPathRoot::Raw;
  }
  if (Value.Equals(TEXT("ProjectContent"), ESearchCase::IgnoreCase)) {
    return EAstralUnrealPathRoot::ProjectContent;
  }
  if (Value.Equals(TEXT("ProjectSaved"), ESearchCase::IgnoreCase)) {
    return EAstralUnrealPathRoot::ProjectSaved;
  }
  if (Value.Equals(TEXT("ProjectPersistentDownload"), ESearchCase::IgnoreCase)) {
    return EAstralUnrealPathRoot::ProjectPersistentDownload;
  }
  return Fallback;
}

static const TCHAR* AstralSamplePathRootName(EAstralUnrealPathRoot Root) {
  switch (Root) {
  case EAstralUnrealPathRoot::ProjectContent:
    return TEXT("ProjectContent");
  case EAstralUnrealPathRoot::ProjectSaved:
    return TEXT("ProjectSaved");
  case EAstralUnrealPathRoot::ProjectPersistentDownload:
    return TEXT("ProjectPersistentDownload");
  case EAstralUnrealPathRoot::Raw:
  default:
    return TEXT("Raw");
  }
}

AAstralSampleActor::AAstralSampleActor() {
  PrimaryActorTick.bCanEverTick = false;
  StreamingChat =
      CreateDefaultSubobject<UAstralStreamingChatComponent>(TEXT("AstralStreamingChat"));
  MultipleConversations = CreateDefaultSubobject<UAstralMultipleConversationsComponent>(
      TEXT("AstralMultipleConversations"));
}

void AAstralSampleActor::BeginPlay() {
  Super::BeginPlay();

  ApplyCommandLineOverrides();

  RunGenerationDemo();
  CancelStreamingDemo();
  RunEmbeddingDemo();
  RunMediaFeedDemo();
  RunPackagedMemorySourceDemo();
  RunSavedCacheDemo();
  RunMemorySearchDemo();
  RunErrorDemo();

  if (FParse::Param(FCommandLine::Get(), TEXT("AstralSampleAutoQuit"))) {
    FGenericPlatformMisc::RequestExit(false);
  }
}

void AAstralSampleActor::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (Session != nullptr) {
    Session->Cancel();
    Session = nullptr;
  }
  if (GenerationModel != nullptr) {
    GenerationModel->Release();
    GenerationModel = nullptr;
  }
  if (Embedder != nullptr) {
    Embedder->Destroy();
    Embedder = nullptr;
  }
  if (EmbeddingModel != nullptr) {
    EmbeddingModel->Release();
    EmbeddingModel = nullptr;
  }
  if (MediaSession != nullptr) {
    MediaSession->Cancel();
    MediaSession = nullptr;
  }
  if (MediaModel != nullptr) {
    MediaModel->Release();
    MediaModel = nullptr;
  }
  if (ContentMemoryModel != nullptr) {
    ContentMemoryModel->Release();
    ContentMemoryModel = nullptr;
  }
  if (SavedCacheModel != nullptr) {
    SavedCacheModel->Release();
    SavedCacheModel = nullptr;
  }

  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: clean shutdown"));

  Super::EndPlay(EndPlayReason);
}

void AAstralSampleActor::ApplyCommandLineOverrides() {
  const TCHAR* CommandLine = FCommandLine::Get();
  FString OverrideValue;

  if (FParse::Value(CommandLine, TEXT("AstralBackend="), OverrideValue)) {
    BackendName = OverrideValue;
  }
  if (FParse::Value(CommandLine, TEXT("AstralMemoryBackend="), OverrideValue)) {
    MemoryBackendName = OverrideValue;
  }
  if (FParse::Value(CommandLine, TEXT("AstralMediaBackend="), OverrideValue)) {
    MediaBackendName = OverrideValue;
  }
  if (FParse::Value(CommandLine, TEXT("AstralModel="), OverrideValue)) {
    ModelPath = OverrideValue;
  }
  if (FParse::Value(CommandLine, TEXT("AstralEmbeddingModel="), OverrideValue)) {
    EmbeddingModelPath = OverrideValue;
  }
  if (FParse::Value(CommandLine, TEXT("AstralMediaPath="), OverrideValue)) {
    MediaPath = OverrideValue;
  }
  if (FParse::Value(CommandLine, TEXT("AstralMediaPathRoot="), OverrideValue)) {
    MediaPathRoot = AstralSampleParsePathRoot(OverrideValue, MediaPathRoot);
  }
  if (FParse::Value(CommandLine, TEXT("AstralPrompt="), OverrideValue)) {
    Prompt = OverrideValue;
  }

  UE_LOG(LogAstralSample, Display,
         TEXT("Astral sample: backend=%s memory_backend=%s media_backend=%s model=%s "
              "embedding_model=%s media_path=%s media_path_root=%s"),
         *BackendName, *MemoryBackendName, *MediaBackendName,
         ModelPath.IsEmpty() ? TEXT("<mock/default>") : *ModelPath,
         EmbeddingModelPath.IsEmpty() ? TEXT("<model/default>") : *EmbeddingModelPath,
         MediaPath.IsEmpty() ? TEXT("<none>") : *MediaPath,
         AstralSamplePathRootName(MediaPathRoot));
}

bool AAstralSampleActor::LoadGenerationModel() {
  GenerationModel = NewObject<UAstralModel>(this);

  FAstralModelDesc Desc{};
  Desc.SourceKind = EAstralModelSourceKind::Path;
  Desc.ModelPath = ModelPath;
  Desc.BackendName = BackendName;
  Desc.ContextSize = 256;
  Desc.BatchSize = 64;

  if (!GenerationModel->Load(Desc)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: model load failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return false;
  }

  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: generation model loaded backend=%s"),
         *BackendName);
  return true;
}

bool AAstralSampleActor::LoadMemoryModel(TObjectPtr<UAstralModel>& Model,
                                         const TArray<uint8>& ModelBytes, const TCHAR* Label) {
  Model = NewObject<UAstralModel>(this);

  FAstralModelDesc Desc{};
  Desc.SourceKind = EAstralModelSourceKind::Memory;
  Desc.ModelBytes = ModelBytes;
  Desc.BackendName = MemoryBackendName;
  Desc.ContextSize = 128;

  if (!Model->Load(Desc)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: %s memory model load failed: %s"), Label,
           UTF8_TO_TCHAR(astral_last_error()));
    return false;
  }

  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: %s memory model loaded from %d bytes"),
         Label, ModelBytes.Num());
  return true;
}

bool AAstralSampleActor::CreateSession() {
  Session = NewObject<UAstralSession>(this);

  FAstralSessionDesc Desc{};
  Desc.MaxTokens = 64;
  Desc.Temperature = 0.0f;
  Desc.TopK = 0;
  Desc.TopP = 1.0f;
  Desc.bStreamEnabled = true;
  Desc.Seed = 7;

  if (!Session->Create(GenerationModel, Desc)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: session create failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return false;
  }

  Session->OnStreamBytesNative().AddUObject(this, &AAstralSampleActor::OnStreamBytes);
  return true;
}

void AAstralSampleActor::RunGenerationDemo() {
  if (!LoadGenerationModel() || !CreateSession()) {
    return;
  }

  if (!Session->FeedPrompt(Prompt, true)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: prompt feed failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }
  if (!Session->Decode()) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: decode failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }

  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: generation decode started"));
}

void AAstralSampleActor::CancelStreamingDemo() {
  if (Session == nullptr || !Session->IsValid()) {
    UE_LOG(LogAstralSample, Warning, TEXT("Astral sample: no active session to cancel"));
    return;
  }

  if (!Session->Cancel()) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: cancel failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }

  const int32 WaitResult = Session->Wait(1000);
  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: canceled stream wait result %d"),
         WaitResult);
}

void AAstralSampleActor::RunEmbeddingDemo() {
  EmbeddingModel = NewObject<UAstralModel>(this);

  FAstralModelDesc Desc{};
  Desc.SourceKind = EAstralModelSourceKind::Path;
  Desc.ModelPath = EmbeddingModelPath.IsEmpty() ? ModelPath : EmbeddingModelPath;
  Desc.BackendName = BackendName;
  Desc.ContextSize = 128;
  Desc.bEmbeddingsOnly = true;

  if (!EmbeddingModel->Load(Desc)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: embedding model load failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }

  Embedder = NewObject<UAstralEmbedder>(this);
  if (!Embedder->Create(EmbeddingModel)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: embedder create failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }

  FTCHARToUTF8 Utf8(TEXT("sample vector"));
  TArray<uint8> Bytes;
  Bytes.Append(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());

  TArray<float> Vector;
  if (!Embedder->EmbedUtf8Bytes(Bytes, Vector)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: embedding failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }

  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: embedding dimension %d"), Vector.Num());
}

void AAstralSampleActor::RunMediaFeedDemo() {
  MediaModel = NewObject<UAstralModel>(this);

  FAstralModelDesc ModelDesc{};
  ModelDesc.SourceKind = EAstralModelSourceKind::Path;
  ModelDesc.ModelPath = ModelPath;
  ModelDesc.BackendName = MediaBackendName;
  ModelDesc.ContextSize = 128;

  if (!MediaModel->Load(ModelDesc)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media model load failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }

  if (!MediaPath.IsEmpty()) {
    FAstralModelMediaDesc MediaDesc{};
    MediaDesc.SourceKind = EAstralModelSourceKind::Path;
    MediaDesc.MediaPath = MediaPath;
    MediaDesc.MediaPathRoot = MediaPathRoot;
    if (!MediaModel->InitMedia(MediaDesc)) {
      UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media projector init failed: %s"),
             UTF8_TO_TCHAR(astral_last_error()));
      return;
    }
    UE_LOG(LogAstralSample, Display,
           TEXT("Astral sample: media projector initialized path=%s root=%s"), *MediaPath,
           AstralSamplePathRootName(MediaPathRoot));
  }

  MediaSession = NewObject<UAstralSession>(this);

  FAstralSessionDesc SessionDesc{};
  SessionDesc.MaxTokens = 16;
  SessionDesc.Temperature = 0.0f;
  SessionDesc.TopK = 0;
  SessionDesc.TopP = 1.0f;
  SessionDesc.bStreamEnabled = false;
  SessionDesc.Seed = 17;

  if (!MediaSession->Create(MediaModel, SessionDesc)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media session create failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }

  TArray<uint8> RgbaBytes;
  RgbaBytes.SetNumZeroed(2 * 2 * 4);

  FAstralImageDesc Image{};
  if (!UAstralMediaLibrary::MakeRGBA8ImageFromBytes(RgbaBytes, 2, 2, Image)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media image descriptor failed"));
    return;
  }

  UTexture2D* Texture = UTexture2D::CreateTransient(2, 1, PF_B8G8R8A8);
  if (Texture == nullptr || Texture->GetPlatformData() == nullptr ||
      Texture->GetPlatformData()->Mips.Num() == 0) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media texture allocation failed"));
    return;
  }

  FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
  void* Locked = Mip.BulkData.Lock(LOCK_READ_WRITE);
  if (Locked == nullptr) {
    Mip.BulkData.Unlock();
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media texture lock failed"));
    return;
  }
  uint8* Bgra = static_cast<uint8*>(Locked);
  Bgra[0] = 30;
  Bgra[1] = 20;
  Bgra[2] = 10;
  Bgra[3] = 255;
  Bgra[4] = 60;
  Bgra[5] = 50;
  Bgra[6] = 40;
  Bgra[7] = 128;
  Mip.BulkData.Unlock();

  FAstralImageDesc TextureImage{};
  if (!UAstralMediaLibrary::MakeRGBA8ImageFromTexture(Texture, TextureImage)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media texture descriptor failed"));
    return;
  }

  TArray<uint8> PcmBytes;
  PcmBytes.SetNumZeroed(4 * static_cast<int32>(sizeof(int16)));

  FAstralAudioDesc Audio{};
  if (!UAstralMediaLibrary::MakePCM16AudioFromBytes(PcmBytes, 1, 16000, Audio)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media audio descriptor failed"));
    return;
  }

  if (!MediaSession->FeedImage(Image, false)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media image feed failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }
  if (!MediaSession->FeedImage(TextureImage, false)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media texture image feed failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }
  if (!MediaSession->FeedAudio(Audio, true)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: media audio feed failed: %s"),
           UTF8_TO_TCHAR(astral_last_error()));
    return;
  }

  UE_LOG(LogAstralSample, Display,
         TEXT("Astral sample: media feed demo loaded %s backend with RGBA byte image, texture "
              "image, and PCM16 audio"),
         *MediaBackendName);
}

void AAstralSampleActor::RunPackagedMemorySourceDemo() {
  const FString ContentModelPath =
      FPaths::Combine(FPaths::ProjectContentDir(), TEXT("AstralSample/Models/mock-model.bytes"));

  TArray<uint8> ModelBytes;
  if (!FFileHelper::LoadFileToArray(ModelBytes, *ContentModelPath)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: packaged content model read failed: %s"),
           *ContentModelPath);
    return;
  }

  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: packaged content bytes read from %s"),
         *ContentModelPath);
  LoadMemoryModel(ContentMemoryModel, ModelBytes, TEXT("packaged content"));
}

void AAstralSampleActor::RunSavedCacheDemo() {
  const FString ContentModelPath =
      FPaths::Combine(FPaths::ProjectContentDir(), TEXT("AstralSample/Models/mock-model.bytes"));
  TArray<uint8> ModelBytes;
  if (!FFileHelper::LoadFileToArray(ModelBytes, *ContentModelPath)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: cache source read failed: %s"),
           *ContentModelPath);
    return;
  }

  const FString CacheDir = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("AstralSample"));
  IFileManager::Get().MakeDirectory(*CacheDir, true);

  const FString CachePath = FPaths::Combine(CacheDir, TEXT("mock-model-cache.bytes"));
  if (!FFileHelper::SaveArrayToFile(ModelBytes, *CachePath)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: saved cache write failed: %s"), *CachePath);
    return;
  }

  TArray<uint8> CachedBytes;
  if (!FFileHelper::LoadFileToArray(CachedBytes, *CachePath)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: saved cache read failed: %s"), *CachePath);
    return;
  }

  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: saved cache bytes read from %s"),
         *CachePath);
  LoadMemoryModel(SavedCacheModel, CachedBytes, TEXT("saved cache"));
}

void AAstralSampleActor::RunMemorySearchDemo() {
  constexpr int32 MemoryDim = 2;
  constexpr int32 MemoryCapacity = 2;
  constexpr int64 MemoryKeyA = 101;
  constexpr int64 MemoryKeyB = 202;
  constexpr int32 MemoryGroup = 7;
  constexpr int32 MemoryDocument = 11;
  constexpr int32 ChunkMaxWords = 2;
  constexpr int32 ExpectedChunkCount = 2;
  constexpr int32 FirstChunkIndex = 0;
  constexpr int32 SecondChunkIndex = 1;
  constexpr int32 TopK = 1;
  constexpr int32 AnyGroup = -1;
  constexpr float VectorOne = 1.0f;
  constexpr float VectorZero = 0.0f;
  const FString DocumentText(TEXT("alpha beta gamma delta"));

  FAstralChunkerDesc ChunkDesc;
  ChunkDesc.Mode = EAstralChunkMode::Word;
  ChunkDesc.MaxUnits = ChunkMaxWords;
  ChunkDesc.OverlapUnits = 0;
  ChunkDesc.DocumentId = MemoryDocument;
  ChunkDesc.GroupId = MemoryGroup;

  TArray<FAstralChunkRange> ChunkRanges;
  int32 ChunkError = 0;
  if (!UAstralBlueprintLibrary::ChunkText(DocumentText, ChunkDesc, ChunkRanges, ChunkError) ||
      ChunkRanges.Num() != ExpectedChunkCount) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: memory search demo failed: chunk %d"),
           ChunkError);
    return;
  }

  FAstralMemoryIndexDesc MemoryDesc;
  MemoryDesc.Dimension = MemoryDim;
  MemoryDesc.Capacity = MemoryCapacity;
  MemoryDesc.Metric = EAstralMemoryMetric::Cosine;

  const FAstralOperationResult CreateResult =
      UAstralBlueprintLibrary::CreateMemoryIndexResult(MemoryDesc);
  if (!CreateResult.bSuccess) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: memory search demo failed: create %d"),
           CreateResult.ErrorCode);
    return;
  }

  TArray<FAstralMemoryRecord> Records;
  Records.Reserve(MemoryCapacity);
  FAstralMemoryRecord RecordA;
  FAstralOperationResult RecordResult = UAstralBlueprintLibrary::MakeMemoryRecordFromChunkResult(
      ChunkRanges[FirstChunkIndex], MemoryKeyA, 0, RecordA);
  if (!RecordResult.bSuccess) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: memory search demo failed: record %d"),
           RecordResult.ErrorCode);
    UAstralBlueprintLibrary::DestroyMemoryIndex(CreateResult.Handle);
    return;
  }
  Records.Add(RecordA);

  FAstralMemoryRecord RecordB;
  RecordResult = UAstralBlueprintLibrary::MakeMemoryRecordFromChunkResult(
      ChunkRanges[SecondChunkIndex], MemoryKeyB, 0, RecordB);
  if (!RecordResult.bSuccess) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: memory search demo failed: record %d"),
           RecordResult.ErrorCode);
    UAstralBlueprintLibrary::DestroyMemoryIndex(CreateResult.Handle);
    return;
  }
  Records.Add(RecordB);

  TArray<float> Vectors;
  Vectors.Reserve(MemoryCapacity * MemoryDim);
  Vectors.Add(VectorOne);
  Vectors.Add(VectorZero);
  Vectors.Add(VectorZero);
  Vectors.Add(VectorOne);

  const FAstralOperationResult AddResult = UAstralBlueprintLibrary::AddMemoryBatchResult(
      CreateResult.Handle, Records, Vectors, MemoryDim);
  if (!AddResult.bSuccess) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: memory search demo failed: add %d"),
           AddResult.ErrorCode);
    UAstralBlueprintLibrary::DestroyMemoryIndex(CreateResult.Handle);
    return;
  }

  TArray<float> Query;
  Query.Reserve(MemoryDim);
  Query.Add(VectorOne);
  Query.Add(VectorZero);

  TArray<FAstralMemorySearchResult> Results;
  const FAstralOperationResult SearchResult = UAstralBlueprintLibrary::SearchMemoryIndexResult(
      CreateResult.Handle, Query, TopK, AnyGroup, Results);
  if (!SearchResult.bSuccess || Results.Num() != TopK) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: memory search demo failed: search %d"),
           SearchResult.ErrorCode);
    UAstralBlueprintLibrary::DestroyMemoryIndex(CreateResult.Handle);
    return;
  }

  FString TopChunkText;
  const FAstralOperationResult ChunkTextResult = UAstralBlueprintLibrary::CopyChunkTextResult(
      DocumentText, ChunkRanges[Results[0].ChunkId], TopChunkText);
  if (!ChunkTextResult.bSuccess) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: memory search demo failed: copy chunk %d"),
           ChunkTextResult.ErrorCode);
    UAstralBlueprintLibrary::DestroyMemoryIndex(CreateResult.Handle);
    return;
  }

  UE_LOG(LogAstralSample, Display, TEXT("Astral sample: RAG chunk text %s"), *TopChunkText);
  UE_LOG(LogAstralSample, Display,
         TEXT("Astral sample: RAG search top key %lld score %.3f group %d"), Results[0].Key,
         Results[0].Score, Results[0].GroupId);
  UAstralBlueprintLibrary::DestroyMemoryIndex(CreateResult.Handle);
}

void AAstralSampleActor::RunErrorDemo() {
  UAstralModel* BadModel = NewObject<UAstralModel>(this);

  FAstralModelDesc BadDesc{};
  BadDesc.SourceKind = EAstralModelSourceKind::Memory;
  BadDesc.BackendName = BackendName;

  if (BadModel->Load(BadDesc)) {
    UE_LOG(LogAstralSample, Error, TEXT("Astral sample: expected memory-source load to fail"));
    BadModel->Release();
    return;
  }

  UE_LOG(LogAstralSample, Warning, TEXT("Astral sample: expected load failure: %s"),
         UTF8_TO_TCHAR(astral_last_error()));
}

void AAstralSampleActor::OnStreamBytes(TConstArrayView<uint8> Bytes) {
  FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
  UE_LOG(LogAstralSample, Display, TEXT("Astral sample stream: %.*s"), Text.Length(), Text.Get());
}
