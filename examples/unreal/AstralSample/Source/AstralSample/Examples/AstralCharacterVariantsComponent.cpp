#include "AstralCharacterVariantsComponent.h"

#include "AstralBlueprintLibrary.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "Containers/StringConv.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

UAstralCharacterVariantsComponent::UAstralCharacterVariantsComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = false;
  ModelDesc.BackendName = TEXT("mock");
  SessionDesc.MaxTokens = 96;
  SessionDesc.Temperature = 0.0f;
  SessionDesc.bStreamEnabled = true;
}

FString UAstralCharacterVariantsComponent::CachePath() const {
  return FPaths::Combine(FPaths::ProjectSavedDir(), CacheFileName);
}

bool UAstralCharacterVariantsComponent::EnsureModel() {
  if (Model != nullptr && Model->IsValid()) {
    return true;
  }
  Model = NewObject<UAstralModel>(this);
  return Model->Load(ModelDesc);
}

bool UAstralCharacterVariantsComponent::PreparePromptCache() {
  if (!EnsureModel()) {
    return false;
  }
  if (CacheHandle != 0) {
    UAstralBlueprintLibrary::DestroyPromptCache(CacheHandle);
  }
  LastOperation = UAstralBlueprintLibrary::CreatePromptCacheResult(CacheDesc);
  if (!LastOperation.bSuccess) {
    CacheHandle = 0;
    return false;
  }
  CacheHandle = LastOperation.Handle;

  TArray<int32> Tokens;
  LastOperation = Model->CountTokensResult(SystemPrompt, true, false, PromptTokenCount);
  if (LastOperation.bSuccess) {
    LastOperation = Model->TokenizeResult(SystemPrompt, true, false, Tokens);
  }
  FAstralPromptCacheKey Key{};
  if (LastOperation.bSuccess) {
    LastOperation = UAstralBlueprintLibrary::MakePromptCacheKeyResult(
        static_cast<int64>(Model->GetHandle()), EAstralPromptSectionKind::System, 1, SystemPrompt,
        Key);
  }
  if (LastOperation.bSuccess) {
    LastOperation = UAstralBlueprintLibrary::PutPromptCacheTokensResult(CacheHandle, Key, Tokens);
  }
  TArray<int32> Restored;
  if (LastOperation.bSuccess) {
    LastOperation = UAstralBlueprintLibrary::GetPromptCacheTokensResult(
        CacheHandle, Key, FMath::Max(1, PromptTokenCount), Restored);
  }
  if (LastOperation.bSuccess) {
    LastOperation = Model->DetokenizeResult(Restored, TokenRoundTrip);
  }
  TArray<uint8> Snapshot;
  if (LastOperation.bSuccess) {
    LastOperation = UAstralBlueprintLibrary::SavePromptCacheResult(CacheHandle, Snapshot);
  }
  if (!LastOperation.bSuccess) {
    return false;
  }
  IFileManager::Get().MakeDirectory(*FPaths::GetPath(CachePath()), true);
  return FFileHelper::SaveArrayToFile(Snapshot, *CachePath());
}

bool UAstralCharacterVariantsComponent::LoadPromptCache() {
  if (!EnsureModel()) {
    return false;
  }
  TArray<uint8> Snapshot;
  if (!FFileHelper::LoadFileToArray(Snapshot, *CachePath())) {
    return false;
  }
  if (CacheHandle != 0) {
    UAstralBlueprintLibrary::DestroyPromptCache(CacheHandle);
  }
  LastOperation = UAstralBlueprintLibrary::LoadPromptCacheResult(CacheDesc, Snapshot);
  CacheHandle = LastOperation.bSuccess ? LastOperation.Handle : 0;
  return LastOperation.bSuccess;
}

bool UAstralCharacterVariantsComponent::Run() {
  ReleaseSession();
  if (!EnsureModel()) {
    return false;
  }
  Session = NewObject<UAstralSession>(this);
  if (!Session->Create(Model, SessionDesc) || !Session->SetSystemPrompt(SystemPrompt)) {
    ReleaseSession();
    return false;
  }
  int64 Caps = 0;
  if (!Model->GetCaps(Caps)) {
    ReleaseSession();
    return false;
  }
  if (!StopSequence.IsEmpty() && UAstralBlueprintLibrary::HasStopSequences(Caps) &&
      !Session->StopAddString(StopSequence)) {
    ReleaseSession();
    return false;
  }
  if (!JsonSchema.IsEmpty() && UAstralBlueprintLibrary::HasJsonSchemaGrammar(Caps) &&
      !Session->SetGrammarJsonSchema(JsonSchema)) {
    ReleaseSession();
    return false;
  }

  if (!AdapterDesc.AdapterPath.IsEmpty()) {
    if (AdapterHandle != 0) {
      Model->ReleaseAdapter(AdapterHandle);
      AdapterHandle = 0;
    }
    if (!Model->LoadAdapter(AdapterDesc, AdapterHandle) ||
        !Session->AddAdapter(AdapterHandle, AdapterScale) ||
        !Session->SetAdapterScale(0, AdapterScale)) {
      ReleaseSession();
      return false;
    }
  }

  Session->OnStreamBytesNative().AddUObject(this,
                                            &UAstralCharacterVariantsComponent::OnStreamBytes);
  if (!Session->FeedPrompt(UserPrompt, true) || !Session->Decode()) {
    ReleaseSession();
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::CreateSessionRequestResult(Session, ActiveRequest);
  if (!LastOperation.bSuccess) {
    ReleaseSession();
    return false;
  }
  Output.Reset();
  bRunning = true;
  SetComponentTickEnabled(true);
  return true;
}

bool UAstralCharacterVariantsComponent::SetVariantScale(float NewScale) {
  AdapterScale = NewScale;
  return Session != nullptr && AdapterHandle != 0 && Session->SetAdapterScale(0, NewScale);
}

bool UAstralCharacterVariantsComponent::Cancel() {
  if (ActiveRequest.OwnerHandle == 0) {
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::CancelRequestResult(ActiveRequest);
  return LastOperation.bSuccess;
}

void UAstralCharacterVariantsComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  if (!bRunning) {
    return;
  }
  FAstralRequestStatus Status{};
  LastOperation = UAstralBlueprintLibrary::GetRequestStatusResult(ActiveRequest, Status);
  if (!LastOperation.bSuccess || UAstralBlueprintLibrary::IsRequestTerminal(Status)) {
    ReleaseSession();
  }
}

void UAstralCharacterVariantsComponent::OnStreamBytes(TConstArrayView<uint8> Bytes) {
  const FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
  Output.AppendChars(Text.Get(), Text.Length());
}

void UAstralCharacterVariantsComponent::ReleaseSession() {
  if (Session != nullptr) {
    Session->OnStreamBytesNative().RemoveAll(this);
    Session->Cancel();
    Session = nullptr;
  }
  ActiveRequest = FAstralRequestRef{};
  bRunning = false;
  SetComponentTickEnabled(false);
}

void UAstralCharacterVariantsComponent::ReleaseOwnedResources() {
  ReleaseSession();
  if (CacheHandle != 0) {
    UAstralBlueprintLibrary::DestroyPromptCache(CacheHandle);
    CacheHandle = 0;
  }
  if (Model != nullptr && AdapterHandle != 0) {
    Model->ReleaseAdapter(AdapterHandle);
    AdapterHandle = 0;
  }
  if (Model != nullptr) {
    Model->Release();
    Model = nullptr;
  }
}

void UAstralCharacterVariantsComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  ReleaseOwnedResources();
  Super::EndPlay(EndPlayReason);
}
