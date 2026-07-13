#include "AstralStreamingChatComponent.h"

#include "AstralBlueprintLibrary.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "Containers/StringConv.h"

UAstralStreamingChatComponent::UAstralStreamingChatComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = false;

  ModelDesc.BackendName = TEXT("mock");
  ModelDesc.ContextSize = 256;
  ModelDesc.BatchSize = 64;

  SessionDesc.MaxTokens = 64;
  SessionDesc.Temperature = 0.0f;
  SessionDesc.TopK = 0;
  SessionDesc.TopP = 1.0f;
  SessionDesc.bStreamEnabled = true;
  SessionDesc.Seed = 7;
}

bool UAstralStreamingChatComponent::EnsureSession() {
  if (Model == nullptr) {
    Model = NewObject<UAstralModel>(this);
  }
  if (!Model->IsValid() && !Model->Load(ModelDesc)) {
    return false;
  }

  if (Session == nullptr) {
    Session = NewObject<UAstralSession>(this);
  }

  if (Session->IsValid()) {
    if (!Session->Reset(SessionDesc)) {
      return false;
    }
  } else if (!Session->Create(Model, SessionDesc)) {
    return false;
  }

  Session->OnStreamBytesNative().RemoveAll(this);
  Session->OnStreamBytesNative().AddUObject(this, &UAstralStreamingChatComponent::OnStreamBytes);
  return true;
}

bool UAstralStreamingChatComponent::Run() {
  if (bRunning && !Cancel()) {
    return false;
  }
  if (bRunning && Session->Wait(1000) == static_cast<int32>(EAstralError::Timeout)) {
    return false;
  }
  if (!EnsureSession()) {
    Finish(false, static_cast<int32>(EAstralError::Invalid));
    return false;
  }

  Output.Reset();
  LastStats = FAstralStats{};
  ActiveRequest = FAstralRequestRef{};
  LastErrorCode = static_cast<int32>(EAstralError::OK);

  if (!SystemPrompt.IsEmpty() && !Session->SetSystemPrompt(SystemPrompt)) {
    Finish(false, static_cast<int32>(EAstralError::State));
    return false;
  }
  if (!Session->FeedPrompt(Prompt, true) || !Session->Decode()) {
    Finish(false, static_cast<int32>(EAstralError::State));
    return false;
  }

  const FAstralOperationResult RequestResult =
      UAstralBlueprintLibrary::CreateSessionRequestResult(Session, ActiveRequest);
  if (!RequestResult.bSuccess) {
    Session->Cancel();
    Finish(false, RequestResult.ErrorCode);
    return false;
  }

  bRunning = true;
  SetComponentTickEnabled(true);
  return true;
}

bool UAstralStreamingChatComponent::Cancel() {
  if (Session == nullptr || !Session->IsValid()) {
    return false;
  }

  if (ActiveRequest.OwnerHandle != 0) {
    const FAstralOperationResult Result =
        UAstralBlueprintLibrary::CancelRequestResult(ActiveRequest);
    if (Result.bSuccess) {
      SetComponentTickEnabled(true);
      return true;
    }
  }
  const bool bCanceled = Session->Cancel();
  SetComponentTickEnabled(bCanceled);
  return bCanceled;
}

void UAstralStreamingChatComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                                  FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  if (!bRunning) {
    return;
  }

  FAstralRequestStatus Status{};
  const FAstralOperationResult Result =
      UAstralBlueprintLibrary::GetRequestStatusResult(ActiveRequest, Status);
  if (!Result.bSuccess) {
    Finish(false, Result.ErrorCode);
    return;
  }
  if (!UAstralBlueprintLibrary::IsRequestTerminal(Status)) {
    return;
  }

  const bool bSuccess = UAstralBlueprintLibrary::IsRequestSuccessful(Status);
  Finish(bSuccess, Status.ErrorCode);
}

void UAstralStreamingChatComponent::OnStreamBytes(TConstArrayView<uint8> Bytes) {
  if (Bytes.IsEmpty()) {
    return;
  }

  const FUTF8ToTCHAR Converted(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
  const FString Text(Converted.Length(), Converted.Get());
  Output.Append(Text);
  if (OnText.IsBound()) {
    OnText.Broadcast(Text);
  }
}

void UAstralStreamingChatComponent::Finish(bool bSuccess, int32 ErrorCode) {
  bRunning = false;
  LastErrorCode = ErrorCode;
  LastStats = Session != nullptr ? Session->GetStats() : FAstralStats{};
  SetComponentTickEnabled(false);
  OnFinished.Broadcast(bSuccess, ErrorCode);
}

void UAstralStreamingChatComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  if (Session != nullptr) {
    Session->OnStreamBytesNative().RemoveAll(this);
    Session->Cancel();
  }
  if (Model != nullptr) {
    Model->Release();
  }
  bRunning = false;
  SetComponentTickEnabled(false);
  Super::EndPlay(EndPlayReason);
}
