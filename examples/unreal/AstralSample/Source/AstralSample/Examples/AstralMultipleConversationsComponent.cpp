#include "AstralMultipleConversationsComponent.h"

#include "AstralBlueprintLibrary.h"
#include "AstralConversation.h"
#include "AstralModel.h"
#include "Containers/StringConv.h"
#include "astral_rt.h"

UAstralMultipleConversationsComponent::UAstralMultipleConversationsComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = false;

  ModelDesc.BackendName = TEXT("mock");
  ModelDesc.ContextSize = 256;
  ModelDesc.BatchSize = 64;

  ExecutorDesc.MaxSlots = 4;
  ExecutorDesc.MaxBatchTokens = 64;

  ConversationDesc.MaxTokens = 64;
  ConversationDesc.Temperature = 0.0f;
  ConversationDesc.TopK = 0;
  ConversationDesc.TopP = 1.0f;
  ConversationDesc.bStreamEnabled = true;
  ConversationDesc.bAutoPumpStream = false;
  ConversationDesc.Seed = 11;

  Prompts = {TEXT("Greet the player as a merchant."),
             TEXT("Report the state of the north gate as a guard."),
             TEXT("Offer one concise quest hook as a scholar.")};
}

bool UAstralMultipleConversationsComponent::EnsureModel() {
  if (Model != nullptr && Model->IsValid()) {
    return true;
  }

  Model = NewObject<UAstralModel>(this);
  if (!Model->Load(ModelDesc)) {
    return false;
  }

  FAstralExecutorDesc Config = ExecutorDesc;
  Config.MaxSlots = FMath::Max(Config.MaxSlots, Prompts.Num());
  return Model->ConfigureExecutor(Config);
}

bool UAstralMultipleConversationsComponent::Run() {
  if (bRunning) {
    Cancel();
    for (int32 Index = 0; Index < ActiveConversationCount; ++Index) {
      if (Conversations.IsValidIndex(Index) && Conversations[Index] != nullptr) {
        Conversations[Index]->Wait(1000);
      }
    }
    bRunning = false;
  }
  if (Prompts.IsEmpty() || Prompts.Num() > ExecutorDesc.MaxSlots || !EnsureModel()) {
    return false;
  }

  ActiveConversationCount = Prompts.Num();
  Conversations.SetNum(FMath::Max(Conversations.Num(), ActiveConversationCount));
  Requests.SetNum(ActiveConversationCount);
  StreamBuffers.SetNum(ActiveConversationCount);
  Terminal.Init(false, ActiveConversationCount);
  Outputs.Init(FString(), ActiveConversationCount);
  Stats.Init(FAstralConversationStats{}, ActiveConversationCount);
  LastErrorCodes.Init(static_cast<int32>(EAstralError::OK), ActiveConversationCount);

  for (int32 Index = 0; Index < ActiveConversationCount; ++Index) {
    StreamBuffers[Index].SetNumUninitialized(4096);
    const bool bReset = Conversations[Index] != nullptr && Conversations[Index]->IsValid();
    if (!StartConversation(Index, Prompts[Index], bReset)) {
      AbortPartialRun(Index);
      return false;
    }
  }

  bRunning = true;
  SetComponentTickEnabled(true);
  return true;
}

void UAstralMultipleConversationsComponent::AbortPartialRun(int32 StartedCount) {
  for (int32 Index = 0; Index < StartedCount; ++Index) {
    if (Conversations.IsValidIndex(Index) && Conversations[Index] != nullptr) {
      Conversations[Index]->Cancel();
      Conversations[Index]->Wait(1000);
    }
  }
  ActiveConversationCount = 0;
  bRunning = false;
  SetComponentTickEnabled(false);
}

bool UAstralMultipleConversationsComponent::StartConversation(int32 Index,
                                                              const FString& UserPrompt,
                                                              bool bReset) {
  if (!Conversations.IsValidIndex(Index)) {
    return false;
  }

  FAstralConversationDesc Desc = ConversationDesc;
  Desc.bStreamEnabled = true;
  Desc.bAutoPumpStream = false;

  if (bReset) {
    if (!Conversations[Index]->Reset(Desc)) {
      return false;
    }
  } else {
    Conversations[Index] = NewObject<UAstralConversation>(this);
    if (!Conversations[Index]->Create(Model, Desc)) {
      return false;
    }
  }

  if (!SystemPrompt.IsEmpty() && !Conversations[Index]->SetSystemPrompt(SystemPrompt)) {
    return false;
  }
  if (!Conversations[Index]->FeedPrompt(UserPrompt, true) || !Conversations[Index]->Decode()) {
    return false;
  }

  Requests[Index] = FAstralRequestRef{};
  const FAstralOperationResult Result = UAstralBlueprintLibrary::CreateConversationRequestResult(
      static_cast<int64>(Conversations[Index]->GetHandle()), Requests[Index]);
  return Result.bSuccess;
}

void UAstralMultipleConversationsComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  if (!bRunning) {
    return;
  }

  bool bAllTerminal = true;
  for (int32 Index = 0; Index < ActiveConversationCount; ++Index) {
    if (Terminal[Index]) {
      continue;
    }
    bAllTerminal = false;

    const int32 BytesRead = Conversations[Index]->StreamRead(StreamBuffers[Index], 0);
    if (BytesRead > 0) {
      const FUTF8ToTCHAR Converted(
          reinterpret_cast<const ANSICHAR*>(StreamBuffers[Index].GetData()), BytesRead);
      const FString Text(Converted.Length(), Converted.Get());
      Outputs[Index].Append(Text);
      OnText.Broadcast(Index, Text);
      continue;
    }
    if (BytesRead == ASTRAL_E_TIMEOUT) {
      continue;
    }
    if (BytesRead < 0) {
      FinishConversation(Index, false, BytesRead);
      continue;
    }

    FAstralRequestStatus Status{};
    const FAstralOperationResult StatusResult =
        UAstralBlueprintLibrary::GetRequestStatusResult(Requests[Index], Status);
    if (!StatusResult.bSuccess) {
      FinishConversation(Index, false, StatusResult.ErrorCode);
      continue;
    }
    FinishConversation(Index, UAstralBlueprintLibrary::IsRequestSuccessful(Status),
                       Status.ErrorCode);
  }

  if (bAllTerminal) {
    bRunning = false;
    SetComponentTickEnabled(false);
  }
}

bool UAstralMultipleConversationsComponent::CancelConversation(int32 Index) {
  if (!Conversations.IsValidIndex(Index) || Conversations[Index] == nullptr ||
      !Conversations[Index]->IsValid()) {
    return false;
  }

  if (Requests.IsValidIndex(Index) && Requests[Index].OwnerHandle != 0) {
    const FAstralOperationResult Result =
        UAstralBlueprintLibrary::CancelRequestResult(Requests[Index]);
    if (Result.bSuccess) {
      return true;
    }
  }
  return Conversations[Index]->Cancel();
}

void UAstralMultipleConversationsComponent::Cancel() {
  bool bCanceledAny = false;
  for (int32 Index = 0; Index < ActiveConversationCount; ++Index) {
    bCanceledAny = CancelConversation(Index) || bCanceledAny;
  }
  if (bCanceledAny) {
    bRunning = true;
    SetComponentTickEnabled(true);
  }
}

bool UAstralMultipleConversationsComponent::ResetConversation(int32 Index,
                                                              const FString& NewPrompt) {
  if (!Conversations.IsValidIndex(Index) || Conversations[Index] == nullptr) {
    return false;
  }

  CancelConversation(Index);
  Conversations[Index]->Wait(1000);
  if (!StartConversation(Index, NewPrompt, true)) {
    return false;
  }

  Outputs[Index].Reset();
  Stats[Index] = FAstralConversationStats{};
  LastErrorCodes[Index] = static_cast<int32>(EAstralError::OK);
  Terminal[Index] = false;
  bRunning = true;
  SetComponentTickEnabled(true);
  return true;
}

void UAstralMultipleConversationsComponent::FinishConversation(int32 Index, bool bSuccess,
                                                               int32 ErrorCode) {
  if (!Terminal.IsValidIndex(Index) || Terminal[Index]) {
    return;
  }
  Terminal[Index] = true;
  if (Stats.IsValidIndex(Index) && Conversations.IsValidIndex(Index) &&
      Conversations[Index] != nullptr) {
    Stats[Index] = Conversations[Index]->GetStats();
  }
  if (LastErrorCodes.IsValidIndex(Index)) {
    LastErrorCodes[Index] = ErrorCode;
  }
  OnConversationFinished.Broadcast(Index, bSuccess, ErrorCode);
}

void UAstralMultipleConversationsComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  Cancel();
  bRunning = false;
  SetComponentTickEnabled(false);
  if (Model != nullptr) {
    Model->Release();
  }
  Super::EndPlay(EndPlayReason);
}
