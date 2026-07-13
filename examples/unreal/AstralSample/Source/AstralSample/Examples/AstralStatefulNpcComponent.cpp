#include "AstralStatefulNpcComponent.h"

#include "AstralBlueprintLibrary.h"
#include "AstralModel.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

UAstralStatefulNpcComponent::UAstralStatefulNpcComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = false;
  ModelDesc.BackendName = TEXT("mock");
  ModelDesc.ContextSize = 512;
  AgentDesc.MaxTokens = 128;
  AgentDesc.bStream = true;
  AgentDesc.OverflowPolicy = EAstralAgentOverflowPolicy::TruncateOldest;
}

FString UAstralStatefulNpcComponent::HistoryPath() const {
  return FPaths::Combine(FPaths::ProjectSavedDir(), HistoryFileName);
}

bool UAstralStatefulNpcComponent::EnsureAgent() {
  if (AgentHandle != 0) {
    return true;
  }
  Model = NewObject<UAstralModel>(this);
  if (!Model->Load(ModelDesc)) {
    return false;
  }

  FAstralToolDesc MoveTool{};
  MoveTool.ToolId = 1;
  MoveTool.Name = TEXT("move_to");
  MoveTool.Description = TEXT("Move an NPC or player to a named destination.");
  MoveTool.JsonSchema =
      TEXT("{\"type\":\"object\",\"properties\":{\"destination\":{\"type\":\"string\"}},")
          TEXT("\"required\":[\"destination\"]}");
  TArray<FAstralToolDesc> Tools{MoveTool};
  LastOperation = UAstralBlueprintLibrary::CreateToolsetResult(Tools, EAstralToolChoiceMode::Auto);
  if (!LastOperation.bSuccess) {
    return false;
  }
  ToolsetHandle = LastOperation.Handle;

  AgentDesc.ModelHandle = static_cast<int64>(Model->GetHandle());
  AgentDesc.ToolsetHandle = ToolsetHandle;
  AgentDesc.SystemPrompt = SystemPrompt;
  AgentDesc.Summary = Summary;
  AgentDesc.MemoryContext = MemoryContext;
  LastOperation = UAstralBlueprintLibrary::CreateAgentResult(AgentDesc);
  if (!LastOperation.bSuccess) {
    return false;
  }
  AgentHandle = LastOperation.Handle;
  LoadHistory();
  return true;
}

bool UAstralStatefulNpcComponent::Ask() {
  if (bRunning && !Cancel()) {
    return false;
  }
  if (!EnsureAgent()) {
    Finish(false, LastOperation.ErrorCode);
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::SetAgentSystemPromptResult(AgentHandle, SystemPrompt);
  if (LastOperation.bSuccess) {
    LastOperation = UAstralBlueprintLibrary::SetAgentSummaryResult(AgentHandle, Summary);
  }
  if (LastOperation.bSuccess) {
    LastOperation =
        UAstralBlueprintLibrary::SetAgentMemoryContextResult(AgentHandle, MemoryContext);
  }
  if (LastOperation.bSuccess) {
    LastOperation =
        UAstralBlueprintLibrary::EnqueueAgentChatResult(AgentHandle, UserMessage, false);
  }
  if (!LastOperation.bSuccess) {
    Finish(false, LastOperation.ErrorCode);
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::CreateAgentChatRequestResult(AgentHandle, ActiveRequest);
  if (!LastOperation.bSuccess) {
    Finish(false, LastOperation.ErrorCode);
    return false;
  }

  Output.Reset();
  LastToolCall = FAstralToolCallResult{};
  bRunning = true;
  SetComponentTickEnabled(true);
  return true;
}

void UAstralStatefulNpcComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                                FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  if (!bRunning) {
    return;
  }

  FString Text;
  LastOperation = UAstralBlueprintLibrary::ReadAgentChatResult(AgentHandle, 0, Text);
  if (LastOperation.bSuccess && !Text.IsEmpty()) {
    Output.Append(Text);
    OnText.Broadcast(Text);
    return;
  }
  if (LastOperation.bTimeout) {
    return;
  }
  if (LastOperation.bEndOfStream) {
    LastOperation = UAstralBlueprintLibrary::GetAgentChatStatusResult(AgentHandle, LastChat);
    if (LastOperation.bSuccess) {
      LastOperation =
          UAstralBlueprintLibrary::GetAgentChatToolCallResultStatus(AgentHandle, LastToolCall);
    }
    SaveHistory();
    UAstralBlueprintLibrary::ReleaseAgentSlotResult(AgentHandle);
    Finish(LastOperation.bSuccess, LastOperation.ErrorCode);
    return;
  }
  Finish(false, LastOperation.ErrorCode);
}

bool UAstralStatefulNpcComponent::Cancel() {
  if (AgentHandle == 0) {
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::CancelAgentChatResult(AgentHandle);
  return LastOperation.bSuccess;
}

bool UAstralStatefulNpcComponent::SaveHistory() {
  if (AgentHandle == 0) {
    return false;
  }
  TArray<uint8> Bytes;
  LastOperation = UAstralBlueprintLibrary::SaveAgentHistoryResult(AgentHandle, Bytes);
  if (!LastOperation.bSuccess) {
    return false;
  }
  IFileManager::Get().MakeDirectory(*FPaths::GetPath(HistoryPath()), true);
  return FFileHelper::SaveArrayToFile(Bytes, *HistoryPath());
}

bool UAstralStatefulNpcComponent::LoadHistory() {
  TArray<uint8> Bytes;
  if (AgentHandle == 0 || !FFileHelper::LoadFileToArray(Bytes, *HistoryPath())) {
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::LoadAgentHistoryResult(AgentHandle, Bytes);
  return LastOperation.bSuccess;
}

bool UAstralStatefulNpcComponent::ClearHistory() {
  if (AgentHandle == 0) {
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::ClearAgentHistoryResult(AgentHandle);
  IFileManager::Get().Delete(*HistoryPath());
  return LastOperation.bSuccess;
}

void UAstralStatefulNpcComponent::Finish(bool bSuccess, int32 ErrorCode) {
  bRunning = false;
  SetComponentTickEnabled(false);
  OnFinished.Broadcast(bSuccess, ErrorCode);
}

void UAstralStatefulNpcComponent::ReleaseOwnedResources() {
  if (AgentHandle != 0) {
    UAstralBlueprintLibrary::CancelAgentChatResult(AgentHandle);
    UAstralBlueprintLibrary::ReleaseAgentSlotResult(AgentHandle);
    UAstralBlueprintLibrary::DestroyAgent(AgentHandle);
    AgentHandle = 0;
  }
  if (ToolsetHandle != 0) {
    UAstralBlueprintLibrary::DestroyToolset(ToolsetHandle);
    ToolsetHandle = 0;
  }
  if (Model != nullptr) {
    Model->Release();
    Model = nullptr;
  }
}

void UAstralStatefulNpcComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  ReleaseOwnedResources();
  Super::EndPlay(EndPlayReason);
}
