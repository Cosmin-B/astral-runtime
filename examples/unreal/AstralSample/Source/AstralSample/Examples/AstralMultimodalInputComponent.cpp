#include "AstralMultimodalInputComponent.h"

#include "AstralBlueprintLibrary.h"
#include "AstralEmbedder.h"
#include "AstralMediaLibrary.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "Containers/StringConv.h"

UAstralMultimodalInputComponent::UAstralMultimodalInputComponent() {
  PrimaryComponentTick.bCanEverTick = false;
  ModelDesc.BackendName = TEXT("mock");
  MediaDesc.MediaPath = TEXT("mock-media");
  SessionDesc.MaxTokens = 96;
  SessionDesc.bStreamEnabled = true;
}

bool UAstralMultimodalInputComponent::EnsureModel() {
  if (Model != nullptr && Model->IsValid()) {
    return true;
  }
  if (MediaDesc.MediaPath.IsEmpty() && MediaDesc.MediaBytes.IsEmpty()) {
    return false;
  }
  Model = NewObject<UAstralModel>(this);
  return Model->Load(ModelDesc) && Model->InitMedia(MediaDesc) && Model->GetMediaInfo(MediaInfo);
}

bool UAstralMultimodalInputComponent::StartGeneration(const FAstralImageDesc* ImageDesc,
                                                      const FAstralAudioDesc* AudioDesc) {
  ReleaseSession();
  Session = NewObject<UAstralSession>(this);
  Output.Reset();
  if (!Session->Create(Model, SessionDesc) || !Session->FeedPrompt(Prompt, false)) {
    ReleaseSession();
    return false;
  }
  Session->OnStreamBytesNative().AddUObject(this, &UAstralMultimodalInputComponent::OnStreamBytes);
  const bool bFed = ImageDesc != nullptr ? Session->FeedImage(*ImageDesc, true)
                                         : Session->FeedAudio(*AudioDesc, true);
  if (!bFed || !Session->Decode()) {
    ReleaseSession();
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::CreateSessionRequestResult(Session, ActiveRequest);
  return LastOperation.bSuccess;
}

void UAstralMultimodalInputComponent::OnStreamBytes(TConstArrayView<uint8> Bytes) {
  const FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
  Output.AppendChars(Text.Get(), Text.Length());
}

bool UAstralMultimodalInputComponent::RunImageGeneration() {
  if (Image == nullptr || !EnsureModel()) {
    return false;
  }
  int64 Caps = 0;
  FAstralImageDesc Desc{};
  if (!Model->GetCaps(Caps) || !UAstralBlueprintLibrary::HasImageInput(Caps) ||
      !UAstralMediaLibrary::MakeRGBA8ImageFromTexture(Image, Desc)) {
    return false;
  }
  return StartGeneration(&Desc, nullptr);
}

bool UAstralMultimodalInputComponent::RunAudioGeneration() {
  if (!EnsureModel()) {
    return false;
  }
  int64 Caps = 0;
  FAstralAudioDesc Desc{};
  if (!Model->GetCaps(Caps) || !UAstralBlueprintLibrary::HasAudioInput(Caps) ||
      !UAstralMediaLibrary::MakePCM16AudioFromBytes(Pcm16Bytes, AudioChannels, AudioSampleRate,
                                                    Desc)) {
    return false;
  }
  return StartGeneration(nullptr, &Desc);
}

bool UAstralMultimodalInputComponent::EmbedImage() {
  if (Image == nullptr || !EnsureModel()) {
    return false;
  }
  int64 Caps = 0;
  FAstralImageDesc ImageDesc{};
  if (!Model->GetCaps(Caps) || !UAstralBlueprintLibrary::HasMultimodalEmbeddings(Caps) ||
      !UAstralMediaLibrary::MakeRGBA8ImageFromTexture(Image, ImageDesc)) {
    return false;
  }
  if (Embedder == nullptr) {
    Embedder = NewObject<UAstralEmbedder>(this);
  }
  if (!Embedder->IsValid() && !Embedder->Create(Model)) {
    return false;
  }
  FAstralAudioDesc EmptyAudio{};
  const FAstralAsyncResult Enqueue =
      Embedder->EnqueueMultimodalResult(Prompt, ImageDesc, EmptyAudio, true, false);
  if (!Enqueue.bSuccess) {
    return false;
  }
  ActiveEmbeddingTicket = Enqueue.Ticket;
  LastOperation = UAstralBlueprintLibrary::CreateEmbeddingRequestResult(
      Embedder, ActiveEmbeddingTicket, ActiveRequest);
  if (!LastOperation.bSuccess) {
    return false;
  }
  const FAstralAsyncResult Collected = Embedder->CollectResult(ActiveEmbeddingTicket, Embedding);
  ActiveEmbeddingTicket = 0;
  return Collected.bSuccess;
}

bool UAstralMultimodalInputComponent::Cancel() {
  if (ActiveRequest.OwnerHandle == 0) {
    return false;
  }
  LastOperation = UAstralBlueprintLibrary::CancelRequestResult(ActiveRequest);
  if (ActiveEmbeddingTicket != 0 && Embedder != nullptr) {
    Embedder->Cancel(ActiveEmbeddingTicket);
    ActiveEmbeddingTicket = 0;
  }
  return LastOperation.bSuccess;
}

void UAstralMultimodalInputComponent::ReleaseSession() {
  if (Session != nullptr) {
    Session->OnStreamBytesNative().RemoveAll(this);
    Session->Cancel();
    Session = nullptr;
  }
  ActiveRequest = FAstralRequestRef{};
}

void UAstralMultimodalInputComponent::ReleaseOwnedResources() {
  Cancel();
  ReleaseSession();
  if (Embedder != nullptr) {
    Embedder->Destroy();
    Embedder = nullptr;
  }
  if (Model != nullptr) {
    Model->Release();
    Model = nullptr;
  }
}

void UAstralMultimodalInputComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  ReleaseOwnedResources();
  Super::EndPlay(EndPlayReason);
}
