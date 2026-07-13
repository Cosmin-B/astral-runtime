#include "AstralConversation.h"
#include "AstralLog.h"
#include "AstralModel.h"
#include "AstralSessionStreamPump.h"
#include "IAstralRT.h"

#include "Containers/StringConv.h"
#include "Containers/UnrealString.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

#include "astral_rt.h"

namespace {

static AstralConvDesc make_native_desc(uint64 ModelHandle, const FAstralConversationDesc& Desc) {
  AstralConvDesc Native{};
  Native.size = sizeof(AstralConvDesc);
  Native.model = static_cast<AstralHandle>(ModelHandle);
  Native.max_tokens = static_cast<uint32_t>(Desc.MaxTokens);
  Native.temperature = Desc.Temperature;
  Native.top_k = static_cast<uint32_t>(Desc.TopK);
  Native.top_p = Desc.TopP;
  Native.stream_enabled = Desc.bStreamEnabled ? 1 : 0;
  Native.seed = static_cast<uint32_t>(Desc.Seed);
  return Native;
}

static AstralToolChoiceMode native_tool_choice(EAstralToolChoiceMode ChoiceMode) {
  switch (ChoiceMode) {
  case EAstralToolChoiceMode::Required:
    return ASTRAL_TOOL_CHOICE_REQUIRED;
  case EAstralToolChoiceMode::TextOrTool:
    return ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL;
  case EAstralToolChoiceMode::Auto:
  default:
    return ASTRAL_TOOL_CHOICE_AUTO;
  }
}

struct FAstralConversationStreamReader {
  UAstralConversation* Conversation = nullptr;

  int32 operator()(TArray<uint8>& OutBuffer, uint32 TimeoutMs) const {
    return Conversation->StreamRead(OutBuffer, TimeoutMs);
  }
};

} // namespace

UAstralConversation::UAstralConversation() {
  TokenBuffer.SetNumUninitialized(4096);
  TickUtf8Buffer.Reserve(4096);
  TickTextScratch.Reserve(4096);
}

bool UAstralConversation::IsCurrentRuntimeGeneration() const {
  return IAstralRT::IsAvailable() && IAstralRT::Get().IsInitialized() &&
         IAstralRT::Get().GetRuntimeGeneration() == RuntimeGeneration;
}

bool UAstralConversation::IsValid() const {
  return ConversationHandle != 0 && IsCurrentRuntimeGeneration();
}

void UAstralConversation::BeginDestroy() {
  UpdateTicker(false);
  if (ConversationHandle != 0 && IsCurrentRuntimeGeneration()) {
    astral_conv_destroy(static_cast<AstralHandle>(ConversationHandle));
  }

  ConversationHandle = 0;
  ModelHandle = 0;
  RuntimeGeneration = 0;
  Super::BeginDestroy();
}

bool UAstralConversation::Create(UAstralModel* Model, const FAstralConversationDesc& Desc) {
  TRACE_CPUPROFILER_EVENT_SCOPE(AstralConversation_Create);

  if (ConversationHandle != 0) {
    if (!IsCurrentRuntimeGeneration()) {
      ConversationHandle = 0;
      ModelHandle = 0;
      RuntimeGeneration = 0;
    } else {
      UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: conversation already created"));
      return false;
    }
  }

  if (Model == nullptr || !Model->IsValid()) {
    UE_LOG(LogAstralRT, Error, TEXT("AstralRT: invalid conversation model"));
    return false;
  }

  const AstralConvDesc Native = make_native_desc(Model->GetHandle(), Desc);
  AstralHandle Out = 0;
  const AstralErr Err = astral_conv_create(&Native, &Out);
  if (Err != ASTRAL_OK) {
    UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_conv_create failed (%d)"),
           static_cast<int32>(Err));
    return false;
  }

  ConversationHandle = static_cast<uint64>(Out);
  ModelHandle = Model->GetHandle();
  RuntimeGeneration = IAstralRT::Get().GetRuntimeGeneration();
  UpdateTicker(Desc.bStreamEnabled && Desc.bAutoPumpStream);
  return true;
}

bool UAstralConversation::SetSystemPrompt(const FString& Prompt) {
  if (!IsValid()) {
    return false;
  }

  FTCHARToUTF8 Utf8(*Prompt);
  AstralSpanU8 Span{};
  Span.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
  Span.len = static_cast<uint32_t>(Utf8.Length());
  return astral_conv_set_system_prompt(static_cast<AstralHandle>(ConversationHandle), Span) ==
         ASTRAL_OK;
}

bool UAstralConversation::FeedPrompt(const FString& Prompt, bool bFinalize) {
  if (!IsValid()) {
    return false;
  }

  FTCHARToUTF8 Utf8(*Prompt);
  AstralSpanU8 Span{};
  Span.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
  Span.len = static_cast<uint32_t>(Utf8.Length());
  return astral_conv_feed(static_cast<AstralHandle>(ConversationHandle), Span, bFinalize ? 1 : 0) ==
         ASTRAL_OK;
}

bool UAstralConversation::FeedImage(const FAstralImageDesc& Image, bool bFinalize) {
  if (!IsValid() || Image.Pixels.Num() == 0) {
    return false;
  }

  AstralImageDesc Native{};
  Native.size = sizeof(AstralImageDesc);
  Native.format = static_cast<AstralImageFormat>(Image.Format);
  Native.width = static_cast<uint32_t>(Image.Width);
  Native.height = static_cast<uint32_t>(Image.Height);
  Native.row_stride = static_cast<uint32_t>(Image.RowStride);
  Native.flags = static_cast<uint32_t>(Image.Flags);
  Native.pixels.data = Image.Pixels.GetData();
  Native.pixels.len = static_cast<uint32_t>(Image.Pixels.Num());
  Native.gpu_device = Image.GpuDevice;
  Native.gpu_route_flags = static_cast<uint32_t>(Image.GpuRouteFlags);
  Native.gpu_device_mask = static_cast<uint64_t>(Image.GpuDeviceMask);
  Native.gpu_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(Image.GpuStream));
  return astral_conv_feed_image(static_cast<AstralHandle>(ConversationHandle), &Native,
                                bFinalize ? 1 : 0) == ASTRAL_OK;
}

bool UAstralConversation::FeedAudio(const FAstralAudioDesc& Audio, bool bFinalize) {
  if (!IsValid() || Audio.Samples.Num() == 0 || Audio.Channels == 0) {
    return false;
  }

  const uint32 BytesPerSample = Audio.Format == EAstralAudioFormat::F32 ? 4u : 2u;
  const uint64 TotalSamples = static_cast<uint64>(Audio.Samples.Num()) / BytesPerSample;
  if (TotalSamples % Audio.Channels != 0) {
    return false;
  }

  AstralAudioDesc Native{};
  Native.size = sizeof(AstralAudioDesc);
  Native.format = static_cast<AstralAudioFormat>(Audio.Format);
  Native.channels = static_cast<uint32_t>(Audio.Channels);
  Native.sample_rate = static_cast<uint32_t>(Audio.SampleRate);
  Native.frame_count =
      Audio.FrameCount > 0 ? static_cast<uint64>(Audio.FrameCount) : TotalSamples / Audio.Channels;
  Native.samples.data = Audio.Samples.GetData();
  Native.samples.len = static_cast<uint32_t>(Audio.Samples.Num());
  Native.flags = static_cast<uint32_t>(Audio.Flags);
  Native.gpu_device = Audio.GpuDevice;
  Native.gpu_route_flags = static_cast<uint32_t>(Audio.GpuRouteFlags);
  Native.gpu_device_mask = static_cast<uint64_t>(Audio.GpuDeviceMask);
  Native.gpu_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(Audio.GpuStream));
  return astral_conv_feed_audio(static_cast<AstralHandle>(ConversationHandle), &Native,
                                bFinalize ? 1 : 0) == ASTRAL_OK;
}

bool UAstralConversation::Decode() {
  return IsValid() &&
         astral_conv_decode(static_cast<AstralHandle>(ConversationHandle)) == ASTRAL_OK;
}

bool UAstralConversation::Cancel() {
  return IsValid() &&
         astral_conv_cancel(static_cast<AstralHandle>(ConversationHandle)) == ASTRAL_OK;
}

int32 UAstralConversation::Wait(int32 TimeoutMs) {
  if (!IsValid()) {
    return static_cast<int32>(ASTRAL_E_INVALID);
  }
  return static_cast<int32>(astral_conv_wait(static_cast<AstralHandle>(ConversationHandle),
                                             static_cast<uint32>(FMath::Max(TimeoutMs, 0))));
}

bool UAstralConversation::Reset(const FAstralConversationDesc& Desc) {
  if (!IsValid() || ModelHandle == 0) {
    return false;
  }

  const AstralConvDesc Native = make_native_desc(ModelHandle, Desc);
  const AstralErr Err = astral_conv_reset(static_cast<AstralHandle>(ConversationHandle), &Native);
  if (Err == ASTRAL_OK) {
    UpdateTicker(Desc.bStreamEnabled && Desc.bAutoPumpStream);
    return true;
  }
  return false;
}

bool UAstralConversation::SetSampler(const FAstralSamplerDesc& Desc) {
  if (!IsValid()) {
    return false;
  }

  AstralSamplerDesc Native{};
  Native.size = sizeof(AstralSamplerDesc);
  Native.temperature = Desc.Temperature;
  Native.top_k = static_cast<uint32_t>(Desc.TopK);
  Native.top_p = Desc.TopP;
  Native.min_p = Desc.MinP;
  Native.typical_p = Desc.TypicalP;
  Native.repeat_penalty = Desc.RepeatPenalty;
  Native.repeat_last_n = Desc.RepeatLastN;
  Native.penalize_nl = Desc.bPenalizeNewline ? 1 : 0;
  Native.presence_penalty = Desc.PresencePenalty;
  Native.frequency_penalty = Desc.FrequencyPenalty;
  return astral_conv_set_sampler(static_cast<AstralHandle>(ConversationHandle), &Native) ==
         ASTRAL_OK;
}

bool UAstralConversation::StopClear() {
  return IsValid() &&
         astral_conv_stop_clear(static_cast<AstralHandle>(ConversationHandle)) == ASTRAL_OK;
}

bool UAstralConversation::StopAddString(const FString& Utf8Text) {
  if (!IsValid()) {
    return false;
  }

  FTCHARToUTF8 Utf8(*Utf8Text);
  AstralSpanU8 Span{};
  Span.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
  Span.len = static_cast<uint32_t>(Utf8.Length());
  return astral_conv_stop_add_utf8(static_cast<AstralHandle>(ConversationHandle), Span) ==
         ASTRAL_OK;
}

bool UAstralConversation::SetGrammarGbnf(const FString& Grammar, const FString& RootSymbol) {
  if (!IsValid()) {
    return false;
  }

  FTCHARToUTF8 GrammarUtf8(*Grammar);
  FTCHARToUTF8 RootUtf8(*RootSymbol);
  AstralSpanU8 GrammarSpan{};
  GrammarSpan.data = reinterpret_cast<const uint8_t*>(GrammarUtf8.Get());
  GrammarSpan.len = static_cast<uint32_t>(GrammarUtf8.Length());
  AstralSpanU8 RootSpan{};
  RootSpan.data = reinterpret_cast<const uint8_t*>(RootUtf8.Get());
  RootSpan.len = static_cast<uint32_t>(RootUtf8.Length());
  return astral_conv_grammar_set_gbnf(static_cast<AstralHandle>(ConversationHandle), GrammarSpan,
                                      RootSpan) == ASTRAL_OK;
}

bool UAstralConversation::SetGrammarJsonSchema(const FString& JsonSchema) {
  if (!IsValid()) {
    return false;
  }

  FTCHARToUTF8 Utf8(*JsonSchema);
  AstralSpanU8 Span{};
  Span.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
  Span.len = static_cast<uint32_t>(Utf8.Length());
  return astral_conv_grammar_set_json_schema(static_cast<AstralHandle>(ConversationHandle), Span) ==
         ASTRAL_OK;
}

bool UAstralConversation::ClearGrammar() {
  return IsValid() &&
         astral_conv_grammar_clear(static_cast<AstralHandle>(ConversationHandle)) == ASTRAL_OK;
}

bool UAstralConversation::SetToolset(int64 ToolsetHandle, EAstralToolChoiceMode ChoiceMode) {
  return IsValid() && ToolsetHandle != 0 &&
         astral_conv_set_toolset(static_cast<AstralHandle>(ConversationHandle),
                                 static_cast<AstralHandle>(ToolsetHandle),
                                 native_tool_choice(ChoiceMode)) == ASTRAL_OK;
}

bool UAstralConversation::ClearToolset() {
  return IsValid() &&
         astral_conv_clear_toolset(static_cast<AstralHandle>(ConversationHandle)) == ASTRAL_OK;
}

int32 UAstralConversation::StreamRead(TArray<uint8>& OutBuffer, uint32 TimeoutMs) {
  if (!IsValid() || OutBuffer.Num() == 0) {
    return static_cast<int32>(ASTRAL_E_INVALID);
  }

  AstralMutSpanU8 Span{};
  Span.data = OutBuffer.GetData();
  Span.len = static_cast<uint32_t>(OutBuffer.Num());
  return astral_conv_stream_read(static_cast<AstralHandle>(ConversationHandle), Span, TimeoutMs);
}

FString UAstralConversation::StreamReadString(int32 TimeoutMs) {
  const int32 BytesRead = StreamRead(TokenBuffer, static_cast<uint32>(FMath::Max(TimeoutMs, 0)));
  if (BytesRead <= 0) {
    return FString();
  }

  FUTF8ToTCHAR Converted(reinterpret_cast<const ANSICHAR*>(TokenBuffer.GetData()), BytesRead);
  return FString(Converted.Length(), Converted.Get());
}

FAstralConversationStats UAstralConversation::GetStats() const {
  FAstralConversationStats Result{};
  if (!IsValid()) {
    return Result;
  }

  AstralConvStats Native{};
  if (astral_conv_stats(static_cast<AstralHandle>(ConversationHandle), &Native) != ASTRAL_OK) {
    return Result;
  }

  Result.SlotId = static_cast<int32>(Native.slot_id);
  Result.PromptTokens = static_cast<int32>(Native.prompt_tokens);
  Result.KvTokens = static_cast<int32>(Native.kv_tokens);
  Result.GeneratedTokens = static_cast<int64>(Native.generated_tokens);
  Result.FirstTokenTimeMs = Native.t_first_token_ms;
  Result.TokensPerSecond = Native.tok_per_s;
  return Result;
}

bool UAstralConversation::TickStream(float DeltaTime) {
  TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Conversation_TickStream);
  (void)DeltaTime;

  if (!IsValid()) {
    TickerHandle.Reset();
    return false;
  }

  const FAstralConversationStreamReader Reader{this};
  const bool bKeepRunning = AstralRT::Private::FAstralSessionStreamPump::Tick(
      Reader, TokenBuffer, TickUtf8Buffer, TickTextScratch, StreamBytesNative, StreamTextNative,
      OnBytesReceived, OnTokenReceived);
  if (!bKeepRunning) {
    TickerHandle.Reset();
  }
  return bKeepRunning;
}

void UAstralConversation::UpdateTicker(bool bEnable) {
  if (bEnable) {
    if (!TickerHandle.IsValid()) {
      TickerHandle = FTSTicker::GetCoreTicker().AddTicker(
          FTickerDelegate::CreateUObject(this, &UAstralConversation::TickStream), 0.0f);
    }
    return;
  }

  if (TickerHandle.IsValid()) {
    FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
    TickerHandle.Reset();
  }
}
