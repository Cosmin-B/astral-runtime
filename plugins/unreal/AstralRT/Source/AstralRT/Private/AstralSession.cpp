#include "AstralSession.h"
#include "AstralLog.h"
#include "AstralSessionStreamPump.h"
#include "AstralModel.h"
#include "IAstralRT.h"

#include "Containers/Ticker.h"
#include "Containers/UnrealString.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

#include "astral_rt.h"

namespace {

struct FAstralSessionStreamReader
{
    UAstralSession* Session = nullptr;

    int32 operator()(TArray<uint8>& OutBuffer, uint32 TimeoutMs) const
    {
        return Session->StreamRead(OutBuffer, TimeoutMs);
    }
};

} // namespace

UAstralSession::UAstralSession()
{
    TokenBuffer.SetNumUninitialized(4096);
    TickUtf8Buffer.Reserve(4096);
    TickTextScratch.Reserve(4096);
}

bool UAstralSession::IsCurrentRuntimeGeneration() const
{
    return IAstralRT::IsAvailable() &&
        IAstralRT::Get().IsInitialized() &&
        IAstralRT::Get().GetRuntimeGeneration() == RuntimeGeneration;
}

bool UAstralSession::IsValid() const
{
    return SessionHandle != 0 && IsCurrentRuntimeGeneration();
}

void UAstralSession::BeginDestroy()
{
    if (TickerHandle.IsValid())
    {
        FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
        TickerHandle.Reset();
    }

    if (SessionHandle != 0 && IsCurrentRuntimeGeneration())
    {
        astral_session_destroy(static_cast<AstralHandle>(SessionHandle));
    }

    SessionHandle = 0;
    ModelHandle = 0;
    RuntimeGeneration = 0;

    Super::BeginDestroy();
}

bool UAstralSession::Create(UAstralModel* Model, const FAstralSessionDesc& Desc)
{
    if (SessionHandle != 0)
    {
        if (!IsCurrentRuntimeGeneration())
        {
            SessionHandle = 0;
            ModelHandle = 0;
            RuntimeGeneration = 0;
        }
        else
        {
            UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: session already created"));
            return false;
        }
    }

    if (Model == nullptr || !Model->IsValid())
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: invalid model"));
        return false;
    }

    if (!IAstralRT::IsAvailable() || !IAstralRT::Get().IsInitialized())
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: runtime not initialized"));
        return false;
    }

    AstralSessionDesc NativeDesc{};
    NativeDesc.model = static_cast<AstralHandle>(Model->GetHandle());
    NativeDesc.max_tokens = static_cast<uint32_t>(Desc.MaxTokens);
    NativeDesc.temperature = Desc.Temperature;
    NativeDesc.top_k = static_cast<uint32_t>(Desc.TopK);
    NativeDesc.top_p = Desc.TopP;
    NativeDesc.stream_enabled = Desc.bStreamEnabled ? 1 : 0;
    NativeDesc.seed = static_cast<uint32_t>(Desc.Seed);

    AstralHandle Out = 0;
    const AstralErr Err = astral_session_create(&NativeDesc, &Out);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_create failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    SessionHandle = static_cast<uint64>(Out);
    ModelHandle = Model->GetHandle();
    RuntimeGeneration = IAstralRT::Get().GetRuntimeGeneration();

    UpdateTicker(Desc.bStreamEnabled);
    return true;
}

bool UAstralSession::FeedPrompt(const FString& Prompt, bool bFinalize)
{
    FTCHARToUTF8 Utf8(*Prompt);
    return FeedPromptRaw(TConstArrayView<uint8>(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length()), bFinalize);
}

bool UAstralSession::SetSystemPrompt(const FString& Prompt)
{
    FTCHARToUTF8 Utf8(*Prompt);
    return SetSystemPromptRaw(TConstArrayView<uint8>(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length()));
}

bool UAstralSession::SetSystemPromptRaw(TConstArrayView<uint8> Utf8Data)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralSession_SetSystemPrompt);

    if (!IsValid())
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: session not created"));
        return false;
    }

    AstralSpanU8 Span{};
    Span.data = Utf8Data.GetData();
    Span.len = static_cast<uint32_t>(Utf8Data.Num());

    const AstralErr Err = astral_session_set_system_prompt(static_cast<AstralHandle>(SessionHandle), Span);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_set_system_prompt failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::FeedPromptRaw(TConstArrayView<uint8> Utf8Data, bool bFinalize)
{
    if (!IsValid())
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: session not created"));
        return false;
    }

    AstralSpanU8 Span{};
    Span.data = Utf8Data.GetData();
    Span.len = static_cast<uint32_t>(Utf8Data.Num());

    const AstralErr Err =
        astral_session_feed(static_cast<AstralHandle>(SessionHandle), Span, bFinalize ? 1 : 0);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_feed failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::FeedImage(const FAstralImageDesc& Image, bool bFinalize)
{
    if (!IsValid())
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: session not created"));
        return false;
    }

    if (Image.Pixels.Num() == 0)
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: image pixels are empty"));
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

    const AstralErr Err =
        astral_session_feed_image(static_cast<AstralHandle>(SessionHandle), &Native, bFinalize ? 1 : 0);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_feed_image failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::FeedAudio(const FAstralAudioDesc& Audio, bool bFinalize)
{
    if (!IsValid())
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: session not created"));
        return false;
    }

    if (Audio.Samples.Num() == 0)
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: audio samples are empty"));
        return false;
    }

    AstralAudioDesc Native{};
    Native.size = sizeof(AstralAudioDesc);
    Native.format = static_cast<AstralAudioFormat>(Audio.Format);
    if (Audio.Channels == 0)
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: audio channels must be > 0"));
        return false;
    }

    uint64 FrameCount = static_cast<uint64>(Audio.FrameCount);
    if (FrameCount == 0)
    {
        const uint32 BytesPerSample = (Audio.Format == EAstralAudioFormat::F32) ? 4u : 2u;
        const uint64 TotalSamples = static_cast<uint64>(Audio.Samples.Num()) / BytesPerSample;
        if (TotalSamples % Audio.Channels != 0)
        {
            UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: audio samples not aligned to channels"));
            return false;
        }
        FrameCount = TotalSamples / Audio.Channels;
    }

    Native.channels = static_cast<uint32_t>(Audio.Channels);
    Native.sample_rate = static_cast<uint32_t>(Audio.SampleRate);
    Native.frame_count = FrameCount;
    Native.samples.data = Audio.Samples.GetData();
    Native.samples.len = static_cast<uint32_t>(Audio.Samples.Num());
    Native.flags = static_cast<uint32_t>(Audio.Flags);
    Native.gpu_device = Audio.GpuDevice;
    Native.gpu_route_flags = static_cast<uint32_t>(Audio.GpuRouteFlags);
    Native.gpu_device_mask = static_cast<uint64_t>(Audio.GpuDeviceMask);
    Native.gpu_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(Audio.GpuStream));

    const AstralErr Err =
        astral_session_feed_audio(static_cast<AstralHandle>(SessionHandle), &Native, bFinalize ? 1 : 0);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_feed_audio failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::Decode()
{
    if (!IsValid())
    {
        UE_LOG(LogAstralRT, Warning, TEXT("AstralRT: session not created"));
        return false;
    }

    const AstralErr Err = astral_session_decode(static_cast<AstralHandle>(SessionHandle));
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_decode failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::Cancel()
{
    if (!IsValid())
    {
        return false;
    }
    const AstralErr Err = astral_session_cancel(static_cast<AstralHandle>(SessionHandle));
    return Err == ASTRAL_OK;
}

int32 UAstralSession::Wait(int32 TimeoutMs)
{
    if (!IsValid())
    {
        return static_cast<int32>(ASTRAL_E_INVALID);
    }
    const uint32 NativeTimeoutMs = static_cast<uint32>(FMath::Max(TimeoutMs, 0));
    return static_cast<int32>(astral_session_wait(static_cast<AstralHandle>(SessionHandle), NativeTimeoutMs));
}

bool UAstralSession::Reset(const FAstralSessionDesc& Desc)
{
    if (!IsValid() || ModelHandle == 0)
    {
        return false;
    }

    AstralSessionDesc NativeDesc{};
    NativeDesc.model = static_cast<AstralHandle>(ModelHandle);
    NativeDesc.max_tokens = static_cast<uint32_t>(Desc.MaxTokens);
    NativeDesc.temperature = Desc.Temperature;
    NativeDesc.top_k = static_cast<uint32_t>(Desc.TopK);
    NativeDesc.top_p = Desc.TopP;
    NativeDesc.stream_enabled = Desc.bStreamEnabled ? 1 : 0;
    NativeDesc.seed = static_cast<uint32_t>(Desc.Seed);

    const AstralErr Err = astral_session_reset(static_cast<AstralHandle>(SessionHandle), &NativeDesc);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_reset failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    UpdateTicker(Desc.bStreamEnabled);
    return true;
}

bool UAstralSession::SetSampler(const FAstralSamplerDesc& Desc)
{
    if (!IsValid())
    {
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
    Native.mirostat = 0;
    Native.mirostat_tau = 0.0f;
    Native.mirostat_eta = 0.0f;

    const AstralErr Err = astral_session_set_sampler(static_cast<AstralHandle>(SessionHandle), &Native);
    return Err == ASTRAL_OK;
}

bool UAstralSession::ClearAdapters()
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralSession_ClearAdapters);

    if (!IsValid())
    {
        return false;
    }

    const AstralErr Err = astral_session_adapters_clear(static_cast<AstralHandle>(SessionHandle));
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_adapters_clear failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::AddAdapter(int64 AdapterHandle, float Scale)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralSession_AddAdapter);

    if (!IsValid() || AdapterHandle == 0)
    {
        return false;
    }

    const AstralErr Err = astral_session_adapters_add(
        static_cast<AstralHandle>(SessionHandle),
        static_cast<AstralHandle>(AdapterHandle),
        Scale
    );
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_session_adapters_add failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::GetAdapterCount(int32& OutCount) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralSession_GetAdapterCount);

    OutCount = 0;
    if (!IsValid())
    {
        return false;
    }

    uint32 Count = 0;
    const AstralErr Err = astral_session_adapters_count(static_cast<AstralHandle>(SessionHandle), &Count);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutCount = static_cast<int32>(Count);
    return true;
}

bool UAstralSession::GetAdapter(int32 Index, int64& OutAdapterHandle, float& OutScale) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralSession_GetAdapter);

    OutAdapterHandle = 0;
    OutScale = 0.0f;
    if (!IsValid() || Index < 0)
    {
        return false;
    }

    AstralHandle Adapter = 0;
    float Scale = 0.0f;
    const AstralErr Err = astral_session_adapters_get(
        static_cast<AstralHandle>(SessionHandle),
        static_cast<uint32>(Index),
        &Adapter,
        &Scale
    );
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutAdapterHandle = static_cast<int64>(Adapter);
    OutScale = Scale;
    return true;
}

bool UAstralSession::StopClear()
{
    if (!IsValid())
    {
        return false;
    }
    const AstralErr Err = astral_session_stop_clear(static_cast<AstralHandle>(SessionHandle));
    return Err == ASTRAL_OK;
}

bool UAstralSession::StopAddUtf8Bytes(TConstArrayView<uint8> Utf8Data)
{
    if (!IsValid())
    {
        return false;
    }

    AstralSpanU8 Span{};
    Span.data = Utf8Data.GetData();
    Span.len = static_cast<uint32_t>(Utf8Data.Num());
    const AstralErr Err = astral_session_stop_add_utf8(static_cast<AstralHandle>(SessionHandle), Span);
    return Err == ASTRAL_OK;
}

bool UAstralSession::StopAddString(const FString& Utf8Text)
{
    FTCHARToUTF8 Utf8(*Utf8Text);
    return StopAddUtf8Bytes(TConstArrayView<uint8>(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length()));
}

int32 UAstralSession::StreamRead(TArray<uint8>& OutBuffer, uint32 TimeoutMs)
{
    if (!IsValid())
    {
        return static_cast<int32>(ASTRAL_E_INVALID);
    }

    if (OutBuffer.Num() == 0)
    {
        return static_cast<int32>(ASTRAL_E_INVALID);
    }

    AstralMutSpanU8 Span{};
    Span.data = OutBuffer.GetData();
    Span.len = static_cast<uint32_t>(OutBuffer.Num());

    return astral_stream_read(static_cast<AstralHandle>(SessionHandle), Span, TimeoutMs);
}

FString UAstralSession::StreamReadString(int32 TimeoutMs)
{
    const uint32 NativeTimeoutMs = static_cast<uint32>(FMath::Max(TimeoutMs, 0));
    const int32 BytesRead = StreamRead(TokenBuffer, NativeTimeoutMs);
    if (BytesRead <= 0)
    {
        return FString();
    }

    FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(TokenBuffer.GetData()), BytesRead);
    return FString(Converter.Length(), Converter.Get());
}

FAstralStats UAstralSession::GetStats() const
{
    FAstralStats Result{};

    if (!IsValid())
    {
        return Result;
    }

    AstralStats Native{};
    const AstralErr Err = astral_session_stats(static_cast<AstralHandle>(SessionHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return Result;
    }

    Result.InitTimeMs = Native.t_init_ms;
    Result.FirstTokenTimeMs = Native.t_first_token_ms;
    Result.TokensPerSecond = Native.tok_per_s;
    Result.BytesCommitted = static_cast<int64>(Native.bytes_committed);
    Result.BytesReserved = static_cast<int64>(Native.bytes_reserved);

    return Result;
}

bool UAstralSession::TickStream(float DeltaTime)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Session_TickStream);

    (void)DeltaTime;

    if (!IsValid())
    {
        TickerHandle.Reset();
        return false;
    }

    const FAstralSessionStreamReader Reader{this};
    const bool keep_running = AstralRT::Private::FAstralSessionStreamPump::Tick(
        Reader,
        TokenBuffer,
        TickUtf8Buffer,
        TickTextScratch,
        StreamBytesNative,
        StreamTextNative,
        OnBytesReceived,
        OnTokenReceived
    );
    if (!keep_running)
    {
        TickerHandle.Reset();
        return false;
    }

    return true;
}

void UAstralSession::UpdateTicker(bool bEnable)
{
    if (bEnable)
    {
        if (!TickerHandle.IsValid())
        {
            TickerHandle = FTSTicker::GetCoreTicker().AddTicker(
                FTickerDelegate::CreateUObject(this, &UAstralSession::TickStream),
                0.0f
            );
        }
        return;
    }

    if (TickerHandle.IsValid())
    {
        FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
        TickerHandle.Reset();
    }
}
