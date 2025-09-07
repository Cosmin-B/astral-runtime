#include "AstralSession.h"
#include "AstralModel.h"
#include "IAstralRT.h"

#include "Containers/Ticker.h"
#include "Containers/StringBuilder.h"
#include "Containers/UnrealString.h"

#include "astral_rt.h"

UAstralSession::UAstralSession()
{
    TokenBuffer.SetNumUninitialized(4096);
    TickUtf8Buffer.Reserve(4096);
    TickTextScratch.Reserve(4096);
}

void UAstralSession::BeginDestroy()
{
    if (TickerHandle.IsValid())
    {
        FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
        TickerHandle.Reset();
    }

    if (SessionHandle != 0 && IAstralRT::IsAvailable() && IAstralRT::Get().IsInitialized())
    {
        astral_session_destroy(static_cast<AstralHandle>(SessionHandle));
    }

    SessionHandle = 0;
    ModelHandle = 0;

    Super::BeginDestroy();
}

bool UAstralSession::Create(UAstralModel* Model, const FAstralSessionDesc& Desc)
{
    if (SessionHandle != 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("AstralRT: session already created"));
        return false;
    }

    if (Model == nullptr || !Model->IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: invalid model"));
        return false;
    }

    if (!IAstralRT::IsAvailable() || !IAstralRT::Get().IsInitialized())
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: runtime not initialized"));
        return false;
    }

    AstralSessionDesc NativeDesc{};
    NativeDesc.model = static_cast<AstralHandle>(Model->GetHandle());
    NativeDesc.max_tokens = Desc.MaxTokens;
    NativeDesc.temperature = Desc.Temperature;
    NativeDesc.top_k = Desc.TopK;
    NativeDesc.top_p = Desc.TopP;
    NativeDesc.stream_enabled = Desc.bStreamEnabled ? 1 : 0;
    NativeDesc.seed = Desc.Seed;

    AstralHandle Out = 0;
    const AstralErr Err = astral_session_create(&NativeDesc, &Out);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: astral_session_create failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    SessionHandle = static_cast<uint64>(Out);
    ModelHandle = Model->GetHandle();

    UpdateTicker(Desc.bStreamEnabled);
    return true;
}

bool UAstralSession::FeedPrompt(const FString& Prompt, bool bFinalize)
{
    FTCHARToUTF8 Utf8(*Prompt);
    return FeedPromptRaw(TConstArrayView<uint8>(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length()), bFinalize);
}

bool UAstralSession::FeedPromptRaw(TConstArrayView<uint8> Utf8Data, bool bFinalize)
{
    if (SessionHandle == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("AstralRT: session not created"));
        return false;
    }

    AstralSpanU8 Span{};
    Span.data = Utf8Data.GetData();
    Span.len = static_cast<uint32_t>(Utf8Data.Num());

    const AstralErr Err =
        astral_session_feed(static_cast<AstralHandle>(SessionHandle), Span, bFinalize ? 1 : 0);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: astral_session_feed failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::Decode()
{
    if (SessionHandle == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("AstralRT: session not created"));
        return false;
    }

    const AstralErr Err = astral_session_decode(static_cast<AstralHandle>(SessionHandle));
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: astral_session_decode failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    return true;
}

bool UAstralSession::Cancel()
{
    if (SessionHandle == 0)
    {
        return false;
    }
    const AstralErr Err = astral_session_cancel(static_cast<AstralHandle>(SessionHandle));
    return Err == ASTRAL_OK;
}

int32 UAstralSession::Wait(uint32 TimeoutMs)
{
    if (SessionHandle == 0)
    {
        return static_cast<int32>(ASTRAL_E_INVALID);
    }
    return static_cast<int32>(astral_session_wait(static_cast<AstralHandle>(SessionHandle), TimeoutMs));
}

bool UAstralSession::Reset(const FAstralSessionDesc& Desc)
{
    if (SessionHandle == 0 || ModelHandle == 0)
    {
        return false;
    }

    AstralSessionDesc NativeDesc{};
    NativeDesc.model = static_cast<AstralHandle>(ModelHandle);
    NativeDesc.max_tokens = Desc.MaxTokens;
    NativeDesc.temperature = Desc.Temperature;
    NativeDesc.top_k = Desc.TopK;
    NativeDesc.top_p = Desc.TopP;
    NativeDesc.stream_enabled = Desc.bStreamEnabled ? 1 : 0;
    NativeDesc.seed = Desc.Seed;

    const AstralErr Err = astral_session_reset(static_cast<AstralHandle>(SessionHandle), &NativeDesc);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: astral_session_reset failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    UpdateTicker(Desc.bStreamEnabled);
    return true;
}

bool UAstralSession::SetSampler(const FAstralSamplerDesc& Desc)
{
    if (SessionHandle == 0)
    {
        return false;
    }

    AstralSamplerDesc Native{};
    Native.size = sizeof(AstralSamplerDesc);
    Native.temperature = Desc.Temperature;
    Native.top_k = Desc.TopK;
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

bool UAstralSession::StopClear()
{
    if (SessionHandle == 0)
    {
        return false;
    }
    const AstralErr Err = astral_session_stop_clear(static_cast<AstralHandle>(SessionHandle));
    return Err == ASTRAL_OK;
}

bool UAstralSession::StopAddUtf8Bytes(TConstArrayView<uint8> Utf8Data)
{
    if (SessionHandle == 0)
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
    if (SessionHandle == 0)
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

FString UAstralSession::StreamReadString(uint32 TimeoutMs)
{
    const int32 BytesRead = StreamRead(TokenBuffer, TimeoutMs);
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

    if (SessionHandle == 0)
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
    Result.BytesCommitted = Native.bytes_committed;
    Result.BytesReserved = Native.bytes_reserved;

    return Result;
}

bool UAstralSession::TickStream(float DeltaTime)
{
    (void)DeltaTime;

    if (SessionHandle == 0)
    {
        TickerHandle.Reset();
        return false;
    }

    const bool want_tick_utf8 =
        StreamBytesNative.IsBound() || OnBytesReceived.IsBound() || OnTokenReceived.IsBound() || StreamTextNative.IsBound();

    if (want_tick_utf8)
    {
        TickUtf8Buffer.Reset();
    }

    constexpr uint32 MaxReadsPerTick = 128;
    constexpr uint32 MaxBytesPerTick = 64u * 1024u;

    uint32 total_bytes = 0;
    bool stop_ticker = false;

    for (uint32 i = 0; i < MaxReadsPerTick; ++i)
    {
        const int32 BytesRead = StreamRead(TokenBuffer, 0);
        if (BytesRead == ASTRAL_E_TIMEOUT)
        {
            break;
        }

        if (BytesRead < 0)
        {
            UE_LOG(LogTemp, Error, TEXT("AstralRT: astral_stream_read failed (%d)"), BytesRead);
            stop_ticker = true;
            break;
        }

        if (BytesRead == 0)
        {
            stop_ticker = true;
            break;
        }

        total_bytes += static_cast<uint32>(BytesRead);

        if (want_tick_utf8)
        {
            TickUtf8Buffer.Append(TokenBuffer.GetData(), BytesRead);
        }

        if (total_bytes >= MaxBytesPerTick)
        {
            break;
        }
    }

    if (want_tick_utf8 && TickUtf8Buffer.Num() > 0)
    {
        if (StreamBytesNative.IsBound())
        {
            StreamBytesNative.Broadcast(TConstArrayView<uint8>(TickUtf8Buffer.GetData(), TickUtf8Buffer.Num()));
        }

        if (OnBytesReceived.IsBound())
        {
            OnBytesReceived.Broadcast(TickUtf8Buffer);
        }

        if (OnTokenReceived.IsBound() || StreamTextNative.IsBound())
        {
            FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(TickUtf8Buffer.GetData()), TickUtf8Buffer.Num());

            if (StreamTextNative.IsBound())
            {
                TStringBuilder<4096> Text;
                Text.Append(Converter.Get(), Converter.Length());
                StreamTextNative.Broadcast(Text.ToView());
            }

            if (OnTokenReceived.IsBound())
            {
                TickTextScratch.Reset();
                TickTextScratch.Reserve(Converter.Length());
                TickTextScratch.AppendChars(Converter.Get(), Converter.Length());
                OnTokenReceived.Broadcast(TickTextScratch);
            }
        }
    }

    if (stop_ticker)
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
