#include "AstralSessionStreamPump.h"

#include "Containers/StringBuilder.h"
#include "Containers/UnrealString.h"

#include "astral_rt.h"

namespace AstralRT::Private {

bool FAstralSessionStreamPump::Tick(
    TFunctionRef<int32(TArray<uint8>&, uint32)> StreamReadFn,
    TArray<uint8>& TokenBuffer,
    TArray<uint8>& TickUtf8Buffer,
    FString& TickTextScratch,
    FAstralStreamBytesNative& StreamBytesNative,
    FAstralStreamTextNative& StreamTextNative,
    FAstralStreamBytesReceived& OnBytesReceived,
    FAstralTokenReceived& OnTokenReceived
)
{
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
        const int32 BytesRead = StreamReadFn(TokenBuffer, 0);
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

    return !stop_ticker;
}

} // namespace AstralRT::Private
