#pragma once

#include "AstralSession.h"

#include "Templates/Function.h"

namespace AstralRT::Private {

struct FAstralSessionStreamPump
{
    static bool Tick(
        TFunctionRef<int32(TArray<uint8>&, uint32)> StreamReadFn,
        TArray<uint8>& TokenBuffer,
        TArray<uint8>& TickUtf8Buffer,
        FString& TickTextScratch,
        FAstralStreamBytesNative& StreamBytesNative,
        FAstralStreamTextNative& StreamTextNative,
        FAstralStreamBytesReceived& OnBytesReceived,
        FAstralTokenReceived& OnTokenReceived
    );
};

} // namespace AstralRT::Private
