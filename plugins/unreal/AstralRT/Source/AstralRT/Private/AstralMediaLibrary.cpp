#include "AstralMediaLibrary.h"

#include "Engine/Texture2D.h"
#include "PixelFormat.h"

bool UAstralMediaLibrary::MakeRGBA8ImageFromBytes(const TArray<uint8>& RgbaBytes,
                                                   int32 Width,
                                                   int32 Height,
                                                   FAstralImageDesc& OutImage)
{
    OutImage = FAstralImageDesc{};

    if (Width <= 0 || Height <= 0)
    {
        return false;
    }

    const int64 ExpectedBytes = static_cast<int64>(Width) * static_cast<int64>(Height) * 4;
    if (ExpectedBytes <= 0 || RgbaBytes.Num() != ExpectedBytes)
    {
        return false;
    }

    OutImage.Format = EAstralImageFormat::RGBA8;
    OutImage.Width = Width;
    OutImage.Height = Height;
    OutImage.RowStride = Width * 4;
    OutImage.Pixels = RgbaBytes;
    return true;
}

bool UAstralMediaLibrary::MakeRGBA8ImageFromTexture(UTexture2D* Texture, FAstralImageDesc& OutImage)
{
    OutImage = FAstralImageDesc{};

    if (Texture == nullptr || Texture->GetPlatformData() == nullptr || Texture->GetPlatformData()->Mips.Num() == 0)
    {
        return false;
    }

    FTexturePlatformData* PlatformData = Texture->GetPlatformData();
    if (PlatformData->PixelFormat != PF_B8G8R8A8)
    {
        return false;
    }

    FTexture2DMipMap& Mip = PlatformData->Mips[0];
    const int32 Width = Mip.SizeX;
    const int32 Height = Mip.SizeY;
    if (Width <= 0 || Height <= 0)
    {
        return false;
    }

    const int64 PixelCount = static_cast<int64>(Width) * static_cast<int64>(Height);
    const int64 ByteCount = PixelCount * 4;
    if (PixelCount <= 0 || ByteCount <= 0 || ByteCount > TNumericLimits<int32>::Max())
    {
        return false;
    }

    const void* Locked = Mip.BulkData.LockReadOnly();
    if (Locked == nullptr)
    {
        Mip.BulkData.Unlock();
        return false;
    }

    OutImage.Format = EAstralImageFormat::RGBA8;
    OutImage.Width = Width;
    OutImage.Height = Height;
    OutImage.RowStride = Width * 4;
    OutImage.Pixels.SetNumUninitialized(static_cast<int32>(ByteCount));

    const uint8* Src = static_cast<const uint8*>(Locked);
    uint8* Dst = OutImage.Pixels.GetData();
    for (int64 i = 0; i < PixelCount; ++i)
    {
        const int64 Offset = i * 4;
        Dst[Offset + 0] = Src[Offset + 2];
        Dst[Offset + 1] = Src[Offset + 1];
        Dst[Offset + 2] = Src[Offset + 0];
        Dst[Offset + 3] = Src[Offset + 3];
    }

    Mip.BulkData.Unlock();
    return true;
}

bool UAstralMediaLibrary::MakePCM16AudioFromBytes(const TArray<uint8>& PcmBytes,
                                                   int32 Channels,
                                                   int32 SampleRate,
                                                   FAstralAudioDesc& OutAudio)
{
    OutAudio = FAstralAudioDesc{};

    if (Channels <= 0 || SampleRate <= 0 || PcmBytes.Num() == 0)
    {
        return false;
    }

    const int64 BytesPerFrame = static_cast<int64>(Channels) * static_cast<int64>(sizeof(int16));
    if ((static_cast<int64>(PcmBytes.Num()) % BytesPerFrame) != 0)
    {
        return false;
    }

    OutAudio.Format = EAstralAudioFormat::I16;
    OutAudio.Channels = Channels;
    OutAudio.SampleRate = SampleRate;
    OutAudio.FrameCount = static_cast<int64>(PcmBytes.Num()) / BytesPerFrame;
    OutAudio.Samples = PcmBytes;
    return true;
}
