#pragma once

#include "CoreMinimal.h"
#include "AstralTypes.h"
#include "Kismet/BlueprintFunctionLibrary.h"

#include "AstralMediaLibrary.generated.h"

class UTexture2D;

/** Boundary helpers that build Astral media descriptors from Unreal-owned payloads. */
UCLASS()
class ASTRALRT_API UAstralMediaLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /** Copy tightly packed RGBA8 bytes into an image descriptor. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Media")
    static bool MakeRGBA8ImageFromBytes(const TArray<uint8>& RgbaBytes,
                                        int32 Width,
                                        int32 Height,
                                        FAstralImageDesc& OutImage);

    /** Copy a CPU-readable PF_B8G8R8A8 texture mip into an RGBA8 image descriptor. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Media")
    static bool MakeRGBA8ImageFromTexture(UTexture2D* Texture, FAstralImageDesc& OutImage);

    /** Copy interleaved signed 16-bit PCM bytes into an audio descriptor. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Media")
    static bool MakePCM16AudioFromBytes(const TArray<uint8>& PcmBytes,
                                        int32 Channels,
                                        int32 SampleRate,
                                        FAstralAudioDesc& OutAudio);
};
