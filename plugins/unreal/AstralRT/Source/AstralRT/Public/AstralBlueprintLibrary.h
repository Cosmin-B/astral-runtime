#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"

#include "AstralBlueprintLibrary.generated.h"

class UAstralEmbedder;
class UAstralModel;
class UAstralSession;

/** Blueprint entry points for common Astral object creation and diagnostics. */
UCLASS()
class ASTRALRT_API UAstralBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /** Create a model wrapper owned by Outer, or by the transient package when Outer is null. */
    UFUNCTION(BlueprintCallable, Category = "Astral", meta = (DefaultToSelf = "Outer"))
    static UAstralModel* CreateAstralModel(UObject* Outer);

    /** Create a session wrapper owned by Outer, or by the transient package when Outer is null. */
    UFUNCTION(BlueprintCallable, Category = "Astral", meta = (DefaultToSelf = "Outer"))
    static UAstralSession* CreateAstralSession(UObject* Outer);

    /** Create an embedder wrapper owned by Outer, or by the transient package when Outer is null. */
    UFUNCTION(BlueprintCallable, Category = "Astral", meta = (DefaultToSelf = "Outer"))
    static UAstralEmbedder* CreateAstralEmbedder(UObject* Outer);

    /** Return the last native Astral error string for the current thread when available. */
    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static FString GetLastAstralError();

    /** Return a stable symbolic name for a native Astral error code. */
    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static FString ErrorCodeName(int32 ErrorCode);

    /** Maximum number of adapters that can be attached to one session. */
    UFUNCTION(BlueprintPure, Category = "Astral|Adapters")
    static int32 MaxSessionAdapters();

    /** True when the capability bitmask contains embeddings support. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasEmbeddings(int64 Caps);

    /** True when extended sampler controls are available for generation. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasSamplerControls(int64 Caps);

    /** True when native stop sequences are available for generation. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasStopSequences(int64 Caps);

    /** True when the capability bitmask contains GPU offload support. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasGpuOffload(int64 Caps);

    /** True when LoRA or adapter loading is supported. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasLora(int64 Caps);

    /** True when the capability bitmask contains image input support. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasImageInput(int64 Caps);

    /** True when the capability bitmask contains audio input support. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasAudioInput(int64 Caps);

    /** True when the capability bitmask contains multimodal embedding support. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasMultimodalEmbeddings(int64 Caps);

    /** True when the capability bitmask contains grammar-constrained decoding support. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasGrammar(int64 Caps);

    /** True when per-token log probability metadata is supported. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasLogprobs(int64 Caps);

    /** True when the capability bitmask contains KV/state save-load support. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasKvState(int64 Caps);

    /** True when slot or sequence selection is supported. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasSlots(int64 Caps);

    /** True when GBNF grammar constraints are supported. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasGbnfGrammar(int64 Caps);

    /** True when JSON schema grammar constraints are supported. */
    UFUNCTION(BlueprintPure, Category = "Astral|Capabilities")
    static bool HasJsonSchemaGrammar(int64 Caps);
};
