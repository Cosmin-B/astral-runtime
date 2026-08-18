#pragma once

#include "CoreMinimal.h"
#include "AstralTypes.h"
#include "Containers/ArrayView.h"
#include "Containers/Ticker.h"
#include "Containers/StringView.h"

#include "AstralSession.generated.h"

class UAstralModel;

/** Invoked by the core ticker. The byte view is valid only until the delegate returns. */
DECLARE_MULTICAST_DELEGATE_OneParam(FAstralStreamBytesNative, TConstArrayView<uint8>);
/** Invoked by the core ticker. The string view is valid only until the delegate returns. */
DECLARE_MULTICAST_DELEGATE_OneParam(FAstralStreamTextNative, FStringView);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FAstralStreamBytesReceived, const TArray<uint8>&, Bytes);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FAstralTokenReceived, const FString&, Token);

/**
 * UObject owner for a native Astral session handle.
 *
 * Streaming is pull-based. C++ callers can consume UTF-8 byte views through the
 * native delegates, while Blueprint delegates trade convenience for per-tick
 * marshaling. The session keeps the model handle value used at creation so
 * Reset can rebuild native state with the same model.
 * Call state-changing methods from one coordinating thread. The ticker invokes stream
 * delegates on the thread that ticks FTSTicker, normally the game thread. Keep the model
 * and runtime initialized until this UObject is destroyed.
 */
UCLASS(BlueprintType)
class ASTRALRT_API UAstralSession : public UObject
{
    GENERATED_BODY()

public:
    UAstralSession();

    /**
     * Create a native session from a loaded model. Stream ticking follows Desc.bStreamEnabled.
     * Returns false for an invalid model or runtime, an existing session, or a native creation
     * failure. The plugin writes the reason to LogAstralRT.
     */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Create(UAstralModel* Model, const FAstralSessionDesc& Desc);

    /** Feed text after converting FString to UTF-8. Returns false and logs the native error on failure. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool FeedPrompt(const FString& Prompt, bool bFinalize = true);

    /** Set the system prompt before user prompt text is fed. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool SetSystemPrompt(const FString& Prompt);

    /** Set the system prompt from caller-owned UTF-8 bytes. */
    bool SetSystemPromptRaw(TConstArrayView<uint8> Utf8Data);

    /** Feed caller-owned UTF-8 bytes. The native runtime copies or consumes them before return. */
    bool FeedPromptRaw(TConstArrayView<uint8> Utf8Data, bool bFinalize = true);

    /** Feed image bytes to a media-enabled model. Pixel data must be valid for the call duration. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool FeedImage(const FAstralImageDesc& Image, bool bFinalize = true);

    /** Feed audio bytes to a media-enabled model. FrameCount may be inferred from sample data. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool FeedAudio(const FAstralAudioDesc& Audio, bool bFinalize = true);

    /** Queue generation once. Returns before completion; read output through StreamRead or the tick delegates. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Decode();

    /** Request cancellation for the current decode. The request is asynchronous; Wait observes completion. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Cancel();

    /**
     * Block for at most TimeoutMs milliseconds. A value of 0 performs a nonblocking check.
     * Returns 0 for OK, -7 for canceled, -4 for timeout, or another native error code.
     * Do not block the game thread when it must continue pumping stream delegates.
     */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    int32 Wait(int32 TimeoutMs = 0);

    /** Reset native session state with new sampling and token limits for the same model. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool Reset(const FAstralSessionDesc& Desc);

    /** Replace sampler settings used by later decode steps. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool SetSampler(const FAstralSamplerDesc& Desc);

    /** Remove every adapter attached to this session. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    bool ClearAdapters();

    /** Attach a model-scoped adapter handle returned by UAstralModel::LoadAdapter. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    bool AddAdapter(int64 AdapterHandle, float Scale = 1.0f);

    /** Return the number of adapters currently attached to this session. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    bool GetAdapterCount(int32& OutCount) const;

    /** Return one attached adapter handle and scale by index. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    bool GetAdapter(int32 Index, int64& OutAdapterHandle, float& OutScale) const;

    /** Update one attached adapter scale between requests. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    bool SetAdapterScale(int32 Index, float Scale);

    /** Bind a native toolset handle for structured output. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    bool SetToolset(int64 ToolsetHandle, EAstralToolChoiceMode ChoiceMode = EAstralToolChoiceMode::Auto);

    /** Clear any structured-output toolset binding. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    bool ClearToolset();

    /** Clear all native stop sequences for this session. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool StopClear();

    /** Add a stop sequence as caller-owned UTF-8 bytes. */
    bool StopAddUtf8Bytes(TConstArrayView<uint8> Utf8Data);

    /** Add a stop sequence after converting FString to UTF-8. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    bool StopAddString(const FString& Utf8Text);

    /** Constrain later generation with a GBNF grammar and optional root symbol. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Grammar")
    bool SetGrammarGbnf(const FString& Grammar, const FString& RootSymbol = TEXT("root"));

    /** Constrain later generation with a JSON schema. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Grammar")
    bool SetGrammarJsonSchema(const FString& JsonSchema);

    /** Remove the active grammar constraint. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Grammar")
    bool ClearGrammar();

    /**
     * Read streamed UTF-8 bytes into caller-owned OutBuffer. TimeoutMs is in milliseconds,
     * and 0 performs a nonblocking read. Returns the byte count, 0 when no bytes are available,
     * or a negative native error code. Use one stream consumer to preserve byte ordering.
     */
    int32 StreamRead(TArray<uint8>& OutBuffer, uint32 TimeoutMs = 0);

    /** Read one byte chunk as FString. Empty may mean timeout, end of stream, or error. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    FString StreamReadString(int32 TimeoutMs = 0);

    /** Snapshot native timing and memory counters. Zeroed fields mean stats were unavailable. */
    UFUNCTION(BlueprintCallable, Category = "Astral")
    FAstralStats GetStats() const;

    /** True while this object owns a native session from the current runtime generation. */
    UFUNCTION(BlueprintPure, Category = "Astral")
    bool IsValid() const;

    uint64 GetHandle() const { return IsValid() ? SessionHandle : 0; }

    /** Raw UTF-8 delivery on the core ticker thread. The view is borrowed for the callback only. */
    FAstralStreamBytesNative& OnStreamBytesNative() { return StreamBytesNative; }

    /** Decoded delivery on the core ticker thread. The view is borrowed for the callback only. */
    FAstralStreamTextNative& OnStreamTextNative() { return StreamTextNative; }

    /** Raw UTF-8 delivery on the core ticker thread. The array belongs to the delegate invocation. */
    UPROPERTY(BlueprintAssignable, Category = "Astral")
    FAstralStreamBytesReceived OnBytesReceived;

    /** Decoded delivery on the core ticker thread; allocates an FString per tick when bound. */
    UPROPERTY(BlueprintAssignable, Category = "Astral")
    FAstralTokenReceived OnTokenReceived;

    virtual void BeginDestroy() override;

private:
    bool IsCurrentRuntimeGeneration() const;

    uint64 SessionHandle = 0;
    uint64 ModelHandle = 0;
    uint64 RuntimeGeneration = 0;

    TArray<uint8> TokenBuffer;
    TArray<uint8> TickUtf8Buffer;
    FString TickTextScratch;

    FAstralStreamBytesNative StreamBytesNative;
    FAstralStreamTextNative StreamTextNative;

    FTSTicker::FDelegateHandle TickerHandle;
    bool TickStream(float DeltaTime);
    void UpdateTicker(bool bEnable);
};
