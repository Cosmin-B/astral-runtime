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

    /** Create a native structured-output toolset. Release it with DestroyToolset. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    static bool CreateToolset(const TArray<FAstralToolDesc>& Tools, EAstralToolChoiceMode ChoiceMode, int64& OutToolsetHandle);

    /** Release a native structured-output toolset handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    static void DestroyToolset(int64 ToolsetHandle);

    /** Parse a completed tool-call JSON payload against a native toolset. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    static bool ParseToolCall(int64 ToolsetHandle, const FString& GeneratedText, FAstralToolCallResult& OutResult);

    /** Split UTF-8 text into native byte ranges. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Chunking")
    static bool ChunkText(const FString& Text, const FAstralChunkerDesc& Desc, TArray<FAstralChunkRange>& OutRanges, int32& OutErrorCode);

    /** Split an already-tokenized sequence into token ranges. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Chunking")
    static bool ChunkTokens(int32 TokenCount, const FAstralChunkerDesc& Desc, TArray<FAstralChunkRange>& OutRanges, int32& OutErrorCode);

    /** Create a native flat vector memory index. Release it with DestroyMemoryIndex. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool CreateMemoryIndex(const FAstralMemoryIndexDesc& Desc, int64& OutMemoryHandle, int32& OutErrorCode);

    /** Release a native memory index handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static void DestroyMemoryIndex(int64 MemoryHandle);

    /** Add records and row-major vectors to a native memory index. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool AddMemoryBatch(
        int64 MemoryHandle,
        const TArray<FAstralMemoryRecord>& Records,
        const TArray<float>& Vectors,
        int32 Dimension,
        int32& OutErrorCode
    );

    /** Search a native memory index with one query vector. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool SearchMemoryIndex(
        int64 MemoryHandle,
        const TArray<float>& Query,
        int32 TopK,
        int32 GroupId,
        TArray<FAstralMemorySearchResult>& OutResults,
        int32& OutErrorCode
    );

    /** Start an incremental memory search. Release it with EndMemorySearch. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool BeginMemorySearch(
        int64 MemoryHandle,
        const TArray<float>& Query,
        int32 TopK,
        int32 GroupId,
        int64& OutCursorHandle,
        int32& OutErrorCode
    );

    /** Fetch the next batch from an incremental memory search. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool FetchMemorySearch(
        int64 CursorHandle,
        int32 MaxResults,
        TArray<FAstralMemorySearchResult>& OutResults,
        int32& OutErrorCode
    );

    /** Release an incremental memory search handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static void EndMemorySearch(int64 CursorHandle);

    /** Create a native agent over an Astral model/executor. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool CreateAgent(const FAstralAgentDesc& Desc, int64& OutAgentHandle, int32& OutErrorCode);

    /** Release a native agent handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static void DestroyAgent(int64 AgentHandle);

    /** Store a native-owned system prompt for the agent. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool SetAgentSystemPrompt(int64 AgentHandle, const FString& SystemPrompt, int32& OutErrorCode);

    /** Add one native-owned history entry. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool AddAgentMessage(int64 AgentHandle, EAstralAgentRole Role, const FString& Text, int32& OutErrorCode);

    /** Clear native-owned chat history. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool ClearAgentHistory(int64 AgentHandle, int32& OutErrorCode);

    /** Start a native agent chat request. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool EnqueueAgentChat(int64 AgentHandle, const FString& UserMessage, bool bWarmupOnly, int32& OutErrorCode);

    /** Cancel a native agent chat request. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool CancelAgentChat(int64 AgentHandle, int32& OutErrorCode);

    /** Poll native agent chat text. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool ReadAgentChat(int64 AgentHandle, int32 TimeoutMs, FString& OutText, bool& bEndOfStream, int32& OutErrorCode);

    /** Read native agent chat status and counters. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool GetAgentChatResult(int64 AgentHandle, FAstralAgentChatResult& OutResult, int32& OutErrorCode);

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
