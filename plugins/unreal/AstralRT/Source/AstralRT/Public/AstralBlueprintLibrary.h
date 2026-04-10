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

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static FString RequestKindName(EAstralRequestKind Kind);

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static FString RequestStateName(EAstralRequestState State);

    /** Maximum number of adapters that can be attached to one session. */
    UFUNCTION(BlueprintPure, Category = "Astral|Adapters")
    static int32 MaxSessionAdapters();

    /** Read native diagnostics for a loaded model-scoped adapter handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    static bool GetAdapterInfo(int64 AdapterHandle, FAstralAdapterInfo& OutInfo, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    static FAstralOperationResult GetAdapterInfoResult(int64 AdapterHandle, FAstralAdapterInfo& OutInfo);

    /** Copy a loaded adapter path into an engine string. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    static bool CopyAdapterPath(int64 AdapterHandle, FString& OutPath, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Adapters")
    static FAstralOperationResult CopyAdapterPathResult(int64 AdapterHandle, FString& OutPath);

    /** Create a native structured-output toolset. Release it with DestroyToolset. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    static bool CreateToolset(const TArray<FAstralToolDesc>& Tools, EAstralToolChoiceMode ChoiceMode, int64& OutToolsetHandle);

    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    static FAstralOperationResult CreateToolsetResult(const TArray<FAstralToolDesc>& Tools, EAstralToolChoiceMode ChoiceMode);

    /** Release a native structured-output toolset handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    static void DestroyToolset(int64 ToolsetHandle);

    /** Parse a completed tool-call JSON payload against a native toolset. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    static bool ParseToolCall(int64 ToolsetHandle, const FString& GeneratedText, FAstralToolCallResult& OutResult);

    UFUNCTION(BlueprintCallable, Category = "Astral|Tools")
    static FAstralOperationResult ParseToolCallResult(int64 ToolsetHandle, const FString& GeneratedText, FAstralToolCallResult& OutResult);

    /** Split UTF-8 text into native byte ranges. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Chunking")
    static bool ChunkText(const FString& Text, const FAstralChunkerDesc& Desc, TArray<FAstralChunkRange>& OutRanges, int32& OutErrorCode);

    /** Copy one emitted text range into an engine string. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Chunking")
    static bool CopyChunkText(const FString& Text, const FAstralChunkRange& Range, FString& OutText, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Chunking")
    static FAstralOperationResult CopyChunkTextResult(const FString& Text, const FAstralChunkRange& Range, FString& OutText);

    UFUNCTION(BlueprintCallable, Category = "Astral|Chunking")
    static FAstralOperationResult MakeMemoryRecordFromChunkResult(
        const FAstralChunkRange& Range,
        int64 Key,
        int32 Flags,
        FAstralMemoryRecord& OutRecord
    );

    /** Split an already-tokenized sequence into token ranges. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Chunking")
    static bool ChunkTokens(int32 TokenCount, const FAstralChunkerDesc& Desc, TArray<FAstralChunkRange>& OutRanges, int32& OutErrorCode);

    /** Create a native flat vector memory index. Release it with DestroyMemoryIndex. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool CreateMemoryIndex(const FAstralMemoryIndexDesc& Desc, int64& OutMemoryHandle, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult CreateMemoryIndexResult(const FAstralMemoryIndexDesc& Desc);

    /** Release a native memory index handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static void DestroyMemoryIndex(int64 MemoryHandle);

    /** Restore a native memory index snapshot. Release it with DestroyMemoryIndex. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool LoadMemoryIndex(
        const FAstralMemoryIndexDesc& Desc,
        const TArray<uint8>& Bytes,
        int64& OutMemoryHandle,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult LoadMemoryIndexResult(const FAstralMemoryIndexDesc& Desc, const TArray<uint8>& Bytes);

    /** Remove every record from a native memory index. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool ClearMemoryIndex(int64 MemoryHandle, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult ClearMemoryIndexResult(int64 MemoryHandle);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult GetMemoryRecordCountResult(int64 MemoryHandle, int32& OutCount);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool GetMemoryStats(int64 MemoryHandle, FAstralMemoryStats& OutStats, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult GetMemoryStatsResult(int64 MemoryHandle, FAstralMemoryStats& OutStats);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool GetMemoryRecord(int64 MemoryHandle, int64 Key, FAstralMemoryRecord& OutRecord, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult GetMemoryRecordResult(int64 MemoryHandle, int64 Key, FAstralMemoryRecord& OutRecord);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool UpdateMemoryRecord(int64 MemoryHandle, int64 Key, const FAstralMemoryRecord& Record, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult UpdateMemoryRecordResult(int64 MemoryHandle, int64 Key, const FAstralMemoryRecord& Record);

    /** Remove one record from a native memory index by key. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool RemoveMemoryRecord(int64 MemoryHandle, int64 Key, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult RemoveMemoryRecordResult(int64 MemoryHandle, int64 Key);

    /** Serialize a native memory index into engine-owned bytes. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool SaveMemoryIndex(int64 MemoryHandle, TArray<uint8>& OutBytes, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult SaveMemoryIndexResult(int64 MemoryHandle, TArray<uint8>& OutBytes);

    /** Add records and row-major vectors to a native memory index. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool AddMemoryBatch(
        int64 MemoryHandle,
        const TArray<FAstralMemoryRecord>& Records,
        const TArray<float>& Vectors,
        int32 Dimension,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult AddMemoryBatchResult(
        int64 MemoryHandle,
        const TArray<FAstralMemoryRecord>& Records,
        const TArray<float>& Vectors,
        int32 Dimension
    );

    /** Search a native memory index with one query vector. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool SearchMemoryIndex(
        int64 MemoryHandle,
        const TArray<float>& Query,
        int32 TopK,
        int32 GroupId,
        TArray<FAstralMemorySearchResult>& OutResults,
        int32& OutErrorCode,
        int32 GraphSearch = 0
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult SearchMemoryIndexResult(
        int64 MemoryHandle,
        const TArray<float>& Query,
        int32 TopK,
        int32 GroupId,
        TArray<FAstralMemorySearchResult>& OutResults,
        int32 GraphSearch = 0
    );

    /** Start an incremental memory search. Release it with EndMemorySearch. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool BeginMemorySearch(
        int64 MemoryHandle,
        const TArray<float>& Query,
        int32 TopK,
        int32 GroupId,
        int64& OutCursorHandle,
        int32& OutErrorCode,
        int32 GraphSearch = 0
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult BeginMemorySearchResult(
        int64 MemoryHandle,
        const TArray<float>& Query,
        int32 TopK,
        int32 GroupId,
        int32 GraphSearch = 0
    );

    /** Fetch the next batch from an incremental memory search. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static bool FetchMemorySearch(
        int64 CursorHandle,
        int32 MaxResults,
        TArray<FAstralMemorySearchResult>& OutResults,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static FAstralOperationResult FetchMemorySearchResult(
        int64 CursorHandle,
        int32 MaxResults,
        TArray<FAstralMemorySearchResult>& OutResults
    );

    /** Release an incremental memory search handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Memory")
    static void EndMemorySearch(int64 CursorHandle);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static bool CreateSessionRequest(UAstralSession* Session, FAstralRequestRef& OutRequest, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static FAstralOperationResult CreateSessionRequestResult(UAstralSession* Session, FAstralRequestRef& OutRequest);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static bool CreateConversationRequest(int64 ConversationHandle, FAstralRequestRef& OutRequest, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static FAstralOperationResult CreateConversationRequestResult(int64 ConversationHandle, FAstralRequestRef& OutRequest);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static bool CreateAgentChatRequest(int64 AgentHandle, FAstralRequestRef& OutRequest, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static FAstralOperationResult CreateAgentChatRequestResult(int64 AgentHandle, FAstralRequestRef& OutRequest);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static bool CreateEmbeddingRequest(
        UAstralEmbedder* Embedder,
        int64 Ticket,
        FAstralRequestRef& OutRequest,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static FAstralOperationResult CreateEmbeddingRequestResult(
        UAstralEmbedder* Embedder,
        int64 Ticket,
        FAstralRequestRef& OutRequest
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static bool CreateMemorySearchRequest(int64 CursorHandle, FAstralRequestRef& OutRequest, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static FAstralOperationResult CreateMemorySearchRequestResult(int64 CursorHandle, FAstralRequestRef& OutRequest);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static bool GetRequestStatus(const FAstralRequestRef& Request, FAstralRequestStatus& OutStatus, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static FAstralOperationResult GetRequestStatusResult(const FAstralRequestRef& Request, FAstralRequestStatus& OutStatus);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static bool WaitRequest(
        const FAstralRequestRef& Request,
        int32 TimeoutMs,
        FAstralRequestStatus& OutStatus,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static FAstralOperationResult WaitRequestResult(
        const FAstralRequestRef& Request,
        int32 TimeoutMs,
        FAstralRequestStatus& OutStatus
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static bool CancelRequest(const FAstralRequestRef& Request, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Async")
    static FAstralOperationResult CancelRequestResult(const FAstralRequestRef& Request);

    UFUNCTION(BlueprintPure, Category = "Astral|Async")
    static bool IsRequestQueued(const FAstralRequestStatus& Status);

    UFUNCTION(BlueprintPure, Category = "Astral|Async")
    static bool IsRequestRunning(const FAstralRequestStatus& Status);

    UFUNCTION(BlueprintPure, Category = "Astral|Async")
    static bool IsRequestCompleted(const FAstralRequestStatus& Status);

    UFUNCTION(BlueprintPure, Category = "Astral|Async")
    static bool IsRequestCanceled(const FAstralRequestStatus& Status);

    UFUNCTION(BlueprintPure, Category = "Astral|Async")
    static bool IsRequestFailed(const FAstralRequestStatus& Status);

    UFUNCTION(BlueprintPure, Category = "Astral|Async")
    static bool IsRequestActive(const FAstralRequestStatus& Status);

    UFUNCTION(BlueprintPure, Category = "Astral|Async")
    static bool IsRequestTerminal(const FAstralRequestStatus& Status);

    UFUNCTION(BlueprintPure, Category = "Astral|Async")
    static bool IsRequestSuccessful(const FAstralRequestStatus& Status);

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static bool IsOperationSuccessful(const FAstralOperationResult& Result);

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static bool IsOperationBackpressure(const FAstralOperationResult& Result);

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static bool IsOperationTimeout(const FAstralOperationResult& Result);

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static bool IsOperationCanceled(const FAstralOperationResult& Result);

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static bool IsOperationUnsupported(const FAstralOperationResult& Result);

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static bool IsOperationNotFound(const FAstralOperationResult& Result);

    UFUNCTION(BlueprintPure, Category = "Astral|Diagnostics")
    static bool IsOperationEndOfStream(const FAstralOperationResult& Result);

    /** Create a native prompt cache. Release it with DestroyPromptCache. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static bool CreatePromptCache(const FAstralPromptCacheDesc& Desc, int64& OutCacheHandle, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static FAstralOperationResult CreatePromptCacheResult(const FAstralPromptCacheDesc& Desc);

    /** Restore a native prompt cache snapshot. Release it with DestroyPromptCache. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static bool LoadPromptCache(
        const FAstralPromptCacheDesc& Desc,
        const TArray<uint8>& Bytes,
        int64& OutCacheHandle,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static FAstralOperationResult LoadPromptCacheResult(const FAstralPromptCacheDesc& Desc, const TArray<uint8>& Bytes);

    /** Release a native prompt cache handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static void DestroyPromptCache(int64 CacheHandle);

    /** Remove all prompt cache entries. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static bool ClearPromptCache(int64 CacheHandle, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static FAstralOperationResult ClearPromptCacheResult(int64 CacheHandle);

    /** Read prompt cache counters and capacity. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static bool GetPromptCacheStats(int64 CacheHandle, FAstralPromptCacheStats& OutStats, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static FAstralOperationResult GetPromptCacheStatsResult(int64 CacheHandle, FAstralPromptCacheStats& OutStats);

    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static FAstralOperationResult MakePromptCacheKeyResult(
        int64 ModelHandle,
        EAstralPromptSectionKind Section,
        int32 Generation,
        const FString& Text,
        FAstralPromptCacheKey& OutKey
    );

    /** Store tokenized prompt section bytes under a native key. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static bool PutPromptCacheTokens(
        int64 CacheHandle,
        const FAstralPromptCacheKey& Key,
        const TArray<int32>& Tokens,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static FAstralOperationResult PutPromptCacheTokensResult(
        int64 CacheHandle,
        const FAstralPromptCacheKey& Key,
        const TArray<int32>& Tokens
    );

    /** Copy cached tokens into an engine-owned array. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static bool GetPromptCacheTokens(
        int64 CacheHandle,
        const FAstralPromptCacheKey& Key,
        int32 MaxTokens,
        TArray<int32>& OutTokens,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static FAstralOperationResult GetPromptCacheTokensResult(
        int64 CacheHandle,
        const FAstralPromptCacheKey& Key,
        int32 MaxTokens,
        TArray<int32>& OutTokens
    );

    /** Serialize prompt cache entries into engine-owned bytes. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static bool SavePromptCache(int64 CacheHandle, TArray<uint8>& OutBytes, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Prompt Cache")
    static FAstralOperationResult SavePromptCacheResult(int64 CacheHandle, TArray<uint8>& OutBytes);

    /** Create a native agent over an Astral model/executor. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool CreateAgent(const FAstralAgentDesc& Desc, int64& OutAgentHandle, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult CreateAgentResult(const FAstralAgentDesc& Desc);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool GetAgentAssignedSlot(int64 AgentHandle, int32& OutSlot, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult GetAgentAssignedSlotResult(int64 AgentHandle, int32& OutSlot);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool ReleaseAgentSlot(int64 AgentHandle, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult ReleaseAgentSlotResult(int64 AgentHandle);

    /** Release a native agent handle. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static void DestroyAgent(int64 AgentHandle);

    /** Store a native-owned system prompt for the agent. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool SetAgentSystemPrompt(int64 AgentHandle, const FString& SystemPrompt, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult SetAgentSystemPromptResult(int64 AgentHandle, const FString& SystemPrompt);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult GetAgentSystemPromptResult(int64 AgentHandle, FString& OutSystemPrompt);

    /** Store a native-owned rolling summary for the agent. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool SetAgentSummary(int64 AgentHandle, const FString& Summary, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult SetAgentSummaryResult(int64 AgentHandle, const FString& Summary);

    /** Copy the native-owned rolling summary into an engine string. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool GetAgentSummary(int64 AgentHandle, FString& OutSummary, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult GetAgentSummaryResult(int64 AgentHandle, FString& OutSummary);

    /** Store native-owned retrieved context for the next agent prompt. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool SetAgentMemoryContext(int64 AgentHandle, const FString& MemoryContext, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult SetAgentMemoryContextResult(int64 AgentHandle, const FString& MemoryContext);

    /** Build native-owned retrieved context from chunk ranges and memory search results. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool SetAgentMemoryContextFromResults(
        int64 AgentHandle,
        const FString& DocumentText,
        const TArray<FAstralChunkRange>& Chunks,
        const TArray<FAstralMemorySearchResult>& Results,
        const FString& Separator,
        int32 MaxBytes,
        int32& OutErrorCode
    );

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult SetAgentMemoryContextFromResultsStatus(
        int64 AgentHandle,
        const FString& DocumentText,
        const TArray<FAstralChunkRange>& Chunks,
        const TArray<FAstralMemorySearchResult>& Results,
        const FString& Separator,
        int32 MaxBytes
    );

    /** Copy the native-owned retrieved context into an engine string. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool GetAgentMemoryContext(int64 AgentHandle, FString& OutMemoryContext, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult GetAgentMemoryContextResult(int64 AgentHandle, FString& OutMemoryContext);

    /** Parse a completed tool-call JSON payload against an agent-bound toolset. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool ParseAgentToolCall(int64 AgentHandle, const FString& GeneratedText, FAstralToolCallResult& OutResult);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult ParseAgentToolCallResult(
        int64 AgentHandle,
        const FString& GeneratedText,
        FAstralToolCallResult& OutResult
    );

    /** Read the parsed tool call captured from the latest drained agent chat stream. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool GetAgentChatToolCallResult(int64 AgentHandle, FAstralToolCallResult& OutResult);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult GetAgentChatToolCallResultStatus(int64 AgentHandle, FAstralToolCallResult& OutResult);

    /** Add one native-owned history entry. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool AddAgentMessage(int64 AgentHandle, EAstralAgentRole Role, const FString& Text, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult AddAgentMessageResult(int64 AgentHandle, EAstralAgentRole Role, const FString& Text);

    /** Clear native-owned chat history. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool ClearAgentHistory(int64 AgentHandle, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult ClearAgentHistoryResult(int64 AgentHandle);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult GetAgentHistoryCountResult(int64 AgentHandle, int32& OutCount);

    /** Serialize native-owned agent prompt state and history. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool SaveAgentHistory(int64 AgentHandle, TArray<uint8>& OutBytes, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult SaveAgentHistoryResult(int64 AgentHandle, TArray<uint8>& OutBytes);

    /** Restore native-owned agent prompt state and history. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool LoadAgentHistory(int64 AgentHandle, const TArray<uint8>& Bytes, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult LoadAgentHistoryResult(int64 AgentHandle, const TArray<uint8>& Bytes);

    /** Start a native agent chat request. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool EnqueueAgentChat(int64 AgentHandle, const FString& UserMessage, bool bWarmupOnly, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult EnqueueAgentChatResult(int64 AgentHandle, const FString& UserMessage, bool bWarmupOnly);

    /** Cancel a native agent chat request. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool CancelAgentChat(int64 AgentHandle, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult CancelAgentChatResult(int64 AgentHandle);

    /** Poll native agent chat text. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool ReadAgentChat(int64 AgentHandle, int32 TimeoutMs, FString& OutText, bool& bEndOfStream, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult ReadAgentChatResult(int64 AgentHandle, int32 TimeoutMs, FString& OutText);

    /** Read native agent chat status and counters. */
    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static bool GetAgentChatResult(int64 AgentHandle, FAstralAgentChatResult& OutResult, int32& OutErrorCode);

    UFUNCTION(BlueprintCallable, Category = "Astral|Agent")
    static FAstralOperationResult GetAgentChatStatusResult(int64 AgentHandle, FAstralAgentChatResult& OutResult);

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
