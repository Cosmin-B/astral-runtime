#include "AstralBlueprintLibrary.h"

#include "AstralEmbedder.h"
#include "AstralLog.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "UObject/Package.h"

#include "astral_rt.h"

namespace {

static constexpr int32 kInlineToolCapacity = 8;
static constexpr int32 kInlineChunkRangeCapacity = 32;
static constexpr int32 kInlineMemoryRecordCapacity = 32;
static constexpr int32 kInlineMemoryResultCapacity = 16;
static constexpr int32 kAgentReadBufferBytes = 4096;
static constexpr int32 kInlineAgentTextBytes = kAgentReadBufferBytes;
static constexpr int32 kNoElements = 0;
static constexpr int32 kEmptyResultCount = 0;
static constexpr int64 kInvalidAstralHandle = 0;
static constexpr AstralHandle kNullNativeHandle = 0;
static constexpr uint64_t kNoByteCount = 0;

static UObject* resolve_outer(UObject* Outer)
{
    return Outer != nullptr ? Outer : GetTransientPackage();
}

static bool has_cap(int64 Caps, AstralCaps Cap)
{
    const AstralCaps NativeCaps = static_cast<AstralCaps>(Caps);
    return (NativeCaps & Cap) != 0;
}

static AstralToolChoiceMode to_native_tool_choice(EAstralToolChoiceMode Mode)
{
    switch (Mode)
    {
    case EAstralToolChoiceMode::Required:
        return ASTRAL_TOOL_CHOICE_REQUIRED;
    case EAstralToolChoiceMode::TextOrTool:
        return ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL;
    case EAstralToolChoiceMode::Auto:
    default:
        return ASTRAL_TOOL_CHOICE_AUTO;
    }
}

static AstralChunkMode to_native_chunk_mode(EAstralChunkMode Mode)
{
    switch (Mode)
    {
    case EAstralChunkMode::Char:
        return ASTRAL_CHUNK_MODE_CHAR;
    case EAstralChunkMode::Word:
        return ASTRAL_CHUNK_MODE_WORD;
    case EAstralChunkMode::Sentence:
        return ASTRAL_CHUNK_MODE_SENTENCE;
    case EAstralChunkMode::Token:
        return ASTRAL_CHUNK_MODE_TOKEN;
    case EAstralChunkMode::None:
    default:
        return ASTRAL_CHUNK_MODE_NONE;
    }
}

static AstralMemoryMetric to_native_memory_metric(EAstralMemoryMetric Metric)
{
    switch (Metric)
    {
    case EAstralMemoryMetric::Dot:
        return ASTRAL_MEMORY_METRIC_DOT;
    case EAstralMemoryMetric::L2:
        return ASTRAL_MEMORY_METRIC_L2;
    case EAstralMemoryMetric::Cosine:
    default:
        return ASTRAL_MEMORY_METRIC_COSINE;
    }
}

static AstralMemoryIndexKind to_native_memory_index_kind(EAstralMemoryIndexKind Kind)
{
    switch (Kind)
    {
    case EAstralMemoryIndexKind::Graph:
        return ASTRAL_MEMORY_INDEX_GRAPH;
    case EAstralMemoryIndexKind::Flat:
    default:
        return ASTRAL_MEMORY_INDEX_FLAT;
    }
}

static AstralMemoryIndexDesc to_native_memory_desc(const FAstralMemoryIndexDesc& Desc)
{
    AstralMemoryIndexDesc Native{};
    Native.size = sizeof(AstralMemoryIndexDesc);
    Native.dim = static_cast<uint32_t>(Desc.Dimension);
    Native.capacity = static_cast<uint32_t>(Desc.Capacity);
    Native.metric = to_native_memory_metric(Desc.Metric);
    Native.index_kind = to_native_memory_index_kind(Desc.IndexKind);
    Native.graph_neighbors = Desc.GraphNeighbors > 0 ? static_cast<uint32_t>(Desc.GraphNeighbors) : 0u;
    Native.graph_search = Desc.GraphSearch > 0 ? static_cast<uint32_t>(Desc.GraphSearch) : 0u;
    return Native;
}

static AstralAgentRole to_native_agent_role(EAstralAgentRole Role)
{
    switch (Role)
    {
    case EAstralAgentRole::System:
        return ASTRAL_AGENT_ROLE_SYSTEM;
    case EAstralAgentRole::Assistant:
        return ASTRAL_AGENT_ROLE_ASSISTANT;
    case EAstralAgentRole::Tool:
        return ASTRAL_AGENT_ROLE_TOOL;
    case EAstralAgentRole::User:
    default:
        return ASTRAL_AGENT_ROLE_USER;
    }
}

static AstralPromptSectionKind to_native_prompt_section(EAstralPromptSectionKind Section)
{
    switch (Section)
    {
    case EAstralPromptSectionKind::Tools:
        return ASTRAL_PROMPT_SECTION_TOOLS;
    case EAstralPromptSectionKind::Memory:
        return ASTRAL_PROMPT_SECTION_MEMORY;
    case EAstralPromptSectionKind::History:
        return ASTRAL_PROMPT_SECTION_HISTORY;
    case EAstralPromptSectionKind::User:
        return ASTRAL_PROMPT_SECTION_USER;
    case EAstralPromptSectionKind::Raw:
        return ASTRAL_PROMPT_SECTION_RAW;
    case EAstralPromptSectionKind::System:
    default:
        return ASTRAL_PROMPT_SECTION_SYSTEM;
    }
}

static AstralPromptCacheEvictionPolicy to_native_prompt_cache_eviction(EAstralPromptCacheEvictionPolicy Policy)
{
    switch (Policy)
    {
    case EAstralPromptCacheEvictionPolicy::Fifo:
    default:
        return ASTRAL_PROMPT_CACHE_EVICT_FIFO;
    }
}

static AstralPromptCacheDesc to_native_prompt_cache_desc(const FAstralPromptCacheDesc& Desc)
{
    AstralPromptCacheDesc Native{};
    Native.size = sizeof(AstralPromptCacheDesc);
    Native.max_entries = static_cast<uint32_t>(Desc.MaxEntries);
    Native.max_tokens = static_cast<uint32_t>(Desc.MaxTokens);
    Native.max_bytes = static_cast<uint32_t>(Desc.MaxBytes);
    Native.eviction_policy = to_native_prompt_cache_eviction(Desc.EvictionPolicy);
    Native.flags = Desc.bTrackStats ? ASTRAL_PROMPT_CACHE_FLAG_TRACK_STATS : 0u;
    return Native;
}

static AstralPromptCacheKey to_native_prompt_cache_key(const FAstralPromptCacheKey& Key)
{
    AstralPromptCacheKey Native{};
    Native.size = sizeof(AstralPromptCacheKey);
    Native.section_kind = to_native_prompt_section(Key.Section);
    Native.model = static_cast<AstralHandle>(Key.ModelHandle);
    Native.key = static_cast<uint64_t>(Key.Key);
    Native.generation = static_cast<uint32_t>(Key.Generation);
    return Native;
}

static AstralRequestKind to_native_request_kind(EAstralRequestKind Kind)
{
    switch (Kind)
    {
    case EAstralRequestKind::Session:
        return ASTRAL_REQUEST_SESSION;
    case EAstralRequestKind::Conversation:
        return ASTRAL_REQUEST_CONVERSATION;
    case EAstralRequestKind::AgentChat:
        return ASTRAL_REQUEST_AGENT_CHAT;
    case EAstralRequestKind::Embedding:
        return ASTRAL_REQUEST_EMBEDDING;
    case EAstralRequestKind::MemorySearch:
        return ASTRAL_REQUEST_MEMORY_SEARCH;
    case EAstralRequestKind::None:
    default:
        return ASTRAL_REQUEST_NONE;
    }
}

static EAstralRequestKind from_native_request_kind(AstralRequestKind Kind)
{
    switch (Kind)
    {
    case ASTRAL_REQUEST_SESSION:
        return EAstralRequestKind::Session;
    case ASTRAL_REQUEST_CONVERSATION:
        return EAstralRequestKind::Conversation;
    case ASTRAL_REQUEST_AGENT_CHAT:
        return EAstralRequestKind::AgentChat;
    case ASTRAL_REQUEST_EMBEDDING:
        return EAstralRequestKind::Embedding;
    case ASTRAL_REQUEST_MEMORY_SEARCH:
        return EAstralRequestKind::MemorySearch;
    case ASTRAL_REQUEST_NONE:
    default:
        return EAstralRequestKind::None;
    }
}

static EAstralRequestState from_native_request_state(AstralRequestState State)
{
    switch (State)
    {
    case ASTRAL_REQUEST_QUEUED:
        return EAstralRequestState::Queued;
    case ASTRAL_REQUEST_RUNNING:
        return EAstralRequestState::Running;
    case ASTRAL_REQUEST_COMPLETED:
        return EAstralRequestState::Completed;
    case ASTRAL_REQUEST_CANCELED:
        return EAstralRequestState::Canceled;
    case ASTRAL_REQUEST_FAILED:
        return EAstralRequestState::Failed;
    case ASTRAL_REQUEST_INVALID:
    default:
        return EAstralRequestState::Invalid;
    }
}

static AstralRequestRef to_native_request_ref(const FAstralRequestRef& Request)
{
    AstralRequestRef Native{};
    Native.size = sizeof(AstralRequestRef);
    Native.kind = to_native_request_kind(Request.Kind);
    Native.owner = static_cast<AstralHandle>(Request.OwnerHandle);
    Native.ticket = static_cast<uint64_t>(Request.Ticket);
    return Native;
}

static FAstralRequestRef from_native_request_ref(const AstralRequestRef& Native)
{
    FAstralRequestRef Request;
    Request.Kind = from_native_request_kind(Native.kind);
    Request.OwnerHandle = static_cast<int64>(Native.owner);
    Request.Ticket = static_cast<int64>(Native.ticket);
    return Request;
}

static FAstralRequestStatus from_native_request_status(const AstralRequestStatus& Native)
{
    FAstralRequestStatus Status;
    Status.Kind = from_native_request_kind(Native.kind);
    Status.State = from_native_request_state(Native.state);
    Status.ErrorCode = static_cast<int32>(Native.result);
    Status.OwnerHandle = static_cast<int64>(Native.owner);
    Status.Ticket = static_cast<int64>(Native.ticket);
    Status.QueueDepth = static_cast<int32>(Native.queue_depth);
    Status.bHasTicket = (Native.flags & ASTRAL_REQUEST_FLAG_TICKET) != 0u;
    Status.bStream = (Native.flags & ASTRAL_REQUEST_FLAG_STREAM) != 0u;
    return Status;
}

static FAstralPromptCacheStats from_native_prompt_cache_stats(const AstralPromptCacheStats& Native)
{
    FAstralPromptCacheStats Stats;
    Stats.Entries = static_cast<int32>(Native.entries);
    Stats.MaxEntries = static_cast<int32>(Native.max_entries);
    Stats.Tokens = static_cast<int32>(Native.tokens);
    Stats.MaxTokens = static_cast<int32>(Native.max_tokens);
    Stats.Bytes = static_cast<int32>(Native.bytes);
    Stats.MaxBytes = static_cast<int32>(Native.max_bytes);
    Stats.Hits = static_cast<int64>(Native.hits);
    Stats.Misses = static_cast<int64>(Native.misses);
    Stats.Evictions = static_cast<int64>(Native.evictions);
    return Stats;
}

static void fill_native_chunker(const FAstralChunkerDesc& Source, const FTCHARToUTF8& DelimitersUtf8, AstralChunkerDesc& Out)
{
    Out = AstralChunkerDesc{};
    Out.size = sizeof(AstralChunkerDesc);
    Out.mode = to_native_chunk_mode(Source.Mode);
    Out.max_units = static_cast<uint32_t>(Source.MaxUnits);
    Out.overlap_units = static_cast<uint32_t>(Source.OverlapUnits);
    Out.document_id = static_cast<uint32_t>(Source.DocumentId);
    Out.group_id = static_cast<uint32_t>(Source.GroupId);
    Out.delimiters.data = reinterpret_cast<const uint8_t*>(DelimitersUtf8.Get());
    Out.delimiters.len = static_cast<uint32_t>(DelimitersUtf8.Length());
}

static FAstralChunkRange from_native_chunk_range(const AstralChunkRange& Native)
{
    FAstralChunkRange Range;
    Range.DocumentId = static_cast<int32>(Native.document_id);
    Range.ChunkId = static_cast<int32>(Native.chunk_id);
    Range.GroupId = static_cast<int32>(Native.group_id);
    Range.ByteBegin = static_cast<int32>(Native.byte_begin);
    Range.ByteEnd = static_cast<int32>(Native.byte_end);
    Range.TokenBegin = static_cast<int32>(Native.token_begin);
    Range.TokenEnd = static_cast<int32>(Native.token_end);
    return Range;
}

static FAstralMemorySearchResult from_native_memory_result(const AstralMemorySearchResult& Native)
{
    FAstralMemorySearchResult Result;
    Result.Key = static_cast<int64>(Native.key);
    Result.GroupId = static_cast<int32>(Native.group_id);
    Result.DocumentId = static_cast<int32>(Native.document_id);
    Result.ChunkId = static_cast<int32>(Native.chunk_id);
    Result.Score = Native.score;
    Result.Flags = static_cast<int32>(Native.flags);
    return Result;
}

static FAstralAgentChatResult from_native_agent_result(const AstralAgentChatResult& Native)
{
    FAstralAgentChatResult Result;
    Result.State = static_cast<int32>(Native.state);
    Result.PromptBytes = static_cast<int32>(Native.prompt_bytes);
    Result.HistoryMessages = static_cast<int32>(Native.history_messages);
    Result.PromptTokens = static_cast<int32>(Native.prompt_tokens);
    Result.PromptCacheReusedTokens = static_cast<int32>(Native.prompt_cache_reused_tokens);
    Result.PromptCacheNewTokens = static_cast<int32>(Native.prompt_cache_new_tokens);
    Result.PromptCacheHits = static_cast<int32>(Native.prompt_cache_hits);
    Result.PromptCacheMisses = static_cast<int32>(Native.prompt_cache_misses);
    Result.LastError = static_cast<int32>(Native.last_error);
    Result.GeneratedTokens = static_cast<int64>(Native.generated_tokens);
    Result.TimeToFirstTokenMs = Native.t_first_token_ms;
    Result.TokensPerSecond = Native.tok_per_s;
    return Result;
}

static FAstralOperationResult make_operation_result(
    AstralErr Err,
    int64 Handle = kInvalidAstralHandle,
    int32 Count = kEmptyResultCount
)
{
    FAstralOperationResult Result;
    Result.bSuccess = Err == ASTRAL_OK;
    Result.ErrorCode = static_cast<int32>(Err);
    Result.Handle = Handle;
    Result.Count = Count;
    Result.bBackpressure = Err == ASTRAL_E_BUSY;
    Result.bTimeout = Err == ASTRAL_E_TIMEOUT;
    Result.bCanceled = Err == ASTRAL_E_CANCELED;
    Result.bUnsupported = Err == ASTRAL_E_UNSUPPORTED;
    Result.bNotFound = Err == ASTRAL_E_NOT_FOUND;
    return Result;
}

static bool chunker_desc_valid_for_blueprint(const FAstralChunkerDesc& Desc)
{
    return Desc.MaxUnits > 0 && Desc.OverlapUnits >= 0 && Desc.OverlapUnits < Desc.MaxUnits &&
           Desc.DocumentId >= 0 && Desc.GroupId >= 0;
}

static FString utf8_span_to_string(AstralSpanU8 Span)
{
    if (Span.data == nullptr || Span.len == 0)
    {
        return FString();
    }
    FUTF8ToTCHAR Converted(reinterpret_cast<const ANSICHAR*>(Span.data), static_cast<int32>(Span.len));
    return FString(Converted.Length(), Converted.Get());
}

} // namespace

UAstralModel* UAstralBlueprintLibrary::CreateAstralModel(UObject* Outer)
{
    return NewObject<UAstralModel>(resolve_outer(Outer));
}

UAstralSession* UAstralBlueprintLibrary::CreateAstralSession(UObject* Outer)
{
    return NewObject<UAstralSession>(resolve_outer(Outer));
}

UAstralEmbedder* UAstralBlueprintLibrary::CreateAstralEmbedder(UObject* Outer)
{
    return NewObject<UAstralEmbedder>(resolve_outer(Outer));
}

FString UAstralBlueprintLibrary::GetLastAstralError()
{
    const char* Last = astral_last_error();
    return Last != nullptr ? FString(UTF8_TO_TCHAR(Last)) : FString();
}

FString UAstralBlueprintLibrary::ErrorCodeName(int32 ErrorCode)
{
    switch (static_cast<AstralErr>(ErrorCode))
    {
    case ASTRAL_OK:
        return TEXT("ASTRAL_OK");
    case ASTRAL_E_INVALID:
        return TEXT("ASTRAL_E_INVALID");
    case ASTRAL_E_NOMEM:
        return TEXT("ASTRAL_E_NOMEM");
    case ASTRAL_E_BUSY:
        return TEXT("ASTRAL_E_BUSY");
    case ASTRAL_E_TIMEOUT:
        return TEXT("ASTRAL_E_TIMEOUT");
    case ASTRAL_E_STATE:
        return TEXT("ASTRAL_E_STATE");
    case ASTRAL_E_BACKEND:
        return TEXT("ASTRAL_E_BACKEND");
    case ASTRAL_E_CANCELED:
        return TEXT("ASTRAL_E_CANCELED");
    case ASTRAL_E_UNSUPPORTED:
        return TEXT("ASTRAL_E_UNSUPPORTED");
    case ASTRAL_E_NOT_FOUND:
        return TEXT("ASTRAL_E_NOT_FOUND");
    default:
        return FString::Printf(TEXT("ASTRAL_E_%d"), ErrorCode);
    }
}

int32 UAstralBlueprintLibrary::MaxSessionAdapters()
{
    return static_cast<int32>(ASTRAL_SESSION_ADAPTERS_MAX);
}

bool UAstralBlueprintLibrary::CreateToolset(
    const TArray<FAstralToolDesc>& Tools,
    EAstralToolChoiceMode ChoiceMode,
    int64& OutToolsetHandle
)
{
    const FAstralOperationResult Result = CreateToolsetResult(Tools, ChoiceMode);
    OutToolsetHandle = Result.Handle;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreateToolsetResult(
    const TArray<FAstralToolDesc>& Tools,
    EAstralToolChoiceMode ChoiceMode
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateToolset);

    if (Tools.Num() == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    int32 Utf8Bytes = 0;
    for (const FAstralToolDesc& Tool : Tools)
    {
        FTCHARToUTF8 NameUtf8(*Tool.Name);
        FTCHARToUTF8 DescriptionUtf8(*Tool.Description);
        FTCHARToUTF8 SchemaUtf8(*Tool.JsonSchema);
        Utf8Bytes += NameUtf8.Length() + DescriptionUtf8.Length() + SchemaUtf8.Length();
    }

    TArray<uint8> Utf8Storage;
    Utf8Storage.Reserve(Utf8Bytes);
    TArray<AstralToolDesc, TInlineAllocator<kInlineToolCapacity>> NativeTools;
    NativeTools.SetNumZeroed(Tools.Num());

    for (int32 Index = 0; Index < Tools.Num(); ++Index)
    {
        const FAstralToolDesc& Tool = Tools[Index];
        FTCHARToUTF8 NameUtf8(*Tool.Name);
        FTCHARToUTF8 DescriptionUtf8(*Tool.Description);
        FTCHARToUTF8 SchemaUtf8(*Tool.JsonSchema);

        AstralToolDesc& Native = NativeTools[Index];
        Native.size = sizeof(AstralToolDesc);
        Native.tool_id = static_cast<uint32_t>(Tool.ToolId);

        const int32 NameOffset = Utf8Storage.Num();
        Utf8Storage.Append(reinterpret_cast<const uint8*>(NameUtf8.Get()), NameUtf8.Length());
        Native.name.data = Utf8Storage.GetData() + NameOffset;
        Native.name.len = static_cast<uint32_t>(NameUtf8.Length());

        const int32 DescriptionOffset = Utf8Storage.Num();
        Utf8Storage.Append(reinterpret_cast<const uint8*>(DescriptionUtf8.Get()), DescriptionUtf8.Length());
        Native.description.data = Utf8Storage.GetData() + DescriptionOffset;
        Native.description.len = static_cast<uint32_t>(DescriptionUtf8.Length());

        const int32 SchemaOffset = Utf8Storage.Num();
        Utf8Storage.Append(reinterpret_cast<const uint8*>(SchemaUtf8.Get()), SchemaUtf8.Length());
        Native.json_schema.data = Utf8Storage.GetData() + SchemaOffset;
        Native.json_schema.len = static_cast<uint32_t>(SchemaUtf8.Length());
    }

    AstralToolsetDesc Desc{};
    Desc.size = sizeof(AstralToolsetDesc);
    Desc.tool_count = static_cast<uint32_t>(NativeTools.Num());
    Desc.choice_mode = to_native_tool_choice(ChoiceMode);
    Desc.tools = NativeTools.GetData();

    AstralHandle Toolset = 0;
    const AstralErr Err = astral_toolset_create(&Desc, &Toolset);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_toolset_create failed (%d)"), static_cast<int32>(Err));
        return make_operation_result(Err);
    }

    return make_operation_result(ASTRAL_OK, static_cast<int64>(Toolset), static_cast<int32>(Tools.Num()));
}

void UAstralBlueprintLibrary::DestroyToolset(int64 ToolsetHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_DestroyToolset);

    if (ToolsetHandle != 0)
    {
        astral_toolset_destroy(static_cast<AstralHandle>(ToolsetHandle));
    }
}

bool UAstralBlueprintLibrary::ParseToolCall(
    int64 ToolsetHandle,
    const FString& GeneratedText,
    FAstralToolCallResult& OutResult
)
{
    return ParseToolCallResult(ToolsetHandle, GeneratedText, OutResult).bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::ParseToolCallResult(
    int64 ToolsetHandle,
    const FString& GeneratedText,
    FAstralToolCallResult& OutResult
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ParseToolCall);

    OutResult = FAstralToolCallResult{};
    if (ToolsetHandle == 0)
    {
        OutResult.ParseStatus = static_cast<int32>(ASTRAL_E_INVALID);
        return make_operation_result(ASTRAL_E_INVALID);
    }

    FTCHARToUTF8 GeneratedUtf8(*GeneratedText);
    AstralSpanU8 Text{};
    Text.data = reinterpret_cast<const uint8_t*>(GeneratedUtf8.Get());
    Text.len = static_cast<uint32_t>(GeneratedUtf8.Length());

    AstralToolCallResult Native{};
    Native.size = sizeof(AstralToolCallResult);
    const AstralErr Err = astral_toolset_parse_call(static_cast<AstralHandle>(ToolsetHandle), Text, &Native);
    if (Err != ASTRAL_OK)
    {
        OutResult.ParseStatus = static_cast<int32>(Err);
        return make_operation_result(Err);
    }

    OutResult.bFound = true;
    OutResult.ParseStatus = Native.parse_status;
    OutResult.ToolId = static_cast<int32>(Native.tool_id);
    OutResult.Name = utf8_span_to_string(Native.name);
    OutResult.ArgumentsJson = utf8_span_to_string(Native.arguments_json);
    return make_operation_result(ASTRAL_OK, ToolsetHandle, static_cast<int32>(Native.parse_status));
}

bool UAstralBlueprintLibrary::ChunkText(
    const FString& Text,
    const FAstralChunkerDesc& Desc,
    TArray<FAstralChunkRange>& OutRanges,
    int32& OutErrorCode
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ChunkText);

    OutRanges.Reset();
    OutErrorCode = static_cast<int32>(ASTRAL_OK);
    if (!chunker_desc_valid_for_blueprint(Desc) || Desc.Mode == EAstralChunkMode::Token)
    {
        OutErrorCode = static_cast<int32>(ASTRAL_E_INVALID);
        return false;
    }

    FTCHARToUTF8 TextUtf8(*Text);
    FTCHARToUTF8 DelimitersUtf8(*Desc.Delimiters);
    AstralChunkerDesc NativeDesc{};
    fill_native_chunker(Desc, DelimitersUtf8, NativeDesc);

    AstralSpanU8 NativeText{};
    NativeText.data = reinterpret_cast<const uint8_t*>(TextUtf8.Get());
    NativeText.len = static_cast<uint32_t>(TextUtf8.Length());

    uint32_t Required = 0;
    AstralErr Err = astral_chunk_count(&NativeDesc, NativeText, &Required);
    if (Err != ASTRAL_OK)
    {
        OutErrorCode = static_cast<int32>(Err);
        return false;
    }
    if (Required == 0)
    {
        return true;
    }

    TArray<AstralChunkRange, TInlineAllocator<kInlineChunkRangeCapacity>> NativeRanges;
    NativeRanges.SetNumZeroed(static_cast<int32>(Required));
    Err = astral_chunk_ranges(&NativeDesc, NativeText, NativeRanges.GetData(), Required, &Required);
    if (Err != ASTRAL_OK)
    {
        OutErrorCode = static_cast<int32>(Err);
        return false;
    }

    OutRanges.Reserve(static_cast<int32>(Required));
    for (uint32_t Index = 0; Index < Required; ++Index)
    {
        OutRanges.Add(from_native_chunk_range(NativeRanges[static_cast<int32>(Index)]));
    }
    return true;
}

bool UAstralBlueprintLibrary::ChunkTokens(
    int32 TokenCount,
    const FAstralChunkerDesc& Desc,
    TArray<FAstralChunkRange>& OutRanges,
    int32& OutErrorCode
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ChunkTokens);

    OutRanges.Reset();
    OutErrorCode = static_cast<int32>(ASTRAL_OK);
    if (!chunker_desc_valid_for_blueprint(Desc) || Desc.Mode != EAstralChunkMode::Token || TokenCount < 0)
    {
        OutErrorCode = static_cast<int32>(ASTRAL_E_INVALID);
        return false;
    }

    FTCHARToUTF8 DelimitersUtf8(*Desc.Delimiters);
    AstralChunkerDesc NativeDesc{};
    fill_native_chunker(Desc, DelimitersUtf8, NativeDesc);

    uint32_t Required = 0;
    AstralErr Err = astral_token_chunk_count(&NativeDesc, static_cast<uint32_t>(TokenCount), &Required);
    if (Err != ASTRAL_OK)
    {
        OutErrorCode = static_cast<int32>(Err);
        return false;
    }
    if (Required == 0)
    {
        return true;
    }

    TArray<AstralChunkRange, TInlineAllocator<kInlineChunkRangeCapacity>> NativeRanges;
    NativeRanges.SetNumZeroed(static_cast<int32>(Required));
    Err = astral_token_chunk_ranges(&NativeDesc, static_cast<uint32_t>(TokenCount), NativeRanges.GetData(), Required, &Required);
    if (Err != ASTRAL_OK)
    {
        OutErrorCode = static_cast<int32>(Err);
        return false;
    }

    OutRanges.Reserve(static_cast<int32>(Required));
    for (uint32_t Index = 0; Index < Required; ++Index)
    {
        OutRanges.Add(from_native_chunk_range(NativeRanges[static_cast<int32>(Index)]));
    }
    return true;
}

bool UAstralBlueprintLibrary::CreateMemoryIndex(
    const FAstralMemoryIndexDesc& Desc,
    int64& OutMemoryHandle,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = CreateMemoryIndexResult(Desc);
    OutMemoryHandle = Result.Handle;
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreateMemoryIndexResult(const FAstralMemoryIndexDesc& Desc)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateMemoryIndex);

    if (Desc.Dimension <= 0 || Desc.Capacity <= 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralMemoryIndexDesc Native = to_native_memory_desc(Desc);

    AstralHandle Handle = kNullNativeHandle;
    const AstralErr Err = astral_memory_create(&Native, &Handle);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, static_cast<int64>(Handle));
}

void UAstralBlueprintLibrary::DestroyMemoryIndex(int64 MemoryHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_DestroyMemoryIndex);

    if (MemoryHandle != kInvalidAstralHandle)
    {
        astral_memory_destroy(static_cast<AstralHandle>(MemoryHandle));
    }
}

bool UAstralBlueprintLibrary::LoadMemoryIndex(
    const FAstralMemoryIndexDesc& Desc,
    const TArray<uint8>& Bytes,
    int64& OutMemoryHandle,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = LoadMemoryIndexResult(Desc, Bytes);
    OutMemoryHandle = Result.Handle;
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::LoadMemoryIndexResult(const FAstralMemoryIndexDesc& Desc, const TArray<uint8>& Bytes)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_LoadMemoryIndex);

    if (Desc.Dimension <= kNoElements || Desc.Capacity <= kNoElements || Bytes.Num() == kNoElements)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralMemoryIndexDesc Native = to_native_memory_desc(Desc);
    AstralSpanU8 Span{};
    Span.data = Bytes.GetData();
    Span.len = static_cast<uint32_t>(Bytes.Num());

    AstralHandle Handle = kNullNativeHandle;
    const AstralErr Err = astral_memory_load(&Native, Span, &Handle);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, static_cast<int64>(Handle));
}

bool UAstralBlueprintLibrary::ClearMemoryIndex(int64 MemoryHandle, int32& OutErrorCode)
{
    const FAstralOperationResult Result = ClearMemoryIndexResult(MemoryHandle);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::ClearMemoryIndexResult(int64 MemoryHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ClearMemoryIndex);

    if (MemoryHandle == kInvalidAstralHandle)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    const AstralErr Err = astral_memory_clear(static_cast<AstralHandle>(MemoryHandle));
    return make_operation_result(Err, MemoryHandle);
}

bool UAstralBlueprintLibrary::RemoveMemoryRecord(int64 MemoryHandle, int64 Key, int32& OutErrorCode)
{
    const FAstralOperationResult Result = RemoveMemoryRecordResult(MemoryHandle, Key);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::RemoveMemoryRecordResult(int64 MemoryHandle, int64 Key)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_RemoveMemoryRecord);

    if (MemoryHandle == kInvalidAstralHandle)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    const AstralErr Err = astral_memory_remove(static_cast<AstralHandle>(MemoryHandle), static_cast<uint64_t>(Key));
    return make_operation_result(Err, MemoryHandle);
}

bool UAstralBlueprintLibrary::SaveMemoryIndex(int64 MemoryHandle, TArray<uint8>& OutBytes, int32& OutErrorCode)
{
    const FAstralOperationResult Result = SaveMemoryIndexResult(MemoryHandle, OutBytes);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::SaveMemoryIndexResult(int64 MemoryHandle, TArray<uint8>& OutBytes)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_SaveMemoryIndex);

    OutBytes.Reset();
    if (MemoryHandle == kInvalidAstralHandle)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    uint64_t ByteCount = kNoByteCount;
    AstralErr Err = astral_memory_save_size(static_cast<AstralHandle>(MemoryHandle), &ByteCount);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    if (ByteCount > static_cast<uint64_t>(TNumericLimits<int32>::Max()))
    {
        return make_operation_result(ASTRAL_E_NOMEM);
    }

    OutBytes.SetNumUninitialized(static_cast<int32>(ByteCount));
    AstralMutSpanU8 Span{};
    Span.data = OutBytes.GetData();
    Span.len = static_cast<uint32_t>(OutBytes.Num());

    uint64_t Written = kNoByteCount;
    Err = astral_memory_save(static_cast<AstralHandle>(MemoryHandle), Span, &Written);
    if (Err != ASTRAL_OK)
    {
        OutBytes.Reset();
        return make_operation_result(Err);
    }

    OutBytes.SetNum(static_cast<int32>(Written), EAllowShrinking::No);
    return make_operation_result(ASTRAL_OK, MemoryHandle, static_cast<int32>(Written));
}

bool UAstralBlueprintLibrary::AddMemoryBatch(
    int64 MemoryHandle,
    const TArray<FAstralMemoryRecord>& Records,
    const TArray<float>& Vectors,
    int32 Dimension,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = AddMemoryBatchResult(MemoryHandle, Records, Vectors, Dimension);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::AddMemoryBatchResult(
    int64 MemoryHandle,
    const TArray<FAstralMemoryRecord>& Records,
    const TArray<float>& Vectors,
    int32 Dimension
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_AddMemoryBatch);

    if (MemoryHandle == 0 || Dimension <= 0 || Records.Num() == 0 || Vectors.Num() != Records.Num() * Dimension)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    TArray<AstralMemoryRecord, TInlineAllocator<kInlineMemoryRecordCapacity>> NativeRecords;
    NativeRecords.SetNumZeroed(Records.Num());
    for (int32 Index = 0; Index < Records.Num(); ++Index)
    {
        const FAstralMemoryRecord& Source = Records[Index];
        AstralMemoryRecord& Native = NativeRecords[Index];
        Native.size = sizeof(AstralMemoryRecord);
        Native.key = static_cast<uint64_t>(Source.Key);
        Native.group_id = static_cast<uint32_t>(Source.GroupId);
        Native.document_id = static_cast<uint32_t>(Source.DocumentId);
        Native.chunk_id = static_cast<uint32_t>(Source.ChunkId);
        Native.flags = static_cast<uint32_t>(Source.Flags);
    }

    const AstralErr Err = astral_memory_add_batch(
        static_cast<AstralHandle>(MemoryHandle),
        NativeRecords.GetData(),
        Vectors.GetData(),
        static_cast<uint32_t>(NativeRecords.Num())
    );
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, MemoryHandle, Records.Num());
}

bool UAstralBlueprintLibrary::SearchMemoryIndex(
    int64 MemoryHandle,
    const TArray<float>& Query,
    int32 TopK,
    int32 GroupId,
    TArray<FAstralMemorySearchResult>& OutResults,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = SearchMemoryIndexResult(MemoryHandle, Query, TopK, GroupId, OutResults);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::SearchMemoryIndexResult(
    int64 MemoryHandle,
    const TArray<float>& Query,
    int32 TopK,
    int32 GroupId,
    TArray<FAstralMemorySearchResult>& OutResults
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_SearchMemoryIndex);

    OutResults.Reset();
    if (MemoryHandle == 0 || Query.Num() == 0 || TopK <= 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralMemorySearchDesc NativeSearch{};
    NativeSearch.size = sizeof(AstralMemorySearchDesc);
    NativeSearch.top_k = static_cast<uint32_t>(TopK);
    NativeSearch.group_id = GroupId < 0 ? ASTRAL_MEMORY_GROUP_ANY : static_cast<uint32_t>(GroupId);

    TArray<AstralMemorySearchResult, TInlineAllocator<kInlineMemoryResultCapacity>> NativeResults;
    NativeResults.SetNumZeroed(TopK);
    uint32_t ResultCount = 0;
    const AstralErr Err = astral_memory_search(
        static_cast<AstralHandle>(MemoryHandle),
        &NativeSearch,
        Query.GetData(),
        NativeResults.GetData(),
        static_cast<uint32_t>(NativeResults.Num()),
        &ResultCount
    );
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutResults.Reserve(static_cast<int32>(ResultCount));
    for (uint32_t Index = 0; Index < ResultCount; ++Index)
    {
        OutResults.Add(from_native_memory_result(NativeResults[static_cast<int32>(Index)]));
    }
    return make_operation_result(ASTRAL_OK, MemoryHandle, static_cast<int32>(ResultCount));
}

bool UAstralBlueprintLibrary::BeginMemorySearch(
    int64 MemoryHandle,
    const TArray<float>& Query,
    int32 TopK,
    int32 GroupId,
    int64& OutCursorHandle,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = BeginMemorySearchResult(MemoryHandle, Query, TopK, GroupId);
    OutCursorHandle = Result.Handle;
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::BeginMemorySearchResult(
    int64 MemoryHandle,
    const TArray<float>& Query,
    int32 TopK,
    int32 GroupId
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_BeginMemorySearch);

    if (MemoryHandle == 0 || Query.Num() == 0 || TopK <= 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralMemorySearchDesc NativeSearch{};
    NativeSearch.size = sizeof(AstralMemorySearchDesc);
    NativeSearch.top_k = static_cast<uint32_t>(TopK);
    NativeSearch.group_id = GroupId < 0 ? ASTRAL_MEMORY_GROUP_ANY : static_cast<uint32_t>(GroupId);

    AstralHandle Cursor = 0;
    const AstralErr Err = astral_memory_search_begin(
        static_cast<AstralHandle>(MemoryHandle),
        &NativeSearch,
        Query.GetData(),
        &Cursor
    );
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    return make_operation_result(ASTRAL_OK, static_cast<int64>(Cursor));
}

bool UAstralBlueprintLibrary::FetchMemorySearch(
    int64 CursorHandle,
    int32 MaxResults,
    TArray<FAstralMemorySearchResult>& OutResults,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = FetchMemorySearchResult(CursorHandle, MaxResults, OutResults);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::FetchMemorySearchResult(
    int64 CursorHandle,
    int32 MaxResults,
    TArray<FAstralMemorySearchResult>& OutResults
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_FetchMemorySearch);

    OutResults.Reset();
    if (CursorHandle == 0 || MaxResults < 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    TArray<AstralMemorySearchResult, TInlineAllocator<kInlineMemoryResultCapacity>> NativeResults;
    NativeResults.SetNumZeroed(MaxResults);
    uint32_t ResultCount = 0;
    const AstralErr Err = astral_memory_search_fetch(
        static_cast<AstralHandle>(CursorHandle),
        NativeResults.GetData(),
        static_cast<uint32_t>(NativeResults.Num()),
        &ResultCount
    );
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutResults.Reserve(static_cast<int32>(ResultCount));
    for (uint32_t Index = 0; Index < ResultCount; ++Index)
    {
        OutResults.Add(from_native_memory_result(NativeResults[static_cast<int32>(Index)]));
    }
    return make_operation_result(ASTRAL_OK, CursorHandle, static_cast<int32>(ResultCount));
}

void UAstralBlueprintLibrary::EndMemorySearch(int64 CursorHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_EndMemorySearch);

    if (CursorHandle != 0)
    {
        astral_memory_search_end(static_cast<AstralHandle>(CursorHandle));
    }
}

bool UAstralBlueprintLibrary::CreateSessionRequest(
    UAstralSession* Session,
    FAstralRequestRef& OutRequest,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = CreateSessionRequestResult(Session, OutRequest);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreateSessionRequestResult(
    UAstralSession* Session,
    FAstralRequestRef& OutRequest
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateSessionRequest);

    OutRequest = FAstralRequestRef{};
    if (Session == nullptr || !Session->IsValid())
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralRequestRef Native{};
    const AstralErr Err = astral_request_from_session(static_cast<AstralHandle>(Session->GetHandle()), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutRequest = from_native_request_ref(Native);
    return make_operation_result(ASTRAL_OK, OutRequest.OwnerHandle);
}

bool UAstralBlueprintLibrary::CreateConversationRequest(
    int64 ConversationHandle,
    FAstralRequestRef& OutRequest,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = CreateConversationRequestResult(ConversationHandle, OutRequest);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreateConversationRequestResult(
    int64 ConversationHandle,
    FAstralRequestRef& OutRequest
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateConversationRequest);

    OutRequest = FAstralRequestRef{};
    if (ConversationHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralRequestRef Native{};
    const AstralErr Err = astral_request_from_conversation(static_cast<AstralHandle>(ConversationHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutRequest = from_native_request_ref(Native);
    return make_operation_result(ASTRAL_OK, OutRequest.OwnerHandle);
}

bool UAstralBlueprintLibrary::CreateAgentChatRequest(
    int64 AgentHandle,
    FAstralRequestRef& OutRequest,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = CreateAgentChatRequestResult(AgentHandle, OutRequest);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreateAgentChatRequestResult(
    int64 AgentHandle,
    FAstralRequestRef& OutRequest
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateAgentChatRequest);

    OutRequest = FAstralRequestRef{};
    if (AgentHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralRequestRef Native{};
    const AstralErr Err = astral_request_from_agent_chat(static_cast<AstralHandle>(AgentHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutRequest = from_native_request_ref(Native);
    return make_operation_result(ASTRAL_OK, OutRequest.OwnerHandle);
}

bool UAstralBlueprintLibrary::CreateEmbeddingRequest(
    UAstralEmbedder* Embedder,
    int64 Ticket,
    FAstralRequestRef& OutRequest,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = CreateEmbeddingRequestResult(Embedder, Ticket, OutRequest);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreateEmbeddingRequestResult(
    UAstralEmbedder* Embedder,
    int64 Ticket,
    FAstralRequestRef& OutRequest
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateEmbeddingRequest);

    OutRequest = FAstralRequestRef{};
    if (Embedder == nullptr || !Embedder->IsValid() || Ticket <= 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralRequestRef Native{};
    const AstralErr Err =
        astral_request_from_embedding(static_cast<AstralHandle>(Embedder->GetHandle()), static_cast<uint64_t>(Ticket), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutRequest = from_native_request_ref(Native);
    return make_operation_result(ASTRAL_OK, OutRequest.OwnerHandle);
}

bool UAstralBlueprintLibrary::CreateMemorySearchRequest(
    int64 CursorHandle,
    FAstralRequestRef& OutRequest,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = CreateMemorySearchRequestResult(CursorHandle, OutRequest);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreateMemorySearchRequestResult(
    int64 CursorHandle,
    FAstralRequestRef& OutRequest
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateMemorySearchRequest);

    OutRequest = FAstralRequestRef{};
    if (CursorHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralRequestRef Native{};
    const AstralErr Err = astral_request_from_memory_search(static_cast<AstralHandle>(CursorHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutRequest = from_native_request_ref(Native);
    return make_operation_result(ASTRAL_OK, CursorHandle);
}

bool UAstralBlueprintLibrary::GetRequestStatus(
    const FAstralRequestRef& Request,
    FAstralRequestStatus& OutStatus,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = GetRequestStatusResult(Request, OutStatus);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::GetRequestStatusResult(
    const FAstralRequestRef& Request,
    FAstralRequestStatus& OutStatus
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_GetRequestStatus);

    OutStatus = FAstralRequestStatus{};
    AstralRequestRef NativeRequest = to_native_request_ref(Request);
    AstralRequestStatus NativeStatus{};
    NativeStatus.size = sizeof(AstralRequestStatus);
    const AstralErr Err = astral_request_state(&NativeRequest, &NativeStatus);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutStatus = from_native_request_status(NativeStatus);
    return make_operation_result(ASTRAL_OK, static_cast<int64>(NativeStatus.owner), static_cast<int32>(NativeStatus.queue_depth));
}

bool UAstralBlueprintLibrary::WaitRequest(
    const FAstralRequestRef& Request,
    int32 TimeoutMs,
    FAstralRequestStatus& OutStatus,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = WaitRequestResult(Request, TimeoutMs, OutStatus);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::WaitRequestResult(
    const FAstralRequestRef& Request,
    int32 TimeoutMs,
    FAstralRequestStatus& OutStatus
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_WaitRequest);

    OutStatus = FAstralRequestStatus{};
    if (TimeoutMs < 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralRequestRef NativeRequest = to_native_request_ref(Request);
    AstralRequestStatus NativeStatus{};
    NativeStatus.size = sizeof(AstralRequestStatus);
    const AstralErr Err = astral_request_wait(&NativeRequest, static_cast<uint32_t>(TimeoutMs), &NativeStatus);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutStatus = from_native_request_status(NativeStatus);
    return make_operation_result(ASTRAL_OK, static_cast<int64>(NativeStatus.owner), static_cast<int32>(NativeStatus.queue_depth));
}

bool UAstralBlueprintLibrary::CancelRequest(const FAstralRequestRef& Request, int32& OutErrorCode)
{
    const FAstralOperationResult Result = CancelRequestResult(Request);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CancelRequestResult(const FAstralRequestRef& Request)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CancelRequest);

    AstralRequestRef NativeRequest = to_native_request_ref(Request);
    const AstralErr Err = astral_request_cancel(&NativeRequest);
    return make_operation_result(Err, Request.OwnerHandle);
}

bool UAstralBlueprintLibrary::CreatePromptCache(const FAstralPromptCacheDesc& Desc, int64& OutCacheHandle, int32& OutErrorCode)
{
    const FAstralOperationResult Result = CreatePromptCacheResult(Desc);
    OutCacheHandle = Result.Handle;
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreatePromptCacheResult(const FAstralPromptCacheDesc& Desc)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreatePromptCache);

    if (Desc.MaxEntries < 0 || Desc.MaxTokens < 0 || Desc.MaxBytes < 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralPromptCacheDesc Native = to_native_prompt_cache_desc(Desc);
    AstralHandle Handle = 0;
    const AstralErr Err = astral_prompt_cache_create(&Native, &Handle);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, static_cast<int64>(Handle));
}

bool UAstralBlueprintLibrary::LoadPromptCache(
    const FAstralPromptCacheDesc& Desc,
    const TArray<uint8>& Bytes,
    int64& OutCacheHandle,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = LoadPromptCacheResult(Desc, Bytes);
    OutCacheHandle = Result.Handle;
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::LoadPromptCacheResult(const FAstralPromptCacheDesc& Desc, const TArray<uint8>& Bytes)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_LoadPromptCache);

    if (Desc.MaxEntries < 0 || Desc.MaxTokens < 0 || Desc.MaxBytes < 0 || Bytes.Num() == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralPromptCacheDesc NativeDesc = to_native_prompt_cache_desc(Desc);
    AstralSpanU8 Span{};
    Span.data = Bytes.GetData();
    Span.len = static_cast<uint32_t>(Bytes.Num());

    AstralHandle Handle = 0;
    const AstralErr Err = astral_prompt_cache_load(&NativeDesc, Span, &Handle);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, static_cast<int64>(Handle));
}

void UAstralBlueprintLibrary::DestroyPromptCache(int64 CacheHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_DestroyPromptCache);

    if (CacheHandle != 0)
    {
        astral_prompt_cache_destroy(static_cast<AstralHandle>(CacheHandle));
    }
}

bool UAstralBlueprintLibrary::ClearPromptCache(int64 CacheHandle, int32& OutErrorCode)
{
    const FAstralOperationResult Result = ClearPromptCacheResult(CacheHandle);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::ClearPromptCacheResult(int64 CacheHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ClearPromptCache);

    if (CacheHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    const AstralErr Err = astral_prompt_cache_clear(static_cast<AstralHandle>(CacheHandle));
    return make_operation_result(Err, CacheHandle);
}

bool UAstralBlueprintLibrary::GetPromptCacheStats(int64 CacheHandle, FAstralPromptCacheStats& OutStats, int32& OutErrorCode)
{
    const FAstralOperationResult Result = GetPromptCacheStatsResult(CacheHandle, OutStats);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::GetPromptCacheStatsResult(int64 CacheHandle, FAstralPromptCacheStats& OutStats)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_GetPromptCacheStats);

    OutStats = FAstralPromptCacheStats();
    if (CacheHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralPromptCacheStats Native{};
    Native.size = sizeof(AstralPromptCacheStats);
    const AstralErr Err = astral_prompt_cache_stats(static_cast<AstralHandle>(CacheHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutStats = from_native_prompt_cache_stats(Native);
    return make_operation_result(ASTRAL_OK, CacheHandle, OutStats.Entries);
}

bool UAstralBlueprintLibrary::PutPromptCacheTokens(
    int64 CacheHandle,
    const FAstralPromptCacheKey& Key,
    const TArray<int32>& Tokens,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = PutPromptCacheTokensResult(CacheHandle, Key, Tokens);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::PutPromptCacheTokensResult(
    int64 CacheHandle,
    const FAstralPromptCacheKey& Key,
    const TArray<int32>& Tokens
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_PutPromptCacheTokens);

    if (CacheHandle == 0 || Key.ModelHandle == 0 || Key.Generation < 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralPromptCacheKey NativeKey = to_native_prompt_cache_key(Key);
    const AstralErr Err = astral_prompt_cache_put_tokens(
        static_cast<AstralHandle>(CacheHandle),
        &NativeKey,
        Tokens.GetData(),
        static_cast<uint32_t>(Tokens.Num())
    );
    return make_operation_result(Err, CacheHandle, Tokens.Num());
}

bool UAstralBlueprintLibrary::GetPromptCacheTokens(
    int64 CacheHandle,
    const FAstralPromptCacheKey& Key,
    int32 MaxTokens,
    TArray<int32>& OutTokens,
    int32& OutErrorCode
)
{
    const FAstralOperationResult Result = GetPromptCacheTokensResult(CacheHandle, Key, MaxTokens, OutTokens);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::GetPromptCacheTokensResult(
    int64 CacheHandle,
    const FAstralPromptCacheKey& Key,
    int32 MaxTokens,
    TArray<int32>& OutTokens
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_GetPromptCacheTokens);

    OutTokens.Reset();
    if (CacheHandle == 0 || Key.ModelHandle == 0 || Key.Generation < 0 || MaxTokens < 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    OutTokens.SetNumZeroed(MaxTokens);
    AstralPromptCacheKey NativeKey = to_native_prompt_cache_key(Key);
    uint32_t TokenCount = 0;
    const AstralErr Err = astral_prompt_cache_get_tokens(
        static_cast<AstralHandle>(CacheHandle),
        &NativeKey,
        OutTokens.GetData(),
        static_cast<uint32_t>(OutTokens.Num()),
        &TokenCount
    );
    if (Err != ASTRAL_OK)
    {
        OutTokens.Reset();
        return make_operation_result(Err);
    }

    OutTokens.SetNum(static_cast<int32>(TokenCount), EAllowShrinking::No);
    return make_operation_result(ASTRAL_OK, CacheHandle, static_cast<int32>(TokenCount));
}

bool UAstralBlueprintLibrary::SavePromptCache(int64 CacheHandle, TArray<uint8>& OutBytes, int32& OutErrorCode)
{
    const FAstralOperationResult Result = SavePromptCacheResult(CacheHandle, OutBytes);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::SavePromptCacheResult(int64 CacheHandle, TArray<uint8>& OutBytes)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_SavePromptCache);

    OutBytes.Reset();
    if (CacheHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    uint32_t ByteCount = 0;
    AstralErr Err = astral_prompt_cache_save_size(static_cast<AstralHandle>(CacheHandle), &ByteCount);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutBytes.SetNumUninitialized(static_cast<int32>(ByteCount));
    AstralMutSpanU8 Span{};
    Span.data = OutBytes.GetData();
    Span.len = ByteCount;

    uint32_t Written = 0;
    Err = astral_prompt_cache_save(static_cast<AstralHandle>(CacheHandle), Span, &Written);
    if (Err != ASTRAL_OK)
    {
        OutBytes.Reset();
        return make_operation_result(Err);
    }

    OutBytes.SetNum(static_cast<int32>(Written), EAllowShrinking::No);
    return make_operation_result(ASTRAL_OK, CacheHandle, static_cast<int32>(Written));
}

bool UAstralBlueprintLibrary::CreateAgent(const FAstralAgentDesc& Desc, int64& OutAgentHandle, int32& OutErrorCode)
{
    const FAstralOperationResult Result = CreateAgentResult(Desc);
    OutAgentHandle = Result.Handle;
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CreateAgentResult(const FAstralAgentDesc& Desc)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateAgent);

    if (Desc.ModelHandle == 0 || Desc.MaxTokens < 0 || Desc.TopK < 0 || Desc.MaxMessages < 0 || Desc.MaxPromptBytes < 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralAgentDesc Native{};
    Native.size = sizeof(AstralAgentDesc);
    Native.model = static_cast<AstralHandle>(Desc.ModelHandle);
    Native.prompt_cache = static_cast<AstralHandle>(Desc.PromptCacheHandle);
    Native.memory_index = static_cast<AstralHandle>(Desc.MemoryIndexHandle);
    Native.toolset = static_cast<AstralHandle>(Desc.ToolsetHandle);
    Native.max_tokens = static_cast<uint32_t>(Desc.MaxTokens);
    Native.temperature = Desc.Temperature;
    Native.top_k = static_cast<uint32_t>(Desc.TopK);
    Native.top_p = Desc.TopP;
    Native.stream_enabled = Desc.bStream ? 1 : 0;
    Native.seed = static_cast<uint32_t>(Desc.Seed);
    Native.tool_choice_mode = to_native_tool_choice(Desc.ToolChoiceMode);
    Native.max_messages = static_cast<uint32_t>(Desc.MaxMessages);
    Native.max_prompt_bytes = static_cast<uint32_t>(Desc.MaxPromptBytes);
    Native.overflow_policy = static_cast<AstralAgentOverflowPolicy>(Desc.OverflowPolicy);

    AstralHandle Handle = 0;
    const AstralErr Err = astral_agent_create(&Native, &Handle);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, static_cast<int64>(Handle));
}

void UAstralBlueprintLibrary::DestroyAgent(int64 AgentHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_DestroyAgent);

    if (AgentHandle != 0)
    {
        astral_agent_destroy(static_cast<AstralHandle>(AgentHandle));
    }
}

bool UAstralBlueprintLibrary::SetAgentSystemPrompt(int64 AgentHandle, const FString& SystemPrompt, int32& OutErrorCode)
{
    const FAstralOperationResult Result = SetAgentSystemPromptResult(AgentHandle, SystemPrompt);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::SetAgentSystemPromptResult(int64 AgentHandle, const FString& SystemPrompt)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_SetAgentSystemPrompt);

    if (AgentHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    FTCHARToUTF8 Utf8(*SystemPrompt);
    AstralSpanU8 Span{};
    Span.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
    Span.len = static_cast<uint32_t>(Utf8.Length());
    const AstralErr Err = astral_agent_set_system_prompt(static_cast<AstralHandle>(AgentHandle), Span);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, AgentHandle, Utf8.Length());
}

bool UAstralBlueprintLibrary::SetAgentSummary(int64 AgentHandle, const FString& Summary, int32& OutErrorCode)
{
    const FAstralOperationResult Result = SetAgentSummaryResult(AgentHandle, Summary);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::SetAgentSummaryResult(int64 AgentHandle, const FString& Summary)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_SetAgentSummary);

    if (AgentHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    FTCHARToUTF8 Utf8(*Summary);
    AstralSpanU8 Span{};
    Span.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
    Span.len = static_cast<uint32_t>(Utf8.Length());
    const AstralErr Err = astral_agent_set_summary(static_cast<AstralHandle>(AgentHandle), Span);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, AgentHandle, Utf8.Length());
}

bool UAstralBlueprintLibrary::GetAgentSummary(int64 AgentHandle, FString& OutSummary, int32& OutErrorCode)
{
    const FAstralOperationResult Result = GetAgentSummaryResult(AgentHandle, OutSummary);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::GetAgentSummaryResult(int64 AgentHandle, FString& OutSummary)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_GetAgentSummary);

    OutSummary.Reset();
    if (AgentHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    uint32_t ByteCount = 0;
    AstralErr Err = astral_agent_get_summary_size(static_cast<AstralHandle>(AgentHandle), &ByteCount);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    if (ByteCount == kNoElements)
    {
        return make_operation_result(ASTRAL_OK, AgentHandle, kEmptyResultCount);
    }

    TArray<uint8, TInlineAllocator<kInlineAgentTextBytes>> Bytes;
    Bytes.SetNumUninitialized(static_cast<int32>(ByteCount));
    AstralMutSpanU8 Out{};
    Out.data = Bytes.GetData();
    Out.len = ByteCount;
    uint32_t Written = 0;
    Err = astral_agent_get_summary(static_cast<AstralHandle>(AgentHandle), Out, &Written);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    AstralSpanU8 TextSpan{};
    TextSpan.data = Bytes.GetData();
    TextSpan.len = Written;
    OutSummary = utf8_span_to_string(TextSpan);
    return make_operation_result(ASTRAL_OK, AgentHandle, static_cast<int32>(Written));
}

bool UAstralBlueprintLibrary::SetAgentMemoryContext(int64 AgentHandle, const FString& MemoryContext, int32& OutErrorCode)
{
    const FAstralOperationResult Result = SetAgentMemoryContextResult(AgentHandle, MemoryContext);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::SetAgentMemoryContextResult(int64 AgentHandle, const FString& MemoryContext)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_SetAgentMemoryContext);

    if (AgentHandle == kInvalidAstralHandle)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    FTCHARToUTF8 Utf8(*MemoryContext);
    AstralSpanU8 Span{};
    Span.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
    Span.len = static_cast<uint32_t>(Utf8.Length());
    const AstralErr Err = astral_agent_set_memory_context(static_cast<AstralHandle>(AgentHandle), Span);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, AgentHandle, Utf8.Length());
}

bool UAstralBlueprintLibrary::GetAgentMemoryContext(int64 AgentHandle, FString& OutMemoryContext, int32& OutErrorCode)
{
    const FAstralOperationResult Result = GetAgentMemoryContextResult(AgentHandle, OutMemoryContext);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::GetAgentMemoryContextResult(int64 AgentHandle, FString& OutMemoryContext)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_GetAgentMemoryContext);

    OutMemoryContext.Reset();
    if (AgentHandle == kInvalidAstralHandle)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    uint32_t ByteCount = 0;
    AstralErr Err = astral_agent_get_memory_context_size(static_cast<AstralHandle>(AgentHandle), &ByteCount);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    if (ByteCount == kNoElements)
    {
        return make_operation_result(ASTRAL_OK, AgentHandle, kEmptyResultCount);
    }

    TArray<uint8, TInlineAllocator<kInlineAgentTextBytes>> Bytes;
    Bytes.SetNumUninitialized(static_cast<int32>(ByteCount));
    AstralMutSpanU8 Out{};
    Out.data = Bytes.GetData();
    Out.len = ByteCount;
    uint32_t Written = 0;
    Err = astral_agent_get_memory_context(static_cast<AstralHandle>(AgentHandle), Out, &Written);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    AstralSpanU8 TextSpan{};
    TextSpan.data = Bytes.GetData();
    TextSpan.len = Written;
    OutMemoryContext = utf8_span_to_string(TextSpan);
    return make_operation_result(ASTRAL_OK, AgentHandle, static_cast<int32>(Written));
}

bool UAstralBlueprintLibrary::ParseAgentToolCall(
    int64 AgentHandle,
    const FString& GeneratedText,
    FAstralToolCallResult& OutResult
)
{
    return ParseAgentToolCallResult(AgentHandle, GeneratedText, OutResult).bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::ParseAgentToolCallResult(
    int64 AgentHandle,
    const FString& GeneratedText,
    FAstralToolCallResult& OutResult
)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ParseAgentToolCall);

    OutResult = FAstralToolCallResult{};
    if (AgentHandle == kInvalidAstralHandle)
    {
        OutResult.ParseStatus = static_cast<int32>(ASTRAL_E_INVALID);
        return make_operation_result(ASTRAL_E_INVALID);
    }

    FTCHARToUTF8 GeneratedUtf8(*GeneratedText);
    AstralSpanU8 Text{};
    Text.data = reinterpret_cast<const uint8_t*>(GeneratedUtf8.Get());
    Text.len = static_cast<uint32_t>(GeneratedUtf8.Length());

    AstralToolCallResult Native{};
    Native.size = sizeof(AstralToolCallResult);
    const AstralErr Err = astral_agent_parse_tool_call(static_cast<AstralHandle>(AgentHandle), Text, &Native);
    if (Err != ASTRAL_OK)
    {
        OutResult.ParseStatus = static_cast<int32>(Err);
        return make_operation_result(Err);
    }

    OutResult.bFound = true;
    OutResult.ParseStatus = Native.parse_status;
    OutResult.ToolId = static_cast<int32>(Native.tool_id);
    OutResult.Name = utf8_span_to_string(Native.name);
    OutResult.ArgumentsJson = utf8_span_to_string(Native.arguments_json);
    return make_operation_result(ASTRAL_OK, AgentHandle, static_cast<int32>(Native.parse_status));
}

bool UAstralBlueprintLibrary::AddAgentMessage(int64 AgentHandle, EAstralAgentRole Role, const FString& Text, int32& OutErrorCode)
{
    const FAstralOperationResult Result = AddAgentMessageResult(AgentHandle, Role, Text);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::AddAgentMessageResult(int64 AgentHandle, EAstralAgentRole Role, const FString& Text)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_AddAgentMessage);

    if (AgentHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    FTCHARToUTF8 Utf8(*Text);
    AstralAgentMessage Native{};
    Native.size = sizeof(AstralAgentMessage);
    Native.role = to_native_agent_role(Role);
    Native.content.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
    Native.content.len = static_cast<uint32_t>(Utf8.Length());

    const AstralErr Err = astral_agent_message_add(static_cast<AstralHandle>(AgentHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, AgentHandle, Utf8.Length());
}

bool UAstralBlueprintLibrary::ClearAgentHistory(int64 AgentHandle, int32& OutErrorCode)
{
    const FAstralOperationResult Result = ClearAgentHistoryResult(AgentHandle);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::ClearAgentHistoryResult(int64 AgentHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ClearAgentHistory);

    const AstralErr Err = astral_agent_history_clear(static_cast<AstralHandle>(AgentHandle));
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, AgentHandle);
}

bool UAstralBlueprintLibrary::SaveAgentHistory(int64 AgentHandle, TArray<uint8>& OutBytes, int32& OutErrorCode)
{
    const FAstralOperationResult Result = SaveAgentHistoryResult(AgentHandle, OutBytes);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::SaveAgentHistoryResult(int64 AgentHandle, TArray<uint8>& OutBytes)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_SaveAgentHistory);

    OutBytes.Reset();
    if (AgentHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    uint32_t ByteCount = 0;
    AstralErr Err = astral_agent_history_save_size(static_cast<AstralHandle>(AgentHandle), &ByteCount);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }

    OutBytes.SetNumUninitialized(static_cast<int32>(ByteCount));
    AstralMutSpanU8 Span{};
    Span.data = OutBytes.GetData();
    Span.len = ByteCount;

    uint32_t Written = 0;
    Err = astral_agent_history_save(static_cast<AstralHandle>(AgentHandle), Span, &Written);
    if (Err != ASTRAL_OK)
    {
        OutBytes.Reset();
        return make_operation_result(Err);
    }

    OutBytes.SetNum(static_cast<int32>(Written), EAllowShrinking::No);
    return make_operation_result(ASTRAL_OK, AgentHandle, static_cast<int32>(Written));
}

bool UAstralBlueprintLibrary::LoadAgentHistory(int64 AgentHandle, const TArray<uint8>& Bytes, int32& OutErrorCode)
{
    const FAstralOperationResult Result = LoadAgentHistoryResult(AgentHandle, Bytes);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::LoadAgentHistoryResult(int64 AgentHandle, const TArray<uint8>& Bytes)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_LoadAgentHistory);

    if (AgentHandle == 0 || Bytes.Num() == kNoElements)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    AstralSpanU8 Span{};
    Span.data = Bytes.GetData();
    Span.len = static_cast<uint32_t>(Bytes.Num());

    const AstralErr Err = astral_agent_history_load(static_cast<AstralHandle>(AgentHandle), Span);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, AgentHandle, Bytes.Num());
}

bool UAstralBlueprintLibrary::EnqueueAgentChat(int64 AgentHandle, const FString& UserMessage, bool bWarmupOnly, int32& OutErrorCode)
{
    const FAstralOperationResult Result = EnqueueAgentChatResult(AgentHandle, UserMessage, bWarmupOnly);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::EnqueueAgentChatResult(int64 AgentHandle, const FString& UserMessage, bool bWarmupOnly)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_EnqueueAgentChat);

    if (AgentHandle == 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    FTCHARToUTF8 Utf8(*UserMessage);
    AstralAgentChatDesc Native{};
    Native.size = sizeof(AstralAgentChatDesc);
    Native.flags = bWarmupOnly ? ASTRAL_AGENT_CHAT_FLAG_WARMUP : ASTRAL_AGENT_CHAT_FLAG_NONE;
    Native.user_message.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
    Native.user_message.len = static_cast<uint32_t>(Utf8.Length());

    const AstralErr Err = astral_agent_chat_enqueue(static_cast<AstralHandle>(AgentHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, AgentHandle, Utf8.Length());
}

bool UAstralBlueprintLibrary::CancelAgentChat(int64 AgentHandle, int32& OutErrorCode)
{
    const FAstralOperationResult Result = CancelAgentChatResult(AgentHandle);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::CancelAgentChatResult(int64 AgentHandle)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CancelAgentChat);

    const AstralErr Err = astral_agent_chat_cancel(static_cast<AstralHandle>(AgentHandle));
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    return make_operation_result(ASTRAL_OK, AgentHandle);
}

bool UAstralBlueprintLibrary::ReadAgentChat(int64 AgentHandle, int32 TimeoutMs, FString& OutText, bool& bEndOfStream, int32& OutErrorCode)
{
    const FAstralOperationResult Result = ReadAgentChatResult(AgentHandle, TimeoutMs, OutText);
    bEndOfStream = Result.bEndOfStream;
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::ReadAgentChatResult(int64 AgentHandle, int32 TimeoutMs, FString& OutText)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ReadAgentChat);

    OutText.Reset();
    if (AgentHandle == 0 || TimeoutMs < 0)
    {
        return make_operation_result(ASTRAL_E_INVALID);
    }

    uint8 Buffer[kAgentReadBufferBytes]{};
    AstralMutSpanU8 Out{};
    Out.data = Buffer;
    Out.len = static_cast<uint32_t>(sizeof(Buffer));
    const int32 Result = astral_agent_chat_stream_read(static_cast<AstralHandle>(AgentHandle), Out, static_cast<uint32_t>(TimeoutMs));
    if (Result == 0)
    {
        FAstralOperationResult Status = make_operation_result(ASTRAL_OK, AgentHandle);
        Status.bEndOfStream = true;
        return Status;
    }
    if (Result < 0)
    {
        return make_operation_result(static_cast<AstralErr>(Result), AgentHandle);
    }

    AstralSpanU8 TextSpan{};
    TextSpan.data = Buffer;
    TextSpan.len = static_cast<uint32_t>(Result);
    OutText = utf8_span_to_string(TextSpan);
    return make_operation_result(ASTRAL_OK, AgentHandle, Result);
}

bool UAstralBlueprintLibrary::GetAgentChatResult(int64 AgentHandle, FAstralAgentChatResult& OutResult, int32& OutErrorCode)
{
    const FAstralOperationResult Result = GetAgentChatStatusResult(AgentHandle, OutResult);
    OutErrorCode = Result.ErrorCode;
    return Result.bSuccess;
}

FAstralOperationResult UAstralBlueprintLibrary::GetAgentChatStatusResult(int64 AgentHandle, FAstralAgentChatResult& OutResult)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_GetAgentChatResult);

    AstralAgentChatResult Native{};
    Native.size = sizeof(AstralAgentChatResult);
    const AstralErr Err = astral_agent_chat_result(static_cast<AstralHandle>(AgentHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err);
    }
    OutResult = from_native_agent_result(Native);
    return make_operation_result(ASTRAL_OK, AgentHandle, OutResult.PromptTokens);
}

bool UAstralBlueprintLibrary::HasEmbeddings(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_EMBEDDINGS);
}

bool UAstralBlueprintLibrary::HasSamplerControls(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_SAMPLER_EXT);
}

bool UAstralBlueprintLibrary::HasStopSequences(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_STOP_SEQS);
}

bool UAstralBlueprintLibrary::HasGpuOffload(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_GPU_OFFLOAD);
}

bool UAstralBlueprintLibrary::HasLora(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_LORA);
}

bool UAstralBlueprintLibrary::HasImageInput(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_IMAGE);
}

bool UAstralBlueprintLibrary::HasAudioInput(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_AUDIO);
}

bool UAstralBlueprintLibrary::HasMultimodalEmbeddings(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_MM_EMBEDDINGS);
}

bool UAstralBlueprintLibrary::HasGrammar(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_GRAMMAR);
}

bool UAstralBlueprintLibrary::HasLogprobs(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_LOGPROBS);
}

bool UAstralBlueprintLibrary::HasKvState(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_KV_STATE);
}

bool UAstralBlueprintLibrary::HasSlots(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_SLOTS);
}

bool UAstralBlueprintLibrary::HasGbnfGrammar(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_GRAMMAR_GBNF);
}

bool UAstralBlueprintLibrary::HasJsonSchemaGrammar(int64 Caps)
{
    return has_cap(Caps, ASTRAL_CAP_GRAMMAR_JSON_SCHEMA);
}
