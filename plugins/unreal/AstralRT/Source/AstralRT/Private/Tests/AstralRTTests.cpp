#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"

#include "AstralBlueprintLibrary.h"
#include "AstralLog.h"
#include "AstralEmbedder.h"
#include "AstralMediaLibrary.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "IAstralRT.h"
#include "astral_rt.h"

#include "Engine/Texture2D.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformProcess.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/Paths.h"
#include "PixelFormat.h"
#include "UObject/Package.h"

namespace {

static bool ensure_astral_initialized(FAutomationTestBase& test) {
    // Force-load module if needed.
    (void)IAstralRT::Get();

    if (!IAstralRT::IsAvailable() || !IAstralRT::Get().IsInitialized()) {
        test.AddError(TEXT("AstralRT runtime is not initialized."));
        return false;
    }
    return true;
}

static bool bytes_equal_ascii(const TArray<uint8>& bytes, const char* lit) {
    if (lit == nullptr) {
        return false;
    }
    const int32 lit_len = FCStringAnsi::Strlen(lit);
    if (bytes.Num() != lit_len) {
        return false;
    }
    return FMemory::Memcmp(bytes.GetData(), lit, lit_len) == 0;
}

static FString bytes_to_ascii_string(const TArray<uint8>& bytes) {
    TArray<ANSICHAR> text;
    text.Reserve(bytes.Num() + 1);
    for (const uint8 b : bytes) {
        text.Add(static_cast<ANSICHAR>(b));
    }
    text.Add('\0');
    return FString(ANSI_TO_TCHAR(text.GetData()));
}

static FString printable_ascii_summary(const TArray<uint8>& bytes, int32 max_chars) {
    FString out;
    out.Reserve(FMath::Min(bytes.Num(), max_chars));
    for (const uint8 b : bytes) {
        if (out.Len() >= max_chars) {
            break;
        }
        if (b >= 0x20 && b < 0x7f) {
            out.AppendChar(static_cast<TCHAR>(b));
        } else {
            out.AppendChar(TEXT(' '));
        }
    }
    return out.TrimStartAndEnd();
}

static FString bytes_to_hex_string(const TArray<uint8>& bytes) {
    FString out;
    out.Reserve(bytes.Num() * 3);
    for (const uint8 b : bytes) {
        out += FString::Printf(TEXT("%02x "), static_cast<uint32>(b));
    }
    return out;
}

static void append_ascii(TArray<uint8>& out, const char* lit) {
    out.Reset();
    if (lit == nullptr) {
        return;
    }
    const int32 lit_len = FCStringAnsi::Strlen(lit);
    out.Append(reinterpret_cast<const uint8*>(lit), lit_len);
}

static bool env_enabled(const TCHAR* name) {
    const FString value = FPlatformMisc::GetEnvironmentVariable(name);
    return value == TEXT("1") || value.Equals(TEXT("true"), ESearchCase::IgnoreCase);
}

static FString readable_model_path_from_env(const TCHAR* path_env,
                                            const TCHAR* require_env,
                                            const TCHAR* test_name,
                                            FAutomationTestBase& test,
                                            bool& out_should_run) {
    const FString ModelPath = FPlatformMisc::GetEnvironmentVariable(path_env);
    const bool RequireProbe = env_enabled(require_env);
    out_should_run = false;
    if (!ModelPath.IsEmpty() && FPaths::FileExists(ModelPath)) {
        out_should_run = true;
        return ModelPath;
    }

    if (RequireProbe) {
        test.AddError(FString::Printf(
            TEXT("%s must name a readable GGUF when %s=1 for %s: %s"),
            path_env,
            require_env,
            test_name,
            ModelPath.IsEmpty() ? TEXT("<empty>") : *ModelPath));
    } else {
        test.AddInfo(FString::Printf(TEXT("%s is not configured; %s not run."), path_env, test_name));
    }
    return FString();
}

static bool run_mock_session_once(AstralHandle session, TArray<uint8>& out_bytes) {
    out_bytes.Reset();
    out_bytes.Reserve(32);

    const char* prompt = "hi";
    AstralSpanU8 prompt_span{};
    prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
    prompt_span.len = static_cast<uint32_t>(FCStringAnsi::Strlen(prompt));

    AstralErr e = astral_session_feed(session, prompt_span, 1);
    if (e != ASTRAL_OK) {
        return false;
    }

    e = astral_session_decode(session);
    if (e != ASTRAL_OK) {
        return false;
    }

    e = astral_session_wait(session, 5000);
    if (e != ASTRAL_OK) {
        return false;
    }

    uint8_t buf[64];
    for (;;) {
        AstralMutSpanU8 out{};
        out.data = buf;
        out.len = sizeof(buf);

        const int32_t n = astral_stream_read(session, out, 0);
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        if (n < 0) {
            return false;
        }
        if (n == 0) {
            break;
        }
        out_bytes.Append(buf, n);
    }
    return true;
}

static int32 drain_stream(UAstralSession& Session, TArray<uint8>& OutBytes) {
    OutBytes.Reset();
    TArray<uint8> Chunk;
    Chunk.SetNumUninitialized(512);

    int32 Total = 0;
    for (;;) {
        const int32 Count = Session.StreamRead(Chunk, 0);
        if (Count == ASTRAL_E_TIMEOUT) {
            continue;
        }
        if (Count < 0) {
            return Count;
        }
        if (Count == 0) {
            return Total;
        }
        OutBytes.Append(Chunk.GetData(), Count);
        Total += Count;
    }
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTModuleInitTest,
    "AstralRT.Module.Init",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTModuleInitTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    const bool ok = ensure_astral_initialized(*this);
    if (!ok) {
        return false;
    }
    TestTrue(TEXT("AstralRT initialized"), ok);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTBlueprintLibraryTest,
    "AstralRT.Blueprint.LibraryHelpers",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTBlueprintLibraryTest::RunTest(const FString& Parameters) {
    (void)Parameters;

    UAstralModel* Model = UAstralBlueprintLibrary::CreateAstralModel(GetTransientPackage());
    UAstralSession* Session = UAstralBlueprintLibrary::CreateAstralSession(GetTransientPackage());
    UAstralEmbedder* Embedder = UAstralBlueprintLibrary::CreateAstralEmbedder(GetTransientPackage());
    TestNotNull(TEXT("Blueprint model factory"), Model);
    TestNotNull(TEXT("Blueprint session factory"), Session);
    TestNotNull(TEXT("Blueprint embedder factory"), Embedder);

    TestEqual(TEXT("timeout error name"),
              UAstralBlueprintLibrary::ErrorCodeName(static_cast<int32>(ASTRAL_E_TIMEOUT)),
              FString(TEXT("ASTRAL_E_TIMEOUT")));
    TestEqual(TEXT("unknown error name"),
              UAstralBlueprintLibrary::ErrorCodeName(-12345),
              FString(TEXT("ASTRAL_E_-12345")));

    const int64 Caps = static_cast<int64>(ASTRAL_CAP_EMBEDDINGS |
                                          ASTRAL_CAP_SAMPLER_EXT |
                                          ASTRAL_CAP_STOP_SEQS |
                                          ASTRAL_CAP_GPU_OFFLOAD |
                                          ASTRAL_CAP_LORA |
                                          ASTRAL_CAP_IMAGE |
                                          ASTRAL_CAP_AUDIO |
                                          ASTRAL_CAP_MM_EMBEDDINGS |
                                          ASTRAL_CAP_GRAMMAR |
                                          ASTRAL_CAP_LOGPROBS |
                                          ASTRAL_CAP_KV_STATE |
                                          ASTRAL_CAP_SLOTS |
                                          ASTRAL_CAP_GRAMMAR_GBNF |
                                          ASTRAL_CAP_GRAMMAR_JSON_SCHEMA);
    TestTrue(TEXT("embeddings cap"), UAstralBlueprintLibrary::HasEmbeddings(Caps));
    TestTrue(TEXT("sampler controls cap"), UAstralBlueprintLibrary::HasSamplerControls(Caps));
    TestTrue(TEXT("stop sequences cap"), UAstralBlueprintLibrary::HasStopSequences(Caps));
    TestTrue(TEXT("gpu cap"), UAstralBlueprintLibrary::HasGpuOffload(Caps));
    TestTrue(TEXT("lora cap"), UAstralBlueprintLibrary::HasLora(Caps));
    TestTrue(TEXT("image cap"), UAstralBlueprintLibrary::HasImageInput(Caps));
    TestTrue(TEXT("audio cap"), UAstralBlueprintLibrary::HasAudioInput(Caps));
    TestTrue(TEXT("mm embeddings cap"), UAstralBlueprintLibrary::HasMultimodalEmbeddings(Caps));
    TestTrue(TEXT("grammar cap"), UAstralBlueprintLibrary::HasGrammar(Caps));
    TestTrue(TEXT("logprobs cap"), UAstralBlueprintLibrary::HasLogprobs(Caps));
    TestTrue(TEXT("kv state cap"), UAstralBlueprintLibrary::HasKvState(Caps));
    TestTrue(TEXT("slots cap"), UAstralBlueprintLibrary::HasSlots(Caps));
    TestTrue(TEXT("gbnf grammar cap"), UAstralBlueprintLibrary::HasGbnfGrammar(Caps));
    TestTrue(TEXT("json schema grammar cap"), UAstralBlueprintLibrary::HasJsonSchemaGrammar(Caps));
    TestFalse(TEXT("empty caps"), UAstralBlueprintLibrary::HasEmbeddings(0));

    constexpr int32 MockTokenCount = 3;
    constexpr int32 MockTokenA = static_cast<int32>('a');
    constexpr int32 MockTokenB = static_cast<int32>('b');
    constexpr int32 MockTokenC = static_cast<int32>('c');
    constexpr int32 MockDetokenizedBytes = 3;
    int32 InvalidTokenCount = 0;
    const FAstralOperationResult InvalidCountTokens = Model->CountTokensResult(TEXT("abc"), false, false, InvalidTokenCount);
    TestFalse(TEXT("invalid model token count fails"), InvalidCountTokens.bSuccess);
    TestEqual(TEXT("invalid model token count error"), InvalidCountTokens.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    const bool ModelLoaded = Model->Load(ModelDesc);
    TestTrue(TEXT("Blueprint model loads for tokenization"), ModelLoaded);

    int32 TokenCount = 0;
    const FAstralOperationResult CountTokens = Model->CountTokensResult(TEXT("abc"), false, false, TokenCount);
    TestTrue(TEXT("token count result succeeds"), CountTokens.bSuccess);
    TestEqual(TEXT("token count result count"), CountTokens.Count, MockTokenCount);
    TestEqual(TEXT("token count out count"), TokenCount, MockTokenCount);

    TArray<int32> Tokens;
    const FAstralOperationResult Tokenize = Model->TokenizeResult(TEXT("abc"), false, false, Tokens);
    TestTrue(TEXT("tokenize result succeeds"), Tokenize.bSuccess);
    TestEqual(TEXT("tokenize result count"), Tokenize.Count, MockTokenCount);
    TestEqual(TEXT("tokenize token count"), Tokens.Num(), MockTokenCount);
    if (Tokens.Num() == MockTokenCount) {
        TestEqual(TEXT("tokenize token a"), Tokens[0], MockTokenA);
        TestEqual(TEXT("tokenize token b"), Tokens[1], MockTokenB);
        TestEqual(TEXT("tokenize token c"), Tokens[2], MockTokenC);
    }

    FString Detokenized;
    const FAstralOperationResult Detokenize = Model->DetokenizeResult(Tokens, Detokenized);
    TestTrue(TEXT("detokenize result succeeds"), Detokenize.bSuccess);
    TestEqual(TEXT("detokenize byte count"), Detokenize.Count, MockDetokenizedBytes);
    TestEqual(TEXT("detokenize text"), Detokenized, FString(TEXT("abc")));
    Model->Release();

    TArray<uint8> EmptyBytes;
    constexpr int64 InvalidTicket = 0;
    constexpr int64 InvalidHandle = 0;
    const FAstralAsyncResult InvalidEnqueue = Embedder->EnqueueUtf8BytesResult(EmptyBytes);
    TestFalse(TEXT("invalid embedder enqueue result fails"), InvalidEnqueue.bSuccess);
    TestEqual(TEXT("invalid embedder enqueue error"), InvalidEnqueue.ErrorCode, static_cast<int32>(ASTRAL_E_STATE));
    TestEqual(TEXT("invalid embedder enqueue ticket"), InvalidEnqueue.Ticket, InvalidTicket);
    TestFalse(TEXT("invalid embedder enqueue backpressure flag"), InvalidEnqueue.bBackpressure);

    TArray<FAstralToolDesc> Tools;
    constexpr int32 ToolLookupId = 7;
    FAstralToolDesc Tool;
    Tool.ToolId = ToolLookupId;
    Tool.Name = TEXT("lookup");
    Tool.Description = TEXT("search memory");
    Tool.JsonSchema = TEXT("{}");
    Tools.Add(Tool);

    const FAstralOperationResult ToolsetResult =
        UAstralBlueprintLibrary::CreateToolsetResult(Tools, EAstralToolChoiceMode::Auto);
    TestTrue(TEXT("toolset result succeeds"), ToolsetResult.bSuccess);
    TestTrue(TEXT("toolset handle valid"), ToolsetResult.Handle != InvalidHandle);
    TestEqual(TEXT("toolset count"), ToolsetResult.Count, Tools.Num());

    FAstralToolCallResult ToolCall;
    const FAstralOperationResult ParseResult =
        UAstralBlueprintLibrary::ParseToolCallResult(ToolsetResult.Handle, TEXT("{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}"), ToolCall);
    TestTrue(TEXT("tool parse result succeeds"), ParseResult.bSuccess);
    TestTrue(TEXT("tool parse found"), ToolCall.bFound);
    TestEqual(TEXT("tool parse id"), ToolCall.ToolId, Tool.ToolId);
    TestEqual(TEXT("tool parse name"), ToolCall.Name, Tool.Name);
    TestEqual(TEXT("tool parse arguments"), ToolCall.ArgumentsJson, FString(TEXT("{\"q\":\"x\"}")));

    FAstralToolCallResult MissingToolCall;
    const FAstralOperationResult MissingTool =
        UAstralBlueprintLibrary::ParseToolCallResult(ToolsetResult.Handle, TEXT("{\"name\":\"missing\",\"arguments\":{}}"), MissingToolCall);
    TestFalse(TEXT("missing tool parse fails"), MissingTool.bSuccess);
    TestTrue(TEXT("missing tool flag"), MissingTool.bNotFound);

    UAstralBlueprintLibrary::DestroyToolset(ToolsetResult.Handle);

    constexpr int32 ChunkMaxWords = 2;
    constexpr int32 ChunkExpectedCount = 2;
    FAstralChunkerDesc ChunkDesc;
    ChunkDesc.Mode = EAstralChunkMode::Word;
    ChunkDesc.MaxUnits = ChunkMaxWords;
    ChunkDesc.OverlapUnits = 0;
    TArray<FAstralChunkRange> ChunkRanges;
    int32 ChunkError = static_cast<int32>(ASTRAL_OK);
    TestTrue(
        TEXT("chunk text ranges"),
        UAstralBlueprintLibrary::ChunkText(TEXT("alpha beta gamma"), ChunkDesc, ChunkRanges, ChunkError)
    );
    TestEqual(TEXT("chunk text count"), ChunkRanges.Num(), ChunkExpectedCount);
    FString FirstChunk;
    const FAstralOperationResult CopyChunk =
        UAstralBlueprintLibrary::CopyChunkTextResult(TEXT("alpha beta gamma"), ChunkRanges[0], FirstChunk);
    TestTrue(TEXT("copy chunk text result succeeds"), CopyChunk.bSuccess);
    TestEqual(TEXT("copy chunk text bytes"), CopyChunk.Count, ChunkRanges[0].ByteEnd - ChunkRanges[0].ByteBegin);
    TestEqual(TEXT("copy chunk text value"), FirstChunk, FString(TEXT("alpha beta")));

    constexpr int32 MemoryDim = 2;
    constexpr int32 MemoryCapacity = 3;
    constexpr int64 MemoryKeyA = 10;
    constexpr int64 MemoryKeyB = 20;
    constexpr int32 MemoryGroupA = 1;
    constexpr int32 MemoryGroupB = 2;
    constexpr int32 MemoryDocumentA = 100;
    constexpr int32 MemoryDocumentB = 200;
    constexpr int32 MemoryChunkA = 1000;
    constexpr int32 MemoryChunkB = 2000;
    constexpr float VectorOne = 1.0f;
    constexpr float VectorZero = 0.0f;
    constexpr int32 SearchTopK = 2;
    constexpr int32 AnyMemoryGroup = -1;
    constexpr int32 CursorFetchLimit = 1;
    constexpr int32 CursorRemainingAfterFetch = SearchTopK - CursorFetchLimit;
    constexpr int32 EmptySearchCount = 0;
    constexpr int32 EmptyByteCount = 0;
    FAstralMemoryIndexDesc MemoryDesc;
    MemoryDesc.Dimension = MemoryDim;
    MemoryDesc.Capacity = MemoryCapacity;
    MemoryDesc.Metric = EAstralMemoryMetric::Cosine;
    const FAstralOperationResult MemoryCreate = UAstralBlueprintLibrary::CreateMemoryIndexResult(MemoryDesc);
    TestTrue(TEXT("memory create result succeeds"), MemoryCreate.bSuccess);
    TestTrue(TEXT("memory handle valid"), MemoryCreate.Handle != InvalidHandle);

    TArray<FAstralMemoryRecord> Records;
    FAstralMemoryRecord RecordA;
    RecordA.Key = MemoryKeyA;
    RecordA.GroupId = MemoryGroupA;
    RecordA.DocumentId = MemoryDocumentA;
    RecordA.ChunkId = MemoryChunkA;
    Records.Add(RecordA);
    FAstralMemoryRecord RecordB;
    RecordB.Key = MemoryKeyB;
    RecordB.GroupId = MemoryGroupB;
    RecordB.DocumentId = MemoryDocumentB;
    RecordB.ChunkId = MemoryChunkB;
    Records.Add(RecordB);
    TArray<float> Vectors;
    Vectors.Add(VectorOne);
    Vectors.Add(VectorZero);
    Vectors.Add(VectorZero);
    Vectors.Add(VectorOne);

    const FAstralOperationResult AddMemory = UAstralBlueprintLibrary::AddMemoryBatchResult(MemoryCreate.Handle, Records, Vectors, MemoryDesc.Dimension);
    TestTrue(TEXT("memory add result succeeds"), AddMemory.bSuccess);
    TestEqual(TEXT("memory add count"), AddMemory.Count, Records.Num());

    TArray<float> Query;
    Query.Add(VectorOne);
    Query.Add(VectorZero);
    TArray<FAstralMemorySearchResult> MemoryResults;
    const FAstralOperationResult SearchMemory =
        UAstralBlueprintLibrary::SearchMemoryIndexResult(MemoryCreate.Handle, Query, SearchTopK, AnyMemoryGroup, MemoryResults);
    TestTrue(TEXT("memory search result succeeds"), SearchMemory.bSuccess);
    TestEqual(TEXT("memory search count"), SearchMemory.Count, SearchTopK);
    TestEqual(TEXT("memory top key"), MemoryResults[0].Key, RecordA.Key);

    const FAstralOperationResult BeginSearch =
        UAstralBlueprintLibrary::BeginMemorySearchResult(MemoryCreate.Handle, Query, SearchTopK, AnyMemoryGroup);
    TestTrue(TEXT("memory cursor begin succeeds"), BeginSearch.bSuccess);
    TestTrue(TEXT("memory cursor handle valid"), BeginSearch.Handle != InvalidHandle);

    FAstralRequestRef SearchRequest;
    const FAstralOperationResult CreateSearchRequest =
        UAstralBlueprintLibrary::CreateMemorySearchRequestResult(BeginSearch.Handle, SearchRequest);
    TestTrue(TEXT("memory cursor request succeeds"), CreateSearchRequest.bSuccess);
    TestEqual(TEXT("memory cursor request kind"), SearchRequest.Kind, EAstralRequestKind::MemorySearch);
    TestEqual(TEXT("memory cursor request owner"), SearchRequest.OwnerHandle, BeginSearch.Handle);

    FAstralRequestStatus SearchRequestStatus;
    const FAstralOperationResult GetSearchStatus =
        UAstralBlueprintLibrary::GetRequestStatusResult(SearchRequest, SearchRequestStatus);
    TestTrue(TEXT("memory cursor status succeeds"), GetSearchStatus.bSuccess);
    TestEqual(TEXT("memory cursor status state"), SearchRequestStatus.State, EAstralRequestState::Completed);
    TestEqual(TEXT("memory cursor status depth"), SearchRequestStatus.QueueDepth, SearchTopK);

    FAstralRequestStatus WaitSearchStatus;
    const FAstralOperationResult WaitSearch =
        UAstralBlueprintLibrary::WaitRequestResult(SearchRequest, EmptySearchCount, WaitSearchStatus);
    TestTrue(TEXT("memory cursor wait succeeds"), WaitSearch.bSuccess);
    TestEqual(TEXT("memory cursor wait state"), WaitSearchStatus.State, EAstralRequestState::Completed);
    TestEqual(TEXT("memory cursor wait depth"), WaitSearchStatus.QueueDepth, SearchTopK);

    const FAstralOperationResult CancelSearch = UAstralBlueprintLibrary::CancelRequestResult(SearchRequest);
    TestFalse(TEXT("memory cursor cancel unsupported"), CancelSearch.bSuccess);
    TestEqual(TEXT("memory cursor cancel code"), CancelSearch.ErrorCode, static_cast<int32>(EAstralError::Unsupported));

    TArray<FAstralMemorySearchResult> CursorResults;
    const FAstralOperationResult FetchSearch =
        UAstralBlueprintLibrary::FetchMemorySearchResult(BeginSearch.Handle, CursorFetchLimit, CursorResults);
    TestTrue(TEXT("memory cursor fetch succeeds"), FetchSearch.bSuccess);
    TestEqual(TEXT("memory cursor fetch count"), FetchSearch.Count, CursorFetchLimit);
    TestEqual(TEXT("memory cursor top key"), CursorResults[0].Key, RecordA.Key);

    FAstralRequestStatus SearchRequestAfterFetch;
    const FAstralOperationResult GetSearchAfterFetch =
        UAstralBlueprintLibrary::GetRequestStatusResult(SearchRequest, SearchRequestAfterFetch);
    TestTrue(TEXT("memory cursor status after fetch succeeds"), GetSearchAfterFetch.bSuccess);
    TestEqual(TEXT("memory cursor status after fetch depth"), SearchRequestAfterFetch.QueueDepth, CursorRemainingAfterFetch);
    UAstralBlueprintLibrary::EndMemorySearch(BeginSearch.Handle);

    TArray<uint8> MemoryBytes;
    const FAstralOperationResult SaveMemory = UAstralBlueprintLibrary::SaveMemoryIndexResult(MemoryCreate.Handle, MemoryBytes);
    TestTrue(TEXT("memory save succeeds"), SaveMemory.bSuccess);
    TestTrue(TEXT("memory save has bytes"), MemoryBytes.Num() > EmptyByteCount);

    const FAstralOperationResult LoadMemory = UAstralBlueprintLibrary::LoadMemoryIndexResult(MemoryDesc, MemoryBytes);
    TestTrue(TEXT("memory load succeeds"), LoadMemory.bSuccess);
    TestTrue(TEXT("memory load handle valid"), LoadMemory.Handle != InvalidHandle);

    TArray<FAstralMemorySearchResult> LoadedMemoryResults;
    const FAstralOperationResult LoadedSearchMemory =
        UAstralBlueprintLibrary::SearchMemoryIndexResult(LoadMemory.Handle, Query, SearchTopK, AnyMemoryGroup, LoadedMemoryResults);
    TestTrue(TEXT("loaded memory search succeeds"), LoadedSearchMemory.bSuccess);
    TestEqual(TEXT("loaded memory search count"), LoadedSearchMemory.Count, SearchTopK);
    TestEqual(TEXT("loaded memory top key"), LoadedMemoryResults[0].Key, RecordA.Key);

    const FAstralOperationResult RemoveMemory = UAstralBlueprintLibrary::RemoveMemoryRecordResult(LoadMemory.Handle, MemoryKeyA);
    TestTrue(TEXT("memory remove succeeds"), RemoveMemory.bSuccess);

    TArray<FAstralMemorySearchResult> RemovedMemoryResults;
    const FAstralOperationResult RemovedSearchMemory =
        UAstralBlueprintLibrary::SearchMemoryIndexResult(LoadMemory.Handle, Query, SearchTopK, AnyMemoryGroup, RemovedMemoryResults);
    TestTrue(TEXT("removed memory search succeeds"), RemovedSearchMemory.bSuccess);
    TestEqual(TEXT("removed memory search count"), RemovedSearchMemory.Count, CursorFetchLimit);
    TestEqual(TEXT("removed memory top key"), RemovedMemoryResults[0].Key, RecordB.Key);

    const FAstralOperationResult ClearMemory = UAstralBlueprintLibrary::ClearMemoryIndexResult(LoadMemory.Handle);
    TestTrue(TEXT("memory clear succeeds"), ClearMemory.bSuccess);

    TArray<FAstralMemorySearchResult> ClearedMemoryResults;
    const FAstralOperationResult ClearedSearchMemory =
        UAstralBlueprintLibrary::SearchMemoryIndexResult(LoadMemory.Handle, Query, SearchTopK, AnyMemoryGroup, ClearedMemoryResults);
    TestTrue(TEXT("cleared memory search succeeds"), ClearedSearchMemory.bSuccess);
    TestEqual(TEXT("cleared memory search count"), ClearedSearchMemory.Count, EmptySearchCount);

    UAstralBlueprintLibrary::DestroyMemoryIndex(LoadMemory.Handle);
    UAstralBlueprintLibrary::DestroyMemoryIndex(MemoryCreate.Handle);

    FAstralMemoryIndexDesc GraphMemoryDesc = MemoryDesc;
    GraphMemoryDesc.IndexKind = EAstralMemoryIndexKind::Graph;
    GraphMemoryDesc.GraphNeighbors = 2;
    GraphMemoryDesc.GraphSearch = 4;
    const FAstralOperationResult GraphMemoryCreate = UAstralBlueprintLibrary::CreateMemoryIndexResult(GraphMemoryDesc);
    TestTrue(TEXT("graph memory create result succeeds"), GraphMemoryCreate.bSuccess);
    const FAstralOperationResult GraphAddMemory =
        UAstralBlueprintLibrary::AddMemoryBatchResult(GraphMemoryCreate.Handle, Records, Vectors, GraphMemoryDesc.Dimension);
    TestTrue(TEXT("graph memory add result succeeds"), GraphAddMemory.bSuccess);
    TArray<FAstralMemorySearchResult> GraphMemoryResults;
    const FAstralOperationResult GraphSearchMemory =
        UAstralBlueprintLibrary::SearchMemoryIndexResult(GraphMemoryCreate.Handle, Query, SearchTopK, AnyMemoryGroup, GraphMemoryResults);
    TestTrue(TEXT("graph memory search result succeeds"), GraphSearchMemory.bSuccess);
    TestEqual(TEXT("graph memory search count"), GraphSearchMemory.Count, SearchTopK);
    TestEqual(TEXT("graph memory top key"), GraphMemoryResults[0].Key, RecordA.Key);
    UAstralBlueprintLibrary::DestroyMemoryIndex(GraphMemoryCreate.Handle);

    constexpr int32 PromptCacheMaxEntries = 4;
    constexpr int32 PromptCacheMaxTokens = 16;
    constexpr int64 PromptCacheModelHandle = 1;
    constexpr int64 PromptCacheSystemKey = 11;
    constexpr int32 PromptCacheGeneration = 1;
    constexpr int32 PromptCacheTokenA = 101;
    constexpr int32 PromptCacheTokenB = 202;
    constexpr int32 PromptCacheReadCapacity = 4;
    constexpr int32 PromptCacheEntryCount = 1;
    constexpr int32 PromptCacheTokenCount = 2;
    FAstralPromptCacheDesc PromptCacheDesc;
    PromptCacheDesc.MaxEntries = PromptCacheMaxEntries;
    PromptCacheDesc.MaxTokens = PromptCacheMaxTokens;
    PromptCacheDesc.bTrackStats = true;

    const FAstralOperationResult PromptCacheCreate = UAstralBlueprintLibrary::CreatePromptCacheResult(PromptCacheDesc);
    TestTrue(TEXT("prompt cache create succeeds"), PromptCacheCreate.bSuccess);
    TestTrue(TEXT("prompt cache handle valid"), PromptCacheCreate.Handle != InvalidHandle);

    FAstralPromptCacheKey PromptCacheKey;
    PromptCacheKey.Section = EAstralPromptSectionKind::System;
    PromptCacheKey.ModelHandle = PromptCacheModelHandle;
    PromptCacheKey.Key = PromptCacheSystemKey;
    PromptCacheKey.Generation = PromptCacheGeneration;

    TArray<int32> PromptCacheTokens;
    PromptCacheTokens.Add(PromptCacheTokenA);
    PromptCacheTokens.Add(PromptCacheTokenB);
    const FAstralOperationResult PromptCachePut =
        UAstralBlueprintLibrary::PutPromptCacheTokensResult(PromptCacheCreate.Handle, PromptCacheKey, PromptCacheTokens);
    TestTrue(TEXT("prompt cache put succeeds"), PromptCachePut.bSuccess);
    TestEqual(TEXT("prompt cache put count"), PromptCachePut.Count, PromptCacheTokens.Num());

    TArray<int32> PromptCacheReadTokens;
    const FAstralOperationResult PromptCacheGet =
        UAstralBlueprintLibrary::GetPromptCacheTokensResult(PromptCacheCreate.Handle, PromptCacheKey, PromptCacheReadCapacity, PromptCacheReadTokens);
    TestTrue(TEXT("prompt cache get succeeds"), PromptCacheGet.bSuccess);
    TestEqual(TEXT("prompt cache get count"), PromptCacheGet.Count, PromptCacheTokenCount);
    TestEqual(TEXT("prompt cache token a"), PromptCacheReadTokens[0], PromptCacheTokenA);
    TestEqual(TEXT("prompt cache token b"), PromptCacheReadTokens[1], PromptCacheTokenB);

    FAstralPromptCacheStats PromptCacheStats;
    const FAstralOperationResult PromptCacheStatsResult =
        UAstralBlueprintLibrary::GetPromptCacheStatsResult(PromptCacheCreate.Handle, PromptCacheStats);
    TestTrue(TEXT("prompt cache stats succeeds"), PromptCacheStatsResult.bSuccess);
    TestEqual(TEXT("prompt cache stats entries"), PromptCacheStats.Entries, PromptCacheEntryCount);
    TestEqual(TEXT("prompt cache stats tokens"), PromptCacheStats.Tokens, PromptCacheTokenCount);

    TArray<uint8> PromptCacheBytes;
    const FAstralOperationResult PromptCacheSave = UAstralBlueprintLibrary::SavePromptCacheResult(PromptCacheCreate.Handle, PromptCacheBytes);
    TestTrue(TEXT("prompt cache save succeeds"), PromptCacheSave.bSuccess);
    TestTrue(TEXT("prompt cache save has bytes"), PromptCacheBytes.Num() > 0);

    const FAstralOperationResult PromptCacheLoad = UAstralBlueprintLibrary::LoadPromptCacheResult(PromptCacheDesc, PromptCacheBytes);
    TestTrue(TEXT("prompt cache load succeeds"), PromptCacheLoad.bSuccess);
    TestTrue(TEXT("prompt cache load handle valid"), PromptCacheLoad.Handle != InvalidHandle);

    TArray<int32> LoadedPromptCacheTokens;
    const FAstralOperationResult LoadedPromptCacheGet =
        UAstralBlueprintLibrary::GetPromptCacheTokensResult(PromptCacheLoad.Handle, PromptCacheKey, PromptCacheReadCapacity, LoadedPromptCacheTokens);
    TestTrue(TEXT("loaded prompt cache get succeeds"), LoadedPromptCacheGet.bSuccess);
    TestEqual(TEXT("loaded prompt cache token count"), LoadedPromptCacheGet.Count, PromptCacheTokenCount);

    const FAstralOperationResult PromptCacheClear = UAstralBlueprintLibrary::ClearPromptCacheResult(PromptCacheCreate.Handle);
    TestTrue(TEXT("prompt cache clear succeeds"), PromptCacheClear.bSuccess);
    FAstralPromptCacheKey MissingPromptCacheKey = PromptCacheKey;
    MissingPromptCacheKey.Key = PromptCacheSystemKey + PromptCacheEntryCount;
    TArray<int32> MissingPromptCacheTokens;
    const FAstralOperationResult MissingPromptCache =
        UAstralBlueprintLibrary::GetPromptCacheTokensResult(PromptCacheCreate.Handle, MissingPromptCacheKey, PromptCacheReadCapacity, MissingPromptCacheTokens);
    TestFalse(TEXT("missing prompt cache lookup fails"), MissingPromptCache.bSuccess);
    TestTrue(TEXT("missing prompt cache flag"), MissingPromptCache.bNotFound);

    UAstralBlueprintLibrary::DestroyPromptCache(PromptCacheLoad.Handle);
    UAstralBlueprintLibrary::DestroyPromptCache(PromptCacheCreate.Handle);

    FAstralMemoryIndexDesc InvalidMemoryDesc;
    const FAstralOperationResult InvalidMemoryCreate = UAstralBlueprintLibrary::CreateMemoryIndexResult(InvalidMemoryDesc);
    TestFalse(TEXT("invalid memory create fails"), InvalidMemoryCreate.bSuccess);
    TestEqual(TEXT("invalid memory create error"), InvalidMemoryCreate.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    const FAstralOperationResult InvalidAgentCancel = UAstralBlueprintLibrary::CancelAgentChatResult(InvalidHandle);
    TestFalse(TEXT("invalid agent cancel fails"), InvalidAgentCancel.bSuccess);
    TestEqual(TEXT("invalid agent cancel error"), InvalidAgentCancel.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    FAstralRequestRef InvalidRequest;
    const FAstralOperationResult InvalidSessionRequest =
        UAstralBlueprintLibrary::CreateSessionRequestResult(nullptr, InvalidRequest);
    TestFalse(TEXT("invalid session request fails"), InvalidSessionRequest.bSuccess);
    TestEqual(TEXT("invalid session request error"), InvalidSessionRequest.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    const FAstralOperationResult InvalidConversationRequest =
        UAstralBlueprintLibrary::CreateConversationRequestResult(InvalidHandle, InvalidRequest);
    TestFalse(TEXT("invalid conversation request fails"), InvalidConversationRequest.bSuccess);
    TestEqual(TEXT("invalid conversation request error"), InvalidConversationRequest.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    const FAstralOperationResult InvalidAgentRequest =
        UAstralBlueprintLibrary::CreateAgentChatRequestResult(InvalidHandle, InvalidRequest);
    TestFalse(TEXT("invalid agent request fails"), InvalidAgentRequest.bSuccess);
    TestEqual(TEXT("invalid agent request error"), InvalidAgentRequest.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    const FAstralOperationResult InvalidEmbeddingRequest =
        UAstralBlueprintLibrary::CreateEmbeddingRequestResult(nullptr, InvalidHandle, InvalidRequest);
    TestFalse(TEXT("invalid embedding request fails"), InvalidEmbeddingRequest.bSuccess);
    TestEqual(TEXT("invalid embedding request error"), InvalidEmbeddingRequest.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    TArray<uint8> AgentHistoryBytes;
    const FAstralOperationResult InvalidAgentSave = UAstralBlueprintLibrary::SaveAgentHistoryResult(InvalidHandle, AgentHistoryBytes);
    TestFalse(TEXT("invalid agent history save fails"), InvalidAgentSave.bSuccess);
    TestEqual(TEXT("invalid agent history save error"), InvalidAgentSave.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    const FAstralOperationResult InvalidAgentLoad = UAstralBlueprintLibrary::LoadAgentHistoryResult(InvalidHandle, AgentHistoryBytes);
    TestFalse(TEXT("invalid agent history load fails"), InvalidAgentLoad.bSuccess);
    TestEqual(TEXT("invalid agent history load error"), InvalidAgentLoad.ErrorCode, static_cast<int32>(ASTRAL_E_INVALID));

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTModuleShutdownRestartTest,
    "AstralRT.Module.ShutdownRestart",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTModuleShutdownRestartTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    IAstralRT& Runtime = IAstralRT::Get();
    Runtime.ShutdownModule();
    TestFalse(TEXT("runtime reports uninitialized after shutdown"), Runtime.IsInitialized());

    Runtime.ShutdownModule();
    TestFalse(TEXT("second shutdown remains uninitialized"), Runtime.IsInitialized());

    Runtime.StartupModule();
    TestTrue(TEXT("runtime reinitializes after startup"), Runtime.IsInitialized());
    if (!Runtime.IsInitialized()) {
        return false;
    }

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(FCStringAnsi::Strlen(backend));
    model_desc.n_ctx = 128;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&model_desc, &model);
    TestEqual(TEXT("mock model loads after module restart"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestTrue(TEXT("mock model handle after module restart"), model != 0);
    if (model != 0) {
        astral_model_release(model);
    }
    return err == ASTRAL_OK && model != 0;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTModuleEnginePreExitTest,
    "AstralRT.Module.EnginePreExit",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTModuleEnginePreExitTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    IAstralRT& Runtime = IAstralRT::Get();
    Runtime.SimulateEnginePreExitForAutomation();
    TestFalse(TEXT("runtime reports uninitialized after engine pre-exit"), Runtime.IsInitialized());

    Runtime.SimulateEnginePreExitForAutomation();
    TestFalse(TEXT("second engine pre-exit remains uninitialized"), Runtime.IsInitialized());

    Runtime.StartupModule();
    TestTrue(TEXT("runtime reinitializes after engine pre-exit"), Runtime.IsInitialized());
    return Runtime.IsInitialized();
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTModuleRuntimeGenerationInvalidationTest,
    "AstralRT.Module.RuntimeGenerationInvalidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTModuleRuntimeGenerationInvalidationTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    IAstralRT& Runtime = IAstralRT::Get();

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model loads before runtime generation change"), ok);
    TestTrue(TEXT("model valid before runtime generation change"), Model->IsValid());

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 17;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session creates before runtime generation change"), ok);
    TestTrue(TEXT("session valid before runtime generation change"), Session->IsValid());

    FAstralRequestRef SessionRequest;
    const FAstralOperationResult CreateSessionRequest =
        UAstralBlueprintLibrary::CreateSessionRequestResult(Session, SessionRequest);
    TestTrue(TEXT("session request succeeds before runtime generation change"), CreateSessionRequest.bSuccess);
    TestEqual(TEXT("session request kind"), SessionRequest.Kind, EAstralRequestKind::Session);
    TestTrue(TEXT("session request owner valid"), SessionRequest.OwnerHandle != 0);

    FAstralRequestStatus SessionRequestStatus;
    const FAstralOperationResult GetSessionRequestStatus =
        UAstralBlueprintLibrary::GetRequestStatusResult(SessionRequest, SessionRequestStatus);
    TestTrue(TEXT("session request status succeeds"), GetSessionRequestStatus.bSuccess);
    TestEqual(TEXT("session request status state"), SessionRequestStatus.State, EAstralRequestState::Queued);
    TestTrue(TEXT("session request queued helper"), UAstralBlueprintLibrary::IsRequestQueued(SessionRequestStatus));
    TestTrue(TEXT("session request active helper"), UAstralBlueprintLibrary::IsRequestActive(SessionRequestStatus));
    TestFalse(TEXT("session request terminal helper"), UAstralBlueprintLibrary::IsRequestTerminal(SessionRequestStatus));
    TestFalse(TEXT("session request successful helper before completion"), UAstralBlueprintLibrary::IsRequestSuccessful(SessionRequestStatus));

    FAstralRequestStatus CompletedStatus = SessionRequestStatus;
    CompletedStatus.State = EAstralRequestState::Completed;
    CompletedStatus.ErrorCode = static_cast<int32>(EAstralError::OK);
    TestTrue(TEXT("completed request helper"), UAstralBlueprintLibrary::IsRequestCompleted(CompletedStatus));
    TestTrue(TEXT("successful request helper"), UAstralBlueprintLibrary::IsRequestSuccessful(CompletedStatus));
    TestTrue(TEXT("completed terminal helper"), UAstralBlueprintLibrary::IsRequestTerminal(CompletedStatus));
    TestFalse(TEXT("completed active helper"), UAstralBlueprintLibrary::IsRequestActive(CompletedStatus));

    CompletedStatus.State = EAstralRequestState::Failed;
    CompletedStatus.ErrorCode = static_cast<int32>(EAstralError::Backend);
    TestTrue(TEXT("failed request helper"), UAstralBlueprintLibrary::IsRequestFailed(CompletedStatus));
    TestTrue(TEXT("failed terminal helper"), UAstralBlueprintLibrary::IsRequestTerminal(CompletedStatus));
    TestFalse(TEXT("failed successful helper"), UAstralBlueprintLibrary::IsRequestSuccessful(CompletedStatus));

    CompletedStatus.State = EAstralRequestState::Canceled;
    CompletedStatus.ErrorCode = static_cast<int32>(EAstralError::Canceled);
    TestTrue(TEXT("canceled request helper"), UAstralBlueprintLibrary::IsRequestCanceled(CompletedStatus));
    TestTrue(TEXT("canceled terminal helper"), UAstralBlueprintLibrary::IsRequestTerminal(CompletedStatus));

    UAstralModel* EmbeddingModel = NewObject<UAstralModel>();
    TestNotNull(TEXT("embedding model allocated"), EmbeddingModel);

    FAstralModelDesc EmbeddingModelDesc = ModelDesc;
    EmbeddingModelDesc.bEmbeddingsOnly = true;
    ok = EmbeddingModel->Load(EmbeddingModelDesc);
    TestTrue(TEXT("embedding model loads before runtime generation change"), ok);
    TestTrue(TEXT("embedding model valid before runtime generation change"), EmbeddingModel->IsValid());

    UAstralEmbedder* Embedder = NewObject<UAstralEmbedder>();
    TestNotNull(TEXT("embedder allocated"), Embedder);
    ok = Embedder->Create(EmbeddingModel);
    TestTrue(TEXT("embedder creates before runtime generation change"), ok);
    TestTrue(TEXT("embedder valid before runtime generation change"), Embedder->IsValid());
    TestTrue(TEXT("embedder dim before runtime generation change"), Embedder->GetDim() > 0);

    Runtime.ShutdownModule();
    TestFalse(TEXT("runtime reports uninitialized after generation shutdown"), Runtime.IsInitialized());
    TestFalse(TEXT("model invalid after runtime generation change"), Model->IsValid());
    TestFalse(TEXT("embedding model invalid after runtime generation change"), EmbeddingModel->IsValid());
    TestFalse(TEXT("session invalid after runtime generation change"), Session->IsValid());
    TestFalse(TEXT("embedder invalid after runtime generation change"), Embedder->IsValid());
    TestEqual(TEXT("embedder dim cleared by generation check"), Embedder->GetDim(), 0);
    TestFalse(TEXT("stale session cancel rejected after generation change"), Session->Cancel());
    TestFalse(TEXT("stale session reset rejected after generation change"), Session->Reset(SessionDesc));
    TestEqual(
        TEXT("stale session wait returns invalid after generation change"),
        Session->Wait(0),
        static_cast<int32>(ASTRAL_E_INVALID));

    Runtime.StartupModule();
    TestTrue(TEXT("runtime reinitializes after generation invalidation"), Runtime.IsInitialized());
    TestFalse(TEXT("stale model remains invalid after runtime restart"), Model->IsValid());
    TestFalse(TEXT("stale embedding model remains invalid after runtime restart"), EmbeddingModel->IsValid());
    TestFalse(TEXT("stale session remains invalid after runtime restart"), Session->IsValid());
    TestFalse(TEXT("stale embedder remains invalid after runtime restart"), Embedder->IsValid());

    ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model reload clears stale generation handle"), ok);
    TestTrue(TEXT("model valid after reload into current generation"), Model->IsValid());

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session recreate clears stale generation handle"), ok);
    TestTrue(TEXT("session valid after recreate into current generation"), Session->IsValid());

    ok = EmbeddingModel->Load(EmbeddingModelDesc);
    TestTrue(TEXT("embedding model reload clears stale generation handle"), ok);
    TestTrue(TEXT("embedding model valid after reload into current generation"), EmbeddingModel->IsValid());

    ok = Embedder->Create(EmbeddingModel);
    TestTrue(TEXT("embedder recreate clears stale generation handle"), ok);
    TestTrue(TEXT("embedder valid after recreate into current generation"), Embedder->IsValid());

    Embedder->Destroy();
    Embedder->ConditionalBeginDestroy();
    EmbeddingModel->Release();
    EmbeddingModel->ConditionalBeginDestroy();
    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return Runtime.IsInitialized();
}

#if WITH_EDITOR
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTModuleEndPIETest,
    "AstralRT.Module.EndPIE",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTModuleEndPIETest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    IAstralRT& Runtime = IAstralRT::Get();
    Runtime.SimulateEditorEndPIEForAutomation();
    TestTrue(TEXT("runtime remains initialized after editor EndPIE"), Runtime.IsInitialized());

    Runtime.SimulateEditorEndPIEForAutomation();
    TestTrue(TEXT("second editor EndPIE remains initialized"), Runtime.IsInitialized());

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(FCStringAnsi::Strlen(backend));
    model_desc.n_ctx = 128;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&model_desc, &model);
    TestEqual(TEXT("mock model loads after editor EndPIE"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestTrue(TEXT("mock model handle after editor EndPIE"), model != 0);
    if (model != 0) {
        astral_model_release(model);
    }
    return Runtime.IsInitialized() && err == ASTRAL_OK && model != 0;
}
#endif

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTFMemoryAllocatorTest,
    "AstralRT.Memory.FMemoryAllocator",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTFMemoryAllocatorTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    IAstralRT& Runtime = IAstralRT::Get();
    Runtime.ResetAllocatorStats();

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load uses FMemory allocator"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 7;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create uses FMemory allocator"), ok);
    if (!ok) {
        Model->Release();
        Model->ConditionalBeginDestroy();
        return false;
    }

    const FAstralRTAllocatorStats LiveStats = Runtime.GetAllocatorStats();
    TestTrue(TEXT("native alloc callback called"), LiveStats.AllocCalls > 0);
    TestTrue(TEXT("native alloc callback tracked bytes"), LiveStats.AllocBytes > 0);

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();

    const FAstralRTAllocatorStats ReleasedStats = Runtime.GetAllocatorStats();
    TestTrue(TEXT("native free callback called"), ReleasedStats.FreeCalls > 0);
    TestTrue(TEXT("native free callback tracked bytes"), ReleasedStats.FreeBytes > 0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockE2ETest,
    "AstralRT.Mock.E2E",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockE2ETest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(FCStringAnsi::Strlen(backend));
    model_desc.n_ctx = 128;

    AstralHandle model = 0;
    AstralErr err = astral_model_load(&model_desc, &model);
    TestEqual(TEXT("astral_model_load"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK || model == 0) {
        return false;
    }

    AstralCaps caps = 0;
    err = astral_model_caps(model, &caps);
    TestEqual(TEXT("astral_model_caps"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestTrue(TEXT("mock caps include stop seqs"), (caps & ASTRAL_CAP_STOP_SEQS) != 0);
    TestTrue(TEXT("mock caps include sampler ext"), (caps & ASTRAL_CAP_SAMPLER_EXT) != 0);

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = 32;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    TestEqual(TEXT("astral_session_create"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK || session == 0) {
        astral_model_release(model);
        return false;
    }

    TArray<uint8> out1;
    const bool ok1 = run_mock_session_once(session, out1);
    TestTrue(TEXT("first decode ok"), ok1);
    TestTrue(TEXT("mock output == mock-backend"), bytes_equal_ascii(out1, "mock-backend"));

    err = astral_session_reset(session, &session_desc);
    TestEqual(TEXT("astral_session_reset"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));

    err = astral_session_stop_clear(session);
    TestEqual(TEXT("astral_session_stop_clear"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    const char* stop = "backend";
    AstralSpanU8 stop_span{};
    stop_span.data = reinterpret_cast<const uint8_t*>(stop);
    stop_span.len = static_cast<uint32_t>(FCStringAnsi::Strlen(stop));
    TestEqual(TEXT("stop span len"), static_cast<int32>(stop_span.len), 7);

    int32_t stop_tokens[16];
    uint32_t stop_token_count = 0;
    err = astral_tokenize(model, stop_span, stop_tokens, 16, 0, 0, &stop_token_count);
    TestEqual(TEXT("stop tokenize"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestEqual(TEXT("stop token count"), static_cast<int32>(stop_token_count), 7);
    if (stop_token_count > 0) {
        TestEqual(TEXT("stop token first"), stop_tokens[0], static_cast<int32>('b'));
        TestEqual(TEXT("stop token last"), stop_tokens[stop_token_count - 1], static_cast<int32>('d'));
    }

    err = astral_session_stop_add_utf8(session, stop_span);
    TestEqual(TEXT("astral_session_stop_add_utf8"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));

    TArray<uint8> out2;
    const bool ok2 = run_mock_session_once(session, out2);
    TestTrue(TEXT("second decode ok"), ok2);
    TestEqual(TEXT("stop output byte count"), out2.Num(), 5);
    TestEqual(TEXT("stop suppresses backend suffix"), bytes_to_ascii_string(out2), FString(TEXT("mock-")));
    if (!bytes_equal_ascii(out2, "mock-")) {
        AddError(FString::Printf(TEXT("stop output hex: %s"), *bytes_to_hex_string(out2)));
    }

    astral_session_destroy(session);
    astral_model_release(model);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockEmbeddingsTest,
    "AstralRT.Mock.Embeddings",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockEmbeddingsTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(FCStringAnsi::Strlen(backend));
    model_desc.n_ctx = 128;
    model_desc.embeddings_only = 1;

    AstralHandle model = 0;
    AstralErr err = astral_model_load(&model_desc, &model);
    TestEqual(TEXT("astral_model_load"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK || model == 0) {
        return false;
    }

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    TestEqual(TEXT("astral_model_embedding_dim"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestEqual(TEXT("mock dim"), static_cast<int32>(dim), 8);

    AstralHandle emb = 0;
    err = astral_embed_create(model, &emb);
    TestEqual(TEXT("astral_embed_create"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK || emb == 0) {
        astral_model_release(model);
        return false;
    }

    const char* text = "abc";
    AstralSpanU8 text_span{};
    text_span.data = reinterpret_cast<const uint8_t*>(text);
    text_span.len = static_cast<uint32_t>(FCStringAnsi::Strlen(text));

    uint64_t ticket = 0;
    err = astral_embed_enqueue(emb, text_span, &ticket);
    TestEqual(TEXT("astral_embed_enqueue"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK) {
        astral_embed_destroy(emb);
        astral_model_release(model);
        return false;
    }

    TArray<float> vec;
    vec.SetNumUninitialized(static_cast<int32>(dim));

    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec.GetData());
    out.len = static_cast<uint32_t>(vec.Num() * sizeof(float));

    err = astral_embed_collect(emb, ticket, out);
    TestEqual(TEXT("astral_embed_collect"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));

    // mock embedder: sum(tokens incl BOS=256) + i, so for "abc": 256 + 97 + 98 + 99 = 550.
    TestEqual(TEXT("vec[0]"), vec[0], 550.0f);
    TestEqual(TEXT("vec[1]"), vec[1], 551.0f);
    TestEqual(TEXT("vec[7]"), vec[7], 557.0f);

    astral_embed_destroy(emb);
    astral_model_release(model);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTRealGenerationSmokeTest,
    "AstralRT.Real.GenerationSmoke",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTRealGenerationSmokeTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    bool ShouldRun = false;
    const FString ModelPath = readable_model_path_from_env(
        TEXT("ASTRAL_UNREAL_TEST_MODEL"),
        TEXT("ASTRAL_UNREAL_REQUIRE_REAL_GENERATION"),
        TEXT("AstralRT.Real.GenerationSmoke"),
        *this,
        ShouldRun);
    if (!ShouldRun) {
        return !env_enabled(TEXT("ASTRAL_UNREAL_REQUIRE_REAL_GENERATION"));
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("cpu");
    ModelDesc.ModelPath = ModelPath;
    ModelDesc.ContextSize = 256;
    ModelDesc.BatchSize = 128;
    ModelDesc.NumThreads = 2;

    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("real generation model load"), ok);
    if (!ok) {
        Model->ConditionalBeginDestroy();
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 8;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 1;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = true;
    SessionDesc.Seed = 19;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("real session create"), ok);
    if (!ok) {
        Model->Release();
        Model->ConditionalBeginDestroy();
        return false;
    }

    ok = Session->FeedPrompt(TEXT("The capital of France is"), true);
    TestTrue(TEXT("real prompt feed"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("real decode"), ok);
    const int32 WaitResult = Session->Wait(120000);
    TestEqual(TEXT("real decode wait"), WaitResult, static_cast<int32>(ASTRAL_OK));

    TArray<uint8> OutBytes;
    const int32 ByteCount = drain_stream(*Session, OutBytes);
    TestTrue(TEXT("real generation produced bytes"), ByteCount > 0);

    const FString Output = printable_ascii_summary(OutBytes, 160);
    const FString Evidence = FString::Printf(
        TEXT("[unreal_generation_smoke] backend=cpu model=%s bytes=%d text=%s"),
        *ModelPath,
        ByteCount,
        *Output);
    AddInfo(Evidence);
    UE_LOG(LogAstralRT, Display, TEXT("%s"), *Evidence);

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return ok && WaitResult == ASTRAL_OK && ByteCount > 0;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTRealEmbeddingProbeTest,
    "AstralRT.Real.EmbeddingProbe",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTRealEmbeddingProbeTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    bool ShouldRun = false;
    const FString ModelPath = readable_model_path_from_env(
        TEXT("ASTRAL_UNREAL_TEST_EMBED_MODEL"),
        TEXT("ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING"),
        TEXT("AstralRT.Real.EmbeddingProbe"),
        *this,
        ShouldRun);
    if (!ShouldRun) {
        return !env_enabled(TEXT("ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING"));
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("cpu");
    ModelDesc.ModelPath = ModelPath;
    ModelDesc.ContextSize = 0;
    ModelDesc.BatchSize = 0;
    ModelDesc.NumThreads = 0;
    ModelDesc.bEmbeddingsOnly = true;

    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("real embedding model load"), ok);
    if (!ok) {
        Model->ConditionalBeginDestroy();
        return false;
    }

    UAstralEmbedder* Embedder = NewObject<UAstralEmbedder>();
    TestNotNull(TEXT("embedder allocated"), Embedder);

    ok = Embedder->Create(Model);
    TestTrue(TEXT("real embedder create"), ok);
    if (!ok) {
        Model->Release();
        Model->ConditionalBeginDestroy();
        return false;
    }

    TArray<uint8> Bytes;
    append_ascii(Bytes, "hi");

    TArray<float> Vec;
    ok = Embedder->EmbedUtf8Bytes(Bytes, Vec);
    TestTrue(TEXT("real embedding collect"), ok);
    TestTrue(TEXT("real embedding dim bounded"), Vec.Num() > 0 && Vec.Num() < 8192);

    double SumAbs = 0.0;
    for (const float Value : Vec) {
        SumAbs += static_cast<double>(FMath::Abs(Value));
    }

    const float First = Vec.Num() > 0 ? Vec[0] : 0.0f;
    const FString Evidence = FString::Printf(
        TEXT("[unreal_embedding_probe] backend=cpu model=%s dim=%d sum_abs=%.6f first=%.6f"),
        *ModelPath,
        Vec.Num(),
        SumAbs,
        First);
    AddInfo(Evidence);
    UE_LOG(LogAstralRT, Display, TEXT("%s"), *Evidence);
    TestTrue(TEXT("real embedding vector has signal"), SumAbs > 0.0);

    constexpr int32 ThroughputIters = 4;
    const char* ThroughputTexts[ThroughputIters] = {
        "gameplay memory",
        "quest state",
        "dialogue recall",
        "inventory note",
    };
    double ThroughputSumAbs = 0.0;
    const double StartSeconds = FPlatformTime::Seconds();
    for (int32 Index = 0; Index < ThroughputIters; ++Index) {
        append_ascii(Bytes, ThroughputTexts[Index]);
        ok = Embedder->EmbedUtf8Bytes(Bytes, Vec);
        TestTrue(TEXT("real embedding throughput collect"), ok);
        TestTrue(TEXT("real embedding throughput dim bounded"), Vec.Num() > 0 && Vec.Num() < 8192);
        for (const float Value : Vec) {
            ThroughputSumAbs += static_cast<double>(FMath::Abs(Value));
        }
    }
    const double ElapsedSeconds = FPlatformTime::Seconds() - StartSeconds;
    const double EmbedsPerSec = ElapsedSeconds > 0.0 ? static_cast<double>(ThroughputIters) / ElapsedSeconds : 0.0;
    const FString ThroughputEvidence = FString::Printf(
        TEXT("[unreal_embedding_throughput] backend=cpu model=%s iters=%d seconds=%.6f embeds_per_sec=%.3f sum_abs=%.6f"),
        *ModelPath,
        ThroughputIters,
        ElapsedSeconds,
        EmbedsPerSec,
        ThroughputSumAbs);
    AddInfo(ThroughputEvidence);
    UE_LOG(LogAstralRT, Display, TEXT("%s"), *ThroughputEvidence);
    TestTrue(TEXT("real embedding throughput positive"), EmbedsPerSec > 0.0);
    TestTrue(TEXT("real embedding throughput vector signal"), ThroughputSumAbs > 0.0);

    Embedder->Destroy();
    Embedder->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return ok && Vec.Num() > 0 && SumAbs > 0.0 && ThroughputSumAbs > 0.0 && EmbedsPerSec > 0.0;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTRealSessionLifecycleTest,
    "AstralRT.Real.SessionLifecycle",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTRealSessionLifecycleTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    bool ShouldRun = false;
    const FString ModelPath = readable_model_path_from_env(
        TEXT("ASTRAL_UNREAL_TEST_MODEL"),
        TEXT("ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE"),
        TEXT("AstralRT.Real.SessionLifecycle"),
        *this,
        ShouldRun);
    if (!ShouldRun) {
        return !env_enabled(TEXT("ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE"));
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("cpu");
    ModelDesc.ModelPath = ModelPath;
    ModelDesc.ContextSize = 512;
    ModelDesc.BatchSize = 128;
    ModelDesc.NumThreads = 2;

    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("real lifecycle model load"), ok);
    if (!ok) {
        Model->ConditionalBeginDestroy();
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 256;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 1;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = true;
    SessionDesc.Seed = 37;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("real lifecycle session create"), ok);
    if (!ok) {
        Model->Release();
        Model->ConditionalBeginDestroy();
        return false;
    }

    ok = Session->FeedPrompt(TEXT("Repeat the word engine until stopped: engine"), true);
    TestTrue(TEXT("real lifecycle prompt feed"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("real lifecycle decode start"), ok);
    FPlatformProcess::Sleep(0.01f);
    const bool CancelOk = Session->Cancel();
    TestTrue(TEXT("real lifecycle cancel request"), CancelOk);
    const int32 CancelWait = Session->Wait(120000);
    TestEqual(TEXT("real lifecycle cancel wait"), CancelWait, static_cast<int32>(ASTRAL_E_CANCELED));

    TArray<uint8> CanceledBytes;
    const int32 CanceledByteCount = drain_stream(*Session, CanceledBytes);
    TestTrue(TEXT("real lifecycle canceled stream drain"), CanceledByteCount >= 0);

    SessionDesc.MaxTokens = 8;
    SessionDesc.Seed = 41;
    ok = Session->Reset(SessionDesc);
    TestTrue(TEXT("real lifecycle reset after cancel"), ok);
    ok = Session->FeedPrompt(TEXT("The capital of France is"), true);
    TestTrue(TEXT("real lifecycle reuse prompt feed"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("real lifecycle reuse decode"), ok);
    const int32 ReuseWait = Session->Wait(120000);
    TestEqual(TEXT("real lifecycle reuse wait"), ReuseWait, static_cast<int32>(ASTRAL_OK));

    TArray<uint8> ReuseBytes;
    const int32 ReuseByteCount = drain_stream(*Session, ReuseBytes);
    TestTrue(TEXT("real lifecycle reuse produced bytes"), ReuseByteCount > 0);

    const FString ReuseOutput = printable_ascii_summary(ReuseBytes, 160);
    const FString Evidence = FString::Printf(
        TEXT("[unreal_session_lifecycle] backend=cpu model=%s cancel_wait=%d canceled_bytes=%d reuse_wait=%d reuse_bytes=%d text=%s"),
        *ModelPath,
        CancelWait,
        CanceledByteCount,
        ReuseWait,
        ReuseByteCount,
        *ReuseOutput);
    AddInfo(Evidence);
    UE_LOG(LogAstralRT, Display, TEXT("%s"), *Evidence);

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return ok &&
        CancelOk &&
        CancelWait == ASTRAL_E_CANCELED &&
        CanceledByteCount >= 0 &&
        ReuseWait == ASTRAL_OK &&
        ReuseByteCount > 0;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockEmbedderQueuePressureTest,
    "AstralRT.Mock.EmbedderQueuePressure",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockEmbedderQueuePressureTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    ModelDesc.bEmbeddingsOnly = true;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralEmbedder* Embedder = NewObject<UAstralEmbedder>();
    TestNotNull(TEXT("embedder allocated"), Embedder);

    ok = Embedder->Create(Model);
    TestTrue(TEXT("embedder create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    constexpr int32 Inflight = 8;
    const char* Texts[Inflight] = {
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
    };
    int64 Tickets[Inflight] = {};
    TArray<uint8> Bytes;
    constexpr int64 InvalidTicket = 0;

    for (int32 i = 0; i < Inflight; ++i) {
        append_ascii(Bytes, Texts[i]);
        ok = Embedder->EnqueueUtf8Bytes(Bytes, Tickets[i]);
        TestTrue(TEXT("enqueue inflight ticket"), ok);
        TestTrue(TEXT("ticket valid"), Tickets[i] > 0);
    }

    FAstralRequestRef EmbeddingRequest;
    const FAstralOperationResult CreateEmbeddingRequest =
        UAstralBlueprintLibrary::CreateEmbeddingRequestResult(Embedder, Tickets[0], EmbeddingRequest);
    TestTrue(TEXT("embedding request succeeds"), CreateEmbeddingRequest.bSuccess);
    TestEqual(TEXT("embedding request kind"), EmbeddingRequest.Kind, EAstralRequestKind::Embedding);
    TestEqual(TEXT("embedding request ticket"), EmbeddingRequest.Ticket, Tickets[0]);

    FAstralRequestStatus EmbeddingRequestStatus;
    const FAstralOperationResult GetEmbeddingRequestStatus =
        UAstralBlueprintLibrary::GetRequestStatusResult(EmbeddingRequest, EmbeddingRequestStatus);
    TestTrue(TEXT("embedding request status succeeds"), GetEmbeddingRequestStatus.bSuccess);
    TestEqual(TEXT("embedding request status ticket"), EmbeddingRequestStatus.Ticket, Tickets[0]);
    TestTrue(TEXT("embedding request has ticket flag"), EmbeddingRequestStatus.bHasTicket);

    int64 OverflowTicket = InvalidTicket;
    append_ascii(Bytes, "overflow");
    const FAstralAsyncResult Overflow = Embedder->EnqueueUtf8BytesResult(Bytes);
    TestFalse(TEXT("overflow result fails"), Overflow.bSuccess);
    TestEqual(TEXT("overflow error code"), Overflow.ErrorCode, static_cast<int32>(ASTRAL_E_BUSY));
    TestTrue(TEXT("overflow backpressure flag"), Overflow.bBackpressure);
    TestEqual(TEXT("overflow result ticket remains zero"), Overflow.Ticket, InvalidTicket);
    ok = Embedder->EnqueueUtf8Bytes(Bytes, OverflowTicket);
    TestFalse(TEXT("overflow returns busy through wrapper"), ok);
    TestEqual(TEXT("overflow wrapper ticket remains zero"), OverflowTicket, InvalidTicket);

    constexpr int32 CanceledIndex = 3;
    ok = Embedder->Cancel(Tickets[CanceledIndex]);
    TestTrue(TEXT("cancel queued embedding ticket"), ok);
    const FAstralAsyncResult StaleCancel = Embedder->CancelResult(Tickets[CanceledIndex]);
    TestFalse(TEXT("cancel stale embedding ticket fails"), StaleCancel.bSuccess);
    TestEqual(TEXT("cancel stale error"), StaleCancel.ErrorCode, static_cast<int32>(ASTRAL_E_NOT_FOUND));

    int64 ReplacementTicket = 0;
    append_ascii(Bytes, "replacement");
    ok = Embedder->EnqueueUtf8Bytes(Bytes, ReplacementTicket);
    TestTrue(TEXT("enqueue after cancel frees capacity"), ok);
    TestTrue(TEXT("replacement ticket valid"), ReplacementTicket > 0);

    TArray<float> Vec;
    int32 Collected = 0;
    double SumAbs = 0.0;
    const double StartSeconds = FPlatformTime::Seconds();
    for (int32 Offset = 0; Offset < Inflight; ++Offset) {
        const int32 Index = Inflight - 1 - Offset;
        if (Index == CanceledIndex) {
            continue;
        }
        ok = Embedder->Collect(Tickets[Index], Vec);
        TestTrue(TEXT("collect out of order"), ok);
        TestEqual(TEXT("vector size"), Vec.Num(), Embedder->GetDim());
        if (Vec.Num() > 0) {
            SumAbs += FMath::Abs(static_cast<double>(Vec[0]));
        }
        ++Collected;
    }

    ok = Embedder->Collect(ReplacementTicket, Vec);
    TestTrue(TEXT("collect replacement ticket"), ok);
    TestEqual(TEXT("replacement vector size"), Vec.Num(), Embedder->GetDim());
    if (Vec.Num() > 0) {
        SumAbs += FMath::Abs(static_cast<double>(Vec[0]));
    }
    ++Collected;
    const double ElapsedSeconds = FPlatformTime::Seconds() - StartSeconds;

    const FAstralAsyncResult StaleCollect = Embedder->CollectResult(Tickets[0], Vec);
    TestFalse(TEXT("stale ticket rejected"), StaleCollect.bSuccess);
    TestEqual(TEXT("stale ticket error"), StaleCollect.ErrorCode, static_cast<int32>(ASTRAL_E_NOT_FOUND));
    const FAstralAsyncResult CanceledCollect = Embedder->CollectResult(Tickets[CanceledIndex], Vec);
    TestFalse(TEXT("canceled ticket rejected"), CanceledCollect.bSuccess);
    TestEqual(TEXT("canceled ticket error"), CanceledCollect.ErrorCode, static_cast<int32>(ASTRAL_E_CANCELED));
    TestTrue(TEXT("canceled ticket flag"), CanceledCollect.bCanceled);

    int64 ReuseTicket = 0;
    append_ascii(Bytes, "reuse");
    ok = Embedder->EnqueueUtf8Bytes(Bytes, ReuseTicket);
    TestTrue(TEXT("enqueue after drain"), ok);
    TestTrue(TEXT("reuse ticket valid"), ReuseTicket > 0);
    ok = Embedder->Collect(ReuseTicket, Vec);
    TestTrue(TEXT("collect reuse ticket"), ok);

    const double EmbedsPerSec = ElapsedSeconds > 0.0 ? static_cast<double>(Collected) / ElapsedSeconds : 0.0;
    const FString Evidence = FString::Printf(
        TEXT("[unreal_embedding_acceptance] batch=%d canceled=1 backpressure=busy seconds=%.6f embeds_per_sec=%.3f sum_abs=%.6f"),
        Collected,
        ElapsedSeconds,
        EmbedsPerSec,
        SumAbs);
    AddInfo(Evidence);
    UE_LOG(LogAstralRT, Display, TEXT("%s"), *Evidence);
    TestEqual(TEXT("acceptance batch count"), Collected, Inflight);
    TestTrue(TEXT("acceptance throughput positive"), EmbedsPerSec > 0.0);
    TestTrue(TEXT("acceptance vector signal"), SumAbs > 0.0);

    Embedder->Destroy();
    Embedder->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockMemoryModelSourceTest,
    "AstralRT.Mock.MemoryModelSource",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockMemoryModelSourceTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.SourceKind = EAstralModelSourceKind::Memory;
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    ModelDesc.ModelBytes.SetNumUninitialized(4);
    ModelDesc.ModelBytes[0] = 'm';
    ModelDesc.ModelBytes[1] = 'o';
    ModelDesc.ModelBytes[2] = 'c';
    ModelDesc.ModelBytes[3] = 'k';

    const bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("memory source model load"), ok);
    TestTrue(TEXT("model valid after memory load"), Model->IsValid());

    Model->Release();
    Model->ConditionalBeginDestroy();
    return ok;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockFailedLoadRecoveryTest,
    "AstralRT.Mock.FailedLoadRecovery",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockFailedLoadRecoveryTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.SourceKind = EAstralModelSourceKind::Memory;
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;

    AddExpectedError(TEXT("AstralRT: memory model source has no bytes"), EAutomationExpectedErrorFlags::Contains, 1);
    bool ok = Model->Load(ModelDesc);
    TestFalse(TEXT("empty memory model load fails"), ok);
    TestFalse(TEXT("failed load leaves model invalid"), Model->IsValid());
    TestEqual(TEXT("failed load leaves handle zero"), Model->GetHandle(), static_cast<uint64>(0));

    int32 Dim = 7;
    ok = Model->GetEmbeddingDim(Dim);
    TestFalse(TEXT("failed load has no embedding dim"), ok);
    TestEqual(TEXT("failed dim query zeroes output"), Dim, 0);

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 11;

    AddExpectedError(TEXT("AstralRT: invalid model"), EAutomationExpectedErrorFlags::Contains, 1);
    ok = Session->Create(Model, SessionDesc);
    TestFalse(TEXT("invalid model create rejected"), ok);
    TestFalse(TEXT("failed create leaves session invalid"), Session->IsValid());

    ModelDesc.ModelBytes.SetNumUninitialized(4);
    ModelDesc.ModelBytes[0] = 'm';
    ModelDesc.ModelBytes[1] = 'o';
    ModelDesc.ModelBytes[2] = 'c';
    ModelDesc.ModelBytes[3] = 'k';

    ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model recovers after failed load"), ok);
    TestTrue(TEXT("recovered model valid"), Model->IsValid());

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create after model recovery"), ok);
    TestTrue(TEXT("recovered session valid"), Session->IsValid());

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->Release();
    TestFalse(TEXT("double release leaves model invalid"), Model->IsValid());
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockSessionCancelResetTest,
    "AstralRT.Mock.SessionCancelReset",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockSessionCancelResetTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ModelPath = TEXT("infinite");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("infinite mock model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 256;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = true;
    SessionDesc.Seed = 7;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    ok = Session->FeedPrompt(TEXT("hi"), true);
    TestTrue(TEXT("first feed"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("first decode"), ok);
    ok = Session->Cancel();
    TestTrue(TEXT("first cancel"), ok);
    TestEqual(TEXT("first wait canceled"), Session->Wait(5000), static_cast<int32>(ASTRAL_E_CANCELED));

    SessionDesc.bStreamEnabled = false;
    ok = Session->Reset(SessionDesc);
    TestTrue(TEXT("reset after cancel"), ok);

    ok = Session->FeedPrompt(TEXT("reuse"), true);
    TestTrue(TEXT("second feed"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("second decode"), ok);
    ok = Session->Cancel();
    TestTrue(TEXT("second cancel"), ok);
    TestEqual(TEXT("second wait canceled"), Session->Wait(5000), static_cast<int32>(ASTRAL_E_CANCELED));

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockDestroyInvalidationTest,
    "AstralRT.Mock.DestroyInvalidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockDestroyInvalidationTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ModelPath = TEXT("infinite");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("infinite mock model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 256;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = true;
    SessionDesc.Seed = 17;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("streaming session create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    ok = Session->FeedPrompt(TEXT("hi"), true);
    TestTrue(TEXT("feed before destroy"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("decode before destroy"), ok);
    ok = Session->Cancel();
    TestTrue(TEXT("cancel before destroy"), ok);
    TestEqual(TEXT("wait canceled before destroy"), Session->Wait(5000), static_cast<int32>(ASTRAL_E_CANCELED));

    Session->ConditionalBeginDestroy();
    TestFalse(TEXT("destroy clears session valid state"), Session->IsValid());
    TestEqual(TEXT("wait after destroy is invalid"), Session->Wait(0), static_cast<int32>(ASTRAL_E_INVALID));
    TestFalse(TEXT("cancel after destroy fails"), Session->Cancel());
    TestFalse(TEXT("reset after destroy fails"), Session->Reset(SessionDesc));

    TArray<uint8> Out;
    Out.SetNumUninitialized(16);
    TestEqual(TEXT("stream read after destroy is invalid"), Session->StreamRead(Out, 0), static_cast<int32>(ASTRAL_E_INVALID));

    Model->Release();
    TestFalse(TEXT("model release clears valid state"), Model->IsValid());
    Model->Release();
    TestFalse(TEXT("second model release stays invalid"), Model->IsValid());
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockMediaFeedTest,
    "AstralRT.Mock.MediaFeed",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockMediaFeedTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 42;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    FAstralImageDesc Image{};
    Image.Format = EAstralImageFormat::RGB8;
    Image.Width = 1;
    Image.Height = 1;
    Image.Pixels.SetNumZeroed(3);
    ok = Session->FeedImage(Image, true);
    TestTrue(TEXT("feed image"), ok);

    FAstralAudioDesc Audio{};
    Audio.Format = EAstralAudioFormat::I16;
    Audio.Channels = 1;
    Audio.SampleRate = 16000;
    Audio.FrameCount = 4;
    Audio.Samples.SetNumZeroed(8);
    ok = Session->FeedAudio(Audio, true);
    TestTrue(TEXT("feed audio"), ok);

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMediaDescriptorHelpersTest,
    "AstralRT.Media.DescriptorHelpers",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMediaDescriptorHelpersTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    TArray<uint8> Rgba;
    Rgba.SetNumZeroed(2 * 2 * 4);

    FAstralImageDesc Image{};
    bool ok = UAstralMediaLibrary::MakeRGBA8ImageFromBytes(Rgba, 2, 2, Image);
    TestTrue(TEXT("RGBA bytes descriptor"), ok);
    TestEqual(TEXT("image width"), static_cast<int32>(Image.Width), 2);
    TestEqual(TEXT("image height"), static_cast<int32>(Image.Height), 2);
    TestEqual(TEXT("image stride"), static_cast<int32>(Image.RowStride), 8);
    TestEqual(TEXT("image byte count"), Image.Pixels.Num(), Rgba.Num());
    const FAstralImageDesc GoodImage = Image;

    TArray<uint8> BadRgba;
    BadRgba.SetNumZeroed(3);
    ok = UAstralMediaLibrary::MakeRGBA8ImageFromBytes(BadRgba, 2, 2, Image);
    TestFalse(TEXT("reject undersized RGBA bytes"), ok);

    UTexture2D* Texture = UTexture2D::CreateTransient(2, 1, PF_B8G8R8A8);
    TestNotNull(TEXT("transient texture allocated"), Texture);
    if (Texture != nullptr && Texture->GetPlatformData() != nullptr && Texture->GetPlatformData()->Mips.Num() > 0) {
        FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
        void* Locked = Mip.BulkData.Lock(LOCK_READ_WRITE);
        TestNotNull(TEXT("texture bulk data locked"), Locked);
        if (Locked != nullptr) {
            uint8* Bgra = static_cast<uint8*>(Locked);
            Bgra[0] = 30;
            Bgra[1] = 20;
            Bgra[2] = 10;
            Bgra[3] = 255;
            Bgra[4] = 60;
            Bgra[5] = 50;
            Bgra[6] = 40;
            Bgra[7] = 128;
        }
        Mip.BulkData.Unlock();
    }

    FAstralImageDesc TextureImage{};
    ok = UAstralMediaLibrary::MakeRGBA8ImageFromTexture(Texture, TextureImage);
    TestTrue(TEXT("texture descriptor"), ok);
    TestEqual(TEXT("texture image format"), static_cast<uint32>(TextureImage.Format), static_cast<uint32>(EAstralImageFormat::RGBA8));
    TestEqual(TEXT("texture image width"), static_cast<int32>(TextureImage.Width), 2);
    TestEqual(TEXT("texture image height"), static_cast<int32>(TextureImage.Height), 1);
    TestEqual(TEXT("texture image stride"), static_cast<int32>(TextureImage.RowStride), 8);
    TestEqual(TEXT("texture image byte count"), TextureImage.Pixels.Num(), 8);
    if (TextureImage.Pixels.Num() == 8) {
        TestEqual(TEXT("texture first red"), TextureImage.Pixels[0], static_cast<uint8>(10));
        TestEqual(TEXT("texture first green"), TextureImage.Pixels[1], static_cast<uint8>(20));
        TestEqual(TEXT("texture first blue"), TextureImage.Pixels[2], static_cast<uint8>(30));
        TestEqual(TEXT("texture first alpha"), TextureImage.Pixels[3], static_cast<uint8>(255));
        TestEqual(TEXT("texture second red"), TextureImage.Pixels[4], static_cast<uint8>(40));
        TestEqual(TEXT("texture second green"), TextureImage.Pixels[5], static_cast<uint8>(50));
        TestEqual(TEXT("texture second blue"), TextureImage.Pixels[6], static_cast<uint8>(60));
        TestEqual(TEXT("texture second alpha"), TextureImage.Pixels[7], static_cast<uint8>(128));
    }

    TArray<uint8> Pcm;
    Pcm.SetNumZeroed(4 * static_cast<int32>(sizeof(int16)));

    FAstralAudioDesc Audio{};
    ok = UAstralMediaLibrary::MakePCM16AudioFromBytes(Pcm, 1, 16000, Audio);
    TestTrue(TEXT("PCM16 descriptor"), ok);
    TestEqual(TEXT("audio format"), static_cast<uint32>(Audio.Format), static_cast<uint32>(EAstralAudioFormat::I16));
    TestEqual(TEXT("audio channels"), static_cast<int32>(Audio.Channels), 1);
    TestEqual(TEXT("audio sample rate"), static_cast<int32>(Audio.SampleRate), 16000);
    TestEqual(TEXT("audio frame count"), static_cast<int32>(Audio.FrameCount), 4);
    const FAstralAudioDesc GoodAudio = Audio;

    TArray<uint8> BadPcm;
    BadPcm.SetNumZeroed(3);
    ok = UAstralMediaLibrary::MakePCM16AudioFromBytes(BadPcm, 2, 16000, Audio);
    TestFalse(TEXT("reject unaligned PCM16 bytes"), ok);

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 13;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    ok = Session->FeedImage(GoodImage, false);
    TestTrue(TEXT("feed helper image"), ok);
    ok = Session->FeedImage(TextureImage, false);
    TestTrue(TEXT("feed texture helper image"), ok);
    ok = Session->FeedAudio(GoodAudio, true);
    TestTrue(TEXT("feed helper audio"), ok);

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockMultimodalEmbedTest,
    "AstralRT.Mock.MultimodalEmbed",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockMultimodalEmbedTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    ModelDesc.bEmbeddingsOnly = true;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralEmbedder* Embedder = NewObject<UAstralEmbedder>();
    TestNotNull(TEXT("embedder allocated"), Embedder);

    ok = Embedder->Create(Model);
    TestTrue(TEXT("embedder create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    FAstralImageDesc Image{};
    Image.Format = EAstralImageFormat::RGB8;
    Image.Width = 1;
    Image.Height = 1;
    Image.RowStride = 3;
    Image.Pixels.SetNumZeroed(3);

    FAstralAudioDesc Audio{};
    Audio.Format = EAstralAudioFormat::I16;
    Audio.Channels = 1;
    Audio.SampleRate = 16000;
    Audio.FrameCount = 4;
    Audio.Samples.SetNumZeroed(8);

    int64 Ticket = 0;
    ok = Embedder->EnqueueMultimodal(TEXT("abc"), Image, Audio, true, false, Ticket);
    TestTrue(TEXT("enqueue multimodal"), ok);
    TestTrue(TEXT("ticket valid"), Ticket > 0);

    TArray<float> Vec;
    ok = Embedder->Collect(Ticket, Vec);
    TestTrue(TEXT("collect embedding"), ok);
    TestTrue(TEXT("vector size"), Vec.Num() == Embedder->GetDim());

    Embedder->Destroy();
    Embedder->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
