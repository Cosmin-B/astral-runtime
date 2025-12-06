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
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_CreateToolset);

    OutToolsetHandle = 0;
    if (Tools.Num() == 0)
    {
        return false;
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
        return false;
    }

    OutToolsetHandle = static_cast<int64>(Toolset);
    return true;
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
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralBlueprint_ParseToolCall);

    OutResult = FAstralToolCallResult{};
    if (ToolsetHandle == 0)
    {
        OutResult.ParseStatus = static_cast<int32>(ASTRAL_E_INVALID);
        return false;
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
        return false;
    }

    OutResult.bFound = true;
    OutResult.ParseStatus = Native.parse_status;
    OutResult.ToolId = static_cast<int32>(Native.tool_id);
    OutResult.Name = utf8_span_to_string(Native.name);
    OutResult.ArgumentsJson = utf8_span_to_string(Native.arguments_json);
    return true;
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
