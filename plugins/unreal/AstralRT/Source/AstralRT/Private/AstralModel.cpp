#include "AstralModel.h"
#include "AstralLog.h"
#include "IAstralRT.h"

#include "Containers/Array.h"
#include "Containers/StringConv.h"
#include "Containers/UnrealString.h"
#include "HAL/Platform.h"
#include "Misc/Paths.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

#include "astral_rt.h"

namespace {

static constexpr int32 kInlineResolvedPathBytes = 512;
static constexpr int32 kInlineDetokenizedBytes = 512;
static constexpr int32 kNoUtf8Chars = 0;
static constexpr uint32 kNoUtf8Bytes = 0;

static FString path_root_dir(EAstralUnrealPathRoot Root)
{
    switch (Root)
    {
    case EAstralUnrealPathRoot::ProjectContent:
        return FPaths::ProjectContentDir();
    case EAstralUnrealPathRoot::ProjectSaved:
        return FPaths::ProjectSavedDir();
    case EAstralUnrealPathRoot::ProjectPersistentDownload:
        return FPaths::ProjectPersistentDownloadDir();
    case EAstralUnrealPathRoot::Raw:
    default:
        return FString();
    }
}

static AstralModelPathRoot native_path_root(EAstralUnrealPathRoot Root)
{
    switch (Root)
    {
    case EAstralUnrealPathRoot::ProjectContent:
        return ASTRAL_MODEL_PATH_ROOT_CONTENT;
    case EAstralUnrealPathRoot::ProjectSaved:
        return ASTRAL_MODEL_PATH_ROOT_SAVED;
    case EAstralUnrealPathRoot::ProjectPersistentDownload:
        return ASTRAL_MODEL_PATH_ROOT_DOWNLOAD;
    case EAstralUnrealPathRoot::Raw:
    default:
        return ASTRAL_MODEL_PATH_ROOT_RAW;
    }
}

static AstralSpanU8 utf8_span(const FTCHARToUTF8& Utf8)
{
    AstralSpanU8 Span{};
    if (Utf8.Length() > kNoUtf8Chars)
    {
        Span.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
        Span.len = static_cast<uint32_t>(Utf8.Length());
    }
    return Span;
}

static FString resolve_unreal_path(const FString& Path, EAstralUnrealPathRoot RootKind)
{
    if (Path.IsEmpty())
    {
        return Path;
    }

    const FString ContentRoot = path_root_dir(EAstralUnrealPathRoot::ProjectContent);
    const FString SavedRoot = path_root_dir(EAstralUnrealPathRoot::ProjectSaved);
    const FString DownloadRoot = path_root_dir(EAstralUnrealPathRoot::ProjectPersistentDownload);

    FTCHARToUTF8 PathUtf8(*Path);
    FTCHARToUTF8 ContentRootUtf8(*ContentRoot);
    FTCHARToUTF8 SavedRootUtf8(*SavedRoot);
    FTCHARToUTF8 DownloadRootUtf8(*DownloadRoot);

    AstralModelPathResolveDesc Desc{};
    Desc.size = sizeof(AstralModelPathResolveDesc);
    Desc.root = native_path_root(RootKind);
    Desc.path = utf8_span(PathUtf8);
    Desc.content_root = utf8_span(ContentRootUtf8);
    Desc.saved_root = utf8_span(SavedRootUtf8);
    Desc.cache_root = utf8_span(SavedRootUtf8);
    Desc.download_root = utf8_span(DownloadRootUtf8);

    uint32 RequiredBytes = kNoUtf8Bytes;
    AstralMutSpanU8 EmptyOut{};
    AstralErr Err = astral_model_path_resolve(&Desc, EmptyOut, &RequiredBytes);
    if (Err != ASTRAL_E_NOMEM && Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_model_path_resolve failed (%d)"), static_cast<int32>(Err));
        return FString();
    }
    if (RequiredBytes > static_cast<uint32>(TNumericLimits<int32>::Max()))
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: resolved path is too large"));
        return FString();
    }

    TArray<uint8, TInlineAllocator<kInlineResolvedPathBytes>> Bytes;
    Bytes.SetNumUninitialized(static_cast<int32>(RequiredBytes));

    AstralMutSpanU8 Out{};
    Out.data = Bytes.GetData();
    Out.len = RequiredBytes;
    Err = astral_model_path_resolve(&Desc, Out, &RequiredBytes);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_model_path_resolve copy failed (%d)"), static_cast<int32>(Err));
        return FString();
    }

    FUTF8ToTCHAR Converted(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), static_cast<int32>(RequiredBytes));
    return FString(Converted.Length(), Converted.Get());
}

static FString resolve_model_path(const FAstralModelDesc& Desc)
{
    return resolve_unreal_path(Desc.ModelPath, Desc.PathRoot);
}

static FString resolve_media_path(const FAstralModelMediaDesc& Desc)
{
    return resolve_unreal_path(Desc.MediaPath, Desc.MediaPathRoot);
}

static FString resolve_adapter_path(const FAstralAdapterDesc& Desc)
{
    return resolve_unreal_path(Desc.AdapterPath, Desc.PathRoot);
}

static FAstralOperationResult make_operation_result(AstralErr Err, int64 Handle, int32 Count = 0)
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

} // namespace

void UAstralModel::BeginDestroy()
{
    Release();
    Super::BeginDestroy();
}

bool UAstralModel::IsCurrentRuntimeGeneration() const
{
    return IAstralRT::IsAvailable() &&
        IAstralRT::Get().IsInitialized() &&
        IAstralRT::Get().GetRuntimeGeneration() == RuntimeGeneration;
}

bool UAstralModel::IsValid() const
{
    return ModelHandle != 0 && IsCurrentRuntimeGeneration();
}

bool UAstralModel::Load(const FAstralModelDesc& Desc)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralModel_Load);

    Release();

    if (!IAstralRT::IsAvailable() || !IAstralRT::Get().IsInitialized())
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: runtime not initialized"));
        return false;
    }

    const FString ResolvedModelPath = resolve_model_path(Desc);
    FTCHARToUTF8 PathUtf8(*ResolvedModelPath);
    FTCHARToUTF8 BackendUtf8(*Desc.BackendName);

    AstralModelDesc Native{};
    Native.size = sizeof(AstralModelDesc);
    Native.source_kind = static_cast<AstralModelSourceKind>(Desc.SourceKind);

    if (Desc.SourceKind == EAstralModelSourceKind::Path)
    {
        if (!ResolvedModelPath.IsEmpty())
        {
            Native.model_path.data = reinterpret_cast<const uint8_t*>(PathUtf8.Get());
            Native.model_path.len = static_cast<uint32_t>(PathUtf8.Length());
        }
    }
    else if (Desc.SourceKind == EAstralModelSourceKind::Memory)
    {
        if (Desc.ModelBytes.Num() == 0)
        {
            UE_LOG(LogAstralRT, Error, TEXT("AstralRT: memory model source has no bytes"));
            return false;
        }
        Native.model_bytes.data = Desc.ModelBytes.GetData();
        Native.model_bytes.len = static_cast<uint32_t>(Desc.ModelBytes.Num());
    }
    else
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: model IO source is not exposed by the Unreal wrapper"));
        return false;
    }

    if (!Desc.BackendName.IsEmpty())
    {
        Native.backend_name.data = reinterpret_cast<const uint8_t*>(BackendUtf8.Get());
        Native.backend_name.len = static_cast<uint32_t>(BackendUtf8.Length());
    }

    Native.gpu_layers = static_cast<uint32_t>(Desc.GpuLayers);
    Native.n_ctx = static_cast<uint32_t>(Desc.ContextSize);
    Native.n_batch = static_cast<uint32_t>(Desc.BatchSize);
    Native.n_threads = static_cast<uint32_t>(Desc.NumThreads);
    Native.embeddings_only = Desc.bEmbeddingsOnly ? 1 : 0;

    AstralHandle Out = 0;
    const AstralErr Err = astral_model_load(&Native, &Out);
    if (Err != ASTRAL_OK)
    {
        const char* Last = astral_last_error();
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_model_load failed (%d): %s"),
               static_cast<int32>(Err),
               Last ? UTF8_TO_TCHAR(Last) : TEXT("<no error>"));
        return false;
    }

    ModelHandle = static_cast<uint64>(Out);
    RuntimeGeneration = IAstralRT::Get().GetRuntimeGeneration();
    return true;
}

bool UAstralModel::GetEmbeddingDim(int32& OutDim) const
{
    OutDim = 0;
    if (!IsValid())
    {
        return false;
    }

    uint32_t Dim = 0;
    const AstralErr Err = astral_model_embedding_dim(static_cast<AstralHandle>(ModelHandle), &Dim);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutDim = static_cast<int32>(Dim);
    return true;
}

bool UAstralModel::LoadAdapter(const FAstralAdapterDesc& Desc, int64& OutAdapterHandle) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralModel_LoadAdapter);

    OutAdapterHandle = 0;
    if (!IsValid())
    {
        return false;
    }

    const FString ResolvedAdapterPath = resolve_adapter_path(Desc);
    FTCHARToUTF8 PathUtf8(*ResolvedAdapterPath);

    AstralAdapterDesc Native{};
    Native.size = sizeof(AstralAdapterDesc);
    Native.path.data = reinterpret_cast<const uint8_t*>(PathUtf8.Get());
    Native.path.len = static_cast<uint32_t>(PathUtf8.Length());

    AstralHandle Out = 0;
    const AstralErr Err = astral_model_adapter_load(static_cast<AstralHandle>(ModelHandle), &Native, &Out);
    if (Err != ASTRAL_OK)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_model_adapter_load failed (%d)"), static_cast<int32>(Err));
        return false;
    }

    OutAdapterHandle = static_cast<int64>(Out);
    return true;
}

void UAstralModel::ReleaseAdapter(int64 AdapterHandle) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralModel_ReleaseAdapter);

    if (AdapterHandle == 0)
    {
        return;
    }

    astral_model_adapter_release(static_cast<AstralHandle>(AdapterHandle));
}

bool UAstralModel::GetCaps(int64& OutCaps) const
{
    OutCaps = 0;
    if (!IsValid())
    {
        return false;
    }

    AstralCaps Caps = 0;
    const AstralErr Err = astral_model_caps(static_cast<AstralHandle>(ModelHandle), &Caps);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutCaps = static_cast<int64>(Caps);
    return true;
}

bool UAstralModel::GetLimits(FAstralModelLimits& OutLimits) const
{
    OutLimits = FAstralModelLimits{};
    if (!IsValid())
    {
        return false;
    }

    AstralModelLimits Native{};
    const AstralErr Err = astral_model_limits(static_cast<AstralHandle>(ModelHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutLimits.VocabSize = static_cast<int32>(Native.vocab_size);
    OutLimits.ContextSize = static_cast<int32>(Native.ctx_size);
    OutLimits.MaxBatch = static_cast<int32>(Native.max_batch);
    OutLimits.MaxSlots = static_cast<int32>(Native.max_slots);
    return true;
}

bool UAstralModel::CountTokens(const FString& Text, bool bAddSpecial, bool bParseSpecial, int32& OutCount) const
{
    return CountTokensResult(Text, bAddSpecial, bParseSpecial, OutCount).bSuccess;
}

FAstralOperationResult UAstralModel::CountTokensResult(
    const FString& Text,
    bool bAddSpecial,
    bool bParseSpecial,
    int32& OutCount
) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralModel_CountTokens);

    OutCount = 0;
    if (!IsValid())
    {
        return make_operation_result(ASTRAL_E_INVALID, static_cast<int64>(ModelHandle));
    }

    FTCHARToUTF8 TextUtf8(*Text);
    AstralSpanU8 Span{};
    Span.data = reinterpret_cast<const uint8_t*>(TextUtf8.Get());
    Span.len = static_cast<uint32_t>(TextUtf8.Length());

    uint32_t Count = 0;
    const AstralErr Err = astral_tokenize_count(
        static_cast<AstralHandle>(ModelHandle),
        Span,
        bAddSpecial ? 1 : 0,
        bParseSpecial ? 1 : 0,
        &Count);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err, static_cast<int64>(ModelHandle));
    }

    OutCount = static_cast<int32>(Count);
    return make_operation_result(ASTRAL_OK, static_cast<int64>(ModelHandle), OutCount);
}

bool UAstralModel::Tokenize(const FString& Text, bool bAddSpecial, bool bParseSpecial, TArray<int32>& OutTokens) const
{
    return TokenizeResult(Text, bAddSpecial, bParseSpecial, OutTokens).bSuccess;
}

FAstralOperationResult UAstralModel::TokenizeResult(
    const FString& Text,
    bool bAddSpecial,
    bool bParseSpecial,
    TArray<int32>& OutTokens
) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralModel_Tokenize);

    OutTokens.Reset();
    int32 Count = 0;
    const FAstralOperationResult CountResult = CountTokensResult(Text, bAddSpecial, bParseSpecial, Count);
    if (!CountResult.bSuccess)
    {
        return CountResult;
    }
    if (Count == 0)
    {
        return make_operation_result(ASTRAL_OK, static_cast<int64>(ModelHandle));
    }

    OutTokens.SetNumUninitialized(Count);
    FTCHARToUTF8 TextUtf8(*Text);
    AstralSpanU8 Span{};
    Span.data = reinterpret_cast<const uint8_t*>(TextUtf8.Get());
    Span.len = static_cast<uint32_t>(TextUtf8.Length());

    uint32_t Written = 0;
    const AstralErr Err = astral_tokenize(
        static_cast<AstralHandle>(ModelHandle),
        Span,
        OutTokens.GetData(),
        static_cast<uint32_t>(OutTokens.Num()),
        bAddSpecial ? 1 : 0,
        bParseSpecial ? 1 : 0,
        &Written);
    if (Err != ASTRAL_OK)
    {
        OutTokens.Reset();
        return make_operation_result(Err, static_cast<int64>(ModelHandle));
    }

    OutTokens.SetNum(static_cast<int32>(Written), EAllowShrinking::No);
    return make_operation_result(ASTRAL_OK, static_cast<int64>(ModelHandle), static_cast<int32>(Written));
}

bool UAstralModel::Detokenize(const TArray<int32>& Tokens, FString& OutText) const
{
    return DetokenizeResult(Tokens, OutText).bSuccess;
}

FAstralOperationResult UAstralModel::DetokenizeResult(const TArray<int32>& Tokens, FString& OutText) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralModel_Detokenize);

    OutText.Reset();
    if (!IsValid())
    {
        return make_operation_result(ASTRAL_E_INVALID, static_cast<int64>(ModelHandle));
    }

    uint32_t ByteCount = 0;
    const AstralErr CountErr = astral_detokenize_count(
        static_cast<AstralHandle>(ModelHandle),
        Tokens.GetData(),
        static_cast<uint32_t>(Tokens.Num()),
        &ByteCount);
    if (CountErr != ASTRAL_OK)
    {
        return make_operation_result(CountErr, static_cast<int64>(ModelHandle));
    }
    if (ByteCount == 0)
    {
        return make_operation_result(ASTRAL_OK, static_cast<int64>(ModelHandle));
    }

    TArray<uint8, TInlineAllocator<kInlineDetokenizedBytes>> Bytes;
    Bytes.SetNumUninitialized(static_cast<int32>(ByteCount));
    AstralMutSpanU8 Out{};
    Out.data = Bytes.GetData();
    Out.len = ByteCount;

    uint32_t Written = 0;
    const AstralErr Err = astral_detokenize(
        static_cast<AstralHandle>(ModelHandle),
        Tokens.GetData(),
        static_cast<uint32_t>(Tokens.Num()),
        Out,
        &Written);
    if (Err != ASTRAL_OK)
    {
        return make_operation_result(Err, static_cast<int64>(ModelHandle));
    }

    FUTF8ToTCHAR Converted(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), static_cast<int32>(Written));
    OutText = FString(Converted.Length(), Converted.Get());
    return make_operation_result(ASTRAL_OK, static_cast<int64>(ModelHandle), static_cast<int32>(Written));
}

bool UAstralModel::InitMedia(const FAstralModelMediaDesc& Desc)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralModel_InitMedia);

    if (!IsValid())
    {
        return false;
    }

    AstralModelMediaDesc Native{};
    Native.size = sizeof(AstralModelMediaDesc);
    Native.source_kind = static_cast<AstralModelSourceKind>(Desc.SourceKind);
    Native.flags = static_cast<uint32_t>(Desc.Flags);
    Native.image_min_tokens = static_cast<uint32_t>(Desc.ImageMinTokens);
    Native.image_max_tokens = static_cast<uint32_t>(Desc.ImageMaxTokens);
    Native.gpu_device = Desc.GpuDevice;
    Native.gpu_route_flags = static_cast<uint32_t>(Desc.GpuRouteFlags);
    Native.gpu_device_mask = static_cast<uint64_t>(Desc.GpuDeviceMask);
    Native.gpu_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(Desc.GpuStream));

    const FString ResolvedMediaPath = resolve_media_path(Desc);
    FTCHARToUTF8 MediaPathUtf8(*ResolvedMediaPath);
    if (Desc.SourceKind == EAstralModelSourceKind::Path)
    {
        if (ResolvedMediaPath.IsEmpty())
        {
            return false;
        }
        Native.media_path.data = reinterpret_cast<const uint8_t*>(MediaPathUtf8.Get());
        Native.media_path.len = static_cast<uint32_t>(MediaPathUtf8.Length());
    }
    else if (Desc.SourceKind == EAstralModelSourceKind::Memory)
    {
        if (Desc.MediaBytes.Num() == 0)
        {
            return false;
        }
        Native.media_bytes.data = Desc.MediaBytes.GetData();
        Native.media_bytes.len = static_cast<uint32_t>(Desc.MediaBytes.Num());
    }
    else
    {
        return false;
    }

    const AstralErr Err = astral_model_media_init(static_cast<AstralHandle>(ModelHandle), &Native);
    return Err == ASTRAL_OK;
}

bool UAstralModel::GetMediaInfo(FAstralMediaInfo& OutInfo) const
{
    OutInfo = FAstralMediaInfo{};
    if (!IsValid())
    {
        return false;
    }

    AstralMediaInfo Native{};
    Native.size = sizeof(AstralMediaInfo);
    const AstralErr Err = astral_model_media_info(static_cast<AstralHandle>(ModelHandle), &Native);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutInfo.SupportsImage = static_cast<int32>(Native.supports_image);
    OutInfo.SupportsAudio = static_cast<int32>(Native.supports_audio);
    OutInfo.AudioSampleRate = static_cast<int32>(Native.audio_sample_rate);
    OutInfo.ImageMinTokens = static_cast<int32>(Native.image_min_tokens);
    OutInfo.ImageMaxTokens = static_cast<int32>(Native.image_max_tokens);
    return true;
}

void UAstralModel::Release()
{
    if (ModelHandle == 0)
    {
        return;
    }

    if (IsCurrentRuntimeGeneration())
    {
        astral_model_release(static_cast<AstralHandle>(ModelHandle));
    }

    ModelHandle = 0;
    RuntimeGeneration = 0;
}
