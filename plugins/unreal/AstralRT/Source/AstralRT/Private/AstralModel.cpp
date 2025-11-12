#include "AstralModel.h"
#include "AstralLog.h"
#include "IAstralRT.h"

#include "Containers/UnrealString.h"
#include "HAL/Platform.h"
#include "Misc/Paths.h"

#include "astral_rt.h"

namespace {

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

static FString resolve_model_path(const FAstralModelDesc& Desc)
{
    if (Desc.ModelPath.IsEmpty() || FPaths::IsRelative(Desc.ModelPath) == false)
    {
        return Desc.ModelPath;
    }

    const FString Root = path_root_dir(Desc.PathRoot);
    if (Root.IsEmpty())
    {
        return FPaths::ConvertRelativePathToFull(Desc.ModelPath);
    }

    return FPaths::ConvertRelativePathToFull(FPaths::Combine(Root, Desc.ModelPath));
}

} // namespace

void UAstralModel::BeginDestroy()
{
    Release();
    Super::BeginDestroy();
}

bool UAstralModel::Load(const FAstralModelDesc& Desc)
{
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
    return true;
}

bool UAstralModel::GetEmbeddingDim(int32& OutDim) const
{
    OutDim = 0;
    if (ModelHandle == 0)
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

bool UAstralModel::GetCaps(int64& OutCaps) const
{
    OutCaps = 0;
    if (ModelHandle == 0)
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
    if (ModelHandle == 0)
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

bool UAstralModel::InitMedia(const FAstralModelMediaDesc& Desc)
{
    if (ModelHandle == 0)
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

    FTCHARToUTF8 MediaPathUtf8(*Desc.MediaPath);
    if (Desc.SourceKind == EAstralModelSourceKind::Path)
    {
        if (Desc.MediaPath.IsEmpty())
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
    if (ModelHandle == 0)
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

    if (IAstralRT::IsAvailable() && IAstralRT::Get().IsInitialized())
    {
        astral_model_release(static_cast<AstralHandle>(ModelHandle));
    }

    ModelHandle = 0;
}
