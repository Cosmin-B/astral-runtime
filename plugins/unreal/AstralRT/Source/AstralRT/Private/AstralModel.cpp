#include "AstralModel.h"
#include "IAstralRT.h"

#include "Containers/UnrealString.h"
#include "HAL/Platform.h"

#include "astral_rt.h"

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
        UE_LOG(LogTemp, Error, TEXT("AstralRT: runtime not initialized"));
        return false;
    }

    FTCHARToUTF8 PathUtf8(*Desc.ModelPath);
    FTCHARToUTF8 BackendUtf8(*Desc.BackendName);

    AstralModelDesc Native{};
    Native.size = sizeof(AstralModelDesc);
    Native.source_kind = ASTRAL_MODEL_SOURCE_PATH;

    if (!Desc.ModelPath.IsEmpty())
    {
        Native.model_path.data = reinterpret_cast<const uint8_t*>(PathUtf8.Get());
        Native.model_path.len = static_cast<uint32_t>(PathUtf8.Length());
    }

    if (!Desc.BackendName.IsEmpty())
    {
        Native.backend_name.data = reinterpret_cast<const uint8_t*>(BackendUtf8.Get());
        Native.backend_name.len = static_cast<uint32_t>(BackendUtf8.Length());
    }

    Native.gpu_layers = Desc.GpuLayers;
    Native.n_ctx = Desc.ContextSize;
    Native.n_batch = Desc.BatchSize;
    Native.n_threads = Desc.NumThreads;
    Native.embeddings_only = Desc.bEmbeddingsOnly ? 1 : 0;

    AstralHandle Out = 0;
    const AstralErr Err = astral_model_load(&Native, &Out);
    if (Err != ASTRAL_OK)
    {
        const char* Last = astral_last_error();
        UE_LOG(LogTemp, Error, TEXT("AstralRT: astral_model_load failed (%d): %s"),
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

bool UAstralModel::GetCaps(uint64& OutCaps) const
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

    OutCaps = static_cast<uint64>(Caps);
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

    OutLimits.VocabSize = Native.vocab_size;
    OutLimits.ContextSize = Native.ctx_size;
    OutLimits.MaxBatch = Native.max_batch;
    OutLimits.MaxSlots = Native.max_slots;
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
    Native.flags = Desc.Flags;
    Native.image_min_tokens = Desc.ImageMinTokens;
    Native.image_max_tokens = Desc.ImageMaxTokens;
    Native.gpu_device = Desc.GpuDevice;
    Native.gpu_route_flags = Desc.GpuRouteFlags;
    Native.gpu_device_mask = Desc.GpuDeviceMask;
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

    OutInfo.SupportsImage = Native.supports_image;
    OutInfo.SupportsAudio = Native.supports_audio;
    OutInfo.AudioSampleRate = Native.audio_sample_rate;
    OutInfo.ImageMinTokens = Native.image_min_tokens;
    OutInfo.ImageMaxTokens = Native.image_max_tokens;
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
