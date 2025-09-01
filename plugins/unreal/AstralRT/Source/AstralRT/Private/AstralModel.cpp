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
