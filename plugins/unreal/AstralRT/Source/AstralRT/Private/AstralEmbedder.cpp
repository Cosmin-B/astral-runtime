#include "AstralEmbedder.h"
#include "AstralLog.h"
#include "AstralModel.h"
#include "IAstralRT.h"

#include "Containers/UnrealString.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

#include "astral_rt.h"

bool UAstralEmbedder::IsCurrentRuntimeGeneration() const
{
    return IAstralRT::IsAvailable() &&
        IAstralRT::Get().IsInitialized() &&
        IAstralRT::Get().GetRuntimeGeneration() == RuntimeGeneration;
}

bool UAstralEmbedder::IsValid() const
{
    return EmbedderHandle != 0 && IsCurrentRuntimeGeneration();
}

int32 UAstralEmbedder::GetDim() const
{
    return IsValid() ? EmbeddingDim : 0;
}

void UAstralEmbedder::BeginDestroy()
{
    Destroy();
    Super::BeginDestroy();
}

bool UAstralEmbedder::Create(UAstralModel* Model)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_Create);

    Destroy();

    if (Model == nullptr || !Model->IsValid())
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: invalid model for embedder"));
        return false;
    }

    if (!IAstralRT::IsAvailable() || !IAstralRT::Get().IsInitialized())
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: runtime not initialized"));
        return false;
    }

    int32 Dim = 0;
    if (!Model->GetEmbeddingDim(Dim) || Dim <= 0)
    {
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: model does not report a valid embedding dim"));
        return false;
    }

    AstralHandle Out = 0;
    const AstralErr Err = astral_embed_create(static_cast<AstralHandle>(Model->GetHandle()), &Out);
    if (Err != ASTRAL_OK || Out == 0)
    {
        const char* Last = astral_last_error();
        UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_embed_create failed (%d): %s"),
               static_cast<int32>(Err),
               Last ? UTF8_TO_TCHAR(Last) : TEXT("<no error>"));
        return false;
    }

    EmbedderHandle = static_cast<uint64>(Out);
    ModelHandle = Model->GetHandle();
    RuntimeGeneration = IAstralRT::Get().GetRuntimeGeneration();
    EmbeddingDim = Dim;
    return true;
}

void UAstralEmbedder::Destroy()
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_Destroy);

    if (EmbedderHandle == 0)
    {
        ModelHandle = 0;
        RuntimeGeneration = 0;
        EmbeddingDim = 0;
        return;
    }

    if (IsCurrentRuntimeGeneration())
    {
        astral_embed_destroy(static_cast<AstralHandle>(EmbedderHandle));
    }

    EmbedderHandle = 0;
    ModelHandle = 0;
    RuntimeGeneration = 0;
    EmbeddingDim = 0;
}

bool UAstralEmbedder::EnqueueUtf8Bytes(const TArray<uint8>& Utf8Bytes, int64& OutTicket)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_EnqueueUtf8Bytes);

    OutTicket = 0;
    if (!IsValid())
    {
        return false;
    }

    AstralSpanU8 Text{};
    Text.data = Utf8Bytes.GetData();
    Text.len = static_cast<uint32_t>(Utf8Bytes.Num());

    uint64_t Ticket = 0;
    const AstralErr Err = astral_embed_enqueue(static_cast<AstralHandle>(EmbedderHandle), Text, &Ticket);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutTicket = static_cast<int64>(Ticket);
    return true;
}

bool UAstralEmbedder::EnqueueImage(const FAstralImageDesc& Image, int64& OutTicket)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_EnqueueImage);

    OutTicket = 0;
    if (!IsValid())
    {
        return false;
    }
    if (Image.Pixels.Num() == 0)
    {
        return false;
    }

    AstralImageDesc Native{};
    Native.size = sizeof(AstralImageDesc);
    Native.format = static_cast<AstralImageFormat>(Image.Format);
    Native.width = static_cast<uint32_t>(Image.Width);
    Native.height = static_cast<uint32_t>(Image.Height);
    Native.row_stride = static_cast<uint32_t>(Image.RowStride);
    Native.flags = static_cast<uint32_t>(Image.Flags);
    Native.pixels.data = Image.Pixels.GetData();
    Native.pixels.len = static_cast<uint32_t>(Image.Pixels.Num());
    Native.gpu_device = Image.GpuDevice;
    Native.gpu_route_flags = static_cast<uint32_t>(Image.GpuRouteFlags);
    Native.gpu_device_mask = static_cast<uint64_t>(Image.GpuDeviceMask);
    Native.gpu_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(Image.GpuStream));

    uint64_t Ticket = 0;
    const AstralErr Err = astral_embed_enqueue_image(static_cast<AstralHandle>(EmbedderHandle), &Native, &Ticket);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutTicket = static_cast<int64>(Ticket);
    return true;
}

bool UAstralEmbedder::EnqueueAudio(const FAstralAudioDesc& Audio, int64& OutTicket)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_EnqueueAudio);

    OutTicket = 0;
    if (!IsValid())
    {
        return false;
    }
    if (Audio.Samples.Num() == 0)
    {
        return false;
    }

    AstralAudioDesc Native{};
    Native.size = sizeof(AstralAudioDesc);
    Native.format = static_cast<AstralAudioFormat>(Audio.Format);
    if (Audio.Channels == 0)
    {
        return false;
    }

    uint64 FrameCount = static_cast<uint64>(Audio.FrameCount);
    if (FrameCount == 0)
    {
        const uint32 BytesPerSample = (Audio.Format == EAstralAudioFormat::F32) ? 4u : 2u;
        const uint64 TotalSamples = static_cast<uint64>(Audio.Samples.Num()) / BytesPerSample;
        if (TotalSamples % Audio.Channels != 0)
        {
            return false;
        }
        FrameCount = TotalSamples / Audio.Channels;
    }

    Native.channels = static_cast<uint32_t>(Audio.Channels);
    Native.sample_rate = static_cast<uint32_t>(Audio.SampleRate);
    Native.frame_count = FrameCount;
    Native.samples.data = Audio.Samples.GetData();
    Native.samples.len = static_cast<uint32_t>(Audio.Samples.Num());
    Native.flags = static_cast<uint32_t>(Audio.Flags);
    Native.gpu_device = Audio.GpuDevice;
    Native.gpu_route_flags = static_cast<uint32_t>(Audio.GpuRouteFlags);
    Native.gpu_device_mask = static_cast<uint64_t>(Audio.GpuDeviceMask);
    Native.gpu_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(Audio.GpuStream));

    uint64_t Ticket = 0;
    const AstralErr Err = astral_embed_enqueue_audio(static_cast<AstralHandle>(EmbedderHandle), &Native, &Ticket);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutTicket = static_cast<int64>(Ticket);
    return true;
}

bool UAstralEmbedder::EnqueueMultimodal(const FString& Text,
                                        const FAstralImageDesc& Image,
                                        const FAstralAudioDesc& Audio,
                                        bool bUseImage,
                                        bool bUseAudio,
                                        int64& OutTicket)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_EnqueueMultimodal);

    OutTicket = 0;
    if (!IsValid())
    {
        return false;
    }

    FTCHARToUTF8 Utf8(*Text);
    AstralSpanU8 TextSpan{};
    TextSpan.data = reinterpret_cast<const uint8_t*>(Utf8.Get());
    TextSpan.len = static_cast<uint32_t>(Utf8.Length());

    const AstralImageDesc* ImagePtr = nullptr;
    AstralImageDesc ImageNative{};
    if (bUseImage)
    {
        if (Image.Pixels.Num() == 0)
        {
            return false;
        }
        ImageNative.size = sizeof(AstralImageDesc);
        ImageNative.format = static_cast<AstralImageFormat>(Image.Format);
        ImageNative.width = static_cast<uint32_t>(Image.Width);
        ImageNative.height = static_cast<uint32_t>(Image.Height);
        ImageNative.row_stride = static_cast<uint32_t>(Image.RowStride);
        ImageNative.flags = static_cast<uint32_t>(Image.Flags);
        ImageNative.pixels.data = Image.Pixels.GetData();
        ImageNative.pixels.len = static_cast<uint32_t>(Image.Pixels.Num());
        ImageNative.gpu_device = Image.GpuDevice;
        ImageNative.gpu_route_flags = static_cast<uint32_t>(Image.GpuRouteFlags);
        ImageNative.gpu_device_mask = static_cast<uint64_t>(Image.GpuDeviceMask);
        ImageNative.gpu_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(Image.GpuStream));
        ImagePtr = &ImageNative;
    }

    const AstralAudioDesc* AudioPtr = nullptr;
    AstralAudioDesc AudioNative{};
    if (bUseAudio)
    {
        if (Audio.Samples.Num() == 0)
        {
            return false;
        }
        AudioNative.size = sizeof(AstralAudioDesc);
        AudioNative.format = static_cast<AstralAudioFormat>(Audio.Format);
        if (Audio.Channels == 0)
        {
            return false;
        }
        uint64 FrameCount = static_cast<uint64>(Audio.FrameCount);
        if (FrameCount == 0)
        {
            const uint32 BytesPerSample = (Audio.Format == EAstralAudioFormat::F32) ? 4u : 2u;
            const uint64 TotalSamples = static_cast<uint64>(Audio.Samples.Num()) / BytesPerSample;
            if (TotalSamples % Audio.Channels != 0)
            {
                return false;
            }
            FrameCount = TotalSamples / Audio.Channels;
        }
        AudioNative.channels = static_cast<uint32_t>(Audio.Channels);
        AudioNative.sample_rate = static_cast<uint32_t>(Audio.SampleRate);
        AudioNative.frame_count = FrameCount;
        AudioNative.samples.data = Audio.Samples.GetData();
        AudioNative.samples.len = static_cast<uint32_t>(Audio.Samples.Num());
        AudioNative.flags = static_cast<uint32_t>(Audio.Flags);
        AudioNative.gpu_device = Audio.GpuDevice;
        AudioNative.gpu_route_flags = static_cast<uint32_t>(Audio.GpuRouteFlags);
        AudioNative.gpu_device_mask = static_cast<uint64_t>(Audio.GpuDeviceMask);
        AudioNative.gpu_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(Audio.GpuStream));
        AudioPtr = &AudioNative;
    }

    uint64_t Ticket = 0;
    const AstralErr Err = astral_embed_enqueue_multimodal(
        static_cast<AstralHandle>(EmbedderHandle), TextSpan, ImagePtr, AudioPtr, &Ticket);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutTicket = static_cast<int64>(Ticket);
    return true;
}

bool UAstralEmbedder::Collect(int64 Ticket, TArray<float>& OutVector)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_Collect);

    if (!IsValid() || Ticket <= 0 || EmbeddingDim <= 0)
    {
        return false;
    }

    OutVector.SetNumUninitialized(EmbeddingDim);

    AstralMutSpanU8 Out{};
    Out.data = reinterpret_cast<uint8_t*>(OutVector.GetData());
    Out.len = static_cast<uint32_t>(OutVector.Num() * sizeof(float));

    const AstralErr Err =
        astral_embed_collect(static_cast<AstralHandle>(EmbedderHandle), static_cast<uint64_t>(Ticket), Out);
    return Err == ASTRAL_OK;
}

bool UAstralEmbedder::Cancel(int64 Ticket)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_Cancel);

    if (!IsValid() || Ticket <= 0)
    {
        return false;
    }

    const AstralErr Err =
        astral_embed_cancel(static_cast<AstralHandle>(EmbedderHandle), static_cast<uint64_t>(Ticket));
    return Err == ASTRAL_OK;
}

bool UAstralEmbedder::EmbedUtf8Bytes(const TArray<uint8>& Utf8Bytes, TArray<float>& OutVector)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_EmbedUtf8Bytes);

    int64 Ticket = 0;
    if (!EnqueueUtf8Bytes(Utf8Bytes, Ticket))
    {
        return false;
    }
    return Collect(Ticket, OutVector);
}

bool UAstralEmbedder::EmbedText(const FString& Text, TArray<float>& OutVector)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AstralRT_Embedder_EmbedText);

    FTCHARToUTF8 Utf8(*Text);
    TArray<uint8> Bytes;
    Bytes.Append(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
    return EmbedUtf8Bytes(Bytes, OutVector);
}
