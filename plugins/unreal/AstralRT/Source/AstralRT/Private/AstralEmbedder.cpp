#include "AstralEmbedder.h"
#include "AstralModel.h"
#include "IAstralRT.h"

#include "Containers/UnrealString.h"

#include "astral_rt.h"

void UAstralEmbedder::BeginDestroy()
{
    Destroy();
    Super::BeginDestroy();
}

bool UAstralEmbedder::Create(UAstralModel* Model)
{
    Destroy();

    if (Model == nullptr || !Model->IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: invalid model for embedder"));
        return false;
    }

    if (!IAstralRT::IsAvailable() || !IAstralRT::Get().IsInitialized())
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: runtime not initialized"));
        return false;
    }

    int32 Dim = 0;
    if (!Model->GetEmbeddingDim(Dim) || Dim <= 0)
    {
        UE_LOG(LogTemp, Error, TEXT("AstralRT: model does not report a valid embedding dim"));
        return false;
    }

    AstralHandle Out = 0;
    const AstralErr Err = astral_embed_create(static_cast<AstralHandle>(Model->GetHandle()), &Out);
    if (Err != ASTRAL_OK || Out == 0)
    {
        const char* Last = astral_last_error();
        UE_LOG(LogTemp, Error, TEXT("AstralRT: astral_embed_create failed (%d): %s"),
               static_cast<int32>(Err),
               Last ? UTF8_TO_TCHAR(Last) : TEXT("<no error>"));
        return false;
    }

    EmbedderHandle = static_cast<uint64>(Out);
    ModelHandle = Model->GetHandle();
    EmbeddingDim = Dim;
    return true;
}

void UAstralEmbedder::Destroy()
{
    if (EmbedderHandle == 0)
    {
        ModelHandle = 0;
        EmbeddingDim = 0;
        return;
    }

    if (IAstralRT::IsAvailable() && IAstralRT::Get().IsInitialized())
    {
        astral_embed_destroy(static_cast<AstralHandle>(EmbedderHandle));
    }

    EmbedderHandle = 0;
    ModelHandle = 0;
    EmbeddingDim = 0;
}

bool UAstralEmbedder::EnqueueUtf8Bytes(const TArray<uint8>& Utf8Bytes, int64& OutTicket)
{
    OutTicket = 0;
    if (EmbedderHandle == 0)
    {
        return false;
    }

    AstralSpanU8 Text{};
    Text.data = Utf8Bytes.GetData();
    Text.len = static_cast<uint32_t>(Utf8Bytes.Num());

    uint64 Ticket = 0;
    const AstralErr Err = astral_embed_enqueue(static_cast<AstralHandle>(EmbedderHandle), Text, &Ticket);
    if (Err != ASTRAL_OK)
    {
        return false;
    }

    OutTicket = static_cast<int64>(Ticket);
    return true;
}

bool UAstralEmbedder::Collect(int64 Ticket, TArray<float>& OutVector)
{
    if (EmbedderHandle == 0 || Ticket <= 0 || EmbeddingDim <= 0)
    {
        return false;
    }

    OutVector.SetNumUninitialized(EmbeddingDim);

    AstralMutSpanU8 Out{};
    Out.data = reinterpret_cast<uint8_t*>(OutVector.GetData());
    Out.len = static_cast<uint32_t>(OutVector.Num() * sizeof(float));

    const AstralErr Err =
        astral_embed_collect(static_cast<AstralHandle>(EmbedderHandle), static_cast<uint64>(Ticket), Out);
    return Err == ASTRAL_OK;
}

bool UAstralEmbedder::EmbedUtf8Bytes(const TArray<uint8>& Utf8Bytes, TArray<float>& OutVector)
{
    int64 Ticket = 0;
    if (!EnqueueUtf8Bytes(Utf8Bytes, Ticket))
    {
        return false;
    }
    return Collect(Ticket, OutVector);
}

bool UAstralEmbedder::EmbedText(const FString& Text, TArray<float>& OutVector)
{
    FTCHARToUTF8 Utf8(*Text);
    TArray<uint8> Bytes;
    Bytes.Append(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
    return EmbedUtf8Bytes(Bytes, OutVector);
}

