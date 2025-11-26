#include "AstralBlueprintLibrary.h"

#include "AstralEmbedder.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "UObject/Package.h"

#include "astral_rt.h"

namespace {

static UObject* resolve_outer(UObject* Outer)
{
    return Outer != nullptr ? Outer : GetTransientPackage();
}

static bool has_cap(int64 Caps, AstralCaps Cap)
{
    const AstralCaps NativeCaps = static_cast<AstralCaps>(Caps);
    return (NativeCaps & Cap) != 0;
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
    default:
        return FString::Printf(TEXT("ASTRAL_E_%d"), ErrorCode);
    }
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
