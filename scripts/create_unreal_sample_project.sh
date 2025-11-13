#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir=""
plugin_mode="symlink"
plugin_dir="${root_dir}/plugins/unreal/AstralRT"

usage() {
  cat <<'USAGE'
Usage: ./scripts/create_unreal_sample_project.sh --out PATH [--plugin-mode symlink|copy|none]

Creates a sidecar Unreal project named AstralSample. Generated project files are
written to PATH and are not intended to be committed to the Astral repo.

The sample C++ actor demonstrates:
  - model load through UAstralModel
  - streaming generation through UAstralSession
  - cancellation through UAstralSession::Cancel
  - embeddings through UAstralEmbedder
  - expected failure logging through astral_last_error
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --out" >&2
        exit 2
      fi
      out_dir="$2"
      shift 2
      ;;
    --plugin-mode)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --plugin-mode" >&2
        exit 2
      fi
      plugin_mode="$2"
      shift 2
      ;;
    --plugin-dir)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --plugin-dir" >&2
        exit 2
      fi
      plugin_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${out_dir}" ]]; then
  echo "--out is required" >&2
  usage >&2
  exit 2
fi

case "${plugin_mode}" in
  symlink|copy|none)
    ;;
  *)
    echo "--plugin-mode must be symlink, copy, or none" >&2
    exit 2
    ;;
esac

out_parent="$(dirname "${out_dir}")"
mkdir -p "${out_parent}"
project_dir="$(cd "${out_parent}" && pwd)/$(basename "${out_dir}")"
mkdir -p \
  "${project_dir}/Config" \
  "${project_dir}/Plugins" \
  "${project_dir}/Source/AstralSample"

if [[ "${plugin_mode}" == "symlink" ]]; then
  rm -rf "${project_dir}/Plugins/AstralRT"
  ln -s "${plugin_dir}" "${project_dir}/Plugins/AstralRT"
elif [[ "${plugin_mode}" == "copy" ]]; then
  rm -rf "${project_dir}/Plugins/AstralRT"
  cp -a "${plugin_dir}" "${project_dir}/Plugins/AstralRT"
fi

cat > "${project_dir}/AstralSample.uproject" <<'EOF'
{
  "FileVersion": 3,
  "EngineAssociation": "5.7",
  "Category": "Samples",
  "Description": "AstralRT Unreal sample project",
  "Modules": [
    {
      "Name": "AstralSample",
      "Type": "Runtime",
      "LoadingPhase": "Default"
    }
  ],
  "Plugins": [
    {
      "Name": "AstralRT",
      "Enabled": true
    }
  ]
}
EOF

cat > "${project_dir}/Config/DefaultEngine.ini" <<'EOF'
[/Script/EngineSettings.GameMapsSettings]
EditorStartupMap=/Game/AstralSample
GameDefaultMap=/Game/AstralSample
GlobalDefaultGameMode=/Script/AstralSample.AstralSampleGameMode
EOF

cat > "${project_dir}/Source/AstralSample.Target.cs" <<'EOF'
using UnrealBuildTool;
using System.Collections.Generic;

public class AstralSampleTarget : TargetRules
{
    public AstralSampleTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        ExtraModuleNames.Add("AstralSample");
    }
}
EOF

cat > "${project_dir}/Source/AstralSampleEditor.Target.cs" <<'EOF'
using UnrealBuildTool;
using System.Collections.Generic;

public class AstralSampleEditorTarget : TargetRules
{
    public AstralSampleEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        ExtraModuleNames.Add("AstralSample");
    }
}
EOF

cat > "${project_dir}/Source/AstralSample/AstralSample.Build.cs" <<'EOF'
using UnrealBuildTool;

public class AstralSample : ModuleRules
{
    public AstralSample(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "AstralRT"
        });
    }
}
EOF

cat > "${project_dir}/Source/AstralSample/AstralSample.cpp" <<'EOF'
#include "Modules/ModuleManager.h"

IMPLEMENT_PRIMARY_GAME_MODULE(FDefaultGameModuleImpl, AstralSample, "AstralSample");
EOF

cat > "${project_dir}/Source/AstralSample/AstralSampleGameMode.h" <<'EOF'
#pragma once

#include "GameFramework/GameModeBase.h"

#include "AstralSampleGameMode.generated.h"

UCLASS()
class ASTRALSAMPLE_API AAstralSampleGameMode : public AGameModeBase
{
    GENERATED_BODY()
};
EOF

cat > "${project_dir}/Source/AstralSample/AstralSampleGameMode.cpp" <<'EOF'
#include "AstralSampleGameMode.h"
EOF

cat > "${project_dir}/Source/AstralSample/AstralSampleActor.h" <<'EOF'
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

#include "AstralSampleActor.generated.h"

class UAstralEmbedder;
class UAstralModel;
class UAstralSession;

UCLASS()
class ASTRALSAMPLE_API AAstralSampleActor : public AActor
{
    GENERATED_BODY()

public:
    AAstralSampleActor();

    UPROPERTY(EditAnywhere, Category = "Astral")
    FString BackendName = TEXT("mock");

    UPROPERTY(EditAnywhere, Category = "Astral")
    FString ModelPath;

    UPROPERTY(EditAnywhere, Category = "Astral")
    FString Prompt = TEXT("Say hello from Astral.");

    UFUNCTION(BlueprintCallable, Category = "Astral")
    void RunGenerationDemo();

    UFUNCTION(BlueprintCallable, Category = "Astral")
    void CancelStreamingDemo();

    UFUNCTION(BlueprintCallable, Category = "Astral")
    void RunEmbeddingDemo();

    UFUNCTION(BlueprintCallable, Category = "Astral")
    void RunErrorDemo();

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    UPROPERTY()
    TObjectPtr<UAstralModel> GenerationModel;

    UPROPERTY()
    TObjectPtr<UAstralSession> Session;

    UPROPERTY()
    TObjectPtr<UAstralModel> EmbeddingModel;

    UPROPERTY()
    TObjectPtr<UAstralEmbedder> Embedder;

    bool LoadGenerationModel();
    bool CreateSession();
    void OnStreamBytes(TConstArrayView<uint8> Bytes);
};
EOF

cat > "${project_dir}/Source/AstralSample/AstralSampleActor.cpp" <<'EOF'
#include "AstralSampleActor.h"

#include "AstralEmbedder.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "AstralTypes.h"
#include "astral_rt.h"

#include "Containers/ArrayView.h"
#include "Containers/StringConv.h"

DEFINE_LOG_CATEGORY_STATIC(LogAstralSample, Log, All);

AAstralSampleActor::AAstralSampleActor()
{
    PrimaryActorTick.bCanEverTick = false;
}

void AAstralSampleActor::BeginPlay()
{
    Super::BeginPlay();

    RunGenerationDemo();
    RunEmbeddingDemo();
    RunErrorDemo();
}

void AAstralSampleActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (Session != nullptr)
    {
        Session->Cancel();
        Session = nullptr;
    }
    if (GenerationModel != nullptr)
    {
        GenerationModel->Release();
        GenerationModel = nullptr;
    }
    if (Embedder != nullptr)
    {
        Embedder->Destroy();
        Embedder = nullptr;
    }
    if (EmbeddingModel != nullptr)
    {
        EmbeddingModel->Release();
        EmbeddingModel = nullptr;
    }

    Super::EndPlay(EndPlayReason);
}

bool AAstralSampleActor::LoadGenerationModel()
{
    GenerationModel = NewObject<UAstralModel>(this);

    FAstralModelDesc Desc{};
    Desc.SourceKind = EAstralModelSourceKind::Path;
    Desc.ModelPath = ModelPath;
    Desc.BackendName = BackendName;
    Desc.ContextSize = 256;
    Desc.BatchSize = 64;

    if (!GenerationModel->Load(Desc))
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: model load failed: %s"), UTF8_TO_TCHAR(astral_last_error()));
        return false;
    }

    return true;
}

bool AAstralSampleActor::CreateSession()
{
    Session = NewObject<UAstralSession>(this);

    FAstralSessionDesc Desc{};
    Desc.MaxTokens = 64;
    Desc.Temperature = 0.0f;
    Desc.TopK = 0;
    Desc.TopP = 1.0f;
    Desc.bStreamEnabled = true;
    Desc.Seed = 7;

    if (!Session->Create(GenerationModel, Desc))
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: session create failed: %s"), UTF8_TO_TCHAR(astral_last_error()));
        return false;
    }

    Session->OnStreamBytesNative().AddUObject(this, &AAstralSampleActor::OnStreamBytes);
    return true;
}

void AAstralSampleActor::RunGenerationDemo()
{
    if (!LoadGenerationModel() || !CreateSession())
    {
        return;
    }

    if (!Session->FeedPrompt(Prompt, true))
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: prompt feed failed: %s"), UTF8_TO_TCHAR(astral_last_error()));
        return;
    }
    if (!Session->Decode())
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: decode failed: %s"), UTF8_TO_TCHAR(astral_last_error()));
    }
}

void AAstralSampleActor::CancelStreamingDemo()
{
    if (Session == nullptr || !Session->IsValid())
    {
        UE_LOG(LogAstralSample, Warning, TEXT("Astral sample: no active session to cancel"));
        return;
    }

    if (!Session->Cancel())
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: cancel failed: %s"), UTF8_TO_TCHAR(astral_last_error()));
        return;
    }

    const int32 WaitResult = Session->Wait(1000);
    UE_LOG(LogAstralSample, Log, TEXT("Astral sample: canceled stream wait result %d"), WaitResult);
}

void AAstralSampleActor::RunEmbeddingDemo()
{
    EmbeddingModel = NewObject<UAstralModel>(this);

    FAstralModelDesc Desc{};
    Desc.SourceKind = EAstralModelSourceKind::Path;
    Desc.ModelPath = ModelPath;
    Desc.BackendName = BackendName;
    Desc.ContextSize = 128;
    Desc.bEmbeddingsOnly = true;

    if (!EmbeddingModel->Load(Desc))
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: embedding model load failed: %s"), UTF8_TO_TCHAR(astral_last_error()));
        return;
    }

    Embedder = NewObject<UAstralEmbedder>(this);
    if (!Embedder->Create(EmbeddingModel))
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: embedder create failed: %s"), UTF8_TO_TCHAR(astral_last_error()));
        return;
    }

    FTCHARToUTF8 Utf8(TEXT("sample vector"));
    TArray<uint8> Bytes;
    Bytes.Append(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());

    TArray<float> Vector;
    if (!Embedder->EmbedUtf8Bytes(Bytes, Vector))
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: embedding failed: %s"), UTF8_TO_TCHAR(astral_last_error()));
        return;
    }

    UE_LOG(LogAstralSample, Log, TEXT("Astral sample: embedding dimension %d"), Vector.Num());
}

void AAstralSampleActor::RunErrorDemo()
{
    UAstralModel* BadModel = NewObject<UAstralModel>(this);

    FAstralModelDesc BadDesc{};
    BadDesc.SourceKind = EAstralModelSourceKind::Memory;
    BadDesc.BackendName = BackendName;

    if (BadModel->Load(BadDesc))
    {
        UE_LOG(LogAstralSample, Error, TEXT("Astral sample: expected memory-source load to fail"));
        BadModel->Release();
        return;
    }

    UE_LOG(LogAstralSample, Warning, TEXT("Astral sample: expected load failure: %s"), UTF8_TO_TCHAR(astral_last_error()));
}

void AAstralSampleActor::OnStreamBytes(TConstArrayView<uint8> Bytes)
{
    FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
    UE_LOG(LogAstralSample, Log, TEXT("Astral sample stream: %.*s"), Text.Length(), Text.Get());
}
EOF

cat > "${project_dir}/README.md" <<'EOF'
# AstralSample

Generated sidecar Unreal sample project for AstralRT.

Build the native plugin package from the Astral repo first:

```bash
cmake --preset unreal-plugin
cmake --build --preset unreal-plugin -j
```

Open `AstralSample.uproject` in UE 5.7, place `AAstralSampleActor` in a map,
and run the map. The actor demonstrates model load, streaming generation,
cancellation, embeddings, and expected error logging through `LogAstralSample`.

Real production sign-off still requires packaging this project on the UE 5.7
release runner and recording the Automation/package logs as release evidence.
EOF

echo "Created Unreal sample project: ${project_dir}"
