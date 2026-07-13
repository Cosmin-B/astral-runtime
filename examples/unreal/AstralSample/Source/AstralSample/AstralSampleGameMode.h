#pragma once

#include "GameFramework/GameModeBase.h"

#include "AstralSampleGameMode.generated.h"

UCLASS()
class ASTRALSAMPLE_API AAstralSampleGameMode : public AGameModeBase {
  GENERATED_BODY()

protected:
  virtual void BeginPlay() override;
};
