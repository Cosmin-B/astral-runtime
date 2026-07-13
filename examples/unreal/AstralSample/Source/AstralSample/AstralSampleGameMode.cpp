#include "AstralSampleGameMode.h"

#include "AstralSampleActor.h"
#include "Engine/World.h"

void AAstralSampleGameMode::BeginPlay() {
  Super::BeginPlay();

  UWorld* World = GetWorld();
  if (World != nullptr) {
    World->SpawnActor<AAstralSampleActor>();
  }
}
