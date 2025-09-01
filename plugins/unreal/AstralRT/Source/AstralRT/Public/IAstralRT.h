#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"
#include "Modules/ModuleManager.h"

/**
 * AstralRT module interface.
 * Manages runtime init/shutdown of the Astral native library.
 */
class ASTRALRT_API IAstralRT : public IModuleInterface
{
public:
    static inline IAstralRT& Get()
    {
        return FModuleManager::LoadModuleChecked<IAstralRT>("AstralRT");
    }

    static inline bool IsAvailable()
    {
        return FModuleManager::Get().IsModuleLoaded("AstralRT");
    }

    virtual bool IsInitialized() const = 0;
};

