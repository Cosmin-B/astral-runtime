#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"
#include "Modules/ModuleManager.h"

struct FAstralRTAllocatorStats
{
    uint64 AllocCalls = 0;
    uint64 FreeCalls = 0;
    uint64 AllocBytes = 0;
    uint64 FreeBytes = 0;
};

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
    virtual void ResetAllocatorStats() = 0;
    virtual FAstralRTAllocatorStats GetAllocatorStats() const = 0;
#if WITH_DEV_AUTOMATION_TESTS
    virtual void SimulateEnginePreExitForAutomation() = 0;
#endif
};
