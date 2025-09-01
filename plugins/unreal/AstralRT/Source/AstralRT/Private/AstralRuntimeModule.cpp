#include "IAstralRT.h"

#include "Modules/ModuleManager.h"
#include "HAL/UnrealMemory.h"

#include "astral_rt.h"

namespace {

static void* UEAlloc(void* user, size_t size, size_t align) {
    (void)user;
    const uint32 alignment = align > 0 ? static_cast<uint32>(align) : 0u;
    return FMemory::Malloc(size, alignment);
}

static void UEFree(void* user, void* ptr, size_t size, size_t align) {
    (void)user;
    (void)size;
    (void)align;
    FMemory::Free(ptr);
}

} // namespace

class FAstralRuntimeModule : public IAstralRT
{
public:
    virtual void StartupModule() override
    {
        AstralAllocator Allocator{};
        Allocator.alloc = &UEAlloc;
        Allocator.free = &UEFree;
        Allocator.user = nullptr;

        AstralInit InitCfg{};
        InitCfg.sys_alloc = Allocator;
        InitCfg.log_cb = nullptr;
        InitCfg.log_user = nullptr;
        InitCfg.reserve_bytes = 2ull << 30;
        InitCfg.thread_count = 0;
        InitCfg.numa_node = 0xFFFFFFFFu;
        InitCfg.enable_hugepages = 0;

        const AstralErr Err = astral_init(&InitCfg);
        if (Err != ASTRAL_OK)
        {
            bInitialized = false;
            UE_LOG(LogTemp, Error, TEXT("AstralRT: astral_init failed (%d)"), static_cast<int32>(Err));
            return;
        }

        bInitialized = true;
        UE_LOG(LogTemp, Log, TEXT("AstralRT: Initialized"));
    }

    virtual void ShutdownModule() override
    {
        if (bInitialized)
        {
            astral_shutdown();
            bInitialized = false;
            UE_LOG(LogTemp, Log, TEXT("AstralRT: Shutdown"));
        }
    }

    virtual bool IsInitialized() const override
    {
        return bInitialized;
    }

private:
    bool bInitialized = false;
};

IMPLEMENT_MODULE(FAstralRuntimeModule, AstralRT)

