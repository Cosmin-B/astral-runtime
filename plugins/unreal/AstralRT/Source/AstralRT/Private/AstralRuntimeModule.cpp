#include "IAstralRT.h"

#include "Modules/ModuleManager.h"
#include "HAL/UnrealMemory.h"
#include "Containers/StringConv.h"

#include "astral_rt.h"

namespace {

DEFINE_LOG_CATEGORY_STATIC(LogAstralRT, Log, All);

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

static void UELog(void* user, int level, AstralSpanU8 msg) {
    (void)user;
    if (!msg.data || msg.len == 0)
    {
        return;
    }

    const ANSICHAR* utf8 = reinterpret_cast<const ANSICHAR*>(msg.data);
    const int32 utf8_len = static_cast<int32>(msg.len);
    const FUTF8ToTCHAR text(utf8, utf8_len);

    const TCHAR* t = text.Get();
    const int32 t_len = text.Length();

    switch (level)
    {
        case 0: // ASTRAL_LOG_ERROR
            UE_LOG(LogAstralRT, Error, TEXT("[Astral] %.*s"), t_len, t);
            break;
        case 1: // ASTRAL_LOG_WARN
            UE_LOG(LogAstralRT, Warning, TEXT("[Astral] %.*s"), t_len, t);
            break;
        case 2: // ASTRAL_LOG_INFO
            UE_LOG(LogAstralRT, Log, TEXT("[Astral] %.*s"), t_len, t);
            break;
        case 3: // ASTRAL_LOG_DEBUG
        case 4: // ASTRAL_LOG_TRACE
        default:
            UE_LOG(LogAstralRT, Verbose, TEXT("[Astral] %.*s"), t_len, t);
            break;
    }
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
        InitCfg.log_cb = &UELog;
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
