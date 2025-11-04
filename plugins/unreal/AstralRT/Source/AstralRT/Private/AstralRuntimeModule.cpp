#include "IAstralRT.h"
#include "AstralLog.h"

#include "Modules/ModuleManager.h"
#include "HAL/UnrealMemory.h"
#include "Containers/StringConv.h"

#include "astral_rt.h"

#include <atomic>

DEFINE_LOG_CATEGORY(LogAstralRT);

namespace {

struct FAllocatorCounters
{
    std::atomic<uint64> AllocCalls{0};
    std::atomic<uint64> FreeCalls{0};
    std::atomic<uint64> AllocBytes{0};
    std::atomic<uint64> FreeBytes{0};

    void Reset()
    {
        AllocCalls.store(0, std::memory_order_relaxed);
        FreeCalls.store(0, std::memory_order_relaxed);
        AllocBytes.store(0, std::memory_order_relaxed);
        FreeBytes.store(0, std::memory_order_relaxed);
    }

    FAstralRTAllocatorStats Snapshot() const
    {
        FAstralRTAllocatorStats Stats{};
        Stats.AllocCalls = AllocCalls.load(std::memory_order_relaxed);
        Stats.FreeCalls = FreeCalls.load(std::memory_order_relaxed);
        Stats.AllocBytes = AllocBytes.load(std::memory_order_relaxed);
        Stats.FreeBytes = FreeBytes.load(std::memory_order_relaxed);
        return Stats;
    }
};

static FAllocatorCounters GAllocatorCounters;

static void* UEAlloc(void* user, size_t size, size_t align) {
    (void)user;
    const uint32 alignment = align > 0 ? static_cast<uint32>(align) : 0u;
    GAllocatorCounters.AllocCalls.fetch_add(1, std::memory_order_relaxed);
    GAllocatorCounters.AllocBytes.fetch_add(static_cast<uint64>(size), std::memory_order_relaxed);
    return FMemory::Malloc(size, alignment);
}

static void UEFree(void* user, void* ptr, size_t size, size_t align) {
    (void)user;
    (void)align;
    GAllocatorCounters.FreeCalls.fetch_add(1, std::memory_order_relaxed);
    GAllocatorCounters.FreeBytes.fetch_add(static_cast<uint64>(size), std::memory_order_relaxed);
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
            GAllocatorCounters.Reset();
            UE_LOG(LogAstralRT, Error, TEXT("AstralRT: astral_init failed (%d)"), static_cast<int32>(Err));
            return;
        }

        bInitialized = true;
        UE_LOG(LogAstralRT, Log, TEXT("AstralRT: Initialized"));
    }

    virtual void ShutdownModule() override
    {
        if (bInitialized)
        {
            astral_shutdown();
            bInitialized = false;
            GAllocatorCounters.Reset();
            UE_LOG(LogAstralRT, Log, TEXT("AstralRT: Shutdown"));
        }
    }

    virtual bool IsInitialized() const override
    {
        return bInitialized;
    }

    virtual void ResetAllocatorStats() override
    {
        GAllocatorCounters.Reset();
    }

    virtual FAstralRTAllocatorStats GetAllocatorStats() const override
    {
        return GAllocatorCounters.Snapshot();
    }

private:
    bool bInitialized = false;
};

IMPLEMENT_MODULE(FAstralRuntimeModule, AstralRT)
