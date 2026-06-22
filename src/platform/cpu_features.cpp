#include "cpu_features.hpp"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
  #elif defined(_MSC_VER)
    #include <intrin.h>
  #endif
#endif

namespace astral::platform {
namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

bool cpuid_leaf(uint32_t leaf, uint32_t subleaf, uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
#if defined(__GNUC__) || defined(__clang__)
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t d = 0;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    *eax = a;
    *ebx = b;
    *ecx = c;
    *edx = d;
    return true;
#elif defined(_MSC_VER)
    int regs[4] = {};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    *eax = static_cast<uint32_t>(regs[0]);
    *ebx = static_cast<uint32_t>(regs[1]);
    *ecx = static_cast<uint32_t>(regs[2]);
    *edx = static_cast<uint32_t>(regs[3]);
    return true;
#else
    (void)leaf;
    (void)subleaf;
    *eax = 0;
    *ebx = 0;
    *ecx = 0;
    *edx = 0;
    return false;
#endif
}

uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax = 0;
    uint32_t edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | static_cast<uint64_t>(eax);
#else
    return 0;
#endif
}

bool detect_x86_avx2() {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;

    if (!cpuid_leaf(0, 0, &eax, &ebx, &ecx, &edx) || eax < 7) {
        return false;
    }

    if (!cpuid_leaf(1, 0, &eax, &ebx, &ecx, &edx)) {
        return false;
    }

    constexpr uint32_t kXsave = 1u << 26;
    constexpr uint32_t kOsxsave = 1u << 27;
    constexpr uint32_t kAvx = 1u << 28;
    if ((ecx & (kXsave | kOsxsave | kAvx)) != (kXsave | kOsxsave | kAvx)) {
        return false;
    }

    constexpr uint64_t kXmmYmmState = (1ull << 1) | (1ull << 2);
    if ((xgetbv0() & kXmmYmmState) != kXmmYmmState) {
        return false;
    }

    if (!cpuid_leaf(7, 0, &eax, &ebx, &ecx, &edx)) {
        return false;
    }

    constexpr uint32_t kAvx2 = 1u << 5;
    return (ebx & kAvx2) != 0;
}

#endif

CpuFeatures detect_cpu_features() {
    CpuFeatures features{};
    features.arch = CpuArch::Unknown;
    features.x86_avx2 = false;
    features.arm_neon = false;

#if defined(__x86_64__) || defined(_M_X64)
    features.arch = CpuArch::X86_64;
    features.x86_avx2 = detect_x86_avx2();
#elif defined(__i386__) || defined(_M_IX86)
    features.arch = CpuArch::X86_32;
    features.x86_avx2 = detect_x86_avx2();
#elif defined(__aarch64__) || defined(_M_ARM64)
    features.arch = CpuArch::Arm64;
    features.arm_neon = true;
#elif defined(__arm__) || defined(_M_ARM)
    features.arch = CpuArch::Arm32;
  #if defined(__ARM_NEON) || defined(__ARM_NEON__)
    features.arm_neon = true;
  #endif
#elif defined(__riscv) && (__riscv_xlen == 64)
    features.arch = CpuArch::RiscV64;
#endif

    return features;
}

} // namespace

const CpuFeatures& cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}

const char* cpu_arch_name(CpuArch arch) {
    switch (arch) {
        case CpuArch::X86_64:
            return "x86_64";
        case CpuArch::X86_32:
            return "x86";
        case CpuArch::Arm64:
            return "arm64";
        case CpuArch::Arm32:
            return "arm";
        case CpuArch::RiscV64:
            return "riscv64";
        case CpuArch::Unknown:
        default:
            return "unknown";
    }
}

const char* cpu_dispatch_tier_name() {
    const CpuFeatures& features = cpu_features();
    if (features.x86_avx2) {
        return "x86_64-avx2";
    }
    if (features.arm_neon) {
        return "arm-neon";
    }
    return "scalar";
}

} // namespace astral::platform
