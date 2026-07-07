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

X86Features x86_features_from_registers(uint32_t leaf1_ecx, uint32_t leaf7_ebx, uint64_t xcr0) {
  constexpr uint32_t kXsave = 1u << 26;
  constexpr uint32_t kOsxsave = 1u << 27;
  constexpr uint32_t kAvx = 1u << 28;
  constexpr uint32_t kF16c = 1u << 29;
  constexpr uint32_t kAvx2 = 1u << 5;
  constexpr uint64_t kXmmYmmState = (1ull << 1) | (1ull << 2);

  const bool avx_state = (leaf1_ecx & (kXsave | kOsxsave | kAvx)) == (kXsave | kOsxsave | kAvx) &&
                         (xcr0 & kXmmYmmState) == kXmmYmmState;
  return X86Features{avx_state && (leaf7_ebx & kAvx2) != 0, avx_state && (leaf1_ecx & kF16c) != 0};
}

bool x86_supports_avx2_f16c(const X86Features& features) {
  return features.avx2 && features.f16c;
}

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

X86Features detect_x86_features() {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  if (!cpuid_leaf(0, 0, &eax, &ebx, &ecx, &edx) || eax < 7) {
    return X86Features{false, false};
  }
  const uint32_t max_leaf = eax;

  if (!cpuid_leaf(1, 0, &eax, &ebx, &ecx, &edx)) {
    return X86Features{false, false};
  }
  const uint32_t leaf1_ecx = ecx;

  constexpr uint32_t kXsave = 1u << 26;
  constexpr uint32_t kOsxsave = 1u << 27;
  uint64_t xcr0 = 0;
  if ((leaf1_ecx & (kXsave | kOsxsave)) == (kXsave | kOsxsave)) {
    xcr0 = xgetbv0();
  }

  uint32_t leaf7_ebx = 0;
  if (max_leaf >= 7 && cpuid_leaf(7, 0, &eax, &ebx, &ecx, &edx)) {
    leaf7_ebx = ebx;
  }
  return x86_features_from_registers(leaf1_ecx, leaf7_ebx, xcr0);
}

#endif

CpuFeatures detect_cpu_features() {
  CpuFeatures features{};
  features.arch = CpuArch::Unknown;
  features.x86_avx2 = false;
  features.x86_f16c = false;
  features.arm_neon = false;

#if defined(__x86_64__) || defined(_M_X64)
  features.arch = CpuArch::X86_64;
  const X86Features x86 = detect_x86_features();
  features.x86_avx2 = x86.avx2;
  features.x86_f16c = x86.f16c;
#elif defined(__i386__) || defined(_M_IX86)
  features.arch = CpuArch::X86_32;
  const X86Features x86 = detect_x86_features();
  features.x86_avx2 = x86.avx2;
  features.x86_f16c = x86.f16c;
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
