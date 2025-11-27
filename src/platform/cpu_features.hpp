#pragma once

#include <cstdint>

namespace astral::platform {

enum class CpuArch : uint8_t {
    Unknown = 0,
    X86_64,
    X86_32,
    Arm64,
    Arm32,
    RiscV64,
};

struct CpuFeatures {
    CpuArch arch;
    bool x86_avx2;
    bool arm_neon;
};

const CpuFeatures& cpu_features();
const char* cpu_arch_name(CpuArch arch);
const char* cpu_dispatch_tier_name();

inline bool cpu_supports_avx2() {
    return cpu_features().x86_avx2;
}

inline bool cpu_supports_neon() {
    return cpu_features().arm_neon;
}

} // namespace astral::platform
