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

struct X86Features {
  bool avx2;
  bool f16c;
};

X86Features x86_features_from_registers(uint32_t leaf1_ecx, uint32_t leaf7_ebx, uint64_t xcr0);
bool x86_supports_avx2_f16c(const X86Features& features);

struct CpuFeatures {
  CpuArch arch;
  bool x86_avx2;
  bool x86_f16c;
  bool arm_neon;
};

const CpuFeatures& cpu_features();
const char* cpu_arch_name(CpuArch arch);
const char* cpu_dispatch_tier_name();

inline bool cpu_supports_avx2() {
  return cpu_features().x86_avx2;
}

inline bool cpu_supports_f16c() {
  return cpu_features().x86_f16c;
}

inline bool cpu_supports_neon() {
  return cpu_features().arm_neon;
}

} // namespace astral::platform
