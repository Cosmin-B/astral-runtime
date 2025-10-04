#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_cuda_parity.sh [options]

Builds a CUDA-enabled preset and runs the CUDA parity test.

Options:
  --preset <name>     CMake preset to use (default: dev-cuda)
  --arch <list>       Override ASTRAL_CUDA_ARCHITECTURES (e.g. "120a-real" or "native")
  --strict            Enable strict token-id parity assertions
  --allow-probes      Allow build/probe-only runs when real CUDA env flags are unset
  --check-env         Check required env policy, then exit
  --help              Show this help

Environment:
  ASTRAL_TEST_CUDA_PARITY_INFER=1     Run the CPU-vs-CUDA inference parity section (requires tests/models/gpt2.Q2_K.gguf)
  ASTRAL_TEST_CUDA_E2E=1              Run the CUDA side of test_cuda_e2e
  ASTRAL_TEST_CUDA_PARITY_STRICT=1    (same as --strict)

Example:
  ASTRAL_TEST_CUDA_PARITY_INFER=1 scripts/run_cuda_parity.sh --preset dev-cuda
EOF
}

preset="dev-cuda"
strict=0
arch_override=""
allow_probes=0
check_env=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --arch) arch_override="${2:-}"; shift 2 ;;
    --strict) strict=1; shift ;;
    --allow-probes) allow_probes=1; shift ;;
    --check-env) check_env=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${strict}" -eq 1 ]]; then
  export ASTRAL_TEST_CUDA_PARITY_STRICT=1
fi

if [[ "${allow_probes}" -ne 1 ]]; then
  if [[ "${ASTRAL_TEST_CUDA_PARITY_INFER:-}" != "1" ]]; then
    echo "Missing ASTRAL_TEST_CUDA_PARITY_INFER=1; pass --allow-probes only for local build discovery." >&2
    exit 2
  fi
  if [[ "${ASTRAL_TEST_CUDA_E2E:-}" != "1" ]]; then
    echo "Missing ASTRAL_TEST_CUDA_E2E=1; pass --allow-probes only for local build discovery." >&2
    exit 2
  fi
fi

if [[ "${check_env}" -eq 1 ]]; then
  if [[ "${allow_probes}" -eq 1 ]]; then
    echo "[cuda_parity] probe-only env policy OK"
  else
    echo "[cuda_parity] real CUDA env policy OK"
  fi
  exit 0
fi

# Prefer the system-installed toolkit under /usr/local/cuda when present (common on CUDA boxes).
if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDAToolkit_ROOT="/usr/local/cuda"
  export CUDA_PATH="/usr/local/cuda"
fi

echo "[cuda_parity] Configure: ${preset}"
if [[ -n "${arch_override}" ]]; then
  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    cmake --preset "${preset}" -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" -DASTRAL_CUDA_ARCHITECTURES="${arch_override}"
  else
    cmake --preset "${preset}" -DASTRAL_CUDA_ARCHITECTURES="${arch_override}"
  fi
else
  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    cmake --preset "${preset}" -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
  else
    cmake --preset "${preset}"
  fi
fi

echo "[cuda_parity] Build: ${preset}"
cmake --build --preset "${preset}" -j

echo "[cuda_parity] Test: test_cuda_parity + test_cuda_e2e"
ctest --preset "${preset}" -R 'test_cuda_(parity|e2e)' -V
