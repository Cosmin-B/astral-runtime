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
  --check-runner      Check GPU/toolkit runner policy, then exit
  --help              Show this help

Environment:
  ASTRAL_TEST_CUDA_PARITY_INFER=1     Run the CPU-vs-CUDA inference parity section (requires tests/models/gpt2.Q2_K.gguf)
  ASTRAL_TEST_CUDA_E2E=1              Run the CUDA side of test_cuda_e2e
  ASTRAL_TEST_CUDA_PARITY_STRICT=1    (same as --strict)
  ASTRAL_CUDA_NVIDIA_SMI              Override nvidia-smi path for runner checks
  ASTRAL_CUDA_NVCC                    Override nvcc path for runner checks

Example:
  ASTRAL_TEST_CUDA_PARITY_INFER=1 scripts/run_cuda_parity.sh --preset dev-cuda
EOF
}

preset="dev-cuda"
strict=0
arch_override=""
allow_probes=0
check_env=0
check_runner=0
cuda_nvcc_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --arch) arch_override="${2:-}"; shift 2 ;;
    --strict) strict=1; shift ;;
    --allow-probes) allow_probes=1; shift ;;
    --check-env) check_env=1; shift ;;
    --check-runner) check_runner=1; shift ;;
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

select_cuda_toolkit() {
  if [[ -n "${ASTRAL_CUDA_NVCC:-}" ]]; then
    cuda_nvcc_path="${ASTRAL_CUDA_NVCC}"
  elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    cuda_nvcc_path="/usr/local/cuda/bin/nvcc"
  else
    cuda_nvcc_path="$(command -v nvcc || true)"
  fi

  if [[ -z "${cuda_nvcc_path}" || ! -x "${cuda_nvcc_path}" ]]; then
    echo "CUDA runner missing nvcc; set ASTRAL_CUDA_NVCC or install the CUDA toolkit." >&2
    return 1
  fi

  local cuda_bin
  cuda_bin="$(cd "$(dirname "${cuda_nvcc_path}")" && pwd)"
  export PATH="${cuda_bin}:${PATH}"
  if [[ "${cuda_bin}" == */bin ]]; then
    export CUDAToolkit_ROOT="${cuda_bin%/bin}"
    export CUDA_PATH="${CUDAToolkit_ROOT}"
  fi
}

require_cuda_runner() {
  local nvidia_smi="${ASTRAL_CUDA_NVIDIA_SMI:-nvidia-smi}"
  if ! command -v "${nvidia_smi}" >/dev/null 2>&1; then
    echo "CUDA runner missing nvidia-smi; run real CUDA release lanes on a GPU host." >&2
    return 1
  fi

  select_cuda_toolkit || return 1

  echo "[cuda_parity] GPU:"
  "${nvidia_smi}" --query-gpu=name,driver_version --format=csv,noheader
  echo "[cuda_parity] nvcc:"
  "${cuda_nvcc_path}" --version | sed -n "1,4p"
}

if [[ "${check_runner}" -eq 1 ]]; then
  if [[ "${allow_probes}" -eq 1 ]]; then
    echo "[cuda_parity] probe-only runner policy OK"
    exit 0
  fi
  require_cuda_runner
  exit $?
fi

if [[ "${allow_probes}" -ne 1 ]]; then
  require_cuda_runner
else
  # Prefer the system toolkit for probe builds when it is present.
  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export CUDAToolkit_ROOT="/usr/local/cuda"
    export CUDA_PATH="/usr/local/cuda"
  fi
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
