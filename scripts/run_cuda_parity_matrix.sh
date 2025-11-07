#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_cuda_parity_matrix.sh [options]

Build + run CUDA parity + CUDA e2e across kernel modes:
  - default (mmq auto)
  - force cuBLAS
  - force MMQ

Options:
  --arch <list>     Override ASTRAL_CUDA_ARCHITECTURES (e.g. "120a-real" or "native")
  --preset-set <s>  Preset group: dev or release (default: dev)
  --strict          Enable strict token-id parity assertions
  --allow-probes    Allow build/probe-only runs when real CUDA env flags are unset
  --check-env       Check required env policy, then exit
  --check-runner    Check GPU/toolkit runner policy, then exit
  --print-plan      Print selected presets and required flags, then exit
  --help            Show help

Environment:
  ASTRAL_TEST_CUDA_PARITY_INFER=1  Enables inference parity section
  ASTRAL_TEST_CUDA_E2E=1           Enables CUDA side of test_cuda_e2e

Example:
  ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 \\
    scripts/run_cuda_parity_matrix.sh --arch 120a-real --strict

Release-candidate gate:
  ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 \\
    scripts/run_cuda_parity_matrix.sh --preset-set release --arch native --strict
EOF
}

arch_override=""
preset_set="dev"
strict=0
allow_probes=0
check_env=0
check_runner=0
print_plan=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) arch_override="${2:-}"; shift 2 ;;
    --preset-set) preset_set="${2:-}"; shift 2 ;;
    --strict) strict=1; shift ;;
    --allow-probes) allow_probes=1; shift ;;
    --check-env) check_env=1; shift ;;
    --check-runner) check_runner=1; shift ;;
    --print-plan) print_plan=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

common_args=()
if [[ -n "${arch_override}" ]]; then
  common_args+=(--arch "${arch_override}")
fi
if [[ "${strict}" -eq 1 ]]; then
  common_args+=(--strict)
fi
if [[ "${allow_probes}" -eq 1 ]]; then
  common_args+=(--allow-probes)
fi
if [[ "${check_env}" -eq 1 ]]; then
  scripts/run_cuda_parity.sh --preset dev-cuda --check-env "${common_args[@]}"
  exit $?
fi
if [[ "${check_runner}" -eq 1 ]]; then
  scripts/run_cuda_parity.sh --preset dev-cuda --check-runner "${common_args[@]}"
  exit $?
fi

case "${preset_set}" in
  dev)
    presets=(dev-cuda dev-cuda-cublas dev-cuda-mmq)
    ;;
  release)
    presets=(release-with-tests-cuda release-with-tests-cuda-cublas release-with-tests-cuda-mmq)
    ;;
  *)
    echo "Unknown preset set: ${preset_set}" >&2
    usage
    exit 2
    ;;
esac

if [[ "${preset_set}" == "release" && "${allow_probes}" -ne 1 && -z "${arch_override}" ]]; then
  echo "Missing --arch <deployed-arch-list> for release CUDA matrix evidence." >&2
  exit 2
fi

if [[ "${print_plan}" -eq 1 ]]; then
  echo "[cuda_matrix] preset set: ${preset_set}"
  echo "[cuda_matrix] arch: ${arch_override:-preset-default}"
  echo "[cuda_matrix] strict: ${strict}"
  echo "[cuda_matrix] require env: ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1"
  for p in "${presets[@]}"; do
    echo "[cuda_matrix] preset: ${p}"
  done
  exit 0
fi

for p in "${presets[@]}"; do
  echo
  echo "=== CUDA parity preset: ${p} ==="
  scripts/run_cuda_parity.sh --preset "${p}" "${common_args[@]}"
done
