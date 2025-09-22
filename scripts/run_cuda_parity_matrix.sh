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
  --help            Show help

Environment:
  ASTRAL_TEST_CUDA_PARITY_INFER=1  Enables inference parity section
  ASTRAL_TEST_CUDA_E2E=1           Enables CUDA side of test_cuda_e2e

Example:
  ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 \\
    scripts/run_cuda_parity_matrix.sh --arch 120a-real --strict

Release-candidate gate:
  ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 \\
    scripts/run_cuda_parity_matrix.sh --preset-set release --strict
EOF
}

arch_override=""
preset_set="dev"
strict=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) arch_override="${2:-}"; shift 2 ;;
    --preset-set) preset_set="${2:-}"; shift 2 ;;
    --strict) strict=1; shift ;;
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

for p in "${presets[@]}"; do
  echo
  echo "=== CUDA parity preset: ${p} ==="
  scripts/run_cuda_parity.sh --preset "${p}" "${common_args[@]}"
done
