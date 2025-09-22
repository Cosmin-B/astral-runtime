#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_release_required_gates.sh [options]

Runs the hard release-candidate gates that must not be treated as best-effort:
native release tests, CUDA parity/e2e matrix, and real MTMD media validation.

Options:
  --cuda-arch <list>    Override ASTRAL_CUDA_ARCHITECTURES for CUDA presets
  --cuda-strict         Enable strict CUDA token-id parity assertions
  --mtmd-bench         Require MTMD feature bench media feed rows
  --help               Show help

MTMD fixtures are supplied through:
  ASTRAL_TEST_VISION_MODEL, ASTRAL_TEST_VISION_MEDIA
  ASTRAL_TEST_AUDIO_MODEL, ASTRAL_TEST_AUDIO_MEDIA
EOF
}

cuda_arch=""
cuda_strict=0
mtmd_bench=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-arch) cuda_arch="${2:-}"; shift 2 ;;
    --cuda-strict) cuda_strict=1; shift ;;
    --mtmd-bench) mtmd_bench=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

echo "[release-gate] Native release tests"
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests -j --output-on-failure

cuda_args=(--preset-set release)
if [[ -n "${cuda_arch}" ]]; then
  cuda_args+=(--arch "${cuda_arch}")
fi
if [[ "${cuda_strict}" -eq 1 ]]; then
  cuda_args+=(--strict)
fi

echo "[release-gate] CUDA release matrix"
ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 \
  scripts/run_cuda_parity_matrix.sh "${cuda_args[@]}"

mtmd_args=()
if [[ "${mtmd_bench}" -eq 1 ]]; then
  mtmd_args+=(--bench)
fi

echo "[release-gate] MTMD real media validation"
scripts/run_multimodal_validation.sh "${mtmd_args[@]}"

echo "[release-gate] required gates passed"
