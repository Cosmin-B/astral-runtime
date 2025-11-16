#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_release_required_gates.sh [options]

Runs the hard release-candidate gates that must not be treated as optional:
native release tests, ASAN/UBSAN and TSan validation, CUDA parity/e2e matrix,
real MTMD media validation, UE 5.7 full/slim container Automation, Unreal
Automation compatibility, UE 5.7 sample packaging, and Unity EditMode ABI
validation.

Options:
  --cuda-arch <list>    Required deployed CUDA architecture list for CUDA presets
  --cuda-strict         Enable strict CUDA token-id parity assertions
  --mtmd-bench         Require MTMD feature bench media feed rows
  --print-plan         Print required lanes and environment checks, then exit
  --skip-sanitizers    Skip ASAN/UBSAN and TSan lanes
  --skip-engine         Skip both Unreal and Unity engine gates
  --skip-unreal         Skip the Unreal container, compatibility, and sample gates
  --skip-unity          Skip the Unity EditMode ABI lane
  --help               Show help

MTMD fixtures are supplied through:
  ASTRAL_TEST_VISION_MODEL, ASTRAL_TEST_VISION_MEDIA
  ASTRAL_TEST_AUDIO_MODEL, ASTRAL_TEST_AUDIO_MEDIA

Engine gates are supplied through:
  Epic GHCR access or cached UE 5.7 full/slim container images
  UNREAL_54_EDITOR, UNREAL_55_EDITOR, UNREAL_56_EDITOR, UNREAL_57_EDITOR
  UNREAL_RUNUAT
  UNITY_EDITOR

Release-candidate runs should not use skip flags.
Release-candidate CUDA runs must name --cuda-arch so the evidence records the
deployed architecture target instead of inheriting preset defaults.
EOF
}

cuda_arch=""
cuda_strict=0
mtmd_bench=0
run_sanitizers=1
run_unreal=1
run_unity=1
print_plan=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-arch) cuda_arch="${2:-}"; shift 2 ;;
    --cuda-strict) cuda_strict=1; shift ;;
    --mtmd-bench) mtmd_bench=1; shift ;;
    --print-plan) print_plan=1; shift ;;
    --skip-sanitizers) run_sanitizers=0; shift ;;
    --skip-engine) run_unreal=0; run_unity=0; shift ;;
    --skip-unreal) run_unreal=0; shift ;;
    --skip-unity) run_unity=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

require_env_for_plan() {
  local missing=0
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "[release-gate] missing environment: ${name}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

print_release_plan() {
  local missing=0

  echo "[release-gate] plan"
  echo "  native release tests: cmake --preset release-with-tests && cmake --build --preset release-with-tests -j && ctest --preset release-with-tests -j --output-on-failure"
  if [[ "${run_sanitizers}" -eq 1 ]]; then
    echo "  sanitizers: scripts/run_asan.sh && scripts/run_tsan.sh"
  else
    echo "  sanitizers: skipped for local diagnosis"
  fi
  echo "  CUDA matrix: ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 scripts/run_cuda_parity_matrix.sh ${cuda_args[*]}"
  echo "  MTMD validation: scripts/run_multimodal_validation.sh ${mtmd_args[*]}"
  if [[ "${run_unreal}" -eq 1 ]]; then
    echo "  Unreal access: scripts/check_unreal_validation_access.sh --check-registry"
    echo "  Unreal native package: cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j"
    echo "  Unreal UE 5.7 full container: scripts/run_unreal_container_ci.sh --variant full --filter AstralRT --skip-native-build"
    echo "  Unreal UE 5.7 slim container: scripts/run_unreal_container_ci.sh --variant slim --filter AstralRT --skip-native-build"
    echo "  Unreal matrix: scripts/run_unreal_compatibility_matrix.sh"
    echo "  Unreal sample package: scripts/run_unreal_sample_package.sh --platform Linux"
  else
    echo "  Unreal container, matrix, and sample package: skipped for local diagnosis"
  fi
  if [[ "${run_unity}" -eq 1 ]]; then
    echo "  Unity EditMode ABI: scripts/run_unity_ci_tests.sh"
  else
    echo "  Unity EditMode ABI: skipped for local diagnosis"
  fi

  require_env_for_plan \
    ASTRAL_TEST_VISION_MODEL \
    ASTRAL_TEST_VISION_MEDIA \
    ASTRAL_TEST_AUDIO_MODEL \
    ASTRAL_TEST_AUDIO_MEDIA || missing=1

  if [[ -z "${cuda_arch}" ]]; then
    echo "[release-gate] missing required option: --cuda-arch <deployed-arch-list>" >&2
    missing=1
  fi

  if [[ "${run_unreal}" -eq 1 ]]; then
    require_env_for_plan \
      UNREAL_54_EDITOR \
      UNREAL_55_EDITOR \
      UNREAL_56_EDITOR \
      UNREAL_57_EDITOR \
      UNREAL_RUNUAT || missing=1
  fi

  if [[ "${run_unity}" -eq 1 ]]; then
    require_env_for_plan UNITY_EDITOR || missing=1
  fi

  if [[ "${missing}" -ne 0 ]]; then
    echo "[release-gate] plan has missing release-candidate environment" >&2
    return 1
  fi
  echo "[release-gate] plan environment OK"
}

cuda_args=(--preset-set release)
if [[ -n "${cuda_arch}" ]]; then
  cuda_args+=(--arch "${cuda_arch}")
fi
if [[ "${cuda_strict}" -eq 1 ]]; then
  cuda_args+=(--strict)
fi

mtmd_args=()
if [[ "${mtmd_bench}" -eq 1 ]]; then
  mtmd_args+=(--bench)
fi

if [[ "${print_plan}" -eq 1 ]]; then
  print_release_plan
  exit $?
fi

if [[ -z "${cuda_arch}" ]]; then
  echo "[release-gate] missing required option: --cuda-arch <deployed-arch-list>" >&2
  exit 2
fi

echo "[release-gate] Native release tests"
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests -j --output-on-failure

if [[ "${run_sanitizers}" -eq 1 ]]; then
  echo "[release-gate] ASAN/UBSAN validation"
  scripts/run_asan.sh

  echo "[release-gate] ThreadSanitizer validation"
  scripts/run_tsan.sh
else
  echo "[release-gate] sanitizer validation skipped"
fi

echo "[release-gate] CUDA release matrix"
ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 \
  scripts/run_cuda_parity_matrix.sh "${cuda_args[@]}"

echo "[release-gate] MTMD real media validation"
scripts/run_multimodal_validation.sh "${mtmd_args[@]}"

if [[ "${run_unreal}" -eq 1 ]]; then
  echo "[release-gate] Unreal validation access"
  scripts/check_unreal_validation_access.sh --check-registry

  echo "[release-gate] Unreal native package"
  cmake --preset unreal-plugin
  cmake --build --preset unreal-plugin -j

  echo "[release-gate] Unreal UE 5.7 full container"
  scripts/run_unreal_container_ci.sh --variant full --filter 'AstralRT' --skip-native-build

  echo "[release-gate] Unreal UE 5.7 slim container"
  scripts/run_unreal_container_ci.sh --variant slim --filter 'AstralRT' --skip-native-build

  echo "[release-gate] Unreal 5.4+ compatibility matrix"
  scripts/run_unreal_compatibility_matrix.sh

  echo "[release-gate] Unreal 5.7 sample package"
  scripts/run_unreal_sample_package.sh --platform Linux
else
  echo "[release-gate] Unreal container, compatibility matrix, and sample package skipped"
fi

if [[ "${run_unity}" -eq 1 ]]; then
  echo "[release-gate] Unity EditMode ABI validation"
  scripts/run_unity_ci_tests.sh
else
  echo "[release-gate] Unity EditMode ABI validation skipped"
fi

echo "[release-gate] required gates passed"
