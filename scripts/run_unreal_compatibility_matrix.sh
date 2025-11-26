#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

versions="5.4 5.5 5.6 5.7"
allow_missing=0
build_native=1
test_filter="${UNREAL_TEST_FILTER:-AstralRT}"

usage() {
  cat <<'EOF'
Usage: scripts/run_unreal_compatibility_matrix.sh [options]

Run AstralRT Unreal Automation against the UE 5.4+ compatibility matrix.

Options:
  --versions "<list>"       Space-separated UE versions (default: "5.4 5.5 5.6 5.7")
  --allow-missing           Skip versions whose editor env var is unset
  --skip-native-build       Do not rebuild the AstralRT ThirdParty package first
  --filter <pattern>        Automation filter (default: AstralRT)
  -h, --help                Show this help

Required editor env vars by version:
  UE 5.4: UNREAL_54_EDITOR=/path/to/UnrealEditor-Cmd
  UE 5.5: UNREAL_55_EDITOR=/path/to/UnrealEditor-Cmd
  UE 5.6: UNREAL_56_EDITOR=/path/to/UnrealEditor-Cmd
  UE 5.7: UNREAL_57_EDITOR=/path/to/UnrealEditor-Cmd

Each version writes logs and Automation reports under:
  build/unreal-ci-results/ue-<version>/

Release-candidate runs should not use --allow-missing.
EOF
}

env_name_for_version() {
  case "$1" in
    5.4) printf 'UNREAL_54_EDITOR\n' ;;
    5.5) printf 'UNREAL_55_EDITOR\n' ;;
    5.6) printf 'UNREAL_56_EDITOR\n' ;;
    5.7) printf 'UNREAL_57_EDITOR\n' ;;
    *)
      echo "Unsupported Unreal version '$1' (expected 5.4, 5.5, 5.6, or 5.7)" >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --versions)
      versions="${2:-}"
      shift 2
      ;;
    --allow-missing)
      allow_missing=1
      shift
      ;;
    --skip-native-build)
      build_native=0
      shift
      ;;
    --filter)
      test_filter="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${versions}" ]]; then
  echo "No Unreal versions requested." >&2
  exit 2
fi

if [[ "${build_native}" -eq 1 ]]; then
  cmake --preset unreal-plugin
  cmake --build --preset unreal-plugin -j
fi

ran_any=0
for version in ${versions}; do
  editor_var="$(env_name_for_version "${version}")"
  editor="${!editor_var:-}"

  if [[ -z "${editor}" ]]; then
    if [[ "${allow_missing}" -eq 1 ]]; then
      echo "[unreal_matrix] Skipping UE ${version}: ${editor_var} is unset"
      continue
    fi
    echo "Missing ${editor_var} for UE ${version}. Set it or pass --allow-missing for local discovery runs." >&2
    exit 2
  fi

  result_dir="${root_dir}/build/unreal-ci-results/ue-${version}"
  echo "[unreal_matrix] UE ${version}: ${editor}"
  UNREAL_EDITOR="${editor}" \
  UNREAL_TEST_FILTER="${test_filter}" \
  ASTRAL_UNREAL_RESULTS_DIR="${result_dir}" \
    "${root_dir}/scripts/run_unreal_ci_tests.sh"
  ran_any=1
done

if [[ "${ran_any}" -eq 0 ]]; then
  echo "No Unreal versions ran." >&2
  exit 2
fi
