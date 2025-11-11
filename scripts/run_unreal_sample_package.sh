#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

out_dir="${ASTRAL_UNREAL_SAMPLE_DIR:-${root_dir}/build/unreal-sample-package/AstralSample}"
archive_dir="${ASTRAL_UNREAL_SAMPLE_ARCHIVE_DIR:-${root_dir}/build/unreal-sample-package/archive}"
platform="${ASTRAL_UNREAL_SAMPLE_PLATFORM:-Linux}"
runuat="${UNREAL_RUNUAT:-}"
unreal_editor="${UNREAL_EDITOR:-}"
engine_dir="${UNREAL_ENGINE_DIR:-}"
build_native=1

usage() {
  cat <<'USAGE'
Usage: scripts/run_unreal_sample_package.sh [options]

Generate the sidecar AstralSample UE 5.7 project outside the repo and package it
with RunUAT BuildCookRun.

Options:
  --out <path>             Generated AstralSample project path
  --archive-dir <path>     BuildCookRun archive output path
  --platform <name>        Unreal target platform (default: Linux)
  --runuat <path>          Path to RunUAT.sh or RunUAT.bat
  --editor <path>          UnrealEditor-Cmd/UnrealEditor path used to infer RunUAT
  --engine-dir <path>      Unreal engine root used to infer RunUAT
  --skip-native-build      Do not rebuild the AstralRT ThirdParty package first
  -h, --help               Show this help

Environment:
  UNREAL_RUNUAT            Same as --runuat
  UNREAL_ENGINE_DIR        Same as --engine-dir
  UNREAL_EDITOR            Same as --editor

The generated sample project is a sidecar artifact and must not be committed to
the Astral repo.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      out_dir="${2:-}"
      shift 2
      ;;
    --archive-dir)
      archive_dir="${2:-}"
      shift 2
      ;;
    --platform)
      platform="${2:-}"
      shift 2
      ;;
    --runuat)
      runuat="${2:-}"
      shift 2
      ;;
    --editor)
      unreal_editor="${2:-}"
      shift 2
      ;;
    --engine-dir)
      engine_dir="${2:-}"
      shift 2
      ;;
    --skip-native-build)
      build_native=0
      shift
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

infer_runuat_from_engine_dir() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    return 1
  fi
  for script in \
    "${candidate}/Engine/Build/BatchFiles/RunUAT.sh" \
    "${candidate}/Engine/Build/BatchFiles/RunUAT.bat"; do
    if [[ -f "${script}" ]]; then
      printf '%s\n' "${script}"
      return 0
    fi
  done
  return 1
}

infer_engine_dir_from_editor() {
  local editor="$1"
  if [[ -z "${editor}" || "${editor}" != */Engine/Binaries/* ]]; then
    return 1
  fi
  printf '%s\n' "${editor%%/Engine/Binaries/*}"
}

if [[ -z "${runuat}" && -n "${engine_dir}" ]]; then
  runuat="$(infer_runuat_from_engine_dir "${engine_dir}" || true)"
fi

if [[ -z "${runuat}" && -n "${unreal_editor}" ]]; then
  inferred_engine_dir="$(infer_engine_dir_from_editor "${unreal_editor}" || true)"
  runuat="$(infer_runuat_from_engine_dir "${inferred_engine_dir}" || true)"
fi

if [[ -z "${runuat}" ]]; then
  cat >&2 <<'EOF'
Missing Unreal RunUAT path.
Set UNREAL_RUNUAT, UNREAL_ENGINE_DIR, or UNREAL_EDITOR so the sample package lane can run.
EOF
  exit 2
fi

if [[ ! -f "${runuat}" ]]; then
  echo "RunUAT path does not exist: ${runuat}" >&2
  exit 2
fi

if [[ "${runuat}" != *.bat && ! -x "${runuat}" ]]; then
  echo "RunUAT is not executable: ${runuat}" >&2
  exit 2
fi

if [[ "${build_native}" -eq 1 ]]; then
  cmake --preset unreal-plugin
  cmake --build --preset unreal-plugin -j
fi

"${root_dir}/scripts/create_unreal_sample_project.sh" \
  --out "${out_dir}" \
  --plugin-mode copy

project_file="${out_dir}/AstralSample.uproject"
if [[ ! -f "${project_file}" ]]; then
  echo "Generated sample project is missing: ${project_file}" >&2
  exit 2
fi

mkdir -p "${archive_dir}"

echo "[unreal_sample] Project: ${project_file}"
echo "[unreal_sample] Archive: ${archive_dir}"
echo "[unreal_sample] RunUAT: ${runuat}"
echo "[unreal_sample] Platform: ${platform}"
echo "[unreal_sample] Plugin mode: copy"
echo "[unreal_sample] BuildCookRun"

"${runuat}" BuildCookRun \
  -project="${project_file}" \
  -noP4 \
  -clientconfig=Development \
  -platform="${platform}" \
  -build \
  -cook \
  -stage \
  -pak \
  -archive \
  -archivedirectory="${archive_dir}" \
  -utf8output

echo "[unreal_sample] OK: ${archive_dir}"
