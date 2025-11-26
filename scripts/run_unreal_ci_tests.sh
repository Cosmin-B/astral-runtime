#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

copy_directory() {
  local src="$1"
  local dst="$2"

  if command -v cmake >/dev/null 2>&1; then
    cmake -E copy_directory "${src}" "${dst}"
    return
  fi

  rm -rf "${dst}"
  mkdir -p "${dst}"
  cp -a "${src}/." "${dst}/"
}

host_platform() {
  case "$(uname -s 2>/dev/null || true)" in
    Linux*) printf 'Linux' ;;
    Darwin*) printf 'Mac' ;;
    MINGW*|MSYS*|CYGWIN*) printf 'Win64' ;;
    *)
      echo "Unsupported host platform for Unreal build: $(uname -s 2>/dev/null || true)" >&2
      return 2
      ;;
  esac
}

engine_root_from_editor() {
  local editor_path="$1"
  local resolved_editor="${editor_path}"

  if [[ "${resolved_editor}" != */* ]]; then
    resolved_editor="$(command -v "${resolved_editor}")"
  fi
  resolved_editor="$(cd "$(dirname "${resolved_editor}")" && pwd)/$(basename "${resolved_editor}")"

  case "${resolved_editor}" in
    */Engine/Binaries/*/UnrealEditor*|*/Engine/Binaries/*/UnrealEditor-Cmd*)
      printf '%s\n' "${resolved_editor%%/Engine/Binaries/*}"
      ;;
  esac
}

run_unreal_build() {
  local project_file="$1"
  local platform="$2"
  local run_ubt="${UNREAL_BUILD_TOOL:-}"

  if [[ -z "${run_ubt}" ]]; then
    local engine_root
    engine_root="$(engine_root_from_editor "${unreal_editor}")"
    if [[ -n "${engine_root}" ]]; then
      run_ubt="${engine_root}/Engine/Build/BatchFiles/RunUBT.sh"
    fi
  fi

  if [[ -z "${run_ubt}" || ! -x "${run_ubt}" ]]; then
    echo "Missing UnrealBuildTool runner. Set UNREAL_BUILD_TOOL to RunUBT.sh." >&2
    exit 2
  fi

  echo "[unreal_ci] Build: AstralCiUnrealProjectEditor ${platform} Development"
  "${run_ubt}" AstralCiUnrealProjectEditor "${platform}" Development \
    -Project="${project_file}" \
    -NoHotReload \
    -NoUBTMakefiles
}

unreal_editor="${UNREAL_EDITOR:-}"
if [[ -z "${unreal_editor}" ]]; then
  echo "Missing UNREAL_EDITOR env var (path to UnrealEditor-Cmd or UnrealEditor executable)." >&2
  exit 2
fi

if [[ "${unreal_editor}" == */* ]]; then
  if [[ ! -x "${unreal_editor}" ]]; then
    echo "UNREAL_EDITOR is not executable: ${unreal_editor}" >&2
    exit 2
  fi
elif ! command -v "${unreal_editor}" >/dev/null 2>&1; then
  echo "UNREAL_EDITOR was not found in PATH: ${unreal_editor}" >&2
  exit 2
fi

source_project_dir="${root_dir}/ci/unreal/AstralCiUnrealProject"
source_project_file="${source_project_dir}/AstralCiUnrealProject.uproject"
plugin_dir="${root_dir}/plugins/unreal/AstralRT"
test_filter="${UNREAL_TEST_FILTER:-AstralRT}"
results_dir="${ASTRAL_UNREAL_RESULTS_DIR:-${root_dir}/build/unreal-ci-results}"

if [[ -n "${ASTRAL_UNREAL_PROJECT:-}" ]]; then
  project_dir="${ASTRAL_UNREAL_PROJECT}"
  project_file="${project_dir}/AstralCiUnrealProject.uproject"
  if [[ ! -f "${project_file}" ]]; then
    project_file="$(find "${project_dir}" -maxdepth 1 -name '*.uproject' -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "${project_file}" || ! -f "${project_file}" ]]; then
    echo "No .uproject found in ASTRAL_UNREAL_PROJECT: ${project_dir}" >&2
    exit 2
  fi
  if [[ ! -f "${project_dir}/Plugins/AstralRT/AstralRT.uplugin" ]]; then
    echo "AstralRT plugin is not installed in project: ${project_dir}/Plugins/AstralRT" >&2
    echo "Install plugins/unreal/AstralRT there, or leave ASTRAL_UNREAL_PROJECT unset to use the staged CI project." >&2
    exit 2
  fi
else
  if [[ ! -f "${source_project_file}" ]]; then
    echo "Unreal CI project scaffold not found: ${source_project_file}" >&2
    exit 2
  fi
  if [[ ! -f "${plugin_dir}/AstralRT.uplugin" ]]; then
    echo "AstralRT plugin not found: ${plugin_dir}" >&2
    exit 2
  fi

  project_dir="${root_dir}/build/unreal-ci-project/AstralCiUnrealProject"
  project_file="${project_dir}/AstralCiUnrealProject.uproject"
  rm -rf "${project_dir}"
  mkdir -p "${project_dir}/Plugins"
  copy_directory "${source_project_dir}" "${project_dir}"
  copy_directory "${plugin_dir}" "${project_dir}/Plugins/AstralRT"
fi

if [[ ! -f "${project_dir}/Plugins/AstralRT/Source/ThirdParty/AstralCore/include/astral_rt.h" ]]; then
  echo "AstralRT ThirdParty headers are missing from the plugin." >&2
  echo "Run: cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j" >&2
  exit 2
fi

third_party_dir="${project_dir}/Plugins/AstralRT/Source/ThirdParty/AstralCore"
native_library=""
case "$(uname -s 2>/dev/null || true)" in
  Linux*) native_library="${third_party_dir}/lib/Linux/libastral_rt.a" ;;
  Darwin*) native_library="${third_party_dir}/lib/Mac/libastral_rt.a" ;;
  MINGW*|MSYS*|CYGWIN*) native_library="${third_party_dir}/lib/Win64/astral_rt.lib" ;;
esac

if [[ -n "${native_library}" && ! -f "${native_library}" ]]; then
  echo "AstralRT native library is missing: ${native_library}" >&2
  echo "Run: cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j" >&2
  exit 2
fi

mkdir -p "${results_dir}"

report_dir="${results_dir}/automation-report"
log_file="${results_dir}/unreal-automation.log"
engine_log_file="${results_dir}/unreal-automation-engine.log"
exec_cmds="Automation RunTests ${test_filter};Quit"
target_platform="$(host_platform)"

echo "[unreal_ci] Project: ${project_file}"
echo "[unreal_ci] Filter: ${test_filter}"
echo "[unreal_ci] Report: ${report_dir}"

run_unreal_build "${project_file}" "${target_platform}"

"${unreal_editor}" "${project_file}" \
  -Unattended \
  -NullRHI \
  -NoSplash \
  -NoSound \
  -NoPause \
  -NoP4 \
  -ExecCmds="${exec_cmds}" \
  -TestExit="Automation Test Queue Empty" \
  -ReportExportPath="${report_dir}" \
  -log="${engine_log_file}" \
  2>&1 | tee "${log_file}"

python3 "${root_dir}/scripts/validate_unreal_automation_results.py" \
  --log "${log_file}" \
  --report-dir "${report_dir}" \
  --filter "${test_filter}"

echo "[unreal_ci] Log: ${log_file}"
