#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
test_filter="${UNREAL_TEST_FILTER:-AstralRT.*}"
results_dir="${root_dir}/build/unreal-ci-results"

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
  cmake -E copy_directory "${source_project_dir}" "${project_dir}"
  cmake -E copy_directory "${plugin_dir}" "${project_dir}/Plugins/AstralRT"
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
exec_cmds="Automation RunTests ${test_filter};Quit"

echo "[unreal_ci] Project: ${project_file}"
echo "[unreal_ci] Filter: ${test_filter}"
echo "[unreal_ci] Report: ${report_dir}"

"${unreal_editor}" "${project_file}" \
  -Unattended \
  -NullRHI \
  -NoSplash \
  -NoSound \
  -NoPause \
  -NoP4 \
  -ExecCmds="${exec_cmds}" \
  -TestExit="Automation Test Queue Empty" \
  -ReportOutputPath="${report_dir}" \
  -log="${log_file}"

echo "[unreal_ci] Log: ${log_file}"
