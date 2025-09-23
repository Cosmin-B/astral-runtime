#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="${root_dir}/ci/unity/AstralCiUnityProject"
results_dir="${root_dir}/build/unity-ci-results"
unity_editor="${UNITY_EDITOR:-}"
build_native=1

usage() {
  cat <<'EOF'
Usage: scripts/run_unity_ci_tests.sh [options]

Builds the Unity native plugin, checks that Unity can load the native runtime,
and runs the package EditMode ABI tests.

Options:
  --editor <path>       Unity editor executable (or UNITY_EDITOR env var)
  --project <path>      Unity project path (default: ci/unity/AstralCiUnityProject)
  --results-dir <path>  Test result directory (default: build/unity-ci-results)
  --skip-build          Do not build the native Unity plugin first
  --help                Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --editor) unity_editor="${2:-}"; shift 2 ;;
    --project) project_dir="${2:-}"; shift 2 ;;
    --results-dir) results_dir="${2:-}"; shift 2 ;;
    --skip-build) build_native=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${unity_editor}" ]]; then
  echo "Missing Unity editor path. Set UNITY_EDITOR or pass --editor." >&2
  exit 2
fi
if [[ ! -x "${unity_editor}" ]]; then
  echo "Unity editor is not executable: ${unity_editor}" >&2
  exit 2
fi

if [[ ! -d "${project_dir}" ]]; then
  echo "Unity CI project not found: ${project_dir}" >&2
  exit 2
fi

if [[ "${build_native}" -eq 1 ]]; then
  echo "[unity-ci] Configure native plugin"
  cmake --preset unity-plugin

  echo "[unity-ci] Build native plugin"
  cmake --build --preset unity-plugin -j
fi

case "$(uname -s)" in
  Linux*) native_library="${root_dir}/plugins/unity/Runtime/Plugins/x86_64/libastral_rt.so" ;;
  Darwin*) native_library="${root_dir}/plugins/unity/Runtime/Plugins/x86_64/libastral_rt.dylib" ;;
  MINGW*|MSYS*|CYGWIN*) native_library="${root_dir}/plugins/unity/Runtime/Plugins/x86_64/astral_rt.dll" ;;
  *) echo "Unsupported host OS for Unity native plugin preflight: $(uname -s)" >&2; exit 2 ;;
esac

if [[ ! -s "${native_library}" ]]; then
  echo "Unity native runtime missing or empty: ${native_library}" >&2
  exit 2
fi

mkdir -p "${results_dir}"
rm -f "${results_dir}/editmode-results.xml"

ASTRAL_UNITY_REQUIRE_NATIVE=1 "${unity_editor}" \
  -batchmode \
  -nographics \
  -quit \
  -projectPath "${project_dir}" \
  -runTests \
  -testPlatform EditMode \
  -testResults "${results_dir}/editmode-results.xml" \
  -logFile "${results_dir}/unity-editmode.log"

if [[ ! -s "${results_dir}/editmode-results.xml" ]]; then
  echo "Unity did not write EditMode results: ${results_dir}/editmode-results.xml" >&2
  exit 1
fi

echo "[unity-ci] Results: ${results_dir}/editmode-results.xml"
