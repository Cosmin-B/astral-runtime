#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="${root_dir}/ci/unity/AstralCiUnityProject"
results_dir="${root_dir}/build/unity-gameci-results"
unity_version=""
image_version="${ASTRAL_UNITY_GAMECI_IMAGE_VERSION:-3.2.2}"
image_component="${ASTRAL_UNITY_GAMECI_COMPONENT:-base}"
image="${ASTRAL_UNITY_GAMECI_IMAGE:-}"
native_build_image="${ASTRAL_UNITY_NATIVE_BUILD_IMAGE:-}"
gameci_docs_url="https://game.ci/docs/docker/docker-images/"
docker_bin="${DOCKER:-docker}"
pull_image=1
build_native=1
dry_run=0

usage() {
  cat <<'EOF'
Usage: scripts/run_unity_gameci_tests.sh [options]

Runs the Unity EditMode ABI lane in a GameCI Unity Editor container.

Options:
  --image <ref>            Docker image (default: unityci/editor:ubuntu-<project-version>-base-3.2.2)
  --unity-version <ver>    Unity editor version (default: ProjectVersion.txt)
  --image-version <ver>    GameCI Docker image version (default: 3.2.2)
  --component <name>       GameCI image component (default: base)
  --project <path>         Unity project path (default: ci/unity/AstralCiUnityProject)
  --results-dir <path>     Result directory (default: build/unity-gameci-results)
  --native-build-image <ref>
                          Build the native plugin in this Linux container before launching Unity
  --skip-build             Do not build the native Unity plugin before container launch
  --skip-pull              Do not pull the Docker image before launch
  --dry-run                Print the resolved docker command and exit
  --help                   Show help

Unity license material is passed only through environment variables when they
already exist in the caller environment. This script does not read or write
license files.
EOF
}

project_unity_version() {
  local version_file="${project_dir}/ProjectSettings/ProjectVersion.txt"
  if [[ ! -f "${version_file}" ]]; then
    echo "Unity project version file not found: ${version_file}" >&2
    exit 2
  fi
  sed -n 's/^m_EditorVersion:[[:space:]]*//p' "${version_file}" | head -n 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="${2:-}"; shift 2 ;;
    --unity-version) unity_version="${2:-}"; shift 2 ;;
    --image-version) image_version="${2:-}"; shift 2 ;;
    --component) image_component="${2:-}"; shift 2 ;;
    --project) project_dir="$(cd "${2:-}" && pwd)"; shift 2 ;;
    --results-dir) results_dir="$(mkdir -p "${2:-}" && cd "${2:-}" && pwd)"; shift 2 ;;
    --native-build-image) native_build_image="${2:-}"; shift 2 ;;
    --skip-build) build_native=0; shift ;;
    --skip-pull) pull_image=0; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -d "${project_dir}" ]]; then
  echo "Unity CI project not found: ${project_dir}" >&2
  exit 2
fi

if [[ -z "${unity_version}" ]]; then
  unity_version="$(project_unity_version)"
fi
if [[ -z "${unity_version}" ]]; then
  echo "Unity version could not be resolved from ${project_dir}" >&2
  exit 2
fi

if [[ -z "${image}" ]]; then
  image="unityci/editor:ubuntu-${unity_version}-${image_component}-${image_version}"
fi

mkdir -p "${results_dir}"

declare -a docker_args=(
  run --rm
  --workdir /workspace
  --volume "${root_dir}:/workspace"
  --volume "${results_dir}:/workspace/build/unity-gameci-results"
  --env ASTRAL_UNITY_REQUIRE_NATIVE=1
)

for env_name in ASTRAL_ENGINE_MEMORY_PERF ASTRAL_ENGINE_MEMORY_MAX_P50_US; do
  if [[ -n "${!env_name:-}" ]]; then
    docker_args+=(--env "${env_name}")
  fi
done

for env_name in UNITY_LICENSE UNITY_EMAIL UNITY_PASSWORD UNITY_SERIAL UNITY_LICENSING_SERVER; do
  if [[ -n "${!env_name:-}" ]]; then
    docker_args+=(--env "${env_name}")
  fi
done

docker_args+=(
  "${image}"
  bash -lc
  "scripts/run_unity_ci_tests.sh --skip-build --editor /opt/unity/Editor/Unity --project ci/unity/AstralCiUnityProject --results-dir build/unity-gameci-results"
)

echo "[unity-gameci] Image: ${image}"
echo "[unity-gameci] Docs: ${gameci_docs_url}"
echo "[unity-gameci] Project: ${project_dir}"
echo "[unity-gameci] Results: ${results_dir}/editmode-results.xml"

if [[ "${dry_run}" -eq 1 ]]; then
  printf '[unity-gameci] Command: %q' "${docker_bin}"
  printf ' %q' "${docker_args[@]}"
  printf '\n'
  exit 0
fi

if ! command -v "${docker_bin}" >/dev/null 2>&1; then
  echo "Docker executable not found: ${docker_bin}" >&2
  exit 2
fi

if [[ "${build_native}" -eq 1 ]]; then
  if [[ -n "${native_build_image}" ]]; then
    echo "[unity-gameci] Build native plugin in ${native_build_image}"
    "${docker_bin}" run --rm \
      --workdir /workspace \
      --volume "${root_dir}:/workspace" \
      "${native_build_image}" \
      bash -lc "set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; apt-get update >/dev/null; apt-get install -y --no-install-recommends ca-certificates build-essential cmake >/dev/null; cmake -S . -B build/unity-linux-ci -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_DIR=/workspace/external/llama.cpp -DASTRAL_BUILD_TESTS=OFF -DASTRAL_BUILD_BENCHMARKS=OFF -DASTRAL_BUILD_UNITY_PLUGIN=ON -DASTRAL_BUILD_STATIC_LIB=OFF -DASTRAL_BUILD_SHARED_LIB=ON; cmake --build build/unity-linux-ci -j\"\$(nproc)\" --target astral_unity_plugin_package; chown -R $(id -u):$(id -g) build/unity-linux-ci plugins/unity/Runtime/Plugins/x86_64/libastral_rt.so"
  else
    echo "[unity-gameci] Configure native plugin"
    cmake --preset unity-plugin

    echo "[unity-gameci] Build native plugin"
    cmake --build --preset unity-plugin -j
  fi
fi

native_library="${root_dir}/plugins/unity/Runtime/Plugins/x86_64/libastral_rt.so"
if [[ ! -s "${native_library}" ]]; then
  echo "Unity native runtime missing or empty: ${native_library}" >&2
  exit 2
fi

if [[ "${pull_image}" -eq 1 ]]; then
  echo "[unity-gameci] Pull image"
  "${docker_bin}" pull "${image}"
fi

"${docker_bin}" "${docker_args[@]}"
python3 "${root_dir}/scripts/validate_unity_editmode_results.py" \
  "${results_dir}/editmode-results.xml"
