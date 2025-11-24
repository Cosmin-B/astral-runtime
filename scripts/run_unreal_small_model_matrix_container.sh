#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_engine="${CONTAINER_ENGINE:-docker}"
pull_timeout="${ASTRAL_UNREAL_PULL_TIMEOUT:-${ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS:-120}}"
ue_version="5.7"
variant="slim"
image=""
digest=""
pull_image=1
print_command=0
build_native=0
models_dir="tests/models"
out_dir="build/unreal-small-model-matrix"
declare -a matrix_args=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_unreal_small_model_matrix_container.sh [options]

Runs the Unreal packaged-sample small-model matrix inside an Epic UE container.
This is for cached full/slim images when the host does not have RunUAT.

Options:
  --ue-version <5.4|5.5|5.6|5.7>
  --variant <slim|full>    UE image variant (default: slim)
  --image <ref>            Override container image
  --digest <sha256:...>    Override image digest
  --skip-pull              Use a locally cached image
  --pull-timeout <dur>     Bound docker pull time (default: 120s)
  --build-native           Rebuild the Unreal ThirdParty package in-container
  --print-command          Print the docker command without running it
  --models-dir <dir>       Forwarded to the matrix runner
  --model <path>           Forwarded to the matrix runner; repeatable
  --preset <name>          Forwarded to the matrix runner; repeatable
  --embedding-model <path> Forwarded to the matrix runner
  --out <dir>              Forwarded to the matrix runner
  --list                   Forwarded to the matrix runner
  --dry-run                Forwarded to the matrix runner
  -h, --help               Show help

By default the wrapper reuses an already staged ThirdParty package and forwards
--skip-native-build to run_unreal_small_model_matrix.sh.
USAGE
}

image_tag_for_version() {
  case "$1:$2" in
    5.4:slim) printf 'ghcr.io/epicgames/unreal-engine:dev-slim-5.4.4\n' ;;
    5.4:full) printf 'ghcr.io/epicgames/unreal-engine:dev-5.4.4\n' ;;
    5.5:slim) printf 'ghcr.io/epicgames/unreal-engine:dev-slim-5.5.4\n' ;;
    5.5:full) printf 'ghcr.io/epicgames/unreal-engine:dev-5.5.4\n' ;;
    5.6:slim) printf 'ghcr.io/epicgames/unreal-engine:dev-slim-5.6.1\n' ;;
    5.6:full) printf 'ghcr.io/epicgames/unreal-engine:dev-5.6.1\n' ;;
    5.7:slim) printf 'ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4\n' ;;
    5.7:full) printf 'ghcr.io/epicgames/unreal-engine:dev-5.7.4\n' ;;
    *)
      echo "Unsupported --ue-version/--variant combination: $1/$2" >&2
      exit 2
      ;;
  esac
}

digest_for_version() {
  case "$1:$2" in
    5.7:slim) printf 'sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n' ;;
    5.7:full) printf 'sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n' ;;
    *) printf '\n' ;;
  esac
}

shell_join() {
  local arg
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ue-version) ue_version="${2:-}"; shift 2 ;;
    --variant) variant="${2:-}"; shift 2 ;;
    --image) image="${2:-}"; shift 2 ;;
    --digest) digest="${2:-}"; shift 2 ;;
    --skip-pull) pull_image=0; shift ;;
    --pull-timeout) pull_timeout="${2:-}"; shift 2 ;;
    --build-native) build_native=1; shift ;;
    --print-command) print_command=1; shift ;;
    --models-dir) models_dir="${2:-}"; shift 2 ;;
    --model) matrix_args+=(--model "${2:-}"); shift 2 ;;
    --preset) matrix_args+=(--preset "${2:-}"); shift 2 ;;
    --embedding-model) matrix_args+=(--embedding-model "${2:-}"); shift 2 ;;
    --out) out_dir="${2:-}"; shift 2 ;;
    --list) matrix_args+=(--list); shift ;;
    --dry-run) matrix_args+=(--dry-run); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${variant}" in
  slim|full) ;;
  *) echo "Unsupported --variant '${variant}' (expected slim or full)" >&2; exit 2 ;;
esac

if ! command -v "${container_engine}" >/dev/null 2>&1; then
  echo "Container engine not found: ${container_engine}" >&2
  exit 2
fi

if [[ -z "${image}" ]]; then
  image="$(image_tag_for_version "${ue_version}" "${variant}")"
  digest="${digest:-$(digest_for_version "${ue_version}" "${variant}")}"
fi

image_ref="${image}"
if [[ -n "${digest}" && "${image}" != *@sha256:* ]]; then
  image_ref="${image}@${digest}"
fi

if [[ "${pull_image}" -eq 1 ]]; then
  echo "[unreal_small_matrix_container] Pull image: ${image_ref}"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${pull_timeout}" "${container_engine}" pull "${image_ref}" </dev/null
  else
    "${container_engine}" pull "${image_ref}" </dev/null
  fi
fi

inner_args=(--models-dir "${models_dir}" --out "${out_dir}" "${matrix_args[@]}")
if [[ "${build_native}" -eq 0 ]]; then
  inner_args+=(--skip-native-build)
fi

inner_command='
set -euo pipefail
cd /workspace/astral
engine_root=""
for candidate in "${UNREAL_ENGINE_DIR:-}" /home/ue4/UnrealEngine /home/ue/UnrealEngine /opt/unreal-engine /UnrealEngine; do
  if [[ -n "${candidate}" && -d "${candidate}/Engine" ]]; then
    engine_root="${candidate}"
    break
  fi
done
if [[ -z "${engine_root}" ]]; then
  echo "[unreal_small_matrix_container] Unreal engine root not found" >&2
  exit 2
fi
echo "[unreal_small_matrix_container] Engine root: ${engine_root}"
'
inner_command+='exec ./scripts/run_unreal_small_model_matrix.sh --engine-dir "${engine_root}"'
if [[ "${#inner_args[@]}" -gt 0 ]]; then
  inner_command+=" $(shell_join "${inner_args[@]}")"
fi

cmd=(
  "${container_engine}" run --rm --init
  -v "${root_dir}:/workspace/astral"
  -w /workspace/astral
  "${image_ref}"
  bash -lc "${inner_command}"
)

printf '[unreal_small_matrix_container] run:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${print_command}" -eq 1 ]]; then
  exit 0
fi

"${cmd[@]}"
