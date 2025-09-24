#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

full_image="ghcr.io/epicgames/unreal-engine:dev-5.7.4"
full_digest="sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce"
slim_image="ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4"
slim_digest="sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6"

container_engine="${CONTAINER_ENGINE:-docker}"
variant="slim"
image=""
digest=""
pull_image=1
build_native=1
test_filter="${UNREAL_TEST_FILTER:-AstralRT.*}"
unreal_editor="${UNREAL_EDITOR:-}"

usage() {
  cat <<'EOF'
Usage: scripts/run_unreal_container_ci.sh [options]

Run AstralRT Unreal Automation in an Epic UE Linux container.

Options:
  --variant <slim|full>     UE 5.7 image variant (default: slim)
  --image <ref>             Override container image reference
  --digest <sha256:...>     Override digest for --image or selected variant
  --skip-pull               Do not pull the image before running
  --skip-native-build       Do not rebuild the AstralRT ThirdParty package
  --editor <path>           UnrealEditor-Cmd path inside the container
  --filter <pattern>        Automation filter (default: AstralRT.*)
  -h, --help                Show this help

Environment:
  CONTAINER_ENGINE          Container CLI (default: docker)
  UNREAL_EDITOR             Same as --editor
  UNREAL_TEST_FILTER        Same as --filter

The default images are private Epic GHCR packages. Authenticate Docker with an
Epic-linked GitHub account before running this script.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      variant="${2:-}"
      shift 2
      ;;
    --image)
      image="${2:-}"
      shift 2
      ;;
    --digest)
      digest="${2:-}"
      shift 2
      ;;
    --skip-pull)
      pull_image=0
      shift
      ;;
    --skip-native-build)
      build_native=0
      shift
      ;;
    --editor)
      unreal_editor="${2:-}"
      shift 2
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

if ! command -v "${container_engine}" >/dev/null 2>&1; then
  echo "Container engine not found: ${container_engine}" >&2
  exit 2
fi

if [[ -z "${image}" ]]; then
  case "${variant}" in
    slim)
      image="${slim_image}"
      digest="${digest:-${slim_digest}}"
      ;;
    full)
      image="${full_image}"
      digest="${digest:-${full_digest}}"
      ;;
    *)
      echo "Unsupported --variant '${variant}' (expected slim or full)" >&2
      exit 2
      ;;
  esac
fi

image_ref="${image}"
if [[ -n "${digest}" && "${image}" != *@sha256:* ]]; then
  image_ref="${image}@${digest}"
fi

if [[ "${pull_image}" -eq 1 ]]; then
  "${container_engine}" pull "${image_ref}"
fi

image_digests="$("${container_engine}" image inspect "${image_ref}" --format '{{range .RepoDigests}}{{println .}}{{end}}' 2>/dev/null || true)"
if [[ -n "${image_digests}" ]]; then
  echo "[unreal_container] Local image digests:"
  printf '%s\n' "${image_digests}"
fi
if [[ -n "${digest}" && "${image_digests}" != *"${digest}"* ]]; then
  echo "Pulled image does not report expected digest ${digest}" >&2
  exit 2
fi

container_script='
set -euo pipefail

cd /workspace/astral

echo "[unreal_container] Image: ${ASTRAL_UNREAL_IMAGE_REF}"
echo "[unreal_container] Test filter: ${UNREAL_TEST_FILTER}"

engine_root=""
for candidate in \
  "${UNREAL_ENGINE_DIR:-}" \
  /home/ue4/UnrealEngine \
  /home/ue/UnrealEngine \
  /opt/unreal-engine \
  /UnrealEngine; do
  if [[ -n "${candidate}" && -d "${candidate}/Engine" ]]; then
    engine_root="${candidate}"
    break
  fi
done

if [[ -n "${engine_root}" ]]; then
  echo "[unreal_container] Engine root: ${engine_root}"
  if [[ -f "${engine_root}/Engine/Build/Build.version" ]]; then
    sed -n "1,80p" "${engine_root}/Engine/Build/Build.version"
  fi
  linux_sdk="${engine_root}/Engine/Config/Linux/Linux_SDK.json"
  if [[ -f "${linux_sdk}" ]]; then
    echo "[unreal_container] Linux SDK: ${linux_sdk}"
    sed -n "1,120p" "${linux_sdk}"
  else
    echo "[unreal_container] Linux SDK metadata not found under ${engine_root}" >&2
  fi
fi

if command -v clang >/dev/null 2>&1; then
  echo "[unreal_container] clang:"
  clang --version | sed -n "1,3p"
else
  echo "[unreal_container] clang not found in PATH" >&2
fi

if [[ "${ASTRAL_UNREAL_BUILD_NATIVE}" == "1" ]]; then
  cmake --preset unreal-plugin
  cmake --build --preset unreal-plugin -j
fi

if [[ -z "${UNREAL_EDITOR:-}" ]]; then
  for candidate in \
    "${engine_root}/Engine/Binaries/Linux/UnrealEditor-Cmd" \
    "${engine_root}/Engine/Binaries/Linux/UnrealEditor" \
    /home/ue4/UnrealEngine/Engine/Binaries/Linux/UnrealEditor-Cmd \
    /home/ue/UnrealEngine/Engine/Binaries/Linux/UnrealEditor-Cmd \
    /opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor-Cmd \
    /UnrealEngine/Engine/Binaries/Linux/UnrealEditor-Cmd; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      export UNREAL_EDITOR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${UNREAL_EDITOR:-}" ]]; then
  echo "Could not locate UnrealEditor-Cmd inside the container; set --editor or UNREAL_EDITOR." >&2
  exit 2
fi

./scripts/run_unreal_ci_tests.sh
'

"${container_engine}" run --rm --init \
  -e "ASTRAL_UNREAL_BUILD_NATIVE=${build_native}" \
  -e "ASTRAL_UNREAL_IMAGE_REF=${image_ref}" \
  -e "UNREAL_EDITOR=${unreal_editor}" \
  -e "UNREAL_TEST_FILTER=${test_filter}" \
  -v "${root_dir}:/workspace/astral" \
  -w /workspace/astral \
  "${image_ref}" \
  bash -lc "${container_script}"
