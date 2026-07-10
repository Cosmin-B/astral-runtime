#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

container_engine="${CONTAINER_ENGINE:-docker}"
pull_timeout="${ASTRAL_UNREAL_PULL_TIMEOUT:-${ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS:-120}}"
required_clang_version="${ASTRAL_UNREAL_REQUIRED_CLANG_VERSION:-}"
required_linux_sdk_toolchain="${ASTRAL_UNREAL_REQUIRED_LINUX_SDK_TOOLCHAIN:-}"
required_linux_sdk_clang="${ASTRAL_UNREAL_REQUIRED_LINUX_SDK_CLANG:-}"
ue_version="5.7"
variant="slim"
image=""
digest=""
pull_image=1
build_native=1
install_cmake=0
test_filter="${UNREAL_TEST_FILTER:-AstralRT}"
unreal_editor="${UNREAL_EDITOR:-}"

usage() {
  cat <<'EOF'
Usage: scripts/run_unreal_container_ci.sh [options]

Run AstralRT Unreal Automation in an Epic UE Linux container.

Options:
  --ue-version <5.4|5.5|5.6|5.7>
                            Unreal version defaults for image/toolchain (default: 5.7)
  --variant <slim|full>     UE image variant (default: slim)
  --image <ref>             Override container image reference
  --digest <sha256:...>     Override digest for --image or selected variant
  --skip-pull               Do not pull the image before running
  --pull-timeout <duration> Bound manifest/pull waits (default: 120s)
  --install-cmake           Install cmake inside the temporary container if missing
  --skip-native-build       Do not rebuild the AstralRT ThirdParty package
  --editor <path>           UnrealEditor-Cmd path inside the container
  --filter <pattern>        Automation filter (default: AstralRT)
  -h, --help                Show this help

Environment:
  CONTAINER_ENGINE          Container CLI (default: docker)
  ASTRAL_UNREAL_PULL_TIMEOUT
                            Manifest and pull timeout, passed to timeout(1)
                            (default: 120)
  ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS
                            Backward-compatible alias for ASTRAL_UNREAL_PULL_TIMEOUT
  ASTRAL_UNREAL_REQUIRED_CLANG_VERSION
                            Override required clang version for the selected UE version
  ASTRAL_UNREAL_REQUIRED_LINUX_SDK_TOOLCHAIN
                            Override Linux SDK toolchain marker for the selected UE version
  ASTRAL_UNREAL_REQUIRED_LINUX_SDK_CLANG
                            Override Linux SDK clang marker for the selected UE version
  ASTRAL_UNREAL_TEST_EMBED_MODEL
                            Absolute model path inside the container for the gated real embedding Automation probe
  ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING
                            Set to 1 with ASTRAL_UNREAL_TEST_EMBED_MODEL to fail hard when the probe cannot run
  ASTRAL_ENGINE_MEMORY_PERF
                            Set to 1 to run the 100k x 384 E5M2 wrapper test
  ASTRAL_ENGINE_MEMORY_MAX_P50_US
                            Maximum accepted wrapper p50 in microseconds (default: 1000)
  ASTRAL_UNREAL_TEST_MODEL
                            Absolute model path inside the container for the gated real generation Automation smoke
  ASTRAL_UNREAL_REQUIRE_REAL_GENERATION
                            Set to 1 with ASTRAL_UNREAL_TEST_MODEL to fail hard when the smoke cannot run
  ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE
                            Set to 1 with ASTRAL_UNREAL_TEST_MODEL to fail hard when the real lifecycle smoke cannot run
  UNREAL_EDITOR             Same as --editor
  UNREAL_TEST_FILTER        Same as --filter

The default images are private Epic GHCR packages. Authenticate Docker with an
Epic-linked GitHub account before running this script.
EOF
}

image_tag_for_version() {
  local version="$1"
  local selected_variant="$2"

  case "${version}:${selected_variant}" in
    5.4:slim) printf 'ghcr.io/epicgames/unreal-engine:dev-slim-5.4.4\n' ;;
    5.4:full) printf 'ghcr.io/epicgames/unreal-engine:dev-5.4.4\n' ;;
    5.5:slim) printf 'ghcr.io/epicgames/unreal-engine:dev-slim-5.5.4\n' ;;
    5.5:full) printf 'ghcr.io/epicgames/unreal-engine:dev-5.5.4\n' ;;
    5.6:slim) printf 'ghcr.io/epicgames/unreal-engine:dev-slim-5.6.1\n' ;;
    5.6:full) printf 'ghcr.io/epicgames/unreal-engine:dev-5.6.1\n' ;;
    5.7:slim) printf 'ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4\n' ;;
    5.7:full) printf 'ghcr.io/epicgames/unreal-engine:dev-5.7.4\n' ;;
    *)
      echo "Unsupported --ue-version/--variant combination: ${version}/${selected_variant}" >&2
      exit 2
      ;;
  esac
}

digest_for_version() {
  local version="$1"
  local selected_variant="$2"

  case "${version}:${selected_variant}" in
    5.7:slim) printf 'sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n' ;;
    5.7:full) printf 'sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n' ;;
    *) printf '\n' ;;
  esac
}

toolchain_for_version() {
  case "$1" in
    5.4) printf '16.0.6 v22 16.0.6\n' ;;
    5.5) printf '18.1.0 v23 18.1.0\n' ;;
    5.6) printf '18.1.0 v25 18.1.0\n' ;;
    5.7) printf '20.1.8 v26 20.1.8\n' ;;
    *)
      echo "Unsupported --ue-version '$1' (expected 5.4, 5.5, 5.6, or 5.7)" >&2
      exit 2
      ;;
  esac
}

is_epic_ghcr_image() {
  [[ "$1" == ghcr.io/epicgames/unreal-engine:* || "$1" == ghcr.io/epicgames/unreal-engine@sha256:* ]]
}

has_ghcr_auth_config() {
  if [[ -n "${DOCKER_AUTH_CONFIG:-}" && "${DOCKER_AUTH_CONFIG}" == *"ghcr.io"* ]]; then
    return 0
  fi

  local docker_config_dir="${DOCKER_CONFIG:-${HOME:-}/.docker}"
  local docker_config="${docker_config_dir}/config.json"
  if [[ ! -f "${docker_config}" ]]; then
    return 1
  fi

  grep -Eq '"ghcr[.]io"|"credsStore"|"credHelpers"' "${docker_config}"
}

fail_epic_ghcr_auth() {
  cat >&2 <<EOF
Epic Unreal container access is not configured for ${container_engine}.
Authenticate with an Epic-linked GitHub account that can read ghcr.io/epicgames/unreal-engine, then rerun this command.
Image: ${image_ref}
EOF
}

pull_container_image() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "${pull_timeout}" "${container_engine}" pull "${image_ref}" </dev/null
  else
    "${container_engine}" pull "${image_ref}" </dev/null
  fi
}

inspect_container_manifest() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "${pull_timeout}" "${container_engine}" manifest inspect "${image_ref}" </dev/null
  else
    "${container_engine}" manifest inspect "${image_ref}" </dev/null
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ue-version)
      ue_version="${2:-}"
      shift 2
      ;;
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
    --pull-timeout)
      pull_timeout="${2:-}"
      shift 2
      ;;
    --install-cmake)
      install_cmake=1
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

case "${variant}" in
  slim|full) ;;
  *)
    echo "Unsupported --variant '${variant}' (expected slim or full)" >&2
    exit 2
    ;;
esac

if [[ -z "${image}" ]]; then
  image="$(image_tag_for_version "${ue_version}" "${variant}")"
  digest="${digest:-$(digest_for_version "${ue_version}" "${variant}")}"
fi

read -r default_clang default_sdk_toolchain default_sdk_clang <<<"$(toolchain_for_version "${ue_version}")"
required_clang_version="${required_clang_version:-${default_clang}}"
required_linux_sdk_toolchain="${required_linux_sdk_toolchain:-${default_sdk_toolchain}}"
required_linux_sdk_clang="${required_linux_sdk_clang:-${default_sdk_clang}}"

if [[ -z "${required_clang_version}" || -z "${required_linux_sdk_toolchain}" || -z "${required_linux_sdk_clang}" ]]; then
  echo "Missing Unreal toolchain requirements for UE ${ue_version}" >&2
  exit 2
fi

image_ref="${image}"
if [[ -n "${digest}" && "${image}" != *@sha256:* ]]; then
  image_ref="${image}@${digest}"
fi

if [[ "${pull_image}" -eq 1 ]]; then
  if is_epic_ghcr_image "${image_ref}" && ! has_ghcr_auth_config; then
    fail_epic_ghcr_auth
    exit 2
  fi
  if is_epic_ghcr_image "${image_ref}"; then
    echo "[unreal_container] Check image access: ${image_ref}"
    if ! inspect_container_manifest >/dev/null; then
      cat >&2 <<EOF
Unable to inspect Unreal container manifest: ${image_ref}
The configured ${container_engine} credentials cannot read ghcr.io/epicgames/unreal-engine, or the pinned image is unavailable.
Authenticate with an Epic-linked GitHub account that has Epic Unreal Engine package access, then rerun this command.
EOF
      exit 2
    fi
  fi
  echo "[unreal_container] Pull image: ${image_ref}"
  pull_status=0
  pull_container_image || pull_status=$?
  if [[ "${pull_status}" -ne 0 ]]; then
    if [[ "${pull_status}" -eq 124 ]]; then
      cat >&2 <<EOF
Timed out pulling Unreal container image after ${pull_timeout}: ${image_ref}
Pre-pull the image manually, increase --pull-timeout, or rerun with --skip-pull after the image is cached.
EOF
      exit 2
    fi
    cat >&2 <<EOF
Unable to pull Unreal container image: ${image_ref}
Authenticate ${container_engine} with an Epic-linked GitHub account that can read ghcr.io/epicgames/unreal-engine, then rerun this command.
EOF
    exit 2
  fi
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
echo "[unreal_container] UE version: ${ASTRAL_UNREAL_VERSION}"
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
    linux_sdk_text="$(sed -n "1,120p" "${linux_sdk}")"
    printf "%s\n" "${linux_sdk_text}"
    if [[ "${linux_sdk_text}" != *"${ASTRAL_UNREAL_REQUIRED_LINUX_SDK_TOOLCHAIN}"* ||
          "${linux_sdk_text}" != *"${ASTRAL_UNREAL_REQUIRED_LINUX_SDK_CLANG}"* ]]; then
      echo "Linux SDK metadata mismatch: expected ${ASTRAL_UNREAL_REQUIRED_LINUX_SDK_TOOLCHAIN} with clang ${ASTRAL_UNREAL_REQUIRED_LINUX_SDK_CLANG}" >&2
      exit 2
    fi
  else
    echo "[unreal_container] Linux SDK metadata not found under ${engine_root}" >&2
    exit 2
  fi
fi

clang_bin=""
if command -v clang >/dev/null 2>&1; then
  clang_bin="$(command -v clang)"
elif [[ -n "${engine_root}" ]]; then
  for candidate in \
    "${engine_root}/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/"*"${ASTRAL_UNREAL_REQUIRED_LINUX_SDK_TOOLCHAIN}"*"${ASTRAL_UNREAL_REQUIRED_LINUX_SDK_CLANG}"*/x86_64-unknown-linux-gnu/bin/clang; do
    if [[ -x "${candidate}" ]]; then
      clang_bin="${candidate}"
      break
    fi
  done
fi

if [[ -z "${clang_bin}" ]]; then
  echo "[unreal_container] clang not found in PATH or UE Linux SDK" >&2
  exit 2
fi

echo "[unreal_container] clang: ${clang_bin}"
clang_version_text="$("${clang_bin}" --version | sed -n "1,3p")"
printf "%s\n" "${clang_version_text}"
if [[ "${clang_version_text}" != *"${ASTRAL_UNREAL_REQUIRED_CLANG_VERSION}"* ]]; then
  echo "clang version mismatch: expected ${ASTRAL_UNREAL_REQUIRED_CLANG_VERSION}" >&2
  exit 2
fi

unreal_cmake_args=()
clangxx_bin="${clang_bin%/clang}/clang++"
ue_sdk_root=""
if [[ "${clang_bin}" == */x86_64-unknown-linux-gnu/bin/clang ]]; then
  ue_sdk_root="${clang_bin%/bin/clang}"
fi
if [[ -x "${clangxx_bin}" && -d "${ue_sdk_root}/include/c++/v1" ]]; then
  ue_c_flags="--target=x86_64-unknown-linux-gnu --sysroot=${ue_sdk_root} -fPIC"
  ue_cxx_flags="--driver-mode=g++ -nostdinc++ -isystem${ue_sdk_root}/include -isystem${ue_sdk_root}/include/c++/v1 ${ue_c_flags} -fexceptions"
  unreal_cmake_args+=(
    "-DCMAKE_C_COMPILER=${clang_bin}"
    "-DCMAKE_CXX_COMPILER=${clangxx_bin}"
    "-DCMAKE_C_FLAGS=${ue_c_flags}"
    "-DCMAKE_CXX_FLAGS=${ue_cxx_flags}"
  )
  echo "[unreal_container] Native package C++ runtime: UE libc++"
fi

if [[ "${ASTRAL_UNREAL_BUILD_NATIVE}" == "1" ]]; then
  if ! command -v cmake >/dev/null 2>&1; then
    if [[ "${ASTRAL_UNREAL_INSTALL_CMAKE}" == "1" ]]; then
      echo "[unreal_container] Install cmake in temporary container"
      apt_prefix=()
      if [[ "$(id -u)" -ne 0 ]]; then
        apt_prefix=(sudo -n)
      fi
      "${apt_prefix[@]}" apt-get update
      "${apt_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake
    else
      echo "[unreal_container] cmake not found in container; rerun with --install-cmake, or prebuild the AstralRT Unreal ThirdParty package with the UE Linux SDK and rerun with --skip-native-build." >&2
      exit 2
    fi
  fi
  rm -rf build/unreal
  cmake --preset unreal-plugin "${unreal_cmake_args[@]}"
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
  -e "ASTRAL_UNREAL_INSTALL_CMAKE=${install_cmake}" \
  -e "ASTRAL_UNREAL_IMAGE_REF=${image_ref}" \
  -e "ASTRAL_UNREAL_VERSION=${ue_version}" \
  -e "ASTRAL_UNREAL_REQUIRED_CLANG_VERSION=${required_clang_version}" \
  -e "ASTRAL_UNREAL_REQUIRED_LINUX_SDK_TOOLCHAIN=${required_linux_sdk_toolchain}" \
  -e "ASTRAL_UNREAL_REQUIRED_LINUX_SDK_CLANG=${required_linux_sdk_clang}" \
  -e "ASTRAL_UNREAL_TEST_MODEL=${ASTRAL_UNREAL_TEST_MODEL:-}" \
  -e "ASTRAL_UNREAL_REQUIRE_REAL_GENERATION=${ASTRAL_UNREAL_REQUIRE_REAL_GENERATION:-}" \
  -e "ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE=${ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE:-}" \
  -e "ASTRAL_UNREAL_TEST_EMBED_MODEL=${ASTRAL_UNREAL_TEST_EMBED_MODEL:-}" \
  -e "ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING=${ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING:-}" \
  -e "ASTRAL_ENGINE_MEMORY_PERF=${ASTRAL_ENGINE_MEMORY_PERF:-}" \
  -e "ASTRAL_ENGINE_MEMORY_MAX_P50_US=${ASTRAL_ENGINE_MEMORY_MAX_P50_US:-}" \
  -e "UNREAL_EDITOR=${unreal_editor}" \
  -e "UNREAL_TEST_FILTER=${test_filter}" \
  -v "${root_dir}:/workspace/astral" \
  -w /workspace/astral \
  "${image_ref}" \
  bash -lc "${container_script}"
