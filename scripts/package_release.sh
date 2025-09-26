#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/package_release.sh [options]

Builds + tests a preset, installs into a staging prefix, and produces zip artifacts under ./dist/.

Options:
  --preset <name>          CMake preset to build (default: release-with-tests)
  --skip-tests             Skip ctest (CI should not use this)
  --install-prefix <path>  Install staging prefix (default: ./dist/stage)
  --out-dir <path>         Output directory for zips (default: ./dist)
  --unity                  Also build+package Unity plugin and zip ./plugins/unity
  --unreal                 Also build+package Unreal plugin and zip ./plugins/unreal/AstralRT
  --sign                   Sign dist/checksums.sha256 after metadata generation
  --sign-tool <tool>       Signing backend for --sign: gpg or minisign
  --sign-key <id-or-path>  GPG key id or minisign secret key path
  --help                   Show this help

Notes:
  - Uses `cmake -E tar --format=zip` (cross-platform; works on Windows GitHub Actions via `shell: bash`).
  - Output file naming: astral-<version>-<os>-<arch>.zip
EOF
}

preset="release-with-tests"
skip_tests=0
install_prefix="${root_dir}/dist/stage"
out_dir="${root_dir}/dist"
do_unity=0
do_unreal=0
do_sign=0
sign_tool=""
sign_key=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --skip-tests) skip_tests=1; shift ;;
    --install-prefix) install_prefix="${2:-}"; shift 2 ;;
    --out-dir) out_dir="${2:-}"; shift 2 ;;
    --unity) do_unity=1; shift ;;
    --unreal) do_unreal=1; shift ;;
    --sign) do_sign=1; shift ;;
    --sign-tool) sign_tool="${2:-}"; shift 2 ;;
    --sign-key) sign_key="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

version="$(sed -n 's/^project(AstralRT VERSION \([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -n 1)"
if [[ -z "${version}" ]]; then
  # Fallback: parse the public header.
  major="$(sed -n 's/^#define ASTRAL_VERSION_MAJOR \([0-9][0-9]*\).*/\1/p' include/astral_rt.h | head -n 1)"
  minor="$(sed -n 's/^#define ASTRAL_VERSION_MINOR \([0-9][0-9]*\).*/\1/p' include/astral_rt.h | head -n 1)"
  patch="$(sed -n 's/^#define ASTRAL_VERSION_PATCH \([0-9][0-9]*\).*/\1/p' include/astral_rt.h | head -n 1)"
  if [[ -n "${major}" && -n "${minor}" && -n "${patch}" ]]; then
    version="${major}.${minor}.${patch}"
  fi
fi
if [[ -z "${version}" ]]; then
  echo "Failed to detect version from CMakeLists.txt/include/astral_rt.h" >&2
  exit 1
fi

case "${install_prefix}" in
  /*) ;;
  *) install_prefix="${root_dir}/${install_prefix}" ;;
esac

case "${out_dir}" in
  /*) ;;
  *) out_dir="${root_dir}/${out_dir}" ;;
esac

os="unknown"
arch="unknown"
case "${OSTYPE:-}" in
  linux*) os="linux" ;;
  darwin*) os="mac" ;;
  msys*|cygwin*|win32*) os="win" ;;
esac

uname_m="$(uname -m 2>/dev/null || echo unknown)"
case "${uname_m}" in
  x86_64|amd64) arch="x86_64" ;;
  aarch64|arm64) arch="arm64" ;;
esac

build_dir="${root_dir}/build/release-test"
case "${preset}" in
  release-with-tests) build_dir="${root_dir}/build/release-test" ;;
  release) build_dir="${root_dir}/build/release" ;;
  dev) build_dir="${root_dir}/build/dev" ;;
  dev-prof) build_dir="${root_dir}/build/dev-prof" ;;
  release-prof) build_dir="${root_dir}/build/release-prof" ;;
  unity-plugin) build_dir="${root_dir}/build/unity" ;;
  unreal-plugin) build_dir="${root_dir}/build/unreal" ;;
  *) build_dir="${root_dir}/build/${preset}" ;;
esac

mkdir -p "${out_dir}"
rm -rf "${install_prefix}"
mkdir -p "${install_prefix}"

echo "[package_release] Configure: ${preset}"
cmake --preset "${preset}"

echo "[package_release] Build: ${preset}"
cmake --build --preset "${preset}" -j

if [[ "${skip_tests}" -eq 0 ]]; then
  echo "[package_release] Test: ${preset}"
  ctest --preset "${preset}" -j
fi

echo "[package_release] Install to staging: ${install_prefix}"
cmake --install "${build_dir}" --prefix "${install_prefix}"
cmake -E copy "${root_dir}/LICENSE" "${root_dir}/NOTICE" "${install_prefix}/"
cmake -E make_directory "${install_prefix}/share/astral/release"
cmake -E copy_directory "${root_dir}/docs/release" "${install_prefix}/share/astral/release"

core_zip="${out_dir}/astral-${version}-${os}-${arch}.zip"
echo "[package_release] Zip core: ${core_zip}"
(
  cd "${install_prefix}"
  cmake -E tar cf "${core_zip}" --format=zip -- .
)

if [[ "${do_unity}" -eq 1 ]]; then
  echo "[package_release] Build Unity plugin"
  cmake --preset unity-plugin
  cmake --build --preset unity-plugin -j
  unity_zip="${out_dir}/astral-${version}-unity-plugin-${os}-${arch}.zip"
  echo "[package_release] Zip Unity plugin: ${unity_zip}"
  cmake -E tar cf "${unity_zip}" --format=zip -- "plugins/unity" "LICENSE" "NOTICE" "docs/release"
fi

if [[ "${do_unreal}" -eq 1 ]]; then
  echo "[package_release] Build Unreal plugin ThirdParty package"
  cmake --preset unreal-plugin
  cmake --build --preset unreal-plugin -j
  unreal_zip="${out_dir}/astral-${version}-unreal-plugin-${os}-${arch}.zip"
  echo "[package_release] Zip Unreal plugin: ${unreal_zip}"
  cmake -E tar cf "${unreal_zip}" --format=zip -- "plugins/unreal/AstralRT" "LICENSE" "NOTICE" "docs/release"
fi

echo "[package_release] Generate release metadata"
"${root_dir}/scripts/generate_abi_layout_report.sh" --out "${out_dir}/abi-layout.json"
"${root_dir}/scripts/generate_release_metadata.sh" "${out_dir}"

if [[ "${do_sign}" -eq 1 ]]; then
  sign_args=(--out-dir "${out_dir}")
  if [[ -n "${sign_tool}" ]]; then
    sign_args+=(--tool "${sign_tool}")
  fi
  if [[ -n "${sign_key}" ]]; then
    sign_args+=(--key "${sign_key}")
  fi
  echo "[package_release] Sign release checksums"
  "${root_dir}/scripts/sign_release_artifacts.sh" "${sign_args[@]}"
fi

echo "[package_release] Done. Artifacts in: ${out_dir}"
