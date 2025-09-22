#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-${root_dir}/dist}"

mkdir -p "${out_dir}"

manifest="${out_dir}/dependency-manifest.json"
checksums="${out_dir}/checksums.sha256"

version="$(sed -n 's/^project(AstralRT VERSION \([0-9][0-9.]*\).*/\1/p' "${root_dir}/CMakeLists.txt" | head -n 1)"
if [[ -z "${version}" ]]; then
  echo "Failed to detect Astral version from CMakeLists.txt" >&2
  exit 1
fi

git_commit="$(git -C "${root_dir}" rev-parse HEAD 2>/dev/null || echo unknown)"
git_dirty="false"
if ! git -C "${root_dir}" diff --quiet --ignore-submodules -- || ! git -C "${root_dir}" diff --cached --quiet --ignore-submodules --; then
  git_dirty="true"
fi

submodule_sha() {
  local path="$1"
  git -C "${root_dir}/${path}" rev-parse HEAD 2>/dev/null || echo unknown
}

submodule_desc() {
  local path="$1"
  git -C "${root_dir}/${path}" describe --tags --always --dirty 2>/dev/null || echo unknown
}

unity_package_version="$(sed -n 's/^[[:space:]]*"version": *"\([^"]*\)".*/\1/p' "${root_dir}/plugins/unity/package.json" | head -n 1)"
unity_version="$(sed -n 's/^[[:space:]]*"unity": *"\([^"]*\)".*/\1/p' "${root_dir}/plugins/unity/package.json" | head -n 1)"
unreal_plugin_version="$(sed -n 's/^[[:space:]]*"VersionName": *"\([^"]*\)".*/\1/p' "${root_dir}/plugins/unreal/AstralRT/AstralRT.uplugin" | head -n 1)"

cat > "${manifest}" <<EOF
{
  "schema": "astral.release.dependency-manifest.v1",
  "astral": {
    "version": "${version}",
    "git_commit": "${git_commit}",
    "git_dirty": ${git_dirty}
  },
  "submodules": [
    {
      "name": "llama.cpp",
      "path": "external/llama.cpp",
      "description": "$(submodule_desc external/llama.cpp)",
      "commit": "$(submodule_sha external/llama.cpp)",
      "license": "MIT"
    },
    {
      "name": "tracy",
      "path": "external/tracy",
      "description": "$(submodule_desc external/tracy)",
      "commit": "$(submodule_sha external/tracy)",
      "license": "BSD-3-Clause"
    }
  ],
  "engine_packages": {
    "unity": {
      "path": "plugins/unity/package.json",
      "version": "${unity_package_version:-unknown}",
      "minimum_unity": "${unity_version:-unknown}"
    },
    "unreal": {
      "path": "plugins/unreal/AstralRT/AstralRT.uplugin",
      "version": "${unreal_plugin_version:-unknown}",
      "compatibility_floor": "5.4",
      "production_target": "5.7"
    }
  }
}
EOF

if command -v sha256sum >/dev/null 2>&1; then
  hash_file() { sha256sum "$1"; }
elif command -v shasum >/dev/null 2>&1; then
  hash_file() { shasum -a 256 "$1"; }
else
  echo "No SHA-256 tool found (sha256sum or shasum)." >&2
  exit 1
fi

: > "${checksums}"
while IFS= read -r file; do
  base="$(basename "${file}")"
  if [[ "${base}" == "$(basename "${manifest}")" || "${base}" == "$(basename "${checksums}")" ]]; then
    continue
  fi
  (cd "${out_dir}" && hash_file "${base}") >> "${checksums}"
done < <(find "${out_dir}" -maxdepth 1 -type f | sort)

echo "[release_metadata] Manifest: ${manifest}"
echo "[release_metadata] Checksums: ${checksums}"
