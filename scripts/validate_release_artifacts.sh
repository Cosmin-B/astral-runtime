#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist_dir="${root_dir}/dist"
expect_unity=0
expect_unreal=0
require_signature=0

usage() {
  cat <<'EOF'
Usage: scripts/validate_release_artifacts.sh [options]

Validates the packaged release artifact set after package_release.sh and
generate_release_metadata.sh have run.

Options:
  --dist <dir>          Release artifact directory (default: ./dist)
  --expect-unity       Require a Unity plugin zip
  --expect-unreal      Require an Unreal plugin zip
  --require-signature  Require checksums.sha256.asc or checksums.sha256.minisig
  --help               Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dist) dist_dir="${2:-}"; shift 2 ;;
    --expect-unity) expect_unity=1; shift ;;
    --expect-unreal) expect_unreal=1; shift ;;
    --require-signature) require_signature=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "${dist_dir}" ]]; then
  echo "[release-artifacts] missing dist directory: ${dist_dir}" >&2
  exit 1
fi

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "[release-artifacts] missing or empty file: ${path}" >&2
    exit 1
  fi
}

require_checksum_entry() {
  local name="$1"
  if ! grep -Eq "[[:space:]]${name//./\\.}$" "${dist_dir}/checksums.sha256"; then
    echo "[release-artifacts] checksums.sha256 does not cover ${name}" >&2
    exit 1
  fi
}

first_match() {
  local pattern="$1"
  find "${dist_dir}" -maxdepth 1 -type f -name "${pattern}" -printf '%f\n' | sort | head -n 1
}

require_file "${dist_dir}/abi-layout.json"
require_file "${dist_dir}/dependency-manifest.json"
require_file "${dist_dir}/checksums.sha256"

core_zip="$(first_match 'astral-*-linux-*.zip')"
if [[ -z "${core_zip}" ]]; then
  core_zip="$(first_match 'astral-*-mac-*.zip')"
fi
if [[ -z "${core_zip}" ]]; then
  core_zip="$(first_match 'astral-*-win-*.zip')"
fi
if [[ -z "${core_zip}" ]]; then
  echo "[release-artifacts] missing core runtime zip" >&2
  exit 1
fi

require_checksum_entry "abi-layout.json"
require_checksum_entry "dependency-manifest.json"
require_checksum_entry "${core_zip}"

if [[ "${expect_unity}" -eq 1 ]]; then
  unity_zip="$(first_match 'astral-*-unity-plugin-*.zip')"
  if [[ -z "${unity_zip}" ]]; then
    echo "[release-artifacts] missing Unity plugin zip" >&2
    exit 1
  fi
  require_checksum_entry "${unity_zip}"
fi

if [[ "${expect_unreal}" -eq 1 ]]; then
  unreal_zip="$(first_match 'astral-*-unreal-plugin-*.zip')"
  if [[ -z "${unreal_zip}" ]]; then
    echo "[release-artifacts] missing Unreal plugin zip" >&2
    exit 1
  fi
  require_checksum_entry "${unreal_zip}"
fi

if [[ "${require_signature}" -eq 1 ]]; then
  if [[ ! -s "${dist_dir}/checksums.sha256.asc" && ! -s "${dist_dir}/checksums.sha256.minisig" ]]; then
    echo "[release-artifacts] missing checksum signature" >&2
    exit 1
  fi
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${dist_dir}" && sha256sum -c checksums.sha256 >/dev/null)
elif command -v shasum >/dev/null 2>&1; then
  (cd "${dist_dir}" && shasum -a 256 -c checksums.sha256 >/dev/null)
else
  echo "[release-artifacts] no SHA-256 verifier found (sha256sum or shasum)" >&2
  exit 1
fi

echo "[release-artifacts] OK: ${dist_dir}"
