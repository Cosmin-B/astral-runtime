#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/sign_release_artifacts.sh [options]

Signs the release checksum file. The checksum file covers each packaged artifact,
so the detached signature authenticates the artifact set for a release.

Options:
  --out-dir <path>       Release output directory (default: ./dist)
  --checksums <path>     Checksum file to sign (default: <out-dir>/checksums.sha256)
  --tool <gpg|minisign>  Signing backend (default: auto)
  --key <id-or-path>     GPG key id or minisign secret key path
  --dry-run              Validate inputs and print the signing command without signing
  --help                 Show this help

Environment:
  ASTRAL_RELEASE_SIGN_TOOL  gpg or minisign
  ASTRAL_RELEASE_SIGN_KEY   GPG key id or minisign secret key path
  ASTRAL_RELEASE_GPG_PASSPHRASE_FILE  File containing the GPG key passphrase
EOF
}

out_dir="${root_dir}/dist"
checksums=""
tool="${ASTRAL_RELEASE_SIGN_TOOL:-auto}"
key="${ASTRAL_RELEASE_SIGN_KEY:-}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) out_dir="${2:-}"; shift 2 ;;
    --checksums) checksums="${2:-}"; shift 2 ;;
    --tool) tool="${2:-}"; shift 2 ;;
    --key) key="${2:-}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${checksums}" ]]; then
  checksums="${out_dir}/checksums.sha256"
fi

if [[ ! -s "${checksums}" ]]; then
  echo "[sign_release] checksum file missing or empty: ${checksums}" >&2
  exit 2
fi

verify_checksum_entries() {
  local checksum_dir
  checksum_dir="$(cd "$(dirname "${checksums}")" && pwd)"

  while read -r _hash file rest; do
    if [[ -z "${file:-}" ]]; then
      continue
    fi
    if [[ -n "${rest:-}" ]]; then
      echo "[sign_release] checksum entry contains spaces or extra fields: ${file} ${rest}" >&2
      exit 2
    fi
    if [[ ! -f "${checksum_dir}/${file}" ]]; then
      echo "[sign_release] checksum entry is missing artifact: ${file}" >&2
      exit 2
    fi
  done < "${checksums}"
}

resolve_tool() {
  if [[ "${tool}" != "auto" ]]; then
    printf '%s\n' "${tool}"
    return
  fi
  if command -v gpg >/dev/null 2>&1; then
    printf '%s\n' "gpg"
    return
  fi
  if command -v minisign >/dev/null 2>&1; then
    printf '%s\n' "minisign"
    return
  fi
  echo "[sign_release] no signing tool found (install gpg or minisign)" >&2
  exit 2
}

verify_checksum_entries
tool="$(resolve_tool)"

case "${tool}" in
  gpg)
    if ! command -v gpg >/dev/null 2>&1; then
      echo "[sign_release] gpg not found" >&2
      exit 2
    fi
    signature="${checksums}.asc"
    cmd=(gpg --batch --yes --armor --detach-sign --output "${signature}")
    if [[ -n "${ASTRAL_RELEASE_GPG_PASSPHRASE_FILE:-}" ]]; then
      cmd+=(--pinentry-mode loopback --passphrase-file "${ASTRAL_RELEASE_GPG_PASSPHRASE_FILE}")
    fi
    if [[ -n "${key}" ]]; then
      cmd+=(--local-user "${key}")
    fi
    cmd+=("${checksums}")
    ;;
  minisign)
    if ! command -v minisign >/dev/null 2>&1; then
      echo "[sign_release] minisign not found" >&2
      exit 2
    fi
    if [[ -z "${key}" ]]; then
      echo "[sign_release] minisign requires --key <secret-key-file>" >&2
      exit 2
    fi
    signature="${checksums}.minisig"
    cmd=(minisign -Sm "${checksums}" -s "${key}" -x "${signature}")
    ;;
  *)
    echo "[sign_release] unsupported signing tool: ${tool}" >&2
    exit 2
    ;;
esac

if [[ "${dry_run}" -eq 1 ]]; then
  printf '[sign_release] dry run:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"

if [[ ! -s "${signature}" ]]; then
  echo "[sign_release] signature was not written: ${signature}" >&2
  exit 1
fi

echo "[sign_release] Signature: ${signature}"
