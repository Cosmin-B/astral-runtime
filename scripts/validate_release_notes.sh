#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
notes_path="${root_dir}/docs/release/RELEASE_NOTES_TEMPLATE.md"

usage() {
  cat <<'EOF'
Usage: scripts/validate_release_notes.sh [options] [file]

Validates that release notes carry the minimum release-candidate evidence.

Options:
  --path <file>  Release notes file to validate
  --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      notes_path="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      if [[ "${1:0:1}" == "-" ]]; then
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
      fi
      notes_path="$1"
      shift
      ;;
  esac
done

if [[ -z "${notes_path}" || ! -s "${notes_path}" ]]; then
  echo "[release-notes] missing or empty release notes: ${notes_path}" >&2
  exit 1
fi

require_regex() {
  local pattern="$1"
  local message="$2"
  if ! grep -Eq "${pattern}" "${notes_path}"; then
    echo "[release-notes] ${message}" >&2
    exit 1
  fi
}

require_text() {
  local text="$1"
  local message="$2"
  if ! grep -Fq "${text}" "${notes_path}"; then
    echo "[release-notes] ${message}" >&2
    exit 1
  fi
}

require_regex '^##[[:space:]]+Compatibility[[:space:]]*$' "missing Compatibility section"
require_regex '^##[[:space:]]+Artifacts[[:space:]]*$' "missing Artifacts section"
require_regex '^##[[:space:]]+Validation Evidence[[:space:]]*$' "missing Validation Evidence section"
require_regex '^##[[:space:]]+Engine Evidence[[:space:]]*$' "missing Engine Evidence section"
require_regex '^##[[:space:]]+Rollback[[:space:]]*$' "missing Rollback section"
require_regex '^##[[:space:]]+Known Gaps[[:space:]]*$' "missing Known Gaps section"

require_text "dependency-manifest.json" "release notes must name dependency-manifest.json"
require_text "abi-layout.json" "release notes must name abi-layout.json"
require_text "checksums.sha256" "release notes must name checksums.sha256"
require_regex 'checksums\.sha256\.(asc|minisig)|signed-waiver' "release notes must name the checksum signature file or waiver"
require_regex 'fingerprint|public verification key|Signing key' "release notes must identify the public verification key or signer fingerprint"
require_regex 'validate_release_artifacts\.sh[^[:cntrl:]]*--require-signature|gpg[[:space:]]+--verify|minisign[[:space:]]+-Vm' "release notes must include a checksum signature verification command"
require_regex 'Unity' "release notes must include Unity evidence"
require_regex 'Unreal' "release notes must include Unreal evidence"
require_regex 'CUDA' "release notes must include CUDA evidence"
require_regex 'MTMD|vision/audio' "release notes must include multimodal evidence"
require_regex 'Previous known-good artifact' "release notes must identify rollback artifact"
require_regex 'issue tracker issue|waiver' "known gaps must carry a waiver or issue tracker issue"

echo "[release-notes] OK: ${notes_path}"
