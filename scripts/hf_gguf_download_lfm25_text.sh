#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/hf_gguf_download_lfm25_text.sh [options]

Downloads the LFM2.5 *text* GGUF set (Base/Instruct/JP/Thinking) into a directory.

Options:
  --out <dir>     Output directory root (default: tests/models/hf-lfm25)
  --token <tok>   HF token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN)
  --help          Show help

Notes:
  - This manifest intentionally excludes LFM2.5-VL and LFM2.5-Audio, which require
    additional multi-file assets (mmproj/vocoder/etc) and dedicated runtime support.
EOF
}

out_dir="tests/models/hf-lfm25"
token="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out_dir="${2:-}"; shift 2 ;;
    --token) token="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

args=(--manifest scripts/hf_gguf_manifest_lfm25_text.json --out "${out_dir}")
if [[ -n "${token}" ]]; then
  args+=(--token "${token}")
fi

./scripts/hf_gguf_download_manifest.sh "${args[@]}"

