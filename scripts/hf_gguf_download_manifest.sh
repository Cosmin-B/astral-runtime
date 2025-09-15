#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/hf_gguf_download_manifest.sh [options]

Downloads GGUF files described by a pinned manifest JSON.
This wraps scripts/hf_gguf_sync.py for each repo entry.

Options:
  --manifest <file>   Manifest JSON (default: scripts/hf_gguf_manifest_full.json)
  --out <dir>         Output directory root (default: tests/models/hf)
  --token <tok>       HF token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN)
  --help              Show help

Notes:
  - Files are stored under <out>/<repo-with-__>/.
  - Each repo can specify mode=all and max_gb_per_file.
EOF
}

manifest="scripts/hf_gguf_manifest_full.json"
out_dir="tests/models/hf"
token="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="${2:-}"; shift 2 ;;
    --out) out_dir="${2:-}"; shift 2 ;;
    --token) token="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f "${manifest}" ]]; then
  echo "Manifest not found: ${manifest}" >&2
  exit 2
fi

mkdir -p "${out_dir}"

common_args=(--out "${out_dir}")
if [[ -n "${token}" ]]; then
  common_args+=(--token "${token}")
fi

python3 - <<'PY' "${manifest}" | while IFS=$'\t' read -r repo mode maxgb; do
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
repos = data.get("repos", [])
for r in repos:
    repo = r.get("repo", "")
    mode = r.get("mode", "all")
    maxgb = r.get("max_gb_per_file", 25)
    if not repo:
        continue
    sys.stdout.write(f"{repo}\t{mode}\t{maxgb}\n")
PY
  if [[ -z "${repo}" ]]; then
    continue
  fi

  echo "[hf-manifest] repo=${repo} mode=${mode} max_gb_per_file=${maxgb}"

  if [[ "${mode}" == "all" ]]; then
    python3 scripts/hf_gguf_sync.py download "${repo}" "${common_args[@]}" --all --max-gb "${maxgb}"
  else
    echo "[hf-manifest] unsupported mode=${mode} (expected: all)" >&2
    exit 2
  fi
done

echo "[hf-manifest] done out=${out_dir}"

