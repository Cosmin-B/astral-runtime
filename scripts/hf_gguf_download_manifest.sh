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
  - Each repo can specify mode=all, revision, max_gb_per_file, and include regexes.
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

python3 - <<'PY' "${manifest}" | while IFS=$'\t' read -r repo mode revision maxgb includes; do
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
repos = data.get("repos", [])
for r in repos:
    repo = r.get("repo", "")
    mode = r.get("mode", "all")
    revision = r.get("revision", "main")
    maxgb = r.get("max_gb_per_file", 25)
    includes = r.get("include", [])
    if isinstance(includes, str):
        includes = [includes]
    if not repo:
        continue
    sys.stdout.write(f"{repo}\t{mode}\t{revision}\t{maxgb}\t{json.dumps(includes, separators=(',', ':'))}\n")
PY
  if [[ -z "${repo}" ]]; then
    continue
  fi

  echo "[hf-manifest] repo=${repo} revision=${revision} mode=${mode} max_gb_per_file=${maxgb}"

  include_args=()
  while IFS= read -r include_re; do
    [[ -n "${include_re}" ]] || continue
    include_args+=(--include "${include_re}")
  done < <(python3 -c 'import json,sys; [print(x) for x in json.loads(sys.argv[1])]' "${includes}")

  if [[ "${mode}" == "all" ]]; then
    python3 scripts/hf_gguf_sync.py download "${repo}" "${common_args[@]}" --revision "${revision}" --all --max-gb "${maxgb}" "${include_args[@]}"
  else
    echo "[hf-manifest] unsupported mode=${mode} (expected: all)" >&2
    exit 2
  fi
done

echo "[hf-manifest] done out=${out_dir}"
