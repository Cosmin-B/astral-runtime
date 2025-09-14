#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_hf_gguf_suite.sh [options]

Downloads a curated set of GGUF repos from Hugging Face into a single output directory.

Options:
  --out <dir>     Output directory (default: tests/models/hf)
  --token <tok>   HF token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN)
  --help          Show help

Notes:
  - This is a pragmatic suite: it downloads 1-2 representative quants per repo by default.
  - To download all quants for a repo, use scripts/hf_gguf_sync.py download ... --all.
EOF
}

out_dir="tests/models/hf"
token="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out_dir="${2:-}"; shift 2 ;;
    --token) token="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

common_args=(--out "${out_dir}")
if [[ -n "${token}" ]]; then
  common_args+=(--token "${token}")
fi

echo "[hf-suite] out=${out_dir}"

# Qwen3 8B (chat/instruct): typical "go-to" quants for local inference.
python3 scripts/hf_gguf_sync.py download Qwen/Qwen3-8B-GGUF "${common_args[@]}" --max-count 2 --max-gb 12 \
  --prefer Q4_K_M --prefer Q5_K_M --prefer Q5_0

# Qwen3 8B embeddings: keep it lightweight by default.
python3 scripts/hf_gguf_sync.py download Qwen/Qwen3-Embedding-8B-GGUF "${common_args[@]}" --max-count 1 --max-gb 12 \
  --prefer Q4_K_M --prefer Q5_K_M --prefer Q5_0

# Qwen3 1.7B: many quant variants exist; pick a few representative ones.
python3 scripts/hf_gguf_sync.py download unsloth/Qwen3-1.7B-GGUF "${common_args[@]}" --max-count 4 --max-gb 3 \
  --prefer Q4 --prefer Q5 --prefer Q8 --prefer Q6

# Qwen3 32B: huge; pick the smallest-ish quant by default.
python3 scripts/hf_gguf_sync.py download bartowski/Qwen_Qwen3-32B-GGUF "${common_args[@]}" --max-count 1 --max-gb 16 \
  --prefer IQ2_XXS --prefer Q2_K

echo "[hf-suite] done"
