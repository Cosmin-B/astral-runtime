#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
tool="${root_dir}/scripts/model_preset_tool.py"
exit_ok=0
exit_usage=2

usage() {
  cat <<'EOF'
Usage: tests/model_downloader.sh [options]

Options:
  --preset <name>       Download a named preset from scripts/model_presets.json
  --dir <path>          Output directory (default: tests/models)
  --dry-run             Print resolved preset, path, URL, checksum, and command
  --list-presets        Print available presets
  --token <token>       Hugging Face token, otherwise HF_TOKEN/HUGGINGFACE_HUB_TOKEN is used
  --url <url>           Custom GGUF URL; requires --file
  --file <name.gguf>    Output filename for custom URL/HF downloads
  --min-bytes <bytes>   Minimum accepted size for custom downloads
  --sha256 <hex>        Optional checksum for custom downloads
  --hf-repo <org/repo>  Custom Hugging Face repo
  --hf-file <file>      Custom Hugging Face file
  --hf-rev <revision>   Custom Hugging Face revision
  -h, --help            Show help
EOF
}

args=()
if [[ -n "${ASTRAL_MODEL_URL:-${ASTRAL_TEST_MODEL_URL:-}}" ]]; then
  args+=(--url "${ASTRAL_MODEL_URL:-${ASTRAL_TEST_MODEL_URL:-}}")
fi
if [[ -n "${ASTRAL_MODEL_FILE:-${ASTRAL_TEST_MODEL_FILE:-}}" ]]; then
  args+=(--file "${ASTRAL_MODEL_FILE:-${ASTRAL_TEST_MODEL_FILE:-}}")
fi
if [[ -n "${ASTRAL_HF_REPO:-${ASTRAL_TEST_HF_REPO:-}}" ]]; then
  args+=(--hf-repo "${ASTRAL_HF_REPO:-${ASTRAL_TEST_HF_REPO:-}}")
fi
if [[ -n "${ASTRAL_HF_FILE:-${ASTRAL_TEST_HF_FILE:-}}" ]]; then
  args+=(--hf-file "${ASTRAL_HF_FILE:-${ASTRAL_TEST_HF_FILE:-}}")
fi
if [[ -n "${ASTRAL_HF_REVISION:-${ASTRAL_TEST_HF_REVISION:-}}" ]]; then
  args+=(--hf-revision "${ASTRAL_HF_REVISION:-${ASTRAL_TEST_HF_REVISION:-}}")
fi
if [[ -n "${ASTRAL_MODEL_MIN_BYTES:-}" ]]; then
  args+=(--min-bytes "${ASTRAL_MODEL_MIN_BYTES}")
fi
if [[ -n "${ASTRAL_MODEL_SHA256:-}" ]]; then
  args+=(--sha256 "${ASTRAL_MODEL_SHA256}")
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) args+=(--preset "${2:-}"); shift 2 ;;
    --dir) args+=(--dir "${2:-}"); shift 2 ;;
    --dry-run) args+=(--dry-run); shift ;;
    --token) args+=(--token "${2:-}"); shift 2 ;;
    --url) args+=(--url "${2:-}"); shift 2 ;;
    --file) args+=(--file "${2:-}"); shift 2 ;;
    --min-bytes) args+=(--min-bytes "${2:-}"); shift 2 ;;
    --sha256) args+=(--sha256 "${2:-}"); shift 2 ;;
    --hf-repo) args+=(--hf-repo "${2:-}"); shift 2 ;;
    --hf-file) args+=(--hf-file "${2:-}"); shift 2 ;;
    --hf-rev|--hf-revision) args+=(--hf-revision "${2:-}"); shift 2 ;;
    --list-presets)
      exec python3 "${tool}" list
      ;;
    --help|-h)
      usage
      exit "${exit_ok}"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit "${exit_usage}"
      ;;
  esac
done

exec python3 "${tool}" download "${args[@]}"
