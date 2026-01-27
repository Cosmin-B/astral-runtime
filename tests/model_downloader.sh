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
  --validate-only       Validate an existing local preset file without downloading
  --print-path          Print the resolved local path for a preset
  --info                Print resolved preset metadata as JSON
  --status              Print existing local file state as JSON
  --list-presets        Print available presets
  --list-package        Print presets marked for packaged samples
  --list-type <type>    Filter --list-presets by all, text, or embedding
  --list-format <fmt>   Print --list-presets as text or json
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

print_path=0
print_info=0
print_status=0
list_presets=0
list_package=0
preset_name=""
output_dir="tests/models"
list_type="all"
list_format="text"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset_name="${2:-}"; args+=(--preset "${preset_name}"); shift 2 ;;
    --dir) output_dir="${2:-}"; args+=(--dir "${output_dir}"); shift 2 ;;
    --dry-run) args+=(--dry-run); shift ;;
    --validate-only) args+=(--validate-only); shift ;;
    --print-path) print_path=1; shift ;;
    --info) print_info=1; shift ;;
    --status) print_status=1; shift ;;
    --list-type) list_type="${2:-}"; shift 2 ;;
    --list-format) list_format="${2:-}"; shift 2 ;;
    --token) args+=(--token "${2:-}"); shift 2 ;;
    --url) args+=(--url "${2:-}"); shift 2 ;;
    --file) args+=(--file "${2:-}"); shift 2 ;;
    --min-bytes) args+=(--min-bytes "${2:-}"); shift 2 ;;
    --sha256) args+=(--sha256 "${2:-}"); shift 2 ;;
    --hf-repo) args+=(--hf-repo "${2:-}"); shift 2 ;;
    --hf-file) args+=(--hf-file "${2:-}"); shift 2 ;;
    --hf-rev|--hf-revision) args+=(--hf-revision "${2:-}"); shift 2 ;;
    --list-presets)
      list_presets=1
      shift
      ;;
    --list-package)
      list_presets=1
      list_package=1
      shift
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

if [[ "${list_presets}" -eq 1 ]]; then
  list_args=(list --type "${list_type}" --format "${list_format}" --dir "${output_dir}")
  if [[ "${list_package}" -eq 1 ]]; then
    list_args+=(--package)
  fi
  exec python3 "${tool}" "${list_args[@]}"
fi

if [[ "${print_path}" -eq 1 ]]; then
  if [[ -z "${preset_name}" ]]; then
    echo "--print-path requires --preset" >&2
    exit "${exit_usage}"
  fi
  exec python3 "${tool}" path "${preset_name}" --dir "${output_dir}"
fi

if [[ "${print_info}" -eq 1 ]]; then
  if [[ -z "${preset_name}" ]]; then
    echo "--info requires --preset" >&2
    exit "${exit_usage}"
  fi
  exec python3 "${tool}" info "${preset_name}" --dir "${output_dir}"
fi

if [[ "${print_status}" -eq 1 ]]; then
  if [[ -z "${preset_name}" ]]; then
    echo "--status requires --preset" >&2
    exit "${exit_usage}"
  fi
  exec python3 "${tool}" status "${preset_name}" --dir "${output_dir}"
fi

exec python3 "${tool}" download "${args[@]}"
