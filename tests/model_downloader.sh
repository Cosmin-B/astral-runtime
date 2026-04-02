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
  --validate-metadata   Validate GGUF metadata against preset fields
  --inspect-metadata    Print GGUF metadata summary for a local preset
  --print-path          Print the resolved local path for a preset
  --info                Print resolved preset metadata as JSON
  --status              Print existing local file state as JSON
  --status-all          Print local file state for selected presets as JSON
  --status-summary      Print aggregate local file state for selected presets as JSON
  --download-plan       Print commands for selected presets that are not ready
  --status-format <fmt> Print --status/--status-all as json or text
  --status-only <state> Filter --status-all by any, ready, missing, partial, invalid, or not-ready
  --list-presets        Print available presets
  --list-package        Print presets marked for packaged samples
  --list-unreal-matrix  Print presets marked for Unreal sample matrix runs
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

Downloads resume incomplete .part files when the server supports ranges.
Progress is printed to stderr during downloads.
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
print_status_all=0
print_status_summary=0
print_download_plan=0
inspect_metadata=0
list_presets=0
list_package=0
list_unreal_matrix=0
preset_name=""
output_dir="tests/models"
list_type="all"
list_format="text"
status_format="json"
status_only="any"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset_name="${2:-}"; args+=(--preset "${preset_name}"); shift 2 ;;
    --dir) output_dir="${2:-}"; args+=(--dir "${output_dir}"); shift 2 ;;
    --dry-run) args+=(--dry-run); shift ;;
    --validate-only) args+=(--validate-only); shift ;;
    --validate-metadata) args+=(--validate-metadata); shift ;;
    --inspect-metadata) inspect_metadata=1; shift ;;
    --print-path) print_path=1; shift ;;
    --info) print_info=1; shift ;;
    --status) print_status=1; shift ;;
    --status-all) print_status_all=1; shift ;;
    --status-summary) print_status_summary=1; shift ;;
    --download-plan) print_download_plan=1; shift ;;
    --status-format) status_format="${2:-}"; shift 2 ;;
    --status-only) status_only="${2:-}"; shift 2 ;;
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
    --list-unreal-matrix)
      list_presets=1
      list_unreal_matrix=1
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

if [[ "${list_presets}" -eq 1 && "${print_status_all}" -eq 0 && "${print_status_summary}" -eq 0 && "${print_download_plan}" -eq 0 ]]; then
  list_args=(list --type "${list_type}" --format "${list_format}" --dir "${output_dir}")
  if [[ "${list_package}" -eq 1 ]]; then
    list_args+=(--package)
  fi
  if [[ "${list_unreal_matrix}" -eq 1 ]]; then
    list_args+=(--unreal-matrix)
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

if [[ "${inspect_metadata}" -eq 1 ]]; then
  if [[ -z "${preset_name}" ]]; then
    echo "--inspect-metadata requires --preset" >&2
    exit "${exit_usage}"
  fi
  exec python3 "${tool}" inspect "${preset_name}" --dir "${output_dir}" --validate
fi

if [[ "${print_status}" -eq 1 ]]; then
  if [[ -z "${preset_name}" ]]; then
    echo "--status requires --preset" >&2
    exit "${exit_usage}"
  fi
  exec python3 "${tool}" status "${preset_name}" --dir "${output_dir}" --format "${status_format}"
fi

if [[ "${print_status_all}" -eq 1 ]]; then
  status_args=(status-all --type "${list_type}" --dir "${output_dir}" --format "${status_format}" --only "${status_only}")
  if [[ "${list_package}" -eq 1 ]]; then
    status_args+=(--package)
  fi
  if [[ "${list_unreal_matrix}" -eq 1 ]]; then
    status_args+=(--unreal-matrix)
  fi
  exec python3 "${tool}" "${status_args[@]}"
fi

if [[ "${print_status_summary}" -eq 1 ]]; then
  summary_args=(status-summary --type "${list_type}" --dir "${output_dir}" --format "${status_format}")
  if [[ "${list_package}" -eq 1 ]]; then
    summary_args+=(--package)
  fi
  if [[ "${list_unreal_matrix}" -eq 1 ]]; then
    summary_args+=(--unreal-matrix)
  fi
  exec python3 "${tool}" "${summary_args[@]}"
fi

if [[ "${print_download_plan}" -eq 1 ]]; then
  plan_args=(download-plan --type "${list_type}" --dir "${output_dir}" --format "${status_format}")
  if [[ "${list_package}" -eq 1 ]]; then
    plan_args+=(--package)
  fi
  if [[ "${list_unreal_matrix}" -eq 1 ]]; then
    plan_args+=(--unreal-matrix)
  fi
  exec python3 "${tool}" "${plan_args[@]}"
fi

exec python3 "${tool}" download "${args[@]}"
