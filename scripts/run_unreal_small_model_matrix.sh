#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

models_dir="${ASTRAL_UNREAL_SMALL_MODELS_DIR:-tests/models}"
out_root="${ASTRAL_UNREAL_SMALL_MATRIX_OUT:-build/unreal-small-model-matrix}"
platform="${ASTRAL_UNREAL_SAMPLE_PLATFORM:-Linux}"
runuat="${UNREAL_RUNUAT:-}"
engine_dir="${UNREAL_ENGINE_DIR:-}"
editor="${UNREAL_EDITOR:-}"
embedding_model="${ASTRAL_UNREAL_SAMPLE_EMBED_MODEL:-}"
sample_prompt="${ASTRAL_UNREAL_SAMPLE_PROMPT:-Say hello from Astral.}"
sample_memory_backend="${ASTRAL_UNREAL_SAMPLE_MEMORY_BACKEND:-mock}"
sample_media_backend="${ASTRAL_UNREAL_SAMPLE_MEDIA_BACKEND:-mock}"
expect_engine_version="${ASTRAL_UNREAL_SAMPLE_EXPECT_ENGINE_VERSION:-}"
model_preset_tool="${root_dir}/scripts/model_preset_tool.py"
validate_runtime=1
download_missing=0
build_native=1
dry_run=0
list_only=0
declare -a selected_models=()
declare -a passthrough_args=()
declare -a normalized_models=()
declare -a known_presets=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_unreal_small_model_matrix.sh [options]

Runs the packaged Unreal sample across downloaded small Hugging Face GGUF text
fixtures. The script keeps each sample log under a stable per-model directory.

Options:
  --models-dir <dir>       Directory containing GGUF fixtures (default: tests/models)
  --model <path>           Add one GGUF path explicitly; repeatable
  --preset <name>          Add a known small-model preset; repeatable
  --embedding-model <path> GGUF embedding fixture passed to the sample when set
  --out <dir>              Output root for generated sample/archive/logs
  --platform <name>        Unreal target platform (default: Linux)
  --runuat <path>          Forwarded to run_unreal_sample_package.sh
  --engine-dir <path>      Forwarded to run_unreal_sample_package.sh
  --editor <path>          Forwarded to run_unreal_sample_package.sh
  --skip-native-build      Reuse an already staged ThirdParty package
  --sample-prompt <text>   Prompt passed to each sample run
  --expect-engine-version <text>
                          Expected UE version substring in runtime logs
  --download-missing       Download missing known presets via tests/model_downloader.sh
  --skip-runtime-validation
                          Do not validate runtime logs after sample launches
  --dry-run                Print commands without running Unreal
  --list                   Print resolved model list and exit
  -h, --help               Show help

Known presets are loaded from scripts/model_presets.json entries marked for the
Unreal sample matrix.

When no --model or --preset is supplied, the runner auto-selects known small
fixtures that already exist under --models-dir.
USAGE
}

preset_filename() {
  python3 "${model_preset_tool}" filename "$1"
}

downloader_preset_for_file() {
  python3 "${model_preset_tool}" preset-for-file "$(basename "$1")" || true
}

slug_for_model() {
  local name
  name="$(basename "$1")"
  name="${name%.gguf}"
  printf '%s\n' "${name//[^A-Za-z0-9._-]/_}"
}

abs_under_root() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${root_dir}/$1" ;;
  esac
}

add_model_once() {
  local model="$1"
  local existing
  for existing in "${selected_models[@]}"; do
    if [[ "${existing}" == "${model}" ]]; then
      return 0
    fi
  done
  selected_models+=("${model}")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) models_dir="${2:-}"; shift 2 ;;
    --model) add_model_once "${2:-}"; shift 2 ;;
    --preset) add_model_once "${models_dir}/$(preset_filename "${2:-}")"; shift 2 ;;
    --embedding-model) embedding_model="${2:-}"; shift 2 ;;
    --out) out_root="${2:-}"; shift 2 ;;
    --platform) platform="${2:-}"; shift 2 ;;
    --runuat) runuat="${2:-}"; shift 2 ;;
    --engine-dir) engine_dir="${2:-}"; shift 2 ;;
    --editor) editor="${2:-}"; shift 2 ;;
    --skip-native-build) build_native=0; shift ;;
    --sample-prompt) sample_prompt="${2:-}"; shift 2 ;;
    --expect-engine-version) expect_engine_version="${2:-}"; shift 2 ;;
    --download-missing) download_missing=1; shift ;;
    --skip-runtime-validation) validate_runtime=0; shift ;;
    --dry-run) dry_run=1; shift ;;
    --list) list_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mapfile -t known_presets < <(python3 "${model_preset_tool}" list --unreal-matrix | awk -F $'\t' '{print $1}')
if [[ "${#known_presets[@]}" -eq 0 ]]; then
  echo "[unreal_small_matrix] no manifest presets marked for the Unreal sample matrix" >&2
  exit 2
fi

models_dir="$(abs_under_root "${models_dir}")"
out_root="$(abs_under_root "${out_root}")"
if [[ -n "${embedding_model}" ]]; then
  embedding_model="$(abs_under_root "${embedding_model}")"
fi

if [[ "${#selected_models[@]}" -eq 0 ]]; then
  for preset in "${known_presets[@]}"; do
    candidate="${models_dir}/$(preset_filename "${preset}")"
    if [[ -f "${candidate}" || "${download_missing}" -eq 1 ]]; then
      add_model_once "${candidate}"
    fi
  done
fi

if [[ "${#selected_models[@]}" -eq 0 ]]; then
  echo "[unreal_small_matrix] no small GGUF fixtures found under ${models_dir}" >&2
  echo "Download one with: ./tests/model_downloader.sh --preset qwen3-0.6b-q8" >&2
  exit 2
fi

if [[ -z "${embedding_model}" && -f "${models_dir}/Qwen3-Embedding-0.6B-Q8_0.gguf" ]]; then
  embedding_model="${models_dir}/Qwen3-Embedding-0.6B-Q8_0.gguf"
fi

for model in "${selected_models[@]}"; do
  model="$(abs_under_root "${model}")"
  if [[ ! -f "${model}" ]]; then
    downloader_preset="$(downloader_preset_for_file "${model}")"
    if [[ "${download_missing}" -eq 1 && -n "${downloader_preset}" ]]; then
      download_cmd=("${root_dir}/tests/model_downloader.sh" --preset "${downloader_preset}" --dir "${models_dir}")
      if [[ "${list_only}" -eq 0 ]]; then
        printf '[unreal_small_matrix] download:'
        printf ' %q' "${download_cmd[@]}"
        printf '\n'
      fi
      if [[ "${dry_run}" -eq 0 && "${list_only}" -eq 0 ]]; then
        "${download_cmd[@]}"
      fi
    fi
  fi
  if [[ ! -f "${model}" && "${dry_run}" -eq 0 && "${list_only}" -eq 0 ]]; then
    echo "[unreal_small_matrix] model not found: ${model}" >&2
    exit 2
  fi
  normalized_models+=("${model}")
done
selected_models=("${normalized_models[@]}")
if [[ -n "${embedding_model}" && ! -f "${embedding_model}" ]]; then
  echo "[unreal_small_matrix] embedding model not found: ${embedding_model}" >&2
  exit 2
fi

if [[ "${list_only}" -eq 1 ]]; then
  echo "[unreal_small_matrix] models:"
  printf '  %s\n' "${selected_models[@]}"
  if [[ -n "${embedding_model}" ]]; then
    echo "[unreal_small_matrix] embedding_model: ${embedding_model}"
  fi
  exit 0
fi

mkdir -p "${out_root}"

if [[ -n "${runuat}" ]]; then
  passthrough_args+=(--runuat "${runuat}")
fi
if [[ -n "${engine_dir}" ]]; then
  passthrough_args+=(--engine-dir "${engine_dir}")
fi
if [[ -n "${editor}" ]]; then
  passthrough_args+=(--editor "${editor}")
fi
if [[ "${build_native}" -eq 0 ]]; then
  passthrough_args+=(--skip-native-build)
fi
if [[ -n "${embedding_model}" ]]; then
  passthrough_args+=(--sample-embedding-model "${embedding_model}")
fi

for model in "${selected_models[@]}"; do
  slug="$(slug_for_model "${model}")"
  out_dir="${out_root}/${slug}/AstralSample"
  archive_dir="${out_root}/${slug}/archive"
  runtime_log="${out_root}/${slug}/runtime.log"
  cmd=(
    "${root_dir}/scripts/run_unreal_sample_package.sh"
    --out "${out_dir}"
    --archive-dir "${archive_dir}"
    --platform "${platform}"
    --run-sample
    --runtime-log "${runtime_log}"
    --sample-model "${model}"
    --sample-memory-backend "${sample_memory_backend}"
    --sample-media-backend "${sample_media_backend}"
    --sample-prompt "${sample_prompt}"
    "${passthrough_args[@]}"
  )

  printf '[unreal_small_matrix] run:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "${dry_run}" -eq 0 ]]; then
    "${cmd[@]}"
  fi

  if [[ "${validate_runtime}" -eq 1 ]]; then
    validate_cmd=(
      "${root_dir}/scripts/validate_unreal_sample_runtime_log.py"
      --log "${runtime_log}"
      --expect-model "$(basename "${model}")"
    )
    if [[ -n "${embedding_model}" ]]; then
      validate_cmd+=(--expect-embedding-model "$(basename "${embedding_model}")")
    fi
    if [[ -n "${expect_engine_version}" ]]; then
      validate_cmd+=(--expect-engine-version "${expect_engine_version}")
    fi

    printf '[unreal_small_matrix] validate:'
    printf ' %q' "${validate_cmd[@]}"
    printf '\n'
    if [[ "${dry_run}" -eq 0 ]]; then
      "${validate_cmd[@]}"
    fi
  fi
done

echo "[unreal_small_matrix] OK"
