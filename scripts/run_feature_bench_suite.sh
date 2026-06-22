#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_feature_bench_suite.sh [options]

Runs the "features" benchmark over a set of GGUF models (one by one) and writes a single log file.

Options:
  --preset <name>       CMake preset / build dir to use (default: dev)
  --backend <cpu|cuda>  Backend to benchmark (default: cpu)
  --models-dir <dir>    Directory to scan for *.gguf, or preset download dir
  --out <file>          Output log file (required)
  --gpu-layers <N>      GPU layers when backend=cuda (default: 16)
  --iters <N>           ASTRAL_BENCH_FEATURE_ITERS (default: 200)
  --tokens <N>          ASTRAL_BENCH_FEATURE_TOKENS (default: 64)
  --tokenize-only       Run only tokenization markers for each model
  --model-presets <csv> Run named presets from scripts/model_presets.json
  --list-type <type>    Select manifest presets by type: all, text, embedding
  --list-package        Select manifest presets marked for packaged samples
  --list-unreal-matrix  Select manifest presets marked for Unreal matrix runs
  --skip-missing        Skip manifest-selected files that are not present
  --filter <regex>      Only run models whose filename matches this regex
  --help                Show help

Examples:
  scripts/run_feature_bench_suite.sh --models-dir tests/models/hf --out benchmarks/results/local-hf-cpu.txt
  scripts/run_feature_bench_suite.sh --preset dev-cuda --backend cuda --gpu-layers 32 --models-dir tests/models/hf --out benchmarks/results/hetzner-hf-cuda.txt
  scripts/run_feature_bench_suite.sh --preset release-with-tests --models-dir tests/models --tokenize-only --out /tmp/tokenizers.txt
  scripts/run_feature_bench_suite.sh --preset release-with-tests --models-dir tests/models --model-presets qwen3-0.6b-q8,qwen3-embed-0.6b-q8 --tokenize-only --out /tmp/tokenizers.txt
EOF
}

preset="dev"
backend="cpu"
models_dir="tests/models"
out_file=""
gpu_layers="16"
iters="200"
tokens="64"
filter_re=""
tokenize_only="0"
model_presets_csv=""
list_type=""
list_package="0"
list_unreal_matrix="0"
skip_missing="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --backend) backend="${2:-}"; shift 2 ;;
    --models-dir) models_dir="${2:-}"; shift 2 ;;
    --out) out_file="${2:-}"; shift 2 ;;
    --gpu-layers) gpu_layers="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --tokens) tokens="${2:-}"; shift 2 ;;
    --tokenize-only) tokenize_only="1"; shift ;;
    --model-presets) model_presets_csv="${2:-}"; shift 2 ;;
    --list-type) list_type="${2:-}"; shift 2 ;;
    --list-package) list_package="1"; shift ;;
    --list-unreal-matrix) list_unreal_matrix="1"; shift ;;
    --skip-missing) skip_missing="1"; shift ;;
    --filter) filter_re="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${out_file}" ]]; then
  echo "Missing required arg: --out" >&2
  usage
  exit 2
fi

build_dir="build/${preset}"
case "${preset}" in
  release-with-tests) build_dir="build/release-test" ;;
  release) build_dir="build/release" ;;
  dev) build_dir="build/dev" ;;
  dev-prof) build_dir="build/dev-prof" ;;
  release-prof) build_dir="build/release-prof" ;;
  unity-plugin) build_dir="build/unity" ;;
  unreal-plugin) build_dir="build/unreal" ;;
esac

bench_bin="${build_dir}/benchmarks/astral_benchmarks"
if [[ ! -x "${bench_bin}" ]]; then
  echo "Benchmark binary not found: ${bench_bin}" >&2
  echo "Build it with: cmake --preset ${preset} && cmake --build --preset ${preset}" >&2
  exit 2
fi

mkdir -p "$(dirname "${out_file}")"
{
  echo "# Astral feature bench suite"
  echo "# date: $(date -Iseconds)"
  echo "# preset: ${preset}"
  echo "# backend: ${backend}"
  echo "# models_dir: ${models_dir}"
  echo "# gpu_layers: ${gpu_layers}"
  echo "# iters: ${iters}"
  echo "# tokens: ${tokens}"
  echo "# tokenize_only: ${tokenize_only}"
  echo "# model_presets: ${model_presets_csv}"
  echo "# list_type: ${list_type}"
  echo "# list_package: ${list_package}"
  echo "# list_unreal_matrix: ${list_unreal_matrix}"
  echo "# skip_missing: ${skip_missing}"
  echo
} > "${out_file}"

models=()
if [[ -n "${model_presets_csv}" ]]; then
  IFS=',' read -r -a preset_values <<< "${model_presets_csv}"
  for model_preset in "${preset_values[@]}"; do
    models+=("$(python3 scripts/model_preset_tool.py path "${model_preset}" --dir "${models_dir}")")
  done
elif [[ -n "${list_type}" || "${list_package}" == "1" || "${list_unreal_matrix}" == "1" ]]; then
  list_args=(list --format text --dir "${models_dir}")
  if [[ -n "${list_type}" ]]; then
    list_args+=(--type "${list_type}")
  fi
  if [[ "${list_package}" == "1" ]]; then
    list_args+=(--package)
  fi
  if [[ "${list_unreal_matrix}" == "1" ]]; then
    list_args+=(--unreal-matrix)
  fi
  while IFS=$'\t' read -r model_preset _; do
    if [[ -n "${model_preset}" ]]; then
      models+=("$(python3 scripts/model_preset_tool.py path "${model_preset}" --dir "${models_dir}")")
    fi
  done < <(python3 scripts/model_preset_tool.py "${list_args[@]}")
else
  mapfile -t models < <(find "${models_dir}" -type f -name "*.gguf" | sort)
fi

if [[ ${#models[@]} -eq 0 ]]; then
  echo "No .gguf files found under: ${models_dir}" >&2
  exit 0
fi

ran=0
for m in "${models[@]}"; do
  base="$(basename "${m}")"
  if [[ ! -f "${m}" ]]; then
    if [[ "${skip_missing}" == "1" ]]; then
      {
        echo
        echo "## skipped missing model: ${m}"
      } >> "${out_file}"
      continue
    fi
    echo "Model file not found: ${m}" >&2
    echo "Download it with: ./tests/model_downloader.sh --dir ${models_dir} --preset <name>" >&2
    exit 2
  fi
  if [[ -n "${filter_re}" ]]; then
    if ! echo "${base}" | rg -q "${filter_re}"; then
      continue
    fi
  fi

  {
    echo
    echo "## model: ${m}"
    echo
  } >> "${out_file}"

  ASTRAL_BENCH_MODEL="${m}" \
  ASTRAL_BENCH_FEATURE_BACKEND="${backend}" \
  ASTRAL_BENCH_GPU_LAYERS="${gpu_layers}" \
  ASTRAL_BENCH_FEATURE_ITERS="${iters}" \
  ASTRAL_BENCH_FEATURE_TOKENS="${tokens}" \
  ASTRAL_BENCH_TOKENIZE_ONLY="${tokenize_only}" \
  "${bench_bin}" --only features >> "${out_file}" 2>&1

  ran=$((ran+1))
done

echo "[suite] done models=${ran} out=${out_file}"
