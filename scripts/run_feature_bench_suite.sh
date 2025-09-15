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
  --models-dir <dir>    Directory to scan for *.gguf (required)
  --out <file>          Output log file (required)
  --gpu-layers <N>      GPU layers when backend=cuda (default: 16)
  --iters <N>           ASTRAL_BENCH_FEATURE_ITERS (default: 200)
  --tokens <N>          ASTRAL_BENCH_FEATURE_TOKENS (default: 64)
  --filter <regex>      Only run models whose filename matches this regex
  --help                Show help

Examples:
  scripts/run_feature_bench_suite.sh --models-dir tests/models/hf --out benchmarks/results/local-hf-cpu.txt
  scripts/run_feature_bench_suite.sh --preset dev-cuda --backend cuda --gpu-layers 32 --models-dir tests/models/hf --out benchmarks/results/hetzner-hf-cuda.txt
EOF
}

preset="dev"
backend="cpu"
models_dir=""
out_file=""
gpu_layers="16"
iters="200"
tokens="64"
filter_re=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --backend) backend="${2:-}"; shift 2 ;;
    --models-dir) models_dir="${2:-}"; shift 2 ;;
    --out) out_file="${2:-}"; shift 2 ;;
    --gpu-layers) gpu_layers="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --tokens) tokens="${2:-}"; shift 2 ;;
    --filter) filter_re="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${models_dir}" || -z "${out_file}" ]]; then
  echo "Missing required args: --models-dir and --out" >&2
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
  echo
} > "${out_file}"

mapfile -t models < <(find "${models_dir}" -type f -name "*.gguf" | sort)
if [[ ${#models[@]} -eq 0 ]]; then
  echo "No .gguf files found under: ${models_dir}" >&2
  exit 0
fi

ran=0
for m in "${models[@]}"; do
  base="$(basename "${m}")"
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
  "${bench_bin}" --only features >> "${out_file}" 2>&1

  ran=$((ran+1))
done

echo "[suite] done models=${ran} out=${out_file}"
