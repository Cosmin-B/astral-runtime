#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_ci_bench_features.sh [options]

Runs the canonical "features" surface benchmark and writes a log file.
Intended for CI smoke + artifact capture (not a strict perf gate).

Options:
  --preset <name>        Build preset / build dir (default: release-with-tests)
  --backend <cpu|cuda>   Backend to benchmark (default: cpu)
  --model <path>         GGUF path (default: tests/models/gpt2.Q2_K.gguf)
  --out <file>           Output log file (default: benchmarks/results/ci-features.txt)
  --iters <N>            ASTRAL_BENCH_FEATURE_ITERS (default: 50)
  --tokens <N>           ASTRAL_BENCH_FEATURE_TOKENS (default: 32)
  --gpu-layers <N>       ASTRAL_BENCH_GPU_LAYERS when backend=cuda (default: 32)
  --help                 Show help

Examples:
  scripts/run_ci_bench_features.sh
  scripts/run_ci_bench_features.sh --backend cuda --preset dev-cuda --gpu-layers 48
EOF
}

preset="release-with-tests"
backend="cpu"
model="tests/models/gpt2.Q2_K.gguf"
out_file="benchmarks/results/ci-features.txt"
iters="50"
tokens="32"
gpu_layers="32"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --backend) backend="${2:-}"; shift 2 ;;
    --model) model="${2:-}"; shift 2 ;;
    --out) out_file="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --tokens) tokens="${2:-}"; shift 2 ;;
    --gpu-layers) gpu_layers="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

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

if [[ ! -f "${model}" ]]; then
  echo "Model not found: ${model}" >&2
  echo "Download one with: ./tests/model_downloader.sh" >&2
  exit 2
fi

mkdir -p "$(dirname "${out_file}")"

{
  echo "# Astral CI feature surface bench"
  echo "# date: $(date -Iseconds)"
  echo "# preset: ${preset}"
  echo "# backend: ${backend}"
  echo "# model: ${model}"
  echo "# iters: ${iters}"
  echo "# tokens: ${tokens}"
  echo "# gpu_layers: ${gpu_layers}"
  echo
} > "${out_file}"

ASTRAL_BENCH_MODEL="${model}" \
ASTRAL_BENCH_FEATURE_BACKEND="${backend}" \
ASTRAL_BENCH_GPU_LAYERS="${gpu_layers}" \
ASTRAL_BENCH_FEATURE_ITERS="${iters}" \
ASTRAL_BENCH_FEATURE_TOKENS="${tokens}" \
ASTRAL_BENCH_RUNTIME_THREADS="1" \
  "${bench_bin}" --only features >> "${out_file}" 2>&1

echo "[ci-bench] wrote ${out_file}"
