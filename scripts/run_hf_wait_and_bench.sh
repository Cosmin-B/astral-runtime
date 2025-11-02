#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_hf_wait_and_bench.sh [options]

Waits for HF GGUF downloads to finish (no *.part files under tests/models/hf),
then runs the HF bench matrix and parses CSV summaries.

Options:
  --models-dir <dir>    (default: tests/models/hf)
  --out-dir <dir>       (default: benchmarks/results/hf-full-<timestamp>)
  --arch <list>         (default: 120a-real)
  --only <cpu|cuda|all> (default: all)
  --iters <N>           (default: 10)
  --tokens <N>          (default: 32)
  --gpu-layers <N>      (default: 48)
  --help                Show help

Notes:
  - This does NOT start downloads; it only waits for existing downloads to complete.
  - Use scripts/hf_gguf_download_manifest.sh to start/resume downloads.
EOF
}

models_dir="tests/models/hf"
out_dir=""
arch="120a-real"
only="all"
iters="10"
tokens="32"
gpu_layers="48"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) models_dir="${2:-}"; shift 2 ;;
    --out-dir) out_dir="${2:-}"; shift 2 ;;
    --arch) arch="${2:-}"; shift 2 ;;
    --only) only="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --tokens) tokens="${2:-}"; shift 2 ;;
    --gpu-layers) gpu_layers="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${out_dir}" ]]; then
  out_dir="benchmarks/results/hf-full-$(date -Iseconds | tr ':' '-')"
fi

echo "[hf-wait] models_dir=${models_dir}"
echo "[hf-wait] out_dir=${out_dir}"

if [[ ! -d "${models_dir}" ]]; then
  echo "Models dir not found: ${models_dir}" >&2
  exit 2
fi

while true; do
  parts="$(find "${models_dir}" -type f -name "*.part" 2>/dev/null | wc -l)"
  ggufs="$(find "${models_dir}" -type f -name "*.gguf" 2>/dev/null | wc -l)"
  echo "[hf-wait] gguf=${ggufs} part=${parts} $(date -Iseconds)"
  if [[ "${parts}" -eq 0 ]]; then
    break
  fi
  sleep 60
done

mkdir -p "${out_dir}"

./scripts/run_hf_bench_matrix.sh \
  --models-dir "${models_dir}" \
  --out-dir "${out_dir}" \
  --arch "${arch}" \
  --only "${only}" \
  --iters "${iters}" \
  --tokens "${tokens}" \
  --gpu-layers "${gpu_layers}" \
  --cuda-preset dev-cuda-hf \
  --cuda-cublas-preset dev-cuda-cublas-hf \
  --cuda-mmq-preset dev-cuda-mmq-hf

for f in "${out_dir}"/*cuda-auto.txt "${out_dir}"/*cuda-cublas.txt "${out_dir}"/*cuda-mmq.txt "${out_dir}"/*cpu.txt; do
  [[ -f "${f}" ]] || continue
  ./scripts/parse_hf_matrix_log.py --in "${f}" --out "${f%.txt}.csv" --require-pass
done

echo "[hf-wait] done: ${out_dir}"
