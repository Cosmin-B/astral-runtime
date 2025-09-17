#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_hf_full_suite.sh [options]

Runs the pinned HF GGUF workflow end-to-end:
  1) Download models listed in scripts/hf_gguf_manifest_full.json into tests/models/hf
  2) Run the feature-surface bench matrix across all downloaded *.gguf

Options:
  --arch <list>         Passed to scripts/run_hf_bench_matrix.sh --arch (e.g. 120a-real)
  --only <cpu|cuda|all> Which benches to run (default: all)
  --iters <N>           Feature bench iters (default: 25)
  --tokens <N>          Logprobs drain tokens (default: 32)
  --gpu-layers <N>      Default GPU layers for CUDA runs (default: 48)
  --help                Show help

Notes:
  - This can be very large (many GB) if mode=all for big repos (e.g. 32B quants).
  - The bench matrix enumerates *.gguf at start; re-run after downloads to include new files.
EOF
}

arch=""
only="all"
iters="25"
tokens="32"
gpu_layers="48"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) arch="${2:-}"; shift 2 ;;
    --only) only="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --tokens) tokens="${2:-}"; shift 2 ;;
    --gpu-layers) gpu_layers="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

./scripts/hf_gguf_download_manifest.sh --out tests/models/hf

bench_args=(--models-dir tests/models/hf --out-dir benchmarks/results/hf --only "${only}" --iters "${iters}" --tokens "${tokens}" --gpu-layers "${gpu_layers}")
if [[ -n "${arch}" ]]; then
  bench_args+=(--arch "${arch}")
fi

./scripts/run_hf_bench_matrix.sh "${bench_args[@]}"

