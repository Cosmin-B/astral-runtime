#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_memory_bench_matrix.sh [options]

Runs the native memory-only feature benchmark across metric, dimension, and
capacity combinations and writes one log file.

Options:
  --preset <name>       CMake preset / build dir to use (default: dev)
  --out <file>          Output log file (required)
  --iters <N>           ASTRAL_BENCH_FEATURE_ITERS (default: 10)
  --dims <list>         Comma-separated dimensions (default: 128,384,768)
  --capacities <list>   Comma-separated capacities (default: 10000)
  --metrics <list>      Comma-separated metrics: cosine,dot,l2 (default: cosine,dot,l2)
  --storage <kind>      Vector storage: f32 or q8 (default: f32)
  --case <name>         One ASTRAL_BENCH_MEMORY_CASE value, such as graph_recall_search
  --graph-search <N>    Graph search budget (default: 64)
  --graph-neighbors <N> Graph neighbor budget (default: 32)
  --recall-queries <N>  Graph recall queries (default: 32)
  --help                Show help

Examples:
  scripts/run_memory_bench_matrix.sh --preset release-with-tests --out /tmp/astral-memory.txt
  scripts/run_memory_bench_matrix.sh --dims 384,768 --capacities 10000,100000 --metrics cosine --out /tmp/memory.txt
EOF
}

preset="dev"
out_file=""
iters="10"
dims="128,384,768"
capacities="10000"
metrics="cosine,dot,l2"
storage="f32"
memory_case=""
graph_search="64"
graph_neighbors="32"
recall_queries="32"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --out) out_file="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --dims) dims="${2:-}"; shift 2 ;;
    --capacities) capacities="${2:-}"; shift 2 ;;
    --metrics) metrics="${2:-}"; shift 2 ;;
    --storage) storage="${2:-}"; shift 2 ;;
    --case) memory_case="${2:-}"; shift 2 ;;
    --graph-search) graph_search="${2:-}"; shift 2 ;;
    --graph-neighbors) graph_neighbors="${2:-}"; shift 2 ;;
    --recall-queries) recall_queries="${2:-}"; shift 2 ;;
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
  echo "Build it with: cmake --preset ${preset} && cmake --build --preset ${preset} --target astral_benchmarks" >&2
  exit 2
fi

IFS=',' read -r -a dim_values <<< "${dims}"
IFS=',' read -r -a capacity_values <<< "${capacities}"
IFS=',' read -r -a metric_values <<< "${metrics}"

mkdir -p "$(dirname "${out_file}")"
{
  echo "# Astral memory bench matrix"
  echo "# date: $(date -Iseconds)"
  echo "# preset: ${preset}"
  echo "# iters: ${iters}"
  echo "# dims: ${dims}"
  echo "# capacities: ${capacities}"
  echo "# metrics: ${metrics}"
  echo "# storage: ${storage}"
  echo "# case: ${memory_case}"
  echo "# graph_search: ${graph_search}"
  echo "# graph_neighbors: ${graph_neighbors}"
  echo "# recall_queries: ${recall_queries}"
  echo
} > "${out_file}"

ran=0
for metric in "${metric_values[@]}"; do
  for dim in "${dim_values[@]}"; do
    for capacity in "${capacity_values[@]}"; do
      {
        echo
        echo "## metric=${metric} dim=${dim} capacity=${capacity}"
        echo
      } >> "${out_file}"

      ASTRAL_BENCH_MEMORY_ONLY=1 \
      ASTRAL_BENCH_FEATURE_ITERS="${iters}" \
      ASTRAL_BENCH_MEMORY_METRIC="${metric}" \
      ASTRAL_BENCH_MEMORY_STORAGE="${storage}" \
      ASTRAL_BENCH_MEMORY_CASE="${memory_case}" \
      ASTRAL_BENCH_MEMORY_DIM="${dim}" \
      ASTRAL_BENCH_MEMORY_CAPACITY="${capacity}" \
      ASTRAL_BENCH_MEMORY_GRAPH_SEARCH="${graph_search}" \
      ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS="${graph_neighbors}" \
      ASTRAL_BENCH_MEMORY_RECALL_QUERIES="${recall_queries}" \
      "${bench_bin}" --only features >> "${out_file}" 2>&1

      ran=$((ran+1))
    done
  done
done

echo "[memory-matrix] done runs=${ran} out=${out_file}"
