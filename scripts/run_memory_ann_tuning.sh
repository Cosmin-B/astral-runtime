#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_memory_ann_tuning.sh [options]

Runs a repeatable ANN tuning sweep for Astral memory graph indexes. The runner
captures f32 and q8 graph recall, top-1 recall, build cost, load cost, edge
counts, and optional budget sweeps for each requested graph shape. Output goes
outside the repository by default.

Options:
  --preset <name>          CMake preset / build dir to use (default: release-with-tests)
  --out-dir <dir>          Output directory (default: /tmp/astral-ann-tuning-<timestamp>)
  --iters <N>              ASTRAL_BENCH_FEATURE_ITERS (default: 8)
  --dim <N>                Vector dimension (default: 384)
  --capacity <N>           Vector count (default: 100000)
  --metric <name>          Metric: cosine, dot, or l2 (default: cosine)
  --neighbors <list>       Comma-separated graph neighbor budgets (default: 16,32)
  --build-search <list>    Comma-separated graph build/search budgets (default: 64,128,256)
  --query-search <list>    Comma-separated per-query budgets (default: 64,128)
  --recall-queries <N>     Recall query count (default: 32)
  --budget-sweep           Capture per-query graph budget sweep lanes
  --recall-detail          Capture per-query recall detail lanes
  --perf                   Wrap benchmark lanes with perf stat
  --perf-bin <path>        Perf executable to use
  --perf-events <csv>      Perf stat event list
  --require-perf           Fail when perf is missing or blocked
  --help                   Show help

Examples:
  scripts/run_memory_ann_tuning.sh --capacity 10000 --iters 4 --out-dir /tmp/ann-smoke
  scripts/run_memory_ann_tuning.sh --neighbors 32 --build-search 256 --query-search 64,128 --perf
EOF
}

preset="release-with-tests"
out_dir=""
iters="8"
dim="384"
capacity="100000"
metric="cosine"
neighbors_csv="16,32"
build_search_csv="64,128,256"
query_search_csv="64,128"
recall_queries="32"
budget_sweep="0"
recall_detail="0"
perf_enabled="0"
require_perf="0"
perf_bin=""
perf_events=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --out-dir) out_dir="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --dim) dim="${2:-}"; shift 2 ;;
    --capacity) capacity="${2:-}"; shift 2 ;;
    --metric) metric="${2:-}"; shift 2 ;;
    --neighbors) neighbors_csv="${2:-}"; shift 2 ;;
    --build-search) build_search_csv="${2:-}"; shift 2 ;;
    --query-search) query_search_csv="${2:-}"; shift 2 ;;
    --recall-queries) recall_queries="${2:-}"; shift 2 ;;
    --budget-sweep) budget_sweep="1"; shift ;;
    --recall-detail) recall_detail="1"; shift ;;
    --perf) perf_enabled="1"; shift ;;
    --perf-bin) perf_bin="${2:-}"; shift 2 ;;
    --perf-events) perf_events="${2:-}"; shift 2 ;;
    --require-perf) require_perf="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${out_dir}" ]]; then
  out_dir="/tmp/astral-ann-tuning-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "${out_dir}"

IFS=',' read -r -a neighbor_values <<< "${neighbors_csv}"
IFS=',' read -r -a build_search_values <<< "${build_search_csv}"
IFS=',' read -r -a query_search_values <<< "${query_search_csv}"

common_args=(
  --preset "${preset}"
  --iters "${iters}"
  --dim "${dim}"
  --capacity "${capacity}"
  --metric "${metric}"
  --recall-queries "${recall_queries}"
  --edge-stats
  --level-stats
)

if [[ "${budget_sweep}" == "1" ]]; then
  common_args+=(--budget-sweep)
fi
if [[ "${recall_detail}" == "1" ]]; then
  common_args+=(--recall-detail)
fi
if [[ "${perf_enabled}" == "1" ]]; then
  common_args+=(--perf)
fi
if [[ "${require_perf}" == "1" ]]; then
  common_args+=(--require-perf)
fi
if [[ -n "${perf_bin}" ]]; then
  common_args+=(--perf-bin "${perf_bin}")
fi
if [[ -n "${perf_events}" ]]; then
  common_args+=(--perf-events "${perf_events}")
fi

{
  echo "# Astral ANN tuning sweep"
  echo "# date: $(date -Iseconds)"
  echo "# preset: ${preset}"
  echo "# iters: ${iters}"
  echo "# dim: ${dim}"
  echo "# capacity: ${capacity}"
  echo "# metric: ${metric}"
  echo "# neighbors: ${neighbors_csv}"
  echo "# build_search: ${build_search_csv}"
  echo "# query_search: ${query_search_csv}"
  echo "# recall_queries: ${recall_queries}"
  echo "# budget_sweep: ${budget_sweep}"
  echo "# recall_detail: ${recall_detail}"
  echo "# perf_enabled: ${perf_enabled}"
  echo
} > "${out_dir}/summary.txt"

run_count=0
for neighbors in "${neighbor_values[@]}"; do
  for build_search in "${build_search_values[@]}"; do
    for query_search in "${query_search_values[@]}"; do
      shape_dir="${out_dir}/n${neighbors}_b${build_search}_q${query_search}"
      scripts/run_memory_search_acceptance.sh \
        "${common_args[@]}" \
        --out-dir "${shape_dir}" \
        --graph-neighbors "${neighbors}" \
        --graph-search "${build_search}" \
        --query-search "${query_search}"

      {
        echo
        echo "## neighbors=${neighbors} build_search=${build_search} query_search=${query_search}"
        echo "# summary: ${shape_dir}/summary.txt"
        grep -nE "features\\.memory|recall_pct|top1_recall_pct|edge_count|level_count" "${shape_dir}/summary.txt" || true
      } >> "${out_dir}/summary.txt"
      run_count=$((run_count + 1))
    done
  done
done

echo "# runs: ${run_count}" >> "${out_dir}/summary.txt"
echo "[memory-ann-tuning] wrote ${out_dir}"
