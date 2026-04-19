#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_memory_search_acceptance.sh [options]

Runs focused native memory-search benchmarks that compare exact flat search,
reduced-vector flat search, and approximate graph search against the flat
oracle. The summary includes recall, latency, and graph build-cost markers.
Logs are written outside the repository by default.

Options:
  --preset <name>       CMake preset / build dir to use (default: release-with-tests)
  --out-dir <dir>       Output directory (default: /tmp/astral-memory-search-<timestamp>)
  --iters <N>           ASTRAL_BENCH_FEATURE_ITERS (default: 10)
  --dim <N>             Vector dimension (default: 384)
  --capacity <N>        Vector count (default: 100000)
  --metric <name>       Metric: cosine, dot, or l2 (default: cosine)
  --graph-search <N>    Graph build/search budget (default: 64)
  --query-search <N>    Per-query graph search budget (default: graph budget)
  --graph-neighbors <N> Graph neighbor budget (default: 32)
  --recall-queries <N>  Recall query count (default: 32)
  --budget-sweep        Also capture graph_recall_search_sweep for f32 and q8
  --perf                Wrap benchmark lanes with perf stat
  --perf-bin <path>     Perf executable to use (default: perf from PATH)
  --perf-events <csv>   Perf stat event list
  --require-perf        Fail when perf is missing or blocked
  --help                Show help

Examples:
  scripts/run_memory_search_acceptance.sh --out-dir /tmp/astral-memory-search
  scripts/run_memory_search_acceptance.sh --capacity 10000 --iters 20 --graph-search 256
  scripts/run_memory_search_acceptance.sh --capacity 10000 --perf --perf-bin /path/to/linux-6.13/tools/perf/perf
EOF
}

preset="release-with-tests"
out_dir=""
iters="10"
dim="384"
capacity="100000"
metric="cosine"
graph_search="64"
query_search=""
graph_neighbors="32"
recall_queries="32"
budget_sweep="0"
perf_enabled="0"
require_perf="0"
perf_bin=""
perf_events="cycles,instructions,branches,branch-misses,cache-references,cache-misses,LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --out-dir) out_dir="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --dim) dim="${2:-}"; shift 2 ;;
    --capacity) capacity="${2:-}"; shift 2 ;;
    --metric) metric="${2:-}"; shift 2 ;;
    --graph-search) graph_search="${2:-}"; shift 2 ;;
    --query-search) query_search="${2:-}"; shift 2 ;;
    --graph-neighbors) graph_neighbors="${2:-}"; shift 2 ;;
    --recall-queries) recall_queries="${2:-}"; shift 2 ;;
    --budget-sweep) budget_sweep="1"; shift ;;
    --perf) perf_enabled="1"; shift ;;
    --perf-bin) perf_bin="${2:-}"; shift 2 ;;
    --perf-events) perf_events="${2:-}"; shift 2 ;;
    --require-perf) require_perf="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${require_perf}" == "1" ]]; then
  perf_enabled="1"
fi

if [[ -z "${query_search}" ]]; then
  query_search="${graph_search}"
fi

if [[ -z "${out_dir}" ]]; then
  out_dir="/tmp/astral-memory-search-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "${out_dir}"

run_case() {
  local name="$1"
  local storage="$2"
  local bench_case="$3"
  local out_file="${out_dir}/${name}.txt"
  local perf_args=()
  if [[ "${perf_enabled}" == "1" ]]; then
    perf_args+=(--perf)
  fi
  if [[ "${require_perf}" == "1" ]]; then
    perf_args+=(--require-perf)
  fi
  if [[ -n "${perf_bin}" ]]; then
    perf_args+=(--perf-bin "${perf_bin}")
  fi
  if [[ -n "${perf_events}" ]]; then
    perf_args+=(--perf-events "${perf_events}")
  fi
  scripts/run_memory_bench_matrix.sh \
    --preset "${preset}" \
    --case "${bench_case}" \
    --dims "${dim}" \
    --capacities "${capacity}" \
    --metrics "${metric}" \
    --storage "${storage}" \
    --iters "${iters}" \
    --graph-search "${graph_search}" \
    --query-search "${query_search}" \
    --graph-neighbors "${graph_neighbors}" \
    --recall-queries "${recall_queries}" \
    "${perf_args[@]}" \
    --out "${out_file}"
}

{
  echo "# Astral memory search acceptance"
  echo "# date: $(date -Iseconds)"
  echo "# preset: ${preset}"
  echo "# iters: ${iters}"
  echo "# dim: ${dim}"
  echo "# capacity: ${capacity}"
  echo "# metric: ${metric}"
  echo "# graph_search: ${graph_search}"
  echo "# query_search: ${query_search}"
  echo "# graph_neighbors: ${graph_neighbors}"
  echo "# recall_queries: ${recall_queries}"
  echo "# budget_sweep: ${budget_sweep}"
  echo "# perf_enabled: ${perf_enabled}"
  echo "# perf_bin: ${perf_bin}"
  echo "# perf_events: ${perf_events}"
  echo
} > "${out_dir}/summary.txt"

run_case "flat_f32_batch" "f32" "flat_search_batch"
run_case "flat_f32_latency" "f32" "flat_search_latency"
run_case "flat_q8_batch" "q8" "flat_search_batch"
run_case "flat_q8_recall" "q8" "flat_q8_recall_search"
run_case "graph_f32_build" "f32" "graph_add_batch"
run_case "graph_f32_latency" "f32" "graph_search_latency"
run_case "graph_f32_recall" "f32" "graph_recall_search"
run_case "graph_f32_top1_recall" "f32" "graph_recall_top1"
run_case "graph_q8_build" "q8" "graph_add_batch"
run_case "graph_q8_recall" "q8" "graph_recall_search"
run_case "graph_q8_top1_recall" "q8" "graph_recall_top1"

if [[ "${budget_sweep}" == "1" ]]; then
  run_case "graph_f32_budget_sweep" "f32" "graph_recall_search_sweep"
  run_case "graph_q8_budget_sweep" "q8" "graph_recall_search_sweep"
fi

for file in "${out_dir}"/*.txt; do
  if [[ "$(basename "${file}")" == "summary.txt" ]]; then
    continue
  fi
  {
    echo
    echo "## $(basename "${file}")"
    grep -nE "features\\.memory|recall_pct" "${file}" || true
    grep -nE "^# perf_stat:" "${file}" || true
  } >> "${out_dir}/summary.txt"
done

echo "[memory-search] wrote ${out_dir}"
