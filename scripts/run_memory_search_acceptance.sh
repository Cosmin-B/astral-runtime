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
Graph lanes also capture snapshot load cost when graph topology is persisted.
Logs are written outside the repository by default.

Options:
  --preset <name>       CMake preset / build dir to use (default: release-with-tests)
  --out-dir <dir>       Output directory (default: /tmp/astral-memory-search-<timestamp>)
  --iters <N>           ASTRAL_BENCH_FEATURE_ITERS (default: 10)
  --dim <N>             Vector dimension (default: 384)
  --capacity <N>        Vector count (default: 100000)
  --metric <name>       Metric: cosine, dot, or l2 (default: cosine)
  --graph-search <N>    Graph build/search budget (default: 64)
  --query-search <N>    Per-query graph search budget (default: index default)
  --graph-neighbors <N> Graph neighbor budget (default: 32)
  --graph-storages <list> Comma-separated graph storage names to run
  --recall-queries <N>  Recall query count (default: 32)
  --budget-sweep        Also capture graph_recall_search_sweep for graph storage lanes
  --recall-detail       Also capture graph_recall_detail for graph storage lanes
  --add-latency         Also capture graph_add_latency p50/p95/p99 lanes
  --level-stats         Also capture deterministic graph level distribution
  --edge-stats          Also capture stored graph edge counts
  --skip-flat-baseline  Skip flat baseline lanes when a caller already captured them
  --perf                Wrap benchmark lanes with perf stat
  --perf-bin <path>     Perf executable to use (default: perf from PATH)
  --perf-events <csv>   Perf stat event list
  --require-perf        Fail when perf is missing or blocked
  --min-recall-pct <N>  Fail if a named graph storage recall is below this percentage
  --recall-storages <list> Comma-separated graph storage names checked by --min-recall-pct
  --max-add-p99-ns <N>  Fail if a named graph storage insert p99 is above this nanosecond budget
  --add-latency-storages <list> Comma-separated graph storage names checked by --max-add-p99-ns
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
graph_storages="f32,q8,q8f32,f6e2m3,f6e2m3f32,f6e3m2,f6e3m2f32,f8e5m2,f8e5m2f32"
recall_queries="32"
budget_sweep="0"
recall_detail="0"
add_latency="0"
level_stats="0"
edge_stats="0"
skip_flat_baseline="0"
perf_enabled="0"
require_perf="0"
perf_bin=""
perf_events="cycles,instructions,branches,branch-misses,cache-references,cache-misses,LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses"
min_recall_pct=""
recall_storages=""
max_add_p99_ns=""
add_latency_storages=""

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
    --graph-storages) graph_storages="${2:-}"; shift 2 ;;
    --recall-queries) recall_queries="${2:-}"; shift 2 ;;
    --budget-sweep) budget_sweep="1"; shift ;;
    --recall-detail) recall_detail="1"; shift ;;
    --add-latency) add_latency="1"; shift ;;
    --level-stats) level_stats="1"; shift ;;
    --edge-stats) edge_stats="1"; shift ;;
    --skip-flat-baseline) skip_flat_baseline="1"; shift ;;
    --perf) perf_enabled="1"; shift ;;
    --perf-bin) perf_bin="${2:-}"; shift 2 ;;
    --perf-events) perf_events="${2:-}"; shift 2 ;;
    --require-perf) require_perf="1"; shift ;;
    --min-recall-pct) min_recall_pct="${2:-}"; shift 2 ;;
    --recall-storages) recall_storages="${2:-}"; shift 2 ;;
    --max-add-p99-ns) max_add_p99_ns="${2:-}"; shift 2 ;;
    --add-latency-storages) add_latency_storages="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${require_perf}" == "1" ]]; then
  perf_enabled="1"
fi

if [[ -n "${max_add_p99_ns}" && -n "${add_latency_storages}" ]]; then
  add_latency="1"
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

storage_enabled() {
  local needle="$1"
  local values="$2"
  local value
  IFS=',' read -r -a storage_values <<< "${values}"
  for value in "${storage_values[@]}"; do
    if [[ "${value}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

run_graph_case() {
  local name="$1"
  local storage="$2"
  local bench_case="$3"
  if storage_enabled "${storage}" "${graph_storages}"; then
    run_case "${name}" "${storage}" "${bench_case}"
  fi
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
  echo "# graph_storages: ${graph_storages}"
  echo "# recall_queries: ${recall_queries}"
  echo "# budget_sweep: ${budget_sweep}"
  echo "# recall_detail: ${recall_detail}"
  echo "# add_latency: ${add_latency}"
  echo "# level_stats: ${level_stats}"
  echo "# edge_stats: ${edge_stats}"
  echo "# skip_flat_baseline: ${skip_flat_baseline}"
  echo "# perf_enabled: ${perf_enabled}"
  echo "# perf_bin: ${perf_bin}"
  echo "# perf_events: ${perf_events}"
  echo "# min_recall_pct: ${min_recall_pct:-none}"
  echo "# recall_storages: ${recall_storages:-none}"
  echo "# max_add_p99_ns: ${max_add_p99_ns:-none}"
  echo "# add_latency_storages: ${add_latency_storages:-none}"
  echo
} > "${out_dir}/summary.txt"

if [[ "${skip_flat_baseline}" != "1" ]]; then
  run_case "flat_f32_batch" "f32" "flat_search_batch"
  run_case "flat_f32_latency" "f32" "flat_search_latency"
  run_case "flat_q8_batch" "q8" "flat_search_batch"
  run_case "flat_q8_recall" "q8" "flat_q8_recall_search"
  run_case "flat_f6e2m3_batch" "f6e2m3" "flat_search_batch"
  run_case "flat_f6e2m3_recall" "f6e2m3" "flat_compact_recall_search"
  run_case "flat_f6e2m3f32_batch" "f6e2m3f32" "flat_search_batch"
  run_case "flat_f6e2m3f32_recall" "f6e2m3f32" "flat_compact_recall_search"
  run_case "flat_f6e3m2_batch" "f6e3m2" "flat_search_batch"
  run_case "flat_f6e3m2_recall" "f6e3m2" "flat_compact_recall_search"
  run_case "flat_f6e3m2f32_batch" "f6e3m2f32" "flat_search_batch"
  run_case "flat_f6e3m2f32_recall" "f6e3m2f32" "flat_compact_recall_search"
  run_case "flat_f8e5m2_batch" "f8e5m2" "flat_search_batch"
  run_case "flat_f8e5m2_recall" "f8e5m2" "flat_compact_recall_search"
  run_case "flat_f8e5m2f32_batch" "f8e5m2f32" "flat_search_batch"
  run_case "flat_f8e5m2f32_recall" "f8e5m2f32" "flat_compact_recall_search"
fi
run_graph_case "graph_f32_build" "f32" "graph_add_batch"
run_graph_case "graph_f32_load" "f32" "graph_load"
run_graph_case "graph_f32_latency" "f32" "graph_search_latency"
run_graph_case "graph_f32_recall" "f32" "graph_recall_search"
run_graph_case "graph_f32_view_recall" "f32" "graph_snapshot_view_recall_search"
run_graph_case "graph_f32_top1_recall" "f32" "graph_recall_top1"
run_graph_case "graph_q8_build" "q8" "graph_add_batch"
run_graph_case "graph_q8_load" "q8" "graph_load"
run_graph_case "graph_q8_recall" "q8" "graph_recall_search"
run_graph_case "graph_q8_view_recall" "q8" "graph_snapshot_view_recall_search"
run_graph_case "graph_q8_top1_recall" "q8" "graph_recall_top1"
run_graph_case "graph_q8f32_build" "q8f32" "graph_add_batch"
run_graph_case "graph_q8f32_load" "q8f32" "graph_load"
run_graph_case "graph_q8f32_recall" "q8f32" "graph_recall_search"
run_graph_case "graph_q8f32_view_recall" "q8f32" "graph_snapshot_view_recall_search"
run_graph_case "graph_q8f32_top1_recall" "q8f32" "graph_recall_top1"
run_graph_case "graph_f6e2m3_build" "f6e2m3" "graph_add_batch"
run_graph_case "graph_f6e2m3_load" "f6e2m3" "graph_load"
run_graph_case "graph_f6e2m3_recall" "f6e2m3" "graph_recall_search"
run_graph_case "graph_f6e2m3_view_recall" "f6e2m3" "graph_snapshot_view_recall_search"
run_graph_case "graph_f6e2m3_top1_recall" "f6e2m3" "graph_recall_top1"
run_graph_case "graph_f6e2m3f32_build" "f6e2m3f32" "graph_add_batch"
run_graph_case "graph_f6e2m3f32_load" "f6e2m3f32" "graph_load"
run_graph_case "graph_f6e2m3f32_recall" "f6e2m3f32" "graph_recall_search"
run_graph_case "graph_f6e2m3f32_view_recall" "f6e2m3f32" "graph_snapshot_view_recall_search"
run_graph_case "graph_f6e2m3f32_top1_recall" "f6e2m3f32" "graph_recall_top1"
run_graph_case "graph_f6e3m2_build" "f6e3m2" "graph_add_batch"
run_graph_case "graph_f6e3m2_load" "f6e3m2" "graph_load"
run_graph_case "graph_f6e3m2_recall" "f6e3m2" "graph_recall_search"
run_graph_case "graph_f6e3m2_view_recall" "f6e3m2" "graph_snapshot_view_recall_search"
run_graph_case "graph_f6e3m2_top1_recall" "f6e3m2" "graph_recall_top1"
run_graph_case "graph_f6e3m2f32_build" "f6e3m2f32" "graph_add_batch"
run_graph_case "graph_f6e3m2f32_load" "f6e3m2f32" "graph_load"
run_graph_case "graph_f6e3m2f32_recall" "f6e3m2f32" "graph_recall_search"
run_graph_case "graph_f6e3m2f32_view_recall" "f6e3m2f32" "graph_snapshot_view_recall_search"
run_graph_case "graph_f6e3m2f32_top1_recall" "f6e3m2f32" "graph_recall_top1"
run_graph_case "graph_f8e5m2_build" "f8e5m2" "graph_add_batch"
run_graph_case "graph_f8e5m2_load" "f8e5m2" "graph_load"
run_graph_case "graph_f8e5m2_recall" "f8e5m2" "graph_recall_search"
run_graph_case "graph_f8e5m2_view_recall" "f8e5m2" "graph_snapshot_view_recall_search"
run_graph_case "graph_f8e5m2_top1_recall" "f8e5m2" "graph_recall_top1"
run_graph_case "graph_f8e5m2f32_build" "f8e5m2f32" "graph_add_batch"
run_graph_case "graph_f8e5m2f32_load" "f8e5m2f32" "graph_load"
run_graph_case "graph_f8e5m2f32_recall" "f8e5m2f32" "graph_recall_search"
run_graph_case "graph_f8e5m2f32_view_recall" "f8e5m2f32" "graph_snapshot_view_recall_search"
run_graph_case "graph_f8e5m2f32_top1_recall" "f8e5m2f32" "graph_recall_top1"

if [[ "${add_latency}" == "1" ]]; then
  run_graph_case "graph_f32_add_latency" "f32" "graph_add_latency"
  run_graph_case "graph_q8_add_latency" "q8" "graph_add_latency"
  run_graph_case "graph_q8f32_add_latency" "q8f32" "graph_add_latency"
  run_graph_case "graph_f6e2m3_add_latency" "f6e2m3" "graph_add_latency"
  run_graph_case "graph_f6e2m3f32_add_latency" "f6e2m3f32" "graph_add_latency"
  run_graph_case "graph_f6e3m2_add_latency" "f6e3m2" "graph_add_latency"
  run_graph_case "graph_f6e3m2f32_add_latency" "f6e3m2f32" "graph_add_latency"
  run_graph_case "graph_f8e5m2_add_latency" "f8e5m2" "graph_add_latency"
  run_graph_case "graph_f8e5m2f32_add_latency" "f8e5m2f32" "graph_add_latency"
fi

if [[ "${budget_sweep}" == "1" ]]; then
  run_graph_case "graph_f32_budget_sweep" "f32" "graph_recall_search_sweep"
  run_graph_case "graph_q8_budget_sweep" "q8" "graph_recall_search_sweep"
  run_graph_case "graph_q8f32_budget_sweep" "q8f32" "graph_recall_search_sweep"
  run_graph_case "graph_f6e2m3_budget_sweep" "f6e2m3" "graph_recall_search_sweep"
  run_graph_case "graph_f6e2m3f32_budget_sweep" "f6e2m3f32" "graph_recall_search_sweep"
  run_graph_case "graph_f6e3m2_budget_sweep" "f6e3m2" "graph_recall_search_sweep"
  run_graph_case "graph_f6e3m2f32_budget_sweep" "f6e3m2f32" "graph_recall_search_sweep"
  run_graph_case "graph_f8e5m2_budget_sweep" "f8e5m2" "graph_recall_search_sweep"
  run_graph_case "graph_f8e5m2f32_budget_sweep" "f8e5m2f32" "graph_recall_search_sweep"
fi

if [[ "${recall_detail}" == "1" ]]; then
  run_graph_case "graph_f32_recall_detail" "f32" "graph_recall_detail"
  run_graph_case "graph_q8_recall_detail" "q8" "graph_recall_detail"
  run_graph_case "graph_q8f32_recall_detail" "q8f32" "graph_recall_detail"
  run_graph_case "graph_f6e2m3_recall_detail" "f6e2m3" "graph_recall_detail"
  run_graph_case "graph_f6e2m3f32_recall_detail" "f6e2m3f32" "graph_recall_detail"
  run_graph_case "graph_f6e3m2_recall_detail" "f6e3m2" "graph_recall_detail"
  run_graph_case "graph_f6e3m2f32_recall_detail" "f6e3m2f32" "graph_recall_detail"
  run_graph_case "graph_f8e5m2_recall_detail" "f8e5m2" "graph_recall_detail"
  run_graph_case "graph_f8e5m2f32_recall_detail" "f8e5m2f32" "graph_recall_detail"
fi

if [[ "${level_stats}" == "1" ]]; then
  run_graph_case "graph_level_stats" "f32" "graph_level_stats"
fi

if [[ "${edge_stats}" == "1" ]]; then
  run_graph_case "graph_f32_edge_stats" "f32" "graph_edge_stats"
  run_graph_case "graph_q8_edge_stats" "q8" "graph_edge_stats"
  run_graph_case "graph_q8f32_edge_stats" "q8f32" "graph_edge_stats"
  run_graph_case "graph_f6e2m3_edge_stats" "f6e2m3" "graph_edge_stats"
  run_graph_case "graph_f6e2m3f32_edge_stats" "f6e2m3f32" "graph_edge_stats"
  run_graph_case "graph_f6e3m2_edge_stats" "f6e3m2" "graph_edge_stats"
  run_graph_case "graph_f6e3m2f32_edge_stats" "f6e3m2f32" "graph_edge_stats"
  run_graph_case "graph_f8e5m2_edge_stats" "f8e5m2" "graph_edge_stats"
  run_graph_case "graph_f8e5m2f32_edge_stats" "f8e5m2f32" "graph_edge_stats"
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

if [[ -n "${min_recall_pct}" && -n "${recall_storages}" ]]; then
  IFS=',' read -r -a recall_storage_values <<< "${recall_storages}"
  for storage in "${recall_storage_values[@]}"; do
    recall_file="${out_dir}/graph_${storage}_recall.txt"
    if [[ ! -f "${recall_file}" ]]; then
      echo "[memory-search] missing recall file for storage=${storage}: ${recall_file}" >&2
      exit 1
    fi
    recall_pct="$(awk '
      /recall_pct=/ {
        for (i = 1; i <= NF; ++i) {
          if (index($i, "recall_pct=") == 1) {
            value = substr($i, length("recall_pct=") + 1)
            if (value != "") {
              print value
              exit
            }
            if ((i + 1) <= NF) {
              print $(i + 1)
              exit
            }
          }
        }
      }
    ' "${recall_file}")"
    if [[ -z "${recall_pct}" ]]; then
      echo "[memory-search] missing recall_pct for storage=${storage}: ${recall_file}" >&2
      exit 1
    fi
    awk -v storage="${storage}" -v recall="${recall_pct}" -v min="${min_recall_pct}" '
      BEGIN {
        if ((recall + 0.0) < (min + 0.0)) {
          printf "[memory-search] recall below threshold: storage=%s recall_pct=%.4f min=%.4f\n",
                 storage, recall + 0.0, min + 0.0 > "/dev/stderr"
          exit 1
        }
      }
    '
  done
fi

if [[ -n "${max_add_p99_ns}" && -n "${add_latency_storages}" ]]; then
  IFS=',' read -r -a add_latency_storage_values <<< "${add_latency_storages}"
  for storage in "${add_latency_storage_values[@]}"; do
    add_latency_file="${out_dir}/graph_${storage}_add_latency.txt"
    if [[ ! -f "${add_latency_file}" ]]; then
      echo "[memory-search] missing add-latency file for storage=${storage}: ${add_latency_file}" >&2
      exit 1
    fi
    add_p99_ns="$(awk '
      /p99=/ {
        for (i = 1; i <= NF; ++i) {
          if (index($i, "p99=") == 1) {
            value = substr($i, length("p99=") + 1)
            if (value != "") {
              print value
              exit
            }
            if ((i + 1) <= NF) {
              print $(i + 1)
              exit
            }
          }
        }
      }
    ' "${add_latency_file}")"
    if [[ -z "${add_p99_ns}" ]]; then
      echo "[memory-search] missing p99 add latency for storage=${storage}: ${add_latency_file}" >&2
      exit 1
    fi
    awk -v storage="${storage}" -v p99="${add_p99_ns}" -v max="${max_add_p99_ns}" '
      BEGIN {
        if ((p99 + 0.0) > (max + 0.0)) {
          printf "[memory-search] add p99 above threshold: storage=%s p99_ns=%.4f max=%.4f\n",
                 storage, p99 + 0.0, max + 0.0 > "/dev/stderr"
          exit 1
        }
      }
    '
  done
fi

echo "[memory-search] wrote ${out_dir}"
