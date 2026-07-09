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
  --iters <N>           ASTRAL_BENCH_FEATURE_ITERS (default: 10);
                        aggregate recall cases use at least --recall-queries
  --dims <list>         Comma-separated dimensions (default: 128,384,768)
  --capacities <list>   Comma-separated capacities (default: 10000)
  --metrics <list>      Comma-separated metrics: cosine,dot,l2 (default: cosine,dot,l2)
  --storage <kind>      Vector storage: f32, q8, q8f32, f6e2m3, f6e2m3f32, f6e3m2, f6e3m2f32, f8e5m2, or f8e5m2f32 (default: f32)
  --storages <list>     Comma-separated vector storage kinds
  --case <name>         One ASTRAL_BENCH_MEMORY_CASE value, such as flat_q8_recall_search
  --graph-search <N>    Graph search budget (default: 64)
  --query-search <N>    Per-query graph search budget (default: index budget)
  --query-searches <list>
                        Comma-separated per-query graph search budgets
  --graph-neighbors <N> Graph neighbor budget (default: 32)
  --batch-queries <N>   ASTRAL_BENCH_MEMORY_BATCH_QUERIES for batch cases
  --runtime-threads <N> ASTRAL_BENCH_RUNTIME_THREADS for benchmark worker lanes
  --recall-queries <N>  Graph recall queries (default: 32)
  --runtime-threads <N> Astral worker threads (default: 1; the caller is separate)
  --perf                Wrap each benchmark invocation with perf stat
  --perf-bin <path>     Perf executable to use (default: perf from PATH)
  --perf-events <csv>   Perf stat event list
  --require-perf        Fail when perf is missing or blocked
  --summary <file>      Write a compact CSV parsed from the log and perf CSVs
  --help                Show help

Examples:
  scripts/run_memory_bench_matrix.sh --preset release-with-tests --out /tmp/astral-memory.txt
  scripts/run_memory_bench_matrix.sh --dims 384,768 --capacities 10000,100000 --metrics cosine --out /tmp/memory.txt
  scripts/run_memory_bench_matrix.sh --case graph_recall_search --perf --perf-bin /path/to/linux-6.13/tools/perf/perf --out /tmp/memory-perf.txt
EOF
}

preset="dev"
out_file=""
iters="10"
dims="128,384,768"
capacities="10000"
metrics="cosine,dot,l2"
storage="f32"
storages=""
memory_case=""
graph_search="64"
query_search=""
query_searches=""
graph_neighbors="32"
batch_queries=""
runtime_threads=""
recall_queries="32"
runtime_threads="1"
perf_enabled="0"
require_perf="0"
perf_bin=""
perf_events="cycles,instructions,branches,branch-misses,cache-references,cache-misses,LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses"
summary_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --out) out_file="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --dims) dims="${2:-}"; shift 2 ;;
    --capacities) capacities="${2:-}"; shift 2 ;;
    --metrics) metrics="${2:-}"; shift 2 ;;
    --storage) storage="${2:-}"; shift 2 ;;
    --storages) storages="${2:-}"; shift 2 ;;
    --case) memory_case="${2:-}"; shift 2 ;;
    --graph-search) graph_search="${2:-}"; shift 2 ;;
    --query-search) query_search="${2:-}"; shift 2 ;;
    --query-searches) query_searches="${2:-}"; shift 2 ;;
    --graph-neighbors) graph_neighbors="${2:-}"; shift 2 ;;
    --batch-queries) batch_queries="${2:-}"; shift 2 ;;
    --runtime-threads) runtime_threads="${2:-}"; shift 2 ;;
    --recall-queries) recall_queries="${2:-}"; shift 2 ;;
    --runtime-threads) runtime_threads="${2:-}"; shift 2 ;;
    --perf) perf_enabled="1"; shift ;;
    --perf-bin) perf_bin="${2:-}"; shift 2 ;;
    --perf-events) perf_events="${2:-}"; shift 2 ;;
    --require-perf) require_perf="1"; shift ;;
    --summary) summary_file="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${require_perf}" == "1" ]]; then
  perf_enabled="1"
fi

if [[ -z "${out_file}" ]]; then
  echo "Missing required arg: --out" >&2
  usage
  exit 2
fi

if [[ -z "${summary_file}" && "${out_file}" == *.txt ]]; then
  summary_file="${out_file%.txt}.csv"
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

if [[ "${perf_enabled}" == "1" ]]; then
  if [[ -z "${perf_bin}" ]] && command -v perf >/dev/null 2>&1; then
    perf_bin="$(command -v perf)"
  fi
  if [[ -z "${perf_bin}" || ! -x "${perf_bin}" ]]; then
    if [[ "${require_perf}" == "1" ]]; then
      echo "perf not found" >&2
      exit 2
    fi
    perf_enabled="0"
  fi
fi

IFS=',' read -r -a dim_values <<< "${dims}"
IFS=',' read -r -a capacity_values <<< "${capacities}"
IFS=',' read -r -a metric_values <<< "${metrics}"
if [[ -n "${storages}" ]]; then
  IFS=',' read -r -a storage_values <<< "${storages}"
else
  storage_values=("${storage}")
fi
if [[ -n "${query_searches}" ]]; then
  IFS=',' read -r -a query_search_values <<< "${query_searches}"
else
  query_search_values=("${query_search}")
fi

recall_case_requires_full_query_set() {
  case "$1" in
    flat_q8_recall_search|flat_compact_recall_search|graph_recall|graph_recall_top1|graph_recall_search|graph_recall_latency|graph_snapshot_view_recall_search|graph_recall_search_sweep)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

effective_iters="${iters}"
if recall_case_requires_full_query_set "${memory_case}" && (( iters < recall_queries )); then
  effective_iters="${recall_queries}"
fi

mkdir -p "$(dirname "${out_file}")"
perf_dir="$(dirname "${out_file}")/$(basename "${out_file}" .txt)-perf"
if [[ "${perf_enabled}" == "1" ]]; then
  mkdir -p "${perf_dir}"
fi
{
  echo "# Astral memory bench matrix"
  echo "# date: $(date -Iseconds)"
  echo "# preset: ${preset}"
  echo "# iters: ${iters}"
  echo "# effective_iters: ${effective_iters}"
  echo "# dims: ${dims}"
  echo "# capacities: ${capacities}"
  echo "# metrics: ${metrics}"
  echo "# storage: ${storage}"
  echo "# storages: ${storages}"
  echo "# case: ${memory_case}"
  echo "# graph_search: ${graph_search}"
  echo "# query_search: ${query_search}"
  echo "# query_searches: ${query_searches}"
  echo "# graph_neighbors: ${graph_neighbors}"
  echo "# batch_queries: ${batch_queries}"
  echo "# runtime_threads: ${runtime_threads}"
  echo "# recall_queries: ${recall_queries}"
  echo "# runtime_threads: ${runtime_threads}"
  echo "# perf_enabled: ${perf_enabled}"
  echo "# perf_bin: ${perf_bin}"
  echo "# perf_events: ${perf_events}"
  echo "# summary: ${summary_file}"
  echo
} > "${out_file}"

ran=0
for metric in "${metric_values[@]}"; do
  for storage_value in "${storage_values[@]}"; do
    for dim in "${dim_values[@]}"; do
      for capacity in "${capacity_values[@]}"; do
        for query_search_value in "${query_search_values[@]}"; do
          {
            echo
            echo "## metric=${metric} storage=${storage_value} dim=${dim} capacity=${capacity} query_search=${query_search_value}"
            if [[ "${perf_enabled}" == "1" ]]; then
              echo "# perf_stat: ${perf_dir}/${metric}-${storage_value}-d${dim}-n${capacity}-q${query_search_value}.csv"
            fi
            echo
          } >> "${out_file}"

          bench_cmd=(
            env
            "ASTRAL_BENCH_MEMORY_ONLY=1"
            "ASTRAL_BENCH_FEATURE_ITERS=${effective_iters}"
            "ASTRAL_BENCH_MEMORY_METRIC=${metric}"
            "ASTRAL_BENCH_MEMORY_STORAGE=${storage_value}"
            "ASTRAL_BENCH_MEMORY_CASE=${memory_case}"
            "ASTRAL_BENCH_MEMORY_DIM=${dim}"
            "ASTRAL_BENCH_MEMORY_CAPACITY=${capacity}"
            "ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=${graph_search}"
            "ASTRAL_BENCH_MEMORY_GRAPH_QUERY_SEARCH=${query_search_value}"
            "ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS=${graph_neighbors}"
            "ASTRAL_BENCH_MEMORY_RECALL_QUERIES=${recall_queries}"
            "ASTRAL_BENCH_RUNTIME_THREADS=${runtime_threads}"
            "${bench_bin}"
            --only
            features
          )
          if [[ -n "${batch_queries}" ]]; then
            bench_cmd=(env "ASTRAL_BENCH_MEMORY_BATCH_QUERIES=${batch_queries}" "${bench_cmd[@]:1}")
          fi
          if [[ -n "${runtime_threads}" ]]; then
            bench_cmd=(env "ASTRAL_BENCH_RUNTIME_THREADS=${runtime_threads}" "${bench_cmd[@]:1}")
          fi
          if [[ "${perf_enabled}" == "1" ]]; then
            perf_prefix="${perf_dir}/${metric}-${storage_value}-d${dim}-n${capacity}-q${query_search_value}"
            if ! "${perf_bin}" stat -x, -e "${perf_events}" -o "${perf_prefix}.csv" -- "${bench_cmd[@]}" \
              >> "${out_file}" 2> "${perf_prefix}.stderr"; then
              if [[ "${require_perf}" == "1" ]]; then
                cat "${perf_prefix}.stderr" >&2
                exit 1
              fi
              {
                echo "# perf unavailable for metric=${metric} storage=${storage_value} dim=${dim} capacity=${capacity} query_search=${query_search_value}"
                cat "${perf_prefix}.stderr"
              } >> "${out_file}"
              "${bench_cmd[@]}" >> "${out_file}" 2>&1
            fi
          else
            "${bench_cmd[@]}" >> "${out_file}" 2>&1
          fi

          ran=$((ran+1))
        done
      done
    done
  done
done

if [[ -n "${summary_file}" ]]; then
  scripts/summarize_memory_bench_matrix.py --in "${out_file}" --out "${summary_file}" --require-rows
fi

echo "[memory-matrix] done runs=${ran} out=${out_file}"
