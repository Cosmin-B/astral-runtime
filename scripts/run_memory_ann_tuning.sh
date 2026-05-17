#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_memory_ann_tuning.sh [options]

Runs a repeatable ANN tuning sweep for Astral memory graph indexes. The runner
captures f32, q8, and f6e2m3 graph recall, top-1 recall, build cost, load
cost, edge counts, and optional budget sweeps for each requested graph shape.
Output goes outside the repository by default.

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
  --summary-only           Rebuild summary best-row sections from an existing results.csv
  --min-recall-pct <pct>   Fail unless one row reaches this recall percentage
  --max-recall-ns <N>      With --min-recall-pct, fail unless one passing row is this fast
  --min-speedup-vs-flat <N> Fail unless one passing row reaches this flat-f32 speedup
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
summary_only="0"
min_recall_pct=""
max_recall_ns=""
min_speedup_vs_flat=""
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
    --summary-only) summary_only="1"; shift ;;
    --min-recall-pct) min_recall_pct="${2:-}"; shift 2 ;;
    --max-recall-ns) max_recall_ns="${2:-}"; shift 2 ;;
    --min-speedup-vs-flat) min_speedup_vs_flat="${2:-}"; shift 2 ;;
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
csv_file="${out_dir}/results.csv"
summary_file="${out_dir}/summary.txt"
if [[ "${summary_only}" == "1" ]]; then
  summary_file="${out_dir}/summary_best.txt"
fi

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
  echo "# date: $(date -Iseconds)"
  if [[ "${summary_only}" == "1" ]]; then
    echo "# Astral ANN tuning summary"
    echo "# summary_only: 1"
    echo "# results_csv: ${csv_file}"
    echo "# min_recall_pct: ${min_recall_pct:-none}"
    echo "# max_recall_ns: ${max_recall_ns:-none}"
    echo "# min_speedup_vs_flat: ${min_speedup_vs_flat:-none}"
  else
    echo "# Astral ANN tuning sweep"
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
    echo "# min_recall_pct: ${min_recall_pct:-none}"
    echo "# max_recall_ns: ${max_recall_ns:-none}"
    echo "# min_speedup_vs_flat: ${min_speedup_vs_flat:-none}"
    echo "# perf_enabled: ${perf_enabled}"
  fi
  echo
} > "${summary_file}"

if [[ "${summary_only}" != "1" ]]; then
  echo "neighbors,build_search,query_search,effective_query_search,storage,build_ns,load_ns,recall_ns,recall_pct,top1_ns,top1_recall_pct,edge_count,base_edges,upper_edges,build_score_evals,build_candidate_visits,flat_f32_latency_ns,speedup_vs_flat_f32" > "${csv_file}"
fi

metric_value() {
  local file="$1"
  local key="$2"
  awk -v key="${key}" '
    index($0, key) {
      value = substr($0, index($0, key) + length(key))
      sub(/^[[:space:]]*/, "", value)
      split(value, parts, /[[:space:]]+/)
      print parts[1]
      exit
    }
  ' "${file}"
}

metric_ns() {
  local file="$1"
  awk '
    /features\.memory/ {
      for (i = 1; i <= NF; ++i) {
        if ($i == "ns/op") {
          print $(i - 1)
          exit
        }
        if ($i ~ /^p50=/) {
          value = $i
          sub(/^p50=/, "", value)
          if (value != "") {
            print value
            exit
          }
          if ((i + 1) <= NF) {
            print $(i + 1)
            exit
          }
        }
        if ($i == "p50=" && (i + 1) <= NF) {
          print $(i + 1)
          exit
        }
      }
    }
  ' "${file}"
}

edge_metric_value() {
  local file="$1"
  local metric="$2"
  local label="${3:-edge_count}"
  awk -v metric="${metric}" -v label="${label}=" '
    index($0, metric) && index($0, label) {
      value = substr($0, index($0, label) + length(label))
      sub(/^[[:space:]]*/, "", value)
      split(value, parts, /[[:space:]]+/)
      print parts[1]
      exit
    }
  ' "${file}"
}

effective_query_search_for() {
  local _build_search="$1"
  local query_search="$2"
  echo "${query_search}"
}

append_csv_row() {
  local shape_dir="$1"
  local neighbors="$2"
  local build_search="$3"
  local query_search="$4"
  local storage="$5"
  local effective_query_search
  effective_query_search="$(effective_query_search_for "${build_search}" "${query_search}")"
  local prefix="graph_${storage}"
  local build_ns load_ns recall_ns recall_pct top1_ns top1_recall edge_count base_edges upper_edges build_score_evals build_candidate_visits speedup_vs_flat
  build_ns="$(metric_ns "${shape_dir}/${prefix}_build.txt")"
  load_ns="$(metric_ns "${shape_dir}/${prefix}_load.txt")"
  recall_ns="$(metric_ns "${shape_dir}/${prefix}_recall.txt")"
  recall_pct="$(metric_value "${shape_dir}/${prefix}_recall.txt" "recall_pct=")"
  top1_ns="$(metric_ns "${shape_dir}/${prefix}_top1_recall.txt")"
  top1_recall="$(metric_value "${shape_dir}/${prefix}_top1_recall.txt" "top1_recall_pct=")"
  edge_count="$(edge_metric_value "${shape_dir}/${prefix}_edge_stats.txt" "features.memory graph_edges")"
  base_edges="$(edge_metric_value "${shape_dir}/${prefix}_edge_stats.txt" "features.memory graph_base_edges")"
  upper_edges="$(edge_metric_value "${shape_dir}/${prefix}_edge_stats.txt" "features.memory graph_upper_edges")"
  build_score_evals="$(edge_metric_value "${shape_dir}/${prefix}_edge_stats.txt" "features.memory graph_build_score_evals" "score_eval_count")"
  build_candidate_visits="$(edge_metric_value "${shape_dir}/${prefix}_edge_stats.txt" "features.memory graph_build_candidate_visits" "candidate_visit_count")"
  speedup_vs_flat="$(awk -v flat="${flat_f32_latency_ns:-}" -v graph="${recall_ns}" 'BEGIN {
    if (flat == "" || graph == "" || graph + 0.0 <= 0.0) {
      print ""
    } else {
      printf "%.4f", (flat + 0.0) / (graph + 0.0)
    }
  }')"
  echo "${neighbors},${build_search},${query_search},${effective_query_search},${storage},${build_ns},${load_ns},${recall_ns},${recall_pct},${top1_ns},${top1_recall},${edge_count},${base_edges},${upper_edges},${build_score_evals},${build_candidate_visits},${flat_f32_latency_ns:-},${speedup_vs_flat}" >> "${csv_file}"
}

append_recall_rows() {
  local threshold="$1"
  local limit="$2"
  local title="$3"
  awk -F, -v threshold="${threshold}" -v limit="${limit}" -v title="${title}" '
    NR == 1 { next }
    threshold == "" || ($9 + 0.0) >= (threshold + 0.0) {
      printf "%012.3f,%012.3f,%s\n", -($9 + 0.0), ($8 + 0.0), $0
    }
  ' "${csv_file}" | sort -t, -k1,1n -k2,2n | head -n "${limit}" | awk -F, -v title="${title}" '
    BEGIN {
      print ""
      print "## " title
    }
    {
      row = $3
      for (i = 4; i <= NF; ++i) {
        row = row "," $i
      }
      print row
      seen = 1
    }
    END {
      if (!seen) {
        print "# none"
      }
    }
  ' >> "${summary_file}"
}

append_fast_rows() {
  local threshold="$1"
  local limit="$2"
  local title="$3"
  awk -F, -v threshold="${threshold}" -v limit="${limit}" -v title="${title}" '
    NR == 1 { next }
    threshold == "" || ($9 + 0.0) >= (threshold + 0.0) {
      printf "%012.3f,%s\n", ($8 + 0.0), $0
    }
  ' "${csv_file}" | sort -t, -k1,1n | head -n "${limit}" | awk -F, -v title="${title}" '
    BEGIN {
      print ""
      print "## " title
    }
    {
      row = $2
      for (i = 3; i <= NF; ++i) {
        row = row "," $i
      }
      print row
      seen = 1
    }
    END {
      if (!seen) {
        print "# none"
      }
    }
  ' >> "${summary_file}"
}

validate_thresholds() {
  if [[ -z "${min_recall_pct}" && -z "${max_recall_ns}" && -z "${min_speedup_vs_flat}" ]]; then
    return 0
  fi
  awk -F, -v min_recall="${min_recall_pct:-0}" -v max_ns="${max_recall_ns}" \
    -v min_speedup="${min_speedup_vs_flat}" '
    NR == 1 {
      next
    }
    {
      recall = $9 + 0.0
      ns = $8 + 0.0
      speedup = $18 + 0.0
      has_speedup = $18 != ""
      if (recall >= (min_recall + 0.0) && (max_ns == "" || ns <= (max_ns + 0.0)) &&
          (min_speedup == "" || (has_speedup && speedup >= (min_speedup + 0.0)))) {
        print $0
        found = 1
        exit
      }
    }
    END {
      if (!found) {
        exit 1
      }
    }
  ' "${csv_file}" > "${out_dir}/threshold_pass.csv" || {
    {
      echo
      echo "# threshold_status: failed"
      echo "# min_recall_pct: ${min_recall_pct:-0}"
      echo "# max_recall_ns: ${max_recall_ns:-none}"
      echo "# min_speedup_vs_flat: ${min_speedup_vs_flat:-none}"
    } >> "${summary_file}"
    echo "No ANN row met min_recall_pct=${min_recall_pct:-0} max_recall_ns=${max_recall_ns:-none} min_speedup_vs_flat=${min_speedup_vs_flat:-none}" >&2
    return 1
  }
  {
    echo
    echo "# threshold_status: passed"
    echo "# min_recall_pct: ${min_recall_pct:-0}"
    echo "# max_recall_ns: ${max_recall_ns:-none}"
    echo "# min_speedup_vs_flat: ${min_speedup_vs_flat:-none}"
    echo "# threshold_row: $(cat "${out_dir}/threshold_pass.csv")"
  } >> "${summary_file}"
}

if [[ "${summary_only}" == "1" ]]; then
  if [[ ! -f "${csv_file}" ]]; then
    echo "Missing CSV for summary-only run: ${csv_file}" >&2
    exit 2
  fi
  append_recall_rows "" 5 "best recall rows"
  append_fast_rows "95" 5 "fastest rows at or above 95 recall"
  validate_thresholds
  echo "# runs: 0" >> "${summary_file}"
  echo "# summary_only: 1" >> "${summary_file}"
  echo "# results_csv: ${csv_file}" >> "${summary_file}"
  echo "[memory-ann-tuning] wrote ${summary_file}"
  exit 0
fi

run_count=0
flat_baseline_captured=0
flat_f32_latency_ns=""
declare -A captured_shape_dirs=()
for neighbors in "${neighbor_values[@]}"; do
  for build_search in "${build_search_values[@]}"; do
    for query_search in "${query_search_values[@]}"; do
      effective_query_search="$(effective_query_search_for "${build_search}" "${query_search}")"
      shape_key="${neighbors}_${build_search}_${effective_query_search}"
      shape_dir="${out_dir}/n${neighbors}_b${build_search}_q${query_search}"
      source_shape_dir="${captured_shape_dirs[${shape_key}]:-}"
      if [[ -z "${source_shape_dir}" ]]; then
        shape_args=()
        if [[ "${flat_baseline_captured}" == "1" ]]; then
          shape_args+=(--skip-flat-baseline)
        fi
        scripts/run_memory_search_acceptance.sh \
          "${common_args[@]}" \
          "${shape_args[@]}" \
          --out-dir "${shape_dir}" \
          --graph-neighbors "${neighbors}" \
          --graph-search "${build_search}" \
          --query-search "${effective_query_search}"

        captured_shape_dirs["${shape_key}"]="${shape_dir}"
        flat_baseline_captured="1"
        if [[ -z "${flat_f32_latency_ns}" ]]; then
          flat_f32_latency_ns="$(metric_ns "${shape_dir}/flat_f32_latency.txt")"
        fi
        source_shape_dir="${shape_dir}"
        {
          echo
          echo "## neighbors=${neighbors} build_search=${build_search} query_search=${query_search} effective_query_search=${effective_query_search}"
          echo "# summary: ${source_shape_dir}/summary.txt"
          grep -nE "features\\.memory|recall_pct|top1_recall_pct|edge_count|level_count" "${source_shape_dir}/summary.txt" || true
        } >> "${summary_file}"
        run_count=$((run_count + 1))
      else
        {
          echo
          echo "## neighbors=${neighbors} build_search=${build_search} query_search=${query_search} effective_query_search=${effective_query_search}"
          echo "# reused: ${source_shape_dir}"
        } >> "${summary_file}"
      fi
      append_csv_row "${source_shape_dir}" "${neighbors}" "${build_search}" "${query_search}" "f32"
      append_csv_row "${source_shape_dir}" "${neighbors}" "${build_search}" "${query_search}" "q8"
      append_csv_row "${source_shape_dir}" "${neighbors}" "${build_search}" "${query_search}" "f6e2m3"
    done
  done
done

append_recall_rows "" 5 "best recall rows"
append_fast_rows "95" 5 "fastest rows at or above 95 recall"
validate_thresholds
echo "# runs: ${run_count}" >> "${summary_file}"
echo "# results_csv: ${csv_file}" >> "${summary_file}"
echo "[memory-ann-tuning] wrote ${out_dir}"
