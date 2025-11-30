#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_primitive_perf_capture.sh [options]

Captures primitive benchmark output, perf counters, symbols, and disassembly.

Options:
  --preset <name>      CMake preset to use (default: release-with-tests)
  --jobs <N>           Build parallelism (default: nproc/sysctl/4)
  --items <N>          Items per producer for the matrix (default: 1000000)
  --out <dir>          Artifact directory (default: ../astral-perf-runs/<timestamp>)
  --pin                Set ASTRAL_BENCH_PIN_THREADS=1
  --no-build           Skip configure/build
  --require-perf       Fail if perf is missing or blocked
  --perf-events <csv>  Override perf stat events
  --help               Show help
USAGE
}

preset="release-with-tests"
jobs=""
items="1000000"
out_dir=""
pin="0"
build="1"
require_perf="0"
perf_events="cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,dTLB-loads,dTLB-load-misses"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --jobs) jobs="${2:-}"; shift 2 ;;
    --items) items="${2:-}"; shift 2 ;;
    --out) out_dir="${2:-}"; shift 2 ;;
    --pin) pin="1"; shift ;;
    --no-build) build="0"; shift ;;
    --require-perf) require_perf="1"; shift ;;
    --perf-events) perf_events="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${items}" in
  ''|*[!0-9]*|0) echo "--items must be a positive integer" >&2; exit 2 ;;
esac

if [[ -z "${jobs}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    jobs="$(sysctl -n hw.ncpu)"
  else
    jobs="4"
  fi
fi

case "${jobs}" in
  ''|*[!0-9]*|0) echo "--jobs must be a positive integer" >&2; exit 2 ;;
esac

build_dir="build/${preset}"
case "${preset}" in
  release-with-tests) build_dir="build/release-test" ;;
  release) build_dir="build/release" ;;
  dev) build_dir="build/dev" ;;
  dev-prof) build_dir="build/dev-prof" ;;
  dev-prof-micro) build_dir="build/dev-prof-micro" ;;
  release-prof) build_dir="build/release-prof" ;;
esac

bench_bin="${build_dir}/benchmarks/astral_benchmarks"

if [[ "${build}" == "1" ]]; then
  cmake --preset "${preset}"
  cmake --build --preset "${preset}" --target astral_benchmarks -j "${jobs}"
fi

if [[ ! -x "${bench_bin}" ]]; then
  echo "benchmark binary not found: ${bench_bin}" >&2
  exit 2
fi

if [[ -z "${out_dir}" ]]; then
  out_dir="../astral-perf-runs/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "${out_dir}"

bench_cmd=("${bench_bin}" --only concurrency-matrix --mpsc-items "${items}")

write_target_report() {
  local bench_log="$1"
  local report="$2"

  awk '
    function ns_for(label, line, fields, count, i) {
      if (index(line, label) != 1) return ""
      count = split(line, fields, " ")
      for (i = 1; i <= count; ++i) {
        if (fields[i] == "ns/op") return fields[i - 1]
      }
      return ""
    }
    function p50_for(label, line, start, value) {
      if (index(line, label) != 1) return ""
      start = index(line, "p50=")
      if (start == 0) return ""
      value = substr(line, start + 4)
      sub(/^[[:space:]]*/, "", value)
      sub(/[[:space:]].*/, "", value)
      return value
    }
    function prod_wall_for(label, line, start, value) {
      if (index(line, label) != 1) return ""
      start = index(line, "prod-wall=")
      if (start == 0) return ""
      value = substr(line, start + 10)
      sub(/^[[:space:]]*/, "", value)
      sub(/[[:space:]].*/, "", value)
      return value
    }
    function emit(name, actual, target, upper) {
      if (actual == "") return
      status = ((actual + 0.0) <= upper) ? "ok" : "review"
      printf "%s\t%s\t%s\t%s\n", name, actual, target, status
    }
    BEGIN {
      print "case\tactual_ns_per_op\ttarget_ns_per_op\tstatus"
    }
    {
      emit("spsc_batch", ns_for("SPSC batch", $0), "<=0.50", 0.50)
      emit("spsc_cached_local", ns_for("SPSC cached local", $0), "<=0.75", 0.75)
      emit("spsc_local_p50", p50_for("SPSC local pcts", $0), "<=5.00", 5.00)
      emit("spsc_transit", ns_for("SPSC 1P/1C transit", $0), "<=25.00", 25.00)
      emit("spsc_fan_in_4p", ns_for("SPSC fan-in 4P/1C", $0), "<=60.00", 60.00)
      emit("mpsc_local_p50", p50_for("MPSC local pcts", $0), "<=15.00", 15.00)
      emit("mpsc_ticket_local_p50", p50_for("MPSC ticket local pcts", $0), "<=15.00", 15.00)
      emit("mpsc_cas_1p", ns_for("MPSC 1P/1C", $0), "<=35.00", 35.00)
      emit("mpsc_cas_2p", ns_for("MPSC 2P/1C", $0), "<=60.00", 60.00)
      emit("mpsc_cas_4p", ns_for("MPSC 4P/1C", $0), "<=80.00", 80.00)
      emit("mpsc_cas_8p", ns_for("MPSC 8P/1C", $0), "<=200.00", 200.00)
      emit("mpsc_ticket_1p", ns_for("MPSC ticket 1P/1C", $0), "<=20.00", 20.00)
      emit("mpsc_ticket_2p", ns_for("MPSC ticket 2P/1C", $0), "<=60.00", 60.00)
      emit("mpsc_ticket_4p", ns_for("MPSC ticket 4P/1C", $0), "<=60.00", 60.00)
      emit("mpsc_ticket_8p", ns_for("MPSC ticket 8P/1C", $0), "<=200.00", 200.00)
      emit("mpsc_cas_split_1p_prod_wall", prod_wall_for("MPSC split 1P", $0), "<=35.00", 35.00)
      emit("mpsc_cas_split_2p_prod_wall", prod_wall_for("MPSC split 2P", $0), "<=80.00", 80.00)
      emit("mpsc_cas_split_4p_prod_wall", prod_wall_for("MPSC split 4P", $0), "<=120.00", 120.00)
      emit("mpsc_cas_split_8p_prod_wall", prod_wall_for("MPSC split 8P", $0), "<=250.00", 250.00)
      emit("mpsc_ticket_split_1p_prod_wall", prod_wall_for("MPSC ticket split 1P", $0), "<=25.00", 25.00)
      emit("mpsc_ticket_split_2p_prod_wall", prod_wall_for("MPSC ticket split 2P", $0), "<=80.00", 80.00)
      emit("mpsc_ticket_split_4p_prod_wall", prod_wall_for("MPSC ticket split 4P", $0), "<=80.00", 80.00)
      emit("mpsc_ticket_split_8p_prod_wall", prod_wall_for("MPSC ticket split 8P", $0), "<=250.00", 250.00)
      emit("mpmc_local_p50", p50_for("MPMC local pcts", $0), "<=25.00", 25.00)
      emit("mpmc_1p1c", ns_for("MPMC 1P/1C", $0), "<=25.00", 25.00)
      emit("mpmc_4p4c_spaced", ns_for("MPMC 4P/4C spaced", $0), "<=80.00", 80.00)
      emit("mpmc_4p4c_dense", ns_for("MPMC 4P/4C dense", $0), "<=150.00", 150.00)
      emit("mpmc_8p8c", ns_for("MPMC 8P/8C", $0), "<=300.00", 300.00)
    }
  ' "${bench_log}" > "${report}"
}

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_rev=$(git rev-parse HEAD)"
  echo "preset=${preset}"
  echo "bench_bin=${bench_bin}"
  echo "items=${items}"
  echo "pin=${pin}"
  echo "command=${bench_cmd[*]}"
  uname -a | sed 's/^/uname=/'
} > "${out_dir}/run.env"

if command -v lscpu >/dev/null 2>&1; then
  lscpu > "${out_dir}/lscpu.txt" 2>&1 || true
fi

if command -v perf >/dev/null 2>&1; then
  set +e
  ASTRAL_BENCH_PIN_THREADS="${pin}" \
    perf stat -x, -e "${perf_events}" -o "${out_dir}/perf-stat.csv" -- "${bench_cmd[@]}" \
    > "${out_dir}/bench.log" 2> "${out_dir}/perf.stderr"
  perf_status=$?
  set -e
  if [[ ${perf_status} -ne 0 ]]; then
    if [[ "${require_perf}" == "1" ]]; then
      exit "${perf_status}"
    fi
    {
      echo "perf stat failed with exit ${perf_status}"
      cat "${out_dir}/perf.stderr"
      echo
      echo "rerunning benchmark without perf"
    } > "${out_dir}/perf-unavailable.txt"
    ASTRAL_BENCH_PIN_THREADS="${pin}" "${bench_cmd[@]}" > "${out_dir}/bench.log" 2>&1
  fi
else
  if [[ "${require_perf}" == "1" ]]; then
    echo "perf not found" >&2
    exit 2
  fi
  echo "perf not found" > "${out_dir}/perf-unavailable.txt"
  ASTRAL_BENCH_PIN_THREADS="${pin}" "${bench_cmd[@]}" > "${out_dir}/bench.log" 2>&1
fi

if command -v nm >/dev/null 2>&1; then
  nm -C --defined-only "${bench_bin}" > "${out_dir}/symbols.txt" 2>&1 || true
fi

objdump_bin=""
if command -v llvm-objdump >/dev/null 2>&1; then
  objdump_bin="llvm-objdump"
elif command -v objdump >/dev/null 2>&1; then
  objdump_bin="objdump"
fi

if [[ -n "${objdump_bin}" ]]; then
  "${objdump_bin}" -Cd --no-show-raw-insn "${bench_bin}" > "${out_dir}/disassembly.txt" 2>&1 || true
  rg -n "SpscRing|SpscFanIn|MpscRing|MpscTicketRing|MpmcQueue|bench_spsc|bench_mpsc|bench_mpmc|fetch_add|pause|wfe|sev" \
    "${out_dir}/disassembly.txt" > "${out_dir}/disassembly-hot-index.txt" 2>/dev/null || true
fi

write_target_report "${out_dir}/bench.log" "${out_dir}/target-report.tsv"

echo "${out_dir}"
