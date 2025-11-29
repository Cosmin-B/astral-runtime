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

echo "${out_dir}"
