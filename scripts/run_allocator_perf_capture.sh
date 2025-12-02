#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_allocator_perf_capture.sh [options]

Captures allocator benchmark output, target ranges, symbols, and disassembly.

Options:
  --preset <name>      CMake preset to use (default: release-with-tests)
  --jobs <N>           Build parallelism (default: nproc/sysctl/4)
  --iters <N>          Allocation iterations (default: 1000000)
  --size <N>           Allocation size in bytes (default: 64)
  --threads <N>        Contention worker count (default: 4)
  --out <dir>          Artifact directory (default: ../astral-alloc-runs/<timestamp>)
  --no-build           Skip configure/build
  --require-targets    Fail if target-report.tsv contains review rows
  --help               Show help
USAGE
}

preset="release-with-tests"
jobs=""
iters="1000000"
size="64"
threads="4"
out_dir=""
build="1"
require_targets="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --jobs) jobs="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --size) size="${2:-}"; shift 2 ;;
    --threads) threads="${2:-}"; shift 2 ;;
    --out) out_dir="${2:-}"; shift 2 ;;
    --no-build) build="0"; shift ;;
    --require-targets) require_targets="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in iters size threads; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*|0) echo "--${value_name} must be a positive integer" >&2; exit 2 ;;
  esac
done

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
  out_dir="../astral-alloc-runs/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "${out_dir}"

bench_cmd=("${bench_bin}" --only alloc --alloc-iters "${iters}" --alloc-size "${size}" --mpsc-producers "${threads}")

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_rev=$(git rev-parse HEAD)"
  echo "preset=${preset}"
  echo "bench_bin=${bench_bin}"
  echo "iters=${iters}"
  echo "size=${size}"
  echo "threads=${threads}"
  echo "command=${bench_cmd[*]}"
  uname -a | sed 's/^/uname=/'
} > "${out_dir}/run.env"

if command -v lscpu >/dev/null 2>&1; then
  lscpu > "${out_dir}/lscpu.txt" 2>&1 || true
fi

"${bench_cmd[@]}" > "${out_dir}/bench.log" 2>&1

awk '
  function ns_for(label, line, fields, count, i) {
    if (index(line, label) != 1) return ""
    count = split(line, fields, " ")
    for (i = 1; i <= count; ++i) {
      if (fields[i] == "ns/op") return fields[i - 1]
    }
    return ""
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
    emit("frame_allocator", ns_for("frame_allocator alloc/reset", $0), "<=2.00", 2.00)
    emit("object_pool_single", ns_for("object_pool acquire/release", $0), "<=10.00", 10.00)
    emit("object_pool_contended", ns_for("object_pool contended", $0), "<=200.00", 200.00)
    emit("runtime_alloc_vm", ns_for("runtime_alloc/free (vm)", $0), "<=25.00", 25.00)
    emit("runtime_alloc_arena", ns_for("runtime_alloc/free (arena)", $0), "<=20.00", 20.00)
  }
' "${out_dir}/bench.log" > "${out_dir}/target-report.tsv"

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
  rg -n "FrameAllocator|ObjectPool|runtime_alloc|runtime_free|bench_frame_allocator|bench_object_pool|bench_runtime_alloc" \
    "${out_dir}/disassembly.txt" > "${out_dir}/disassembly-hot-index.txt" 2>/dev/null || true
fi

if [[ "${require_targets}" == "1" ]] && awk -F '\t' 'NR > 1 && $4 != "ok" { found = 1 } END { exit found ? 0 : 1 }' "${out_dir}/target-report.tsv"; then
  echo "target-report.tsv contains review rows" >&2
  exit 1
fi

echo "${out_dir}"
