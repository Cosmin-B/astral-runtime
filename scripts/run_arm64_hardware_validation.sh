#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_arm64_hardware_validation.sh [options]

Runs the native ARM64 evidence lane. This script is for real ARM64 Linux
hardware; QEMU/cross builds are covered by the embedded-arm64-ci preset.

Options:
  --preset <name>      Native CMake preset to build (default: release-with-tests)
  --embedded-preset <name>
                       Native embedded preset for example smokes (default: embedded-native)
  --jobs <N>           Build/test parallelism (default: nproc/sysctl/4)
  --items <N>          Items per producer for primitive perf capture (default: 1000000)
  --out <dir>          Evidence directory (default: ../astral-arm64-runs/<timestamp>)
  --allow-non-arm64    Run on the current host anyway
  --skip-build         Skip configure/build
  --require-perf       Fail if perf counters are unavailable
  --help               Show help
USAGE
}

preset="release-with-tests"
embedded_preset="embedded-native"
jobs=""
items="1000000"
out_dir=""
allow_non_arm64="0"
skip_build="0"
require_perf="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --embedded-preset) embedded_preset="${2:-}"; shift 2 ;;
    --jobs) jobs="${2:-}"; shift 2 ;;
    --items) items="${2:-}"; shift 2 ;;
    --out) out_dir="${2:-}"; shift 2 ;;
    --allow-non-arm64) allow_non_arm64="1"; shift ;;
    --skip-build) skip_build="1"; shift ;;
    --require-perf) require_perf="1"; shift ;;
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

machine="$(uname -m)"
case "${machine}" in
  aarch64|arm64)
    ;;
  *)
    if [[ "${allow_non_arm64}" != "1" ]]; then
      echo "native ARM64 hardware is required; current machine is ${machine}" >&2
      echo "use embedded-arm64-ci for QEMU/cross validation, or pass --allow-non-arm64 for a script smoke only" >&2
      exit 2
    fi
    ;;
esac

if [[ -z "${out_dir}" ]]; then
  out_dir="../astral-arm64-runs/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "${out_dir}/logs"
root_abs="$(pwd -P)"
out_abs="$(cd "${out_dir}" && pwd -P)"
case "${out_abs}/" in
  "${root_abs}/"*)
    echo "evidence directory must be outside the repository: ${out_abs}" >&2
    exit 2
    ;;
esac

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_rev=$(git rev-parse HEAD)"
  echo "machine=${machine}"
  echo "preset=${preset}"
  echo "embedded_preset=${embedded_preset}"
  echo "jobs=${jobs}"
  echo "items=${items}"
  uname -a | sed 's/^/uname=/'
} > "${out_dir}/run.env"

if command -v lscpu >/dev/null 2>&1; then
  lscpu > "${out_dir}/lscpu.txt" 2>&1 || true
fi

if [[ "${skip_build}" != "1" ]]; then
  cmake --preset "${preset}" 2>&1 | tee "${out_dir}/logs/configure.log"
  cmake --build --preset "${preset}" -j "${jobs}" 2>&1 | tee "${out_dir}/logs/build.log"
  cmake --preset "${embedded_preset}" 2>&1 | tee "${out_dir}/logs/configure-embedded.log"
  cmake --build --preset "${embedded_preset}" -j "${jobs}" 2>&1 | tee "${out_dir}/logs/build-embedded.log"
fi

ctest --preset "${preset}" -j "${jobs}" \
  -R '^(test_concurrency|test_platform|test_utf8|test_memory|test_model_sources)$' \
  --output-on-failure 2>&1 | tee "${out_dir}/logs/ctest-primitives.log"

test_build_dir="build/${preset}"
case "${preset}" in
  release-with-tests) test_build_dir="build/release-test" ;;
  release) test_build_dir="build/release" ;;
  dev) test_build_dir="build/dev" ;;
esac

embedded_build_dir="build/${embedded_preset}"
case "${embedded_preset}" in
  embedded-native) embedded_build_dir="build/embedded-native" ;;
  embedded-x86_64) embedded_build_dir="build/embedded-x86_64" ;;
esac

if [[ -x "${embedded_build_dir}/examples/embedded/astral_embedded_cli" ]]; then
  "${embedded_build_dir}/examples/embedded/astral_embedded_cli" \
    --backend mock --prompt hi --tokens 64 --sink none --reset \
    > "${out_dir}/logs/embedded-cli.log" 2>&1
else
  echo "missing embedded CLI: ${embedded_build_dir}/examples/embedded/astral_embedded_cli" \
    > "${out_dir}/logs/embedded-cli.log"
  exit 1
fi

if [[ -x "${embedded_build_dir}/examples/embedded/astral_concurrency_contract" ]]; then
  "${embedded_build_dir}/examples/embedded/astral_concurrency_contract" \
    > "${out_dir}/logs/embedded-concurrency-contract.log" 2>&1
else
  echo "missing embedded concurrency contract: ${embedded_build_dir}/examples/embedded/astral_concurrency_contract" \
    > "${out_dir}/logs/embedded-concurrency-contract.log"
  exit 1
fi

perf_args=()
if [[ "${require_perf}" == "1" ]]; then
  perf_args+=(--require-perf)
fi

perf_dir="${out_dir}/primitive-perf"
scripts/run_primitive_perf_capture.sh \
  --preset "${preset}" \
  --no-build \
  --pin \
  --items "${items}" \
  --out "${perf_dir}" \
  "${perf_args[@]}" > "${out_dir}/logs/primitive-perf.path"

{
  echo "# ARM64 hardware validation"
  echo
  echo "Commit: $(git rev-parse HEAD)"
  echo "Machine: ${machine}"
  echo "Preset: ${preset}"
  echo
  echo "Evidence:"
  echo "- logs/ctest-primitives.log"
  echo "- logs/embedded-cli.log"
  echo "- logs/embedded-concurrency-contract.log"
  echo "- primitive-perf/bench.log"
  echo "- primitive-perf/target-report.tsv"
  echo "- primitive-perf/disassembly.txt"
  echo "- primitive-perf/perf-stat.csv, when perf is available"
} > "${out_dir}/review.md"

echo "${out_dir}"
