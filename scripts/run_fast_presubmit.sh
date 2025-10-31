#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

preset="dev"
jobs=""
log_path=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_fast_presubmit.sh [--preset dev|dev-prof|dev-prof-micro] [--jobs N] [--log PATH]

Runs the local native presubmit lane:
  cmake --preset <preset>
  cmake --build --preset <preset> -j <jobs>
  ctest --preset <preset> -j <jobs> --output-on-failure

This lane is intentionally limited to native CMake and CTest work.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --preset" >&2
        exit 2
      fi
      preset="$2"
      shift 2
      ;;
    --jobs)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --jobs" >&2
        exit 2
      fi
      jobs="$2"
      shift 2
      ;;
    --log)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --log" >&2
        exit 2
      fi
      log_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${preset}" in
  dev|dev-prof|dev-prof-micro)
    ;;
  *)
    echo "unsupported preset for fast presubmit: ${preset}" >&2
    exit 2
    ;;
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
  ''|*[!0-9]*|0)
    echo "jobs must be a positive integer" >&2
    exit 2
    ;;
esac

if [[ -z "${log_path}" ]]; then
  log_path="build/test-logs/ctest-${preset}-fast-presubmit.log"
fi

mkdir -p "$(dirname "${log_path}")"

cmake --preset "${preset}"
cmake --build --preset "${preset}" -j "${jobs}"
ctest --preset "${preset}" -j "${jobs}" --output-on-failure 2>&1 | tee "${log_path}"
exit "${PIPESTATUS[0]}"
