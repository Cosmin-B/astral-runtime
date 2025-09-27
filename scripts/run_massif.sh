#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build/dev"
test_bin="${build_dir}/tests/gate_allocations"
massif_out="${root_dir}/massif.out"
massif_report="${root_dir}/massif_report.txt"

if ! command -v valgrind >/dev/null 2>&1; then
  echo "[massif] Valgrind is not installed" >&2
  exit 1
fi

echo "[massif] Configure dev preset"
cmake --preset dev

echo "[massif] Build gate_allocations"
cmake --build --preset dev --target gate_allocations -j

if [[ ! -x "${test_bin}" ]]; then
  echo "[massif] Missing test binary: ${test_bin}" >&2
  exit 1
fi

echo "[massif] Run Massif: ${test_bin}"
valgrind \
  --tool=massif \
  --massif-out-file="${massif_out}" \
  --detailed-freq=1 \
  --max-snapshots=100 \
  --threshold=0.1 \
  --time-unit=ms \
  "${test_bin}"

if command -v ms_print >/dev/null 2>&1; then
  ms_print "${massif_out}" > "${massif_report}"
  echo "[massif] Report: ${massif_report}"
else
  echo "[massif] ms_print not found; raw output: ${massif_out}"
fi
