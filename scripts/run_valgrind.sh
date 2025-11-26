#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build/dev"
test_bin="${build_dir}/tests/gate_allocations"
log_file="${root_dir}/valgrind_memcheck.log"

if ! command -v valgrind >/dev/null 2>&1; then
  echo "[valgrind] Valgrind is not installed" >&2
  exit 1
fi

echo "[valgrind] Configure dev preset"
cmake --preset dev

echo "[valgrind] Build gate_allocations"
cmake --build --preset dev --target gate_allocations -j

if [[ ! -x "${test_bin}" ]]; then
  echo "[valgrind] Missing test binary: ${test_bin}" >&2
  exit 1
fi

echo "[valgrind] Run Memcheck: ${test_bin}"
valgrind \
  --tool=memcheck \
  --leak-check=full \
  --show-leak-kinds=all \
  --track-origins=yes \
  --error-exitcode=99 \
  --log-file="${log_file}" \
  "${test_bin}"

grep -q "ERROR SUMMARY: 0 errors" "${log_file}"
grep -q "definitely lost: 0 bytes" "${log_file}"

echo "[valgrind] OK: ${log_file}"
