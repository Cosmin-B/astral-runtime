#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build/asan"

rm -rf "${build_dir}"

echo "[asan] Configure: ${build_dir}"
cmake -S "${root_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_C_FLAGS="-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address -fsanitize=undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address -fsanitize=undefined" \
  -DASTRAL_BUILD_TESTS=ON \
  -DASTRAL_BUILD_BENCHMARKS=OFF

echo "[asan] Build memory and concurrency gates"
cmake --build "${build_dir}" --target test_concurrency gate_allocations gate_rss_cap -j

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:check_initialization_order=1:strict_init_order=1:detect_stack_use_after_return=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1}"

echo "[asan] Run memory and concurrency gates"
ctest --test-dir "${build_dir}" -R '^(test_concurrency|gate_allocations|gate_rss_cap)$' -V --output-on-failure
