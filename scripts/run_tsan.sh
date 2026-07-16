#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build/tsan"

rm -rf "${build_dir}"

echo "[tsan] Configure: ${build_dir}"
cmake -S "${root_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g" \
  -DCMAKE_C_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=thread" \
  -DASTRAL_BUILD_TESTS=ON \
  -DASTRAL_BUILD_TSAN_TESTS=ON \
  -DASTRAL_BUILD_BENCHMARKS=OFF

echo "[tsan] Build concurrency, memory, and inference race gates"
cmake --build "${build_dir}" \
  --target test_concurrency_tsan test_memory_tsan test_inference_tsan -j

export TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1:second_deadlock_stack=1:history_size=7}"

echo "[tsan] Run ThreadSanitizer gates"
ctest --test-dir "${build_dir}" -L tsan -V --output-on-failure
