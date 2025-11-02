#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${ASTRAL_TEST_MODEL:-${ROOT_DIR}/tests/models/gpt2.Q2_K.gguf}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "[INFO] Model not found at ${MODEL_PATH}, downloading default test model..."
  "${ROOT_DIR}/tests/model_downloader.sh"
fi

echo "[INFO] Building + running allocation gate (release-with-tests)..."
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j

echo "[INFO] Running embedded preset gate (release-with-tests)..."
ctest --preset release-with-tests -R gate_embedded_presets -V

# Keep the reserve low-ish to mimic edge defaults.
ASTRAL_GATE_CPU_ALLOC=1 ASTRAL_MODEL_MIN_BYTES=70000000 ctest --preset release-with-tests -R gate_allocations -V

echo "[INFO] Running I/O gate (release-with-tests)..."
ASTRAL_GATE_CPU_IO=1 ASTRAL_MODEL_MIN_BYTES=70000000 ctest --preset release-with-tests -R gate_io_syscalls -V

echo "[INFO] Running RSS cap gate (release-with-tests)..."
RSS_CAP="${ASTRAL_RSS_MAX_MB:-2048}"
ASTRAL_RSS_MAX_MB="${RSS_CAP}" ASTRAL_MODEL_MIN_BYTES=70000000 ctest --preset release-with-tests -R gate_rss_cap -V

echo "[INFO] Running embedded smoke (embedded-x86_64)..."
"${ROOT_DIR}/scripts/run_embedded_smoke.sh"
