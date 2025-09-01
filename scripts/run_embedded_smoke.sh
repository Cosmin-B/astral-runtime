#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

MODEL_PATH="${ASTRAL_TEST_MODEL:-${ROOT_DIR}/tests/models/gpt2.Q2_K.gguf}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "[INFO] Model not found at ${MODEL_PATH}, downloading default test model..."
  "${ROOT_DIR}/tests/model_downloader.sh"
fi

echo "[INFO] Configuring embedded-x86_64 preset..."
cmake --preset embedded-x86_64

echo "[INFO] Building embedded-x86_64 preset..."
cmake --build --preset embedded-x86_64 -j

echo "[INFO] Running embedded CLI smoke..."
"${ROOT_DIR}/build/embedded-x86_64/examples/embedded/astral_embedded_cli" \
  --model "${MODEL_PATH}" \
  --prompt "hi" \
  --sink none \
  --tokens 64 \
  --reset
