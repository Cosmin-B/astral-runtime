#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_multimodal_validation.sh [options]

Builds an ASTRAL_ENABLE_MTMD=ON preset and runs the required real media gate.
The gate fails before CTest when any required model/projector fixture is absent.

Options:
  --preset <name>          CMake preset to build/test (default: release-with-tests-mtmd)
  --backend <cpu|cuda>     Backend for feature bench validation (default: cpu)
  --model <path>           Text GGUF for feature bench (default: tests/models/gpt2.Q2_K.gguf)
  --vision-model <path>    Vision model GGUF
  --vision-media <path>    Vision projector/media GGUF
  --audio-model <path>     Audio model GGUF
  --audio-media <path>     Audio projector/media GGUF
  --check-fixtures         Validate fixture paths and sizes, then exit
  --bench                  Also require feature bench media feed rows
  --out <file>             Bench log path (default: benchmarks/results/mtmd-features.txt)
  --gpu-layers <N>         Bench GPU layers when backend=cuda (default: 32)
  --help                   Show help

Environment fallbacks:
  ASTRAL_TEST_VISION_MODEL, ASTRAL_TEST_VISION_MEDIA
  ASTRAL_TEST_AUDIO_MODEL, ASTRAL_TEST_AUDIO_MEDIA
  ASTRAL_MTMD_MIN_MODEL_BYTES, ASTRAL_MTMD_MIN_MEDIA_BYTES
EOF
}

preset="release-with-tests-mtmd"
backend="cpu"
model="tests/models/gpt2.Q2_K.gguf"
vision_model="${ASTRAL_TEST_VISION_MODEL:-}"
vision_media="${ASTRAL_TEST_VISION_MEDIA:-}"
audio_model="${ASTRAL_TEST_AUDIO_MODEL:-}"
audio_media="${ASTRAL_TEST_AUDIO_MEDIA:-}"
min_model_bytes="${ASTRAL_MTMD_MIN_MODEL_BYTES:-$((100 * 1024 * 1024))}"
min_media_bytes="${ASTRAL_MTMD_MIN_MEDIA_BYTES:-$((1 * 1024 * 1024))}"
check_fixtures=0
bench=0
out_file="benchmarks/results/mtmd-features.txt"
gpu_layers="32"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --backend) backend="${2:-}"; shift 2 ;;
    --model) model="${2:-}"; shift 2 ;;
    --vision-model) vision_model="${2:-}"; shift 2 ;;
    --vision-media) vision_media="${2:-}"; shift 2 ;;
    --audio-model) audio_model="${2:-}"; shift 2 ;;
    --audio-media) audio_media="${2:-}"; shift 2 ;;
    --check-fixtures) check_fixtures=1; shift ;;
    --bench) bench=1; shift ;;
    --out) out_file="${2:-}"; shift 2 ;;
    --gpu-layers) gpu_layers="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

require_file() {
  local label="$1"
  local path="$2"
  local min_bytes="$3"

  if [[ -z "${path}" ]]; then
    echo "[mtmd] missing ${label}" >&2
    exit 2
  fi
  if [[ ! -f "${path}" ]]; then
    echo "[mtmd] ${label} not found: ${path}" >&2
    exit 2
  fi

  local bytes
  bytes="$(wc -c < "${path}")"
  if (( bytes < min_bytes )); then
    echo "[mtmd] ${label} too small: ${path} (${bytes} bytes, need ${min_bytes})" >&2
    exit 2
  fi
}

require_file "vision model" "${vision_model}" "${min_model_bytes}"
require_file "vision media/projector" "${vision_media}" "${min_media_bytes}"
require_file "audio model" "${audio_model}" "${min_model_bytes}"
require_file "audio media/projector" "${audio_media}" "${min_media_bytes}"

if [[ "${check_fixtures}" -eq 1 ]]; then
  echo "[mtmd] fixture preflight OK"
  echo "[mtmd] vision model: ${vision_model}"
  echo "[mtmd] vision media/projector: ${vision_media}"
  echo "[mtmd] audio model: ${audio_model}"
  echo "[mtmd] audio media/projector: ${audio_media}"
  exit 0
fi

echo "[mtmd] Configure: ${preset}"
cmake --preset "${preset}"

echo "[mtmd] Build: ${preset}"
cmake --build --preset "${preset}" -j

echo "[mtmd] Test: test_media with required real fixtures"
ASTRAL_TEST_REQUIRE_MEDIA=1 \
ASTRAL_TEST_VISION_MODEL="${vision_model}" \
ASTRAL_TEST_VISION_MEDIA="${vision_media}" \
ASTRAL_TEST_AUDIO_MODEL="${audio_model}" \
ASTRAL_TEST_AUDIO_MEDIA="${audio_media}" \
  ctest --preset "${preset}" -R '^test_media$' -V

if [[ "${bench}" -eq 1 ]]; then
  echo "[mtmd] Bench: media feed surface"
  ASTRAL_BENCH_VISION_MODEL="${vision_model}" \
  ASTRAL_BENCH_VISION_MEDIA="${vision_media}" \
  ASTRAL_BENCH_AUDIO_MODEL="${audio_model}" \
  ASTRAL_BENCH_AUDIO_MEDIA="${audio_media}" \
    scripts/run_ci_bench_features.sh \
      --preset "${preset}" \
      --backend "${backend}" \
      --model "${model}" \
      --out "${out_file}" \
      --gpu-layers "${gpu_layers}"

  if grep -q '\[bench\] media init failed' "${out_file}"; then
    echo "[mtmd] media bench initialization failed; see ${out_file}" >&2
    exit 1
  fi
  grep -q 'features.media feed_image' "${out_file}" || {
    echo "[mtmd] missing vision feed benchmark row in ${out_file}" >&2
    exit 1
  }
  grep -q 'features.media feed_audio' "${out_file}" || {
    echo "[mtmd] missing audio feed benchmark row in ${out_file}" >&2
    exit 1
  }
fi

echo "[mtmd] validation passed"
