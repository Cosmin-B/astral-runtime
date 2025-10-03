#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

# Prefer the system-installed toolkit under /usr/local/cuda when present (common on CUDA boxes).
if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDAToolkit_ROOT="/usr/local/cuda"
  export CUDA_PATH="/usr/local/cuda"
fi

usage() {
  cat <<'EOF'
Usage: scripts/run_hf_bench_matrix.sh [options]

Runs the canonical feature-surface benchmark across a directory of downloaded HF GGUF files.
Produces consolidated logs for CPU and CUDA.

This is intended for a CUDA machine (e.g. Hetzner) but CPU runs work anywhere.

Options:
  --models-dir <dir>        Directory containing downloaded *.gguf (default: tests/models/hf)
  --out-dir <dir>           Output directory (default: benchmarks/results)
  --cpu-preset <preset>     CPU preset (default: release-with-tests)
  --cuda-preset <preset>    CUDA preset (default: dev-cuda)
  --cuda-cublas-preset <p>  Forced cuBLAS preset (default: dev-cuda-cublas)
  --cuda-mmq-preset <p>     Forced MMQ preset (default: dev-cuda-mmq)
  --iters <N>               Feature bench iters (default: 25)
  --tokens <N>              Logprobs drain tokens (default: 32)
  --gpu-layers <N>          Default GPU layers for CUDA runs (default: 48)
  --arch <list>             ASTRAL_CUDA_ARCHITECTURES override for CUDA configure (e.g. 120a-real)
  --only <cpu|cuda|all>     Which runs to execute (default: all)
  --allow-failures          Write failed rows but exit successfully for local investigation
  --help                    Show help

Behavior:
  - For each "generative" model file, runs features with:
      model=<file>, embed_model=<baseline embedding GGUF (MiniLM if present)>
  - For each "embedding" model file, runs features with:
      model=<baseline generative GGUF (gpt2)>, embed_model=<file>

Heuristic classification:
  - A file is treated as "embedding" if its filename contains 'embed', 'embedding', 'bge-', or 'jina-embeddings'.
EOF
}

models_dir="tests/models/hf"
out_dir="benchmarks/results"
cpu_preset="release-with-tests"
cuda_preset="dev-cuda"
cuda_cublas_preset="dev-cuda-cublas"
cuda_mmq_preset="dev-cuda-mmq"
iters="25"
tokens="32"
gpu_layers="48"
arch_override=""
only="all"
allow_failures=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) models_dir="${2:-}"; shift 2 ;;
    --out-dir) out_dir="${2:-}"; shift 2 ;;
    --cpu-preset) cpu_preset="${2:-}"; shift 2 ;;
    --cuda-preset) cuda_preset="${2:-}"; shift 2 ;;
    --cuda-cublas-preset) cuda_cublas_preset="${2:-}"; shift 2 ;;
    --cuda-mmq-preset) cuda_mmq_preset="${2:-}"; shift 2 ;;
    --iters) iters="${2:-}"; shift 2 ;;
    --tokens) tokens="${2:-}"; shift 2 ;;
    --gpu-layers) gpu_layers="${2:-}"; shift 2 ;;
    --arch) arch_override="${2:-}"; shift 2 ;;
    --only) only="${2:-}"; shift 2 ;;
    --allow-failures) allow_failures=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "${out_dir}"

if [[ ! -d "${models_dir}" ]]; then
  echo "Models dir not found: ${models_dir}" >&2
  exit 2
fi

# Baselines (keep these stable so comparisons are meaningful).
baseline_gen="tests/models/gpt2.Q2_K.gguf"
baseline_emb="tests/models/all-MiniLM-L6-v2-Q2_K.gguf"

if [[ ! -f "${baseline_gen}" ]]; then
  echo "Missing baseline generative model: ${baseline_gen}" >&2
  echo "Download with: ./tests/model_downloader.sh --preset gpt2-q2k" >&2
  exit 2
fi
if [[ ! -f "${baseline_emb}" ]]; then
  echo "Missing baseline embedding model: ${baseline_emb}" >&2
  echo "Download with: ./tests/model_downloader.sh --preset embed-minilm-q2k" >&2
  exit 2
fi

timestamp="$(date -Iseconds | tr ':' '-')"

cpu_log="${out_dir}/hf-matrix-${timestamp}-cpu.txt"
cuda_log="${out_dir}/hf-matrix-${timestamp}-cuda-auto.txt"
cuda_cublas_log="${out_dir}/hf-matrix-${timestamp}-cuda-cublas.txt"
cuda_mmq_log="${out_dir}/hf-matrix-${timestamp}-cuda-mmq.txt"

declare -A built_presets=()

ensure_built() {
  local preset="$1"
  local backend="$2"

  if [[ -n "${built_presets[${preset}]:-}" ]]; then
    return 0
  fi

  local cmake_args=()
  if [[ "${backend}" == "cuda" && -x "/usr/local/cuda/bin/nvcc" ]]; then
    cmake_args+=("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
  fi
  if [[ -n "${arch_override}" && "${backend}" == "cuda" ]]; then
    cmake_args+=("-DASTRAL_CUDA_ARCHITECTURES=${arch_override}")
  fi

  if [[ ${#cmake_args[@]} -gt 0 ]]; then
    cmake --preset "${preset}" "${cmake_args[@]}" >/dev/null
  else
    cmake --preset "${preset}" >/dev/null
  fi

  cmake --build --preset "${preset}" -j >/dev/null
  built_presets["${preset}"]="1"
}

choose_cuda_layers() {
  local model="$1"

  if [[ ! -f "${model}" ]]; then
    echo "${gpu_layers}"
    return 0
  fi

  # Heuristic: large models often OOM when too many layers are offloaded.
  local bytes
  bytes="$(stat -c '%s' "${model}" 2>/dev/null || echo 0)"

  # > 40 GB -> start small; > 20 GB -> moderate; else default.
  if [[ "${bytes}" -gt 42949672960 ]]; then
    echo "16"
  elif [[ "${bytes}" -gt 21474836480 ]]; then
    echo "32"
  else
    echo "${gpu_layers}"
  fi
}

run_one() {
  local preset="$1"
  local backend="$2"
  local model="$3"
  local embed_model="$4"
  local out="$5"
  local layers="$6"

  {
    echo
    echo "## preset=${preset} backend=${backend}"
    echo "model=${model}"
    echo "embed_model=${embed_model}"
  } >> "${out}"

  ensure_built "${preset}" "${backend}"

  local attempt_layers="${layers}"
  local attempt=0
  local ok="0"
  while true; do
    attempt=$((attempt + 1))
    echo "[bench] attempt=${attempt} gpu_layers=${attempt_layers}" >> "${out}"

    if ./scripts/run_ci_bench_features.sh \
        --preset "${preset}" \
        --backend "${backend}" \
        --model "${model}" \
        --embed-model "${embed_model}" \
        --gpu-layers "${attempt_layers}" \
        --iters "${iters}" \
        --tokens "${tokens}" \
        --out /tmp/astral_hf_bench_tmp.txt; then
      ok="1"
      cat /tmp/astral_hf_bench_tmp.txt >> "${out}"
      break
    fi

    cat /tmp/astral_hf_bench_tmp.txt >> "${out}" || true
    if [[ "${backend}" != "cuda" ]]; then
      break
    fi
    if [[ "${attempt_layers}" -le 1 || "${attempt}" -ge 4 ]]; then
      break
    fi
    attempt_layers=$((attempt_layers / 2))
  done

  if [[ "${ok}" != "1" ]]; then
    echo "[bench] FAILED model=${model}" >> "${out}"
  fi
}

is_embedding_file() {
  local base
  base="$(basename "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "${base}" == *"embed"* ]] || [[ "${base}" == *"embedding"* ]] || [[ "${base}" == bge-* ]] || [[ "${base}" == *"jina-embeddings"* ]]
}

is_aux_gguf_file() {
  local base
  base="$(basename "$1" | tr '[:upper:]' '[:lower:]')"
  # Common multimodal aux assets that are not standalone LLMs:
  # - mmproj-*: vision/audio projector weights
  # - vocoder-*: audio vocoder weights
  # - tokenizer-*: audio tokenizer/codec weights
  [[ "${base}" == mmproj-* ]] || [[ "${base}" == vocoder-* ]] || [[ "${base}" == tokenizer-* ]]
}

mapfile -t ggufs < <(find "${models_dir}" -type f -name "*.gguf" | sort)
if [[ ${#ggufs[@]} -eq 0 ]]; then
  echo "No .gguf files found under: ${models_dir}" >&2
  exit 2
fi

{
  echo "# Astral HF bench matrix"
  echo "# date: $(date -Iseconds)"
  echo "# models_dir: ${models_dir}"
  echo "# iters: ${iters}"
  echo "# tokens: ${tokens}"
  echo "# baseline_gen: ${baseline_gen}"
  echo "# baseline_emb: ${baseline_emb}"
  echo "# arch_override: ${arch_override}"
  echo
} > "${cpu_log}"

if [[ "${only}" == "cuda" || "${only}" == "all" ]]; then
  {
    echo "# Astral HF bench matrix (CUDA auto)"
    echo "# date: $(date -Iseconds)"
    echo "# models_dir: ${models_dir}"
    echo "# iters: ${iters}"
    echo "# tokens: ${tokens}"
    echo "# gpu_layers: ${gpu_layers}"
    echo "# baseline_gen: ${baseline_gen}"
    echo "# baseline_emb: ${baseline_emb}"
    echo "# arch_override: ${arch_override}"
    echo
  } > "${cuda_log}"
  cp "${cuda_log}" "${cuda_cublas_log}"
  cp "${cuda_log}" "${cuda_mmq_log}"
fi

for m in "${ggufs[@]}"; do
  if is_aux_gguf_file "${m}"; then
    # These are typically paired with a main model and require dedicated multimodal support.
    # Skip them for the current feature-surface bench.
    if [[ "${only}" == "cpu" || "${only}" == "all" ]]; then
      {
        echo
        echo "## preset=${cpu_preset} backend=cpu"
        echo "model=${m}"
        echo "embed_model=(skipped: aux gguf)"
        echo "[bench] SKIPPED aux_gguf=1"
      } >> "${cpu_log}"
    fi
    if [[ "${only}" == "cuda" || "${only}" == "all" ]]; then
      for out in "${cuda_log}" "${cuda_cublas_log}" "${cuda_mmq_log}"; do
        {
          echo
          echo "## preset=${cuda_preset} backend=cuda"
          echo "model=${m}"
          echo "embed_model=(skipped: aux gguf)"
          echo "[bench] SKIPPED aux_gguf=1"
        } >> "${out}"
      done
    fi
    continue
  fi

  if is_embedding_file "${m}"; then
    # Benchmark this embedding model as the embedding surface; use stable generative model for KV/grammar/logprobs.
    if [[ "${only}" == "cpu" || "${only}" == "all" ]]; then
      run_one "${cpu_preset}" "cpu" "${baseline_gen}" "${m}" "${cpu_log}" "0"
    fi
    if [[ "${only}" == "cuda" || "${only}" == "all" ]]; then
      run_one "${cuda_preset}" "cuda" "${baseline_gen}" "${m}" "${cuda_log}" "${gpu_layers}"
      run_one "${cuda_cublas_preset}" "cuda" "${baseline_gen}" "${m}" "${cuda_cublas_log}" "${gpu_layers}"
      run_one "${cuda_mmq_preset}" "cuda" "${baseline_gen}" "${m}" "${cuda_mmq_log}" "${gpu_layers}"
    fi
  else
    # Benchmark this model as the generative surface; use stable embedding model for embedding roundtrip.
    if [[ "${only}" == "cpu" || "${only}" == "all" ]]; then
      run_one "${cpu_preset}" "cpu" "${m}" "${baseline_emb}" "${cpu_log}" "0"
    fi
    if [[ "${only}" == "cuda" || "${only}" == "all" ]]; then
      layers="$(choose_cuda_layers "${m}")"
      run_one "${cuda_preset}" "cuda" "${m}" "${baseline_emb}" "${cuda_log}" "${layers}"
      run_one "${cuda_cublas_preset}" "cuda" "${m}" "${baseline_emb}" "${cuda_cublas_log}" "${layers}"
      run_one "${cuda_mmq_preset}" "cuda" "${m}" "${baseline_emb}" "${cuda_mmq_log}" "${layers}"
    fi
  fi
done

echo "[hf-matrix] wrote:"
echo "  ${cpu_log}"
matrix_logs=("${cpu_log}")
if [[ "${only}" == "cuda" || "${only}" == "all" ]]; then
  echo "  ${cuda_log}"
  echo "  ${cuda_cublas_log}"
  echo "  ${cuda_mmq_log}"
  matrix_logs+=("${cuda_log}" "${cuda_cublas_log}" "${cuda_mmq_log}")
fi

if grep -H '^\[bench\] FAILED' "${matrix_logs[@]}"; then
  if [[ "${allow_failures}" -eq 1 ]]; then
    echo "[hf-matrix] failed rows recorded; continuing because --allow-failures was set" >&2
  else
    echo "[hf-matrix] failed rows recorded; rerun with --allow-failures only for local investigation" >&2
    exit 1
  fi
fi
