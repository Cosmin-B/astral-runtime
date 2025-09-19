#!/usr/bin/env bash
set -euo pipefail

# Lightweight “cron-friendly” watchdog for long-running download/bench tasks on a remote box.
#
# Responsibilities:
# - Keep HF downloads running when there are pending *.part files.
# - Keep wait+bench jobs running so full matrices execute automatically once downloads finish.
# - Never deletes files; only restarts jobs if missing.

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

log_dir="benchmarks/results"
mkdir -p "${log_dir}"
log_file="${log_dir}/watchdog.log"

now="$(date -Iseconds)"

repo_download_running() {
  pgrep -af "python3 scripts/hf_gguf_sync.py download" >/dev/null 2>&1
}

ensure_download() {
  local models_dir="$1"   # tests/models/hf or tests/models/hf-lfm25
  local script="$2"       # scripts/hf_gguf_download_manifest.sh or scripts/hf_gguf_download_lfm25_all.sh
  local log="$3"          # log path

  local parts
  parts="$(find "${models_dir}" -type f -name "*.part" 2>/dev/null | wc -l)"
  if [[ "${parts}" -eq 0 ]]; then
    return 0
  fi

  if repo_download_running; then
    return 0
  fi

  echo "[${now}] start download: ${script} -> ${models_dir}" >> "${log_file}"
  nohup "./${script}" > "${log}" 2>&1 </dev/null &
}

ensure_wait_and_bench() {
  local models_dir="$1"
  local log="$2"

  if pgrep -af "scripts/run_hf_wait_and_bench.sh --models-dir ${models_dir}" >/dev/null 2>&1; then
    return 0
  fi

  echo "[${now}] start wait+bench: ${models_dir}" >> "${log_file}"
  nohup ./scripts/run_hf_wait_and_bench.sh --models-dir "${models_dir}" --arch 120a-real --only all --iters 10 --tokens 32 --gpu-layers 48 > "${log}" 2>&1 </dev/null &
}

echo "[${now}] tick" >> "${log_file}"

# Main HF matrix (large set).
ensure_download "tests/models/hf" "scripts/hf_gguf_download_manifest.sh --out tests/models/hf" "${log_dir}/hf-download.log"
ensure_wait_and_bench "tests/models/hf" "${log_dir}/hf-wait-and-bench.log"

# LFM2.5 set (text + VL + audio).
ensure_download "tests/models/hf-lfm25" "scripts/hf_gguf_download_lfm25_all.sh --out tests/models/hf-lfm25" "${log_dir}/hf-lfm25-download.log"
ensure_wait_and_bench "tests/models/hf-lfm25" "${log_dir}/hf-lfm25-wait-and-bench.log"

