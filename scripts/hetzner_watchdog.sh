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

usage() {
  cat <<'EOF'
Usage: scripts/hetzner_watchdog.sh [options]

Keeps long-running HF download and bench jobs alive on a remote runner.

Options:
  --dry-run                 Print commands that would be started
  --log-dir <dir>           Log directory (default: benchmarks/results)
  --hf-models-dir <dir>     Main HF matrix directory (default: tests/models/hf)
  --lfm25-models-dir <dir>  LFM2.5 matrix directory (default: tests/models/hf-lfm25)
  --help                    Show help
EOF
}

dry_run=0
log_dir="benchmarks/results"
hf_models_dir="tests/models/hf"
lfm25_models_dir="tests/models/hf-lfm25"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --log-dir) log_dir="${2:-}"; shift 2 ;;
    --hf-models-dir) hf_models_dir="${2:-}"; shift 2 ;;
    --lfm25-models-dir) lfm25_models_dir="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "${log_dir}"
log_file="${log_dir}/watchdog.log"

now="$(date -Iseconds)"

repo_download_running() {
  pgrep -af "python3 scripts/hf_gguf_sync.py download" >/dev/null 2>&1
}

ensure_download() {
  local models_dir="$1"   # tests/models/hf or tests/models/hf-lfm25
  local log="$2"          # log path
  shift 2
  local cmd=("$@")

  local parts
  parts="$(find "${models_dir}" -type f -name "*.part" 2>/dev/null | wc -l)"
  if [[ "${parts}" -eq 0 ]]; then
    return 0
  fi

  if [[ "${dry_run}" -eq 0 ]] && repo_download_running; then
    return 0
  fi

  echo "[${now}] start download: ${cmd[*]} -> ${models_dir}" >> "${log_file}"
  if [[ "${dry_run}" -eq 1 ]]; then
    printf '[watchdog] would start download:'
    printf ' %q' "${cmd[@]}"
    printf ' > %q\n' "${log}"
    return 0
  fi
  nohup "${cmd[@]}" > "${log}" 2>&1 </dev/null &
}

ensure_wait_and_bench() {
  local models_dir="$1"
  local log="$2"

  if [[ "${dry_run}" -eq 0 ]] && pgrep -af "scripts/run_hf_wait_and_bench.sh --models-dir ${models_dir}" >/dev/null 2>&1; then
    return 0
  fi

  echo "[${now}] start wait+bench: ${models_dir}" >> "${log_file}"
  local cmd=(./scripts/run_hf_wait_and_bench.sh --models-dir "${models_dir}" --arch 120a-real --only all --iters 10 --tokens 32 --gpu-layers 48)
  if [[ "${dry_run}" -eq 1 ]]; then
    printf '[watchdog] would start wait+bench:'
    printf ' %q' "${cmd[@]}"
    printf ' > %q\n' "${log}"
    return 0
  fi
  nohup "${cmd[@]}" > "${log}" 2>&1 </dev/null &
}

echo "[${now}] tick" >> "${log_file}"

# Main HF matrix (large set).
ensure_download "${hf_models_dir}" "${log_dir}/hf-download.log" ./scripts/hf_gguf_download_manifest.sh --out "${hf_models_dir}"
ensure_wait_and_bench "${hf_models_dir}" "${log_dir}/hf-wait-and-bench.log"

# LFM2.5 set (text + VL + audio).
ensure_download "${lfm25_models_dir}" "${log_dir}/hf-lfm25-download.log" ./scripts/hf_gguf_download_lfm25_all.sh --out "${lfm25_models_dir}"
ensure_wait_and_bench "${lfm25_models_dir}" "${log_dir}/hf-lfm25-wait-and-bench.log"
