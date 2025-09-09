#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

usage() {
  cat <<'EOF'
Usage: scripts/run_tracy_capture.sh [--preset <dev-prof|release-prof|dev-prof-micro|release-prof-micro>]

Builds a profiling preset and runs a small workload so you can attach the Tracy UI.

Notes:
  - Astral uses Tracy on-demand + localhost-only by default. Start the Tracy UI, then run this script.
  - This script does not start the Tracy UI for you.
EOF
}

preset="dev-prof"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f external/tracy/public/TracyClient.cpp ]]; then
  echo "Tracy submodule missing. Run: ./scripts/setup_tracy_submodule.sh" >&2
  exit 1
fi

echo "[tracy] Configure+build: ${preset}"
cmake --preset "${preset}"
cmake --build --preset "${preset}" -j

build_dir="build/${preset}"
case "${preset}" in
  dev-prof) build_dir="build/dev-prof" ;;
  dev-prof-micro) build_dir="build/dev-prof-micro" ;;
  release-prof) build_dir="build/release-prof" ;;
  release-prof-micro) build_dir="build/release-prof-micro" ;;
esac

bench="${build_dir}/benchmarks/astral_benchmarks"
if [[ ! -x "${bench}" ]]; then
  echo "Benchmark binary not found: ${bench}" >&2
  echo "Try building a preset that enables benchmarks (dev-prof inherits dev which enables benchmarks)." >&2
  exit 1
fi

echo "[tracy] Start Tracy UI, then run this workload to trigger zones."
echo "[tracy] Running: ${bench} --alloc --lifecycle"
"${bench}" --alloc --lifecycle

