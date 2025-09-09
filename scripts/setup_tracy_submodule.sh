#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TRACY_PATH="external/tracy"
TRACY_URL="https://github.com/wolfpld/tracy.git"

if [ -f "${TRACY_PATH}/public/TracyClient.cpp" ]; then
  echo "Tracy already present at ${TRACY_PATH}"
  exit 0
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not inside a git repository" >&2
  exit 2
fi

if git config -f .gitmodules --get-regexp "^submodule\\.${TRACY_PATH//\//\\.}\\.path$" >/dev/null 2>&1; then
  echo "Initializing existing Tracy submodule..."
  git submodule update --init --recursive "${TRACY_PATH}"
else
  echo "Adding Tracy submodule at ${TRACY_PATH}..."
  git submodule add "${TRACY_URL}" "${TRACY_PATH}"
  git submodule update --init --recursive "${TRACY_PATH}"
fi

if [ ! -f "${TRACY_PATH}/public/TracyClient.cpp" ]; then
  echo "error: Tracy submodule setup did not produce ${TRACY_PATH}/public/TracyClient.cpp" >&2
  exit 3
fi

echo "Tracy ready: ${TRACY_PATH}"

