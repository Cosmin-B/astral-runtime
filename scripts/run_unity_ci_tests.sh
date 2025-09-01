#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/ci/unity/AstralCiUnityProject"

UNITY_EDITOR="${UNITY_EDITOR:-}"
if [ -z "${UNITY_EDITOR}" ]; then
  echo "Missing UNITY_EDITOR env var (path to Unity editor executable)."
  exit 2
fi

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Unity CI project not found: ${PROJECT_DIR}"
  exit 2
fi

RESULTS_DIR="${ROOT_DIR}/build/unity-ci-results"
mkdir -p "${RESULTS_DIR}"

"${UNITY_EDITOR}" \
  -batchmode \
  -nographics \
  -quit \
  -projectPath "${PROJECT_DIR}" \
  -runTests \
  -testPlatform EditMode \
  -testResults "${RESULTS_DIR}/editmode-results.xml" \
  -logFile "${RESULTS_DIR}/unity-editmode.log"

echo "[INFO] Results: ${RESULTS_DIR}/editmode-results.xml"

