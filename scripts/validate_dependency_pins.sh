#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pins_path="${root_dir}/docs/release/dependency-pins.tsv"

usage() {
  cat <<'EOF'
Usage: scripts/validate_dependency_pins.sh [options]

Checks docs/release/dependency-pins.tsv against the current working tree.

Options:
  --pins <file>  Pin manifest to validate
  --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pins)
      pins_path="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${pins_path}" || ! -s "${pins_path}" ]]; then
  echo "[dependency-pins] missing or empty pin manifest: ${pins_path}" >&2
  exit 1
fi

json_string_value() {
  local file="$1"
  local key="$2"
  local escaped
  escaped="$(printf '%s' "${key}" | sed 's/[][\\.^$*]/\\&/g')"
  sed -n 's/^[[:space:]]*"'"${escaped}"'": *"\([^"]*\)".*/\1/p' "${file}" | head -n 1
}

project_version() {
  sed -n 's/^project(AstralRT VERSION \([0-9][0-9.]*\).*/\1/p' "${root_dir}/CMakeLists.txt" | head -n 1
}

actual_value() {
  local type="$1"
  local path="$2"
  local key="$3"

  case "${type}" in
    project)
      case "${key}" in
        version) project_version ;;
        *) return 2 ;;
      esac
      ;;
    submodule)
      case "${key}" in
        commit) git -C "${root_dir}/${path}" rev-parse HEAD ;;
        description) git -C "${root_dir}/${path}" describe --tags --always --dirty ;;
        *) return 2 ;;
      esac
      ;;
    unity_package|unity_dependency|unreal_plugin)
      json_string_value "${root_dir}/${path}" "${key}"
      ;;
    unreal_support)
      case "${key}" in
        compatibility_floor) printf '5.4\n' ;;
        production_target) printf '5.7\n' ;;
        *) return 2 ;;
      esac
      ;;
    github_workflow)
      case "${key}" in
        android_ndk_version)
          sed -n 's/^[[:space:]]*ndk-version:[[:space:]]*//p' "${root_dir}/${path}" | sort -u | paste -sd, -
          ;;
        *) return 2 ;;
      esac
      ;;
    *)
      return 2
      ;;
  esac
}

checked=0
while IFS=$'\t' read -r type path key expected rest; do
  if [[ -z "${type}" || "${type:0:1}" == "#" ]]; then
    continue
  fi
  if [[ -n "${rest:-}" || -z "${path}" || -z "${key}" || -z "${expected}" ]]; then
    echo "[dependency-pins] invalid manifest row: ${type} ${path} ${key} ${expected} ${rest:-}" >&2
    exit 1
  fi

  actual="$(actual_value "${type}" "${path}" "${key}")"
  if [[ -z "${actual}" ]]; then
    echo "[dependency-pins] could not read ${type} ${path} ${key}" >&2
    exit 1
  fi
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[dependency-pins] ${type} ${path} ${key}: expected ${expected}, got ${actual}" >&2
    exit 1
  fi
  checked=$((checked + 1))
done < "${pins_path}"

if [[ "${checked}" -eq 0 ]]; then
  echo "[dependency-pins] no pins checked in ${pins_path}" >&2
  exit 1
fi

echo "[dependency-pins] OK (${checked} pins)"
