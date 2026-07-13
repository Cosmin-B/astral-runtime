#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
template_dir="${root_dir}/examples/unreal/AstralSample"
plugin_dir="${root_dir}/plugins/unreal/AstralRT"
plugin_mode="symlink"
out_dir=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/create_unreal_sample_project.sh --out PATH [options]

Creates a sidecar Unreal project named AstralSample from the maintained source
template. Generated project files are written to PATH and are not intended to
be committed to the Astral repository.

Options:
  --out PATH                    Output project directory (required)
  --plugin-mode symlink|copy|none
                                How AstralRT is added (default: symlink)
  --plugin-dir PATH             AstralRT source for symlink/copy mode
  --help                        Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --out" >&2
        exit 2
      fi
      out_dir="$2"
      shift 2
      ;;
    --plugin-mode)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --plugin-mode" >&2
        exit 2
      fi
      plugin_mode="$2"
      shift 2
      ;;
    --plugin-dir)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --plugin-dir" >&2
        exit 2
      fi
      plugin_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${out_dir}" ]]; then
  echo "--out is required" >&2
  usage >&2
  exit 2
fi

case "${plugin_mode}" in
  symlink|copy|none)
    ;;
  *)
    echo "--plugin-mode must be symlink, copy, or none" >&2
    exit 2
    ;;
esac

if [[ ! -d "${template_dir}" ]]; then
  echo "Unreal sample template not found: ${template_dir}" >&2
  exit 2
fi
if [[ "${plugin_mode}" != "none" && ! -d "${plugin_dir}" ]]; then
  echo "AstralRT plugin directory not found: ${plugin_dir}" >&2
  exit 2
fi

out_parent="$(dirname "${out_dir}")"
mkdir -p "${out_parent}"
project_dir="$(cd "${out_parent}" && pwd)/$(basename "${out_dir}")"
mkdir -p "${project_dir}"
cp -a "${template_dir}/." "${project_dir}/"

mkdir -p "${project_dir}/Plugins"
rm -rf "${project_dir}/Plugins/AstralRT"
if [[ "${plugin_mode}" == "symlink" ]]; then
  ln -s "${plugin_dir}" "${project_dir}/Plugins/AstralRT"
elif [[ "${plugin_mode}" == "copy" ]]; then
  cp -a "${plugin_dir}" "${project_dir}/Plugins/AstralRT"
fi

echo "Created Unreal sample project: ${project_dir}"
