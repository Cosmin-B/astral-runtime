#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="squidfunk/mkdocs-material:9.7.6@sha256:868ad4d39fb5865b72d00173ade00f4eae2b38dde7ff790a011cc44ce4a8ff8e"
output_dir="${root_dir}/build/docs-site"
staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/astral-docs-site.XXXXXX")"

cleanup() {
  rm -rf "${staging_dir}"
}
trap cleanup EXIT

if python3 -c 'import importlib.metadata; raise SystemExit(importlib.metadata.version("mkdocs-material") != "9.7.6")' >/dev/null 2>&1; then
  ASTRAL_DOCS_SITE_DIR="${staging_dir}" NO_MKDOCS_2_WARNING=true \
    python3 -m mkdocs build --strict --config-file "${root_dir}/docs-site/mkdocs.yml"
elif command -v docker >/dev/null 2>&1; then
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "${root_dir}:/docs" \
    --volume "${staging_dir}:/site" \
    --workdir /docs \
    --env ASTRAL_DOCS_SITE_DIR=/site \
    --env NO_MKDOCS_2_WARNING=true \
    "${image}" \
    build --strict --config-file docs-site/mkdocs.yml
else
  printf '%s\n' 'MkDocs Material 9.7.6 or Docker is required to build the documentation site.' >&2
  exit 1
fi

rm -rf "${output_dir}"
mkdir -p "${output_dir}"
cp -a "${staging_dir}/." "${output_dir}/"
