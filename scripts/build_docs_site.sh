#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
npm ci --prefix "${root_dir}/docs-site"
npm run build --prefix "${root_dir}/docs-site"
