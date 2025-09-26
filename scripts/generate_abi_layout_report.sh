#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_path=""
check_only=0

usage() {
  cat <<'EOF'
Usage: scripts/generate_abi_layout_report.sh [options]

Generate a JSON report of public Astral C ABI struct sizes and alignments.

Options:
  --out <file>   Write report to a file instead of stdout
  --check        Generate and sanity-check the report, but do not print it
  -h, --help     Show this help

Environment:
  CXX            C++ compiler to use (default: c++)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      out_path="${2:-}"
      shift 2
      ;;
    --check)
      check_only=1
      shift
      ;;
    -h|--help)
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

compiler="${CXX:-c++}"
if ! command -v "${compiler}" >/dev/null 2>&1; then
  echo "C++ compiler not found: ${compiler}" >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

src="${tmp_dir}/abi_layout_report.cpp"
bin="${tmp_dir}/abi_layout_report"
report="${tmp_dir}/abi_layout_report.json"

cat > "${src}" <<'CPP'
#include <cstddef>
#include <cstdio>
#include "astral_rt.h"

static bool first_struct = true;

#define EMIT_STRUCT(T)                                                        \
    do {                                                                      \
        if (!first_struct) {                                                  \
            std::printf(",\n");                                              \
        }                                                                     \
        std::printf("    {\"name\":\"%s\",\"size\":%zu,\"align\":%zu}",    \
                    #T, sizeof(T), alignof(T));                               \
        first_struct = false;                                                 \
    } while (0)

int main() {
    std::printf("{\n");
    std::printf("  \"schema\":\"astral.abi.layout.v1\",\n");
    std::printf("  \"version\":{\"major\":%u,\"minor\":%u,\"patch\":%u},\n",
                ASTRAL_VERSION_MAJOR, ASTRAL_VERSION_MINOR, ASTRAL_VERSION_PATCH);
    std::printf("  \"pointer_size\":%zu,\n", sizeof(void*));
    std::printf("  \"size_t_size\":%zu,\n", sizeof(size_t));
    std::printf("  \"structs\":[\n");

    EMIT_STRUCT(AstralSpanU8);
    EMIT_STRUCT(AstralMutSpanU8);
    EMIT_STRUCT(AstralImageDesc);
    EMIT_STRUCT(AstralAudioDesc);
    EMIT_STRUCT(AstralAllocator);
    EMIT_STRUCT(AstralInit);
    EMIT_STRUCT(AstralArenaDesc);
    EMIT_STRUCT(AstralInit2);
    EMIT_STRUCT(AstralModelIO);
    EMIT_STRUCT(AstralModelDesc);
    EMIT_STRUCT(AstralModelInfo);
    EMIT_STRUCT(AstralModelLimits);
    EMIT_STRUCT(AstralModelMediaDesc);
    EMIT_STRUCT(AstralMediaInfo);
    EMIT_STRUCT(AstralSessionDesc);
    EMIT_STRUCT(AstralExecutorDesc);
    EMIT_STRUCT(AstralExecutorTuning);
    EMIT_STRUCT(AstralConvDesc);
    EMIT_STRUCT(AstralConvStats);
    EMIT_STRUCT(AstralSamplerDesc);
    EMIT_STRUCT(AstralTokenMeta);
    EMIT_STRUCT(AstralAdapterDesc);
    EMIT_STRUCT(AstralStats);

    std::printf("\n  ]\n");
    std::printf("}\n");
    return 0;
}
CPP

"${compiler}" -std=c++20 -I"${root_dir}/include" "${src}" -o "${bin}"
"${bin}" > "${report}"

if [[ "${check_only}" -eq 1 ]]; then
  grep -q '"schema":"astral.abi.layout.v1"' "${report}"
  grep -q '"name":"AstralModelDesc"' "${report}"
  grep -q '"name":"AstralSessionDesc"' "${report}"
  grep -q '"name":"AstralStats"' "${report}"
  exit 0
fi

if [[ -n "${out_path}" ]]; then
  mkdir -p "$(dirname "${out_path}")"
  cp "${report}" "${out_path}"
else
  cat "${report}"
fi
