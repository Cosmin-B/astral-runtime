#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_path=""
check_only=0

usage() {
  cat <<'EOF'
Usage: scripts/generate_abi_layout_report.sh [options]

Generate a JSON report of public Astral C ABI layout and constant values.

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
static bool first_constant = true;

#define EMIT_STRUCT(T)                                                        \
    do {                                                                      \
        if (!first_struct) {                                                  \
            std::printf(",\n");                                              \
        }                                                                     \
        std::printf("    {\"name\":\"%s\",\"size\":%zu,\"align\":%zu}",    \
                    #T, sizeof(T), alignof(T));                               \
        first_struct = false;                                                 \
    } while (0)

#define EMIT_CONST(C)                                                         \
    do {                                                                      \
        if (!first_constant) {                                                \
            std::printf(",\n");                                              \
        }                                                                     \
        std::printf("    {\"name\":\"%s\",\"value\":%lld}", #C,             \
                    static_cast<long long>(C));                               \
        first_constant = false;                                               \
    } while (0)

int main() {
    std::printf("{\n");
    std::printf("  \"schema\":\"astral.abi.layout.v1\",\n");
    std::printf("  \"version\":{\"major\":%u,\"minor\":%u,\"patch\":%u},\n",
                ASTRAL_VERSION_MAJOR, ASTRAL_VERSION_MINOR, ASTRAL_VERSION_PATCH);
    std::printf("  \"pointer_size\":%zu,\n", sizeof(void*));
    std::printf("  \"size_t_size\":%zu,\n", sizeof(size_t));
#if defined(_WIN32)
    std::printf("  \"calling_convention\":\"__cdecl\",\n");
#else
    std::printf("  \"calling_convention\":\"c\",\n");
#endif
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
    EMIT_STRUCT(AstralToolDesc);
    EMIT_STRUCT(AstralToolsetDesc);
    EMIT_STRUCT(AstralToolInfo);
    EMIT_STRUCT(AstralToolCallResult);
    EMIT_STRUCT(AstralChunkerDesc);
    EMIT_STRUCT(AstralChunkRange);
    EMIT_STRUCT(AstralMemoryIndexDesc);
    EMIT_STRUCT(AstralMemoryRecord);
    EMIT_STRUCT(AstralMemorySearchDesc);
    EMIT_STRUCT(AstralMemorySearchResult);
    EMIT_STRUCT(AstralStats);

    std::printf("\n  ],\n");
    std::printf("  \"constants\":[\n");

    EMIT_CONST(ASTRAL_IMAGE_FORMAT_RGB8);
    EMIT_CONST(ASTRAL_IMAGE_FORMAT_RGBA8);
    EMIT_CONST(ASTRAL_IMAGE_FORMAT_RGB_F32);
    EMIT_CONST(ASTRAL_AUDIO_FORMAT_F32);
    EMIT_CONST(ASTRAL_AUDIO_FORMAT_I16);
    EMIT_CONST(ASTRAL_MEDIA_FLAG_USE_GPU);
    EMIT_CONST(ASTRAL_MEDIA_FLAG_WARMUP);
    EMIT_CONST(ASTRAL_GPU_ROUTE_NONE);
    EMIT_CONST(ASTRAL_GPU_ROUTE_DEVICE);
    EMIT_CONST(ASTRAL_GPU_ROUTE_DEVICE_MASK);
    EMIT_CONST(ASTRAL_GPU_ROUTE_STREAM);
    EMIT_CONST(ASTRAL_OK);
    EMIT_CONST(ASTRAL_E_INVALID);
    EMIT_CONST(ASTRAL_E_NOMEM);
    EMIT_CONST(ASTRAL_E_BUSY);
    EMIT_CONST(ASTRAL_E_TIMEOUT);
    EMIT_CONST(ASTRAL_E_STATE);
    EMIT_CONST(ASTRAL_E_BACKEND);
    EMIT_CONST(ASTRAL_E_CANCELED);
    EMIT_CONST(ASTRAL_E_UNSUPPORTED);
    EMIT_CONST(ASTRAL_E_NOT_FOUND);
    EMIT_CONST(ASTRAL_LOG_ERROR);
    EMIT_CONST(ASTRAL_LOG_WARN);
    EMIT_CONST(ASTRAL_LOG_INFO);
    EMIT_CONST(ASTRAL_LOG_DEBUG);
    EMIT_CONST(ASTRAL_LOG_TRACE);
    EMIT_CONST(ASTRAL_MEMMODE_VM);
    EMIT_CONST(ASTRAL_MEMMODE_ARENA_BORROWED);
    EMIT_CONST(ASTRAL_MEMMODE_ARENA_OWNED);
    EMIT_CONST(ASTRAL_MODEL_SOURCE_PATH);
    EMIT_CONST(ASTRAL_MODEL_SOURCE_MEMORY);
    EMIT_CONST(ASTRAL_MODEL_SOURCE_IO);
    EMIT_CONST(ASTRAL_GPU_SPLIT_NONE);
    EMIT_CONST(ASTRAL_GPU_SPLIT_LAYER);
    EMIT_CONST(ASTRAL_GPU_SPLIT_ROW);
    EMIT_CONST(ASTRAL_GPU_CFG_NONE);
    EMIT_CONST(ASTRAL_GPU_CFG_MAIN);
    EMIT_CONST(ASTRAL_GPU_CFG_SPLIT_MODE);
    EMIT_CONST(ASTRAL_GPU_CFG_DEVICES);
    EMIT_CONST(ASTRAL_GPU_CFG_DEVICE_MASK);
    EMIT_CONST(ASTRAL_GPU_CFG_TENSOR_SPLIT);
    EMIT_CONST(ASTRAL_CAP_NONE);
    EMIT_CONST(ASTRAL_CAP_SAMPLER_EXT);
    EMIT_CONST(ASTRAL_CAP_STOP_SEQS);
    EMIT_CONST(ASTRAL_CAP_EMBEDDINGS);
    EMIT_CONST(ASTRAL_CAP_GPU_OFFLOAD);
    EMIT_CONST(ASTRAL_CAP_LORA);
    EMIT_CONST(ASTRAL_CAP_GRAMMAR);
    EMIT_CONST(ASTRAL_CAP_LOGPROBS);
    EMIT_CONST(ASTRAL_CAP_KV_STATE);
    EMIT_CONST(ASTRAL_CAP_SLOTS);
    EMIT_CONST(ASTRAL_CAP_GRAMMAR_GBNF);
    EMIT_CONST(ASTRAL_CAP_GRAMMAR_JSON_SCHEMA);
    EMIT_CONST(ASTRAL_CAP_IMAGE);
    EMIT_CONST(ASTRAL_CAP_AUDIO);
    EMIT_CONST(ASTRAL_CAP_MM_EMBEDDINGS);
    EMIT_CONST(ASTRAL_SESSION_IDLE);
    EMIT_CONST(ASTRAL_SESSION_FEEDING_PROMPT);
    EMIT_CONST(ASTRAL_SESSION_DECODING);
    EMIT_CONST(ASTRAL_SESSION_COMPLETED);
    EMIT_CONST(ASTRAL_SESSION_CANCELED);
    EMIT_CONST(ASTRAL_SESSION_FAILED);
    EMIT_CONST(ASTRAL_SESSION_ADAPTERS_MAX);
    EMIT_CONST(ASTRAL_TOOL_CHOICE_AUTO);
    EMIT_CONST(ASTRAL_TOOL_CHOICE_REQUIRED);
    EMIT_CONST(ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL);
    EMIT_CONST(ASTRAL_CHUNK_MODE_NONE);
    EMIT_CONST(ASTRAL_CHUNK_MODE_CHAR);
    EMIT_CONST(ASTRAL_CHUNK_MODE_WORD);
    EMIT_CONST(ASTRAL_CHUNK_MODE_SENTENCE);
    EMIT_CONST(ASTRAL_CHUNK_MODE_TOKEN);
    EMIT_CONST(ASTRAL_CHUNK_FLAG_KEEP_EMPTY);
    EMIT_CONST(ASTRAL_MEMORY_METRIC_DOT);
    EMIT_CONST(ASTRAL_MEMORY_METRIC_COSINE);
    EMIT_CONST(ASTRAL_MEMORY_METRIC_L2);
    EMIT_CONST(ASTRAL_MEMORY_INDEX_FLAT);
    EMIT_CONST(ASTRAL_MEMORY_GROUP_ANY);

    std::printf("\n  ]\n");
    std::printf("}\n");
    return 0;
}
CPP

"${compiler}" -std=c++17 -I"${root_dir}/include" "${src}" -o "${bin}"
"${bin}" > "${report}"

if [[ "${check_only}" -eq 1 ]]; then
  grep -q '"schema":"astral.abi.layout.v1"' "${report}"
  grep -q '"calling_convention":"' "${report}"
  grep -q '"name":"AstralModelDesc"' "${report}"
  grep -q '"name":"AstralSessionDesc"' "${report}"
  grep -q '"name":"AstralStats"' "${report}"
  grep -q '"name":"AstralToolCallResult"' "${report}"
  grep -q '"name":"ASTRAL_E_UNSUPPORTED","value":-8' "${report}"
  grep -q '"name":"ASTRAL_E_NOT_FOUND","value":-9' "${report}"
  grep -q '"name":"ASTRAL_MODEL_SOURCE_IO","value":2' "${report}"
  grep -q '"name":"ASTRAL_SESSION_FAILED","value":5' "${report}"
  grep -q '"name":"ASTRAL_SESSION_ADAPTERS_MAX","value":8' "${report}"
  grep -q '"name":"ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL","value":2' "${report}"
  grep -q '"name":"AstralChunkRange"' "${report}"
  grep -q '"name":"ASTRAL_CHUNK_MODE_TOKEN","value":4' "${report}"
  grep -q '"name":"AstralMemorySearchResult"' "${report}"
  grep -q '"name":"ASTRAL_MEMORY_METRIC_L2","value":2' "${report}"
  grep -q '"name":"ASTRAL_CAP_MM_EMBEDDINGS","value":134217728' "${report}"
  grep -q '"name":"ASTRAL_GPU_ROUTE_STREAM","value":4' "${report}"
  exit 0
fi

if [[ -n "${out_path}" ]]; then
  mkdir -p "$(dirname "${out_path}")"
  cp "${report}" "${out_path}"
else
  cat "${report}"
fi
