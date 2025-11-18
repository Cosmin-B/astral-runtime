#!/bin/bash
# model_downloader.sh - Download a GGUF model for integration tests
#
# Default: GPT-2 (Q2_K quantization, small download)
#
# Overrides (optional):
# - ASTRAL_TEST_MODEL_URL / ASTRAL_MODEL_URL: GGUF download URL
# - ASTRAL_TEST_MODEL_FILE / ASTRAL_MODEL_FILE: output filename (under tests/models/)
# - ASTRAL_HF_REPO / ASTRAL_TEST_HF_REPO: Hugging Face repo id (org/name)
# - ASTRAL_HF_FILE / ASTRAL_TEST_HF_FILE: Hugging Face filename (in repo)
# - ASTRAL_HF_REVISION / ASTRAL_TEST_HF_REVISION: Hugging Face revision (default: main)
# - HF_TOKEN: optional token for gated/private repos (used as Authorization header)
# - ASTRAL_MODEL_MIN_BYTES: minimum expected file size (bytes)
#
# CLI overrides (take precedence over env):
#   ./model_downloader.sh --url <url> --file <name.gguf> --min-bytes <N>
#   ./model_downloader.sh --hf-repo <org/repo> --hf-file <name.gguf> [--hf-rev <rev>]
#   ./model_downloader.sh --preset <name>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR="${SCRIPT_DIR}/models"

DEFAULT_MODEL_FILE="gpt2.Q2_K.gguf"
DEFAULT_MODEL_URL="https://huggingface.co/RichardErkhov/openai-community_-_gpt2-gguf/resolve/main/gpt2.Q2_K.gguf"
DEFAULT_MIN_BYTES=70000000

HF_REPO="${ASTRAL_HF_REPO:-${ASTRAL_TEST_HF_REPO:-}}"
HF_FILE="${ASTRAL_HF_FILE:-${ASTRAL_TEST_HF_FILE:-}}"
HF_REVISION="${ASTRAL_HF_REVISION:-${ASTRAL_TEST_HF_REVISION:-main}}"

MODEL_URL="${ASTRAL_MODEL_URL:-${ASTRAL_TEST_MODEL_URL:-}}"
MODEL_FILE_NAME="${ASTRAL_MODEL_FILE:-${ASTRAL_TEST_MODEL_FILE:-$DEFAULT_MODEL_FILE}}"
MODEL_MIN_BYTES="${ASTRAL_MODEL_MIN_BYTES:-$DEFAULT_MIN_BYTES}"

usage() {
    cat <<EOF
Usage: $0 [--preset <name>] [--url <url>] [--hf-repo <org/repo> --hf-file <file.gguf> [--hf-rev <rev>]] \\
          [--file <name.gguf>] [--min-bytes <N>] [--dir <path>] [--list-presets]

Environment overrides:
  ASTRAL_TEST_MODEL_URL / ASTRAL_MODEL_URL
  ASTRAL_TEST_MODEL_FILE / ASTRAL_MODEL_FILE
  ASTRAL_TEST_HF_REPO / ASTRAL_HF_REPO
  ASTRAL_TEST_HF_FILE / ASTRAL_HF_FILE
  ASTRAL_TEST_HF_REVISION / ASTRAL_HF_REVISION
  ASTRAL_MODEL_MIN_BYTES
  HF_TOKEN (optional auth token for private/gated Hugging Face repos)

After download, point integration test at the model:
  export ASTRAL_TEST_MODEL="\${PWD}/tests/models/<name.gguf>"
EOF
}

list_presets() {
    cat <<EOF
Presets:
  gpt2-q2k
    - Inference smoke default (small generative GGUF)
  embed-minilm-q2k
    - Embeddings model (BERT encoder; 384-dim class), small
  embed-bge-small-q4km
    - Embeddings model (BGE small; 384-dim class), small-ish
  embed-nomic-v1-5-q2k
    - Embeddings model (Nomic v1.5; larger than MiniLM/BGE)
  liquid-lfm2-350m-q4km
    - Small LiquidAI generative model (still hundreds of MB)
  qwen3-0.6b-q8
    - Small Qwen3 generative GGUF from the official Qwen repo
  qwen3-embed-0.6b-q8
    - Small Qwen3 embeddings GGUF from the official Qwen repo

You can override any preset with --url/--hf-repo/--hf-file/--min-bytes.
EOF
}

apply_preset() {
    case "$1" in
        gpt2-q2k)
            MODEL_FILE_NAME="gpt2.Q2_K.gguf"
            MODEL_URL="https://huggingface.co/RichardErkhov/openai-community_-_gpt2-gguf/resolve/main/gpt2.Q2_K.gguf"
            MODEL_MIN_BYTES=70000000
            HF_REPO=""
            HF_FILE=""
            HF_REVISION="main"
            ;;
        embed-minilm-q2k)
            MODEL_FILE_NAME="all-MiniLM-L6-v2-Q2_K.gguf"
            HF_REPO="second-state/All-MiniLM-L6-v2-Embedding-GGUF"
            HF_FILE="all-MiniLM-L6-v2-Q2_K.gguf"
            HF_REVISION="main"
            MODEL_URL=""
            MODEL_MIN_BYTES=15000000
            ;;
        embed-bge-small-q4km)
            MODEL_FILE_NAME="bge-small-en-v1.5-q4_k_m.gguf"
            HF_REPO="CompendiumLabs/bge-small-en-v1.5-gguf"
            HF_FILE="bge-small-en-v1.5-q4_k_m.gguf"
            HF_REVISION="main"
            MODEL_URL=""
            MODEL_MIN_BYTES=25000000
            ;;
        embed-nomic-v1-5-q2k)
            MODEL_FILE_NAME="nomic-embed-text-v1.5.Q2_K.gguf"
            HF_REPO="nomic-ai/nomic-embed-text-v1.5-GGUF"
            HF_FILE="nomic-embed-text-v1.5.Q2_K.gguf"
            HF_REVISION="main"
            MODEL_URL=""
            MODEL_MIN_BYTES=80000000
            ;;
        liquid-lfm2-350m-q4km)
            MODEL_FILE_NAME="LFM2-350M-Q4_K_M.gguf"
            HF_REPO="LiquidAI/LFM2-350M-GGUF"
            HF_FILE="LFM2-350M-Q4_K_M.gguf"
            HF_REVISION="main"
            MODEL_URL=""
            MODEL_MIN_BYTES=200000000
            ;;
        qwen3-0.6b-q8)
            MODEL_FILE_NAME="Qwen3-0.6B-Q8_0.gguf"
            HF_REPO="Qwen/Qwen3-0.6B-GGUF"
            HF_FILE="Qwen3-0.6B-Q8_0.gguf"
            HF_REVISION="main"
            MODEL_URL=""
            MODEL_MIN_BYTES=550000000
            ;;
        qwen3-embed-0.6b-q8)
            MODEL_FILE_NAME="Qwen3-Embedding-0.6B-Q8_0.gguf"
            HF_REPO="Qwen/Qwen3-Embedding-0.6B-GGUF"
            HF_FILE="Qwen3-Embedding-0.6B-Q8_0.gguf"
            HF_REVISION="main"
            MODEL_URL=""
            MODEL_MIN_BYTES=550000000
            ;;
        *)
            echo "Unknown preset: $1"
            echo ""
            list_presets
            exit 1
            ;;
    esac
}

build_hf_url() {
    local repo="$1"
    local file="$2"
    local rev="$3"
    if [ -z "$repo" ] || [ -z "$file" ]; then
        return 1
    fi
    if [ -z "$rev" ]; then
        rev="main"
    fi
    echo "https://huggingface.co/${repo}/resolve/${rev}/${file}"
}

PRESET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --list-presets)
            list_presets
            exit 0
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --url)
            MODEL_URL="$2"
            shift 2
            ;;
        --hf-repo)
            HF_REPO="$2"
            shift 2
            ;;
        --hf-file)
            HF_FILE="$2"
            shift 2
            ;;
        --hf-rev|--hf-revision)
            HF_REVISION="$2"
            shift 2
            ;;
        --file)
            MODEL_FILE_NAME="$2"
            shift 2
            ;;
        --min-bytes)
            MODEL_MIN_BYTES="$2"
            shift 2
            ;;
        --dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -n "$PRESET" ]; then
    apply_preset "$PRESET"
fi

if [ -z "${MODEL_URL}" ]; then
    # If URL wasn't provided, prefer HF repo+file if set; otherwise fall back to the default URL.
    if [ -n "${HF_REPO}" ] && [ -n "${HF_FILE}" ]; then
        MODEL_URL="$(build_hf_url "${HF_REPO}" "${HF_FILE}" "${HF_REVISION}")"
    else
        MODEL_URL="${DEFAULT_MODEL_URL}"
    fi
fi

MODEL_FILE="${MODEL_DIR}/${MODEL_FILE_NAME}"

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -ge "$MODEL_MIN_BYTES" ]; then
        info "Model already exists at: $MODEL_FILE"
        info "File size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "${FILE_SIZE} bytes")"
        exit 0
    else
        warn "Model file exists but is too small ($FILE_SIZE bytes), re-downloading..."
        rm -f "$MODEL_FILE"
    fi
fi

# Create model directory
mkdir -p "$MODEL_DIR"

# Check for download tool (wget or curl)
DOWNLOAD_TOOL=""
if command -v wget >/dev/null 2>&1; then
    DOWNLOAD_TOOL="wget"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOAD_TOOL="curl"
else
    error "Neither wget nor curl is available"
    error "Please install wget or curl and try again"
    error "  Ubuntu/Debian: sudo apt-get install wget"
    error "  macOS: brew install wget"
    exit 1
fi

info "Downloading model: ${MODEL_FILE_NAME}"
info "URL: ${MODEL_URL}"
info "This may take a few minutes depending on your connection..."

# Optional Hugging Face token for gated/private models.
AUTH_HEADER=""
if [ -n "${HF_TOKEN:-}" ]; then
    AUTH_HEADER="Authorization: Bearer ${HF_TOKEN}"
fi

# Download with progress
if [ "$DOWNLOAD_TOOL" = "wget" ]; then
    WGET_HEADERS=()
    if [ -n "${AUTH_HEADER}" ]; then
        WGET_HEADERS+=(--header "${AUTH_HEADER}")
    fi
    wget --progress=bar:force:noscroll \
         --timeout=30 \
         --tries=3 \
         "${WGET_HEADERS[@]}" \
         -O "$MODEL_FILE.tmp" \
         "$MODEL_URL" || {
        error "Download failed"
        rm -f "$MODEL_FILE.tmp"
        exit 1
    }
elif [ "$DOWNLOAD_TOOL" = "curl" ]; then
    CURL_HEADERS=()
    if [ -n "${AUTH_HEADER}" ]; then
        CURL_HEADERS+=(-H "${AUTH_HEADER}")
    fi
    curl -L \
         --progress-bar \
         --connect-timeout 30 \
         --retry 3 \
         "${CURL_HEADERS[@]}" \
         -o "$MODEL_FILE.tmp" \
         "$MODEL_URL" || {
        error "Download failed"
        rm -f "$MODEL_FILE.tmp"
        exit 1
    }
fi

# Verify download
if [ ! -f "$MODEL_FILE.tmp" ]; then
    error "Downloaded file not found"
    exit 1
fi

FILE_SIZE=$(stat -c%s "$MODEL_FILE.tmp" 2>/dev/null || stat -f%z "$MODEL_FILE.tmp" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -lt "$MODEL_MIN_BYTES" ]; then
    error "Downloaded file is too small ($FILE_SIZE bytes), expected >=${MODEL_MIN_BYTES}"
    rm -f "$MODEL_FILE.tmp"
    exit 1
fi

# Move to final location
mv "$MODEL_FILE.tmp" "$MODEL_FILE"

info "Model downloaded successfully!"
info "Location: $MODEL_FILE"
info "Size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "${FILE_SIZE} bytes")"

exit 0
