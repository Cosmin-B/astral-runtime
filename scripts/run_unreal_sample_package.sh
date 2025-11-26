#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

out_dir="${ASTRAL_UNREAL_SAMPLE_DIR:-${root_dir}/build/unreal-sample-package/AstralSample}"
archive_dir="${ASTRAL_UNREAL_SAMPLE_ARCHIVE_DIR:-${root_dir}/build/unreal-sample-package/archive}"
platform="${ASTRAL_UNREAL_SAMPLE_PLATFORM:-Linux}"
runuat="${UNREAL_RUNUAT:-}"
unreal_editor="${UNREAL_EDITOR:-}"
engine_dir="${UNREAL_ENGINE_DIR:-}"
build_native=1
run_sample="${ASTRAL_UNREAL_SAMPLE_RUN:-0}"
sample_backend="${ASTRAL_UNREAL_SAMPLE_BACKEND:-}"
sample_memory_backend="${ASTRAL_UNREAL_SAMPLE_MEMORY_BACKEND:-mock}"
sample_media_backend="${ASTRAL_UNREAL_SAMPLE_MEDIA_BACKEND:-mock}"
sample_model="${ASTRAL_UNREAL_SAMPLE_MODEL:-}"
sample_embedding_model="${ASTRAL_UNREAL_SAMPLE_EMBED_MODEL:-}"
sample_media_path="${ASTRAL_UNREAL_SAMPLE_MEDIA_PATH:-}"
sample_media_path_root="${ASTRAL_UNREAL_SAMPLE_MEDIA_PATH_ROOT:-Raw}"
sample_prompt="${ASTRAL_UNREAL_SAMPLE_PROMPT:-Say hello from Astral.}"
sample_runtime_log="${ASTRAL_UNREAL_SAMPLE_RUNTIME_LOG:-}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_unreal_sample_package.sh [options]

Generate the sidecar AstralSample UE 5.7 project outside the repo and package it
with RunUAT BuildCookRun.

Options:
  --out <path>             Generated AstralSample project path
  --archive-dir <path>     BuildCookRun archive output path
  --platform <name>        Unreal target platform (default: Linux)
  --runuat <path>          Path to RunUAT.sh or RunUAT.bat
  --editor <path>          UnrealEditor-Cmd/UnrealEditor path used to infer RunUAT
  --engine-dir <path>      Unreal engine root used to infer RunUAT
  --skip-native-build      Do not rebuild the AstralRT ThirdParty package first
  --run-sample             Launch the archived sample after packaging
  --runtime-log <path>     Capture packaged sample output to this log
  --sample-backend <name>  Backend for generation/embedding demos
  --sample-memory-backend <name>
                          Backend for packaged Content/Saved byte demos
  --sample-media-backend <name>
                          Backend for sample image/audio feed demo
  --sample-model <path>    GGUF path passed as -AstralModel
  --sample-embedding-model <path>
                          GGUF path passed as -AstralEmbeddingModel
  --sample-media-path <path>
                          Projector/encoder GGUF path passed as -AstralMediaPath
  --sample-media-path-root <Raw|ProjectContent|ProjectSaved|ProjectPersistentDownload>
                          Path root passed as -AstralMediaPathRoot (default: Raw)
  --sample-prompt <text>   Prompt passed as -AstralPrompt
  -h, --help               Show this help

Environment:
  UNREAL_RUNUAT            Same as --runuat
  UNREAL_ENGINE_DIR        Same as --engine-dir
  UNREAL_EDITOR            Same as --editor
  ASTRAL_UNREAL_SAMPLE_RUN
  ASTRAL_UNREAL_SAMPLE_BACKEND
  ASTRAL_UNREAL_SAMPLE_MEMORY_BACKEND
  ASTRAL_UNREAL_SAMPLE_MEDIA_BACKEND
  ASTRAL_UNREAL_SAMPLE_MODEL
  ASTRAL_UNREAL_SAMPLE_EMBED_MODEL
  ASTRAL_UNREAL_SAMPLE_MEDIA_PATH
  ASTRAL_UNREAL_SAMPLE_MEDIA_PATH_ROOT
  ASTRAL_UNREAL_SAMPLE_PROMPT
  ASTRAL_UNREAL_SAMPLE_RUNTIME_LOG

The generated sample project is a sidecar artifact and must not be committed to
the Astral repo.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      out_dir="${2:-}"
      shift 2
      ;;
    --archive-dir)
      archive_dir="${2:-}"
      shift 2
      ;;
    --platform)
      platform="${2:-}"
      shift 2
      ;;
    --runuat)
      runuat="${2:-}"
      shift 2
      ;;
    --editor)
      unreal_editor="${2:-}"
      shift 2
      ;;
    --engine-dir)
      engine_dir="${2:-}"
      shift 2
      ;;
    --skip-native-build)
      build_native=0
      shift
      ;;
    --run-sample)
      run_sample=1
      shift
      ;;
    --runtime-log)
      sample_runtime_log="${2:-}"
      shift 2
      ;;
    --sample-backend)
      sample_backend="${2:-}"
      shift 2
      ;;
    --sample-memory-backend)
      sample_memory_backend="${2:-}"
      shift 2
      ;;
    --sample-media-backend)
      sample_media_backend="${2:-}"
      shift 2
      ;;
    --sample-model)
      sample_model="${2:-}"
      shift 2
      ;;
    --sample-embedding-model)
      sample_embedding_model="${2:-}"
      shift 2
      ;;
    --sample-media-path)
      sample_media_path="${2:-}"
      shift 2
      ;;
    --sample-media-path-root)
      sample_media_path_root="${2:-}"
      shift 2
      ;;
    --sample-prompt)
      sample_prompt="${2:-}"
      shift 2
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

infer_runuat_from_engine_dir() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    return 1
  fi
  for script in \
    "${candidate}/Engine/Build/BatchFiles/RunUAT.sh" \
    "${candidate}/Engine/Build/BatchFiles/RunUAT.bat"; do
    if [[ -f "${script}" ]]; then
      printf '%s\n' "${script}"
      return 0
    fi
  done
  return 1
}

infer_engine_dir_from_editor() {
  local editor="$1"
  if [[ -z "${editor}" || "${editor}" != */Engine/Binaries/* ]]; then
    return 1
  fi
  printf '%s\n' "${editor%%/Engine/Binaries/*}"
}

if [[ -z "${runuat}" && -n "${engine_dir}" ]]; then
  runuat="$(infer_runuat_from_engine_dir "${engine_dir}" || true)"
fi

if [[ -z "${runuat}" && -n "${unreal_editor}" ]]; then
  inferred_engine_dir="$(infer_engine_dir_from_editor "${unreal_editor}" || true)"
  runuat="$(infer_runuat_from_engine_dir "${inferred_engine_dir}" || true)"
fi

if [[ -z "${runuat}" ]]; then
  cat >&2 <<'EOF'
Missing Unreal RunUAT path.
Set UNREAL_RUNUAT, UNREAL_ENGINE_DIR, or UNREAL_EDITOR so the sample package lane can run.
EOF
  exit 2
fi

case "${sample_media_path_root}" in
  Raw|ProjectContent|ProjectSaved|ProjectPersistentDownload)
    ;;
  *)
    echo "--sample-media-path-root must be Raw, ProjectContent, ProjectSaved, or ProjectPersistentDownload" >&2
    exit 2
    ;;
esac

if [[ ! -f "${runuat}" ]]; then
  echo "RunUAT path does not exist: ${runuat}" >&2
  exit 2
fi

if [[ "${runuat}" != *.bat && ! -x "${runuat}" ]]; then
  echo "RunUAT is not executable: ${runuat}" >&2
  exit 2
fi

if [[ -z "${sample_backend}" ]]; then
  if [[ -n "${sample_model}" || -n "${sample_embedding_model}" ]]; then
    sample_backend="cpu"
  else
    sample_backend="mock"
  fi
fi

if [[ "${build_native}" -eq 1 ]]; then
  cmake --preset unreal-plugin
  cmake --build --preset unreal-plugin -j
fi

"${root_dir}/scripts/create_unreal_sample_project.sh" \
  --out "${out_dir}" \
  --plugin-mode copy

project_file="${out_dir}/AstralSample.uproject"
if [[ ! -f "${project_file}" ]]; then
  echo "Generated sample project is missing: ${project_file}" >&2
  exit 2
fi

mkdir -p "${archive_dir}"

echo "[unreal_sample] Project: ${project_file}"
echo "[unreal_sample] Archive: ${archive_dir}"
echo "[unreal_sample] RunUAT: ${runuat}"
echo "[unreal_sample] Platform: ${platform}"
echo "[unreal_sample] Plugin mode: copy"
echo "[unreal_sample] BuildCookRun"

"${runuat}" BuildCookRun \
  -project="${project_file}" \
  -noP4 \
  -clientconfig=Development \
  -platform="${platform}" \
  -build \
  -cook \
  -stage \
  -pak \
  -archive \
  -archivedirectory="${archive_dir}" \
  -utf8output

echo "[unreal_sample] OK: ${archive_dir}"

if [[ "${run_sample}" != "1" ]]; then
  exit 0
fi

if [[ "${platform}" != "Linux" ]]; then
  echo "Packaged sample launch currently supports Linux archives only, got: ${platform}" >&2
  exit 2
fi

sample_exe="${archive_dir}/${platform}/AstralSample.sh"
if [[ ! -x "${sample_exe}" ]]; then
  echo "Packaged sample executable is missing or not executable: ${sample_exe}" >&2
  exit 2
fi

runtime_args=(
  -NullRHI
  -Unattended
  -NoSplash
  -NoSound
  -AstralSampleAutoQuit
  -log
  -stdout
  "-AstralBackend=${sample_backend}"
  "-AstralMemoryBackend=${sample_memory_backend}"
  "-AstralMediaBackend=${sample_media_backend}"
)
if [[ -n "${sample_model}" ]]; then
  runtime_args+=("-AstralModel=${sample_model}")
fi
if [[ -n "${sample_embedding_model}" ]]; then
  runtime_args+=("-AstralEmbeddingModel=${sample_embedding_model}")
fi
if [[ -n "${sample_media_path}" ]]; then
  runtime_args+=("-AstralMediaPath=${sample_media_path}")
  runtime_args+=("-AstralMediaPathRoot=${sample_media_path_root}")
fi
if [[ -n "${sample_prompt}" ]]; then
  runtime_args+=("-AstralPrompt=${sample_prompt}")
fi

echo "[unreal_sample] Runtime: ${sample_exe}"
echo "[unreal_sample] Runtime backend: ${sample_backend}"
echo "[unreal_sample] Runtime memory backend: ${sample_memory_backend}"
echo "[unreal_sample] Runtime media backend: ${sample_media_backend}"
if [[ -n "${sample_media_path}" ]]; then
  echo "[unreal_sample] Runtime media path: ${sample_media_path}"
  echo "[unreal_sample] Runtime media path root: ${sample_media_path_root}"
fi
if [[ -n "${sample_runtime_log}" ]]; then
  mkdir -p "$(dirname "${sample_runtime_log}")"
  echo "[unreal_sample] Runtime log: ${sample_runtime_log}"
  set +e
  "${sample_exe}" "${runtime_args[@]}" 2>&1 | tee "${sample_runtime_log}"
  runtime_status=${PIPESTATUS[0]}
  set -e
else
  set +e
  "${sample_exe}" "${runtime_args[@]}"
  runtime_status=$?
  set -e
fi

if [[ "${runtime_status}" -ne 0 ]]; then
  echo "[unreal_sample] Runtime failed: ${runtime_status}" >&2
  exit "${runtime_status}"
fi

echo "[unreal_sample] Runtime OK"
