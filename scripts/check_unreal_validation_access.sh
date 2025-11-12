#!/usr/bin/env bash
set -euo pipefail

full_image="ghcr.io/epicgames/unreal-engine:dev-5.7.4"
full_digest="sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce"
slim_image="ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4"
slim_digest="sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6"

container_engine="${CONTAINER_ENGINE:-docker}"
timeout_seconds="${ASTRAL_UNREAL_ACCESS_TIMEOUT_SECONDS:-5}"
check_registry=0
allow_missing=0

usage() {
  cat <<'EOF'
Usage: scripts/check_unreal_validation_access.sh [options]

Check whether this machine can run Astral's real Unreal validation lanes.

Options:
  --check-registry          Probe Epic GHCR manifests for the pinned UE 5.7 images
  --allow-missing           Print diagnostics but exit 0 when access is incomplete
  --container-engine <cmd>  Container CLI to inspect images/manifests (default: docker)
  --timeout <seconds>       Timeout for registry probes (default: 5)
  -h, --help                Show this help

Ready means either:
  - Both pinned UE 5.7 full/slim container images are cached locally or their
    manifests are readable with --check-registry, or
  - UNREAL_54_EDITOR, UNREAL_55_EDITOR, UNREAL_56_EDITOR, and UNREAL_57_EDITOR
    all point to executable UnrealEditor-Cmd/UnrealEditor binaries.

This script only inspects access. It does not pull images, build the plugin,
package the sample, or start UnrealEditor.
EOF
}

has_ghcr_auth_config() {
  if [[ -n "${DOCKER_AUTH_CONFIG:-}" && "${DOCKER_AUTH_CONFIG}" == *"ghcr.io"* ]]; then
    return 0
  fi

  local docker_config_dir="${DOCKER_CONFIG:-${HOME:-}/.docker}"
  local docker_config="${docker_config_dir}/config.json"
  if [[ ! -f "${docker_config}" ]]; then
    return 1
  fi

  grep -Eq '"ghcr[.]io"|"credsStore"|"credHelpers"' "${docker_config}"
}

run_with_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_seconds}" "$@"
  else
    "$@"
  fi
}

image_has_digest() {
  local ref="$1"
  local digest="$2"
  local digests

  if ! command -v "${container_engine}" >/dev/null 2>&1; then
    return 1
  fi

  digests="$("${container_engine}" image inspect "${ref}" --format '{{range .RepoDigests}}{{println .}}{{end}}' 2>/dev/null || true)"
  [[ "${digests}" == *"${digest}"* ]]
}

manifest_is_readable() {
  local ref="$1"

  if ! command -v "${container_engine}" >/dev/null 2>&1; then
    return 1
  fi

  run_with_timeout "${container_engine}" manifest inspect "${ref}" >/dev/null 2>&1
}

check_container_variant() {
  local name="$1"
  local image="$2"
  local digest="$3"
  local ref="${image}@${digest}"

  echo "[unreal_access] UE 5.7 ${name} image: ${ref}"

  if image_has_digest "${ref}" "${digest}"; then
    echo "[unreal_access] OK: ${name} image cached with expected digest"
    return 0
  fi

  echo "[unreal_access] MISSING: ${name} image is not cached with expected digest"

  if [[ "${check_registry}" -eq 0 ]]; then
    echo "[unreal_access] SKIP: ${name} registry manifest not checked"
    return 1
  fi

  if ! has_ghcr_auth_config; then
    echo "[unreal_access] MISSING: Epic GHCR auth is not configured for ${container_engine}"
    return 1
  fi

  if manifest_is_readable "${ref}"; then
    echo "[unreal_access] OK: ${name} registry manifest is readable"
    return 0
  fi

  echo "[unreal_access] MISSING: ${name} registry manifest is not readable"
  return 1
}

editor_var_for_version() {
  case "$1" in
    5.4) printf 'UNREAL_54_EDITOR\n' ;;
    5.5) printf 'UNREAL_55_EDITOR\n' ;;
    5.6) printf 'UNREAL_56_EDITOR\n' ;;
    5.7) printf 'UNREAL_57_EDITOR\n' ;;
    *)
      echo "Unsupported Unreal version '$1'" >&2
      exit 2
      ;;
  esac
}

check_editor_matrix() {
  local ready=1
  local version var editor

  for version in 5.4 5.5 5.6 5.7; do
    var="$(editor_var_for_version "${version}")"
    editor="${!var:-}"
    if [[ -n "${editor}" && -x "${editor}" ]]; then
      echo "[unreal_access] OK: UE ${version} editor ${var}=${editor}"
      continue
    fi

    if [[ -z "${editor}" ]]; then
      echo "[unreal_access] MISSING: ${var} is unset"
    else
      echo "[unreal_access] MISSING: ${var} is not executable: ${editor}"
    fi
    ready=0
  done

  [[ "${ready}" -eq 1 ]]
}

resolve_runuat() {
  local editor engine_dir candidate

  if [[ -n "${UNREAL_RUNUAT:-}" && -x "${UNREAL_RUNUAT}" ]]; then
    printf '%s\n' "${UNREAL_RUNUAT}"
    return 0
  fi

  if [[ -n "${UNREAL_ENGINE_DIR:-}" ]]; then
    candidate="${UNREAL_ENGINE_DIR%/}/Engine/Build/BatchFiles/RunUAT.sh"
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    candidate="${UNREAL_ENGINE_DIR%/}/Build/BatchFiles/RunUAT.sh"
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  for editor in "${UNREAL_57_EDITOR:-}" "${UNREAL_EDITOR:-}"; do
    if [[ "${editor}" == *"/Engine/Binaries/"* ]]; then
      engine_dir="${editor%%/Engine/Binaries/*}/Engine"
      candidate="${engine_dir}/Build/BatchFiles/RunUAT.sh"
      if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  done

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-registry)
      check_registry=1
      shift
      ;;
    --allow-missing)
      allow_missing=1
      shift
      ;;
    --container-engine)
      container_engine="${2:-}"
      shift 2
      ;;
    --timeout)
      timeout_seconds="${2:-}"
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

echo "[unreal_access] Container engine: ${container_engine}"
if command -v "${container_engine}" >/dev/null 2>&1; then
  echo "[unreal_access] OK: container engine found"
else
  echo "[unreal_access] MISSING: container engine not found"
fi

container_ready=0
if check_container_variant "full" "${full_image}" "${full_digest}"; then
  full_ready=1
else
  full_ready=0
fi
if check_container_variant "slim" "${slim_image}" "${slim_digest}"; then
  slim_ready=1
else
  slim_ready=0
fi
if [[ "${full_ready}" -eq 1 && "${slim_ready}" -eq 1 ]]; then
  container_ready=1
  echo "[unreal_access] READY: UE 5.7 full/slim container access is available"
fi

if check_editor_matrix; then
  editor_ready=1
  echo "[unreal_access] READY: UE 5.4-5.7 editor matrix is configured"
else
  editor_ready=0
fi

if runuat="$(resolve_runuat)"; then
  echo "[unreal_access] OK: RunUAT available: ${runuat}"
else
  echo "[unreal_access] MISSING: RunUAT path is not configured"
fi

if [[ "${container_ready}" -eq 1 || "${editor_ready}" -eq 1 ]]; then
  exit 0
fi

cat >&2 <<'EOF'
[unreal_access] BLOCKED: real Unreal validation needs Epic GHCR access for both pinned UE 5.7 images or executable UNREAL_54_EDITOR/UNREAL_55_EDITOR/UNREAL_56_EDITOR/UNREAL_57_EDITOR paths.
EOF

if [[ "${allow_missing}" -eq 1 ]]; then
  exit 0
fi
exit 2
