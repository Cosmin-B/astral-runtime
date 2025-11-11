#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path


REQUIRED_LANES = (
    "native_dev_ctest",
    "native_release_ctest",
    "release_required_gates",
    "sanitizer_validation",
    "comment_review",
    "unreal_57_full_container",
    "unreal_57_slim_container",
    "unreal_compatibility_matrix",
    "unity_editmode_abi",
    "cuda_parity_matrix",
    "multimodal_validation",
    "hf_model_matrix",
    "windows_large_pages",
    "release_artifacts",
    "release_signing",
    "dependency_pins",
    "release_notes",
)

PRE_SIGN_EXCLUDED_LANES = ("release_signing",)

REQUIRED_COMMAND_TOKENS = {
    "native_dev_ctest": (
        "cmake --preset dev",
        "cmake --build --preset dev",
        "ctest --preset dev",
    ),
    "native_release_ctest": (
        "cmake --preset release-with-tests",
        "cmake --build --preset release-with-tests",
        "ctest --preset release-with-tests",
    ),
    "release_required_gates": (
        "run_release_required_gates.sh",
        "--cuda-arch",
        "--cuda-strict",
        "--mtmd-bench",
    ),
    "sanitizer_validation": (
        "run_asan.sh",
        "run_tsan.sh",
    ),
    "comment_review": (
        "inventory_comments.py",
        "--format review-tsv",
        "--fail-orphan-markers",
    ),
    "unreal_57_full_container": (
        "run_unreal_container_ci.sh",
        "--variant full",
        "--filter AstralRT.*",
        "ghcr.io/epicgames/unreal-engine:dev-5.7.4",
        "sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce",
    ),
    "unreal_57_slim_container": (
        "run_unreal_container_ci.sh",
        "--variant slim",
        "--filter AstralRT.*",
        "ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4",
        "sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6",
    ),
    "unreal_compatibility_matrix": (
        "UNREAL_54_EDITOR",
        "UNREAL_55_EDITOR",
        "UNREAL_56_EDITOR",
        "UNREAL_57_EDITOR",
        "run_unreal_compatibility_matrix.sh",
        "5.4",
        "5.5",
        "5.6",
        "5.7",
        "--filter AstralRT.*",
    ),
    "unity_editmode_abi": ("UNITY_EDITOR", "run_unity_ci_tests.sh"),
    "cuda_parity_matrix": (
        "ASTRAL_TEST_CUDA_PARITY_INFER=1",
        "ASTRAL_TEST_CUDA_E2E=1",
        "run_cuda_parity_matrix.sh",
        "--preset-set release",
        "--arch",
        "--strict",
    ),
    "multimodal_validation": ("run_multimodal_validation.sh", "--bench"),
    "hf_model_matrix": ("run_hf_full_suite.sh", "--arch", "--only all"),
    "windows_large_pages": ("run_windows_large_page_validation.ps1",),
    "release_artifacts": ("validate_release_artifacts.sh", "--require-signature"),
    "release_signing": ("release-sign",),
    "dependency_pins": ("validate_dependency_pins.sh",),
    "release_notes": ("validate_release_notes.sh",),
}

FORBIDDEN_COMMAND_TOKENS = {
    "unreal_compatibility_matrix": ("--allow-missing",),
}

COMMENT_REVIEW_HEADER = "decision\tissue\tnotes\tpath\tline\tkind\tmarker\tbead\ttext"

REQUIRED_ARTIFACT_NAMES = {
    "sanitizer_validation": ("asan.log", "tsan.log"),
    "unreal_57_full_container": ("unreal-57-full-container.log",),
    "unreal_57_slim_container": ("unreal-57-slim-container.log",),
    "unreal_compatibility_matrix": ("unreal-compatibility-matrix.log",),
    "release_artifacts": (
        "checksums.sha256",
        "abi-layout.json",
        "dependency-manifest.json",
        "release-sbom.spdx.json",
    ),
    "release_signing": ("checksums.sha256.asc",),
    "hf_model_matrix": ("hf-model-matrix.log", "hf-model-matrix-summary.csv"),
    "multimodal_validation": ("multimodal-validation.log", "mtmd-features.txt"),
}

UNREAL_CONTAINER_EVIDENCE = {
    "unreal_57_full_container": {
        "artifact": "unreal-57-full-container.log",
        "image": "ghcr.io/epicgames/unreal-engine:dev-5.7.4",
        "digest": "sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce",
    },
    "unreal_57_slim_container": {
        "artifact": "unreal-57-slim-container.log",
        "image": "ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4",
        "digest": "sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6",
    },
}

UNREAL_CONTAINER_COMMON_TOKENS = (
    "[unreal_container] Check image access:",
    "[unreal_container] Pull image:",
    "[unreal_container] Local image digests:",
    "[unreal_container] Image:",
    "[unreal_container] Test filter: AstralRT.*",
    "[unreal_container] clang:",
    "[unreal_container] Linux SDK:",
    "v26",
    "20.1.8",
    "Unreal ThirdParty provenance OK",
    "[unreal_ci] Filter: AstralRT.*",
    "[unreal-results] OK:",
)

UNREAL_CONTAINER_FAILURE_TOKENS = (
    "Epic Unreal container access is not configured",
    "Unable to inspect Unreal container manifest",
    "Unable to pull Unreal container image",
    "Pulled image does not report expected digest",
    "Linux SDK metadata mismatch",
    "clang version mismatch",
    "Could not locate UnrealEditor-Cmd inside the container",
)

UNREAL_COMPATIBILITY_VERSIONS = ("5.4", "5.5", "5.6", "5.7")

UNREAL_COMPATIBILITY_FAILURE_TOKENS = (
    "[unreal_matrix] Skipping UE",
    "Missing UNREAL_",
    "No Unreal versions ran",
    "Unsupported Unreal version",
    "Automation output contains failure marker",
    "missing or empty Unreal log",
    "Automation report directory has no non-empty files",
)


def fail(message):
    print(f"[release-evidence] {message}", file=sys.stderr)
    return 1


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        chunk = handle.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(1024 * 1024)
    return digest.hexdigest()


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def resolve_artifacts(lane_name, lane, base_dir):
    artifacts = lane.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{lane_name}.artifacts must list at least one artifact")

    resolved = []
    for index, artifact in enumerate(artifacts):
        field = f"{lane_name}.artifacts[{index}]"
        if isinstance(artifact, str):
            artifact = {"path": artifact}
        if not isinstance(artifact, dict):
            raise ValueError(f"{field} must be a string path or object")

        require_text(artifact.get("path"), f"{field}.path")
        relative_path = Path(artifact["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{field}.path must stay under the evidence directory")

        path = base_dir / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{field}.path is missing or empty: {path}")

        expected_sha = artifact.get("sha256")
        if expected_sha is not None:
            require_text(expected_sha, f"{field}.sha256")
            if sha256_file(path) != expected_sha.lower():
                raise ValueError(f"{field}.sha256 does not match: {path}")
        resolved.append(path)
    return resolved


def validate_comment_review_artifacts(paths):
    review_tsv = next((path for path in paths if path.name == "comment-review.tsv"), None)
    summary_log = next((path for path in paths if path.name == "comment-inventory-summary.log"), None)
    if review_tsv is None:
        raise ValueError("comment_review.artifacts must include comment-review.tsv")
    if summary_log is None:
        raise ValueError("comment_review.artifacts must include comment-inventory-summary.log")

    first_line = review_tsv.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if first_line != COMMENT_REVIEW_HEADER:
        raise ValueError("comment_review comment-review.tsv header is invalid")

    summary = summary_log.read_text(encoding="utf-8", errors="replace")
    if "orphan_markers=0" not in summary:
        raise ValueError("comment_review summary must report orphan_markers=0")


def validate_multimodal_artifacts(paths):
    features = next((path for path in paths if path.name == "mtmd-features.txt"), None)
    if features is None:
        raise ValueError("multimodal_validation.artifacts must include mtmd-features.txt")

    text = features.read_text(encoding="utf-8", errors="replace")
    if "[bench] media init failed" in text:
        raise ValueError("multimodal_validation bench output reports media init failure")
    if "features.media feed_image" not in text:
        raise ValueError("multimodal_validation bench output is missing features.media feed_image")
    if "features.media feed_audio" not in text:
        raise ValueError("multimodal_validation bench output is missing features.media feed_audio")


def validate_unreal_container_artifacts(lane_name, paths):
    expected = UNREAL_CONTAINER_EVIDENCE[lane_name]
    log_path = next((path for path in paths if path.name == expected["artifact"]), None)
    if log_path is None:
        raise ValueError(f"{lane_name}.artifacts must include {expected['artifact']}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    for token in (expected["image"], expected["digest"], *UNREAL_CONTAINER_COMMON_TOKENS):
        if token not in text:
            raise ValueError(f"{lane_name} log is missing {token}")
    for token in UNREAL_CONTAINER_FAILURE_TOKENS:
        if token in text:
            raise ValueError(f"{lane_name} log contains failure marker {token}")


def validate_unreal_compatibility_artifacts(paths):
    log_path = next((path for path in paths if path.name == "unreal-compatibility-matrix.log"), None)
    if log_path is None:
        raise ValueError("unreal_compatibility_matrix.artifacts must include unreal-compatibility-matrix.log")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    for version in UNREAL_COMPATIBILITY_VERSIONS:
        for token in (
            f"[unreal_matrix] UE {version}:",
            f"build/unreal-ci-results/ue-{version}",
            "[unreal_ci] Filter: AstralRT.*",
            "[unreal-results] OK:",
        ):
            if token not in text:
                raise ValueError(f"unreal_compatibility_matrix log is missing {token}")
    for token in UNREAL_COMPATIBILITY_FAILURE_TOKENS:
        if token in text:
            raise ValueError(f"unreal_compatibility_matrix log contains failure marker {token}")


def require_artifact_names(lane_name, paths):
    required = REQUIRED_ARTIFACT_NAMES.get(lane_name, ())
    if not required:
        return
    available = {path.name for path in paths}
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"{lane_name}.artifacts must include {', '.join(missing)}")


def validate_lane_artifacts(lane_name, lane, base_dir):
    paths = resolve_artifacts(lane_name, lane, base_dir)
    require_artifact_names(lane_name, paths)
    if lane_name == "comment_review":
        validate_comment_review_artifacts(paths)
    elif lane_name in UNREAL_CONTAINER_EVIDENCE:
        validate_unreal_container_artifacts(lane_name, paths)
    elif lane_name == "unreal_compatibility_matrix":
        validate_unreal_compatibility_artifacts(paths)
    elif lane_name == "multimodal_validation":
        validate_multimodal_artifacts(paths)


def validate_manifest(data, base_dir, phase):
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    if data.get("schema") != "astral.release.evidence.v1":
        raise ValueError("schema must be astral.release.evidence.v1")

    release = data.get("release")
    if not isinstance(release, dict):
        raise ValueError("release must be an object")
    require_text(release.get("version"), "release.version")
    require_text(release.get("git_commit"), "release.git_commit")

    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object")

    required_lanes = REQUIRED_LANES
    if phase == "pre-sign":
        required_lanes = tuple(lane for lane in REQUIRED_LANES if lane not in PRE_SIGN_EXCLUDED_LANES)

    missing = [lane for lane in required_lanes if lane not in evidence]
    if missing:
        raise ValueError(f"missing required lane(s): {', '.join(missing)}")

    for lane_name in required_lanes:
        lane = evidence[lane_name]
        if not isinstance(lane, dict):
            raise ValueError(f"{lane_name} must be an object")
        if lane.get("status") != "pass":
            raise ValueError(f"{lane_name}.status must be pass")
        require_text(lane.get("command"), f"{lane_name}.command")
        for token in REQUIRED_COMMAND_TOKENS.get(lane_name, ()):
            if token not in lane["command"]:
                raise ValueError(f"{lane_name}.command must include {token}")
        for token in FORBIDDEN_COMMAND_TOKENS.get(lane_name, ()):
            if token in lane["command"]:
                raise ValueError(f"{lane_name}.command must not include {token}")
        validate_lane_artifacts(lane_name, lane, base_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Astral release candidate evidence manifests."
    )
    parser.add_argument("manifest", help="Path to release-evidence.json")
    parser.add_argument(
        "--base-dir",
        help="Directory artifact paths are relative to (default: manifest directory)",
    )
    parser.add_argument(
        "--phase",
        choices=("complete", "pre-sign"),
        default="complete",
        help="Evidence phase to validate (default: complete)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = Path(args.base_dir) if args.base_dir else manifest_path.parent

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        validate_manifest(data, base_dir, args.phase)
    except OSError as exc:
        return fail(str(exc))
    except json.JSONDecodeError as exc:
        return fail(f"invalid JSON: {exc}")
    except ValueError as exc:
        return fail(str(exc))

    print(f"[release-evidence] OK: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
