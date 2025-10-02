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


def validate_artifacts(lane_name, lane, base_dir):
    artifacts = lane.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{lane_name}.artifacts must list at least one artifact")

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


def validate_manifest(data, base_dir):
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

    missing = [lane for lane in REQUIRED_LANES if lane not in evidence]
    if missing:
        raise ValueError(f"missing required lane(s): {', '.join(missing)}")

    for lane_name in REQUIRED_LANES:
        lane = evidence[lane_name]
        if not isinstance(lane, dict):
            raise ValueError(f"{lane_name} must be an object")
        if lane.get("status") != "pass":
            raise ValueError(f"{lane_name}.status must be pass")
        require_text(lane.get("command"), f"{lane_name}.command")
        validate_artifacts(lane_name, lane, base_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Astral release candidate evidence manifests."
    )
    parser.add_argument("manifest", help="Path to release-evidence.json")
    parser.add_argument(
        "--base-dir",
        help="Directory artifact paths are relative to (default: manifest directory)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = Path(args.base_dir) if args.base_dir else manifest_path.parent

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        validate_manifest(data, base_dir)
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
