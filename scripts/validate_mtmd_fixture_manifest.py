#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REV_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_ROLES = ("vision", "audio")
ROLE_REQUIRED_FILES = {
    "vision": ("model", "projector"),
    "audio": ("model", "projector", "vocoder", "tokenizer"),
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def _require_string(obj: Dict[str, Any], key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}.{key} must be a non-empty string")
    return value


def _require_list(obj: Dict[str, Any], key: str, where: str) -> List[str]:
    value = obj.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}.{key} must be a non-empty list")
    out: List[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{where}.{key}[{i}] must be a non-empty string")
        out.append(item)
    return out


def _validate_regexes(patterns: Iterable[str], where: str) -> None:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"{where}.include has invalid regex {pattern!r}: {exc}") from exc


def validate_manifest(data: Dict[str, Any]) -> None:
    if data.get("schema") != "astral.mtmd-fixtures.v1":
        raise ValueError("schema must be astral.mtmd-fixtures.v1")

    if _require_string(data, "license_id", "manifest") != "lfm1.0":
        raise ValueError("manifest.license_id must be lfm1.0")
    _require_string(data, "license_name", "manifest")
    license_url = _require_string(data, "license_url", "manifest")
    if not license_url.startswith("https://huggingface.co/"):
        raise ValueError("manifest.license_url must point at Hugging Face")

    repos = data.get("repos")
    if not isinstance(repos, list) or not repos:
        raise ValueError("manifest.repos must be a non-empty list")

    seen_roles = set()
    for index, repo in enumerate(repos):
        where = f"repos[{index}]"
        if not isinstance(repo, dict):
            raise ValueError(f"{where} must be an object")
        role = _require_string(repo, "role", where)
        if role not in ROLE_REQUIRED_FILES:
            raise ValueError(f"{where}.role must be one of {', '.join(ROLE_REQUIRED_FILES)}")
        if role in seen_roles:
            raise ValueError(f"duplicate role {role}")
        seen_roles.add(role)

        _require_string(repo, "repo", where)
        revision = _require_string(repo, "revision", where)
        if not REV_RE.match(revision):
            raise ValueError(f"{where}.revision must be a 40-character commit SHA")
        if repo.get("mode") != "all":
            raise ValueError(f"{where}.mode must be all")
        max_gb = repo.get("max_gb_per_file")
        if not isinstance(max_gb, (int, float)) or max_gb <= 0:
            raise ValueError(f"{where}.max_gb_per_file must be a positive number")

        includes = _require_list(repo, "include", where)
        _validate_regexes(includes, where)

        required_files = repo.get("required_files")
        if not isinstance(required_files, dict):
            raise ValueError(f"{where}.required_files must be an object")
        for key in ROLE_REQUIRED_FILES[role]:
            filename = _require_string(required_files, key, f"{where}.required_files")
            if not filename.endswith(".gguf"):
                raise ValueError(f"{where}.required_files.{key} must be a GGUF filename")
            if not any(re.compile(pattern).search(filename) for pattern in includes):
                raise ValueError(f"{where}.required_files.{key} is not matched by include regexes")

    missing = [role for role in REQUIRED_ROLES if role not in seen_roles]
    if missing:
        raise ValueError(f"missing required MTMD roles: {', '.join(missing)}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate Astral MTMD fixture manifests.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)

    try:
        validate_manifest(_load(args.manifest))
    except Exception as exc:
        print(f"[mtmd-fixtures] {exc}", file=sys.stderr)
        return 1

    print(f"[mtmd-fixtures] OK: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
