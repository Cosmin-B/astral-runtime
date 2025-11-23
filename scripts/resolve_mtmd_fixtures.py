#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from validate_mtmd_fixture_manifest import _load, validate_manifest


KEYS = {
    ("vision", "model"): "vision_model",
    ("vision", "projector"): "vision_media",
    ("audio", "model"): "audio_model",
    ("audio", "projector"): "audio_media",
    ("audio", "vocoder"): "audio_vocoder",
    ("audio", "tokenizer"): "audio_tokenizer",
}


def resolve_fixture_paths(manifest: Path, fixture_dir: Path) -> Dict[str, Path]:
    data = _load(manifest)
    validate_manifest(data)

    resolved: Dict[str, Path] = {}
    for repo in data["repos"]:
        role = repo["role"]
        required_files = repo["required_files"]
        for file_key, filename in required_files.items():
            output_key = KEYS.get((role, file_key))
            if output_key is not None:
                resolved[output_key] = fixture_dir / filename

    required = ("vision_model", "vision_media", "audio_model", "audio_media")
    missing = [key for key in required if key not in resolved]
    if missing:
        raise ValueError(f"manifest did not resolve required keys: {', '.join(missing)}")
    return resolved


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Resolve Astral MTMD fixture paths from a pinned manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        resolved = resolve_fixture_paths(args.manifest, args.fixture_dir)
    except Exception as exc:
        print(f"[mtmd-fixtures] {exc}", file=sys.stderr)
        return 1

    for key in sorted(resolved):
        print(f"{key}\t{resolved[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
