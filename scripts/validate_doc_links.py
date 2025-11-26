#!/usr/bin/env python3
"""Validate local Markdown links in maintained repository docs."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
SCAN_SUFFIXES = {".md", ".txt"}
SKIP_DIRS = {
    ".git",
    "build",
    "external",
    "__pycache__",
}
SKIP_FILES = {
    Path("docs/FEATURE_PARITY.md"),
}


@dataclass(frozen=True)
class BrokenLink:
    path: Path
    line: int
    target: str
    resolved: Path


def iter_doc_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_DIRS and (rel_dir / name) not in SKIP_DIRS
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(root)
            if rel in SKIP_FILES or path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            files.append(path)
    return sorted(files)


def is_external_or_anchor(target: str) -> bool:
    lower = target.lower()
    return (
        not target
        or target.startswith("#")
        or "://" in target
        or lower.startswith("mailto:")
        or lower.startswith("tel:")
    )


def strip_markdown_target(raw: str) -> str:
    target = raw.strip()
    if not target:
        return target
    if target.startswith("<"):
        end = target.find(">")
        if end != -1:
            target = target[1:end]
    else:
        # Drop an optional Markdown title: [text](path "title").
        quote_positions = [pos for pos in (target.find(' "'), target.find(" '")) if pos != -1]
        if quote_positions:
            target = target[: min(quote_positions)]
    return target.split("#", 1)[0].strip()


def should_skip_target(target: str) -> bool:
    return (
        is_external_or_anchor(target)
        or any(ch in target for ch in "*?<>")
        or target.startswith("UNREAL_EDITOR=")
        or target.startswith("UNITY_EDITOR=")
    )


def resolve_target(root: Path, doc_path: Path, target: str) -> Path | None:
    if target.startswith("/"):
        absolute = Path(target)
        try:
            absolute.relative_to(root)
        except ValueError:
            return None
        return absolute
    return (doc_path.parent / target).resolve()


def validate(root: Path) -> list[BrokenLink]:
    root = root.resolve()
    broken: list[BrokenLink] = []
    for path in iter_doc_files(root):
        in_fence = False
        fence_marker = ""
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = ""
                continue
            if in_fence:
                continue

            for match in LINK_RE.finditer(line):
                target = strip_markdown_target(match.group(2))
                if should_skip_target(target):
                    continue
                resolved = resolve_target(root, path, target)
                if resolved is None:
                    continue
                if not resolved.exists():
                    broken.append(
                        BrokenLink(
                            path=path.relative_to(root),
                            line=line_no,
                            target=target,
                            resolved=resolved,
                        )
                    )
    return broken


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan")
    args = parser.parse_args()

    root = Path(args.root)
    broken = validate(root)
    if broken:
        for item in broken:
            print(
                f"{item.path}:{item.line}: broken local Markdown link "
                f"{item.target!r} -> {item.resolved}",
                file=sys.stderr,
            )
        return 1
    print("[doc-links] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
