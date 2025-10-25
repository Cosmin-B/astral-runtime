#!/usr/bin/env python3
"""Validate that maintained error-code references match AstralErr."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ENUM_RE = re.compile(r"enum\s*\{(?P<body>.*?)\};", re.S)
CODE_RE = re.compile(r"\b(ASTRAL_OK|ASTRAL_E_[A-Z0-9_]+)\b")
SCAN_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cs",
    ".cxx",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
}
SKIP_DIRS = {
    ".issue tracker",
    ".git",
    "__pycache__",
    "build",
    "dist",
    "external",
}
SKIP_FILES = {
    Path("docs/FEATURE_PARITY.md"),
}


@dataclass(frozen=True)
class StaleReference:
    path: Path
    line: int
    code: str


def parse_header_codes(header: Path) -> list[str]:
    text = header.read_text(encoding="utf-8")
    alias_pos = text.find("typedef int32_t AstralErr;")
    if alias_pos < 0:
        raise ValueError(f"could not find AstralErr typedef in {header}")
    match = ENUM_RE.search(text, alias_pos)
    if not match:
        raise ValueError(f"could not find AstralErr code enum in {header}")
    codes: list[str] = []
    for code in CODE_RE.findall(match.group("body")):
        if code not in codes:
            codes.append(code)
    if not codes:
        raise ValueError(f"found AstralErr enum in {header}, but no codes")
    return codes


def should_scan_file(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    if rel in SKIP_FILES or path.suffix.lower() not in SCAN_SUFFIXES:
        return False
    return True


def iter_scan_files(root: Path) -> list[Path]:
    root = root.resolve()
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
            if should_scan_file(root, path):
                files.append(path)
    return sorted(files)


def find_stale_references(root: Path, allowed_codes: set[str]) -> list[StaleReference]:
    root = root.resolve()
    stale: list[StaleReference] = []
    for path in iter_scan_files(root):
        rel = path.relative_to(root)
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            for code in CODE_RE.findall(line):
                if code not in allowed_codes:
                    stale.append(StaleReference(rel, line_no, code))
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--doc", required=True, type=Path)
    parser.add_argument(
        "--scan-root",
        type=Path,
        help="Optional repository root to scan for stale ASTRAL_E_* references.",
    )
    args = parser.parse_args()

    try:
        header_codes = parse_header_codes(args.header)
        doc_text = args.doc.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[error-taxonomy] {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[error-taxonomy] {exc}", file=sys.stderr)
        return 2

    header_code_set = set(header_codes)
    doc_codes = set(CODE_RE.findall(doc_text))
    missing = [code for code in header_codes if code not in doc_codes]
    stale = sorted(code for code in doc_codes if code not in header_code_set)

    if missing:
        print(f"[error-taxonomy] missing documented codes: {', '.join(missing)}", file=sys.stderr)
        return 1
    if stale:
        print(f"[error-taxonomy] stale documented codes: {', '.join(stale)}", file=sys.stderr)
        return 1
    if args.scan_root is not None:
        try:
            stale_refs = find_stale_references(args.scan_root, header_code_set)
        except OSError as exc:
            print(f"[error-taxonomy] {exc}", file=sys.stderr)
            return 2
        if stale_refs:
            for item in stale_refs:
                print(
                    f"[error-taxonomy] stale public error reference: "
                    f"{item.path}:{item.line}: {item.code}",
                    file=sys.stderr,
                )
            return 1

    print(f"[error-taxonomy] OK: {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
