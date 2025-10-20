#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


DEFAULT_SCAN_PATHS = (
    "include",
    "src",
    "backend_plugins",
    "plugins",
    "tests",
    "examples",
    "scripts",
    "docs",
    "cmake",
    "CMakeLists.txt",
    "README.md",
)

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".h",
    ".hpp",
}
HASH_SUFFIXES = {
    ".cmake",
    ".py",
    ".sh",
}
DOC_SUFFIXES = {
    ".md",
    ".txt",
}
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "Binaries",
    "Intermediate",
    "Saved",
    ".issue tracker",
}

MARKER_TOKENS = ("TO" + "DO", "FIX" + "ME", "HA" + "CK")
MARKER_RE = re.compile(r"\b(" + "|".join(MARKER_TOKENS) + r")\b")
BEADS_RE = re.compile(r"workspace-[A-Za-z0-9]+")


@dataclass(frozen=True)
class Entry:
    path: str
    line: int
    kind: str
    marker: str
    bead: str
    text: str


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def iter_files(root: Path, scan_paths: Sequence[str]) -> Iterator[Path]:
    for item in scan_paths:
        path = root / item
        if not path.exists() or is_excluded(path):
            continue
        if path.is_file():
            if should_scan(path):
                yield path
            continue
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and should_scan(candidate) and not is_excluded(candidate):
                yield candidate


def should_scan(path: Path) -> bool:
    if path.name == "CMakeLists.txt":
        return True
    return path.suffix in SOURCE_SUFFIXES or path.suffix in HASH_SUFFIXES or path.suffix in DOC_SUFFIXES


def normalize_text(text: str) -> str:
    parts: List[str] = []
    for raw in text.splitlines():
        part = raw.strip()
        while part.startswith("*"):
            part = part[1:].lstrip()
        if part.startswith("/") or part.startswith("!"):
            part = part[1:].lstrip()
        if part:
            parts.append(part)
    return " ".join(" ".join(parts).split())


def marker_for(text: str) -> str:
    match = MARKER_RE.search(text)
    if not match:
        return ""
    return match.group(1)


def bead_for(text: str) -> str:
    match = BEADS_RE.search(text)
    if not match:
        return ""
    return match.group(0)


def make_entry(path: Path, root: Path, line: int, kind: str, text: str) -> Entry | None:
    clean = normalize_text(text)
    if not clean:
        return None
    return Entry(
        path=str(path.relative_to(root)),
        line=line,
        kind=kind,
        marker=marker_for(clean),
        bead=bead_for(clean),
        text=clean,
    )


def collect_c_family(path: Path, root: Path, lines: Sequence[str]) -> List[Entry]:
    entries: List[Entry] = []
    in_block = False
    block_start = 0
    block_parts: List[str] = []

    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        cursor = 0
        while cursor < len(line):
            if in_block:
                end = line.find("*/", cursor)
                if end < 0:
                    block_parts.append(line[cursor:])
                    break
                block_parts.append(line[cursor:end])
                entry = make_entry(path, root, block_start, "block", "\n".join(block_parts))
                if entry:
                    entries.append(entry)
                block_parts = []
                in_block = False
                cursor = end + 2
                continue

            slash = line.find("//", cursor)
            block = line.find("/*", cursor)
            if slash < 0 and block < 0:
                break
            if slash >= 0 and (block < 0 or slash < block):
                entry = make_entry(path, root, line_no, "line", line[slash + 2 :])
                if entry:
                    entries.append(entry)
                break

            end = line.find("*/", block + 2)
            if end < 0:
                in_block = True
                block_start = line_no
                block_parts = [line[block + 2 :]]
                break
            entry = make_entry(path, root, line_no, "block", line[block + 2 : end])
            if entry:
                entries.append(entry)
            cursor = end + 2

    if in_block:
        entry = make_entry(path, root, block_start, "block", "\n".join(block_parts))
        if entry:
            entries.append(entry)
    return entries


def collect_hash_comments(path: Path, root: Path, lines: Sequence[str]) -> List[Entry]:
    entries: List[Entry] = []
    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        stripped = line.lstrip()
        if line_no == 1 and stripped.startswith("#!"):
            continue
        comment = hash_comment_text(line)
        if comment:
            entry = make_entry(path, root, line_no, "line", comment)
            if entry:
                entries.append(entry)
    return entries


def hash_comment_text(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            return line[index + 1 :]
    return ""


def collect_docs(path: Path, root: Path, lines: Sequence[str]) -> List[Entry]:
    entries: List[Entry] = []
    in_fence = False
    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        entry = make_entry(path, root, line_no, "doc", stripped)
        if entry:
            entries.append(entry)
    return entries


def collect_entries(root: Path, scan_paths: Sequence[str]) -> List[Entry]:
    entries: List[Entry] = []
    for path in iter_files(root, scan_paths):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        if path.suffix in SOURCE_SUFFIXES:
            entries.extend(collect_c_family(path, root, lines))
        elif path.name == "CMakeLists.txt" or path.suffix in HASH_SUFFIXES:
            entries.extend(collect_hash_comments(path, root, lines))
        elif path.suffix in DOC_SUFFIXES:
            entries.extend(collect_docs(path, root, lines))
    return entries


def print_tsv(entries: Iterable[Entry], limit: int) -> None:
    print("path\tline\tkind\tmarker\tbead\ttext")
    count = 0
    for entry in entries:
        if limit and count >= limit:
            break
        text = entry.text.replace("\t", " ").replace("\n", " ")
        print(f"{entry.path}\t{entry.line}\t{entry.kind}\t{entry.marker}\t{entry.bead}\t{text}")
        count += 1


def limited(entries: Sequence[Entry], limit: int) -> Sequence[Entry]:
    if limit <= 0:
        return entries
    return entries[:limit]


def print_summary(entries: Sequence[Entry]) -> None:
    markers = [entry for entry in entries if entry.marker]
    orphan_markers = [entry for entry in markers if not entry.bead]
    files = {entry.path for entry in entries}
    docs = sum(1 for entry in entries if entry.kind == "doc")
    comments = len(entries) - docs
    print(
        "comment_inventory "
        f"files={len(files)} comments={comments} doc_lines={docs} "
        f"markers={len(markers)} orphan_markers={len(orphan_markers)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory maintained Astral comments and doc prose for human review."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to scan")
    parser.add_argument(
        "--path",
        action="append",
        dest="scan_paths",
        help="Relative path to scan; repeat to override the default maintained paths",
    )
    parser.add_argument("--format", choices=("tsv", "json", "summary"), default="tsv")
    parser.add_argument("--limit", type=int, default=0, help="Maximum TSV rows to print; 0 prints all rows")
    parser.add_argument(
        "--fail-orphan-markers",
        action="store_true",
        help="Return non-zero when tracked cleanup markers lack a issue tracker issue id",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    scan_paths = tuple(args.scan_paths) if args.scan_paths else DEFAULT_SCAN_PATHS
    entries = collect_entries(root, scan_paths)
    orphan_markers = [entry for entry in entries if entry.marker and not entry.bead]

    if args.format == "summary":
        print_summary(entries)
    elif args.format == "json":
        print(json.dumps([asdict(entry) for entry in limited(entries, args.limit)], indent=2))
    else:
        print_tsv(entries, args.limit)

    if args.fail_orphan_markers and orphan_markers:
        for entry in orphan_markers:
            print(
                f"[comment-inventory] orphan {entry.marker} marker at {entry.path}:{entry.line}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
