#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
DEFAULT_SUFFIX = ".cpp"
FORMAT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
}


def run_text(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


def changed_ranges(repo: Path, path: str) -> list[tuple[int, int]]:
    diff = run_text(["git", "diff", "--cached", "-U0", "--", path], repo)
    ranges: list[tuple[int, int]] = []
    for line in diff.splitlines():
        match = HUNK_RE.match(line)
        if match is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count == 0:
            continue
        ranges.append((start, start + count - 1))
    return ranges


def staged_file(repo: Path, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f":{path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def check_file(repo: Path, clang_format: str, style_file: Path, path: str) -> int:
    suffix = Path(path).suffix
    if suffix not in FORMAT_EXTENSIONS:
        suffix = DEFAULT_SUFFIX
    ranges = changed_ranges(repo, path)
    if not ranges:
        return 0

    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(staged_file(repo, path))
        handle.flush()
        command = [
            clang_format,
            "--dry-run",
            "--Werror",
            f"--style=file:{style_file}",
        ]
        for start, end in ranges:
            command.append(f"--lines={start}:{end}")
        command.append(handle.name)
        result = subprocess.run(command, cwd=repo, text=True, stderr=subprocess.PIPE)
        if result.returncode != 0:
            sys.stderr.write(f"[pre-commit] clang-format changed lines in {path}\n")
            sys.stderr.write(result.stderr.replace(handle.name, path))
        return result.returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang-format", default="clang-format")
    parser.add_argument("--repo", default=".")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    style_file = repo / ".clang-format"
    failed = 0
    for path in args.paths:
        if check_file(repo, args.clang_format, style_file, path) != 0:
            failed = 1
    return failed


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
