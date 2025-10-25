#!/usr/bin/env python3
"""Validate that the public error taxonomy matches AstralErr."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ENUM_RE = re.compile(r"enum\s*\{(?P<body>.*?)\};", re.S)
CODE_RE = re.compile(r"\b(ASTRAL_OK|ASTRAL_E_[A-Z0-9_]+)\b")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--doc", required=True, type=Path)
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

    doc_codes = set(CODE_RE.findall(doc_text))
    missing = [code for code in header_codes if code not in doc_codes]
    stale = sorted(code for code in doc_codes if code not in set(header_codes))

    if missing:
        print(f"[error-taxonomy] missing documented codes: {', '.join(missing)}", file=sys.stderr)
        return 1
    if stale:
        print(f"[error-taxonomy] stale documented codes: {', '.join(stale)}", file=sys.stderr)
        return 1

    print(f"[error-taxonomy] OK: {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
