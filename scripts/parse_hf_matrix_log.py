#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


FEATURE_ALIASES = {
    "features.embed enqueue+collect": "embed_mops",
    "features.kv state_save": "kv_save_mops",
    "features.kv state_load": "kv_load_mops",
    "features.grammar set_gbnf": "grammar_mops",
    "features.logprobs meta_drain": "logprobs_mops",
}


@dataclass
class Block:
    preset: str = ""
    backend: str = ""
    model: str = ""
    embed_model: str = ""
    metrics: Dict[str, str] = field(default_factory=dict)
    kv_bytes: str = ""
    failed: bool = False


_RE_BLOCK = re.compile(r"^## preset=(?P<preset>\S+)\s+backend=(?P<backend>\S+)\s*$")
_RE_MODEL = re.compile(r"^model=(?P<path>.+)\s*$")
_RE_EMBED_MODEL = re.compile(r"^embed_model=(?P<path>.+)\s*$")
_RE_RESULT = re.compile(
    r"^(?P<name>features\.[^\s].*?)\s{2,}(?P<mops>[0-9]+\.[0-9]+)\s+Mops/s\b"
)
_RE_KV_BYTES = re.compile(r"^features\.kv bytes\s+.*\s+(?P<bytes>[0-9]+)\s+bytes\s*$")


def _basename(p: str) -> str:
    p = p.strip()
    if not p:
        return ""
    return os.path.basename(p)


def parse_blocks(lines: Iterable[str]) -> List[Block]:
    blocks: List[Block] = []
    cur: Optional[Block] = None

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        if cur.preset or cur.backend or cur.model or cur.embed_model or cur.metrics or cur.kv_bytes or cur.failed:
            blocks.append(cur)
        cur = None

    for raw in lines:
        line = raw.rstrip("\n")

        m = _RE_BLOCK.match(line)
        if m:
            flush()
            cur = Block(preset=m.group("preset"), backend=m.group("backend"))
            continue

        if cur is None:
            continue

        if line.startswith("[bench] FAILED"):
            cur.failed = True
            continue

        m = _RE_MODEL.match(line)
        if m:
            cur.model = m.group("path").strip()
            continue

        m = _RE_EMBED_MODEL.match(line)
        if m:
            cur.embed_model = m.group("path").strip()
            continue

        m = _RE_KV_BYTES.match(line)
        if m:
            cur.kv_bytes = m.group("bytes")
            continue

        m = _RE_RESULT.match(line)
        if m:
            name = m.group("name").strip()
            key = FEATURE_ALIASES.get(name)
            if key:
                cur.metrics[key] = m.group("mops")
            continue

    flush()
    return blocks


def write_csv(blocks: List[Block], out_file: str) -> None:
    fieldnames = [
        "preset",
        "backend",
        "model",
        "embed_model",
        "status",
        "embed_mops",
        "kv_save_mops",
        "kv_load_mops",
        "grammar_mops",
        "logprobs_mops",
        "kv_bytes",
    ]
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for b in blocks:
            row = {k: "" for k in fieldnames}
            row["preset"] = b.preset
            row["backend"] = b.backend
            row["model"] = _basename(b.model)
            row["embed_model"] = _basename(b.embed_model)
            row["status"] = "FAIL" if b.failed else "OK"
            row["kv_bytes"] = b.kv_bytes
            for k, v in b.metrics.items():
                row[k] = v
            w.writerow(row)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Parse Astral HF bench-matrix logs into CSV.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input log file produced by scripts/run_hf_bench_matrix.sh")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    ap.add_argument("--require-pass", action="store_true", help="Fail when the log has no blocks or any failed block")
    args = ap.parse_args(argv)

    with open(args.in_path, "r", encoding="utf-8", errors="replace") as f:
        blocks = parse_blocks(f)

    if not blocks:
        print(f"[parse] no blocks found in {args.in_path}", file=sys.stderr)
        if args.require_pass:
            return 1

    if args.require_pass:
        failed = [b for b in blocks if b.failed]
        if failed:
            print(f"[parse] failed HF matrix block(s): {len(failed)}", file=sys.stderr)
            for block in failed[:10]:
                print(f"[parse] failed preset={block.preset} backend={block.backend} model={_basename(block.model)}", file=sys.stderr)
            return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    write_csv(blocks, args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
