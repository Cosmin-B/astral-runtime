#!/usr/bin/env python3
"""
hf_gguf_sync.py

List and download *.gguf files from Hugging Face repositories.

This is intentionally small and dependency-light:
  - Prefers using the Hugging Face REST API (no auth needed for public repos).
  - Downloads via HTTPS streaming to a target directory.

Usage examples:
  python3 scripts/hf_gguf_sync.py list Qwen/Qwen3-8B-GGUF
  python3 scripts/hf_gguf_sync.py download Qwen/Qwen3-8B-GGUF --out tests/models/hf --prefer q4 --max-gb 25
  python3 scripts/hf_gguf_sync.py download Qwen/Qwen3-8B-GGUF --out tests/models/hf --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


HF_API = "https://huggingface.co"


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int


def _http_get_json(url: str, token: Optional[str]) -> object:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "astral-hf-gguf-sync/0.1")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} for {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed for {url}: {e}") from e


def _http_download(url: str, dst: Path, token: Optional[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "astral-hf-gguf-sync/0.1")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req, timeout=60) as resp:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None

        downloaded = 0
        last_pct = -1
        last_print_mib = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = int((downloaded * 100) // total_bytes)
                    mib = int(downloaded // (1024 * 1024))
                    # Throttle progress output: print on percent change or every 64 MiB.
                    if pct != last_pct or (mib - last_print_mib) >= 64:
                        sys.stderr.write(
                            f"\r  {dst.name}: {pct}% ({mib} MiB/{total_bytes//(1024*1024)} MiB)"
                        )
                        sys.stderr.flush()
                        last_pct = pct
                        last_print_mib = mib
        if total_bytes:
            sys.stderr.write("\n")

    tmp.replace(dst)


def list_repo_files(repo: str, token: Optional[str]) -> List[FileInfo]:
    # Prefer the repo tree endpoint because it reliably includes sizes (including for LFS files),
    # while /api/models/<repo> often omits "size" for siblings.
    repo_q = urllib.parse.quote(repo, safe="/")

    tree_url = f"{HF_API}/api/models/{repo_q}/tree/main?recursive=1"
    data = _http_get_json(tree_url, token)
    if isinstance(data, list):
        out: List[FileInfo] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            if it.get("type") != "file":
                continue
            path = it.get("path")
            size = it.get("size")
            if isinstance(path, str) and isinstance(size, int):
                out.append(FileInfo(path=path, size=size))
        return out

    # Fallback: /api/models/<repo> (may not include sizes for GGUF repos).
    model_url = f"{HF_API}/api/models/{repo_q}"
    data = _http_get_json(model_url, token)
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected API response (expected object)")

    siblings = data.get("siblings", [])
    out2: List[FileInfo] = []
    for s in siblings:
        if not isinstance(s, dict):
            continue
        path = s.get("rfilename")
        size = s.get("size")
        if isinstance(path, str) and isinstance(size, int):
            out2.append(FileInfo(path=path, size=size))
        elif isinstance(path, str):
            # Unknown size; list anyway (but --max-gb cannot filter these).
            out2.append(FileInfo(path=path, size=0))
    return out2


def filter_gguf(files: Iterable[FileInfo]) -> List[FileInfo]:
    out = [f for f in files if f.path.lower().endswith(".gguf")]
    out.sort(key=lambda x: (x.size, x.path))
    return out


def _pick_preferred(files: List[FileInfo], prefer: List[str], max_count: int) -> List[FileInfo]:
    if not prefer:
        return files[:max_count]

    prefer_patterns: List[Tuple[str, re.Pattern[str]]] = []
    for p in prefer:
        prefer_patterns.append((p, re.compile(re.escape(p), re.IGNORECASE)))

    chosen: List[FileInfo] = []
    used = set()
    for label, pat in prefer_patterns:
        for f in files:
            if f.path in used:
                continue
            if pat.search(Path(f.path).name):
                chosen.append(f)
                used.add(f.path)
                if len(chosen) >= max_count:
                    return chosen
    # Fill remaining with smallest.
    for f in files:
        if f.path in used:
            continue
        chosen.append(f)
        if len(chosen) >= max_count:
            break
    return chosen


def fmt_size(n: int) -> str:
    gb = n / (1024.0 * 1024.0 * 1024.0)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    mb = n / (1024.0 * 1024.0)
    return f"{mb:.1f} MB"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_list = sub.add_parser("list", help="List GGUF files in a repo")
    ap_list.add_argument("repo", help="Repo id, e.g. Qwen/Qwen3-8B-GGUF")

    ap_dl = sub.add_parser("download", help="Download GGUF files from a repo")
    ap_dl.add_argument("repo", help="Repo id, e.g. Qwen/Qwen3-8B-GGUF")
    ap_dl.add_argument("--out", required=True, help="Output root directory (models stored under <out>/<repo>/)")
    ap_dl.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
                       help="HF token (or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN)")
    ap_dl.add_argument("--max-gb", type=float, default=25.0, help="Skip any single file larger than this (default: 25GB)")
    ap_dl.add_argument("--all", action="store_true", help="Download all GGUF files (still respects --max-gb)")
    ap_dl.add_argument("--max-count", type=int, default=4, help="Max number of GGUF files to download when not --all")
    ap_dl.add_argument("--prefer", action="append", default=[],
                       help="Prefer filenames containing this substring (repeatable), e.g. --prefer Q4 --prefer Q5")
    ap_dl.add_argument("--include", action="append", default=[],
                       help="Only include filenames matching this regex (repeatable)")

    args = ap.parse_args(argv)

    token = getattr(args, "token", None)
    try:
        all_files = list_repo_files(args.repo, token)
    except Exception as e:
        sys.stderr.write(f"[hf] failed to query repo {args.repo}: {e}\n")
        return 2

    ggufs = filter_gguf(all_files)
    if args.cmd == "list":
        if not ggufs:
            print("(no .gguf files found)")
            return 0
        for f in ggufs:
            print(f"{fmt_size(f.size):>9}  {f.path}")
        return 0

    assert args.cmd == "download"
    if not ggufs:
        sys.stderr.write("[hf] no .gguf files found; skipping\n")
        return 0

    include_res: List[re.Pattern[str]] = []
    for s in args.include:
        try:
            include_res.append(re.compile(s))
        except re.error as e:
            sys.stderr.write(f"[hf] invalid --include regex '{s}': {e}\n")
            return 2

    if include_res:
        ggufs = [f for f in ggufs if any(r.search(f.path) for r in include_res)]

    max_bytes = int(args.max_gb * 1024.0 * 1024.0 * 1024.0)
    ggufs = [f for f in ggufs if f.size <= max_bytes]
    if not ggufs:
        sys.stderr.write("[hf] all gguf files filtered out (size/regex)\n")
        return 0

    if args.all:
        chosen = ggufs
    else:
        chosen = _pick_preferred(ggufs, args.prefer, args.max_count)

    out_root = Path(args.out).expanduser().resolve()
    repo_dir = out_root / args.repo.replace("/", "__")
    repo_dir.mkdir(parents=True, exist_ok=True)

    sys.stderr.write(f"[hf] repo={args.repo} gguf_files={len(ggufs)} download={len(chosen)} out={repo_dir}\n")

    for f in chosen:
        url = f"{HF_API}/{args.repo}/resolve/main/{urllib.parse.quote(f.path, safe='/')}"
        dst = repo_dir / Path(f.path).name
        if dst.exists() and dst.stat().st_size == f.size:
            sys.stderr.write(f"[hf] skip (exists): {dst.name}\n")
            continue
        sys.stderr.write(f"[hf] download: {f.path} ({fmt_size(f.size)})\n")
        _http_download(url, dst, token)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
