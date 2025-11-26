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
import time
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
    # Backwards-compat shim.
    _http_download_resume(url, dst, token, expected_size=None)


def _http_download_resume(url: str, dst: Path, token: Optional[str], expected_size: Optional[int]) -> None:
    """
    Robust downloader with resume support:
      - Writes to <dst>.part and renames on success.
      - If a .part exists, resumes using HTTP Range when possible.
      - Retries transient read/timeout errors.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    chunk_size = 1024 * 1024
    connect_timeout_s = 60
    max_attempts = 8

    last_pct = -1
    last_print_mib = 0

    def open_request(start: int) -> urllib.response.addinfourl:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "astral-hf-gguf-sync/0.2")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        if start > 0:
            req.add_header("Range", f"bytes={start}-")
        return urllib.request.urlopen(req, timeout=connect_timeout_s)

    attempt = 0
    while True:
        attempt += 1
        start = tmp.stat().st_size if tmp.exists() else 0

        if expected_size is not None and start == expected_size:
            tmp.replace(dst)
            return

        if expected_size is not None and start > expected_size:
            # Corrupted partial; start over.
            tmp.unlink(missing_ok=True)
            start = 0

        try:
            with open_request(start) as resp:
                # Determine total size: prefer expected_size from HF API.
                total_bytes = expected_size
                if total_bytes is None:
                    total = resp.headers.get("Content-Length")
                    total_bytes = int(total) if total and total.isdigit() else None
                # If server ignored Range and returned full payload, restart the file.
                if start > 0 and getattr(resp, "status", 200) == 200:
                    tmp.unlink(missing_ok=True)
                    start = 0
                    return _http_download_resume(url, dst, token, expected_size)

                downloaded = start
                mode = "ab" if start > 0 else "wb"
                with open(tmp, mode) as f:
                    while True:
                        try:
                            chunk = resp.read(chunk_size)
                        except TimeoutError:
                            raise
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_bytes:
                            pct = int((downloaded * 100) // total_bytes)
                            mib = int(downloaded // (1024 * 1024))
                            if pct != last_pct or (mib - last_print_mib) >= 64:
                                sys.stderr.write(
                                    f"\r  {dst.name}: {pct}% ({mib} MiB/{total_bytes//(1024*1024)} MiB)"
                                )
                                sys.stderr.flush()
                                last_pct = pct
                                last_print_mib = mib
                if total_bytes:
                    sys.stderr.write("\n")

        except urllib.error.HTTPError as e:
            # If Range is not satisfiable, try restarting.
            if e.code == 416 and tmp.exists():
                tmp.unlink(missing_ok=True)
                if attempt < max_attempts:
                    continue
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code} for {url}: {body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt >= max_attempts:
                raise RuntimeError(f"Download failed after {attempt} attempts: {e}") from e
            sleep_s = min(60, 2 ** (attempt - 1))
            sys.stderr.write(f"\n[hf] transient download error ({e}); retrying in {sleep_s}s (attempt {attempt}/{max_attempts})\n")
            sys.stderr.flush()
            time.sleep(sleep_s)
            continue

        # Finished this connection. Verify size if known.
        final_size = tmp.stat().st_size if tmp.exists() else 0
        if expected_size is not None and final_size != expected_size:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Incomplete download for {dst.name}: got {final_size} bytes, expected {expected_size}"
                )
            # Continue/resume with a new request.
            continue

        tmp.replace(dst)
        return


def list_repo_files(repo: str, token: Optional[str], revision: str = "main") -> List[FileInfo]:
    # Prefer the repo tree endpoint because it reliably includes sizes (including for LFS files),
    # while /api/models/<repo> often omits "size" for siblings.
    repo_q = urllib.parse.quote(repo, safe="/")

    revision_q = urllib.parse.quote(revision, safe="")
    tree_url = f"{HF_API}/api/models/{repo_q}/tree/{revision_q}?recursive=1"
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
    ap_list.add_argument("--revision", default="main", help="Model revision to list (default: main)")

    ap_dl = sub.add_parser("download", help="Download GGUF files from a repo")
    ap_dl.add_argument("repo", help="Repo id, e.g. Qwen/Qwen3-8B-GGUF")
    ap_dl.add_argument("--out", required=True, help="Output root directory (models stored under <out>/<repo>/)")
    ap_dl.add_argument("--revision", default="main", help="Model revision to download (default: main)")
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
    revision = getattr(args, "revision", "main")
    try:
        all_files = list_repo_files(args.repo, token, revision)
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
        url = f"{HF_API}/{args.repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(f.path, safe='/')}"
        dst = repo_dir / Path(f.path).name
        if dst.exists() and dst.stat().st_size == f.size:
            sys.stderr.write(f"[hf] skip (exists): {dst.name}\n")
            continue
        sys.stderr.write(f"[hf] download: {f.path} ({fmt_size(f.size)})\n")
        _http_download_resume(url, dst, token, expected_size=f.size if f.size > 0 else None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
