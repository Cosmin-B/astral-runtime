#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List


BASE_TOKENS = (
    "AstralSampleAutoQuit",
    "Mounted IoStore container",
    "Mounted Pak file",
    "Astral sample: media feed demo loaded",
    "texture image",
    "PCM16 audio",
    "Astral sample: packaged content bytes read",
    "Astral sample: packaged content memory model loaded",
    "Astral sample: saved cache bytes read",
    "Astral sample: saved cache memory model loaded",
    "Astral sample: RAG search top key",
)

REAL_MODEL_TOKENS = (
    "Astral sample: backend=cpu",
    "Astral sample: canceled stream wait result",
)

EMBEDDING_TOKENS = (
    "embedding_model=",
    "Astral sample: embedding dimension",
)

FAILURE_TOKENS = (
    "Runtime failed",
    "Fatal error",
    "Assertion failed",
    "Ensure condition failed",
    "AutomationTool exiting with ExitCode=1",
    "packaged content model read failed",
    "saved cache write failed",
    "saved cache read failed",
    "memory model load failed",
    "memory search demo failed",
    "media feed demo failed",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _require_tokens(text: str, tokens: Iterable[str], context: str) -> None:
    for token in tokens:
        if token not in text:
            raise ValueError(f"{context} is missing {token}")


def validate_runtime_log(
    path: Path,
    *,
    expect_model: str = "",
    expect_embedding_model: str = "",
    expect_engine_version: str = "",
    require_real_model: bool = False,
    require_embedding: bool = False,
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty Unreal sample runtime log: {path}")

    text = _read(path)
    _require_tokens(text, BASE_TOKENS, "Unreal sample runtime log")

    if expect_engine_version and expect_engine_version not in text:
        raise ValueError(f"Unreal sample runtime log is missing engine version {expect_engine_version}")

    if expect_model:
        if expect_model not in text:
            raise ValueError(f"Unreal sample runtime log is missing model path {expect_model}")
        require_real_model = True

    if expect_embedding_model:
        if expect_embedding_model not in text:
            raise ValueError(f"Unreal sample runtime log is missing embedding model path {expect_embedding_model}")
        require_embedding = True

    if require_real_model:
        _require_tokens(text, REAL_MODEL_TOKENS, "Unreal sample runtime log")

    if require_embedding:
        _require_tokens(text, EMBEDDING_TOKENS, "Unreal sample runtime log")

    for token in FAILURE_TOKENS:
        if token in text:
            raise ValueError(f"Unreal sample runtime log contains failure marker {token}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate packaged Unreal sample runtime evidence.")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--expect-model", default="")
    parser.add_argument("--expect-embedding-model", default="")
    parser.add_argument("--expect-engine-version", default="")
    parser.add_argument("--require-real-model", action="store_true")
    parser.add_argument("--require-embedding", action="store_true")
    args = parser.parse_args(argv)

    try:
        validate_runtime_log(
            args.log,
            expect_model=args.expect_model,
            expect_embedding_model=args.expect_embedding_model,
            expect_engine_version=args.expect_engine_version,
            require_real_model=args.require_real_model,
            require_embedding=args.require_embedding,
        )
    except Exception as exc:
        print(f"[unreal-sample-runtime] {exc}", file=sys.stderr)
        return 1

    print(f"[unreal-sample-runtime] OK: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
