#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


EXIT_OK = 0
EXPECTED_USAGE_EXIT = 2
EXPECTED_ARG_COUNT = 2
EXPECTED_QWEN_CONTEXT = 40960
EXPECTED_QWEN_SIZE_BYTES = 639446688
EXPECTED_EMBED_DIMENSION = 1024
MODEL_TYPE_EMBEDDING = "embedding"
QWEN_TEXT_PRESET = "qwen3-0.6b-q8"
QWEN_EMBED_PRESET = "qwen3-embed-0.6b-q8"
QWEN_TEXT_FILE = "Qwen3-0.6B-Q8_0.gguf"
QWEN_TEXT_REPO = "Qwen/Qwen3-0.6B-GGUF"
UNKNOWN_PRESET = "missing-preset"
CUSTOM_OUTPUT_DIR = "build/model-preset-smoke"


def run_tool(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(root / "scripts" / "model_preset_tool.py"), *args]
    return subprocess.run(
        command,
        cwd=root,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main(argv: list[str]) -> int:
    if len(argv) != EXPECTED_ARG_COUNT:
        print("usage: model_preset_tool_smoke.py <repo-root>", file=sys.stderr)
        return EXPECTED_USAGE_EXIT

    root = Path(argv[1]).resolve()

    manifest = run_tool(root, "validate-manifest").stdout
    require("manifest OK" in manifest, "manifest validator did not report success")

    info = json.loads(run_tool(root, "info", QWEN_TEXT_PRESET, "--dir", CUSTOM_OUTPUT_DIR).stdout)
    require(info["name"] == QWEN_TEXT_PRESET, "wrong preset name")
    require(info["repo"] == QWEN_TEXT_REPO, "wrong repository")
    require(info["filename"] == QWEN_TEXT_FILE, "wrong filename")
    require(info["context_length"] == EXPECTED_QWEN_CONTEXT, "wrong context length")
    require(info["size_bytes"] == EXPECTED_QWEN_SIZE_BYTES, "wrong size")
    require(info["path"].endswith(f"{CUSTOM_OUTPUT_DIR}/{QWEN_TEXT_FILE}"), "wrong resolved path")
    require(QWEN_TEXT_PRESET in info["download_command"], "download command does not name preset")

    embed = json.loads(run_tool(root, "info", QWEN_EMBED_PRESET).stdout)
    require(embed["embedding_dimension"] == EXPECTED_EMBED_DIMENSION, "wrong embedding dimension")
    require(embed["model_type"] == MODEL_TYPE_EMBEDDING, "wrong embedding model type")

    dry_run = run_tool(root, "download", "--preset", QWEN_TEXT_PRESET, "--dry-run").stdout
    for marker in ("preset:", "path:", "url:", "size_bytes:", "sha256:", "command:"):
        require(marker in dry_run, f"dry run missed {marker}")

    unknown = run_tool(root, "download", "--preset", UNKNOWN_PRESET, "--dry-run", check=False)
    require(unknown.returncode == EXPECTED_USAGE_EXIT, "unknown preset should fail")
    require("Unknown preset" in unknown.stderr, "unknown preset error should name the problem")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
