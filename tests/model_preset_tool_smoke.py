#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


EXIT_OK = 0
EXPECTED_USAGE_EXIT = 2
EXPECTED_ARG_COUNT = 2
EXPECTED_QWEN_CONTEXT = 40960
EXPECTED_QWEN_SIZE_BYTES = 639446688
EXPECTED_EMBED_DIMENSION = 1024
INVALID_CUSTOM_MIN_BYTES = 0
VALID_CUSTOM_MIN_BYTES = 1
MODEL_TYPE_EMBEDDING = "embedding"
MODEL_TYPE_TEXT = "text"
QWEN_TEXT_PRESET = "qwen3-0.6b-q8"
QWEN_EMBED_PRESET = "qwen3-embed-0.6b-q8"
QWEN_TEXT_FILE = "Qwen3-0.6B-Q8_0.gguf"
QWEN_TEXT_REPO = "Qwen/Qwen3-0.6B-GGUF"
UNKNOWN_PRESET = "missing-preset"
CUSTOM_OUTPUT_DIR = "build/model-preset-smoke"
CUSTOM_MODEL_URL = "https://example.test/model.gguf"
CUSTOM_MODEL_FILE = "model.gguf"
CUSTOM_BAD_MODEL_FILE = "../model.gguf"
CUSTOM_BAD_SHA256 = "abc"
STATUS_MISSING = "missing"
STATUS_PARTIAL = "partial"
STATUS_INVALID = "invalid"
STATUS_READY = "ready"
PACKAGE_PRESET = "gpt2-q2k"


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
    require(info["include_in_package"] is False, "Qwen preset should not package by default")
    require(info["path"].endswith(f"{CUSTOM_OUTPUT_DIR}/{QWEN_TEXT_FILE}"), "wrong resolved path")
    require(QWEN_TEXT_PRESET in info["download_command"], "download command does not name preset")

    embed = json.loads(run_tool(root, "info", QWEN_EMBED_PRESET).stdout)
    require(embed["embedding_dimension"] == EXPECTED_EMBED_DIMENSION, "wrong embedding dimension")
    require(embed["model_type"] == MODEL_TYPE_EMBEDDING, "wrong embedding model type")
    require(embed["include_in_package"] is False, "Qwen embedding preset should not package by default")

    embedding_presets = json.loads(run_tool(root, "list", "--type", MODEL_TYPE_EMBEDDING, "--format", "json").stdout)
    require(any(row["name"] == QWEN_EMBED_PRESET for row in embedding_presets), "embedding list missed Qwen embed preset")
    require(all(row["model_type"] == MODEL_TYPE_EMBEDDING for row in embedding_presets), "embedding list mixed model types")

    embedding_status = json.loads(
        run_tool(root, "status-all", "--type", MODEL_TYPE_EMBEDDING, "--dir", CUSTOM_OUTPUT_DIR).stdout
    )
    require(any(row["name"] == QWEN_EMBED_PRESET for row in embedding_status), "embedding status missed Qwen embed preset")
    require(all(row["model_type"] == MODEL_TYPE_EMBEDDING for row in embedding_status), "embedding status mixed model types")
    require(all("status" in row and "download_command" in row for row in embedding_status), "status rows are incomplete")

    text_presets = run_tool(root, "list", "--type", MODEL_TYPE_TEXT).stdout
    require(QWEN_TEXT_PRESET in text_presets, "text list missed Qwen text preset")
    require(QWEN_EMBED_PRESET not in text_presets, "text list included embedding preset")

    package_presets = json.loads(run_tool(root, "list", "--package", "--format", "json").stdout)
    require(any(row["name"] == PACKAGE_PRESET for row in package_presets), "package list missed default text preset")
    require(all(row["include_in_package"] is True for row in package_presets), "package list included disabled preset")

    package_status = run_tool(root, "status-all", "--package", "--format", "text", "--dir", CUSTOM_OUTPUT_DIR).stdout
    require(PACKAGE_PRESET in package_status, "package status missed default text preset")
    require(QWEN_TEXT_PRESET not in package_status, "package status included disabled preset")

    dry_run = run_tool(root, "download", "--preset", QWEN_TEXT_PRESET, "--dry-run").stdout
    for marker in ("preset:", "path:", "url:", "size_bytes:", "sha256:", "command:"):
        require(marker in dry_run, f"dry run missed {marker}")

    with tempfile.TemporaryDirectory() as temp_dir:
        missing_status = json.loads(run_tool(root, "status", QWEN_TEXT_PRESET, "--dir", temp_dir).stdout)
        require(missing_status["status"] == STATUS_MISSING, "missing status did not report missing")
        require(missing_status["present_bytes"] == 0, "missing status should have no present bytes")
        require(missing_status["partial_bytes"] == 0, "missing status should have no partial bytes")
        require(QWEN_TEXT_PRESET in missing_status["download_command"], "status should include repeatable command")

        part_path = Path(temp_dir) / f"{QWEN_TEXT_FILE}.part"
        part_path.write_bytes(b"partial")
        partial_status = json.loads(run_tool(root, "status", QWEN_TEXT_PRESET, "--dir", temp_dir).stdout)
        require(partial_status["status"] == STATUS_PARTIAL, "partial status did not report partial")
        require(partial_status["partial_bytes"] == len(b"partial"), "partial byte count mismatch")

        model_path = Path(temp_dir) / QWEN_TEXT_FILE
        model_path.write_bytes(b"invalid")
        invalid_status = json.loads(run_tool(root, "status", QWEN_TEXT_PRESET, "--dir", temp_dir).stdout)
        require(invalid_status["status"] == STATUS_INVALID, "invalid status did not report invalid")
        require("size mismatch" in invalid_status["error"], "invalid status should name size mismatch")

    status_text = run_tool(root, "status", QWEN_TEXT_PRESET, "--dir", CUSTOM_OUTPUT_DIR, "--format", "text").stdout
    for marker in ("preset:", "status:", "path:", "present_bytes:", "partial_bytes:", "expected_bytes:", "command:"):
        require(marker in status_text, f"status text missed {marker}")

    unknown = run_tool(root, "download", "--preset", UNKNOWN_PRESET, "--dry-run", check=False)
    require(unknown.returncode == EXPECTED_USAGE_EXIT, "unknown preset should fail")
    require("Unknown preset" in unknown.stderr, "unknown preset error should name the problem")

    bad_file = run_tool(
        root,
        "download",
        "--url",
        CUSTOM_MODEL_URL,
        "--file",
        CUSTOM_BAD_MODEL_FILE,
        "--dry-run",
        check=False,
    )
    require(bad_file.returncode == EXPECTED_USAGE_EXIT, "custom path traversal should fail")
    require("custom filename" in bad_file.stderr, "custom path traversal error should name the filename")

    bad_size = run_tool(
        root,
        "download",
        "--url",
        CUSTOM_MODEL_URL,
        "--file",
        CUSTOM_MODEL_FILE,
        "--min-bytes",
        str(INVALID_CUSTOM_MIN_BYTES),
        "--dry-run",
        check=False,
    )
    require(bad_size.returncode == EXPECTED_USAGE_EXIT, "custom non-positive min size should fail")
    require("custom min-bytes" in bad_size.stderr, "custom min size error should name the field")

    bad_sha = run_tool(
        root,
        "download",
        "--url",
        CUSTOM_MODEL_URL,
        "--file",
        CUSTOM_MODEL_FILE,
        "--min-bytes",
        str(VALID_CUSTOM_MIN_BYTES),
        "--sha256",
        CUSTOM_BAD_SHA256,
        "--dry-run",
        check=False,
    )
    require(bad_sha.returncode == EXPECTED_USAGE_EXIT, "custom short checksum should fail")
    require("custom download" in bad_sha.stderr, "custom checksum error should name the download")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
