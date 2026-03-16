#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import struct
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
UNREAL_MATRIX_PRESET = "gemma3-270m-q4km"
UNREAL_MATRIX_FILE = "gemma-3-270m-q4_k_m.gguf"
TINY_PRESET = "tiny-embed"
TINY_LABEL = "Tiny synthetic embedding GGUF"
TINY_REPO = "local/tiny"
TINY_FILE = "tiny-embed.gguf"
TINY_LICENSE = "Synthetic smoke fixture."
TINY_REVISION = "main"
TINY_ARCH = "qwen3"
TINY_CONTEXT = 40960
TINY_EMBEDDING = 1024
TINY_POOLING = 1
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TENSOR_COUNT = 0
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8


def gguf_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def gguf_uint32(key: str, value: int) -> bytes:
    return gguf_string(key) + struct.pack("<II", GGUF_TYPE_UINT32, value)


def gguf_text(key: str, value: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(value)


def write_tiny_gguf(path: Path, include_pooling: bool = True) -> tuple[int, str]:
    entries = [
        gguf_text("general.architecture", TINY_ARCH),
        gguf_uint32(f"{TINY_ARCH}.context_length", TINY_CONTEXT),
        gguf_uint32(f"{TINY_ARCH}.embedding_length", TINY_EMBEDDING),
    ]
    if include_pooling:
        entries.append(gguf_uint32(f"{TINY_ARCH}.pooling_type", TINY_POOLING))
    data = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, GGUF_TENSOR_COUNT, len(entries)) + b"".join(entries)
    path.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def write_manifest(path: Path, size_bytes: int, sha256: str, embedding_dimension: int = TINY_EMBEDDING) -> None:
    payload = {
        "version": 1,
        "download_dir": "tests/models",
        "default_preset": TINY_PRESET,
        "huggingface_base_url": "https://huggingface.co",
        "presets": [
            {
                "name": TINY_PRESET,
                "label": TINY_LABEL,
                "model_type": MODEL_TYPE_EMBEDDING,
                "repo": TINY_REPO,
                "filename": TINY_FILE,
                "revision": TINY_REVISION,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "license_note": TINY_LICENSE,
                "context_length": TINY_CONTEXT,
                "embedding_dimension": embedding_dimension,
                "include_in_package": False,
                "include_in_unreal_sample_matrix": False,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def run_downloader(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [str(root / "tests" / "model_downloader.sh"), *args]
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
    embedding_summary = json.loads(
        run_tool(root, "status-summary", "--type", MODEL_TYPE_EMBEDDING, "--dir", CUSTOM_OUTPUT_DIR).stdout
    )
    require(embedding_summary["total"] == len(embedding_status), "embedding summary total mismatched rows")
    require(
        embedding_summary["not_ready"]
        == embedding_summary["missing"] + embedding_summary["partial"] + embedding_summary["invalid"],
        "embedding summary not-ready count mismatched",
    )
    require(embedding_summary["expected_bytes"] > 0, "embedding summary missed expected bytes")
    missing_embedding_status = json.loads(
        run_tool(
            root,
            "status-all",
            "--type",
            MODEL_TYPE_EMBEDDING,
            "--dir",
            CUSTOM_OUTPUT_DIR,
            "--only",
            STATUS_MISSING,
        ).stdout
    )
    require(
        any(row["name"] == QWEN_EMBED_PRESET for row in missing_embedding_status),
        "missing embedding status missed Qwen embed preset",
    )
    require(all(row["status"] == STATUS_MISSING for row in missing_embedding_status), "missing filter mixed statuses")

    text_presets = run_tool(root, "list", "--type", MODEL_TYPE_TEXT).stdout
    require(QWEN_TEXT_PRESET in text_presets, "text list missed Qwen text preset")
    require(QWEN_EMBED_PRESET not in text_presets, "text list included embedding preset")

    package_presets = json.loads(run_tool(root, "list", "--package", "--format", "json").stdout)
    require(any(row["name"] == PACKAGE_PRESET for row in package_presets), "package list missed default text preset")
    require(all(row["include_in_package"] is True for row in package_presets), "package list included disabled preset")

    package_status = run_tool(root, "status-all", "--package", "--format", "text", "--dir", CUSTOM_OUTPUT_DIR).stdout
    require(PACKAGE_PRESET in package_status, "package status missed default text preset")
    require(QWEN_TEXT_PRESET not in package_status, "package status included disabled preset")
    package_summary = json.loads(
        run_tool(root, "status-summary", "--package", "--format", "json", "--dir", CUSTOM_OUTPUT_DIR).stdout
    )
    require(package_summary["total"] == len(package_presets), "package summary total mismatched preset count")

    unreal_matrix_presets = json.loads(run_tool(root, "list", "--unreal-matrix", "--format", "json").stdout)
    require(
        any(row["name"] == UNREAL_MATRIX_PRESET for row in unreal_matrix_presets),
        "Unreal matrix list missed expected text preset",
    )
    require(
        all(row["include_in_unreal_sample_matrix"] is True for row in unreal_matrix_presets),
        "Unreal matrix list included disabled preset",
    )
    require(
        all(row["model_type"] == MODEL_TYPE_TEXT for row in unreal_matrix_presets),
        "Unreal matrix list included non-text preset",
    )
    unreal_matrix_status = run_tool(
        root,
        "status-all",
        "--unreal-matrix",
        "--format",
        "text",
        "--dir",
        CUSTOM_OUTPUT_DIR,
    ).stdout
    require(UNREAL_MATRIX_PRESET in unreal_matrix_status, "Unreal matrix status missed expected preset")
    require(PACKAGE_PRESET not in unreal_matrix_status, "Unreal matrix status included package-only preset")
    wrapper_unreal_matrix = run_downloader(
        root,
        "--list-unreal-matrix",
        "--list-format",
        "json",
    ).stdout
    require(UNREAL_MATRIX_PRESET in wrapper_unreal_matrix, "wrapper Unreal matrix list missed expected preset")
    wrapper_unreal_matrix_status = run_downloader(
        root,
        "--status-all",
        "--list-unreal-matrix",
        "--status-format",
        "text",
        "--dir",
        CUSTOM_OUTPUT_DIR,
    ).stdout
    require(UNREAL_MATRIX_PRESET in wrapper_unreal_matrix_status, "wrapper Unreal matrix status missed expected preset")
    require(PACKAGE_PRESET not in wrapper_unreal_matrix_status, "wrapper Unreal matrix status included package-only preset")
    preset_for_file = run_tool(root, "preset-for-file", UNREAL_MATRIX_FILE).stdout.strip()
    require(preset_for_file == UNREAL_MATRIX_PRESET, "filename lookup did not resolve Unreal matrix preset")

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

        not_ready = json.loads(
            run_tool(root, "status-all", "--type", MODEL_TYPE_TEXT, "--dir", temp_dir, "--only", "not-ready").stdout
        )
        require(any(row["name"] == QWEN_TEXT_PRESET for row in not_ready), "not-ready status missed invalid file")
        require(all(row["status"] != STATUS_READY for row in not_ready), "not-ready filter included ready status")
        text_summary = json.loads(run_tool(root, "status-summary", "--type", MODEL_TYPE_TEXT, "--dir", temp_dir).stdout)
        require(text_summary["invalid"] >= 1, "text summary missed invalid file")
        require(text_summary["partial_bytes"] == len(b"partial"), "text summary missed partial byte count")
        require(
            text_summary["not_ready"] == text_summary["missing"] + text_summary["partial"] + text_summary["invalid"],
            "text summary not-ready count mismatched",
        )

        wrapper_not_ready = run_downloader(
            root,
            "--status-all",
            "--list-type",
            MODEL_TYPE_TEXT,
            "--status-only",
            "not-ready",
            "--status-format",
            "json",
            "--dir",
            temp_dir,
        ).stdout
        require(QWEN_TEXT_PRESET in wrapper_not_ready, "wrapper not-ready status missed invalid file")
        wrapper_summary = run_downloader(
            root,
            "--status-summary",
            "--list-type",
            MODEL_TYPE_TEXT,
            "--status-format",
            "text",
            "--dir",
            temp_dir,
        ).stdout
        require("not_ready:" in wrapper_summary, "wrapper summary text missed not-ready count")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        model_path = temp_path / TINY_FILE
        manifest_path = temp_path / "manifest.json"
        size_bytes, sha256 = write_tiny_gguf(model_path)
        write_manifest(manifest_path, size_bytes, sha256)

        metadata = json.loads(
            run_tool(
                root,
                "--manifest",
                str(manifest_path),
                "inspect",
                TINY_PRESET,
                "--dir",
                temp_dir,
                "--validate",
            ).stdout
        )
        require(metadata["architecture"] == TINY_ARCH, "metadata reader missed architecture")
        require(metadata["context_length"] == TINY_CONTEXT, "metadata reader missed context length")
        require(metadata["embedding_dimension"] == TINY_EMBEDDING, "metadata reader missed embedding dimension")
        require(metadata["supports_embeddings"] is True, "metadata reader missed embedding pooling")

        valid_file = run_tool(
            root,
            "--manifest",
            str(manifest_path),
            "validate-file",
            "--preset",
            TINY_PRESET,
            "--dir",
            temp_dir,
            "--validate-metadata",
        ).stdout
        require("valid:" in valid_file, "metadata file validation did not report success")

        bad_manifest_path = temp_path / "bad-manifest.json"
        write_manifest(bad_manifest_path, size_bytes, sha256, embedding_dimension=EXPECTED_EMBED_DIMENSION + 1)
        bad_metadata = run_tool(
            root,
            "--manifest",
            str(bad_manifest_path),
            "validate-file",
            "--preset",
            TINY_PRESET,
            "--dir",
            temp_dir,
            "--validate-metadata",
            check=False,
        )
        require(bad_metadata.returncode == EXPECTED_USAGE_EXIT, "metadata mismatch should fail")
        require("embedding_dimension" in bad_metadata.stderr, "metadata mismatch should name embedding dimension")

        no_pool_path = temp_path / "no-pool.gguf"
        no_pool_size, no_pool_sha = write_tiny_gguf(no_pool_path, include_pooling=False)
        no_pool_manifest = temp_path / "no-pool-manifest.json"
        write_manifest(no_pool_manifest, no_pool_size, no_pool_sha)
        no_pool = run_tool(
            root,
            "--manifest",
            str(no_pool_manifest),
            "validate-file",
            "--preset",
            TINY_PRESET,
            "--path",
            str(no_pool_path),
            "--validate-metadata",
            check=False,
        )
        require(no_pool.returncode == EXPECTED_USAGE_EXIT, "embedding metadata without pooling should fail")
        require("embedding pooling" in no_pool.stderr, "embedding metadata error should name pooling")

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
