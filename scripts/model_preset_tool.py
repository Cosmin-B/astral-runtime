#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "model_presets.json"
MANIFEST_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_BASE_SECONDS = 2
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
PROGRESS_STEP_BYTES = 64 * 1024 * 1024
SHA256_HEX_LENGTH = 64
PERCENT_SCALE = 100
HTTP_STATUS_OK = 200
EXIT_OK = 0
EXIT_USAGE = 2
MIN_CUSTOM_MODEL_BYTES = 1
ENV_TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
CUSTOM_PRESET_NAME = "custom"
CUSTOM_PRESET_LABEL = "Custom GGUF model"
CUSTOM_MODEL_TYPE = "custom"
DEFAULT_REVISION = "main"
GGUF_SUFFIX = ".gguf"
MODEL_TYPE_TEXT = "text"
MODEL_TYPE_EMBEDDING = "embedding"
MODEL_TYPE_CUSTOM = CUSTOM_MODEL_TYPE
KNOWN_MODEL_TYPES = (MODEL_TYPE_TEXT, MODEL_TYPE_EMBEDDING, MODEL_TYPE_CUSTOM)
LIST_FORMAT_TEXT = "text"
LIST_FORMAT_JSON = "json"
MODEL_TYPE_ALL = "all"
SHA256_HEX_DIGITS = frozenset("0123456789abcdef")
MODEL_STATUS_READY = "ready"
MODEL_STATUS_MISSING = "missing"
MODEL_STATUS_PARTIAL = "partial"
MODEL_STATUS_INVALID = "invalid"
MODEL_STATUS_NOT_READY = "not-ready"
MODEL_STATUS_ANY = "any"
MODEL_STATUS_FILTERS = (
    MODEL_STATUS_ANY,
    MODEL_STATUS_READY,
    MODEL_STATUS_MISSING,
    MODEL_STATUS_PARTIAL,
    MODEL_STATUS_INVALID,
    MODEL_STATUS_NOT_READY,
)
PARTIAL_DOWNLOAD_SUFFIX = ".part"
GGUF_MAGIC = b"GGUF"
GGUF_HEADER_BYTES = 24
GGUF_UINT8 = 0
GGUF_INT8 = 1
GGUF_UINT16 = 2
GGUF_INT16 = 3
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64 = 10
GGUF_INT64 = 11
GGUF_FLOAT64 = 12
GGUF_MAX_METADATA_ENTRIES = 100000
GGUF_MAX_STRING_BYTES = 64 * 1024 * 1024
GGUF_MAX_ARRAY_ITEMS = 1024 * 1024
GGUF_CONTEXT_SUFFIX = ".context_length"
GGUF_EMBEDDING_SUFFIX = ".embedding_length"
GGUF_POOLING_SUFFIX = ".pooling_type"

GGUF_SCALAR_FORMATS = {
    GGUF_UINT8: "B",
    GGUF_INT8: "b",
    GGUF_UINT16: "H",
    GGUF_INT16: "h",
    GGUF_UINT32: "I",
    GGUF_INT32: "i",
    GGUF_FLOAT32: "f",
    GGUF_BOOL: "?",
    GGUF_UINT64: "Q",
    GGUF_INT64: "q",
    GGUF_FLOAT64: "d",
}


@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    model_type: str
    repo: str
    filename: str
    revision: str
    size_bytes: int
    sha256: str
    license_note: str
    context_length: Optional[int]
    embedding_dimension: Optional[int]
    include_in_package: bool
    include_in_unreal_sample_matrix: bool
    huggingface_base_url: str
    direct_url: str = ""

    @property
    def url(self) -> str:
        if self.direct_url:
            return self.direct_url
        repo = urllib.parse.quote(self.repo, safe="/")
        revision = urllib.parse.quote(self.revision, safe="")
        filename = urllib.parse.quote(self.filename, safe="/")
        return f"{self.huggingface_base_url}/{repo}/resolve/{revision}/{filename}"


def _repo_root_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def _read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def _require_str(row: Dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"preset {row.get('name', '<unknown>')} has invalid {key}")
    return value


def _require_int(row: Dict[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"preset {row.get('name', '<unknown>')} has invalid {key}")
    return value


def _require_sha256(value: str, owner: str) -> None:
    if len(value) != SHA256_HEX_LENGTH:
        raise ValueError(f"{owner} has invalid sha256 length")
    if any(ch not in SHA256_HEX_DIGITS for ch in value):
        raise ValueError(f"{owner} has non-lowercase sha256")


def _require_positive_size(value: int, owner: str) -> None:
    if value < MIN_CUSTOM_MODEL_BYTES:
        raise ValueError(f"{owner} must be positive")


def _require_gguf_filename(filename: str, owner: str) -> None:
    if Path(filename).name != filename:
        raise ValueError(f"{owner} must be a basename")
    if not filename.endswith(GGUF_SUFFIX):
        raise ValueError(f"{owner} must end with {GGUF_SUFFIX}")


def _optional_int(row: Dict[str, Any], key: str) -> Optional[int]:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"preset {row.get('name', '<unknown>')} has invalid {key}")
    return value


def _load_presets(manifest_path: Path) -> List[Preset]:
    data = _read_manifest(manifest_path)
    base_url = data.get("huggingface_base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("manifest has invalid huggingface_base_url")

    rows = data.get("presets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest has no presets")

    out: List[Preset] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("preset rows must be objects")
        name = _require_str(row, "name")
        if name in seen:
            raise ValueError(f"duplicate preset name: {name}")
        seen.add(name)
        sha256 = _require_str(row, "sha256")
        _require_sha256(sha256, f"preset {name}")
        out.append(
            Preset(
                name=name,
                label=_require_str(row, "label"),
                model_type=_require_str(row, "model_type"),
                repo=_require_str(row, "repo"),
                filename=_require_str(row, "filename"),
                revision=_require_str(row, "revision"),
                size_bytes=_require_int(row, "size_bytes"),
                sha256=sha256,
                license_note=_require_str(row, "license_note"),
                context_length=_optional_int(row, "context_length"),
                embedding_dimension=_optional_int(row, "embedding_dimension"),
                include_in_package=bool(row.get("include_in_package", False)),
                include_in_unreal_sample_matrix=bool(row.get("include_in_unreal_sample_matrix", False)),
                huggingface_base_url=base_url.rstrip("/"),
            )
        )
    return out


def _validate_manifest(data: Dict[str, Any], presets: List[Preset]) -> None:
    version = data.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(f"manifest has unsupported version: {version}")

    download_dir = data.get("download_dir")
    if not isinstance(download_dir, str) or not download_dir:
        raise ValueError("manifest has invalid download_dir")

    default_name = data.get("default_preset")
    if not isinstance(default_name, str) or not default_name:
        raise ValueError("manifest has invalid default_preset")
    preset_names = {preset.name for preset in presets}
    if default_name not in preset_names:
        raise ValueError(f"default preset is not defined: {default_name}")

    filenames = set()
    for preset in presets:
        if preset.model_type not in KNOWN_MODEL_TYPES:
            raise ValueError(f"preset {preset.name} has unsupported model_type: {preset.model_type}")
        _require_gguf_filename(preset.filename, f"preset {preset.name} filename")
        if preset.filename in filenames:
            raise ValueError(f"duplicate preset filename: {preset.filename}")
        filenames.add(preset.filename)
        if preset.model_type == MODEL_TYPE_EMBEDDING and preset.embedding_dimension is None:
            raise ValueError(f"embedding preset {preset.name} is missing embedding_dimension")
        if preset.model_type == MODEL_TYPE_TEXT and preset.embedding_dimension is not None:
            raise ValueError(f"text preset {preset.name} should not set embedding_dimension")
        row = next(item for item in data["presets"] if item["name"] == preset.name)
        if not isinstance(row.get("include_in_package"), bool):
            raise ValueError(f"preset {preset.name} has invalid include_in_package")
        if not isinstance(row.get("include_in_unreal_sample_matrix"), bool):
            raise ValueError(f"preset {preset.name} has invalid include_in_unreal_sample_matrix")


def _default_preset_name(manifest_path: Path) -> str:
    data = _read_manifest(manifest_path)
    value = data.get("default_preset")
    if not isinstance(value, str) or not value:
        raise ValueError("manifest has invalid default_preset")
    return value


def _find_preset(presets: Iterable[Preset], name: str) -> Preset:
    for preset in presets:
        if preset.name == name:
            return preset
    raise KeyError(name)


def _preset_for_filename(presets: Iterable[Preset], filename: str) -> Optional[Preset]:
    basename = Path(filename).name
    for preset in presets:
        if preset.filename == basename:
            return preset
    return None


def _format_bytes(size: int) -> str:
    mib = 1024 * 1024
    gib = mib * 1024
    if size >= gib:
        return f"{size / gib:.2f} GiB"
    return f"{size / mib:.1f} MiB"


def _token_from_env() -> str:
    for name in ENV_TOKEN_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class _GGUFReader:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle

    def _take(self, size: int) -> bytes:
        out = self._handle.read(size)
        if len(out) != size:
            raise ValueError("truncated GGUF metadata")
        return out

    def _unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack_from("<" + fmt, self._take(size))[0]

    def string(self) -> str:
        size = int(self._unpack("Q"))
        if size > GGUF_MAX_STRING_BYTES:
            raise ValueError(f"GGUF string is too large: {size}")
        return self._take(size).decode("utf-8")

    def value(self) -> Any:
        value_type = int(self._unpack("I"))
        return self.typed_value(value_type)

    def typed_value(self, value_type: int) -> Any:
        if value_type == GGUF_STRING:
            return self.string()
        if value_type == GGUF_ARRAY:
            item_type = int(self._unpack("I"))
            item_count = int(self._unpack("Q"))
            if item_count > GGUF_MAX_ARRAY_ITEMS:
                raise ValueError(f"GGUF array is too large: {item_count}")
            return [self.typed_value(item_type) for _ in range(item_count)]
        fmt = GGUF_SCALAR_FORMATS.get(value_type)
        if fmt is None:
            raise ValueError(f"unsupported GGUF metadata value type: {value_type}")
        return self._unpack(fmt)


def _read_gguf_metadata(path: Path) -> Dict[str, Any]:
    if path.stat().st_size < GGUF_HEADER_BYTES:
        raise ValueError(f"{path.name} is too small for a GGUF header")
    with path.open("rb") as handle:
        reader = _GGUFReader(handle)
        magic = reader._take(len(GGUF_MAGIC))
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path.name} is not a GGUF file")
        reader._unpack("I")
        reader._unpack("Q")
        metadata_count = int(reader._unpack("Q"))
        if metadata_count > GGUF_MAX_METADATA_ENTRIES:
            raise ValueError(f"{path.name} has too many GGUF metadata entries: {metadata_count}")

        metadata: Dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.string()
            metadata[key] = reader.value()
    return metadata


def _first_int_metadata(metadata: Dict[str, Any], suffix: str) -> Optional[int]:
    for key, value in metadata.items():
        if key.endswith(suffix) and type(value) is int:
            return int(value)
    return None


def _metadata_record(path: Path) -> Dict[str, Any]:
    metadata = _read_gguf_metadata(path)
    context_length = _first_int_metadata(metadata, GGUF_CONTEXT_SUFFIX)
    embedding_dimension = _first_int_metadata(metadata, GGUF_EMBEDDING_SUFFIX)
    return {
        "path": str(path),
        "format": "GGUF",
        "metadata_entries": len(metadata),
        "architecture": metadata.get("general.architecture", ""),
        "context_length": context_length,
        "embedding_dimension": embedding_dimension,
        "supports_embeddings": any(key.endswith(GGUF_POOLING_SUFFIX) for key in metadata),
    }


def _validate_gguf_metadata(path: Path, preset: Preset) -> Dict[str, Any]:
    record = _metadata_record(path)
    if preset.context_length is not None and record["context_length"] != preset.context_length:
        raise ValueError(
            f"{path.name} context_length mismatch: got {record['context_length']}, expected {preset.context_length}"
        )
    if preset.embedding_dimension is not None and record["embedding_dimension"] != preset.embedding_dimension:
        raise ValueError(
            f"{path.name} embedding_dimension mismatch: got {record['embedding_dimension']}, "
            f"expected {preset.embedding_dimension}"
        )
    if preset.model_type == MODEL_TYPE_EMBEDDING and not record["supports_embeddings"]:
        raise ValueError(f"{path.name} metadata does not advertise embedding pooling")
    return record


def _verify_existing(path: Path, preset: Preset) -> None:
    if not path.exists():
        raise FileNotFoundError(f"model file is missing: {path}")
    size = path.stat().st_size
    if preset.sha256:
        if size != preset.size_bytes:
            raise ValueError(f"{path.name} size mismatch: got {size}, expected {preset.size_bytes}")
        sha256 = _file_sha256(path)
        if sha256 != preset.sha256:
            raise ValueError(f"{path.name} sha256 mismatch: got {sha256}, expected {preset.sha256}")
    elif size < preset.size_bytes:
        raise ValueError(f"{path.name} size mismatch: got {size}, expected at least {preset.size_bytes}")


def _resolved_output_path(args: argparse.Namespace, preset: Preset) -> Path:
    return _repo_root_path(args.dir) / preset.filename


def _format_download_command(preset: Preset, output_dir: Path) -> str:
    script = str(Path("tests") / "model_downloader.sh")
    output = str(output_dir)
    if preset.name == CUSTOM_PRESET_NAME:
        parts = [script, "--url", preset.url, "--file", preset.filename, "--dir", output]
    else:
        parts = [script, "--preset", preset.name, "--dir", output]
    return " ".join(shlex.quote(part) for part in parts)


def _preset_record(preset: Preset, output_dir: Path) -> Dict[str, Any]:
    output_path = output_dir / preset.filename
    return {
        "name": preset.name,
        "label": preset.label,
        "model_type": preset.model_type,
        "repo": preset.repo,
        "filename": preset.filename,
        "revision": preset.revision,
        "url": preset.url,
        "path": str(output_path),
        "size_bytes": preset.size_bytes,
        "sha256": preset.sha256,
        "license_note": preset.license_note,
        "context_length": preset.context_length,
        "embedding_dimension": preset.embedding_dimension,
        "include_in_package": preset.include_in_package,
        "include_in_unreal_sample_matrix": preset.include_in_unreal_sample_matrix,
        "download_command": _format_download_command(preset, output_dir),
    }


def _status_record(preset: Preset, output_dir: Path) -> Dict[str, Any]:
    output_path = output_dir / preset.filename
    part_path = output_path.with_name(output_path.name + PARTIAL_DOWNLOAD_SUFFIX)
    record = _preset_record(preset, output_dir)
    record["part_path"] = str(part_path)
    record["present_bytes"] = 0
    record["partial_bytes"] = part_path.stat().st_size if part_path.exists() else 0
    record["expected_bytes"] = preset.size_bytes
    record["sha256_ok"] = False
    record["status"] = MODEL_STATUS_MISSING
    record["error"] = ""

    if output_path.exists():
        record["present_bytes"] = output_path.stat().st_size
        try:
            _verify_existing(output_path, preset)
            record["sha256_ok"] = True
            record["status"] = MODEL_STATUS_READY
        except Exception as exc:
            record["status"] = MODEL_STATUS_INVALID
            record["error"] = str(exc)
        return record

    if part_path.exists():
        record["status"] = MODEL_STATUS_PARTIAL
    return record


def _request(url: str, token: str, start: int) -> urllib.request.Request:
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "astral-model-downloader/1.0")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if start:
        request.add_header("Range", f"bytes={start}-")
    return request


def _download(preset: Preset, out_path: Path, token: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".part")
    last_percent = -1
    last_bytes = 0

    for attempt in range(DEFAULT_RETRY_COUNT):
        start = tmp_path.stat().st_size if tmp_path.exists() else 0
        if start > preset.size_bytes:
            tmp_path.unlink(missing_ok=True)
            start = 0

        try:
            with urllib.request.urlopen(_request(preset.url, token, start), timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                if start and getattr(response, "status", HTTP_STATUS_OK) == HTTP_STATUS_OK:
                    tmp_path.unlink(missing_ok=True)
                    start = 0
                mode = "ab" if start else "wb"
                downloaded = start
                with tmp_path.open(mode) as handle:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded * PERCENT_SCALE) // preset.size_bytes
                        if percent != last_percent or downloaded - last_bytes >= PROGRESS_STEP_BYTES:
                            sys.stderr.write(
                                f"\r  {preset.filename}: {percent}% ({_format_bytes(downloaded)}/{_format_bytes(preset.size_bytes)})"
                            )
                            sys.stderr.flush()
                            last_percent = percent
                            last_bytes = downloaded
                sys.stderr.write("\n")
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            if attempt + 1 >= DEFAULT_RETRY_COUNT:
                raise RuntimeError(f"download failed for {preset.name}: {exc}") from exc
            sleep_seconds = DEFAULT_RETRY_BASE_SECONDS ** attempt
            sys.stderr.write(f"\n[model] transient download error; retrying in {sleep_seconds}s\n")
            time.sleep(sleep_seconds)
            continue

        if not tmp_path.exists():
            raise FileNotFoundError(f"download did not produce {tmp_path}")
        final_size = tmp_path.stat().st_size
        size_ok = final_size == preset.size_bytes if preset.sha256 else final_size >= preset.size_bytes
        if size_ok:
            sha256 = _file_sha256(tmp_path)
            if preset.sha256 and sha256 != preset.sha256:
                tmp_path.unlink(missing_ok=True)
                raise ValueError(f"{preset.filename} sha256 mismatch: got {sha256}, expected {preset.sha256}")
            tmp_path.replace(out_path)
            return

    raise RuntimeError(f"incomplete download for {preset.name}")


def _print_preset(preset: Preset, path: Path) -> None:
    print(f"preset: {preset.name}")
    print(f"label: {preset.label}")
    print(f"type: {preset.model_type}")
    print(f"path: {path}")
    print(f"url: {preset.url}")
    print(f"size_bytes: {preset.size_bytes}")
    print(f"sha256: {preset.sha256 or 'not pinned'}")
    print(f"context_length: {preset.context_length if preset.context_length is not None else 'unknown'}")
    print(f"embedding_dimension: {preset.embedding_dimension if preset.embedding_dimension is not None else 'n/a'}")


def cmd_list(args: argparse.Namespace) -> int:
    presets = _load_presets(Path(args.manifest))
    selected = _select_presets(args, presets)
    if args.format == LIST_FORMAT_JSON:
        output_dir = _repo_root_path(args.dir)
        json.dump([_preset_record(preset, output_dir) for preset in selected], sys.stdout, indent=2, sort_keys=True)
        print()
        return EXIT_OK
    for preset in selected:
        print(f"{preset.name}\t{preset.filename}\t{preset.model_type}\t{preset.label}")
    return EXIT_OK


def cmd_filename(args: argparse.Namespace) -> int:
    preset = _find_preset(_load_presets(Path(args.manifest)), args.preset)
    print(preset.filename)
    return EXIT_OK


def cmd_preset_for_file(args: argparse.Namespace) -> int:
    preset = _preset_for_filename(_load_presets(Path(args.manifest)), args.filename)
    if preset is None:
        return EXIT_USAGE
    print(preset.name)
    return EXIT_OK


def _select_download_preset(args: argparse.Namespace, presets: List[Preset], manifest_path: Path) -> Preset:
    if args.url or args.hf_repo or args.hf_file or args.file:
        if not args.file:
            raise ValueError("custom downloads require --file")
        _require_gguf_filename(args.file, "custom filename")
        _require_positive_size(args.min_bytes, "custom min-bytes")
        if args.sha256:
            _require_sha256(args.sha256, "custom download")
        if args.url:
            url = args.url
            repo = ""
            filename = args.file
            revision = args.hf_revision or DEFAULT_REVISION
        elif args.hf_repo and args.hf_file:
            repo = args.hf_repo
            filename = args.hf_file
            revision = args.hf_revision or DEFAULT_REVISION
            repo_q = urllib.parse.quote(repo, safe="/")
            rev_q = urllib.parse.quote(revision, safe="")
            file_q = urllib.parse.quote(filename, safe="/")
            url = f"https://huggingface.co/{repo_q}/resolve/{rev_q}/{file_q}"
        else:
            raise ValueError("custom Hugging Face downloads require --hf-repo and --hf-file")
        return Preset(
            name=CUSTOM_PRESET_NAME,
            label=CUSTOM_PRESET_LABEL,
            model_type=CUSTOM_MODEL_TYPE,
            repo=repo,
            filename=args.file,
            revision=revision,
            size_bytes=args.min_bytes,
            sha256=args.sha256,
            license_note="Custom model; verify license terms before redistribution.",
            context_length=None,
            embedding_dimension=None,
            include_in_package=False,
            include_in_unreal_sample_matrix=False,
            huggingface_base_url="https://huggingface.co",
            direct_url=url,
        )

    preset_name = args.preset or _default_preset_name(manifest_path)
    try:
        return _find_preset(presets, preset_name)
    except KeyError as exc:
        raise ValueError(f"Unknown preset: {preset_name}") from exc


def cmd_path(args: argparse.Namespace) -> int:
    preset = _find_preset(_load_presets(Path(args.manifest)), args.preset)
    print(_resolved_output_path(args, preset))
    return EXIT_OK


def cmd_info(args: argparse.Namespace) -> int:
    preset = _find_preset(_load_presets(Path(args.manifest)), args.preset)
    output_dir = _repo_root_path(args.dir)
    record = _preset_record(preset, output_dir)
    if args.format == "json":
        json.dump(record, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print_preset(preset, output_dir / preset.filename)
        print(f"command: {record['download_command']}")
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    preset = _find_preset(_load_presets(Path(args.manifest)), args.preset)
    output_dir = _repo_root_path(args.dir)
    record = _status_record(preset, output_dir)
    if args.format == "json":
        json.dump(record, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(f"preset: {preset.name}")
        print(f"status: {record['status']}")
        print(f"path: {record['path']}")
        print(f"present_bytes: {record['present_bytes']}")
        print(f"partial_bytes: {record['partial_bytes']}")
        print(f"expected_bytes: {record['expected_bytes']}")
        if record["error"]:
            print(f"error: {record['error']}")
        print(f"command: {record['download_command']}")
    return EXIT_OK


def _select_presets(args: argparse.Namespace, presets: List[Preset]) -> List[Preset]:
    selected = presets
    if args.unreal_matrix:
        selected = [preset for preset in selected if preset.include_in_unreal_sample_matrix]
    if args.package:
        selected = [preset for preset in selected if preset.include_in_package]
    if args.type != MODEL_TYPE_ALL:
        selected = [preset for preset in selected if preset.model_type == args.type]
    return selected


def cmd_status_all(args: argparse.Namespace) -> int:
    output_dir = _repo_root_path(args.dir)
    presets = _select_presets(args, _load_presets(Path(args.manifest)))
    records = [_status_record(preset, output_dir) for preset in presets]
    if args.only == MODEL_STATUS_NOT_READY:
        records = [record for record in records if record["status"] != MODEL_STATUS_READY]
    elif args.only != MODEL_STATUS_ANY:
        records = [record for record in records if record["status"] == args.only]
    if args.format == "json":
        json.dump(records, sys.stdout, indent=2, sort_keys=True)
        print()
        return EXIT_OK

    for record in records:
        print(
            f"{record['name']}\t{record['status']}\t"
            f"{record['present_bytes']}/{record['expected_bytes']}\t{record['path']}"
        )
    return EXIT_OK


def _status_summary_record(records: List[Dict[str, object]]) -> Dict[str, int]:
    ready = 0
    missing = 0
    partial = 0
    invalid = 0
    expected_bytes = 0
    present_bytes = 0
    partial_bytes = 0

    for record in records:
        status = record["status"]
        if status == MODEL_STATUS_READY:
            ready += 1
        elif status == MODEL_STATUS_MISSING:
            missing += 1
        elif status == MODEL_STATUS_PARTIAL:
            partial += 1
        elif status == MODEL_STATUS_INVALID:
            invalid += 1
        expected_bytes += int(record["expected_bytes"])
        present_bytes += int(record["present_bytes"])
        partial_bytes += int(record["partial_bytes"])

    return {
        "total": len(records),
        "ready": ready,
        "missing": missing,
        "partial": partial,
        "invalid": invalid,
        "not_ready": missing + partial + invalid,
        "expected_bytes": expected_bytes,
        "present_bytes": present_bytes,
        "partial_bytes": partial_bytes,
    }


def cmd_status_summary(args: argparse.Namespace) -> int:
    output_dir = _repo_root_path(args.dir)
    presets = _select_presets(args, _load_presets(Path(args.manifest)))
    summary = _status_summary_record([_status_record(preset, output_dir) for preset in presets])
    if args.format == "json":
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()
        return EXIT_OK

    for key, value in summary.items():
        print(f"{key}: {value}")
    return EXIT_OK


def cmd_download_plan(args: argparse.Namespace) -> int:
    output_dir = _repo_root_path(args.dir)
    presets = _select_presets(args, _load_presets(Path(args.manifest)))
    records = [_status_record(preset, output_dir) for preset in presets]
    records = [record for record in records if record["status"] != MODEL_STATUS_READY]
    if args.format == "json":
        json.dump(records, sys.stdout, indent=2, sort_keys=True)
        print()
        return EXIT_OK

    for record in records:
        print(record["download_command"])
    return EXIT_OK


def cmd_validate_file(args: argparse.Namespace) -> int:
    preset = _find_preset(_load_presets(Path(args.manifest)), args.preset)
    output_path = Path(args.path).expanduser() if args.path else _resolved_output_path(args, preset)
    _verify_existing(output_path, preset)
    if args.validate_metadata:
        _validate_gguf_metadata(output_path, preset)
    print(f"valid: {output_path}")
    return EXIT_OK


def cmd_inspect(args: argparse.Namespace) -> int:
    preset = _find_preset(_load_presets(Path(args.manifest)), args.preset)
    output_path = Path(args.path).expanduser() if args.path else _resolved_output_path(args, preset)
    record = _validate_gguf_metadata(output_path, preset) if args.validate else _metadata_record(output_path)
    record["preset"] = preset.name
    record["model_type"] = preset.model_type
    if args.format == "json":
        json.dump(record, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(f"preset: {preset.name}")
        print(f"path: {output_path}")
        print(f"architecture: {record['architecture'] or 'unknown'}")
        print(f"context_length: {record['context_length'] if record['context_length'] is not None else 'unknown'}")
        print(
            f"embedding_dimension: "
            f"{record['embedding_dimension'] if record['embedding_dimension'] is not None else 'n/a'}"
        )
        print(f"supports_embeddings: {str(record['supports_embeddings']).lower()}")
        print(f"metadata_entries: {record['metadata_entries']}")
    return EXIT_OK


def cmd_download(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    presets = _load_presets(manifest_path)
    preset = _select_download_preset(args, presets, manifest_path)

    output_dir = _repo_root_path(args.dir)
    output_path = output_dir / preset.filename
    _print_preset(preset, output_path)
    print(f"command: {_format_download_command(preset, output_dir)}")

    if args.dry_run:
        return EXIT_OK

    if args.validate_only:
        _verify_existing(output_path, preset)
        if args.validate_metadata:
            _validate_gguf_metadata(output_path, preset)
        print(f"valid: {output_path}")
        return EXIT_OK

    if output_path.exists():
        _verify_existing(output_path, preset)
        if args.validate_metadata:
            _validate_gguf_metadata(output_path, preset)
        print(f"ready: {output_path}")
        return EXIT_OK

    _download(preset, output_path, args.token or _token_from_env())
    _verify_existing(output_path, preset)
    if args.validate_metadata:
        _validate_gguf_metadata(output_path, preset)
    print(f"downloaded: {output_path}")
    return EXIT_OK


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    data = _read_manifest(manifest_path)
    presets = _load_presets(manifest_path)
    _validate_manifest(data, presets)
    if not presets:
        raise ValueError("no presets")
    print(f"manifest OK: {args.manifest} presets={len(presets)}")
    return EXIT_OK


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--unreal-matrix", action="store_true")
    list_parser.add_argument("--package", action="store_true")
    list_parser.add_argument("--type", choices=(MODEL_TYPE_ALL, MODEL_TYPE_TEXT, MODEL_TYPE_EMBEDDING), default=MODEL_TYPE_ALL)
    list_parser.add_argument("--format", choices=(LIST_FORMAT_TEXT, LIST_FORMAT_JSON), default=LIST_FORMAT_TEXT)
    list_parser.add_argument("--dir", default="tests/models")
    list_parser.set_defaults(func=cmd_list)

    filename_parser = sub.add_parser("filename")
    filename_parser.add_argument("preset")
    filename_parser.set_defaults(func=cmd_filename)

    path_parser = sub.add_parser("path")
    path_parser.add_argument("preset")
    path_parser.add_argument("--dir", default="tests/models")
    path_parser.set_defaults(func=cmd_path)

    info_parser = sub.add_parser("info")
    info_parser.add_argument("preset")
    info_parser.add_argument("--dir", default="tests/models")
    info_parser.add_argument("--format", choices=("json", "text"), default="json")
    info_parser.set_defaults(func=cmd_info)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("preset")
    status_parser.add_argument("--dir", default="tests/models")
    status_parser.add_argument("--format", choices=("json", "text"), default="json")
    status_parser.set_defaults(func=cmd_status)

    status_all_parser = sub.add_parser("status-all")
    status_all_parser.add_argument("--unreal-matrix", action="store_true")
    status_all_parser.add_argument("--package", action="store_true")
    status_all_parser.add_argument("--type", choices=(MODEL_TYPE_ALL, MODEL_TYPE_TEXT, MODEL_TYPE_EMBEDDING), default=MODEL_TYPE_ALL)
    status_all_parser.add_argument("--dir", default="tests/models")
    status_all_parser.add_argument("--format", choices=("json", "text"), default="json")
    status_all_parser.add_argument("--only", choices=MODEL_STATUS_FILTERS, default=MODEL_STATUS_ANY)
    status_all_parser.set_defaults(func=cmd_status_all)

    status_summary_parser = sub.add_parser("status-summary")
    status_summary_parser.add_argument("--unreal-matrix", action="store_true")
    status_summary_parser.add_argument("--package", action="store_true")
    status_summary_parser.add_argument(
        "--type", choices=(MODEL_TYPE_ALL, MODEL_TYPE_TEXT, MODEL_TYPE_EMBEDDING), default=MODEL_TYPE_ALL
    )
    status_summary_parser.add_argument("--dir", default="tests/models")
    status_summary_parser.add_argument("--format", choices=("json", "text"), default="json")
    status_summary_parser.set_defaults(func=cmd_status_summary)

    download_plan_parser = sub.add_parser("download-plan")
    download_plan_parser.add_argument("--unreal-matrix", action="store_true")
    download_plan_parser.add_argument("--package", action="store_true")
    download_plan_parser.add_argument(
        "--type", choices=(MODEL_TYPE_ALL, MODEL_TYPE_TEXT, MODEL_TYPE_EMBEDDING), default=MODEL_TYPE_ALL
    )
    download_plan_parser.add_argument("--dir", default="tests/models")
    download_plan_parser.add_argument("--format", choices=("json", "text"), default="json")
    download_plan_parser.set_defaults(func=cmd_download_plan)

    preset_for_file_parser = sub.add_parser("preset-for-file")
    preset_for_file_parser.add_argument("filename")
    preset_for_file_parser.set_defaults(func=cmd_preset_for_file)

    validate_file_parser = sub.add_parser("validate-file")
    validate_file_parser.add_argument("--preset", required=True)
    validate_file_parser.add_argument("--dir", default="tests/models")
    validate_file_parser.add_argument("--path", default="")
    validate_file_parser.add_argument("--validate-metadata", action="store_true")
    validate_file_parser.set_defaults(func=cmd_validate_file)

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("preset")
    inspect_parser.add_argument("--dir", default="tests/models")
    inspect_parser.add_argument("--path", default="")
    inspect_parser.add_argument("--validate", action="store_true")
    inspect_parser.add_argument("--format", choices=("json", "text"), default="json")
    inspect_parser.set_defaults(func=cmd_inspect)

    download_parser = sub.add_parser("download")
    download_parser.add_argument("--preset", default="")
    download_parser.add_argument("--dir", default="tests/models")
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.add_argument("--validate-only", action="store_true")
    download_parser.add_argument("--token", default="")
    download_parser.add_argument("--url", default="")
    download_parser.add_argument("--hf-repo", default="")
    download_parser.add_argument("--hf-file", default="")
    download_parser.add_argument("--hf-revision", default="")
    download_parser.add_argument("--file", default="")
    download_parser.add_argument("--min-bytes", type=int, default=MIN_CUSTOM_MODEL_BYTES)
    download_parser.add_argument("--sha256", default="")
    download_parser.add_argument("--validate-metadata", action="store_true")
    download_parser.set_defaults(func=cmd_download)

    validate_parser = sub.add_parser("validate-manifest")
    validate_parser.set_defaults(func=cmd_validate_manifest)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
