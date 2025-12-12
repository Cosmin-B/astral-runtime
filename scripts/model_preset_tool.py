#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "model_presets.json"
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
        if len(sha256) != SHA256_HEX_LENGTH:
            raise ValueError(f"preset {name} has invalid sha256 length")
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
                include_in_unreal_sample_matrix=bool(row.get("include_in_unreal_sample_matrix", False)),
                huggingface_base_url=base_url.rstrip("/"),
            )
        )
    return out


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
    if preset.name == CUSTOM_PRESET_NAME:
        return f"{Path('tests') / 'model_downloader.sh'} --url {preset.url} --file {preset.filename} --dir {output_dir}"
    return f"{Path('tests') / 'model_downloader.sh'} --preset {preset.name} --dir {output_dir}"


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
    selected = presets
    if args.unreal_matrix:
        selected = [preset for preset in presets if preset.include_in_unreal_sample_matrix]
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


def cmd_validate_file(args: argparse.Namespace) -> int:
    preset = _find_preset(_load_presets(Path(args.manifest)), args.preset)
    output_path = Path(args.path).expanduser() if args.path else _resolved_output_path(args, preset)
    _verify_existing(output_path, preset)
    print(f"valid: {output_path}")
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
        print(f"valid: {output_path}")
        return EXIT_OK

    if output_path.exists():
        _verify_existing(output_path, preset)
        print(f"ready: {output_path}")
        return EXIT_OK

    _download(preset, output_path, args.token or _token_from_env())
    _verify_existing(output_path, preset)
    print(f"downloaded: {output_path}")
    return EXIT_OK


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    presets = _load_presets(Path(args.manifest))
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
    list_parser.set_defaults(func=cmd_list)

    filename_parser = sub.add_parser("filename")
    filename_parser.add_argument("preset")
    filename_parser.set_defaults(func=cmd_filename)

    path_parser = sub.add_parser("path")
    path_parser.add_argument("preset")
    path_parser.add_argument("--dir", default="tests/models")
    path_parser.set_defaults(func=cmd_path)

    preset_for_file_parser = sub.add_parser("preset-for-file")
    preset_for_file_parser.add_argument("filename")
    preset_for_file_parser.set_defaults(func=cmd_preset_for_file)

    validate_file_parser = sub.add_parser("validate-file")
    validate_file_parser.add_argument("--preset", required=True)
    validate_file_parser.add_argument("--dir", default="tests/models")
    validate_file_parser.add_argument("--path", default="")
    validate_file_parser.set_defaults(func=cmd_validate_file)

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
