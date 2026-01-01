# Model Presets

Astral model presets describe known GGUF fixtures and sample models without
committing model files. The manifest lives at `scripts/model_presets.json` and
records preset name, label, model type, Hugging Face repository, filename,
revision, size, SHA-256, license note, context length, embedding dimension, and
sample-matrix eligibility.

## Tooling

- `scripts/model_preset_tool.py list`
- `scripts/model_preset_tool.py list --type embedding --format json`
- `scripts/model_preset_tool.py filename <preset>`
- `scripts/model_preset_tool.py path <preset> --dir <dir>`
- `scripts/model_preset_tool.py info <preset> --dir <dir>`
- `scripts/model_preset_tool.py validate-file --preset <preset> --dir <dir>`
- `tests/model_downloader.sh --preset <preset> --dry-run`
- `tests/model_downloader.sh --preset <preset> --validate-only`
- `tests/model_downloader.sh --preset <preset> --info`
- `tests/model_downloader.sh --preset <preset> --print-path`
- `tests/model_downloader.sh --list-presets --list-type text`
- `tests/model_downloader.sh --list-presets --list-type embedding --list-format json`

`--dry-run` prints the resolved preset, output path, URL, byte size, checksum,
and repeatable downloader command without touching the network. `--validate-only`
checks an existing local file against the manifest size and SHA-256 and returns a
non-zero exit code for missing, truncated, or checksum-drifted files.

`info` prints a stable JSON record with the preset name, model type, repository,
revision, URL, resolved local path, byte size, checksum, context length,
embedding dimension, sample-matrix flag, license note, and repeatable downloader
command. `list --format json` prints the same records for every selected preset,
optionally filtered by `--type text` or `--type embedding`. Engine setup tools
can consume this output without scraping dry-run text.

## Ownership

The preset manifest is source-controlled. GGUF files, partial downloads, and
download logs stay outside commits. The downloader writes to `tests/models` by
default and resumes through a `.part` file before replacing the final path after
checksum validation.

## Engine Use

Unreal and Unity wrappers should use preset names only for setup tools,
samples, and editor workflows. Runtime model loading still receives a concrete
filesystem path through the native model descriptor. Packaged builds can pass
engine root directories to `astral_model_path_resolve` before load so Unity,
Unreal, and native hosts share the same UTF-8 path sizing and join behavior.

## Validation

```bash
python3 scripts/model_preset_tool.py validate-manifest
python3 scripts/model_preset_tool.py list --type embedding --format json
python3 scripts/model_preset_tool.py info qwen3-0.6b-q8 --dir tests/models
./tests/model_downloader.sh --preset qwen3-0.6b-q8 --dry-run
./tests/model_downloader.sh --preset qwen3-embed-0.6b-q8 --dry-run
./tests/model_downloader.sh --list-presets --list-type text
./tests/model_downloader.sh --preset qwen3-0.6b-q8 --print-path
```

Expected evidence markers include `manifest OK`, `preset: qwen3-0.6b-q8`,
`"download_command"`, `sha256:`, and a resolved `.gguf` path.
