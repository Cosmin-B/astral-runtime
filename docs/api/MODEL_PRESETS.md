# Model Presets

Astral model presets describe known GGUF fixtures and sample models without
committing model files. The manifest lives at `scripts/model_presets.json` and
records preset name, label, model type, Hugging Face repository, filename,
revision, size, SHA-256, license note, context length, embedding dimension, and
package/sample-matrix eligibility.

## Tooling

- `scripts/model_preset_tool.py list`
- `scripts/model_preset_tool.py list --package`
- `scripts/model_preset_tool.py list --type embedding --format json`
- `scripts/model_preset_tool.py filename <preset>`
- `scripts/model_preset_tool.py path <preset> --dir <dir>`
- `scripts/model_preset_tool.py info <preset> --dir <dir>`
- `scripts/model_preset_tool.py inspect <preset> --dir <dir> --validate`
- `scripts/model_preset_tool.py status-all --type embedding --format json`
- `scripts/model_preset_tool.py status-all --only not-ready --format json`
- `scripts/model_preset_tool.py status-summary --type embedding --format json`
- `scripts/model_preset_tool.py download-plan --type embedding --format text`
- `scripts/model_preset_tool.py validate-file --preset <preset> --dir <dir> --validate-metadata`
- `tests/model_downloader.sh --preset <preset> --dry-run`
- `tests/model_downloader.sh --preset <preset> --validate-only --validate-metadata`
- `tests/model_downloader.sh --preset <preset> --inspect-metadata`
- `tests/model_downloader.sh --preset <preset> --info`
- `tests/model_downloader.sh --preset <preset> --status`
- `tests/model_downloader.sh --status-all --list-type embedding --status-format json`
- `tests/model_downloader.sh --status-all --status-only not-ready --status-format json`
- `tests/model_downloader.sh --status-summary --list-type embedding --status-format json`
- `tests/model_downloader.sh --download-plan --list-type embedding --status-format text`
- `tests/model_downloader.sh --preset <preset> --print-path`
- `tests/model_downloader.sh --list-presets --list-type text`
- `tests/model_downloader.sh --list-package --list-format json`
- `tests/model_downloader.sh --list-unreal-matrix --list-format json`
- `tests/model_downloader.sh --list-presets --list-type embedding --list-format json`

`--dry-run` prints the resolved preset, output path, URL, byte size, checksum,
and repeatable downloader command without touching the network. `--validate-only`
checks an existing local file against the manifest size and SHA-256 and returns a
non-zero exit code for missing, truncated, or checksum-drifted files.
`--validate-metadata` also reads GGUF header metadata without loading tensor
data and checks manifest context length, embedding dimension, and embedding
pooling support for embedding presets.

`info` prints a stable JSON record with the preset name, model type, repository,
revision, URL, resolved local path, byte size, checksum, context length,
embedding dimension, package flag, sample-matrix flag, license note, and
repeatable downloader command. `list --format json` prints the same records for
every selected preset, optionally filtered by `--type text` or `--type
embedding`. Engine setup tools can consume this output without scraping dry-run
text. The generated downloader command is shell-quoted so paths with spaces can
be copied or displayed without changing the argument boundaries.

`status` prints the same preset metadata plus local file state. `status-all`
prints those records for the selected manifest rows so setup tools can inspect a
whole local model directory without shelling out once per preset. The status
value is `missing`, `partial`, `invalid`, or `ready`; each record includes
final-file bytes, `.part` bytes, expected bytes, checksum result, an error
string for invalid files, and the repeatable downloader command. Engine setup
screens can use this before starting a first-run download.
Use `status-all --only missing`, `partial`, `invalid`, `ready`, or `not-ready`
when a setup UI only needs rows that require download or repair. The
`tests/model_downloader.sh` wrapper exposes the same filter as `--status-only`.
Use `status-summary` when an engine setup panel, CI check, or editor bootstrap
only needs aggregate readiness counts and byte totals for the selected preset
set. The summary reports total, ready, missing, partial, invalid, not-ready,
expected bytes, present bytes, and partial bytes without network access.
Use `download-plan` when a setup tool needs repeatable commands for only the
selected presets that are not ready. JSON output returns the same status records
as `status-all`; text output prints one downloader command per missing, partial,
or invalid preset.

`inspect` prints GGUF metadata derived from the local file header: architecture,
context length, embedding dimension, embedding support, and metadata entry count.
Use `--validate` to make mismatches fail with a non-zero exit code before an
engine sample or local run tries to load the model.

Custom downloads are accepted through `--url` or `--hf-repo` plus `--hf-file`.
The wrapper rejects custom filenames that are not local `.gguf` basenames,
non-positive minimum byte counts, and malformed SHA-256 pins before it touches
the network.

## Ownership

The preset manifest is source-controlled. GGUF files, partial downloads, and
download logs stay outside commits. The downloader writes to `tests/models` by
default and resumes through a `.part` file before replacing the final path after
checksum validation. Download progress is written to stderr so setup tools can
keep stdout reserved for JSON, paths, or repeatable commands.

## Engine Use

Unreal and Unity wrappers should use preset names only for setup tools,
samples, and editor workflows. Runtime model loading still receives a concrete
filesystem path through the native model descriptor. Packaged builds can read
`include_in_package` from the shared manifest, then pass engine root directories
to `astral_model_path_resolve` before load so Unity, Unreal, and native hosts
share the same UTF-8 path sizing and join behavior.

## Validation

```bash
python3 scripts/model_preset_tool.py validate-manifest
python3 scripts/model_preset_tool.py list --type embedding --format json
python3 scripts/model_preset_tool.py list --package --format json
python3 scripts/model_preset_tool.py info qwen3-0.6b-q8 --dir tests/models
python3 scripts/model_preset_tool.py inspect qwen3-embed-0.6b-q8 --dir tests/models --validate
python3 scripts/model_preset_tool.py status qwen3-0.6b-q8 --dir tests/models
python3 scripts/model_preset_tool.py status-all --type embedding --format json --dir tests/models
python3 scripts/model_preset_tool.py status-all --only not-ready --format json --dir tests/models
python3 scripts/model_preset_tool.py status-summary --type embedding --format json --dir tests/models
python3 scripts/model_preset_tool.py download-plan --type embedding --format text --dir tests/models
./tests/model_downloader.sh --preset qwen3-0.6b-q8 --dry-run
./tests/model_downloader.sh --preset qwen3-embed-0.6b-q8 --inspect-metadata
./tests/model_downloader.sh --preset qwen3-0.6b-q8 --status
./tests/model_downloader.sh --status-all --list-package --status-format text
./tests/model_downloader.sh --status-all --status-only not-ready --status-format json
./tests/model_downloader.sh --status-summary --list-package --status-format text
./tests/model_downloader.sh --download-plan --list-type embedding --status-format text
./tests/model_downloader.sh --preset qwen3-embed-0.6b-q8 --dry-run
./tests/model_downloader.sh --list-presets --list-type text
./tests/model_downloader.sh --list-package --list-format json
./tests/model_downloader.sh --list-unreal-matrix --list-format json
./tests/model_downloader.sh --preset qwen3-0.6b-q8 --print-path
```

Expected evidence markers include `manifest OK`, `preset: qwen3-0.6b-q8`,
`"download_command"`, `"status"`, `"not_ready"`, `model_downloader.sh`,
`sha256:`, and a resolved `.gguf` path.
