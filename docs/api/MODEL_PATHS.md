# Model Paths

`astral_model_path_resolve` converts an engine-relative model path into the
UTF-8 filesystem path passed to `AstralModelDesc::model_path`.

The API is a sizing and join helper. It does not touch the filesystem, create
directories, canonicalize paths, or validate model bytes. Hosts provide the root
strings they already trust:

- `ASTRAL_MODEL_PATH_ROOT_RAW`: return `path` unchanged.
- `ASTRAL_MODEL_PATH_ROOT_CONTENT`: join with `content_root`.
- `ASTRAL_MODEL_PATH_ROOT_SAVED`: join with `saved_root`.
- `ASTRAL_MODEL_PATH_ROOT_CACHE`: join with `cache_root`.
- `ASTRAL_MODEL_PATH_ROOT_DOWNLOAD`: join with `download_root`.

Absolute POSIX, Windows drive, and separator-rooted paths pass through unchanged.
Relative paths join with one `/` if the selected root does not already end with
`/` or `\`.

## Runtime Flow

Use the resolver at setup time, before calling `astral_model_load()`.

1. Choose a root kind for the model source.
2. Fill the matching root span from the host runtime.
3. Call `astral_model_path_resolve()` with a null or zero-length output buffer
   to get the required byte count.
4. Allocate a caller-owned UTF-8 buffer and call the resolver again.
5. Pass the resolved span to `AstralModelDesc::model_path`.

Packaged-content models and downloaded-cache models should use the same manifest
entry from `scripts/model_presets.json`. The preset tells setup tooling which
file, size, and SHA-256 to expect; this resolver only turns the chosen engine
root plus relative path into the concrete load path.

## Ownership

Input spans are borrowed for the call. The caller owns `out_path` and can pass a
null or undersized buffer to retrieve the required byte count in `out_len`. The
count excludes a trailing NUL.

## Engine Roots

Native hosts usually pass `RAW` for absolute paths supplied on the command line
and `CACHE` or `DOWNLOAD` for managed model directories.

Unity should use `CONTENT` for `StreamingAssets` model files and `SAVED`,
`CACHE`, or `DOWNLOAD` for files under `Application.persistentDataPath` after
checksum validation. The Unity wrapper exposes this through `AstralModelPath`
helpers so managed code can keep path policy at the engine boundary.

Unreal should use `CONTENT` for packaged project content, `SAVED` for files
under the project saved directory, and `RAW` for editor-selected absolute paths.
The Unreal wrapper maps `FAstralModelDesc::PathRoot` to this ABI so packaged
projects and editor tools share the same UTF-8 sizing behavior.

For mobile first-run downloads, download into the engine's persistent writable
directory, verify size and checksum through the shared preset tooling, then load
through the resolved persistent/download path. Do not put model files, partial
downloads, or runtime logs in source control.

## Example

```c
const char model_rel[] = "Models/model.gguf";
const char content_root[] = "/GameProject/Content";

AstralModelPathResolveDesc path_desc = {0};
path_desc.size = sizeof(path_desc);
path_desc.root = ASTRAL_MODEL_PATH_ROOT_CONTENT;
path_desc.path.data = (const uint8_t*)model_rel;
path_desc.path.len = (uint32_t)strlen(model_rel);
path_desc.content_root.data = (const uint8_t*)content_root;
path_desc.content_root.len = (uint32_t)strlen(content_root);

uint32_t bytes = 0;
AstralErr err = astral_model_path_resolve(&path_desc, (AstralMutSpanU8){0}, &bytes);
```

After allocating `bytes`, call again and pass the returned buffer to
`AstralModelDesc::model_path`.

For a downloaded-cache model, switch `path_desc.root` to
`ASTRAL_MODEL_PATH_ROOT_DOWNLOAD`, fill `download_root`, and keep `path` as the
manifest filename or a relative path under the download directory.

## Validation

```bash
cmake --build --preset release-with-tests -j8 --target test_abi_invalid_args
ctest --preset release-with-tests -R '^test_abi_invalid_args$' --output-on-failure
ctest --preset release-with-tests -R '^(gate_source_scans|gate_doc_links)$' --output-on-failure
```
