# Model Paths

`astral_model_path_resolve` converts an engine-relative model path into the
UTF-8 filesystem path passed to `AstralModelDesc::model_path`.

The API does not touch the filesystem and does not canonicalize paths. Hosts
provide the root strings they already trust:

- `ASTRAL_MODEL_PATH_ROOT_RAW`: return `path` unchanged.
- `ASTRAL_MODEL_PATH_ROOT_CONTENT`: join with `content_root`.
- `ASTRAL_MODEL_PATH_ROOT_SAVED`: join with `saved_root`.
- `ASTRAL_MODEL_PATH_ROOT_CACHE`: join with `cache_root`.
- `ASTRAL_MODEL_PATH_ROOT_DOWNLOAD`: join with `download_root`.

Absolute POSIX, Windows drive, and separator-rooted paths pass through unchanged.
Relative paths join with one `/` if the selected root does not already end with
`/` or `\`.

## Ownership

Input spans are borrowed for the call. The caller owns `out_path` and can pass a
null or undersized buffer to retrieve the required byte count in `out_len`. The
count excludes a trailing NUL.

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

## Validation

```bash
cmake --build --preset release-with-tests -j8 --target test_abi_invalid_args
ctest --preset release-with-tests -R '^test_abi_invalid_args$' --output-on-failure
```
