# LoRA Adapters

Astral loads LoRA adapters as model-scoped native handles, then attaches them to
sessions between requests. Adapter changes are setup operations; decoding must
be canceled or completed before changing the active adapter set.

## C ABI

- `AstralAdapterDesc`
- `AstralAdapterInfo`
- `astral_model_adapter_load()`
- `astral_model_adapter_info()`
- `astral_model_adapter_path_copy()`
- `astral_model_adapter_release()`
- `astral_session_adapters_clear()`
- `astral_session_adapters_add()`
- `astral_session_adapters_count()`
- `astral_session_adapters_get()`
- `astral_session_adapters_set_scale()`

`AstralAdapterDesc::path` is a UTF-8 path span owned by the caller for the
duration of `astral_model_adapter_load()`. The returned adapter handle retains
the native model and a path copy until released.

## Ownership

Adapter handles belong to the model that loaded them. A session retains each
adapter it attaches and releases those references when adapters are cleared or
the session is destroyed. The application still owns the original adapter handle
and must call `astral_model_adapter_release()` when it is no longer needed.

Each session can attach up to `ASTRAL_SESSION_ADAPTERS_MAX` adapters. Attempts to
attach an adapter from a different model, exceed the fixed session capacity, or
change adapters while decoding return a native error instead of growing hidden
state.

## Performance

Adapter file loading is a cold path. Attach, clear, query, and scale updates are
bounded setup-time operations over the fixed session adapter array. The decode
and stream-read paths do not parse adapter paths or inspect wrapper-owned
collections.

## Unity

`AstralAdapter` owns a native adapter handle and releases it with `Dispose()`.
Use `AstralModel.LoadAdapter()` to load an adapter, then
`AstralSession.AddAdapter()`, `ClearAdapters()`, `GetAdapterCount()`,
`GetAdapter()`, and `SetAdapterScale()` between requests. `GetInfo()` and
`GetPath()` expose native adapter diagnostics for UI and logs without parsing
session state strings.

## Unreal

`UAstralModel::LoadAdapter()` returns an adapter handle for
`UAstralSession::AddAdapter()`. Blueprint callers can inspect adapter count,
adapter handle, path, reference count, and scale without using comma-separated
strings. Use `UAstralBlueprintLibrary::GetAdapterInfoResult()` and
`CopyAdapterPathResult()` for loaded-adapter diagnostics.

## Example

```c
enum {
    kPrimaryAdapterIndex = 0,
};
const float kInitialAdapterScale = 0.75f;
const float kUpdatedAdapterScale = 0.5f;

AstralAdapterDesc desc = {0};
desc.size = sizeof(AstralAdapterDesc);
desc.path = adapter_path;

AstralHandle adapter = 0;
AstralErr err = astral_model_adapter_load(model, &desc, &adapter);
if (err == ASTRAL_OK) {
    err = astral_session_adapters_add(session, adapter, kInitialAdapterScale);
    err = astral_session_adapters_set_scale(session, kPrimaryAdapterIndex, kUpdatedAdapterScale);
    AstralAdapterInfo info = {0};
    info.size = sizeof(AstralAdapterInfo);
    err = astral_model_adapter_info(adapter, &info);
    astral_model_adapter_release(adapter);
}
```

## Validation

```bash
cmake --build --preset unity-plugin -j8
ctest --preset dev -R '^(test_inference|test_abi_invalid_args|gate_source_scans|gate_doc_links)$' --output-on-failure
```
