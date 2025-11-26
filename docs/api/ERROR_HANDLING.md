# Error Handling

Astral's public C ABI reports failures through `AstralErr`. Functions that return
payloads write to output parameters only when they return `ASTRAL_OK`, unless the
function documentation says otherwise. Engine integrations should log both the
numeric code and `astral_last_error()` for failures that come from the runtime or
backend.

## Error Codes

| Code | Meaning | Caller response |
|---|---|---|
| `ASTRAL_OK` | Operation completed. | Use output parameters returned by the call. |
| `ASTRAL_E_INVALID` | The caller passed an invalid handle, null pointer, malformed span, bad struct size, or unsupported configuration value. | Treat as an integration bug. Validate at the engine boundary before retrying. |
| `ASTRAL_E_NOMEM` | Astral could not reserve, commit, or allocate required memory. Arena-backed sessions can also return this when fixed pools are exhausted. | Release unused sessions/models, reduce configured memory, or report a hard load failure to the user. |
| `ASTRAL_E_BUSY` | A startup-only or registration resource is already in use or full. Dynamic backend loading returns this for provider name collisions or a full plugin table. | Do not retry in a hot loop. Resolve the conflicting registration or reduce plugin count. |
| `ASTRAL_E_TIMEOUT` | A wait or stream read did not produce data before the requested deadline. | Poll again, yield the engine tick, or continue frame work. This is not a fatal decode failure. |
| `ASTRAL_E_STATE` | The object is in the wrong state for the requested operation. Examples include changing session options while decoding or reading a stream concurrently from two consumers. | Fix call ordering. Cancel/wait/reset before reconfiguring, and keep stream ownership single-consumer. |
| `ASTRAL_E_BACKEND` | The selected provider failed below the ABI layer, such as model load, tokenization, decode, or llama.cpp integration failure. | Log `astral_last_error()` and provider context. Retrying only helps after changing the model, backend, or input. |
| `ASTRAL_E_CANCELED` | The operation ended because the caller canceled it. | Treat as an expected control-flow result when cancellation was requested. |
| `ASTRAL_E_UNSUPPORTED` | The selected build, backend, or model does not implement the requested feature. | Query capabilities where available and route to a supported path instead of relying on fallback behavior. |

## Logging

Use `astral_error_string(err)` for stable labels and `astral_last_error()` for
the latest thread-local detail string. `astral_last_error()` is useful for logs,
but it is not a structured API contract and should not drive gameplay logic.

For Unreal, prefer `LogAstralRT` messages that include the failing API name,
the integer code, and the last-error text:

```cpp
const AstralErr Err = astral_session_decode(Session);
if (Err != ASTRAL_OK)
{
    UE_LOG(LogAstralRT, Error, TEXT("astral_session_decode failed (%d): %s"),
           static_cast<int32>(Err),
           UTF8_TO_TCHAR(astral_last_error()));
}
```

For Unity or standalone C# bindings, convert the last-error pointer immediately
after the failing call. Later Astral calls on the same thread may overwrite it.

## Stream Reads

`astral_stream_read()` and `astral_conv_stream_read()` return byte counts instead
of `AstralErr`:

- `> 0`: bytes written to the caller's buffer.
- `0`: stream ended.
- `< 0`: negative `AstralErr`.

`ASTRAL_E_TIMEOUT` from stream reads means no bytes were available before the
deadline. A normal engine tick should treat it as "try again later", not as a
failure. Other negative values should be logged and surfaced.

## Ownership Boundaries

`astral_handle_valid()` only reports whether a handle currently resolves in the
runtime table. It does not make stale handle use safe and it does not detect every
use-after-free pattern. Engine wrappers should own handles with a single release
site and should clear stored handles after destroy/release calls.

Validate inputs at engine or ABI boundaries. Once decode, stream, and embedding
hot paths receive validated objects and spans, do not add extra defensive checks
inside the steady-state loops.
