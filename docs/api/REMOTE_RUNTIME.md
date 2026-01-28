# Remote Runtime

Astral can load a `remote` backend that forwards completion, tokenization, and
embedding work to an HTTP service while preserving the same native session and
embedding handles used by local providers.

This path is intended for editor tools, local service deployments, and hosted
runtime experiments where the engine should not own a separate client stack.
Latency and retry behavior are bounded by the remote service and network path,
so the local CPU/CUDA providers remain the low-latency path.

## Model Setup

Select the backend with `AstralModelDesc::backend_name = "remote"`.
`model_path` contains the base URL, for example `http://127.0.0.1:8080`.
If an API key is needed, pass it as `model_bytes`; the provider sends it as:

```text
Authorization: Bearer <key>
```

The provider performs `GET /health` during model load. A non-200 response maps
to an Astral error before a model handle is returned.

`https://` URLs require cpp-httplib built with OpenSSL support. Builds without
that support reject HTTPS during model load with `ASTRAL_E_UNSUPPORTED` before
opening a socket.

## Endpoints

The current provider uses a small text protocol:

- `GET /health`: returns `200` when the service is ready.
- `POST /tokenize`: request body is UTF-8 text; response is a comma-separated
  or JSON-like list of integer token ids.
- `POST /completion`: request body is the prompt text; response body is the
  generated text.
- `POST /embeddings`: request body is UTF-8 text; response is a comma-separated
  or JSON-like list of float values.

If `/tokenize` is not available, the provider falls back to byte tokens so basic
session flow can still run against very small loopback services.

## Ownership

Remote model, session, and embedder handles are regular Astral handles. Prompt
and response buffers inside the provider are fixed-capacity native storage.
Callers still own output buffers for tokenization, detokenization, stream reads,
and embedding collection.

Remote requests are not part of the local decode inner loop optimization model.
They run at request boundaries and carry network latency. Native sampling sees
a deterministic token stream produced from the remote response body.

## Errors

Connection failures map to `ASTRAL_E_TIMEOUT`. HTTP `404` maps to
`ASTRAL_E_NOT_FOUND`, `408`/`504` to `ASTRAL_E_TIMEOUT`, `405`/`501` to
`ASTRAL_E_UNSUPPORTED`, and other non-200 statuses to `ASTRAL_E_BACKEND`.
The provider retries transient HTTP statuses (`408`, `429`, `500`, `502`,
`503`, `504`) once before returning the final mapped error.

## Example

```c
AstralModelDesc desc = {0};
desc.size = sizeof(desc);
desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
desc.backend_name.data = (const uint8_t*)"remote";
desc.backend_name.len = 6;
desc.model_path.data = (const uint8_t*)"http://127.0.0.1:8080";
desc.model_path.len = 21;
desc.model_bytes.data = (const uint8_t*)"dev-key";
desc.model_bytes.len = 7;

AstralHandle model = 0;
AstralErr err = astral_model_load(&desc, &model);
```

After load, use `astral_session_create`, `astral_session_feed`,
`astral_session_decode`, and `astral_stream_read` exactly as with local
providers.

## Validation

```bash
cmake --build --preset release-with-tests --target test_backend -j
ctest --preset release-with-tests -R '^test_backend$' --output-on-failure
```

Expected markers:

- `backend_remote_loopback_completion_and_embeddings` passes.
- `backend_remote_auth_failure` passes.
- `backend_remote_https_requires_tls_build` passes.
- `backend_remote_health_retry_and_timeout_status` passes.
