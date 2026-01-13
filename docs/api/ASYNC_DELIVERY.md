# Async Delivery

Astral keeps async work native and lets engine wrappers decide how to deliver
results on their main threads. Public async calls return native status codes,
tickets, or stream bytes instead of invoking callbacks from runtime hot paths.

## C ABI

Unified request refs:

- `astral_request_from_session()`
- `astral_request_from_conversation()`
- `astral_request_from_agent_chat()`
- `astral_request_from_embedding()`
- `astral_request_state()`
- `astral_request_cancel()`
- `astral_request_wait()`

Generation sessions:

- `astral_session_decode()`
- `astral_session_cancel()`
- `astral_session_state()`
- `astral_session_wait()`
- `astral_stream_read()`
- `astral_stream_read_meta()`

Continuous-batching conversations:

- `astral_conv_decode()`
- `astral_conv_cancel()`
- `astral_conv_state()`
- `astral_conv_wait()`
- `astral_conv_stream_read()`
- `astral_conv_stream_read_meta()`

Agents:

- `astral_agent_chat_enqueue()`
- `astral_agent_chat_cancel()`
- `astral_agent_chat_stream_read()`
- `astral_agent_chat_result()`

Embeddings:

- `astral_embed_enqueue()`
- `astral_embed_enqueue_image()`
- `astral_embed_enqueue_audio()`
- `astral_embed_enqueue_multimodal()`
- `astral_embed_collect()`
- `astral_embed_cancel()`

## Ownership

Stream buffers and embedding vectors are caller-owned. Session and conversation
streams have one consumer. Embedding enqueue calls return a ticket that remains
valid until `astral_embed_collect()` consumes the result or `astral_embed_cancel()`
releases queued work.

`AstralRequestRef` is a value type for engine queues. It carries the owner
handle, request kind, and embedding ticket when one exists. It does not extend
the lifetime of the owner handle.

## Error Behavior

Bounded queues report `ASTRAL_E_BUSY` when capacity is exhausted. Polling calls
use `ASTRAL_E_TIMEOUT` when no data is ready before the requested deadline.
Canceled operations return `ASTRAL_E_CANCELED` at the wait/result boundary.
Stale or unknown tickets return `ASTRAL_E_INVALID`.

## Performance

The native runtime avoids callback invocation from token, stream, embedding, and
queue hot paths. Engines poll or wait at their boundary and then marshal only the
data they need. Ticketed embedding queues keep capacity bounded and allow callers
to shed queued work without growing memory.

## Unity

`AstralSession` exposes `Decode()`, `Cancel()`, `GetState()`, `WaitResult()`,
and `ReadStream()`. `AstralAgent` exposes chat enqueue, cancel, result, and
stream reads. `AstralEmbedder` exposes ticketed enqueue, `Collect()`, and
`Cancel()` for queued embedding work. Wrappers can store `AstralRequestRef` in
their managed queues and poll `AstralRequestStatus` before dispatching callbacks.

## Unreal

Unreal result structs carry native error codes, tickets, and backpressure flags.
Blueprint delegates are delivered by the plugin on the game thread; native C++
callers can keep using direct poll and wait calls. `AstralRequestStatus` gives
Blueprint queues one state enum across generation, conversations, agent chat,
and embedding tickets.

## Validation

```bash
cmake --build --preset unity-plugin -j8
ctest --preset dev -R '^(test_inference|test_media|test_abi_invalid_args|gate_source_scans|gate_doc_links)$' --output-on-failure
```
