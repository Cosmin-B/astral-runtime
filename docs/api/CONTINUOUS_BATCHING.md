# Continuous Batching (v0.2+)

Astral v0.2 introduces **continuous batching across slots** via **conversations** (`astral_conv_*`). A conversation is an independent prompt+generation stream that runs inside a **model-scoped executor**. The executor batches work from multiple conversations into a single provider eval call when possible.

This keeps provider selection and model ownership **out of per-token paths**: the hot loop is a single executor thread per model.

## C API overview

### 1) Configure the model executor (once per model)

Call `astral_model_executor_configure()` before creating any conversations:

- `AstralExecutorDesc.max_slots`: maximum concurrent conversations for this model.
- `AstralExecutorDesc.max_batch_tokens`: per-tick token cap (keep `<= n_batch` for your provider/model).
- `AstralExecutorDesc.worker_hint`: reserved for ABI compatibility; set it to `0`.

Continuous batching requires a threads-enabled Astral build. Configuration returns
`ASTRAL_E_UNSUPPORTED` when `ASTRAL_ENABLE_THREADS=OFF`.

### 2) Create conversations (slots)

`astral_conv_create()` returns a handle that is bound to one slot in the executor.

Conversations are **not thread-safe** for multi-producer use; keep single-threaded control per conversation (similar to sessions).

### 3) Feed + decode

- `astral_conv_feed()`: tokenizes and appends prompt tokens.
  - For an “empty prompt” run, call with `{NULL, 0}` and `finalize=1` so the provider can still see BOS.
- `astral_conv_decode()`: transitions the conversation to decoding and schedules it in the executor loop.

### 4) Streaming + meta

If `AstralConvDesc.stream_enabled=1`:

- `astral_conv_stream_read()` returns UTF-8 bytes (detokenized).
- `astral_conv_stream_read_meta()` returns optional per-token metadata events.
  - Enable meta events via `astral_conv_set_logprobs()` (non-zero `n_probs`).

### 4b) Per-slot grammar (constrained decoding)

Conversations support per-slot grammar binding (provider-dependent):

- `astral_conv_grammar_set_gbnf()`
- `astral_conv_grammar_set_json_schema()`
- `astral_conv_grammar_clear()`

Grammar compilation/binding happens in the executor thread before the conversation advances decoding.

### 5) Lifecycle

- `astral_conv_cancel()`: requests cancellation.
- `astral_conv_wait()`: joins completion/cancel/failure state.
- `astral_conv_reset()`: resets for reuse (must not be decoding).

## Provider requirements

Continuous batching uses optional backend ops from `include/astral_backend.h`:

- `session_create_ex(..., max_slots, ...)`
- `session_batch_eval(tokens[], token_count, out_output_count)`
- `session_batch_logits(output_index, out_view)`
- `session_slot_reset(slot_id)` clears provider-owned slot state; the scheduler also resets Astral-owned slot counters before reuse.

Per-slot grammar binding uses additional optional ops:

- `session_grammar_set_gbnf_for_slot(session_ctx, slot_id, ...)`
- `session_grammar_set_json_schema_for_slot(session_ctx, slot_id, ...)`
- `session_grammar_clear_for_slot(session_ctx, slot_id)`
- `session_apply_grammar_for_slot(session_ctx, slot_id, tokens, logits, count)`

The built-in `cpu` and `mock` backends implement these.

## Notes / current limitations

- `astral_session_*` and `astral_conv_*` are intentionally separate: sessions are per-session worker based; conversations are executor based.

## Scheduling and tuning

Continuous batching is a trade-off: you can bias toward **throughput** (large batches, many active slots) or **latency**
(smaller batches, fewer slots contending for each tick).

Current knobs:

- `AstralExecutorDesc.max_batch_tokens`: global per-tick token cap. Keep this `<= model n_batch` for best behavior.
- `astral_model_executor_tune()`:
  - `AstralExecutorTuning.max_prompt_tokens_per_slot_tick`: per-slot prompt token cap per tick (helps control TTFT vs bulk throughput).

Scheduling policy (current):

- Prompt ingest: round-robin across slots with a per-slot cap.
- Decode: at most 1 token per slot per tick (round-robin) to avoid starvation.

## Multi-model story

Executors are **model-scoped**: each `AstralHandle model` may have its own executor configured via
`astral_model_executor_configure()`. This enables:

- Multiple independent models in the same process (each with its own conversations/slots).
- Independent scheduling and tuning per model (e.g. different `max_slots` / `max_batch_tokens`).

Practical constraint:

- Each active model executor owns one dedicated thread for its provider session and slot scheduler.
  Executor threads do not occupy runtime worker-pool lanes, so bounded session, embedding, and memory-index
  jobs continue to use the full `AstralInit.thread_count` budget. Account for one additional thread per active
  model executor when sizing the process.

## Unity API

Unity exposes the same executor/conversation split:

```csharp
using Astral.Runtime;
using Unity.Collections;

model.ConfigureExecutor(AstralExecutorConfig.Default);

using var conv = AstralConversation.Create(model, AstralConversationConfig.Default);
using var buffer = new NativeArray<byte>(
    AstralConversation.DefaultStreamBufferBytes,
    Allocator.Persistent,
    NativeArrayOptions.UninitializedMemory);

conv.SetSystemPrompt("Answer as an in-game navigator.");
conv.Feed("Where should I go next?", finalize: true);
conv.Decode();

while (conv.GetState() == AstralSessionState.Decoding)
{
    int bytes = conv.ReadStream(buffer, AstralConversation.NonBlockingTimeoutMs);
    if (bytes > AstralNative.ASTRAL_OK)
    {
        ConsumeUtf8(buffer, bytes);
    }
}
```

`AstralConversation` also exposes per-slot grammar, toolset, logprob metadata,
stop sequences, media feed, cancellation, reset, and stats APIs. Use
`ReadStream(NativeArray<byte>)` and `ReadStreamMeta(NativeArray<AstralTokenMeta>)`
for allocation-free polling; `ReadStreamAsString()` is only a convenience path.
