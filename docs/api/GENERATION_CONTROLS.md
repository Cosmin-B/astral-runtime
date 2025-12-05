# Generation Controls (C ABI)

This page describes Astral’s provider-agnostic “generation controls” surface in `astral/include/astral_rt.h`.

## Capabilities / Limits

Before using optional features, query `astral_model_caps()` and `astral_model_limits()`:

- Core (always on): `ASTRAL_CAP_SAMPLER_EXT`, `ASTRAL_CAP_STOP_SEQS`, `ASTRAL_CAP_LOGPROBS`
- Provider-dependent:
  - `ASTRAL_CAP_KV_STATE`: `astral_session_state_*`
  - `ASTRAL_CAP_LORA`: `astral_model_adapter_*` + `astral_session_adapters_*`
  - `ASTRAL_CAP_GRAMMAR` + `ASTRAL_CAP_GRAMMAR_GBNF`: `astral_session_set_grammar_gbnf`
  - `ASTRAL_CAP_GRAMMAR` + `ASTRAL_CAP_GRAMMAR_JSON_SCHEMA`: `astral_session_set_grammar_json_schema`
  - `ASTRAL_CAP_SLOTS`: `astral_session_set_slot`

## Sampler

Use `astral_session_set_sampler(session, &AstralSamplerDesc)` to configure sampling.

Notes:
- `AstralSamplerDesc.size` must be `sizeof(AstralSamplerDesc)`.
- This call is **not** allowed while decoding (cancel + wait first).

Supported knobs (core-implemented, provider-agnostic):
- Temperature / top-k / top-p
- `min_p`, `typical_p`
- Repeat/presence/frequency penalties:
  - `repeat_penalty`, `repeat_last_n`, `penalize_nl`
  - `presence_penalty`, `frequency_penalty`
- Mirostat:
  - `mirostat` (0/1/2), `mirostat_tau`, `mirostat_eta`

### Penalty Prompt (token span)

For “penalize repeats of a fixed prefix/system prompt”, use:

- `astral_session_penalty_prompt_set_tokens(session, tokens, count)`

The token ids are counted once and included in the penalty counts for each generated token (without per-token work in wrappers).

## Stop Sequences

Stop sequences are matched by **token ids** (tokenized once) and are **suppressed** from the UTF-8 stream.

- `astral_session_stop_clear(session)`
- `astral_session_stop_add_utf8(session, utf8)`
- `astral_session_stop_set_utf8(session, seqs, count)` (bulk)

## Logprobs / Per-token Metadata (side-channel)

The UTF-8 stream (`astral_stream_read`) remains bytes-first and unchanged. Optional per-token metadata is published via a parallel stream:

- Enable/disable: `astral_session_set_logprobs(session, n_probs)` (`0` disables, clamped to `ASTRAL_LOGPROBS_MAX`)
- Consume: `astral_stream_read_meta(session, out_events, capacity, timeout_ms)`

`AstralTokenMeta.logprob` is `log(p(token))` in the sampling distribution actually used for the token (post filters).

## KV State Save/Load

If `ASTRAL_CAP_KV_STATE` is set:

- `astral_session_state_size(session, &bytes)`
- `astral_session_state_save(session, out_buf, &written)`
- `astral_session_state_load(session, state_bytes)`

Preconditions:
- Not allowed while decoding (cancel + wait first).

## Adapters (LoRA)

If `ASTRAL_CAP_LORA` is set:

- Load: `astral_model_adapter_load(model, &AstralAdapterDesc, &out_adapter)`
- Release: `astral_model_adapter_release(adapter)`
- Attach to session:
  - `astral_session_adapters_clear(session)`
  - `astral_session_adapters_add(session, adapter, scale)`
  - `astral_session_adapters_count(session, &out_count)`
  - `astral_session_adapters_get(session, index, &out_adapter, &out_scale)`

Adapters are model-scoped; sessions retain references to attached adapters.
Each session can hold up to `ASTRAL_SESSION_ADAPTERS_MAX` adapters.

## Grammar (GBNF)

If `ASTRAL_CAP_GRAMMAR_GBNF` is set:

- `astral_session_set_grammar_gbnf(session, gbnf_text, root_symbol)`
- `astral_session_clear_grammar(session)`

Grammar constraints are applied in the core sampling loop; providers own any compiled grammar state.

## Grammar (JSON schema)

If `ASTRAL_CAP_GRAMMAR_JSON_SCHEMA` is set:

- `astral_session_set_grammar_json_schema(session, json_schema_utf8)`
- `astral_session_clear_grammar(session)`

Notes:
- JSON schema is compiled provider-side into an internal grammar representation (typically GBNF).
- Providers may reject unsupported schemas (e.g. external `$ref`s) with `ASTRAL_E_INVALID`.

## Slots

If `ASTRAL_CAP_SLOTS` is set:

- `astral_session_set_slot(session, slot_id)`

Slot selection is configured once per session; it must not introduce per-token provider selection overhead.

Notes (CPU/llama backend):
- Multi-slot sessions require creating the llama context with `n_seq_max > 1`. Astral controls this via
  `ASTRAL_LLAMA_MAX_SLOTS` at session creation time (default: `1`).
- If `ASTRAL_LLAMA_MAX_SLOTS=1`, `astral_session_set_slot(session, 0)` is valid; `slot_id > 0` returns `ASTRAL_E_INVALID`.
