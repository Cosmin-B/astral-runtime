# Native Agents

Astral agents own system prompt bytes, memory context bytes, chat history,
prompt assembly, and a conversation handle in native memory. Engine wrappers
pass strings and handles through the ABI; they do not build prompts or manage
history themselves.

## C ABI

- `AstralAgentDesc`
- `AstralAgentMessage`
- `AstralAgentChatDesc`
- `AstralAgentChatResult`
- `astral_agent_create()`
- `astral_agent_destroy()`
- `astral_agent_set_system_prompt()`
- `astral_agent_get_system_prompt_size()`
- `astral_agent_get_system_prompt()`
- `astral_agent_set_summary()`
- `astral_agent_get_summary_size()`
- `astral_agent_get_summary()`
- `astral_agent_set_memory_context()`
- `astral_agent_get_memory_context_size()`
- `astral_agent_get_memory_context()`
- `astral_agent_message_add()`
- `astral_agent_history_clear()`
- `astral_agent_history_count()`
- `astral_agent_history_save_size()`
- `astral_agent_history_save()`
- `astral_agent_history_load()`
- `astral_agent_chat_enqueue()`
- `astral_agent_chat_cancel()`
- `astral_agent_chat_stream_read()`
- `astral_agent_chat_result()`

Agents run on the existing model-scoped conversation executor. Configure the
executor before creating agents for a model. Toolsets and prompt caches can be
bound at creation time and are forwarded to native prompt setup.

## Ownership

The agent copies system prompt, rolling summary, retrieved memory context, and
history content into native storage. Input spans only need to remain valid for
the duration of the call. Chat enqueue assembles one bounded prompt buffer in
this order: system prompt, summary, memory context, history, current user turn,
assistant prefix. The temporary buffer is released before the call returns.
When `AstralAgentDesc::prompt_cache` is set, the agent looks up the assembled
prompt in the native prompt cache during request setup. Cache hits feed cached
token spans directly into the conversation prompt buffer; misses tokenize once,
insert the token span, and then feed those tokens.

`astral_agent_history_save()` serializes the system prompt and history entries
into a caller-provided buffer. Current snapshots include the rolling summary and
memory context. `astral_agent_history_load()` replaces the current native
prompt, summary, memory context, and history copy after validating the payload.

`AstralAgentDesc::overflow_policy` controls history overflow before prompt
decode starts. `ASTRAL_AGENT_OVERFLOW_REJECT` is the default and returns
`ASTRAL_E_NOMEM` when adding history beyond `max_messages` or when the assembled
prompt exceeds `max_prompt_bytes`. `ASTRAL_AGENT_OVERFLOW_TRUNCATE_OLDEST`
removes oldest history messages during message add or prompt setup until the
configured bounds are met.

## Thread Safety

Creation and destruction are control-path operations. A single control thread
should mutate one agent's system prompt, summary, memory context, or history.
Chat streaming follows the same rule as conversations: one consumer may read
from the stream while decode is active.

## Performance

Prompt assembly is outside the decode loop. The assembled buffer is bounded by
`AstralAgentDesc::max_prompt_bytes`, and history growth is bounded by
`max_messages`. The decode hot path remains the existing conversation executor,
stream ring, sampler, grammar, and backend slot machinery.

`AstralAgentChatResult` reports `prompt_cache_reused_tokens`,
`prompt_cache_new_tokens`, `prompt_cache_hits`, and `prompt_cache_misses` for
the most recent request. These counters describe agent prompt setup only; they
do not imply backend KV-prefix reuse.

Unreal wrapper functions use `TRACE_CPUPROFILER_EVENT_SCOPE` around meaningful
agent operations, including history save/load helpers that copy native snapshot
bytes into engine-owned arrays. Unity exposes `AstralAgent` as a thin owned
handle over the same ABI; prompt assembly, summary storage, history snapshots,
and streaming remain native.

## Example

```c
enum {
    kBytesPerKiB = 1024,
    kMaxPromptKiB = 64,
    kMaxTokens = 128,
    kMaxMessages = 64,
    kMaxPromptBytes = kMaxPromptKiB * kBytesPerKiB,
};

AstralAgentDesc desc = {0};
desc.size = sizeof(AstralAgentDesc);
desc.model = model;
desc.prompt_cache = cache;
desc.max_tokens = kMaxTokens;
desc.stream_enabled = 1;
desc.max_messages = kMaxMessages;
desc.max_prompt_bytes = kMaxPromptBytes;

AstralHandle agent = 0;
AstralErr err = astral_agent_create(&desc, &agent);

AstralSpanU8 summary = {0};
err = astral_agent_set_summary(agent, summary);

AstralSpanU8 memory_context = {0};
err = astral_agent_set_memory_context(agent, memory_context);

AstralAgentMessage message = {0};
message.size = sizeof(AstralAgentMessage);
message.role = ASTRAL_AGENT_ROLE_USER;
message.content = user_text;
err = astral_agent_message_add(agent, &message);
```

## Validation

```bash
cmake --build --preset dev -j8 --target test_inference test_abi_invalid_args
ctest --preset dev -R '^(test_inference|test_abi_invalid_args|gate_abi_layout_report|gate_source_scans|gate_doc_links|gate_unreal_header_mirror)$' --output-on-failure
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=1000 ./build/dev/benchmarks/astral_benchmarks --features
```

Expected markers include `features.agent prompt_warmup` and
`features.agent prompt_cache_warmup`.
