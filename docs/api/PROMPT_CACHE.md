# Prompt Cache API

Astral prompt caches store tokenized prompt sections behind an explicit native
handle. They are intended for setup and prompt assembly work: system prompts,
tool instructions, memory context, chat history, current user text, and raw
caller-defined sections.

## Entry Points

- `astral_prompt_cache_create(desc, out_cache)` creates a bounded cache.
- `astral_prompt_cache_destroy(cache)` releases all cached token storage.
- `astral_prompt_cache_clear(cache)` removes all entries without changing the
  cache budget.
- `astral_prompt_cache_save_size(cache, out_bytes)` returns the byte count for
  serializing the current token entries.
- `astral_prompt_cache_save(cache, out_bytes, out_len)` writes a cache snapshot
  into a caller-owned byte buffer.
- `astral_prompt_cache_load(desc, bytes, out_cache)` restores a snapshot into a
  newly created bounded cache.
- `astral_prompt_cache_put_tokens(cache, key, tokens, token_count)` copies a
  caller-owned token span into the cache.
- `astral_prompt_cache_get_tokens(cache, key, out_tokens, max_tokens,
  out_token_count)` copies cached tokens into a caller-owned buffer.
- `astral_prompt_cache_get_token_view(cache, key, out_tokens,
  out_token_count)` returns a read-only token pointer owned by the cache.
- `astral_prompt_cache_stats(cache, out_stats)` reports occupancy and optional
  hit/miss/eviction counters.
- `astral_session_set_system_prompt(session, system_prompt)` tokenizes and
  feeds the system prompt before user prompt chunks.
- `astral_conv_set_system_prompt(conv, system_prompt)` does the same for a
  conversation slot.

## Ownership And Lifetime

Cache descriptors define `max_entries`, `max_tokens`, and optionally
`max_bytes`. Token entries are copied into native runtime memory and stay valid
until the entry is replaced, evicted, cleared, or the cache is destroyed.
Save/load stores token entries and keys only; callers are responsible for using
snapshots with compatible model identity and tokenizer behavior.

`astral_prompt_cache_get_token_view()` is the fastest path because it avoids a
token copy. The returned pointer is valid only until the next cache mutation or
destroy call. Engine wrappers should use the copy API unless they can keep that
lifetime local and obvious.

Unity exposes `AstralPromptCache` as an owned handle. The wrapper keeps token
input/output in `NativeArray<int>` and serializes cache snapshots through
managed byte arrays; native lookup and eviction behavior remain unchanged.

Agents may bind a prompt cache through `AstralAgentDesc::prompt_cache`. The
agent uses the native view path during chat setup, so repeated identical prompt
assemblies can skip tokenization and feed cached tokens directly into the
conversation prompt buffer.

System prompts must be set before normal prompt text is fed. A late call returns
`ASTRAL_E_STATE`.

## Performance Model

The cache uses a fixed-size open-addressed table with a power-of-two backing
array and bounded token storage. Lookups allocate no memory. The default path
does not write hit/miss counters; enable `ASTRAL_PROMPT_CACHE_FLAG_TRACK_STATS`
only for diagnostics because counters add hot-path stores.

Use `astral_prompt_cache_get_token_view()` for native prompt assembly when the
cache lifetime is already controlled. Use `astral_prompt_cache_get_tokens()`
when crossing ABI or engine ownership boundaries.

## Minimal Example

```c
AstralPromptCacheDesc desc = {0};
desc.size = sizeof(desc);
desc.max_entries = 128;
desc.max_tokens = 32768;

AstralHandle cache = 0;
AstralErr err = astral_prompt_cache_create(&desc, &cache);
if (err != ASTRAL_OK) return err;

AstralPromptCacheKey key = {0};
key.size = sizeof(key);
key.section_kind = ASTRAL_PROMPT_SECTION_SYSTEM;
key.model = model;
key.key = 42;
key.generation = 1;

err = astral_prompt_cache_put_tokens(cache, &key, tokens, token_count);
```

Fast native lookup:

```c
const int32_t* view = NULL;
uint32_t count = 0;
err = astral_prompt_cache_get_token_view(cache, &key, &view, &count);
```

## Validation

```bash
cmake --preset dev
cmake --build --preset dev -j8 --target test_prompt_cache astral_benchmarks
ctest --preset dev -R '^test_prompt_cache$' --output-on-failure
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=10000000 ./build/dev/benchmarks/astral_benchmarks --only features
```

Expected evidence markers include `test_prompt_cache Passed`,
`features.prompt_cache get`, `features.prompt_cache view`,
`features.prompt_cache miss`, and `features.agent prompt_cache_warmup`.
