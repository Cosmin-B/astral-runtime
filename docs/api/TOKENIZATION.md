# Tokenization API

Astral exposes model-scoped tokenization through the C ABI. Native callers pass
UTF-8 byte spans and caller-owned token or byte buffers. Unreal and Unity
wrappers only convert engine strings and arrays around the same native calls.

## Entry Points

- `astral_tokenize_count(model, text, add_special, parse_special, out_count)`
  returns the token count for a UTF-8 span without writing token ids.
- `astral_tokenize(model, text, out_tokens, max_tokens, add_special,
  parse_special, out_count)` writes one span into a caller-owned token buffer.
- `astral_tokenize_batch(model, requests, request_count, out_offsets,
  out_tokens, max_tokens, out_count)` tokenizes many spans into one token
  buffer. `out_offsets` has `request_count + 1` entries.
- `astral_detokenize_count(model, tokens, count, out_len)` returns the UTF-8
  byte count needed for token ids.
- `astral_detokenize(model, tokens, count, out_text, out_len)` writes UTF-8
  bytes into a caller-owned buffer.

Unity exposes the same flow on `AstralModel`: `CountTokens()` for sizing,
`Tokenize()` for caller-owned or newly allocated `NativeArray<int>` output, and
`Detokenize()` for managed string conversion outside hot paths.
`CountTokensBatch()` and `TokenizeBatch()` take caller-owned
`NativeArray<AstralTokenizeRequest>` and offset buffers for many UTF-8 spans in
one native call.

Unreal exposes compatibility bool helpers on `UAstralModel` plus
`CountTokensResult()`, `TokenizeResult()`, and `DetokenizeResult()` for
Blueprint graphs that need the native error code and output count.

## Ownership And Lifetime

The model handle owns tokenizer metadata. Text spans, token arrays, offsets, and
output byte buffers are caller-owned and only need to remain valid for the call.
The functions are safe to call from multiple threads on the same model when the
backend tokenizer is thread-safe, which is part of the Astral backend contract.

## Performance Model

Sizing calls let callers allocate once. Batch tokenization has two paths:
passing `NULL` tokens computes offsets and total count only, while passing a
token buffer writes each request directly into the supplied storage and still
returns the full offsets and required total on `ASTRAL_E_NOMEM`. The core API
does not allocate per string or per token. Public ABI functions validate
handles and buffer pointers; provider hot paths receive caller-owned output
spans.

## Minimal Example

```c
uint32_t count = 0;
AstralErr err = astral_tokenize_count(model, text, 1, 0, &count);
if (err != ASTRAL_OK) return err;

int32_t* tokens = allocate_tokens(count);
err = astral_tokenize(model, text, tokens, count, 1, 0, &count);
```

For batch sizing:

```c
AstralTokenizeRequest reqs[2] = {0};
reqs[0].text = first;
reqs[1].text = second;

uint32_t offsets[3] = {0};
uint32_t total = 0;
AstralErr err = astral_tokenize_batch(model, reqs, 2, offsets, NULL, 0, &total);
```

## Validation

```bash
cmake --preset dev
cmake --build --preset dev -j8 --target test_tokenization test_abi_invalid_args
ctest --preset dev -R '^(test_tokenization|test_abi_invalid_args|gate_unreal_header_mirror|gate_source_scans)$' --output-on-failure
ASTRAL_BENCH_TOKENIZE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=1000 ASTRAL_BENCH_MODEL=tests/models/gpt2.Q2_K.gguf ./build/dev/benchmarks/astral_benchmarks --only features
scripts/run_feature_bench_suite.sh --preset dev --models-dir tests/models --tokenize-only --out /tmp/astral-tokenizers.txt
```

Expected evidence markers include `test_tokenization Passed` and
`gate_unreal_header_mirror Passed`. Tokenization benchmark markers are
`features.tokenize count` and `features.tokenize batch`.
