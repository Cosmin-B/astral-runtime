# Native Chunking

Astral chunking turns UTF-8 text or token counts into stable ranges without
allocating per chunk. It is intended for RAG ingest, editor tooling, and engine
wrappers that need deterministic slices while keeping string materialization out
of the native hot path.

## C ABI

- `AstralChunkerDesc`
- `AstralChunkRange`
- `astral_chunk_count()`
- `astral_chunk_ranges()`
- `astral_chunk_text_copy()`
- `astral_token_chunk_count()`
- `astral_token_chunk_ranges()`

`AstralChunkerDesc::mode` selects the splitter:

- `ASTRAL_CHUNK_MODE_NONE`: one range for the full input.
- `ASTRAL_CHUNK_MODE_CHAR`: UTF-8 codepoint ranges.
- `ASTRAL_CHUNK_MODE_WORD`: ASCII whitespace-delimited word ranges.
- `ASTRAL_CHUNK_MODE_SENTENCE`: delimiter-terminated sentence ranges.
- `ASTRAL_CHUNK_MODE_TOKEN`: token ranges over already-tokenized input.

Sentence mode uses `.`, `!`, `?`, and newline when `delimiters` is empty. A
non-empty delimiter span replaces that default set. It does not extend it.

Text chunk ranges fill `byte_begin` and `byte_end`. Token chunk ranges fill
`token_begin` and `token_end`. The unused half of the range is set to a sentinel
value for text ranges and zero for token ranges.

Use `astral_memory_record_from_chunk()` to convert a chunk range into an
`AstralMemoryRecord` for RAG ingest. The helper preserves document id, chunk id,
and group id so native, Unreal, and Unity callers do not duplicate metadata
mapping code.
After search, callers can use the returned `chunk_id` to copy the selected
range with `astral_chunk_text_copy()` or to resolve text from their own document
store.

## Ownership

All range buffers are caller-owned. The native chunker does not retain the input
span and does not allocate while counting or writing ranges. `out_count` receives
the required range count, so callers can call `astral_chunk_count()` first or
retry `astral_chunk_ranges()` after `ASTRAL_E_NOMEM`.

`astral_chunk_text_copy()` is a convenience function for selected text ranges. It
copies into a caller-owned UTF-8 buffer and reports the required byte count.
Pass a null output span with zero length to size the selected range before
allocating or reusing a destination buffer.

## Unity

`AstralChunker` wraps the native API for Unity callers. The primary methods take
caller-owned `NativeArray<byte>` and `NativeArray<AstralChunkRange>` buffers, so
ingest code can count and emit chunk ranges without managed allocations.

Use `AstralChunker.TextDesc()` for UTF-8 text ranges and
`AstralChunker.TokenDesc()` for token-count planning. String overloads are
available for setup and editor tooling, but runtime ingest paths should keep text
as UTF-8 bytes and reuse output buffers.
`CountTextBytes()` uses the native size-only copy path before Unity allocates a
temporary string buffer for selected chunks.

## Performance

The range path scans the input span directly and emits POD records. Overlap is
handled by restarting at the overlapping unit boundary, so overlap should stay
small relative to `max_units`. Token count sizing is constant time. Token range
emission walks chunk boundaries only and does not touch token memory, which
keeps it suitable for chunk planning after tokenization.

## Example

```c
enum {
    kWordsPerChunk = 128,
    kWordOverlap = 16,
};

AstralChunkerDesc desc = {0};
desc.size = sizeof(AstralChunkerDesc);
desc.mode = ASTRAL_CHUNK_MODE_WORD;
desc.max_units = kWordsPerChunk;
desc.overlap_units = kWordOverlap;

uint32_t count = 0;
AstralErr err = astral_chunk_count(&desc, text, &count);
if (err == ASTRAL_OK) {
    AstralChunkRange* ranges = allocate_ranges(count);
    err = astral_chunk_ranges(&desc, text, ranges, count, &count);
}
```

## Validation

```bash
cmake --build --preset dev -j8 --target test_inference test_abi_invalid_args astral_benchmarks
ctest --preset dev -R '^(test_inference|test_abi_invalid_args|gate_abi_layout_report|gate_source_scans|gate_doc_links)$' --output-on-failure
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=200000 ./build/dev/benchmarks/astral_benchmarks --only features
```

Native tests include `inference_rag_ingest_chunk_search_mock`, which plans
document chunks, converts ranges to memory records, searches the flat index, and
copies the retrieved chunk text.
Benchmark output should include `features.chunk word_ranges`,
`features.chunk token_count`, and `features.chunk token_ranges`.
