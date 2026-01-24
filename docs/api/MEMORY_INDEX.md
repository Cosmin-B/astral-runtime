# Native Memory Index

Astral memory indexes store embedding vectors in native memory and return stable
keys and scores for retrieval. The flat index is the exact correctness baseline.
The graph index adds a bounded native ANN path for larger all-group searches.

## C ABI

- `AstralMemoryIndexDesc`
- `AstralMemoryRecord`
- `AstralMemorySearchDesc`
- `AstralMemorySearchResult`
- `astral_memory_create()`
- `astral_memory_destroy()`
- `astral_memory_add_batch()`
- `astral_memory_remove()`
- `astral_memory_clear()`
- `astral_memory_search()`
- `astral_memory_search_begin()`
- `astral_memory_search_fetch()`
- `astral_memory_search_end()`
- `astral_request_from_memory_search()`
- `astral_memory_save_size()`
- `astral_memory_save()`
- `astral_memory_load()`
- `astral_memory_record_from_chunk()`

The index is fixed-dimension and fixed-capacity. Add operations copy vectors
into native storage and use a bounded native key table for update/remove lookup.
Search returns result metadata first; callers decide whether to fetch text or
engine objects for the selected keys.

`AstralMemoryIndexDesc::index_kind` selects storage/search behavior:

- `ASTRAL_MEMORY_INDEX_FLAT` scans contiguous row-major vectors and returns exact
  top-k results.
- `ASTRAL_MEMORY_INDEX_GRAPH` builds a bounded adjacency graph during ingest and
  uses a fixed-size candidate pool for all-group top-k search. Set
  `graph_neighbors` and `graph_search` to tune recall/latency, or leave them
  zero for native defaults. Group-filtered searches use the exact flat scanner.

`astral_memory_record_from_chunk()` maps an `AstralChunkRange` into an
`AstralMemoryRecord` before `astral_memory_add_batch()`. It keeps document,
chunk, and group metadata consistent with the native chunker while the caller
supplies the stable record key and any application flags.

Incremental search cursors snapshot the top-k result set at begin time and let
callers fetch fixed-size batches without re-running the vector scan. The cursor
owns its compact native result buffer and must be released with
`astral_memory_search_end()`. Use `astral_request_from_memory_search()` when an
engine queue needs to track a cursor through the unified request status API.

## Metrics

- `ASTRAL_MEMORY_METRIC_DOT`: higher dot product ranks first.
- `ASTRAL_MEMORY_METRIC_COSINE`: vectors are stored with cached norms and scored
  against the query norm.
- `ASTRAL_MEMORY_METRIC_L2`: negative squared distance ranks first.

Use `ASTRAL_MEMORY_GROUP_ANY` to search all groups, or set
`AstralMemorySearchDesc::group_id` to restrict results.

## Performance

The flat index stores vectors row-major and uses AVX2 or NEON dot and L2 kernels
when the compiler target supports them. The scalar fallback is unrolled by four
lanes.
Search keeps the top-k result set in the caller-provided output array, avoiding
heap allocation during query execution. The flat scanner dispatches by metric
before entering the vector loop, so dot, cosine, and L2 searches do not branch
through a generic scorer for every stored vector.
Batch ingest uses the same fixed-capacity vector storage and a free-slot cursor
so sequential adds do not scan old slots to find the next open row.

The graph index allocates adjacency, candidate, and visited buffers at creation
time. Query execution reuses those buffers and the same SIMD scoring kernels.
Add/update/remove are colder ingest operations; updates and removals may rebuild
the graph to keep neighbor links consistent.

Feature benchmarks accept `ASTRAL_BENCH_MEMORY_CAPACITY`,
`ASTRAL_BENCH_MEMORY_DIM`, and `ASTRAL_BENCH_MEMORY_METRIC` (`cosine`, `dot`,
or `l2`) so local runs can cover 100, 1k, 10k, and 100k vector scans without
changing source. Set `ASTRAL_BENCH_MEMORY_SWEEP=1` to run the built-in
100/1k/10k/100k flat-index sweep in one invocation.

Unreal and Unity wrappers expose the same native descriptors and result records.
Wrapper arrays are converted at the engine boundary; the native index owns vector
storage and search ordering. Unreal's `FAstralMemoryIndexDesc` exposes flat and
graph index modes plus graph neighbor/search budgets; `0` keeps the native
defaults.

## Unity

`AstralMemoryIndex` owns the native index handle and releases it with
`Dispose()`. `AddBatch()`, `Search()`, and cursor `Fetch()` operate on
caller-owned `NativeArray` buffers, which keeps vectors and result records out of
managed collections during retrieval.

Use `AstralMemoryIndex.Record()` to create records with the ABI size field set,
and `AstralMemoryIndex.SearchDesc()` for search descriptors. `Save()` and
`Load()` are snapshot helpers for editor tooling and staged RAG data.

## Example

```c
enum {
    kDim = 384,
    kCapacity = 4096,
    kTopK = 8,
};

AstralMemoryIndexDesc desc = {0};
desc.size = sizeof(AstralMemoryIndexDesc);
desc.dim = kDim;
desc.capacity = kCapacity;
desc.metric = ASTRAL_MEMORY_METRIC_COSINE;
desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

AstralHandle index = 0;
AstralErr err = astral_memory_create(&desc, &index);

AstralMemorySearchDesc search = {0};
search.size = sizeof(AstralMemorySearchDesc);
search.top_k = kTopK;
search.group_id = ASTRAL_MEMORY_GROUP_ANY;
```

Graph index:

```c
desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
desc.graph_neighbors = 16;
desc.graph_search = 64;
err = astral_memory_create(&desc, &index);
```

## Chunked Document Ingest

For document-backed retrieval, keep text storage outside the memory index and
store only stable metadata in native records:

1. Use `astral_chunk_count()` and `astral_chunk_ranges()` on the source UTF-8
   document.
2. Convert each selected `AstralChunkRange` with
   `astral_memory_record_from_chunk()`.
3. Embed each chunk through the embedding API or a caller-owned embedding
   pipeline.
4. Add records and vectors with one `astral_memory_add_batch()` call.
5. Search returns keys, scores, document id, chunk id, and group id. Use the
   returned chunk id to fetch text with `astral_chunk_text_copy()` or with your
   own document store.

Native tests cover that path in `inference_rag_ingest_chunk_search_mock`.

## Validation

```bash
cmake --build --preset dev -j8 --target test_inference test_abi_invalid_args astral_benchmarks
ctest --preset dev -R '^(test_inference|test_abi_invalid_args|gate_abi_layout_report|gate_source_scans|gate_doc_links)$' --output-on-failure
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=200000 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=1000 ASTRAL_BENCH_MEMORY_CAPACITY=100000 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=1000 ASTRAL_BENCH_MEMORY_SWEEP=1 ./build/dev/benchmarks/astral_benchmarks --only features
```

Native tests include `inference_memory_index_graph_mock` for graph search,
filtered exact fallback, save/load, and remove/rebuild behavior.

Expected markers include `features.memory add_batch`,
`features.memory flat_search_top1`, `features.memory flat_search`,
`features.memory graph_search`, and `features.memory cursor_begin_fetch`. Sweep runs also include
`features.memory top1_100`, `features.memory top1_1k`,
`features.memory top1_10k`, and `features.memory top1_100k`.
