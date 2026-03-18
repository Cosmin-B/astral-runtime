# Native Memory Index

Astral memory indexes store embedding vectors in native memory and return stable
keys and scores for retrieval. The flat index is the exact correctness baseline.
The graph index is a bounded native candidate graph for larger all-group
searches. Treat it as an approximate index: tune it against the target dataset
and keep the flat index as the recall oracle.

## C ABI

- `AstralMemoryIndexDesc`
- `AstralMemoryRecord`
- `AstralMemorySearchDesc`
- `AstralMemorySearchResult`
- `AstralMemoryStats`
- `astral_memory_create()`
- `astral_memory_destroy()`
- `astral_memory_count()`
- `astral_memory_stats()`
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
  uses fixed-size frontier and top-candidate pools for all-group top-1 and top-k
  search.
  Set `graph_neighbors` and `graph_search` to tune recall/latency, or leave
  them zero for native defaults. Group-filtered searches use the exact flat
  scanner. Use the graph recall benchmark before choosing this path for
  retrieval. Graph construction assigns deterministic upper levels from each
  record key, descends those levels before base-layer expansion, and keeps a few
  deterministic spread links at the base layer to avoid purely local
  neighborhoods.

Choose the flat index when exact ranking, filtered search, or small-to-medium
corpora matter more than avoiding a full scan. Choose the graph index only after
measuring recall on the target vectors. Higher `graph_neighbors` and
`graph_search` values usually improve recall, but they also increase ingest
cost, memory traffic, and query latency; high-recall graph settings can be
slower than exact flat search on smaller collections.

`astral_memory_record_from_chunk()` maps an `AstralChunkRange` into an
`AstralMemoryRecord` before `astral_memory_add_batch()`. It keeps document,
chunk, and group metadata consistent with the native chunker while the caller
supplies the stable record key and any application flags.

Incremental search cursors snapshot the top-k result set at begin time and let
callers fetch fixed-size batches without re-running the vector scan. The cursor
owns its compact native result buffer and must be released with
`astral_memory_search_end()`. Use `astral_request_from_memory_search()` when an
engine queue needs to track a cursor through the unified request status API.
Canceling a memory-search request marks the cursor canceled and clears its
remaining result depth; it does not release the cursor handle.

`astral_memory_stats()` reports the current index shape and byte footprint for
capacity sizing. `vector_bytes` covers row-major vector storage,
`metadata_bytes` covers native index metadata, slots, active-slot storage, and
the key table, and `graph_bytes` covers graph-only adjacency, frontier,
scratch, level, and visited buffers. `total_bytes` is the runtime footprint
sum, while `save_bytes` matches the current serialized snapshot size returned
by `astral_memory_save_size()`.

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

The graph index allocates adjacency, level, frontier, top-candidate, and visited
buffers at creation time. Query execution reuses those buffers and the same SIMD
scoring kernels. Graph search first performs greedy upper-level routing, then
expands the base layer until the best frontier candidate can no longer improve
the bounded top-candidate pool. Add/update/remove are colder ingest operations;
updates and removals may rebuild the graph to keep neighbor links consistent.
Treat flat search as the recall oracle when tuning graph search. The
`features.memory graph_recall` benchmark reports aggregate top-k overlap
between graph search and exact flat search across deterministic, high-entropy
recall queries spread across the indexed capacity, so graph improvements can be
judged by both latency and recall. Do not select a graph configuration solely
because it is faster than flat search; use the recall target for the dataset
and embedding model as the acceptance check.
`features.memory graph_recall_search` precomputes the exact flat oracle outside
the timed region, then reports graph-only query latency with the same recall
percentage. Use that marker for ANN perf counters when exact-search work would
otherwise dominate the profile.

Feature benchmarks accept `ASTRAL_BENCH_MEMORY_CAPACITY`,
`ASTRAL_BENCH_MEMORY_DIM`, `ASTRAL_BENCH_MEMORY_METRIC` (`cosine`, `dot`, or
`l2`), `ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS` up to the native graph-neighbor
limit, and `ASTRAL_BENCH_MEMORY_GRAPH_SEARCH`. Set
`ASTRAL_BENCH_MEMORY_RECALL_QUERIES` to choose how many deterministic query
vectors are rotated through the graph recall benchmark. These controls let local
runs cover vector scans and graph recall/latency tuning without changing source.
Set
`ASTRAL_BENCH_MEMORY_SWEEP=1` to run the built-in 100/1k/10k/100k flat-index
sweep in one invocation. Set
`ASTRAL_BENCH_MEMORY_ONLY=1` when collecting hardware counters so tokenization,
prompt cache, tool, chunk, and agent benchmarks do not dilute the memory-index
counter profile. Use `scripts/run_memory_bench_matrix.sh` to run the
memory-only benchmark across multiple metrics, dimensions, and capacities in one
log.

For release tuning, capture both `features.memory flat_search_top1` and
`features.memory graph_recall` for the same dimension, metric, capacity,
neighbor count, and search budget. A graph run is useful only when its recall
meets the product target and its latency beats the exact flat baseline for that
dataset. Keep the flat index available as the correctness oracle while tuning
new embedding models or document distributions.

Unreal and Unity wrappers expose the same native descriptors and result records.
Wrapper arrays are converted at the engine boundary; the native index owns vector
storage and search ordering. Unreal's `FAstralMemoryIndexDesc` exposes flat and
graph index modes plus graph neighbor/search budgets; `0` keeps the native
defaults. `FAstralMemoryStats` and Unity `AstralMemoryIndex.GetStats()` expose
the same native footprint counters for engine-side budgeting and telemetry.

## Unity

`AstralMemoryIndex` owns the native index handle and releases it with
`Dispose()`. `AddBatch()`, `Search()`, and cursor `Fetch()` operate on
caller-owned `NativeArray` buffers, which keeps vectors and result records out of
managed collections during retrieval.

Use `AstralMemoryIndex.Record()` to create records with the ABI size field set,
and `AstralMemoryIndex.SearchDesc()` for search descriptors. `Save()` and
`Load()` are snapshot helpers for editor tooling and staged RAG data.
`AstralRequest.FromMemorySearch(cursor)` wraps a search cursor for queue-depth
polling without exposing raw handles to gameplay code.

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

AstralMemoryStats stats = {0};
stats.size = sizeof(AstralMemoryStats);
err = astral_memory_stats(index, &stats);

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
6. For agents, `astral_agent_set_memory_context_from_results()` can copy the
   selected chunk byte ranges into native agent memory in search-result order.

Native tests cover that path in `inference_rag_ingest_chunk_search_mock`.

## Validation

```bash
cmake --build --preset dev -j8 --target test_inference test_abi_invalid_args astral_benchmarks
ctest --preset dev -R '^(test_inference|test_abi_invalid_args|gate_abi_layout_report|gate_source_scans|gate_doc_links)$' --output-on-failure
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=200000 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=1000 ASTRAL_BENCH_MEMORY_CAPACITY=100000 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=1000 ASTRAL_BENCH_MEMORY_SWEEP=1 ./build/dev/benchmarks/astral_benchmarks --only features
perf stat -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses -- env ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=10 ASTRAL_BENCH_MEMORY_SWEEP=1 ASTRAL_BENCH_MEMORY_DIM=384 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=64 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=256 ASTRAL_BENCH_MEMORY_RECALL_QUERIES=64 ./build/dev/benchmarks/astral_benchmarks --only features
scripts/run_memory_bench_matrix.sh --preset dev --dims 128,384,768 --capacities 10000 --metrics cosine,dot,l2 --out /tmp/astral-memory-matrix.txt
```

Native tests include `inference_memory_index_graph_mock` for graph search,
filtered exact fallback, save/load, and remove/rebuild behavior.

Expected markers include `features.memory add_batch`,
`features.memory graph_add_batch`,
`features.memory flat_search_top1`, `features.memory flat_search`,
`features.memory graph_top1`, `features.memory graph_search`,
`features.memory graph_recall`, `features.memory graph_recall_search`, and
`features.memory cursor_begin_fetch`.
Sweep runs also include
`features.memory top1_100`, `features.memory top1_1k`,
`features.memory top1_10k`, and `features.memory top1_100k`.
