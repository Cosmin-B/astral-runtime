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
- `AstralMemoryStorageKind`
- `astral_memory_create()`
- `astral_memory_destroy()`
- `astral_memory_count()`
- `astral_memory_stats()`
- `astral_memory_get_record()`
- `astral_memory_update_record()`
- `astral_memory_add_batch()`
- `astral_memory_remove()`
- `astral_memory_clear()`
- `astral_memory_search()`
- `astral_memory_search_batch()`
- `astral_memory_search_begin()`
- `astral_memory_search_fetch()`
- `astral_memory_search_end()`
- `astral_request_from_memory_search()`
- `astral_memory_save_size()`
- `astral_memory_save()`
- `astral_memory_snapshot_info()`
- `astral_memory_load()`
- `astral_memory_record_from_chunk()`

The index is fixed-dimension and fixed-capacity. Add operations copy vectors
into native storage and use a bounded native key table for update/remove lookup.
Search returns result metadata first; callers decide whether to fetch text or
engine objects for the selected keys.
`astral_memory_search_batch()` accepts a row-major query matrix and writes each
query's results into a fixed `top_k` stride in the output array. `out_counts[i]`
reports how many results were written for query `i`, so callers can batch RAG
lookups without repeated ABI crossings while still using caller-owned buffers.
Flat batch search processes small query chunks against the vector table in one
pass per chunk, so multi-query RAG can reuse cache-hot vector rows instead of
re-running a full scan for every query.

Single-query flat searches above 32,768 records and through top-k 10 can shard
contiguous record ranges when the runtime has at least one worker. Each worker
and the caller keep a private bounded top-k; the caller merges only those final
candidates, so the vector scan does not share a heap or perform atomic updates.
`AstralInit::thread_count` is the number of worker threads; the API caller is
separate. For a latency test restricted to four CPU cores, use three workers so
the caller does not oversubscribe the core set.

Dense pure-E5M2 dot and cosine shards choose their storage and metric path once
before scanning. The common 384-dimensional x86 path uses a fixed-bound F16C
conversion/dot kernel. ARM64 uses native NEON FP16 conversion when available
and NEON integer bit construction on baseline ARMv8 targets.

`AstralMemoryIndexDesc::index_kind` selects storage/search behavior:

- `ASTRAL_MEMORY_INDEX_FLAT` scans contiguous row-major vectors and returns exact
  top-k results.
- `ASTRAL_MEMORY_INDEX_GRAPH` builds a bounded adjacency graph during ingest and
  uses fixed-size frontier and top-candidate pools for all-group top-1 and top-k
  search.
  Set `graph_neighbors`, `graph_search`, and `graph_query_search` to tune
  recall/latency, or leave them zero for the native defaults. `graph_search`
  controls construction expansion. `graph_query_search` controls the index
  default query expansion; zero uses `graph_search`. Group-filtered searches use
  the exact flat scanner. Use the graph recall benchmark before choosing this
  path for retrieval. Graph construction assigns deterministic upper levels from
  each record key, descends those levels before base-layer expansion, selects a
  diverse bounded neighbor set from the construction candidate pool, and
  accepts reverse links into the wider base-layer capacity.

`AstralMemoryIndexDesc::storage_kind` selects vector storage:

- `ASTRAL_MEMORY_STORAGE_F32` is the default exact float32 storage.
- `ASTRAL_MEMORY_STORAGE_Q8` stores each vector as signed 8-bit values plus one
  per-vector scale. Add/load accept float32 vectors. Save uses a versioned
  snapshot: q8 indexes write the per-vector scale and signed bytes, while older
  float32 snapshots remain readable. Q8 search is approximate and is intended
  for memory-footprint-sensitive retrieval. Graph q8 uses q8 pair scoring
  during construction and q8-vs-f32 scoring for queries, so measure recall
  against the flat f32 oracle before using it for retrieval.
- `ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK` stores the q8 vector block plus a hidden
  float32 rerank block. Graph construction keeps q8 routing behavior; result
  scoring uses the float32 block so the final ordering is exact within the
  candidate set. Use it when compact routing helps memory locality but the
  returned scores need f32 fidelity.
- `ASTRAL_MEMORY_STORAGE_F6_E2M3` stores scaled E2M3 values in signed bytes and
  uses the same compact integer dot path as q8, with Float6 scaling applied once
  at the score boundary. It is approximate; validate recall against flat f32 on
  the target embeddings before using it for retrieval.
- `ASTRAL_MEMORY_STORAGE_F6_E3M2` stores scaled E3M2 values in 16-bit lanes.
  It keeps a wider Float6 dynamic range than E2M3 while still using contiguous
  compact storage and integer dot paths. It is approximate; validate recall and
  latency against flat f32 on the target embeddings before using it for
  retrieval.
- `ASTRAL_MEMORY_STORAGE_F8_E5M2` stores scaled E5M2 bytes. It trades q8's
  integer dot path for wider dynamic range and compact snapshot/search support;
  validate recall and latency against flat f32 before choosing it for retrieval.
- `ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK` stores the E5M2 byte block plus a
  hidden float32 rerank block. Graph construction, graph routing, final result
  ordering, and snapshot scoring use the f32 block; the E5M2 block is retained
  for compact snapshot payloads and storage inspection.

Graph snapshots include the routing topology when the load descriptor matches
the saved graph neighbor, search, and level capacities. If those knobs differ,
load keeps the records and vectors and rebuilds the graph for the requested
shape. Use `features.memory graph_load` beside `features.memory graph_add_batch`
when deciding whether to ship a prebuilt graph snapshot or rebuild at startup.

Choose the flat index when exact ranking, filtered search, or small-to-medium
corpora matter more than avoiding a full scan. Choose the graph index only after
measuring recall on the target vectors. Higher `graph_neighbors`,
`graph_search`, and `graph_query_search` values usually improve recall, but they
also increase ingest cost, memory traffic, and query latency; high-recall graph
settings can be slower than exact flat search on smaller collections.

`astral_memory_record_from_chunk()` maps an `AstralChunkRange` into an
`AstralMemoryRecord` before `astral_memory_add_batch()`. It keeps document,
chunk, and group metadata consistent with the native chunker while the caller
supplies the stable record key and any application flags.
`astral_memory_get_record()` fetches stored metadata by key without running a
vector search. Missing keys return `ASTRAL_E_NOT_FOUND`.

`astral_memory_add_batch()` validates the complete batch before publishing any
record. Invalid metadata, capacity failure, allocation failure, or non-finite
vector values leave the index unchanged. Repeated keys in one batch resolve to
the last record and vector supplied for that key. Search queries also reject
non-finite values.
`astral_memory_update_record()` changes stored metadata without touching the
vector payload. If the new record has a different key, the key table is renamed
and graph indexes rebuild their routing links.

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
scratch, level, and visited buffers. Graph indexes also report
`graph_edges`, split into `graph_base_edges` and `graph_upper_edges`, so
callers can inspect how dense the stored routing graph became after ingest or
load. `graph_build_score_evals` and `graph_build_candidate_visits` report
cold-path construction work since the current graph topology was built, which
helps compare ANN tuning changes against build cost. `total_bytes` is the
runtime footprint sum, while `save_bytes` matches the current serialized
snapshot size returned by `astral_memory_save_size()`. For compact indexes,
`save_bytes` reflects the compact snapshot rather than an expanded float32
copy.

Graph indexes store up to `graph_neighbors` links on upper layers and up to
`2 * graph_neighbors` links on the base layer, capped by the native graph
limits. Use higher neighbor counts only when recall data justifies the extra
base-layer memory and search cost.

`astral_memory_snapshot_info()` validates a saved snapshot without creating an
index and reports the record, scale, vector, and graph byte ranges. New
snapshots store records, compact scales, and vectors in separate contiguous
blocks so callers can inspect or memory-map vector payloads before deciding
whether to load the full native index. Older snapshots still load through
`astral_memory_load()`, but their reported strides describe the legacy
interleaved layout.

`astral_memory_snapshot_search()` runs an exact flat scan directly over a saved
snapshot span. It is intended for read-only memory-mapped snapshots and oracle
checks: callers can search staged f32, q8, f6, or f8 snapshots without allocating a
native index handle or copying vector storage first. Graph topology in the
snapshot is ignored by this path; use `astral_memory_load()` when you need the
saved graph routing structure.

`astral_memory_snapshot_map()` opens a snapshot file as a read-only mapped
snapshot view, validates it, and returns an `AstralHandle`. Use
`astral_memory_snapshot_view_search()` to search the mapped bytes repeatedly
without copying them into a native index, `astral_memory_snapshot_view_info()`
to read the validated layout, and `astral_memory_snapshot_unmap()` to close the
view. The view path reuses the layout validated when the file was mapped.
Mapped graph snapshots also reuse the saved topology for all-group searches;
group-filtered searches fall back to exact mapped flat scan because the saved
graph is built across the full corpus. Graph snapshots persist construction and
default query expansion separately; mapped graph views use the saved query
expansion unless `AstralMemorySearchDesc::graph_search` overrides it.

## Metrics

- `ASTRAL_MEMORY_METRIC_DOT`: higher dot product ranks first.
- `ASTRAL_MEMORY_METRIC_COSINE`: vectors are stored with cached norms and scored
  against the query norm.
- `ASTRAL_MEMORY_METRIC_L2`: negative squared distance ranks first.

Use `ASTRAL_MEMORY_GROUP_ANY` to search all groups, or set
`AstralMemorySearchDesc::group_id` to restrict results.
For graph indexes, `AstralMemorySearchDesc::graph_search` can override the
search budget for one query without rebuilding the index. Leave it zero to use
`AstralMemoryIndexDesc::graph_query_search`. The value is clamped to
preallocated graph scratch, so create the index with enough construction or
default query expansion for the largest per-query budget you plan to test.

## Search Selection

Start with the flat index. It is the recall oracle, supports every metric and
group filter, and has the simplest operating model. Move away from it only when
the target corpus and latency budget justify the extra tuning work.

| Mode | Use when | Measure before shipping |
| --- | --- | --- |
| Flat f32 | Exact ranking matters, filtered search is common, or the corpus is small enough for full scans. | `flat_search_top1`, `flat_search`, memory bandwidth counters, and `astral_memory_stats()`. |
| Flat compact | The corpus is memory-bandwidth-bound and approximate scores are acceptable. | `flat_compact_recall_search` versus flat f32 on the target vectors, compact ingest cost, `flat_search_top1`, and `vector_bytes`. |
| Graph f32 | All-group approximate search needs lower latency than a full scan. | `graph_recall_search` against flat f32, `graph_top1`, `graph_add_batch`, `graph_load`, and the chosen `graph_neighbors`/`graph_search`/`graph_query_search` settings. |
| Graph compact | Approximate graph search also needs lower vector footprint. | The graph f32 measurements above, plus compact recall drop, graph build cost, and `graph_bytes`/`vector_bytes`. |
| Graph q8+f32 rerank | Compact graph routing is useful but final score ordering needs f32 fidelity. | The graph compact measurements above, plus q8+f32 `graph_recall_search` and `graph_add_batch` at the target capacity. For 100k x 384 cosine vectors, `graph_neighbors=64`, `graph_search=128`, and `graph_query_search=1536` is the current high-recall starting point. |
| Graph E5M2+f32 rerank | E5M2 snapshot payloads are useful but graph quality and final ordering need f32 fidelity. | `graph_recall_search`, `graph_snapshot_view_recall_search`, `graph_add_batch`, and `graph_load` with `ASTRAL_BENCH_MEMORY_STORAGE=f8e5m2f32` at the target capacity. Compare against q8+f32 before making it the default. |

Do not use the graph index for group-filtered retrieval unless the group is
large enough to justify all-group search followed by application filtering.
Native group-filtered graph searches use the exact scanner because sparse group
filters usually destroy graph locality.

For release candidates, capture one exact flat run and one candidate run with
the same dimension, metric, capacity, query set, and corpus. A graph or compact
configuration is ready only when it meets the product recall target and beats
the exact baseline on the deployment hardware. Keep the exact flat path
available in tooling and tests so recall can be rechecked when the embedding
model or document distribution changes.

## Performance

The flat index stores vectors row-major and uses AVX2 or NEON dot and L2 kernels
when the compiler target supports them. The scalar fallback is unrolled by four
lanes.
Search keeps the top-k result set in the caller-provided output array, avoiding
heap allocation during query execution. The flat scanner dispatches by metric
before entering the vector loop, so dot, cosine, and L2 searches do not branch
through a generic scorer for every stored vector.
Batch flat search precomputes query scales for a bounded stack chunk, scans the
stored rows once per chunk, and writes each query's top-k set into its fixed
output stride.
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
`features.memory graph_recall_top1` reports exact top-1 matches against the
flat oracle, which is useful when tuning search that only needs the best hit.
`features.memory graph_recall_search` precomputes the exact flat oracle outside
the timed region, then reports graph-only query latency with the same recall
percentage. Use that marker for ANN perf counters when exact-search work would
otherwise dominate the profile. `features.memory graph_recall_latency` uses the
same deterministic query preparation and reports p50/p95/p99 for the graph-only
search calls. Aggregate recall lanes must sample the full deterministic query
set before their percentage is meaningful. The matrix runner records both
requested `iters` and `effective_iters`; for aggregate recall cases it raises
`effective_iters` to at least `ASTRAL_BENCH_MEMORY_RECALL_QUERIES`.

Feature benchmarks accept `ASTRAL_BENCH_MEMORY_CAPACITY`,
`ASTRAL_BENCH_MEMORY_DIM`, `ASTRAL_BENCH_MEMORY_METRIC` (`cosine`, `dot`, or
`l2`), `ASTRAL_BENCH_MEMORY_STORAGE` (`f32`, `q8`, `q8f32`, `f6e2m3`,
`f6e3m2`, or `f8e5m2`),
`ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS` up to the native graph-neighbor limit,
and `ASTRAL_BENCH_MEMORY_GRAPH_SEARCH`. Set
`ASTRAL_BENCH_MEMORY_GRAPH_QUERY_SEARCH` to time a lower per-query budget
against an index built with a larger graph search capacity. Set
`ASTRAL_BENCH_MEMORY_RECALL_QUERIES` to choose how many deterministic query
vectors are rotated through the graph recall benchmark. These controls let local
runs cover vector scans and graph recall/latency tuning without changing source.
Set `ASTRAL_BENCH_MEMORY_CASE` to one case name, such as
`flat_search_batch`, `flat_q8_recall_search`, or `graph_recall_search`, when
collecting a narrow profile and avoiding unrelated index rebuilds.
Set
`ASTRAL_BENCH_MEMORY_SWEEP=1` to run the built-in 100/1k/10k/100k flat-index
sweep in one invocation. Set
`ASTRAL_BENCH_MEMORY_ONLY=1` when collecting hardware counters so tokenization,
prompt cache, tool, chunk, and agent benchmarks do not dilute the memory-index
counter profile. Use `scripts/run_memory_bench_matrix.sh` to run the
memory-only benchmark across multiple metrics, dimensions, and capacities in one
log. Use `scripts/run_memory_search_acceptance.sh` to capture the common exact
flat, exact flat latency, reduced flat, q8 recall, graph build cost, graph
snapshot load cost, latency, f32/q8 graph recall, and f32/q8 graph top-1 recall lanes into one
output directory for a single dataset shape. Recall lanes run with an effective
iteration count no smaller than the requested recall-query count, even when the
requested iteration count is lower. Add `--budget-sweep` when tuning
graph recall/latency so the runner also captures `graph_recall_search_sweep`
for f32 and q8 graph indexes using one build and multiple per-query budgets.
Add `--add-latency` when tuning insertion so it also captures
`features.memory graph_add_latency` p50/p95/p99 lanes for graph storage
profiles.
Use the `graph_search_batch` benchmark case with
`ASTRAL_BENCH_RUNTIME_THREADS` greater than 1 to measure the per-worker graph
scratch search path.
Add `--recall-detail` when aggregate recall hides hard queries; it captures one
`features.memory graph_recall_qNNN_rowMMMMMM` lane per deterministic query row
with the same graph and oracle setup, followed by a
`features.memory graph_recall_match_qNNN_rowMMMMMM` row that reports the matched
top-k count. Add `--level-stats` to capture deterministic upper level node
counts for the same capacity and neighbor budget. Add `--edge-stats` to capture
stored total, base-layer, upper-layer, and graph construction-work counts
through the public memory stats API. Both runners can wrap each
benchmark lane in `perf stat` with `--perf`, `--perf-bin`, `--perf-events`, and
`--require-perf`.
Perf CSV and stderr files are written beside the runner logs, so hardware
counter evidence stays with the sidecar capture instead of entering the
repository.
`scripts/run_memory_bench_matrix.sh` also writes a compact CSV summary next to
`.txt` logs by default, or to the path passed with `--summary`. The summary
keeps one row per metric/storage/dimension/capacity/query-budget run and folds
in useful perf ratios when counter CSVs are present: IPC, branch miss rate,
cache miss rate, LLC load miss rate, L1D load miss rate, and DTLB load miss
rate. Use it for quick sweeps; keep the raw log and perf CSVs for deeper
inspection.
Use `scripts/run_memory_ann_tuning.sh` when comparing several graph shapes. It
wraps the acceptance runner across neighbor, build-search, and per-query search
budgets, then writes one roll-up summary with f32/q8 recall, top-1 recall,
build, optional insert-latency, load, level, and edge-count markers for every
shape. It also writes `results.csv` with one f32 and one q8 row per graph shape so large sweeps can be
sorted without manually parsing each lane log. The CSV records both the
requested per-query budget and the effective budget after clamping to the graph
build/search capacity, plus edge counts and graph construction-work counts, so
rows with a larger requested query budget are not mistaken for distinct runtime
work and build-cost changes remain visible. It also records the first exact
flat f32 latency baseline and each graph row's speedup against that baseline.
When multiple requested budgets clamp to the same effective budget for a graph
shape, the runner reuses the first capture and still writes one CSV row per
requested budget. The first captured shape includes the exact flat baseline
lanes; later graph shapes skip those duplicate flat lanes because the flat
baseline only depends on the dataset shape, metric, and storage. Flat batch
search can split query rows across the runtime worker pool for cache-resident
corpora when it is already initialized with multiple workers; larger corpora
use record-range sharding for f32 batches with small top-k, followed by a local
top-k merge. Large single-query f32 searches use the same record-range sharding
when the runtime worker pool is available. Graph batch search stays serial until
graph query scratch and visit state are per-query. The summary also includes
compact best-row sections sorted by recall first and latency second, including a
high-recall section for rows at or above 95% recall. Use `--summary-only` with
`--out-dir <existing-dir>` to write `summary_best.txt` from an existing
`results.csv` without rerunning benchmark lanes or overwriting the full
per-shape summary. The default output directory is under `/tmp`; pass
`--out-dir` to place the capture in a sidecar evidence folder.
Use `--min-recall-pct` and `--max-recall-ns` when a tuning run should fail
unless at least one row meets the chosen recall and latency envelope. The same
threshold path also accepts `--min-speedup-vs-flat` for the graph-vs-exact
baseline requirement. These checks work with `--summary-only`, so existing CSV
captures can be checked without rerunning the benchmark lanes.

For release tuning, capture `features.memory flat_search_top1`,
`features.memory flat_search_batch`, `features.memory flat_search_latency`,
`features.memory flat_q8_recall_search`, `features.memory graph_add_batch`,
`features.memory graph_add_latency`, `features.memory graph_load`,
`features.memory snapshot_search`,
`features.memory snapshot_view_search`,
`features.memory graph_snapshot_view_search`, `features.memory graph_search_latency`,
`features.memory graph_recall_search`,
`features.memory graph_snapshot_view_recall_search`, and
`features.memory graph_recall_top1`
for the same dimension, metric,
capacity, neighbor count, and search budget. A graph run is useful only when its
recall meets the product target, its latency beats the exact flat baseline, and
its ingest cost is acceptable for that dataset. Keep the flat index available as
the correctness oracle while tuning new embedding models or document
distributions.

The required release recall lane uses 10,000 deterministic 384-dimensional
cosine vectors, 32 recall queries, and build/query search budgets of 256. It
requires at least 99% aggregate top-k recall for `f32`, `q8`, `q8f32`,
`f6e3m2f32`, and `f8e5m2f32`. Raw six-bit and E5M2 graph storage profiles are
capacity-oriented alternatives and are not part of that high-recall set.
When aggregate top-k recall moves in the right direction but remains unstable,
run `graph_recall_detail` to see whether failures are isolated to a few query
regions or spread across the corpus. Run `graph_level_stats` beside large graph
tuning when changing neighbor budgets, since very sparse upper levels reduce
the value of greedy routing before base-layer expansion. Run `graph_edge_stats`
when changing insertion or pruning behavior so recall changes can be compared
against the actual stored edge count.

Unreal and Unity wrappers expose the same native descriptors and result records.
Wrapper arrays are converted at the engine boundary; the native index owns vector
storage and search ordering. Unreal's `FAstralMemoryIndexDesc` exposes flat and
graph index modes plus graph neighbor/search budgets; `0` keeps the native
defaults. Unreal `SearchMemoryIndexBatchResult()` and Unity
`AstralMemoryIndex.SearchBatch()` call the native row-major batch search path
for multi-query retrieval. `FAstralMemoryStats` and Unity
`AstralMemoryIndex.GetStats()` expose the same native footprint and graph-edge
counters for engine-side budgeting and telemetry.

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
desc.storage_kind = ASTRAL_MEMORY_STORAGE_F32;

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
desc.storage_kind = ASTRAL_MEMORY_STORAGE_F32;
desc.graph_neighbors = 32;
desc.graph_search = 64;
desc.graph_query_search = 512;
err = astral_memory_create(&desc, &index);
```

Reduced flat storage:

```c
desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
desc.storage_kind = ASTRAL_MEMORY_STORAGE_Q8;
err = astral_memory_create(&desc, &index);
```

Reduced graph storage:

```c
desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
desc.storage_kind = ASTRAL_MEMORY_STORAGE_Q8;
desc.graph_neighbors = 32;
desc.graph_search = 64;
desc.graph_query_search = 512;
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
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_recall_search ASTRAL_BENCH_FEATURE_ITERS=64 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_load ASTRAL_BENCH_FEATURE_ITERS=8 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=256 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=flat_search_batch ASTRAL_BENCH_FEATURE_ITERS=64 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=flat_search_latency ASTRAL_BENCH_FEATURE_ITERS=256 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=flat_q8_recall_search ASTRAL_BENCH_FEATURE_ITERS=64 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_search_latency ASTRAL_BENCH_FEATURE_ITERS=256 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_recall_search_sweep ASTRAL_BENCH_FEATURE_ITERS=64 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=512 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_recall_search ASTRAL_BENCH_FEATURE_ITERS=64 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=256 ASTRAL_BENCH_MEMORY_GRAPH_QUERY_SEARCH=64 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_recall_detail ASTRAL_BENCH_FEATURE_ITERS=32 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_RECALL_QUERIES=16 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_level_stats ASTRAL_BENCH_MEMORY_CAPACITY=100000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS=32 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_edge_stats ASTRAL_BENCH_MEMORY_CAPACITY=100000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS=32 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=64 ASTRAL_BENCH_MEMORY_CAPACITY=10000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=256 ASTRAL_BENCH_MEMORY_RECALL_QUERIES=64 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_STORAGE=q8 ASTRAL_BENCH_FEATURE_ITERS=10 ASTRAL_BENCH_MEMORY_CAPACITY=100000 ASTRAL_BENCH_MEMORY_DIM=384 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_STORAGE=q8f32 ASTRAL_BENCH_FEATURE_ITERS=10 ASTRAL_BENCH_MEMORY_CAPACITY=100000 ASTRAL_BENCH_MEMORY_DIM=384 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_recall_search ASTRAL_BENCH_FEATURE_ITERS=32 ASTRAL_BENCH_MEMORY_STORAGE=q8f32 ASTRAL_BENCH_MEMORY_CAPACITY=100000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS=64 ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=128 ASTRAL_BENCH_MEMORY_GRAPH_QUERY_SEARCH=1536 ./build/dev/benchmarks/astral_benchmarks --only features
ASTRAL_BENCH_MEMORY_ONLY=1 ASTRAL_BENCH_MEMORY_CASE=graph_add_batch ASTRAL_BENCH_FEATURE_ITERS=1 ASTRAL_BENCH_MEMORY_STORAGE=q8f32 ASTRAL_BENCH_MEMORY_CAPACITY=100000 ASTRAL_BENCH_MEMORY_DIM=384 ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS=64 ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=128 ASTRAL_BENCH_MEMORY_GRAPH_QUERY_SEARCH=1536 ./build/dev/benchmarks/astral_benchmarks --only features
scripts/run_memory_bench_matrix.sh --preset dev --dims 128,384,768 --capacities 10000 --metrics cosine,dot,l2 --out /tmp/astral-memory-matrix.txt
scripts/run_memory_bench_matrix.sh --preset dev --case graph_recall_search --dims 384 --capacities 10000,100000 --metrics cosine --storage q8 --out /tmp/astral-memory-graph-q8.txt
scripts/run_memory_bench_matrix.sh --preset dev --case graph_recall_search --dims 384 --capacities 10000,100000 --metrics cosine --storage f6e3m2 --out /tmp/astral-memory-graph-f6e3m2.txt
scripts/run_memory_bench_matrix.sh --preset dev --case graph_recall_search --dims 384 --capacities 10000,100000 --metrics cosine --storage q8f32 --out /tmp/astral-memory-graph-q8f32.txt
scripts/run_memory_bench_matrix.sh --preset dev --case graph_recall_search --dims 384 --capacities 10000,100000 --metrics cosine --storage f8e5m2f32 --out /tmp/astral-memory-graph-f8e5m2f32.txt
scripts/run_memory_search_acceptance.sh --preset dev --capacity 100000 --dim 384 --metric cosine --out-dir /tmp/astral-memory-search
scripts/run_memory_search_acceptance.sh --preset dev --capacity 10000 --dim 384 --metric cosine --perf --perf-bin /path/to/linux-6.13/tools/perf/perf --out-dir /tmp/astral-memory-search-perf
```

Native tests include `inference_memory_index_graph_mock` for graph search,
filtered exact fallback, save/load, and remove/rebuild behavior.

Expected markers include `features.memory add_batch`,
`features.memory graph_add_batch`, `features.memory graph_add_latency`,
`features.memory graph_load`,
`features.memory flat_search_top1`, `features.memory flat_search`,
`features.memory flat_search_latency`, `features.memory flat_search_batch`,
`features.memory flat_q8_recall_search`,
`features.memory graph_top1`, `features.memory graph_search`,
`features.memory graph_search_latency`,
`features.memory graph_recall`, `features.memory graph_recall_top1`,
`features.memory graph_recall_search`, `features.memory graph_edges`,
`features.memory graph_base_edges`, `features.memory graph_upper_edges`,
`features.memory graph_build_score_evals`,
`features.memory graph_build_candidate_visits`, and
`features.memory cursor_begin_fetch`.
The focused graph recall sweep case emits
`features.memory graph_recall_s<N>` markers for each per-search budget.
Sweep runs also include
`features.memory top1_100`, `features.memory top1_1k`,
`features.memory top1_10k`, and `features.memory top1_100k`.
