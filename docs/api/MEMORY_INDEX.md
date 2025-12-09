# Native Memory Index

Astral memory indexes store embedding vectors in native contiguous memory and
return stable keys and scores for retrieval. The first implementation is a flat
index for small and medium data sets; it is the correctness baseline for later
ANN indexes.

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
- `astral_memory_save_size()`
- `astral_memory_save()`
- `astral_memory_load()`

The index is fixed-dimension and fixed-capacity. Add operations copy vectors
into native storage. Search returns result metadata first; callers decide whether
to fetch text or engine objects for the selected keys.

Incremental search cursors snapshot the top-k result set at begin time and let
callers fetch fixed-size batches without re-running the vector scan. The cursor
owns its compact native result buffer and must be released with
`astral_memory_search_end()`.

## Metrics

- `ASTRAL_MEMORY_METRIC_DOT`: higher dot product ranks first.
- `ASTRAL_MEMORY_METRIC_COSINE`: vectors are stored with cached norms and scored
  against the query norm.
- `ASTRAL_MEMORY_METRIC_L2`: negative squared distance ranks first.

Use `ASTRAL_MEMORY_GROUP_ANY` to search all groups, or set
`AstralMemorySearchDesc::group_id` to restrict results.

## Performance

The flat index stores vectors row-major and uses AVX2 dot and L2 kernels when
the compiler target supports them. The scalar fallback is unrolled by four
lanes.
Search keeps the top-k result set in the caller-provided output array, avoiding
heap allocation during query execution.

Feature benchmarks accept `ASTRAL_BENCH_MEMORY_CAPACITY`,
`ASTRAL_BENCH_MEMORY_DIM`, and `ASTRAL_BENCH_MEMORY_METRIC` (`cosine`, `dot`,
or `l2`) so local runs can cover 100, 1k, 10k, and 100k vector scans without
changing source.

Unreal and Unity wrappers expose the same native descriptors and result records.
Wrapper arrays are converted at the engine boundary; the native index owns vector
storage and search ordering.

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

## Validation

```bash
cmake --build --preset dev -j8 --target test_inference test_abi_invalid_args astral_benchmarks
ctest --preset dev -R '^(test_inference|test_abi_invalid_args|gate_abi_layout_report|gate_source_scans|gate_doc_links)$' --output-on-failure
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=200000 ./build/dev/benchmarks/astral_benchmarks --features
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=1000 ASTRAL_BENCH_MEMORY_CAPACITY=100000 ./build/dev/benchmarks/astral_benchmarks --features
```
