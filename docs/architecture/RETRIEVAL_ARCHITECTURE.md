# Retrieval architecture

Astral's memory index stores embedding vectors and stable application keys in a
fixed-capacity native allocation. A query returns keys and scores into
caller-owned storage. The caller decides which text, asset, entity, or engine
object those keys refer to.

Storage and traversal are separate decisions:

| Stage | Flat path | Graph path |
| --- | --- | --- |
| Encode | f32, q8, Float6, E5M2, or compact plus f32 | The same storage families |
| Select candidates | Scan every matching active row | Descend upper levels, then expand a bounded base-layer set |
| Order results | Exact for scores produced by the stored representation | Approximate candidate set, with optional f32 reranking |
| Filters | Group filters stay on this path | Sparse group filters bypass the graph |

## Why both traversals exist

Flat search has work proportional to the active record count, but its memory
access is contiguous and its stored-score ordering is exhaustive. It is also
the native path for group-filtered queries.

Graph search scores fewer rows when routing works well. It descends sparse
upper levels greedily, then expands a bounded candidate set on the base layer.
Increasing the search budget spends more time and scratch capacity to recover
more of the flat result.

Record count alone does not pick the graph. Dimension, format, filter density,
cache behavior, top-k, and the dataset's recall curve all matter. Compact rows
can be cheap enough to scan that graph routing does not win.

## Storage families

Add and load APIs accept f32 vectors, then encode them into the selected layout.

| Storage | Native payload | Scoring behavior |
| --- | --- | --- |
| f32 | Four bytes per component | Reference scores and flat recall oracle |
| q8 | Signed byte plus one scale per vector | Integer SIMD dot path with approximate scores |
| q8 + f32 rerank | q8 block and hidden f32 block | Compact graph routing, then f32 ordering within candidates |
| Float6 E2M3 | Scaled values in signed bytes | Compact integer dot path |
| Float6 E3M2 | Scaled values in 16-bit lanes | Wider exponent range and a separate kernel family |
| Float6 + f32 rerank | E2M3 or E3M2 block and hidden f32 block | Compact routing, then f32 ordering within candidates |
| E5M2 | One byte per component plus one scale | Wide range, low precision, and x86, ARM, or scalar widening |
| E5M2 + f32 rerank | E5M2 block and hidden f32 block | Dual-format snapshot with f32 graph and result scoring |

Compact traversal is approximate relative to f32 even when every row is
visited. A graph adds another source of approximation because it may never
visit a row that flat search would rank. Reranking improves ordering inside the
candidate set, but cannot recover an unvisited candidate.

Snapshots keep records, scales, vectors, and graph topology in contiguous
ranges. A mapped snapshot can search those ranges without copying vector data
into a mutable index first.

## Flat ownership

The flat loop selects metric and storage kernels before scanning. Large
single-query scans can split contiguous record ranges across runtime workers
and the API caller. Each participant maintains a private top-k, and the caller
merges those bounded results without a shared heap or atomic result updates.

Batch search instead scans cache-hot vector rows for small chunks of queries.
Record sharding and query batching solve different ownership problems, so the
runtime chooses between them from query count, record count, and top-k.

## Graph ownership

Graph construction assigns deterministic upper levels from record keys. Level
zero has twice the configured upper-level neighbor capacity. Insertions descend
from the entry point, run bounded candidate expansion, select neighbors, and
update reverse links inside fixed capacities.

Queries use separate construction and search budgets. Candidate arrays,
visited generations, top scores, and rerank scratch are allocated when the
index is created. Group filters leave the graph because a topology built across
the full corpus may route through many rows outside one sparse group.

## E5M2 widening

E5M2 and FP16 share sign and exponent widths. On x86 with AVX2 and F16C, Astral
widens E5M2 bytes into FP16-shaped lanes before converting them to f32 for the
dot product. ARM64 uses NEON conversion where available, with scalar conversion
retained for other machines. Runtime dispatch checks AVX2 and F16C separately.

The common 384-component x86 path scores two adjacent rows per outer-loop
iteration and falls back to the one-row kernel for an odd tail. This is an
implementation detail, not a universal performance claim. Use the benchmark
fixture documented in the [memory index API](../api/MEMORY_INDEX.md) on the
target CPU and dataset before choosing a storage lane.

## Choosing a configuration

| Need | Start with | Measure next |
| --- | --- | --- |
| Exact stored-score ordering or group filters | Flat f32 or flat compact | Query latency, top-k, and compact recall against flat f32 |
| Smaller resident vectors | Flat q8 or E5M2 | Recall on application embeddings and ingest cost |
| Lower unfiltered query work at larger scale | Graph f32 | Recall distribution and latency at several budgets |
| Compact routing with stable final scores | q8 plus f32 rerank | Candidate recall before rerank and total vector bytes |
| Fast startup from a prepared index | Mapped snapshot | Query latency and page behavior on deployment storage |

The [memory index API](../api/MEMORY_INDEX.md) contains descriptors, example
calls, and reproducible benchmark commands.
