# Runtime architecture

Astral is the native control plane around model execution. The application
creates handles through a C ABI, submits bounded work, and drains results into
its own buffers. A provider loads the model and performs token or embedding
evaluation. Keeping those responsibilities separate lets local CPU, CUDA
offload, dynamic plugins, and remote execution use the same request lifecycle.

## The boundary and its owners

The public ABI uses sized descriptors, explicit handles, byte spans, and
integer error codes. It does not export C++ standard-library types or provider
objects. A descriptor's `size` field lets the runtime reject an incompatible
layout before reading optional fields.

| Object | Owner | Release point |
| --- | --- | --- |
| Input spans | Caller | After the operation returns, unless the operation documents a copy |
| Model, session, conversation, agent, and index handles | Astral | Matching release or destroy call |
| Provider model and session contexts | Selected provider | Astral calls the provider destroy operation |
| Stream, embedding, and search output buffers | Caller | Caller decides when to reuse or free them |

Handle objects retain the native state they depend on. Sessions and embedders
retain their model. An agent retains prompt state and borrows a conversation
slot only while a chat request needs decode.

## Providers execute models

Providers implement the POD function table in `include/astral_backend.h`. The
core calls it for model loading, sessions, tokenization, evaluation, logits,
embeddings, adapters, and slot operations when the provider supplies them.

The built-in names are `cpu`, `cuda`, `mock`, and `remote`. Dynamic providers
use the same table through a shared-library entry point. Astral validates the
table at load time and keeps the library resident while its provider can still
be selected. An explicit provider name is authoritative, and an unknown name
fails instead of silently changing execution environments.

## A session request

A normal asynchronous session has one decode producer and one stream consumer:

1. The caller feeds text, tokens, or supported media into session-owned prompt
   storage.
2. `astral_session_decode()` places one bounded job on the runtime worker pool.
3. That worker owns provider session mutation for the decode.
4. The provider evaluates tokens and exposes logits. Astral applies grammar,
   sampling, stop rules, and token metadata.
5. Detokenized bytes enter a bounded SPSC ring. The producer waits when the
   ring is full.
6. The caller polls or waits, copies bytes into its own buffer, and releases
   the handle after completion and stream drain.

Reads return byte chunks and may stop inside a UTF-8 code point. Text consumers
therefore keep an incremental decoder across reads. The stream remains
byte-oriented so the native boundary does not impose a managed allocation or
encoding policy.

## Continuous batching changes the producer

A continuous-batching model owns one executor thread and one provider session
with several slots. The executor becomes the decode producer for every active
slot on that model. Prompt ingest runs round-robin with a configurable per-slot
allowance, while decode contributes at most one token per active slot per tick.

Conversation creation is bounded. When every configured slot is occupied it
returns `ASTRAL_E_BUSY`, leaving the application to wait, reject, or shed work.
Slot snapshots use epoch reclamation so the executor enters one read epoch for
a scheduling pass instead of paying reference-count traffic for every slot.

## Agents reuse the executor

An agent owns bounded copies of its system prompt, rolling summary, retrieved
memory context, and message history. Chat enqueue assembles those sections with
the current user turn and assistant prefix. Once prepared, agent chat enters
the same model executor as a direct conversation.

Retrieval remains explicit. The caller can inspect stable memory-index keys or
ask Astral to copy selected chunks into the agent context. The index does not
retain application documents or engine objects behind those keys.

## Engine-paced delivery

Unity and Unreal wrap the ABI, not model execution. Unity can poll streams into
`NativeArray<byte>`. Unreal result objects carry native request state and move
Blueprint delegate delivery onto the game thread. Native callers can poll or
wait directly. Decode, embedding, and retrieval hot loops do not invoke an
application callback.

## Platform layers

The core has Linux, macOS, and Windows implementations for virtual memory,
threads, time, mapped files, and CPU feature detection. Runtime dispatch selects
x86 AVX2/F16C or ARM NEON kernels only when the build and running CPU support
them. Scalar implementations remain available for portability and kernel
correctness tests.

Embedded presets use arena-backed core memory, turn off dynamic providers, and
disable threads. Session and embedding surfaces remain available through
compatible providers. Model executors and agents require the threaded desktop
profile.

## Where to go deeper

- [Provider boundary](BACKEND_ARCHITECTURE.md)
- [Concurrency and thread ownership](CONCURRENCY_MODEL.md)
- [Core and provider memory](MEMORY_ARCHITECTURE.md)
- [Flat, graph, and compact retrieval](RETRIEVAL_ARCHITECTURE.md)
- [Asynchronous request states](../api/ASYNC_DELIVERY.md)
- [Continuous batching API](../api/CONTINUOUS_BATCHING.md)
- [Agent API](../api/AGENT_RUNTIME.md)
