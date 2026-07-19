# Backend Architecture

This document describes the backend contract implemented by the current
runtime. Backend availability is a build and release property; consult the
[feature matrix](../FEATURE_MATRIX.md) before making a platform support claim.

## Boundary

Astral exposes models, sessions, embeddings, and conversations through the C
ABI in
[`include/astral_rt.h`](https://github.com/Cosmin-B/astral-runtime/blob/main/include/astral_rt.h).
Providers implement the function table in
[`include/astral_backend.h`](https://github.com/Cosmin-B/astral-runtime/blob/main/include/astral_backend.h).
The core owns argument validation, handles, scheduling, sampling, streaming,
and feature orchestration.
The selected provider owns its model and session contexts.

The provider table is a POD C structure. It does not expose C++ vtables or
standard-library types across a shared-library boundary. Public request and
configuration descriptors carry size fields. The provider descriptor and its
operation table currently do not, so separately built plugins must use the
exact header revision shipped with the host runtime.

## Built-In Providers

The registry is a fixed-capacity array initialized on first use. Current builds
register these providers:

| Name | Role | Availability |
| --- | --- | --- |
| `cpu` | llama.cpp model, session, tokenizer, grammar, adapter, and embedding operations | Desktop and supported embedded builds |
| `mock` | Deterministic tests and wrapper validation | All normal test builds |
| `remote` | HTTP-backed remote inference and embeddings | Builds with the remote transport enabled |
| `cuda` | The llama.cpp operation table with GGML CUDA offload | `ASTRAL_ENABLE_CUDA=ON` only |

The CUDA provider is not a second inference implementation. CUDA builds reuse
the CPU provider operation table, while llama.cpp routes eligible work through
GGML CUDA according to the model configuration.

## Selection

`AstralModelDesc.backend_name` is authoritative when non-empty. An unknown or
unavailable name fails model loading instead of silently selecting another
provider.

Without an explicit name, a model with `gpu_layers == 0` selects `cpu`. A model
requesting GPU layers prefers `cuda`, then any registered `metal` or `directml`
provider, and finally falls back to `cpu`. The feature matrix records which of
those names are actually supplied by maintained builds.

Provider registration occurs during startup. Registry mutation is not a
concurrent operation; selection is read-only after initialization.

## Model Sources

The public model descriptor accepts path, memory, and callback-based I/O
sources. Provider support is independent of the ABI surface:

- The llama.cpp provider consumes paths directly.
- Desktop memory and I/O sources may be materialized to a temporary file when
  the pinned llama.cpp API has no in-memory loader.
- Embedded profiles disable that materialization path unless their platform
  contract explicitly permits it.
- The mock provider accepts synthetic paths used by the test suite.

Applications that require a syscall-free model load must validate the selected
provider and profile. A successful compile does not establish that property.

## Ownership And Lifetime

- `astral_model_load` creates a core model object and a provider-owned model
  context.
- Sessions and embedders retain the model through core handles; release them
  before releasing the model.
- A provider's model and session contexts remain opaque to the core.
- Provider-returned views, including logits, obey the lifetime documented on
  the provider operation. Callers must not retain those pointers across a
  mutating provider call.
- Dynamic plugin libraries remain loaded while their provider can be selected.

Provider allocations are not automatically covered by Astral's arena. The core
allocator governs Astral-owned objects; each provider must document and test its
own allocation behavior.

## Threading

The core determines which thread invokes each provider operation. A provider
must honor the operation-level threading contract in `astral_backend.h`:

- model metadata and tokenization operations must be thread-safe for a shared
  model context. No capability flag allows a provider to opt out.
- provider session state is owned by its assigned decode or model-executor
  thread;
- continuous-batching slot operations execute on the model executor;
- provider destruction must not race an operation using the same context.

The [concurrency model](CONCURRENCY_MODEL.md) describes the core ownership and
reclamation rules around these calls.

## Dynamic Plugins

`astral_backend_load_plugin` loads a shared library and resolves
`astral_backend_plugin_provider_v0`. The `v0` entry point selects the contract,
but the current provider and operation-table structs have no version or size
field. Registration checks the provider name, ten required operations, registry
capacity, and duplicate names. It cannot make an older, shorter table safe to
read. Until the table gains an explicit size contract, rebuild external plugins
against the same `astral_backend.h` revision as the host. Plugin loading is a
startup operation and is disabled by embedded presets.

The sample and toy plugins under
[`backend_plugins`](https://github.com/Cosmin-B/astral-runtime/tree/main/backend_plugins)
are the maintained build references. They compile through the root build. There
is not yet a standalone external plugin example. A plugin should be tested
through `test_provider_harness`, not only by loading the library directly.

## Validation

Relevant local gates are:

```bash
cmake --build --preset release-with-tests -j --target \
  test_backend test_provider_harness test_model_sources
ctest --preset release-with-tests --output-on-failure \
  -R '^(test_backend|test_provider_harness|test_model_sources)$'
```

CUDA, remote transport, media, and real-model support require their dedicated
release evidence in addition to these provider-agnostic tests.
