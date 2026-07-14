# Coding Standards

These rules apply to Astral core, platform code, providers, tests, and engine
wrappers. Public ABI declarations in `include/` take precedence over examples
or prose.

## Scope

- Keep changes within the subsystem that owns the behavior.
- Do not combine a functional change with broad formatting or unrelated
  refactoring.
- Add an abstraction only when it removes measured complexity or matches an
  established ownership boundary.
- Keep private investigation notes, benchmark captures, machine paths, and
  credentials outside the product repository.

## Language And Formatting

- Core C++ uses C++17 unless a platform preset explicitly requires otherwise.
- Public headers expose a C ABI and must compile as C11 and C++17.
- Use the repository `.clang-format`; do not reformat untouched files.
- New source text is ASCII unless the file or test intentionally covers UTF-8.
- Comments explain ownership, ordering, invariants, or non-obvious platform
  constraints. Do not narrate ordinary assignments.
- Warnings are errors for Astral-owned code in maintained builds.

## Public ABI

- Do not expose C++ standard-library types, exceptions, references, templates,
  vtables, or compiler-specific layouts.
- Use fixed-width integer types, explicit spans, opaque handles, and POD
  descriptors.
- New extensible descriptors start with `uint32_t size`. Append fields; do not
  reorder existing fields after an ABI release.
- Validate descriptor size, required pointers, lengths, enum ranges, and handle
  state at the ABI boundary.
- Catch implementation exceptions before returning across the ABI.
- Return `AstralErr` or a documented sentinel. Set thread-local error context
  when a generic code needs more detail.
- Keep the mirrored Unreal header byte-for-byte synchronized through the
  maintained gate.

## Ownership

Every shared object or buffer must have an identifiable owner:

- spans borrow storage for the documented call or view lifetime;
- handles own or retain internal objects until their matching release call;
- model executors own provider session mutation for conversation slots;
- stream rings have one producer and one consumer;
- retirement queues retain ownership on overflow;
- borrowed arenas outlive runtime shutdown.

Do not use a shared atomic as a substitute for an ownership decision. Prefer
thread-local state, fixed producer lanes, owner-local cursors, and immutable
snapshots where the topology permits them.

## Memory

- Steady-state decode, sampling, stream transfer, and maintained search kernels
  must not introduce hidden heap allocation or VM/file syscalls.
- Prefer fixed-capacity arrays, stack-backed builders, frame allocators, and
  owner-local pools on bounded paths.
- `StackStringBuilder` is the default when truncation is acceptable.
  `StringBuilder` may spill only where a cold runtime-allocation path is part of
  the contract.
- Arena modes are hard boundaries for Astral-owned allocation. Never add a
  silent host-allocation fallback after exhaustion.
- Check reserve, commit, map, and allocator failures before publishing state or
  counters.
- Provider memory has a separate contract; do not describe third-party
  allocation as core arena memory.

## Concurrency

- State the producer, consumer, collector, and destruction roles before
  selecting a queue or reclamation scheme.
- Use SPSC for one producer and one consumer. Use fixed SPSC fan-in lanes when
  producer identities are stable.
- Use the MPSC ticket ring only when producer backpressure is acceptable. Public
  APIs that promise immediate saturation feedback use non-blocking operations.
- Avoid compare-and-swap retry loops on hot paths when ownership or an
  unconditional ticket can provide the required ordering.
- Sequence values publish slot ownership; reservation alone does not publish
  data.
- Use acquire/release ordering by default. A `seq_cst` operation requires a
  written cross-variable ordering argument and an adversarial test.
- Keep frequently written fields on separate cache-line-aligned objects, not
  merely in an aligned array.
- Blocking and wait loops must have a corresponding wake path on ARM.
- Reclamation must never free on queue overflow. The caller keeps ownership and
  retries or applies an explicit cold-path policy.

See [the concurrency model](../architecture/CONCURRENCY_MODEL.md) for the
implemented primitive contracts.

## Performance

- Measure the production topology and data shape before and after a hot-path
  change.
- Inspect optimized assembly for instruction count, dependency chains, spills,
  branches, and accidental scalarization.
- Use runtime CPU feature checks for optional instructions. Compile-time ISA
  macros are not runtime dispatch.
- Keep a scalar implementation for architecture-specific kernels.
- Do not add branch hints, prefetch, alignment, unrolling, or wider vector code
  without evidence on a relevant target.
- Report fixture size, dimensions, search settings, compiler, flags, CPU, thread
  pinning, and statistic with every published number.
- Keep raw counter output and disassembly captures outside the repository;
  repository docs contain reproducible commands and supported conclusions.

## Engine Wrappers

- Unity and Unreal wrappers remain thin owners of native handles and caller
  buffers. Product behavior belongs in the native runtime.
- Unity hot paths use caller-owned `NativeArray<byte>` storage and avoid managed
  string conversion per token.
- Unreal hot paths use caller-owned byte arrays and
  `TRACE_CPUPROFILER_EVENT_SCOPE` for engine-visible profiling.
- Do not call engine APIs from native worker threads unless the wrapper's public
  contract explicitly schedules that transition.
- Update wrapper tests, package layouts, examples, and the mirrored header with
  every ABI change.

## Tests

Tests scale with the behavioral risk:

- boundary changes need invalid pointer, size, enum, state, and transactional
  failure coverage;
- queue changes need empty, full, wraparound, batch, contention, and weak-memory
  coverage;
- lifetime changes need concurrent create/destroy churn and sanitizer runs;
- allocator changes need exhaustion, reuse, no-spill, accounting, and commit
  failure coverage;
- SIMD changes need scalar parity, tails, non-finite or corrupt input behavior,
  runtime dispatch, and target-architecture execution;
- wrapper changes need native ABI layout plus real engine-runner evidence before
  release support is claimed.

Use the narrow target while iterating, then run the release preset for the
completed slice:

```bash
git diff --check
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests --output-on-failure
```

Run `scripts/run_asan.sh` and `scripts/run_tsan.sh` for memory, ownership, or
concurrency changes. Run the subsystem's benchmark or acceptance script for
performance changes.

## Documentation

- Document current behavior, ownership, failure rules, and validation commands.
- Put future work in `ROADMAP.md`, not in API or architecture reference pages.
- The feature matrix records capability status; architecture prose does not
  override it.
- Do not publish benchmark placeholders, unsourced speedups, or machine-specific
  absolute claims.
- Keep examples compilable against the canonical public header.

## Review Checklist

- The public and internal ownership contracts agree.
- Failure is explicit and leaves state valid.
- Hot paths contain no new allocation, syscall, lock, or retry behavior without
  evidence.
- Runtime dispatch covers every optional instruction used.
- Tests exercise the changed boundary and the relevant concurrency topology.
- Documentation and wrappers match the public ABI.
- Repository-facing text contains only product-facing language.
- `git diff --check`, the focused tests, and the release validation pass.
