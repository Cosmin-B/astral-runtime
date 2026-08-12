# Core memory primitives

This directory contains the small allocators used by the runtime core:

- `frame_allocator.hpp`: owner-local bump allocation over caller-provided,
  accessible memory. Reset invalidates every allocation from the frame.
- `object_pool.hpp`: fixed-capacity intrusive pools in shared and owner-local
  forms.
- `stats.hpp`: the POD memory counters used by allocator diagnostics.
- `test_memory.cpp`: allocator contract and capacity tests.

These types do not reserve or commit virtual memory themselves. Callers provide
their backing storage and own its lifetime. The shared object pool synchronizes
access. The frame allocator and local pool require one owner.

Run the maintained tests with:

```bash
ctest --preset release-with-tests -R '^test_memory$' --output-on-failure
```

See [memory architecture](../../docs/architecture/MEMORY_ARCHITECTURE.md) for
runtime arenas, scratch ownership, provider boundaries, and allocation gates.
