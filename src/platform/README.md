# Platform layer

This directory keeps operating-system and CPU details out of the runtime core.

| Surface | Files | Implementations |
| --- | --- | --- |
| Virtual memory | `vm.h`, `vm_*.cpp` | Linux, macOS, Windows |
| Threads and waits | `thread.h`, `thread_*.cpp` | POSIX, Windows |
| Mapped files | `file_map.h`, `file_map.cpp` | Platform-selected implementation |
| CPU dispatch | `cpu_features.hpp`, `cpu_features.cpp` | x86 AVX2/F16C and ARM feature checks |
| Atomics and cache lines | `atomics.h`, `atomics.cpp`, `cacheline.hpp` | Portable wrappers and runtime queries |
| Time and compiler helpers | `time.h`, `compiler.hpp` | Shared low-level contracts |

Virtual-memory calls return failure rather than throwing. A reservation must be
committed before an allocator touches it, and the original base and size must
be retained for release. Large-page requests are optional and platform
dependent.

Runtime dispatch checks both build-time availability and the running CPU before
selecting architecture-specific kernels. Scalar implementations remain the
fallback.

See [low-level primitives](../../docs/architecture/LOW_LEVEL_PRIMITIVES.md) for
the detailed contracts and [embedded profiles](../../docs/EMBEDDED_PROFILE.md)
for builds that replace virtual memory with an arena.
