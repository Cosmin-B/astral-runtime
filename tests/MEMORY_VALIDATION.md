# Memory Validation

Astral's current memory validation is handled by CTest gates that run against
the maintained runtime paths:

- `test_memory`: unit coverage for memory primitives and allocator behavior.
- `test_arena`: runtime arena creation, reset, and reserve behavior.
- `gate_allocations`: steady-state allocation guard for mock streaming paths, with
  CPU-model coverage enabled by `ASTRAL_GATE_CPU_ALLOC=1`.
- `gate_rss_cap`: process RSS cap check for embedded-style validation.
- `gate_model_churn_soak`: repeated model/session load, decode, reset, unload,
  and RSS-drift sampling. It runs quick mock churn by default; real GGUF churn
  is enabled with `ASTRAL_SOAK_MODEL`.
- `test_memory_tsan`: ThreadSanitizer variant for memory tests on supported
  compilers.

Run the default release memory gates:

```bash
cmake --preset release-with-tests
cmake --build --preset release-with-tests -j
ctest --preset release-with-tests -R '^(test_memory|test_arena|gate_allocations|gate_rss_cap|gate_model_churn_soak)$' -V
```

Run the sanitizer lane:

```bash
./scripts/run_asan.sh
./scripts/run_tsan.sh
```

Run Valgrind Memcheck on the allocation gate:

```bash
./scripts/run_valgrind.sh
```

Run Valgrind Massif on the allocation gate:

```bash
./scripts/run_massif.sh
```

Real GGUF allocation checks are opt-in because they require local model
fixtures:

```bash
ASTRAL_GATE_CPU_ALLOC=1 ASTRAL_MODEL_MIN_BYTES=70000000 \
  ctest --preset release-with-tests -R '^gate_allocations$' -V
```

Run the real-model churn probe on a release runner with a local GGUF:

```bash
ASTRAL_SOAK_MODEL=/models/model.gguf ASTRAL_SOAK_REAL_CYCLES=8 \
  ASTRAL_SOAK_RSS_DRIFT_MB=1024 \
  ctest --preset release-with-tests -R '^gate_model_churn_soak$' -V
```

The remaining production gap is sustained model churn under real Unity/Unreal
hosts: custom engine allocators, long streams, and engine-owned media buffers.
