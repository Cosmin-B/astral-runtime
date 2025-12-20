# Hot-Path Profiling

Astral keeps engine profiling and native profiling separate so captures stay
useful without adding noise to inner loops.

## Native Runtime

Native code uses Astral's Tracy wrapper macros from `src/utils/trace.hpp`.
Profiling builds enable those zones through the existing `*-prof` presets.
Micro zones are compiled only in micro-profiling presets.

Useful native zones cover model load, tokenization, prompt ingestion, decode
work, stream reads, embedding enqueue/collect/cancel, worker scheduling, and
queue wait points. Avoid adding zones inside tiny token, character, or vector
inner loops unless the micro-profiling preset needs that evidence.

## Unreal

Unreal plugin code uses `TRACE_CPUPROFILER_EVENT_SCOPE` for wrapper work that
shows up in Unreal Insights: stream pumping, session tick handoff, embedder
enqueue/collect/cancel, Blueprint tool/memory/prompt-cache operations, and agent
chat helpers.

Unreal code should not use Astral Tracy macros. Native runtime code should not
include Unreal profiler scopes.

## Unity

Unity wrappers keep profiling at the engine boundary. The native plugin can be
built with the Unity profiling presets when Tracy capture is required, while
Unity Editor profiling remains external validation evidence.

## Validation

```bash
cmake --build --preset unity-plugin -j8
ctest --preset dev -R '^gate_source_scans$' --output-on-failure
```

The source gate keeps required Unreal CPU scopes in place and rejects profiler
macro drift across the native/Unreal boundary.
