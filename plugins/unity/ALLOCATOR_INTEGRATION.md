# Unity Native Allocator Integration

Astral's Unity package can pass a Unity-backed allocator into `astral_init()`.
When `AstralConfig.useUnityAllocator` is enabled, `AstralRuntime.Initialize()`
uses `AstralAllocatorBridge.CreateUnityAllocator()` and wires the returned
`AstralNative.AstralAllocator` into `AstralInit.sys_alloc`.

This is an ownership boundary, not a hot-path feature. The allocator is used by
native runtime setup, model/session allocation, and other runtime-owned storage.
Token buffers, embedding vectors, memory-index search results, and stream reads
still use caller-owned spans or Unity wrapper buffers at the API boundary.

## Contract

- The bridge uses `UnsafeUtility.Malloc()` and `UnsafeUtility.Free()` with
  `Allocator.Persistent`.
- Allocation and free callbacks are rooted in static fields so native calls do
  not outlive their delegates.
- The callbacks are marked with `MonoPInvokeCallback` for IL2CPP builds.
- Callback failures return a null pointer or log the free failure; exceptions
  are not allowed to cross the C ABI.
- `AstralAllocatorBridge.GetMemoryStats()` reports the bridge counters since
  the latest allocator creation.
- `AstralAllocatorBridge.ValidateNoLeaks()` compares bridge-level allocated and
  freed byte counts. It is a debugging check, not a substitute for target
  platform profiling.

Use the Unity allocator when you want Unity-owned native allocation and editor
diagnostics. Use the internal allocator when you need to isolate Astral runtime
memory from Unity during native debugging or standalone smoke runs.

## Initialization

```csharp
using Astral.Runtime;

var config = AstralConfig.Default;
config.useUnityAllocator = true;

int err;
if (!AstralRuntime.Initialize(config, out err))
{
    Debug.LogError(AstralRuntime.GetErrorString(err));
}
```

`AstralConfig.Default` currently enables the Unity allocator. Call
`AstralRuntime.Shutdown()` after models, sessions, embedders, memory indexes,
and agents have been released.

```csharp
AstralAllocatorBridge.LogMemoryStats();
AstralRuntime.Shutdown();
```

`AstralRuntime.Shutdown()` calls `ValidateNoLeaks()` when the Unity allocator was
used. A leak warning means some native handle or runtime-owned allocation is
still live from the bridge's point of view.

## Platform Notes

The package layout supports desktop native plugins plus Android ARM64 and iOS
staging paths. Current repository evidence covers Unity 6000.0/GameCI import,
managed compile, and Linux native plugin preload when the native plugin is built
in an Ubuntu 22.04 container. Licensed EditMode execution and player builds
still need dedicated Unity runner evidence.

Do not treat editor import or managed compile as proof of allocator performance
on iOS, Android, or IL2CPP players. Capture Unity Profiler or Memory Profiler
evidence on the target runner before using allocator numbers in release notes.

## Troubleshooting

If initialization fails, check the `astral_init()` error code first. If the
Unity callback logs an allocation failure, reduce the native reserve size,
release stale handles, or retry with `useUnityAllocator = false` to isolate
whether the issue is the allocator bridge or native runtime setup.

If `ValidateNoLeaks()` reports live memory at shutdown, release every Astral
handle before calling `AstralRuntime.Shutdown()`:

- `AstralModel.Release()`
- `AstralSession.Dispose()` or `Destroy()`
- `AstralEmbedder.Dispose()`
- `AstralMemoryIndex.Dispose()`
- `AstralAgent.Dispose()`
- `AstralAdapter.Dispose()`

For IL2CPP player issues, verify that the target package contains the correct
native library for the platform and architecture, then run a player build with
allocator logging enabled.

## Validation

Local checks that do not require a licensed Unity editor:

```bash
ctest --preset release-with-tests -R '^(gate_source_scans|gate_doc_links|gate_unity_mobile_package_layout)$' --output-on-failure
```

Unity runner validation, when license material is available:

```bash
scripts/run_unity_ci_tests.sh
```

Generated Unity `Library/`, player builds, logs, license files, and activation
responses must remain outside the Astral repository.
