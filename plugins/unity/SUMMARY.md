# Astral Unity Plugin - Summary

This directory is a Unity Package Manager package that wraps Astral's C ABI for Unity (Mono + IL2CPP).

## Layout

**Docs**
- `README.md`: install + usage
- `IMPLEMENTATION.md`: marshaling + low-level notes
- `ALLOCATOR_INTEGRATION.md`: allocator bridge design (not implemented yet)
- `package.json`: UPM manifest

**Runtime**
- `Runtime/AstralNative.cs`: P/Invoke declarations for `astral_rt.h`
- `Runtime/AstralNativeArray.cs`: helpers for UTF-8 spans and `NativeArray<byte>`
- `Runtime/AstralRuntime.cs`: runtime init/shutdown
- `Runtime/AstralModel.cs`: model lifetime wrapper (typed 64-bit handle)
- `Runtime/AstralSession.cs`: session wrapper (feed/decode/stream/cancel/wait/reset)
- `Runtime/AstralAllocator.cs`: Unity allocator bridge (TODO)
- `Runtime/AstralLogging.cs`: logging callback bridge (TODO)
- `Runtime/AstralJobSystem.cs`: jobs/Burst integration (TODO)
- `Runtime/AstralExample.cs`: example usage
- `Runtime/Plugins/README.md`: where native binaries go per platform
- `Runtime/Astral.Runtime.asmdef`: runtime asmdef

**Editor**
- `Editor/AstralEditorUtilities.cs`: basic validation helpers
- `Editor/Astral.Editor.asmdef`: editor asmdef

**Samples**
- `Samples~/BasicChat/BasicChatExample.cs`

**Tests**
- `Tests/Editor/AstralAbiTests.cs`: ABI layout + mock-backend smoke (skips if native lib missing)

## ABI notes

- `AstralHandle` is a `ulong` in C# (0 is invalid), matching the 64-bit C ABI handle.
- Strings are passed as UTF-8 spans (`{ byte* data, uint len }`, 16 bytes on 64-bit due to padding), not NUL-terminated strings.

## Current status

- Implemented: backend name + seed wiring, model/session lifetimes, streaming read loop, cancel/wait/state, session reset.
- TODO: Unity allocator bridge, logging callbacks, packaging native binaries for each platform, jobs/Burst integration, embeddings wrapper.
