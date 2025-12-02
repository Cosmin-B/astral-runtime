# Unity Native Allocator Integration

Complete guide to Unity allocator bridge for Astral runtime.

## Overview

The Unity allocator bridge enables Astral to use Unity's native memory allocator (`Allocator.Persistent`) instead of internal malloc. This provides:

1. **Memory Tracking**: All Astral allocations visible in Unity Profiler
2. **Zero Surprises**: No hidden memory usage outside Unity's tracking
3. **Thread-Safe**: Unity allocator allocator is thread-safe
4. **IL2CPP Compatible**: Uses `MonoPInvokeCallback` for all callbacks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Astral Runtime (C++)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Model     │  │   Session    │  │   Backend    │       │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
│         └─────────────────┴──────────────────┘               │
│                           │                                  │
│                    AstralAllocator                           │
│                    (function pointers)                       │
└────────────────────────────┬────────────────────────────────┘
                             │ C ABI
┌────────────────────────────┴────────────────────────────────┐
│                Unity C# Layer                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │     AstralAllocatorBridge.cs                         │   │
│  │  ┌────────────────┐  ┌──────────────────────┐       │   │
│  │  │  UnityAlloc()  │  │    UnityFree()       │       │   │
│  │  │ [MonoPInvoke]  │  │  [MonoPInvoke]       │       │   │
│  │  └────────┬───────┘  └────────┬─────────────┘       │   │
│  └───────────┼──────────────────┼─────────────────────┘   │
│              │                   │                          │
│  ┌───────────▼───────────────────▼─────────────────────┐   │
│  │     UnsafeUtility.Malloc/Free                       │   │
│  │     (Unity allocator Allocator)                       │   │
│  └──────────────────────┬──────────────────────────────┘   │
└─────────────────────────┼──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│              Unity Memory Profiler                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Native Memory:                                      │  │
│  │    - Astral Model: 2.1 GB                           │  │
│  │    - Astral Session: 156 MB                         │  │
│  │    - Astral Backend: 4.3 GB                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Allocator Bridge (`AstralAllocator.cs`)

**Key Components**:

```csharp
public static class AstralAllocatorBridge
{
    // Delegate types (must match C ABI)
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr AllocFn(IntPtr user, UIntPtr size, UIntPtr align);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void FreeFn(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align);

    // Keep delegates alive (GC roots)
    private static AllocFn s_allocDelegate;
    private static FreeFn s_freeDelegate;

    // Create allocator
    public static AstralNative.AstralAllocator CreateUnityAllocator()
    {
        s_allocDelegate = UnityAlloc;
        s_freeDelegate = UnityFree;

        return new AstralNative.AstralAllocator
        {
            alloc = Marshal.GetFunctionPointerForDelegate(s_allocDelegate),
            free = Marshal.GetFunctionPointerForDelegate(s_freeDelegate),
            user = IntPtr.Zero
        };
    }
}
```

**Critical Requirements**:

1. **Delegate Lifetime**: Delegates MUST be kept alive in static fields
   - GC can collect delegates if not rooted
   - Use static fields to prevent collection

2. **MonoPInvokeCallback**: Required for IL2CPP
   ```csharp
   [MonoPInvokeCallback(typeof(AllocFn))]
   private static IntPtr UnityAlloc(IntPtr user, UIntPtr size, UIntPtr align)
   ```

3. **Exception Handling**: Never throw across C ABI boundary
   ```csharp
   try {
       // Allocation logic
   } catch (Exception ex) {
       Debug.LogError($"Allocation failed: {ex}");
       return IntPtr.Zero; // Return null on failure
   }
   ```

4. **Thread-Safety**: Unity allocator is thread-safe
   - `UnsafeUtility.Malloc/Free` uses Unity allocator
   - Unity runtime is thread-safe across all platforms

### 2. Allocation Callback

```csharp
[MonoPInvokeCallback(typeof(AllocFn))]
private static IntPtr UnityAlloc(IntPtr user, UIntPtr size, UIntPtr align)
{
    try
    {
        unsafe
        {
            void* ptr = UnsafeUtility.Malloc(
                (long)size,
                (int)align,
                Allocator.Persistent
            );

            if (ptr == null)
            {
                Debug.LogError($"Unity allocator failed: size={size}, align={align}");
                return IntPtr.Zero;
            }

            // Track allocation (thread-safe)
            Interlocked.Add(ref s_totalAllocated, (long)size);
            Interlocked.Increment(ref s_allocCount);

            return new IntPtr(ptr);
        }
    }
    catch (Exception ex)
    {
        Debug.LogError($"UnityAlloc exception: {ex.Message}");
        return IntPtr.Zero;
    }
}
```

**Design Notes**:

- **Allocator.Persistent**: Long-lived allocations (not GC'd)
- **Alignment**: Power-of-2 (validated by Unity)
- **Error Handling**: Return null on failure (never throw)
- **Tracking**: Thread-safe counters for debugging

### 3. Free Callback

```csharp
[MonoPInvokeCallback(typeof(FreeFn))]
private static void UnityFree(IntPtr user, IntPtr ptr, UIntPtr size, UIntPtr align)
{
    try
    {
        if (ptr == IntPtr.Zero) return; // Null-safe

        unsafe
        {
            UnsafeUtility.Free(ptr.ToPointer(), Allocator.Persistent);
        }

        // Track deallocation
        Interlocked.Add(ref s_totalFreed, (long)size);
        Interlocked.Increment(ref s_freeCount);
    }
    catch (Exception ex)
    {
        Debug.LogError($"UnityFree exception: {ex.Message}");
    }
}
```

**Design Notes**:

- **Null-Safe**: Handle null pointers gracefully
- **Matching Allocator**: MUST use same allocator as `UnityAlloc` (Persistent)
- **No Exceptions**: Catch all exceptions, log errors
- **Tracking**: Thread-safe counters for leak detection

### 4. Memory Tracking

```csharp
// Get statistics
public static (long allocated, long freed, int allocCount, int freeCount) GetMemoryStats()
{
    return (
        Interlocked.Read(ref s_totalAllocated),
        Interlocked.Read(ref s_totalFreed),
        Volatile.Read(ref s_allocCount),
        Volatile.Read(ref s_freeCount)
    );
}

// Validate no leaks
public static bool ValidateNoLeaks()
{
    var stats = GetMemoryStats();
    long delta = stats.allocated - stats.freed;
    int countDelta = stats.allocCount - stats.freeCount;

    if (delta != 0 || countDelta != 0)
    {
        Debug.LogWarning($"Memory leak: {delta} bytes, {countDelta} allocations");
        return false;
    }

    return true;
}
```

## Usage Patterns

### Basic Initialization

```csharp
using Astral.Runtime;

// Initialize with Unity allocator
var config = AstralConfig.Default;
config.useUnityAllocator = true;

var init = config.ToNative();
AstralNative.astral_init(ref init);
```

### Custom Configuration

```csharp
var config = new AstralConfig
{
    useUnityAllocator = true,    // Use Unity allocator
    reserveBytes = 4UL << 30,    // 4GB virtual memory
    enableHugepages = true,      // Try huge pages (if OS supports)
    threadCount = 0,             // Auto-detect
    minLogLevel = 0              // Errors only
};

var init = config.ToNative();
AstralNative.astral_init(ref init);
```

### Memory Profiling

```csharp
// Before shutdown
AstralAllocatorBridge.LogMemoryStats();
// Output:
//   [Astral] Memory Stats:
//     Total Allocated: 6,442,450,944 bytes (156 allocations)
//     Total Freed: 6,442,450,944 bytes (156 frees)
//     Current Usage: 0 bytes (0 live allocations)

// Validate no leaks
if (!AstralAllocatorBridge.ValidateNoLeaks())
{
    Debug.LogError("Memory leaks detected!");
}

AstralNative.astral_shutdown();
```

## Unity Profiler Integration

### Viewing Memory Usage

1. **Open Unity Profiler**:
   - Window → Analysis → Profiler
   - Enable "Memory" module

2. **Run Scene**:
   - Press Play
   - Initialize Astral runtime
   - Load model

3. **Check Native Memory**:
   - Profiler → Memory → Native Memory
   - Look for "Persistent" allocations
   - Astral allocations appear as "Native Allocations"

### Memory Profiler Package

1. **Install Memory Profiler**:
   - Window → Package Manager
   - Search "Memory Profiler"
   - Install

2. **Capture Snapshot**:
   - Window → Analysis → Memory Profiler
   - Click "Capture New Snapshot"

3. **View Astral Memory**:
   - Native Allocations → Persistent
   - Filter by allocation size/count
   - Track allocation stacks (debug builds only)

### Debugging Leaks

```csharp
public class AstralManager : MonoBehaviour
{
    void OnDestroy()
    {
        // Log statistics
        AstralAllocatorBridge.LogMemoryStats();

        // Check for leaks
        if (!AstralAllocatorBridge.ValidateNoLeaks())
        {
            Debug.LogWarning("Memory leaks detected before shutdown!");

            // Get detailed stats
            var stats = AstralAllocatorBridge.GetMemoryStats();
            Debug.Log($"Live allocations: {stats.allocCount - stats.freeCount}");
            Debug.Log($"Live memory: {stats.allocated - stats.freed} bytes");
        }

        // Shutdown
        AstralNative.astral_shutdown();
    }
}
```

## Performance Characteristics

### Allocation Cost

| Operation | Unity Allocator | Internal Allocator |
|-----------|----------------|-------------------|
| Malloc    | ~50-100 ns     | ~30-50 ns        |
| Free      | ~50-100 ns     | ~30-50 ns        |
| Overhead  | +20-50 ns      | Baseline         |

**Notes**:
- Unity allocator is slightly slower due to tracking overhead
- Performance impact is negligible (allocations are not in hot path)
- Benefits (memory tracking, leak detection) outweigh cost

### Memory Overhead

| Allocator | Overhead per Allocation |
|-----------|-------------------------|
| Unity     | ~16-32 bytes           |
| Internal  | ~8-16 bytes            |

**Notes**:
- Unity allocator tracks metadata (file, line, size)
- Overhead is minimal compared to typical allocation sizes (MB+)

### Thread-Safety

- **Unity allocator**: Thread-safe across all platforms
- **Lock-Free**: Uses atomic operations internally
- **No Contention**: Separate allocator per thread (thread-local caches)

## Platform-Specific Notes

### Windows

- **Allocator**: Unity runtime uses `VirtualAlloc` + custom allocator
- **Huge Pages**: Requires `SeLockMemoryPrivilege` (admin)
- **Unity Profiler**: Native memory tracked correctly

### macOS

- **Allocator**: Unity runtime uses `mach_vm_allocate` + custom allocator
- **Huge Pages**: Not supported
- **Unity Profiler**: Native memory tracked correctly

### Linux

- **Allocator**: Unity runtime uses `mmap` + custom allocator
- **Huge Pages**: Requires `/proc/sys/vm/nr_hugepages` configured
- **Unity Profiler**: Native memory tracked correctly

### iOS

- **Allocator**: Unity runtime uses `mach_vm_allocate` + custom allocator
- **Memory Limit**: Strict memory limits (2GB+ can trigger warnings)
- **Unity Profiler**: Native memory tracked correctly

### Android

- **Allocator**: Unity runtime uses `mmap` + custom allocator
- **Memory Limit**: Varies by device (1-4GB typical)
- **Unity Profiler**: Native memory tracked correctly

## Troubleshooting

### Issue: Allocations not visible in Unity Profiler

**Solution**:
- Ensure `useUnityAllocator = true` in config
- Check Unity Profiler is recording "Native Memory"
- Try Memory Profiler package for detailed view

### Issue: Memory leaks detected

**Solution**:
```csharp
// Check statistics before shutdown
AstralAllocatorBridge.LogMemoryStats();

// Common causes:
// 1. Model not released
AstralNative.astral_model_release(model);

// 2. Session not destroyed
AstralNative.astral_session_destroy(session);

// 3. Embedder not destroyed
AstralNative.astral_embed_destroy(embedder);
```

### Issue: Allocation failures

**Solution**:
```csharp
// Check error logs
// [Astral] Unity allocator failed: size=X, align=Y

// Common causes:
// 1. Out of memory (reduce reserveBytes)
// 2. Invalid alignment (must be power-of-2)
// 3. Unity allocator not initialized (call after Unity initialized)
```

### Issue: Crashes in IL2CPP builds

**Solution**:
- Ensure `[MonoPInvokeCallback]` on all callbacks
- Keep delegate references alive (static fields)
- Test on target platform early

## Best Practices

1. **Always use Unity allocator in production**:
   ```csharp
   var config = AstralConfig.Default;
   config.useUnityAllocator = true; // Always true
   ```

2. **Validate no leaks before shutdown**:
   ```csharp
   void OnDestroy() {
       AstralAllocatorBridge.ValidateNoLeaks();
       AstralNative.astral_shutdown();
   }
   ```

3. **Log memory stats periodically**:
   ```csharp
   void Update() {
       if (Input.GetKeyDown(KeyCode.M)) {
           AstralAllocatorBridge.LogMemoryStats();
       }
   }
   ```

4. **Profile early and often**:
   - Use Unity Profiler during development
   - Capture Memory Profiler snapshots before/after loading
   - Track memory growth over time

5. **Test on target platforms**:
   - IL2CPP builds behave differently than Mono
   - Test on iOS/Android early
   - Verify allocator works correctly

## References

- `plugins/unity/Runtime/AstralAllocator.cs` - Implementation
- `include/astral_rt.h` - C ABI reference
- `docs/architecture/MEMORY_ARCHITECTURE.md` - Memory architecture
- Unity allocator documentation - Unity allocator internals

## License

See project LICENSE file.
