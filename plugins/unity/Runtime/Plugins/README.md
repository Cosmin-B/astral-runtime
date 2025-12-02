# Astral Native Plugins

This directory contains platform-specific native libraries for Astral runtime.

## Directory Structure

```
Plugins/
├── x86_64/
│   ├── libastral_rt.so      # Linux x86_64
│   └── astral_rt.dll        # Windows x86_64
├── arm64/
│   ├── libastral_rt.so      # Linux ARM64 (Android)
│   └── libastral_rt.dylib   # macOS ARM64 (Apple Silicon)
└── iOS/
    └── libastral_rt.a       # iOS static library (ARM64)
```

## Build Instructions

Note: `cmake --preset unity-plugin` now also packages the built native library into this `Runtime/Plugins/<arch>/` folder automatically as part of the build. The manual `cp` commands below are still valid if you prefer explicit copying.

### Linux (x86_64)
```bash
cd <astral-repo>
cmake --preset unity-plugin
cmake --build build/unity
# (optional) manual copy:
# cp build/unity/libastral_rt.so plugins/unity/Runtime/Plugins/x86_64/
```

### Windows (x86_64)
```bash
cmake --preset unity-plugin -G "Visual Studio 17 2022"
cmake --build build/unity --config Release --target astral_rt
cp build/unity/Release/astral_rt.dll plugins/unity/Runtime/Plugins/x86_64/
```

### macOS (ARM64)
```bash
cmake --preset unity-plugin -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build/unity --target astral_rt
cp build/unity/libastral_rt.dylib plugins/unity/Runtime/Plugins/arm64/
```

### Android (ARM64)
```bash
# Requires Android NDK
export ANDROID_NDK=/path/to/android-ndk
cmake --preset unity-plugin \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21
cmake --build build/unity --target astral_rt
cp build/unity/libastral_rt.so plugins/unity/Runtime/Plugins/arm64/
```

### iOS (ARM64)
```bash
# Requires Xcode
cmake --preset unity-plugin \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0
cmake --build build/unity --target astral_rt
cp build/unity/libastral_rt.a plugins/unity/Runtime/Plugins/iOS/
```

## Unity Import Settings

Unity will automatically detect platform-specific plugins based on directory structure.
Ensure the following import settings in Unity Editor:

### x86_64/libastral_rt.so (Linux)
- Platform: Linux x86_64
- Load on startup: Yes

### x86_64/astral_rt.dll (Windows)
- Platform: Windows x86_64
- Load on startup: Yes

### arm64/libastral_rt.so (Android)
- Platform: Android
- CPU: ARM64
- Load on startup: Yes

### arm64/libastral_rt.dylib (macOS)
- Platform: macOS
- CPU: ARM64
- Load on startup: Yes

### iOS/libastral_rt.a (iOS)
- Platform: iOS
- CPU: ARM64
- Load on startup: Yes

## Troubleshooting

### DllNotFoundException
- Ensure native library is in correct platform directory
- Check Unity import settings (Platform, CPU, Load on startup)
- Verify library architecture matches Unity build target

### EntryPointNotFoundException
- P/Invoke signature mismatch between C# and C ABI
- Check function name, calling convention, and parameter types
- Use `nm -D libastral_rt.so` (Linux) or `dumpbin /EXPORTS astral_rt.dll` (Windows) to verify exports

### IL2CPP Crashes
- Ensure all P/Invoke declarations use `CallingConvention.Cdecl`
- Verify struct layouts with `StructLayout(LayoutKind.Sequential)`
- Check for missing `allowUnsafeCode: true` in asmdef
