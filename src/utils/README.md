# Astral Runtime Utilities

This directory contains core utility functions for the Astral runtime.

## Components

### 1. UTF-8 Validation and Utilities (`utf8.hpp`, `utf8.cpp`)

**Features**:
- Immutable UTF-8 span (`Span`) matching `AstralSpanU8` from C ABI
- Scalar UTF-8 validation with strict correctness checks
- SIMD-accelerated validation (AVX2 on x86-64, NEON on ARM)
- Code point counting (distinguishes bytes vs. code points)

**Validation Rules**:
- Rejects invalid byte sequences
- Rejects overlong encodings (security requirement)
- Rejects surrogate pairs (U+D800-U+DFFF)
- Rejects code points > U+10FFFF

**Performance**:
- Scalar validation: 200-400 MB/s
- SIMD validation: 1-2 GB/s (AVX2), 800 MB/s - 1.5 GB/s (NEON)
- Automatic runtime detection and fallback

**API**:
```cpp
namespace astral::utf8 {
    struct Span;                         // Immutable UTF-8 span
    bool validate(Span input);           // Validate UTF-8 (scalar)
    bool validate_simd(Span input);      // Validate UTF-8 (SIMD + fallback)
    uint32_t count_codepoints(Span s);   // Count code points (not bytes)
}
```

### 2. String Builder (`string_builder.hpp`)

**Features**:
- Append-only UTF-8 string builder with inline object storage
- Zero-copy `freeze()` to snapshot current content
- Type-safe append methods (UTF-8 spans, integers, floats)
- Explicit spill or truncation behavior when inline capacity is exhausted

**Memory Discipline**:
- Appends within inline capacity do not allocate
- Spill uses the runtime allocator only on overflow
- Truncate mode never allocates
- Each builder is independently owned; sharing requires external synchronization

**API**:
```cpp
namespace astral::utf8 {
    template<uint32_t InlineCapacity = 256,
             typename SpillAllocator = RuntimeStringAllocator,
             StringOverflowPolicy OverflowPolicy = StringOverflowPolicy::Spill>
    class StringBuilder {
        StringBuilder();

        bool append(Span utf8);
        bool append_u32(uint32_t val);
        bool append_i32(int32_t val);
        bool append_f32(float val, uint32_t decimals = 2);
        bool append_char(char c);

        Span freeze() const;  // Zero-copy snapshot
        const char* c_str() const;
        void reset();
    };
    template<uint32_t InlineCapacity = 256,
             typename SpillAllocator = RuntimeStringAllocator>
    using StackStringBuilder =
        StringBuilder<InlineCapacity,
                      SpillAllocator,
                      StringOverflowPolicy::Truncate>;
}
```

**Usage Example**:
```cpp
StackStringBuilder<256> sb;

sb.append(Span::from_cstr("Token: "));
sb.append_u32(42);
sb.append(Span::from_cstr(" (score: "));
sb.append_f32(0.95f, 2);
sb.append_char(')');

Span result = sb.freeze(); // "Token: 42 (score: 0.95)"
```

### 3. Logging (`logging.hpp`, `logging.cpp`)

**Features**:
- Non-blocking logging with callback dispatch
- Thread-local buffers (4KB per thread)
- Log level filtering
- Automatic slow callback detection (>10ms warning)

**Design Principles**:
- Never block on logging
- No allocations per log call
- Drop logs if callback dispatch exceeds the 10ms logging budget

**API**:
```cpp
namespace astral::logging {
    enum class Level { Error, Warn, Info, Debug, Trace };

    void set_callback(LogCallback cb, void* user);
    void set_min_level(Level level);
    void log(Level level, const char* fmt, ...);

    // Convenience wrappers
    void error(const char* fmt, ...);
    void warn(const char* fmt, ...);
    void info(const char* fmt, ...);
    void debug(const char* fmt, ...);
    void trace(const char* fmt, ...);
}
```

**Usage Example**:
```cpp
void my_log_callback(void* user, int level, const uint8_t* msg, uint32_t len) {
    // Forward to engine logger
    UnityLog(level, std::string_view((const char*)msg, len));
}

logging::set_callback(my_log_callback, nullptr);
logging::set_min_level(logging::Level::Info);

logging::info("Model loaded: %s", model_path);
logging::warn("High memory usage: %u MB", mem_mb);
```

## Build

All utilities are compiled as static libraries:
- `astral_utf8` (UTF-8 validation and utilities)
- `astral_logging` (Logging system)

String builder is header-only (depends on `FrameAllocator`).

### CMake Integration

```cmake
add_subdirectory(src/utils)

target_link_libraries(your_target PRIVATE
    astral_utf8
    astral_logging
)
```

### Manual Build

```bash
g++ -std=c++17 -O2 -mavx2 -c src/utils/utf8.cpp
g++ -std=c++17 -O2 -c src/utils/logging.cpp
ar rcs libastral_utils.a utf8.o logging.o
```

## Testing

Run the test suite:
```bash
g++ -std=c++17 -I./src -O2 -o test_utils \
    src/utils/test_utils.cpp \
    src/utils/utf8.cpp \
    src/utils/logging.cpp

./test_utils
```

All 17 tests should pass:
- UTF-8 span operations (equals, slice, from_cstr)
- UTF-8 validation (ASCII, multibyte, invalid, overlong, surrogates)
- Code point counting
- String builder (append, freeze, reset)
- Logging (basic, formatted, level filtering)

## Design Notes

### UTF-8 Everywhere

All strings in Astral are UTF-8. NO UTF-16, NO Latin-1.
- C ABI uses `AstralSpanU8` (pointer + length)
- Internal C++ uses `utf8::Span` (same layout)
- No NUL termination assumed
- Validation rejects non-canonical encodings

### Hot Path Allocation Boundaries

- UTF-8 validation: No allocations (stack-based state machine)
- String builder: Allocates from `FrameAllocator` only
- Logging: Thread-local buffers (allocated once per thread)

### SIMD Acceleration

- AVX2 path for x86-64 (enabled with `-mavx2`)
- NEON path for ARM (enabled with `-march=armv8-a`)
- Runtime CPU detection with automatic fallback
- Small inputs (<64 bytes) use scalar path (SIMD overhead not worth it)

### Thread Safety

- **UTF-8 validation**: Thread-safe (pure functions)
- **String builder**: NOT thread-safe (one per thread/session)
- **Logging**: Thread-safe (TLS buffers + atomic callback pointer)

## Performance Characteristics

| Operation | Cost | Notes |
|-----------|---------|-------|
| `validate()` | Input-size dependent | Scalar validation |
| `validate_simd()` | Input-size dependent | SIMD with scalar fallback |
| `count_codepoints()` | Input-size dependent | SIMD where supported |
| `StringBuilder::append()` | Input-size dependent | Inline until capacity is exhausted |
| `StringBuilder::freeze()` | Constant time | Returns a span over owned storage |
| `logging::log()` | Callback dependent | Formats into thread-local storage |

## References

- Unicode Standard 15.0, Chapter 3 (Conformance)
- "Validating UTF-8 In Less Than One Instruction Per Byte" (Lemire et al., 2021)
  https://arxiv.org/abs/2010.03090
- `docs/rules/CODING_STANDARDS.md` - UTF-8 and string handling rules
