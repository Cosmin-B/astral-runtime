# Astral Coding Standards

## Core Principles

1. **Performance First**: Every line of code in the hot path must justify its existence
2. **Zero Trust in Allocations**: Assume dynamic allocation is a bug until proven necessary
3. **Explicit Over Implicit**: No hidden costs, no magic, no surprises
4. **Stable ABI**: C ABI must never change within a major version
5. **Portability**: Code must run on armv7, armv8, arm64, x86-64 without platform-specific hacks
6. **Data-Oriented Design**: Prefer data-oriented layouts; think in arrays and data flow, not objects
7. **Simple Architecture**: Keep core simple; add flexibility via small, explicit hooks, not frameworks

## Language Features

### C++17 Usage

**Allowed**:
- `constexpr` for compile-time computation
- Structured bindings
- `if constexpr` for platform selection
- Internal compiler macros from `src/platform/compiler.hpp` for force-inline,
  no-inline, branch hints, and restrict-like annotations

**Forbidden**:
- Modules (not yet cross-platform stable)
- Coroutines (allocation implications)
- Concepts
- Ranges (STL dependency)
- `std::format` (allocation, binary size)
- `std::span` (reimplemented as `Span` to avoid STL dependency)
- `std::bit_cast`
- `[[likely]]`, `[[unlikely]]`; use `ASTRAL_LIKELY` and `ASTRAL_UNLIKELY`
- Designated initializers
- Three-way comparison (`<=>`)
- Any STL containers (`std::vector`, `std::string`, etc.)
- Any STL allocators or memory facilities

### RTTI and Exceptions

- **RTTI**: Forbidden across C ABI; allowed internally only if zero-cost
- **Exceptions**: Forbidden across C ABI; allowed in C++ wrapper layer only
- **Alternative**: Return codes + out params for C ABI; `Result<T>` for C++ wrapper

## Naming Conventions

### C ABI (`astral_rt.h`)

```c
// Functions: astral_<module>_<action>
AstralErr astral_session_create(...);
void      astral_model_release(...);

// Types: Astral<Name>
typedef struct AstralSpanU8 { ... } AstralSpanU8;
typedef uint64_t AstralHandle; // 64-bit tagged handle; 0 = invalid

// Enums: ASTRAL_<NAME>
enum {
  ASTRAL_OK = 0,
  ASTRAL_E_NOMEM = -2,
};

// Callbacks: Astral<Purpose>Fn
typedef void (*AstralLogFn)(void* user, int level, AstralSpanU8 msg);
```

### C++ Internal (`src/`)

```cpp
// Namespaces: astral::module
namespace astral::memory { ... }
namespace astral::concurrency { ... }

// Classes: PascalCase
class FrameAllocator { ... };
class MpmcQueue { ... };

// Functions: snake_case
void commit_pages(void* addr, size_t size);
void enqueue_wait(Token tok);

// Members: snake_case with trailing underscore
class Session {
  FrameAllocator allocator_;
  uint32_t max_tokens_;
};

// Constants: kPascalCase
constexpr size_t kCacheLineSize = 64;
constexpr uint32_t kDefaultBatchSize = 512;

// Template parameters: TName
template<typename TAllocator>
class Queue { ... };
```

### Platform Shims (`src/platform/`)

```cpp
// Platform-specific functions: lowercase, OS prefix if needed
void* vm_reserve(size_t size);
void  vm_commit(void* addr, size_t size);

// Conditionals: Clean separation
#if defined(_WIN32)
  // Windows impl
#elif defined(__linux__)
  // Linux impl
#elif defined(__APPLE__)
  // macOS impl
#else
  #error "Unsupported platform"
#endif
```

## Memory Management Rules

### Allocation Discipline

1. **No `new`/`delete`**: Use allocators or placement new
2. **No `malloc`/`free` Directly**: Go through `AstralAllocator` interface
3. **Prefer Stack**: Use stack for small, fixed-size objects
4. **Frame Allocators**: For short-lived objects tied to a frame/request
5. **Object Pools**: For recycled objects (tokens, callback entries)

### Memory Lifetime

```cpp
// GOOD: Explicit lifetime, clear ownership
struct Session {
  FrameAllocator allocator;

  void reset() {
    allocator.reset(); // All frame memory freed
  }
};

// BAD: Hidden allocation
std::string token_to_string(uint32_t token) {
  return tokenizer.decode(token); // Heap alloc!
}

// GOOD: Caller-provided buffer
void token_to_utf8(uint32_t token, Span<uint8_t> out, uint32_t* out_len) {
  // Write to out, set out_len
}
```

### Alignment

- **Default**: 16 bytes (SIMD-friendly)
- **Cache Line**: 64 bytes for shared atomics
- **Page**: 4096 bytes for mmap'd regions
- **Huge Page**: 2MB or 1GB if `enable_hugepages`

```cpp
// Use alignas for static alignment
struct alignas(128) MpmcNode {
  std::atomic<uint64_t> ticket;
  Token data;
};

// Use allocator alignment param for dynamic
void* alloc(size_t size, size_t align = 16);
```

## Concurrency Rules

### Atomics

- **Memory Order**: Be explicit; never use `seq_cst` without justification
- **Load**: `memory_order_acquire` for reading shared state
- **Store**: `memory_order_release` for publishing shared state
- **RMW**: `acq_rel` for read-modify-write (fetch_add, exchange)
- **Relaxed**: Only for statistics or non-synchronizing counters

```cpp
// GOOD: Explicit memory order
uint64_t head = head_.load(std::memory_order_acquire);
tail_.store(new_tail, std::memory_order_release);

// BAD: Default seq_cst (too strong, slower)
uint64_t head = head_.load();
```

### Synchronization Guidelines

1. **ABA Protection**: Use tagged pointers or epoch-based reclamation
2. **Bounded Queues**: Prefer bounded over unbounded (no allocation in enqueue)
3. **Avoid Compare-and-Swap**: v0.1 code avoids CAS; design around per-slot sequence + tickets or small spinlocks
4. **Progress Guarantee**: Prefer bounded waiting with WFE/SEV or clear timeouts

```cpp
// GOOD: CAS-free MPMC enqueue (ticket + per-slot sequence)
uint64_t pos = enqueue_pos_.fetch_add(1, std::memory_order_relaxed);
Slot& slot = buffer_[pos & mask];
while (slot.seq.load(std::memory_order_acquire) != pos) {
  cpu_wait_for_event(); // WFE on ARM, pause fallback elsewhere
}
slot.data = item;
slot.seq.store(pos + 1, std::memory_order_release);
cpu_signal_event();
```

### Thread Safety

- **Document Ownership**: Each class must document thread-safety in header comment
- **Immutable After Init**: Prefer immutable config structs
- **Per-Thread State**: Use thread-local for hot accumulators (tokens/sec counters)

```cpp
// Thread-safe: external synchronization required
// Not thread-safe: single-threaded use only
class FrameAllocator {
  // NOT thread-safe; use one per thread/session
};

// Thread-safe: lock-free enqueue/dequeue
class MpmcQueue {
  // Thread-safe; multiple producers, multiple consumers
};
```

## Error Handling

### C ABI

```c
// Return AstralErr (int32_t)
AstralErr astral_session_create(const AstralSessionDesc* desc, AstralHandle* out);

// Check every call
AstralHandle session;
AstralErr err = astral_session_create(&desc, &session);
if (err != ASTRAL_OK) {
  // handle error
}
```

### C++ Wrapper

```cpp
// Result<T> monad (internal)
template<typename T>
struct Result {
  T value;
  AstralErr error;

  bool ok() const { return error == ASTRAL_OK; }
  T unwrap() const { if (!ok()) throw AstralException(error); return value; }
};

// Usage
auto result = session.decode();
if (!result.ok()) {
  log_error(result.error);
  return;
}
```

### Logging

- **Non-Blocking**: Never block on log callback
- **Levels**: ERROR, WARN, INFO, DEBUG, TRACE
- **Rate Limiting**: Drop logs if callback is slow

```cpp
void log(LogLevel level, const char* fmt, ...) {
  if (!log_cb_) return;
  // Format to thread-local buffer
  // Call log_cb_; if it doesn't return within 1ms, drop
}
```

## UTF-8 and String Handling

### Span-Based Strings

```cpp
// Immutable span
struct Span {
  const uint8_t* data;
  uint32_t len;

  // No NUL terminator assumed
  bool equals(Span other) const;
  Span slice(uint32_t start, uint32_t end) const;
};

// Mutable span (rare; typically write-only)
struct MutSpan {
  uint8_t* data;
  uint32_t len;
};
```

### String Builder

```cpp
// Append-only builder over arena
class StringBuilder {
public:
  void append(Span utf8);
  void append_u32(uint32_t val); // Integer to UTF-8
  Span freeze();                 // Snapshot current content
  void reset();                  // Reset bump pointer
};

// Usage
StringBuilder sb(allocator);
sb.append(Span::from_cstr("Token: "));
sb.append_u32(token_id);
Span result = sb.freeze(); // Zero-copy slice
```

### UTF-8 Validation

- **Incremental**: Validate on input, not on every access
- **Fast Path**: SIMD-accelerated validation for large buffers
- **Fallback**: Scalar loop for small inputs or unaligned buffers

```cpp
bool validate_utf8(Span input) {
  // Check for invalid sequences, overlong encodings, etc.
}
```

## Platform Abstraction

### Virtual Memory

```cpp
// src/platform/vm.h
namespace astral::platform {

void* vm_reserve(size_t size);
void  vm_commit(void* addr, size_t size);
void  vm_decommit(void* addr, size_t size);
void  vm_release(void* addr, size_t size);

bool  vm_try_hugepages(void* addr, size_t size);

} // namespace astral::platform
```

### Atomics and Intrinsics

```cpp
// src/platform/atomics.h
namespace astral::platform {

// Cache line size (runtime detect)
uint32_t cache_line_size();

// Pause/yield for spin loops
void cpu_pause();

// Compiler fences (no-op on hardware, barrier for optimizer)
void compiler_fence();

} // namespace astral::platform
```

### SIMD

- **Detect at Runtime**: Use CPUID / getauxval for feature detection
- **Fallback**: Always provide scalar implementation
- **Intrinsics**: Prefer compiler intrinsics over inline asm

```cpp
// Embed dispatch
void embed_batch_impl_avx2(Span<float> vectors);
void embed_batch_impl_neon(Span<float> vectors);
void embed_batch_impl_scalar(Span<float> vectors);

// Runtime dispatch
using EmbedBatchFn = void (*)(Span<float>);
EmbedBatchFn embed_batch = detect_simd_and_select();
```

## Testing and Validation

### Unit Tests

```cpp
// Minimal test framework (no external deps)
#define TEST(name) \
  void test_##name(); \
  static TestRegistrar reg_##name(#name, test_##name); \
  void test_##name()

// Assertions
#define ASSERT_EQ(a, b) if ((a) != (b)) { test_fail(__FILE__, __LINE__, #a " != " #b); }
#define ASSERT_TRUE(x) if (!(x)) { test_fail(__FILE__, __LINE__, #x); }

// Example
TEST(frame_allocator_basic) {
  FrameAllocator alloc(1 << 20); // 1 MB
  void* p1 = alloc.alloc(64, 16);
  void* p2 = alloc.alloc(128, 16);
  ASSERT_TRUE(p1 != nullptr);
  ASSERT_TRUE(p2 != nullptr);
  alloc.reset();
  void* p3 = alloc.alloc(64, 16);
  ASSERT_EQ(p1, p3); // Same address after reset
}
```

### Benchmarks

```cpp
// Microbenchmark harness
void benchmark_mpmc_enqueue() {
  MpmcQueue<Token, 1024> queue;

  auto start = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < 1'000'000; ++i) {
    queue.enqueue_wait(Token{i});
  }
  auto end = std::chrono::steady_clock::now();

  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double ops_per_sec = 1'000'000.0 / (elapsed_ms / 1000.0);
  printf("MPMC enqueue: %.2f Mops/s\n", ops_per_sec / 1e6);
}
```

## Documentation Standards

### Header Comments

```cpp
// Every public function must have a doc comment
/// Loads a GGUF model from disk and returns an opaque handle.
///
/// @param desc Model configuration (path, GPU layers, context size).
/// @param out_model Output handle; valid only if return is ASTRAL_OK.
/// @return ASTRAL_OK on success; ASTRAL_E_NOMEM if reserve failed;
///         ASTRAL_E_INVALID if desc is null or path is invalid.
///
/// Thread-safety: Safe to call from multiple threads.
/// Memory: Allocates from global reserve; does not use engine allocator.
AstralErr astral_model_load(const AstralModelDesc* desc, AstralHandle* out_model);
```

### Implementation Comments

- **Why, Not What**: Explain non-obvious decisions, not the syntax
- **Performance Notes**: Document hot path optimizations
- **Concurrency**: Explain memory ordering choices

```cpp
// GOOD: Explains rationale
// Use relaxed load because we only increment for stats; no synchronization needed.
uint64_t count = counter_.load(std::memory_order_relaxed);

// BAD: States the obvious
// Load the counter
uint64_t count = counter_.load(std::memory_order_relaxed);
```

## Dependency Policy

### Allowed Dependencies

- **llama.cpp**: Core inference backend (vendored, specific commit pinned)
- **C Standard Library**: malloc/free for fallback, memcpy, etc. (only via wrappers)
- **Platform APIs**: POSIX, Win32, CoreFoundation (only in `src/platform/`)

### Forbidden Dependencies

- **STL Containers**: No `std::vector`, `std::map`, `std::string`, etc.
- **STL Allocators**: No `std::allocator`
- **Boost**: Too large, allocation-heavy
- **Threading Libraries**: No `std::thread` (use platform thread APIs)
- **JSON Libraries**: Implement minimal parser if needed, or accept at init only

### External Approval Required

Any dependency not listed above must be approved by the project lead. Justify:
1. Why it's needed
2. Binary size impact
3. Allocation behavior
4. Platform support (armv7, arm64, x86-64)

## Code Review Checklist

Before submitting code, ensure:

- [ ] No dynamic allocations in hot path (profiled)
- [ ] All functions documented (C ABI + internal)
- [ ] Error codes checked (C ABI calls)
- [ ] Memory ordering explicit (atomics)
- [ ] UTF-8 validated on input
- [ ] Platform abstraction used (no `#ifdef` in core logic)
- [ ] Tests added (unit + integration)
- [ ] Benchmarks run (no regressions)
- [ ] Clang-format applied
- [ ] No compiler warnings (-Wall -Wextra -Werror)

## Formatting

Use `.clang-format`:

```yaml
BasedOnStyle: LLVM
IndentWidth: 2
ColumnLimit: 100
PointerAlignment: Left
AllowShortFunctionsOnASingleLine: Inline
BreakBeforeBraces: Attach
```

Run before commit:

```bash
clang-format -i include/*.h src/**/*.cpp
```

## Performance Mantras

1. **Measure, Don't Guess**: Profile before optimizing
2. **Allocations Are Bugs**: Every allocation in the hot path must be justified in writing
3. **Cache Lines Matter**: Align hot structures to 64 bytes
4. **Branches Are Slow**: Use `ASTRAL_LIKELY` / `ASTRAL_UNLIKELY` after profiling
5. **Data-Oriented Design**: Think in terms of data flow, not objects

### Branch Hints: `ASTRAL_LIKELY` and `ASTRAL_UNLIKELY`

**When to Use**: Mark branches where one path is taken >95% of the time. Modern CPUs can predict up to 4096 unique branches; beyond that, misprediction rates increase dramatically.

**Performance Impact**:
- Correctly predicted branch: 0.7ns
- Mispredicted branch: 3.7ns (5x slower!)
- Branch with hint: Helps compiler optimize code layout (hot code together, cold code separate)

**Evidence**: From less_slow.cpp benchmarks on Intel Sapphire Rapids.

#### Error Paths (Boundary Code Only)

```cpp
// GOOD: Mark rare boundary failures before entering the hot loop.
AstralErr decode_token_boundary(const uint8_t* input, uint32_t len, Token* out) {
    if (input == nullptr) ASTRAL_UNLIKELY {
        return ASTRAL_E_INVALID;
    }

    if (len == 0) ASTRAL_UNLIKELY {
        return ASTRAL_E_INVALID;
    }

    if (len > MAX_TOKEN_LEN) ASTRAL_UNLIKELY {
        return ASTRAL_E_INVALID;
    }

    // The parser receives validated input; do not repeat these guards inside it.
    *out = parse_token(input, len);
    return ASTRAL_OK;
}
```

#### Success vs Failure Patterns

```cpp
// GOOD: Mark rare success as unlikely.
bool try_claim_slot(std::atomic<uint64_t>& slot) {
    // Claim via exchange. Returns true only for the first claimer.
    uint64_t prev = slot.exchange(1, std::memory_order_acquire);
    return prev == 0;
}

// GOOD: Mark common success as likely.
bool validate_token_id(uint32_t id, uint32_t vocab_size) {
    if (id < vocab_size) ASTRAL_LIKELY {
        return true;  // >99% of tokens are valid
    }
    return false;
}
```

#### When NOT to Use

```cpp
// BAD: Unpredictable branch (token sampling with high temperature)
// Don't use hints - use branchless CMOV instead
if (temperature > 0.8f) {  // 50/50 split, don't hint
    return sample_random(logits);
}

// BAD: Balanced branch
if (user_preference == MODE_A) {  // Could be 50/50
    // No hint - let branch predictor learn
}

// BAD: Already obvious to compiler
for (size_t i = 0; i < len; ++i) {
    if (i == 0) ASTRAL_UNLIKELY {  // Compiler already knows this!
        // Loop entry optimization
    }
}
```

#### Guidelines

| Branch Pattern | Hint | Justification |
|---------------|------|---------------|
| Null pointer checks | `ASTRAL_UNLIKELY` | >99% not null |
| Bounds checks (valid input) | `ASTRAL_LIKELY` | >95% in bounds |
| Error returns | `ASTRAL_UNLIKELY` | >95% success |
| MPMC queue operations | None | Unpredictable under contention |
| Token sampling (temp >0.5) | None | Use branchless CMOV |
| Early loop exit | `ASTRAL_UNLIKELY` | Most loops complete |

### Trivially Copyable Types

**Why It Matters**: Trivially copyable types enable `memcpy` optimization (single `rep movsb` instruction) instead of element-wise copy. **10-20% speedup** for bulk operations.

**Requirement**: All hot-path structs (TokenBuffer, EmbeddingSlice, QueueSlot) MUST be trivially copyable.

#### Verification with Static Assertions

```cpp
// REQUIRED: Add static assertion for all C ABI types
struct TokenBuffer {
    uint32_t* ids;
    uint32_t len;
    uint32_t capacity;
};

// This ensures std::vector and other STL containers can use memcpy
static_assert(std::is_trivially_copyable_v<TokenBuffer>,
              "TokenBuffer must be trivially copyable for bulk operations");

// Example: EmbeddingSlice
struct EmbeddingSlice {
    float* data;
    uint32_t dim;
    uint32_t stride;
};

static_assert(std::is_trivially_copyable_v<EmbeddingSlice>,
              "EmbeddingSlice must be trivially copyable for efficient transfer");
```

#### What Makes a Type Trivially Copyable?

**Allowed**:
- Plain data members (POD types: int, float, pointers)
- `= default` copy constructor/assignment
- No virtual functions
- No user-defined destructor

**Forbidden**:
- `std::string` (has destructor)
- `std::vector` (has destructor)
- `std::pair` (NOT trivially copyable in most STL implementations!)
- `std::tuple` (NOT trivially copyable)
- User-defined copy constructor
- User-defined assignment operator
- Virtual functions

```cpp
// GOOD: Trivially copyable
struct GoodStruct {
    uint32_t a;
    float b;
    void* ptr;
};
static_assert(std::is_trivially_copyable_v<GoodStruct>);

// BAD: NOT trivially copyable (std::pair)
struct BadStruct {
    std::pair<int, float> data;  // Fails on most compilers!
};
static_assert(!std::is_trivially_copyable_v<BadStruct>);

// GOOD: Use plain POD struct instead
struct GoodReplacement {
    int first;
    float second;
};
static_assert(std::is_trivially_copyable_v<GoodReplacement>);
```

#### Benchmark Evidence

From less_slow.cpp measurements:

| Type | Copy Latency (1M elements) | Speedup |
|------|---------------------------|---------|
| Plain POD struct | 488 μs | Baseline |
| `std::pair<int, float>` | 600 μs | **20% slower** |
| `std::tuple<int, float>` | 600 μs | **20% slower** |

**Recommendation**: Always use plain POD structs for hot-path data. Verify with `static_assert`.

### Compiler Optimization Flags

#### Release Build Flags (CMakeLists.txt)

```cmake
# Performance-critical flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Function alignment: 2-5% I-cache miss reduction
    # Why: Aligns hot functions to 32-byte boundaries for better fetch
    # Trade-off: +5-10% binary size
    add_compile_options(-falign-functions=32)

    # Loop alignment: Improves branch prediction
    add_compile_options(-falign-loops=32)

    # Link-time optimization: Enables cross-translation-unit inlining
    # Why: ASTRAL_FORCE_INLINE works across compilation units
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)  # Enables LTO

    # Platform-specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # AVX2 baseline for modern x86
        add_compile_options(-mavx2 -mfma)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        # ARM NEON is always available on ARMv8
        add_compile_options(-march=armv8-a+fp+simd)
    endif()
endif()
```

#### Flags to AVOID

```cmake
# FORBIDDEN: Fast-math breaks IEEE 754 semantics
# Risk: Numerical instability in softmax/logits
# add_compile_options(-ffast-math)  # NEVER

# FORBIDDEN: Unsafe stack optimization
# Risk: Buffer overflows harder to detect
# add_compile_options(-fomit-frame-pointer)  # Debug nightmare

# FORBIDDEN: Architecture-specific without runtime detection
# Risk: Crashes on older CPUs (AVX-512 not universal)
# add_compile_options(-mavx512f)  # Use runtime dispatch instead
```

#### Debug Build Optimizations

```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Enable sanitizers
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)

    # Add stack protector
    add_compile_options(-fstack-protector-strong)

    # Keep debug symbols
    add_compile_options(-g3)
endif()
```

## Anti-Patterns

### Forbidden

```cpp
// BAD: Hidden allocation
std::string format_token(uint32_t id) { ... }

// BAD: Undefined memory ordering
std::atomic<int> counter;
counter++; // What memory order?

// BAD: Exception across C ABI
extern "C" AstralErr foo() {
  throw std::runtime_error("oops"); // NEVER
}

// BAD: STL container
std::vector<Token> tokens; // Use custom container

// BAD: Unchecked error
astral_session_create(&desc, &session); // What if it fails?
```

### Encouraged

```cpp
// GOOD: Explicit buffer
void format_token(uint32_t id, MutSpan out, uint32_t* out_len);

// GOOD: Explicit memory order
counter_.fetch_add(1, std::memory_order_relaxed);

// GOOD: Error code propagation
extern "C" AstralErr foo() {
  if (/* error */) return ASTRAL_E_INVALID;
  return ASTRAL_OK;
}

// GOOD: Custom container with allocator
Array<Token> tokens(allocator);

// GOOD: Error handling
AstralErr err = astral_session_create(&desc, &session);
if (err != ASTRAL_OK) { /* handle */ }
```

## Header Hygiene

### Include What You Use (IWYU)

- **Principle**: Every header must include exactly what it uses, nothing more
- **Self-Contained**: Headers must be self-contained (compile standalone)
- **No Transitive Dependencies**: Never rely on transitive includes

```cpp
// GOOD: Explicit includes
#pragma once
#include <cstdint>
#include "memory/frame_allocator.hpp"

class Session {
  FrameAllocator allocator_;
};
```

```cpp
// BAD: Implicit dependency
#pragma once
// Missing #include "memory/frame_allocator.hpp"

class Session {
  FrameAllocator allocator_; // Compile error if header used alone
};
```

### Single-Include Headers

All headers must be usable with a single `#include` statement:

```cpp
// user.cpp
#include "astral/session.hpp" // Should compile without other includes
```

### Include Guards

Use `#pragma once` (widely supported, cleaner):

```cpp
// GOOD
#pragma once

struct Token { ... };
```

```cpp
// BAD: Traditional include guards (verbose, error-prone)
#ifndef ASTRAL_TOKEN_H
#define ASTRAL_TOKEN_H

struct Token { ... };

#endif // ASTRAL_TOKEN_H
```

### No `using namespace` in Headers

Never pollute the global namespace from headers:

```cpp
// BAD: Pollutes namespace for all includers
#pragma once
using namespace std; // NEVER

void foo(string s); // Ambiguous
```

```cpp
// GOOD: Explicit namespace in headers
#pragma once

void foo(std::string s); // Clear (or better: use Span, not std::string)
```

```cpp
// GOOD: using namespace in .cpp files (local scope)
// session.cpp
#include "session.hpp"

using namespace astral::memory; // OK in .cpp

void Session::init() {
  FrameAllocator alloc(...); // No need for astral::memory::
}
```

## Time Measurement

### Monotonic Clock Only

**Never use wall-clock time for performance measurement** (affected by NTP, DST, manual adjustments).

```cpp
// BAD: Wall-clock time (can jump backwards)
auto start = std::chrono::system_clock::now();
// ... work ...
auto end = std::chrono::system_clock::now();
auto duration = end - start; // Unreliable!
```

```cpp
// GOOD: Monotonic clock
auto start = std::chrono::steady_clock::now();
// ... work ...
auto end = std::chrono::steady_clock::now();
auto duration = end - start; // Reliable
```

### Prefer Hardware Counters

For hot paths, use platform-specific high-resolution timers:

#### x86-64: `rdtsc`

```cpp
inline uint64_t read_tsc() {
#if defined(__x86_64__) || defined(_M_X64)
  uint32_t lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
#else
  #error "rdtsc not available"
#endif
}

uint64_t start = read_tsc();
// ... work ...
uint64_t end = read_tsc();
uint64_t cycles = end - start;
```

#### macOS: `mach_absolute_time`

```cpp
#include <mach/mach_time.h>

uint64_t start = mach_absolute_time();
// ... work ...
uint64_t end = mach_absolute_time();

// Convert to nanoseconds
mach_timebase_info_data_t info;
mach_timebase_info(&info);
uint64_t nanos = (end - start) * info.numer / info.denom;
```

#### Windows: `QueryPerformanceCounter`

```cpp
#include <windows.h>

LARGE_INTEGER freq, start, end;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start);
// ... work ...
QueryPerformanceCounter(&end);

double seconds = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
```

#### Linux: `clock_gettime(CLOCK_MONOTONIC)`

```cpp
#include <time.h>

struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);
// ... work ...
clock_gettime(CLOCK_MONOTONIC, &end);

uint64_t nanos = (end.tv_sec - start.tv_sec) * 1000000000ULL +
                 (end.tv_nsec - start.tv_nsec);
```

### Avoid `std::chrono` in Hot Paths

`std::chrono` has overhead (virtual calls, division). Use raw platform APIs for critical paths:

```cpp
// GOOD: Direct platform API
uint64_t start = read_tsc();
decode_tokens();
uint64_t cycles = read_tsc() - start;
```

```cpp
// BAD: std::chrono overhead in hot path
auto start = std::chrono::high_resolution_clock::now();
decode_tokens();
auto end = std::chrono::high_resolution_clock::now();
// Division, nanosecond conversion, potential virtual calls
```

### Time Utilities

Provide platform-abstracted timing functions:

```cpp
// src/platform/time.h
namespace astral::platform {

// High-resolution monotonic timestamp (platform-specific units)
uint64_t monotonic_now();

// Convert timestamp difference to nanoseconds
uint64_t to_nanos(uint64_t ticks);

// Convert timestamp difference to seconds
double to_seconds(uint64_t ticks);

} // namespace astral::platform
```

## Summary

Write code as if every cycle counts, because in game engines, it does. Prefer simplicity and explicitness over cleverness. When in doubt, measure.
