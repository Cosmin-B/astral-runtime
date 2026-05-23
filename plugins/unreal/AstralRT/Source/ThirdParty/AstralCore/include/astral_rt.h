/**
 * astral_rt.h - Astral Runtime C ABI
 *
 * Stable C interface for the Astral native runtime.
 * All functions return error codes; no exceptions.
 * All strings are UTF-8 spans (pointer + length).
 *
 * Version: 0.1.0
 * License: See LICENSE and NOTICE at the repository root.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

// ============================================================================
// API Export and Calling Convention
// ============================================================================

// DLL export/import for Windows
#if defined(_WIN32) || defined(_WIN64)
  #ifdef ASTRAL_BUILD_DLL
    #define ASTRAL_API __declspec(dllexport)
  #elif defined(ASTRAL_USE_DLL)
    #define ASTRAL_API __declspec(dllimport)
  #else
    #define ASTRAL_API
  #endif
  #define ASTRAL_CALL __cdecl
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define ASTRAL_API __attribute__((visibility("default")))
  #else
    #define ASTRAL_API
  #endif
  #define ASTRAL_CALL
#endif

// ============================================================================
// Version
// ============================================================================

#define ASTRAL_VERSION_MAJOR 0
#define ASTRAL_VERSION_MINOR 1
#define ASTRAL_VERSION_PATCH 0

/**
 * Get Astral runtime version.
 * Thread-safety: Safe to call from any thread.
 *
 * @param major Output: major version (may be NULL)
 * @param minor Output: minor version (may be NULL)
 * @param patch Output: patch version (may be NULL)
 */
ASTRAL_API void ASTRAL_CALL astral_version(uint32_t* major, uint32_t* minor, uint32_t* patch);

// ============================================================================
// Common Types
// ============================================================================

/**
 * UTF-8 string span (immutable).
 * No NUL terminator assumed.
 *
 *  Explicit padding ensures consistent layout across 32-bit and 64-bit platforms.
 * On 64-bit: sizeof(ptr)=8, sizeof(uint32_t)=4, padding=4 → total=16 bytes
 * On 32-bit: sizeof(ptr)=4, sizeof(uint32_t)=4, padding=0 → total=8 bytes
 */
typedef struct AstralSpanU8 {
    const uint8_t* data;
    uint32_t len;
#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
    uint32_t _padding;  // Explicit padding for 64-bit platforms
#endif
} AstralSpanU8;

/**
 * UTF-8 string span (mutable).
 * Used for output buffers.
 *
 *  Explicit padding ensures consistent layout across 32-bit and 64-bit platforms.
 */
typedef struct AstralMutSpanU8 {
    uint8_t* data;
    uint32_t len;
#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
    uint32_t _padding;  // Explicit padding for 64-bit platforms
#endif
} AstralMutSpanU8;

typedef struct AstralTokenizeRequest {
    AstralSpanU8 text;
    uint8_t add_special;
    uint8_t parse_special;
    uint16_t _reserved;
} AstralTokenizeRequest;

/**
 * Opaque handle (model, session, embedder).
 *
 * v0.1 ABI hardening:
 * - Handles are 64-bit values encoding (type, slot index, generation).
 * - 0 is always invalid.
 */
typedef uint64_t AstralHandle;

typedef uint32_t AstralPromptSectionKind;
enum {
    ASTRAL_PROMPT_SECTION_SYSTEM = 1,
    ASTRAL_PROMPT_SECTION_TOOLS = 2,
    ASTRAL_PROMPT_SECTION_MEMORY = 3,
    ASTRAL_PROMPT_SECTION_HISTORY = 4,
    ASTRAL_PROMPT_SECTION_USER = 5,
    ASTRAL_PROMPT_SECTION_RAW = 6,
};

typedef uint32_t AstralPromptCacheEvictionPolicy;
enum {
    ASTRAL_PROMPT_CACHE_EVICT_FIFO = 0,
};

typedef uint32_t AstralPromptCacheFlags;
enum {
    ASTRAL_PROMPT_CACHE_FLAG_TRACK_STATS = 1u << 0,
};

typedef struct AstralPromptCacheDesc {
    uint32_t size;
    uint32_t max_entries;
    uint32_t max_tokens;
    uint32_t max_bytes;
    AstralPromptCacheEvictionPolicy eviction_policy;
    AstralPromptCacheFlags flags;
} AstralPromptCacheDesc;

typedef struct AstralPromptCacheKey {
    uint32_t size;
    uint32_t section_kind;
    AstralHandle model;
    uint64_t key;
    uint32_t generation;
    uint32_t _reserved0;
} AstralPromptCacheKey;

typedef struct AstralPromptCacheStats {
    uint32_t size;
    uint32_t entries;
    uint32_t max_entries;
    uint32_t tokens;
    uint32_t max_tokens;
    uint32_t bytes;
    uint32_t max_bytes;
    uint32_t _reserved0;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
} AstralPromptCacheStats;

typedef uint32_t AstralToolChoiceMode;
enum {
    ASTRAL_TOOL_CHOICE_AUTO = 0,
    ASTRAL_TOOL_CHOICE_REQUIRED = 1,
    ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL = 2,
};

typedef struct AstralToolDesc {
    uint32_t size;
    uint32_t tool_id;
    AstralSpanU8 name;
    AstralSpanU8 description;
    AstralSpanU8 json_schema;
} AstralToolDesc;

typedef struct AstralToolsetDesc {
    uint32_t size;
    uint32_t tool_count;
    AstralToolChoiceMode choice_mode;
    uint32_t _reserved0;
    const AstralToolDesc* tools;
} AstralToolsetDesc;

typedef struct AstralToolInfo {
    uint32_t size;
    uint32_t tool_id;
    AstralSpanU8 name;
    AstralSpanU8 description;
    AstralSpanU8 json_schema;
} AstralToolInfo;

typedef struct AstralToolCallResult {
    uint32_t size;
    uint32_t tool_id;
    int32_t parse_status;
    uint32_t _reserved0;
    AstralSpanU8 name;
    AstralSpanU8 arguments_json;
} AstralToolCallResult;

typedef uint32_t AstralChunkMode;
enum {
    ASTRAL_CHUNK_MODE_NONE = 0,
    ASTRAL_CHUNK_MODE_CHAR = 1,
    ASTRAL_CHUNK_MODE_WORD = 2,
    ASTRAL_CHUNK_MODE_SENTENCE = 3,
    ASTRAL_CHUNK_MODE_TOKEN = 4,
};

typedef uint32_t AstralChunkFlags;
enum {
    ASTRAL_CHUNK_FLAG_KEEP_EMPTY = 1u << 0,
};

typedef struct AstralChunkerDesc {
    uint32_t size;
    AstralChunkMode mode;
    uint32_t max_units;
    uint32_t overlap_units;
    uint32_t document_id;
    uint32_t group_id;
    AstralChunkFlags flags;
    uint32_t _reserved0;
    AstralSpanU8 delimiters;
} AstralChunkerDesc;

typedef struct AstralChunkRange {
    uint32_t size;
    uint32_t document_id;
    uint32_t chunk_id;
    uint32_t group_id;
    uint32_t byte_begin;
    uint32_t byte_end;
    uint32_t token_begin;
    uint32_t token_end;
} AstralChunkRange;

typedef uint32_t AstralMemoryMetric;
enum {
    ASTRAL_MEMORY_METRIC_DOT = 0,
    ASTRAL_MEMORY_METRIC_COSINE = 1,
    ASTRAL_MEMORY_METRIC_L2 = 2,
};

typedef uint32_t AstralMemoryIndexKind;
enum {
    ASTRAL_MEMORY_INDEX_FLAT = 0,
    ASTRAL_MEMORY_INDEX_GRAPH = 1,
};

typedef uint32_t AstralMemoryStorageKind;
enum {
  ASTRAL_MEMORY_STORAGE_F32 = 0,
  ASTRAL_MEMORY_STORAGE_Q8 = 1,
  ASTRAL_MEMORY_STORAGE_F6_E2M3 = 2,
};

enum {
    ASTRAL_MEMORY_GROUP_ANY = 0xFFFFFFFFu,
};

typedef struct AstralMemoryIndexDesc {
    uint32_t size;
    uint32_t dim;
    uint32_t capacity;
    AstralMemoryMetric metric;
    AstralMemoryIndexKind index_kind;
    uint32_t graph_neighbors;
    /* Graph construction expansion for ASTRAL_MEMORY_INDEX_GRAPH; 0 selects the runtime default. */
    uint32_t graph_search;
    /* Default query expansion for ASTRAL_MEMORY_INDEX_GRAPH; 0 selects graph_search. */
    uint32_t graph_query_search;
    AstralMemoryStorageKind storage_kind;
} AstralMemoryIndexDesc;

typedef struct AstralMemoryRecord {
    uint32_t size;
    uint32_t group_id;
    uint64_t key;
    uint32_t document_id;
    uint32_t chunk_id;
    uint32_t flags;
    uint32_t _reserved0;
} AstralMemoryRecord;

typedef struct AstralMemorySearchDesc {
    uint32_t size;
    uint32_t top_k;
    uint32_t group_id;
    uint32_t flags;
    /* Per-query graph expansion; 0 uses the index construction expansion. */
    uint32_t graph_search;
} AstralMemorySearchDesc;

typedef struct AstralMemorySearchResult {
    uint32_t size;
    uint32_t group_id;
    uint64_t key;
    uint32_t document_id;
    uint32_t chunk_id;
    float score;
    uint32_t flags;
} AstralMemorySearchResult;

typedef struct AstralMemoryStats {
    uint32_t size;
    uint32_t dim;
    uint32_t capacity;
    uint32_t count;
    AstralMemoryMetric metric;
    AstralMemoryIndexKind index_kind;
    uint32_t graph_neighbors;
    uint32_t graph_search;
    uint32_t graph_query_search;
    uint32_t graph_levels;
    AstralMemoryStorageKind storage_kind;
    uint64_t vector_bytes;
    uint64_t metadata_bytes;
    uint64_t graph_bytes;
    uint64_t graph_edges;
    uint64_t graph_base_edges;
    uint64_t graph_upper_edges;
    uint64_t graph_build_score_evals;
    uint64_t graph_build_candidate_visits;
    uint64_t total_bytes;
    uint64_t save_bytes;
} AstralMemoryStats;

typedef struct AstralMemorySnapshotInfo {
  uint32_t size;
  uint32_t version;
  uint32_t dim;
  uint32_t count;
  AstralMemoryMetric metric;
  AstralMemoryIndexKind index_kind;
  AstralMemoryStorageKind storage_kind;
  uint32_t flags;
  uint64_t record_offset;
  uint64_t record_stride;
  uint64_t vector_offset;
  uint64_t vector_stride;
  uint64_t scale_offset;
  uint64_t scale_stride;
  uint64_t graph_offset;
  uint64_t graph_bytes;
  uint64_t total_bytes;
} AstralMemorySnapshotInfo;

// Compile-time validation: Ensure struct sizes are correct
// Use static_assert for C++ and _Static_assert for C
#ifdef __cplusplus
#define ASTRAL_STATIC_ASSERT(expr, msg) static_assert(expr, msg)
#else
#define ASTRAL_STATIC_ASSERT(expr, msg) _Static_assert(expr, msg)
#endif

#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
ASTRAL_STATIC_ASSERT(sizeof(AstralSpanU8) == 16, "AstralSpanU8 must be 16 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMutSpanU8) == 16, "AstralMutSpanU8 must be 16 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralPromptCacheKey) == 32,
                     "AstralPromptCacheKey must be 32 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralToolDesc) == 56, "AstralToolDesc must be 56 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralToolsetDesc) == 24,
                     "AstralToolsetDesc must be 24 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralToolInfo) == 56, "AstralToolInfo must be 56 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralToolCallResult) == 48,
                     "AstralToolCallResult must be 48 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralChunkerDesc) == 48,
                     "AstralChunkerDesc must be 48 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralChunkRange) == 32, "AstralChunkRange must be 32 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemoryIndexDesc) == 36,
                     "AstralMemoryIndexDesc must be 36 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemoryRecord) == 32,
                     "AstralMemoryRecord must be 32 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemorySearchDesc) == 20,
                     "AstralMemorySearchDesc must be 20 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemorySearchResult) == 32,
                     "AstralMemorySearchResult must be 32 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemoryStats) == 128,
                     "AstralMemoryStats must be 128 bytes on 64-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemorySnapshotInfo) == 104,
                     "AstralMemorySnapshotInfo must be 104 bytes on 64-bit");
#else
ASTRAL_STATIC_ASSERT(sizeof(AstralSpanU8) == 8, "AstralSpanU8 must be 8 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMutSpanU8) == 8, "AstralMutSpanU8 must be 8 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralPromptCacheKey) == 32,
                     "AstralPromptCacheKey must be 32 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralToolDesc) == 32, "AstralToolDesc must be 32 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralToolsetDesc) == 20,
                     "AstralToolsetDesc must be 20 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralToolInfo) == 32, "AstralToolInfo must be 32 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralToolCallResult) == 32,
                     "AstralToolCallResult must be 32 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralChunkerDesc) == 40,
                     "AstralChunkerDesc must be 40 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralChunkRange) == 32, "AstralChunkRange must be 32 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemoryIndexDesc) == 36,
                     "AstralMemoryIndexDesc must be 36 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemoryRecord) == 32,
                     "AstralMemoryRecord must be 32 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemorySearchDesc) == 20,
                     "AstralMemorySearchDesc must be 20 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemorySearchResult) == 32,
                     "AstralMemorySearchResult must be 32 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemoryStats) == 128,
                     "AstralMemoryStats must be 128 bytes on 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralMemorySnapshotInfo) == 104,
                     "AstralMemorySnapshotInfo must be 104 bytes on 32-bit");
#endif

// ============================================================================
// Media Types (Vision / Audio)
// ============================================================================

typedef uint32_t AstralImageFormat;
enum {
    ASTRAL_IMAGE_FORMAT_RGB8 = 0,   /** RGB, 8-bit per channel */
    ASTRAL_IMAGE_FORMAT_RGBA8 = 1,  /** RGBA, 8-bit per channel (alpha ignored) */
    ASTRAL_IMAGE_FORMAT_RGB_F32 = 2 /** RGB, float32 per channel (0..1 assumed) */
};

typedef uint32_t AstralAudioFormat;
enum {
    ASTRAL_AUDIO_FORMAT_F32 = 0, /** float32 PCM */
    ASTRAL_AUDIO_FORMAT_I16 = 1  /** int16 PCM */
};

typedef uint32_t AstralMediaFlags;
enum {
    ASTRAL_MEDIA_FLAG_USE_GPU = 1u << 0,  /** Ask the media projector to use its GPU path during init. */
    ASTRAL_MEDIA_FLAG_WARMUP  = 1u << 1   /** Run media encoder warmup during init. */
};

typedef uint32_t AstralGpuRouteFlags;
enum {
    ASTRAL_GPU_ROUTE_NONE = 0,
    ASTRAL_GPU_ROUTE_DEVICE = 1u << 0,       /** Use gpu_device as an explicit device index. */
    ASTRAL_GPU_ROUTE_DEVICE_MASK = 1u << 1,  /** Use gpu_device_mask bitset to choose a device. */
    ASTRAL_GPU_ROUTE_STREAM = 1u << 2        /** Use gpu_stream when supported by the backend. */
};

/**
 * Image descriptor (size-tagged).
 *
 * Notes:
 * - `size` must be set to sizeof(AstralImageDesc).
 * - `row_stride` is in bytes; 0 means tightly packed.
 * - GPU routing fields are advisory backend inputs; set gpu_route_flags for fields the caller wants consumed.
 */
typedef struct AstralImageDesc {
    uint32_t size;        // sizeof(AstralImageDesc)
    AstralImageFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t row_stride;  // bytes; 0 = width * bytes_per_pixel
    uint32_t flags;
    AstralSpanU8 pixels;  // raw pixel bytes
    int32_t gpu_device;         // requested backend device index when ASTRAL_GPU_ROUTE_DEVICE is set
    uint32_t gpu_route_flags;   // AstralGpuRouteFlags
    uint64_t gpu_device_mask;   // requested device bitset when ASTRAL_GPU_ROUTE_DEVICE_MASK is set
    void* gpu_stream;           // backend-specific stream handle when ASTRAL_GPU_ROUTE_STREAM is set
} AstralImageDesc;

/**
 * Audio descriptor (size-tagged).
 *
 * Notes:
 * - `size` must be set to sizeof(AstralAudioDesc).
 * - `frame_count` is per-channel frames.
 * - GPU routing fields are advisory backend inputs; set gpu_route_flags for fields the caller wants consumed.
 */
typedef struct AstralAudioDesc {
    uint32_t size;         // sizeof(AstralAudioDesc)
    AstralAudioFormat format;
    uint32_t channels;
    uint32_t sample_rate;
    uint64_t frame_count;  // per-channel frames
    AstralSpanU8 samples;  // raw PCM bytes
    uint32_t flags;
    uint32_t _padding0;
    int32_t gpu_device;         // requested backend device index when ASTRAL_GPU_ROUTE_DEVICE is set
    uint32_t gpu_route_flags;   // AstralGpuRouteFlags
    uint64_t gpu_device_mask;   // requested device bitset when ASTRAL_GPU_ROUTE_DEVICE_MASK is set
    void* gpu_stream;           // backend-specific stream handle when ASTRAL_GPU_ROUTE_STREAM is set
} AstralAudioDesc;

// ---------------------------------------------------------------------------
// Tunables / Limits
// ---------------------------------------------------------------------------

// Maximum number of per-token logprob entries supported by the ABI.
// This keeps the meta side-channel fixed-size and allocation-free.
#define ASTRAL_LOGPROBS_MAX 16u

// Maximum number of LoRA/adapters attached to one session.
#define ASTRAL_SESSION_ADAPTERS_MAX 8u

/**
 * Error codes.
 * All functions return AstralErr (0 = success).
 */
typedef int32_t AstralErr;

enum {
    ASTRAL_OK = 0,          /** Success */
    ASTRAL_E_INVALID = -1,  /** Invalid parameter (null pointer, bad config) */
    ASTRAL_E_NOMEM = -2,    /** Out of memory (reserve/commit failed) */
    ASTRAL_E_BUSY = -3,     /** Resource busy (queue full, retry) */
    ASTRAL_E_TIMEOUT = -4,  /** Operation timed out */
    ASTRAL_E_STATE = -5,    /** Invalid state (e.g., decode before feed) */
    ASTRAL_E_BACKEND = -6,  /** Backend error (llama.cpp failed) */
    ASTRAL_E_CANCELED = -7,    /** Operation canceled */
    ASTRAL_E_UNSUPPORTED = -8, /** Operation unsupported */
    ASTRAL_E_NOT_FOUND = -9,   /** Requested object or cache entry was not found */
};

// ============================================================================
// Allocator
// ============================================================================

/**
 * Allocator function: allocate `size` bytes with alignment `align`.
 * @param user User data (passed from AstralAllocator.user)
 * @param size Allocation size in bytes
 * @param align Alignment in bytes (power of 2)
 * @return Pointer to allocated memory, or NULL if out of memory
 */
typedef void* (*AstralAllocFn)(void* user, size_t size, size_t align);

/**
 * Free function: free memory allocated by AstralAllocFn.
 * @param user User data
 * @param ptr Pointer to free
 * @param size Size of allocation (for tracking)
 * @param align Alignment (for tracking)
 */
typedef void (*AstralFreeFn)(void* user, void* ptr, size_t size, size_t align);

/**
 * Custom allocator interface.
 * If alloc/free are NULL, Astral uses internal allocator.
 */
typedef struct AstralAllocator {
    AstralAllocFn alloc;  /** Allocation function */
    AstralFreeFn free;    /** Free function */
    void* user;           /** User data (passed to alloc/free) */
} AstralAllocator;

// ============================================================================
// Logging
// ============================================================================

/**
 * Log levels.
 */
enum {
    ASTRAL_LOG_ERROR = 0,
    ASTRAL_LOG_WARN = 1,
    ASTRAL_LOG_INFO = 2,
    ASTRAL_LOG_DEBUG = 3,
    ASTRAL_LOG_TRACE = 4,
};

/**
 * Logging callback.
 * WARNING: Must be non-blocking. Astral drops logs if callback is slow (>10ms).
 * @param user User data (passed from AstralInit.log_user)
 * @param level Log level (ASTRAL_LOG_*)
 * @param msg UTF-8 log message
 */
typedef void (*AstralLogFn)(void* user, int level, AstralSpanU8 msg);

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialization configuration.
 */
typedef struct AstralInit {
    AstralAllocator sys_alloc;   /** System allocator (optional; use internal if null) */
    AstralLogFn log_cb;          /** Logging callback (optional) */
    void* log_user;              /** User data for log callback */
#if !(defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8))
    // Ensure `reserve_bytes` is 8-byte aligned on 32-bit ABIs.
    uint32_t _padding0;
#endif
    uint64_t reserve_bytes;      /** Virtual memory to reserve (e.g., 2GB = 2ULL << 30) */
    uint32_t thread_count;       /** Worker threads (0 = auto-detect) */
    uint32_t numa_node;          /** NUMA node to pin memory (0xFFFFFFFF = any) */
    uint8_t enable_hugepages;    /** Try huge pages (2MB/1GB) if available */
    uint8_t _padding1[7];        /** Reserved/padding (keeps layout stable) */
} AstralInit;

// ============================================================================
// Initialization (Extended)
// ============================================================================

typedef uint32_t AstralMemoryMode;
enum {
    // Default behavior: reserve/commit virtual memory using platform VM primitives.
    ASTRAL_MEMMODE_VM = 0,

    // Embedded-friendly: use a user-provided arena (borrowed; Astral will not free it).
    ASTRAL_MEMMODE_ARENA_BORROWED = 1,

    // Embedded-friendly: Astral allocates and owns a single arena using the provided allocator.
    ASTRAL_MEMMODE_ARENA_OWNED = 2,
};

/**
 * Arena configuration for ASTRAL_MEMMODE_ARENA_*.
 *
 * The arena is used as a backing store for Astral-managed regions such as per-session scratch blocks.
 *
 * Determinism:
 * - Sessions acquire fixed-size blocks from the arena (see session_block_* fields).
 * - If no blocks remain, session creation fails with ASTRAL_E_NOMEM.
 */
typedef struct AstralArenaDesc {
    void* base; /** Arena base address (required for BORROWED; optional for OWNED). */
#if !(defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8))
    uint32_t _padding0;
#endif
    uint64_t size; /** Arena size in bytes (required). */

    AstralAllocator alloc; /** Allocator used to allocate/free the arena in OWNED mode (optional; falls back to sys_alloc). */

    uint32_t session_block_size;  /** Fixed scratch block size (bytes). 0 = default. */
    uint32_t session_block_count; /** Number of fixed scratch blocks. 0 = auto from arena size. */

    // Additional arena partitioning knobs (optional; 0 = default):
    // - _reserved[0]: worker_scratch_bytes_per_worker (default: 256 KiB; 0 disables worker scratch)
    // - _reserved[1]: runtime_heap_bytes (default: 2 MiB; 0 disables runtime arena heap)
    // - _reserved[2..3]: reserved for future use
    uint32_t _reserved[4];
} AstralArenaDesc;

/**
 * Extended initialization configuration supporting embedded-friendly arena modes.
 *
 * Notes:
 * - For ASTRAL_MEMMODE_VM, AstralInit.reserve_bytes behaves as in astral_init().
 * - For ASTRAL_MEMMODE_ARENA_*, reserve_bytes is ignored; use AstralArenaDesc.size instead.
 * - enable_hugepages is ignored in arena modes.
 * - In arena modes, if `base.thread_count` is 0, Astral defaults to 1 worker thread for determinism.
 */
typedef struct AstralInit2 {
    AstralInit base;
    AstralMemoryMode memory_mode;
    uint32_t flags;
    AstralArenaDesc arena;
} AstralInit2;

// Compile-time validation for critical structs
ASTRAL_STATIC_ASSERT(sizeof(AstralAllocator) == sizeof(void*) * 3, "AstralAllocator layout mismatch");
ASTRAL_STATIC_ASSERT(sizeof(AstralErr) == 4, "AstralErr must be 32-bit");
ASTRAL_STATIC_ASSERT(sizeof(AstralHandle) == 8, "AstralHandle must be 64-bit");

/**
 * Initialize Astral runtime.
 * Must be called once before any other Astral functions.
 * Thread-safety: Not thread-safe; call from main thread only.
 *
 * @param cfg Initialization configuration (must not be NULL)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if cfg is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_init(const AstralInit* cfg);

/**
 * Initialize Astral runtime (extended).
 *
 * Prefer this API for embedded/robotics targets that cannot rely on virtual memory.
 *
 * @param cfg Initialization configuration (must not be NULL)
 * @return ASTRAL_OK on success; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_init2(const AstralInit2* cfg);

/**
 * Shutdown Astral runtime.
 * Drains work queues, joins worker threads, releases memory.
 * All handles must be released before calling this.
 * Thread-safety: Not thread-safe; call from main thread only.
 */
ASTRAL_API void ASTRAL_CALL astral_shutdown(void);

/**
 * Check if a handle is valid.
 * Returns 1 if valid, 0 if invalid/null.
 * Thread-safety: Safe to call from any thread.
 *
 * NOTE: A handle being "valid" only means it's non-null and the internal magic matches.
 * It does not prove lifetime ownership; passing a freed handle is invalid API usage and can alias reused memory.
 *
 * @param handle Handle to validate
 * @return 1 if valid, 0 if invalid
 */
ASTRAL_API int ASTRAL_CALL astral_handle_valid(AstralHandle handle);

/**
 * Get human-readable error string for an error code.
 * Returns a static string (no need to free).
 * Thread-safety: Safe to call from any thread.
 *
 * @param err Error code (from any Astral function)
 * @return Static error message string (UTF-8, NUL-terminated)
 */
ASTRAL_API const char* ASTRAL_CALL astral_error_string(AstralErr err);

/**
 * Get last error context for the calling thread.
 *
 * Returns a pointer to a thread-local, NUL-terminated UTF-8 string.
 * The pointer remains valid until the next Astral call on the same thread.
 *
 * Thread-safety: Safe to call from any thread (thread-local storage).
 */
ASTRAL_API const char* ASTRAL_CALL astral_last_error(void);

/**
 * Clear last error context for the calling thread.
 * Thread-safety: Safe to call from any thread (thread-local storage).
 */
ASTRAL_API void ASTRAL_CALL astral_clear_last_error(void);

// ============================================================================
// Backend Plugins
// ============================================================================

/**
 * Load a backend plugin and register its provider.
 *
 * The plugin must export the symbol `astral_backend_plugin_provider_v0()` defined in
 * `astral_backend_plugin.h`. The returned provider is registered by name and can then
 * be selected via `AstralModelDesc.backend_name`.
 *
 * Thread-safety: Not thread-safe; call during startup (before concurrent model loads).
 *
 * @param path Absolute path to a shared library (UTF-8; no NUL bytes)
 * @return ASTRAL_OK on success; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_backend_load_plugin(AstralSpanU8 path);

// ============================================================================
// Model
// ============================================================================

// ============================================================================
// Model (Extended Source / Embedded-Friendly)
// ============================================================================

typedef uint32_t AstralModelSourceKind;
enum {
    ASTRAL_MODEL_SOURCE_PATH = 0,   // `model_path`
    ASTRAL_MODEL_SOURCE_MEMORY = 1, // `model_bytes`
    ASTRAL_MODEL_SOURCE_IO = 2,     // `io`
};

typedef uint32_t AstralModelPathRoot;
enum {
    ASTRAL_MODEL_PATH_ROOT_RAW = 0,
    ASTRAL_MODEL_PATH_ROOT_CONTENT = 1,
    ASTRAL_MODEL_PATH_ROOT_SAVED = 2,
    ASTRAL_MODEL_PATH_ROOT_CACHE = 3,
    ASTRAL_MODEL_PATH_ROOT_DOWNLOAD = 4,
};

typedef uint32_t AstralModelPathResolveFlags;
enum {
    ASTRAL_MODEL_PATH_RESOLVE_NONE = 0,
};

typedef struct AstralModelPathResolveDesc {
    uint32_t size;
    AstralModelPathRoot root;
    AstralSpanU8 path;
    AstralSpanU8 content_root;
    AstralSpanU8 saved_root;
    AstralSpanU8 cache_root;
    AstralSpanU8 download_root;
    AstralModelPathResolveFlags flags;
    uint32_t _reserved0;
} AstralModelPathResolveDesc;

enum {
    ASTRAL_MODEL_PATH_RESOLVE_DESC_BYTES_64 = 96,
    ASTRAL_MODEL_PATH_RESOLVE_DESC_BYTES_32 = 56,
};

/**
 * Model IO interface for embedded builds (no filesystem required).
 *
 * Contract:
 * - `read_at` must be deterministic and thread-safe for concurrent calls.
 * - `read_at` must not throw exceptions across the C ABI; return 0 on failure.
 * - Returning fewer bytes than requested is allowed (EOF); 0 means failure or EOF.
 */
typedef struct AstralModelIO {
    void* user;
    uint64_t (ASTRAL_CALL * size)(void* user);
    uint32_t (ASTRAL_CALL * read_at)(void* user, uint64_t offset, void* dst, uint32_t dst_len);
} AstralModelIO;

typedef int32_t AstralGpuSplitMode;
enum {
    ASTRAL_GPU_SPLIT_NONE  = 0, // single GPU
    ASTRAL_GPU_SPLIT_LAYER = 1, // split layers/KV across GPUs
    ASTRAL_GPU_SPLIT_ROW   = 2  // split layers/KV + tensor parallelism if supported
};

typedef uint32_t AstralGpuConfigFlags;
enum {
    ASTRAL_GPU_CFG_NONE          = 0,
    ASTRAL_GPU_CFG_MAIN          = 1u << 0,
    ASTRAL_GPU_CFG_SPLIT_MODE    = 1u << 1,
    ASTRAL_GPU_CFG_DEVICES       = 1u << 2,
    ASTRAL_GPU_CFG_DEVICE_MASK   = 1u << 3,
    ASTRAL_GPU_CFG_TENSOR_SPLIT  = 1u << 4
};

/**
 * Model configuration (size-tagged).
 *
 * Notes:
 * - `size` must be set to sizeof(AstralModelDesc).
 * - `source_kind` selects the model source.
 * - GPU config fields are honored only when their bit is set in `gpu_flags`.
 */
typedef struct AstralModelDesc {
    uint32_t size;              // sizeof(AstralModelDesc)
    AstralModelSourceKind source_kind;
    uint32_t _padding0;

    // Sources (selected by `source_kind`)
    AstralSpanU8 model_path;    // PATH
    AstralSpanU8 model_bytes;   // MEMORY
    AstralModelIO io;           // IO

    // Common options
    AstralSpanU8 backend_name;  // Optional backend override
    uint32_t gpu_layers;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_threads;
    uint8_t embeddings_only;
#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
    uint8_t _padding1[7];
#else
    uint8_t _padding1[3];
#endif

    // Optional GPU configuration (llama.cpp). Honor only when flagged in gpu_flags.
    int32_t gpu_main;            // index into selected device list
    int32_t gpu_split_mode;      // AstralGpuSplitMode
    uint32_t gpu_flags;          // AstralGpuConfigFlags
    uint32_t _padding2;
    uint64_t gpu_device_mask;    // optional device bitset (0 = ignore)
    const int32_t* gpu_devices;  // optional device index list
    uint32_t gpu_device_count;   // number of entries in gpu_devices
    uint32_t _padding3;
    const float* gpu_tensor_split; // optional tensor split ratios
    uint32_t gpu_tensor_split_count; // entries in gpu_tensor_split
    uint32_t _padding4;
} AstralModelDesc;

/**
 * Model metadata.
 */
typedef struct AstralModelInfo {
    uint32_t vocab_size;   /** Vocabulary size (tokens) */
    uint32_t ctx_size;     /** Effective context size (tokens) */
    int32_t token_bos;     /** BOS token id (or <0 if none) */
    int32_t token_eos;     /** EOS token id (or <0 if none) */
} AstralModelInfo;

// ============================================================================
// Capabilities / Limits
// ============================================================================

typedef uint64_t AstralCaps;
enum {
    ASTRAL_CAP_NONE = 0,

    // Core features (provider-agnostic).
    ASTRAL_CAP_SAMPLER_EXT = 1ull << 0,   // Extended sampler controls via AstralSamplerDesc
    ASTRAL_CAP_STOP_SEQS   = 1ull << 1,   // Stop sequence support (tokenized; core-owned)

    // Optional features (provider-dependent).
    ASTRAL_CAP_EMBEDDINGS  = 1ull << 16,  // Embeddings API supported for this model
    ASTRAL_CAP_GPU_OFFLOAD = 1ull << 17,  // Build/provider supports GPU offload
    ASTRAL_CAP_LORA        = 1ull << 18,  // LoRA/adapters supported
    ASTRAL_CAP_GRAMMAR     = 1ull << 19,  // Grammar-constrained decoding supported
    ASTRAL_CAP_LOGPROBS    = 1ull << 20,  // Per-token metadata/logprobs supported
    ASTRAL_CAP_KV_STATE    = 1ull << 21,  // Session KV/state save/load supported
    ASTRAL_CAP_SLOTS       = 1ull << 22,  // Slot/sequence selection supported
    ASTRAL_CAP_GRAMMAR_GBNF = 1ull << 23, // GBNF grammar supported
    ASTRAL_CAP_GRAMMAR_JSON_SCHEMA = 1ull << 24, // JSON schema grammar supported
    ASTRAL_CAP_IMAGE       = 1ull << 25,  // Vision input supported
    ASTRAL_CAP_AUDIO       = 1ull << 26,  // Audio input supported
    ASTRAL_CAP_MM_EMBEDDINGS = 1ull << 27 // Multimodal embeddings supported
};

typedef struct AstralModelLimits {
    uint32_t vocab_size;
    uint32_t ctx_size;
    uint32_t max_batch;  // 0 if unknown/unreported
    uint32_t max_slots;  // 0 if not applicable
} AstralModelLimits;

/**
 * Media (vision/audio) configuration for a model.
 *
 * Notes:
 * - `size` must be set to sizeof(AstralModelMediaDesc).
 * - `source_kind` mirrors AstralModelDesc sources (PATH / MEMORY / IO).
 * - GPU routing fields are advisory backend inputs; set gpu_route_flags for fields the caller wants consumed.
 */
typedef struct AstralModelMediaDesc {
    uint32_t size;              // sizeof(AstralModelMediaDesc)
    AstralModelSourceKind source_kind;
    uint32_t flags;             // AstralMediaFlags
    uint32_t image_min_tokens;  // 0 = use model metadata
    uint32_t image_max_tokens;  // 0 = use model metadata
    uint32_t _padding0;

    AstralSpanU8 media_path;    // PATH
    AstralSpanU8 media_bytes;   // MEMORY
    AstralModelIO media_io;     // IO

    // GPU routing requests for media projector/encoder.
    int32_t gpu_device;         // requested backend device index when ASTRAL_GPU_ROUTE_DEVICE is set
    uint32_t gpu_route_flags;   // AstralGpuRouteFlags
    uint64_t gpu_device_mask;   // requested device bitset when ASTRAL_GPU_ROUTE_DEVICE_MASK is set
    void* gpu_stream;           // backend-specific stream handle when ASTRAL_GPU_ROUTE_STREAM is set
} AstralModelMediaDesc;

/**
 * Media (vision/audio) info for a model.
 *
 * Notes:
 * - `size` must be set to sizeof(AstralMediaInfo).
 */
typedef struct AstralMediaInfo {
    uint32_t size;              // sizeof(AstralMediaInfo)
    uint32_t supports_image;    // 1 if vision input supported
    uint32_t supports_audio;    // 1 if audio input supported
    uint32_t audio_sample_rate; // 0 if unknown / not applicable
    uint32_t image_min_tokens;  // 0 if unknown
    uint32_t image_max_tokens;  // 0 if unknown
    uint32_t _padding0;
} AstralMediaInfo;

/**
 * Get model metadata.
 * Thread-safety: Safe to call from multiple threads.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_info(AstralHandle model, AstralModelInfo* out_info);

/**
 * Query capability bits for a model.
 *
 * Thread-safety: Safe to call from multiple threads.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_caps(AstralHandle model, AstralCaps* out_caps);

/**
 * Query model limits. Providers write 0 for limits they cannot report.
 *
 * Thread-safety: Safe to call from multiple threads.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_limits(AstralHandle model, AstralModelLimits* out_limits);

/**
 * Initialize media (vision/audio) support for a model.
 *
 * Notes:
 * - Must be called before creating sessions or embedders that use media.
 * - If media is already initialized, returns ASTRAL_E_STATE.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_media_init(AstralHandle model, const AstralModelMediaDesc* desc);

/**
 * Query media (vision/audio) info for a model.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_media_info(AstralHandle model, AstralMediaInfo* out_info);

/**
 * Get embedding vector dimension for a model.
 *
 * This is the number of float32 values produced by `astral_embed_collect()`.
 *
 * Thread-safety: Safe to call from multiple threads.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_embedding_dim(AstralHandle model, uint32_t* out_dim);

/**
 * Tokenize UTF-8 text to token ids.
 *
 * Thread-safety: Safe to call from multiple threads on the same model.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_tokenize(
    AstralHandle model,
    AstralSpanU8 text,
    int32_t* out_tokens,
    uint32_t max_tokens,
    uint8_t add_special,
    uint8_t parse_special,
    uint32_t* out_count
);

/**
 * Count tokens for UTF-8 text without writing token ids.
 *
 * Thread-safety: Safe to call from multiple threads on the same model.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_tokenize_count(
    AstralHandle model,
    AstralSpanU8 text,
    uint8_t add_special,
    uint8_t parse_special,
    uint32_t* out_count
);

/**
 * Tokenize many UTF-8 spans into one caller-owned token buffer.
 *
 * `out_offsets` must have `request_count + 1` entries. On success,
 * `out_offsets[i]` is the first token for request `i`, and the final entry is
 * the total token count. If `out_tokens == NULL` and `max_tokens == 0`, this
 * function only writes offsets and the required total token count.
 *
 * Thread-safety: Safe to call from multiple threads on the same model.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_tokenize_batch(
    AstralHandle model,
    const AstralTokenizeRequest* requests,
    uint32_t request_count,
    uint32_t* out_offsets,
    int32_t* out_tokens,
    uint32_t max_tokens,
    uint32_t* out_count
);

/**
 * Detokenize token ids to UTF-8.
 *
 * Thread-safety: Safe to call from multiple threads on the same model.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_detokenize(
    AstralHandle model,
    const int32_t* tokens,
    uint32_t count,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
);

/**
 * Count UTF-8 bytes required to detokenize token ids.
 *
 * Thread-safety: Safe to call from multiple threads on the same model.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_detokenize_count(
    AstralHandle model,
    const int32_t* tokens,
    uint32_t count,
    uint32_t* out_len
);

/**
 * Count text chunks for a UTF-8 span without writing ranges.
 *
 * Thread-safety: Safe to call from multiple threads.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_chunk_count(
    const AstralChunkerDesc* desc,
    AstralSpanU8 text,
    uint32_t* out_count
);

/**
 * Write UTF-8 byte ranges into a caller-owned range buffer.
 *
 * `out_ranges` must contain at least `max_ranges` entries. `out_count` receives
 * the required range count even when `max_ranges` is too small.
 *
 * Thread-safety: Safe to call from multiple threads.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_chunk_ranges(
    const AstralChunkerDesc* desc,
    AstralSpanU8 text,
    AstralChunkRange* out_ranges,
    uint32_t max_ranges,
    uint32_t* out_count
);

/**
 * Copy one text range to a caller-owned UTF-8 buffer.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_chunk_text_copy(
    AstralSpanU8 text,
    const AstralChunkRange* range,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
);

/**
 * Count token chunks for an already-tokenized sequence.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_token_chunk_count(
    const AstralChunkerDesc* desc,
    uint32_t token_count,
    uint32_t* out_count
);

/**
 * Write token chunk ranges for an already-tokenized sequence.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_token_chunk_ranges(
    const AstralChunkerDesc* desc,
    uint32_t token_count,
    AstralChunkRange* out_ranges,
    uint32_t max_ranges,
    uint32_t* out_count
);

ASTRAL_API AstralErr ASTRAL_CALL astral_memory_create(const AstralMemoryIndexDesc* desc, AstralHandle* out_index);
ASTRAL_API void ASTRAL_CALL astral_memory_destroy(AstralHandle index);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_count(AstralHandle index, uint32_t* out_count);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_stats(AstralHandle index, AstralMemoryStats* out_stats);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_clear(AstralHandle index);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_get_record(AstralHandle index, uint64_t key, AstralMemoryRecord* out_record);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_update_record(AstralHandle index, uint64_t key, const AstralMemoryRecord* record);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_add_batch(
    AstralHandle index,
    const AstralMemoryRecord* records,
    const float* vectors,
    uint32_t count
);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_remove(AstralHandle index, uint64_t key);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_search(
    AstralHandle index,
    const AstralMemorySearchDesc* desc,
    const float* query,
    AstralMemorySearchResult* out_results,
    uint32_t max_results,
    uint32_t* out_count
);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_search_batch(
    AstralHandle index,
    const AstralMemorySearchDesc* desc,
    const float* queries,
    uint32_t query_count,
    AstralMemorySearchResult* out_results,
    uint32_t max_results,
    uint32_t* out_counts
);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_search_begin(
    AstralHandle index,
    const AstralMemorySearchDesc* desc,
    const float* query,
    AstralHandle* out_cursor
);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_search_fetch(
    AstralHandle cursor,
    AstralMemorySearchResult* out_results,
    uint32_t max_results,
    uint32_t* out_count
);
ASTRAL_API void ASTRAL_CALL astral_memory_search_end(AstralHandle cursor);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_save_size(AstralHandle index, uint64_t* out_bytes);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_save(AstralHandle index, AstralMutSpanU8 out_bytes, uint64_t* out_written);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_snapshot_info(AstralSpanU8 bytes,
                                                             AstralMemorySnapshotInfo* out_info);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_snapshot_search(
    AstralSpanU8 bytes, const AstralMemorySearchDesc* desc, const float* query,
    AstralMemorySearchResult* out_results, uint32_t max_results, uint32_t* out_count);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_load(
    const AstralMemoryIndexDesc* desc,
    AstralSpanU8 bytes,
    AstralHandle* out_index
);
ASTRAL_API AstralErr ASTRAL_CALL astral_memory_record_from_chunk(
    const AstralChunkRange* range,
    uint64_t key,
    uint32_t flags,
    AstralMemoryRecord* out_record
);

ASTRAL_API AstralErr ASTRAL_CALL astral_model_path_resolve(
    const AstralModelPathResolveDesc* desc,
    AstralMutSpanU8 out_path,
    uint32_t* out_len);

/**
 * Load a GGUF model.
 * Thread-safety: Safe to call from multiple threads.
 *
 * @param desc Model configuration (must not be NULL)
 * @param out_model Output: model handle (must not be NULL; valid only if return is ASTRAL_OK)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if desc/out_model is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_load(const AstralModelDesc* desc, AstralHandle* out_model);

/**
 * Load a model from a selected source (path / memory / custom IO).
 *
 * Notes:
 * - This is an embedded-friendly surface; not all backends support all sources yet.
 * - If a source is unsupported by the selected backend, returns ASTRAL_E_UNSUPPORTED.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_load2(const AstralModelDesc* desc, AstralHandle* out_model);

/**
 * Release a model.
 * Thread-safety: Not thread-safe; must not be in use by any session.
 *
 * @param model Model handle (from astral_model_load; must not be NULL)
 */
ASTRAL_API void ASTRAL_CALL astral_model_release(AstralHandle model);

// ============================================================================
// Session
// ============================================================================

/**
 * Session configuration.
 */
typedef struct AstralSessionDesc {
    AstralHandle model;        /** Model handle (from astral_model_load) */
    uint32_t max_tokens;       /** Maximum tokens to generate */
    float temperature;         /** Sampling temperature (0.0 = greedy, 1.0 = diverse) */
    uint32_t top_k;            /** Top-K sampling */
    float top_p;               /** Top-P (nucleus) sampling */
    uint8_t stream_enabled;    /** Enable token streaming (1 = yes, 0 = no) */
    uint8_t _padding0[3];
    uint32_t seed;             /** RNG seed for sampling (0 = auto) */
} AstralSessionDesc;

// ============================================================================
// Continuous batching (conversations)
// ============================================================================

/**
 * Model executor configuration (v0.2+).
 *
 * An executor is a model-scoped decode engine that can run multiple independent
 * conversations ("slots") through a single backend session context, enabling
 * continuous batching across slots.
 *
 * Notes:
 * - `size` must be set to sizeof(AstralExecutorDesc).
 * - Must be configured before creating any conversations for a model.
 */
typedef struct AstralExecutorDesc {
    uint32_t size;             // sizeof(AstralExecutorDesc)
    uint32_t max_slots;        // maximum number of concurrent conversations for this model (>=1)
    uint32_t max_batch_tokens; // per-tick batch token cap (<= model n_batch is recommended)
    uint32_t worker_hint;      // 0 = auto, otherwise preferred runtime worker id
} AstralExecutorDesc;

/**
 * Executor tuning (v0.2+).
 *
 * Allows adjusting non-ABI-critical scheduling knobs at runtime.
 *
 * Notes:
 * - `size` must be set to sizeof(AstralExecutorTuning).
 * - If the executor is not yet created, the runtime returns ASTRAL_E_STATE without changing tuning.
 */
typedef struct AstralExecutorTuning {
    uint32_t size;                            // sizeof(AstralExecutorTuning)
    uint32_t max_prompt_tokens_per_slot_tick; // prompt tokens per slot per tick (0 = leave unchanged)
} AstralExecutorTuning;

/**
 * Conversation configuration (v0.2+).
 *
 * A conversation is an independent token stream (prompt + generation) that runs
 * inside a model-scoped executor slot.
 *
 * Notes:
 * - `size` must be set to sizeof(AstralConvDesc).
 * - Sampling defaults are taken from temperature/top_k/top_p; for full control,
 *   use `astral_conv_set_sampler()` after creation.
 */
typedef struct AstralConvDesc {
    uint32_t size;           // sizeof(AstralConvDesc)
    AstralHandle model;      // model handle
    uint32_t max_tokens;     // maximum tokens to generate
    float temperature;       // legacy sampler defaults
    uint32_t top_k;          // legacy sampler defaults
    float top_p;             // legacy sampler defaults
    uint8_t stream_enabled;  // enable UTF-8 streaming
    uint8_t _padding0[3];
    uint32_t seed;           // RNG seed (0 = auto)
} AstralConvDesc;

/**
 * Conversation statistics (v0.2+).
 *
 * Note: these are approximate and are primarily intended for scheduling/observability.
 */
typedef struct AstralConvStats {
    uint32_t slot_id;          // executor slot id
    uint32_t prompt_tokens;    // tokenized prompt length
    uint32_t kv_tokens;        // tokens evaluated in KV for this slot (n_past)
    uint32_t _padding0;
    uint64_t generated_tokens; // tokens generated so far (or total at completion)
    double t_first_token_ms;   // time-to-first-token in ms (0 if not available yet)
    double tok_per_s;          // generated token throughput (0 if not available yet)
} AstralConvStats;

/**
 * Extended sampler controls (v0.2+).
 *
 * Notes:
 * - `size` must be set to sizeof(AstralSamplerDesc).
 * - Any fields not supported by the current build/provider will be clamped/ignored
 *   (and capabilities should be checked via `astral_model_caps()`).
 */
typedef struct AstralSamplerDesc {
    uint32_t size;              // sizeof(AstralSamplerDesc)
    float temperature;          // 0.0 = greedy
    uint32_t top_k;             // 0 = disabled
    float top_p;                // <=0 or >=1 = disabled
    float min_p;                // 0.0 = disabled
    float typical_p;            // 1.0 = disabled
    float repeat_penalty;       // 1.0 = disabled
    int32_t repeat_last_n;      // 0 = disabled, -1 = use ctx size (treated as a clamp)
    uint8_t penalize_nl;        // 1 = penalize newline tokens
    uint8_t _padding0[3];
    float presence_penalty;     // 0.0 = disabled
    float frequency_penalty;    // 0.0 = disabled
    uint32_t mirostat;          // 0 = disabled, 1 = mirostat, 2 = mirostat v2
    float mirostat_tau;         // target surprise
    float mirostat_eta;         // learning rate
} AstralSamplerDesc;

/**
 * Per-token metadata event (side-channel stream).
 *
 * `top_*` arrays contain up to `top_n` entries, sorted by probability (descending)
 * within the sampling distribution actually used for the token (post filters).
 */
typedef struct AstralTokenMeta {
    uint32_t token_id;
    uint32_t top_n; // <= ASTRAL_LOGPROBS_MAX
    float logprob;  // log(p(token_id)) in the used sampling distribution (0 for greedy)
    uint32_t top_token_ids[ASTRAL_LOGPROBS_MAX];
    float top_logprobs[ASTRAL_LOGPROBS_MAX];
} AstralTokenMeta;

/**
 * LoRA adapter description.
 */
typedef struct AstralAdapterDesc {
    uint32_t size;      // sizeof(AstralAdapterDesc)
    AstralSpanU8 path;  // UTF-8 path to adapter file
} AstralAdapterDesc;

typedef struct AstralAdapterInfo {
    uint32_t size;
    AstralHandle model;
    uint32_t path_bytes;
    uint32_t refcount;
} AstralAdapterInfo;

// Compile-time layout validation for configs.
#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
  ASTRAL_STATIC_ASSERT(sizeof(AstralInit) == 64, "AstralInit must be 64 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralModelDesc) == 168, "AstralModelDesc must be 168 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralModelPathResolveDesc) == ASTRAL_MODEL_PATH_RESOLVE_DESC_BYTES_64, "AstralModelPathResolveDesc must be 96 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralSessionDesc) == 32, "AstralSessionDesc must be 32 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralExecutorDesc) == 16, "AstralExecutorDesc must be 16 bytes");
  ASTRAL_STATIC_ASSERT(sizeof(AstralExecutorTuning) == 8, "AstralExecutorTuning must be 8 bytes");
  ASTRAL_STATIC_ASSERT(sizeof(AstralConvDesc) == 40, "AstralConvDesc must be 40 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralConvStats) == 40, "AstralConvStats must be 40 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralSamplerDesc) == 56, "AstralSamplerDesc must be 56 bytes");
  ASTRAL_STATIC_ASSERT(sizeof(AstralTokenMeta) == 140, "AstralTokenMeta must be 140 bytes");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAdapterDesc) == 24, "AstralAdapterDesc must be 24 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAdapterInfo) == 24, "AstralAdapterInfo must be 24 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralImageDesc) == 64, "AstralImageDesc must be 64 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAudioDesc) == 72, "AstralAudioDesc must be 72 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralModelMediaDesc) == 104, "AstralModelMediaDesc must be 104 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralMediaInfo) == 28, "AstralMediaInfo must be 28 bytes on 64-bit");
#else
  ASTRAL_STATIC_ASSERT(sizeof(AstralInit) == 48, "AstralInit must be 48 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralModelDesc) == 116, "AstralModelDesc must be 116 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralModelPathResolveDesc) == ASTRAL_MODEL_PATH_RESOLVE_DESC_BYTES_32, "AstralModelPathResolveDesc must be 56 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralSessionDesc) == 32, "AstralSessionDesc must be 32 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralExecutorDesc) == 16, "AstralExecutorDesc must be 16 bytes");
  ASTRAL_STATIC_ASSERT(sizeof(AstralExecutorTuning) == 8, "AstralExecutorTuning must be 8 bytes");
  ASTRAL_STATIC_ASSERT(sizeof(AstralConvDesc) == 36, "AstralConvDesc must be 36 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralConvStats) == 40, "AstralConvStats must be 40 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralSamplerDesc) == 56, "AstralSamplerDesc must be 56 bytes");
  ASTRAL_STATIC_ASSERT(sizeof(AstralTokenMeta) == 140, "AstralTokenMeta must be 140 bytes");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAdapterDesc) == 12, "AstralAdapterDesc must be 12 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAdapterInfo) == 20, "AstralAdapterInfo must be 20 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralImageDesc) == 52, "AstralImageDesc must be 52 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAudioDesc) == 60, "AstralAudioDesc must be 60 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralModelMediaDesc) == 72, "AstralModelMediaDesc must be 72 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralMediaInfo) == 28, "AstralMediaInfo must be 28 bytes on 32-bit");
#endif


/**
 * Session state (mirrors internal state machine).
 */
typedef uint32_t AstralSessionState;
enum {
    ASTRAL_SESSION_IDLE = 0,
    ASTRAL_SESSION_FEEDING_PROMPT = 1,
    ASTRAL_SESSION_DECODING = 2,
    ASTRAL_SESSION_COMPLETED = 3, /** Completed successfully (EOS or max_tokens) */
    ASTRAL_SESSION_CANCELED = 4,  /** Completed due to cancellation */
    ASTRAL_SESSION_FAILED = 5,    /** Completed due to error */
};

typedef uint32_t AstralRequestKind;
enum {
    ASTRAL_REQUEST_NONE = 0,
    ASTRAL_REQUEST_SESSION = 1,
    ASTRAL_REQUEST_CONVERSATION = 2,
    ASTRAL_REQUEST_AGENT_CHAT = 3,
    ASTRAL_REQUEST_EMBEDDING = 4,
    ASTRAL_REQUEST_MEMORY_SEARCH = 5,
};

typedef uint32_t AstralRequestState;
enum {
    ASTRAL_REQUEST_INVALID = 0,
    ASTRAL_REQUEST_QUEUED = 1,
    ASTRAL_REQUEST_RUNNING = 2,
    ASTRAL_REQUEST_COMPLETED = 3,
    ASTRAL_REQUEST_CANCELED = 4,
    ASTRAL_REQUEST_FAILED = 5,
};

typedef uint32_t AstralRequestFlags;
enum {
    ASTRAL_REQUEST_FLAG_STREAM = 1u << 0,
    ASTRAL_REQUEST_FLAG_TICKET = 1u << 1,
};

typedef struct AstralRequestRef {
    uint32_t size;
    AstralRequestKind kind;
    AstralHandle owner;
    uint64_t ticket;
} AstralRequestRef;

typedef struct AstralRequestStatus {
    uint32_t size;
    AstralRequestKind kind;
    AstralRequestState state;
    AstralRequestFlags flags;
    AstralHandle owner;
    uint64_t ticket;
    AstralErr result;
    uint32_t queue_depth;
} AstralRequestStatus;

typedef uint32_t AstralAgentRole;
enum {
    ASTRAL_AGENT_ROLE_SYSTEM = 1,
    ASTRAL_AGENT_ROLE_USER = 2,
    ASTRAL_AGENT_ROLE_ASSISTANT = 3,
    ASTRAL_AGENT_ROLE_TOOL = 4,
};

typedef uint32_t AstralAgentFlags;
enum {
    ASTRAL_AGENT_FLAG_NONE = 0,
};

enum {
    ASTRAL_AGENT_SLOT_AUTO = 0,
};

typedef uint32_t AstralAgentChatFlags;
enum {
    ASTRAL_AGENT_CHAT_FLAG_NONE = 0,
    ASTRAL_AGENT_CHAT_FLAG_WARMUP = 1u << 0,
};

typedef uint32_t AstralAgentOverflowPolicy;
enum {
    ASTRAL_AGENT_OVERFLOW_REJECT = 0,
    ASTRAL_AGENT_OVERFLOW_TRUNCATE_OLDEST = 1,
};

typedef struct AstralAgentDesc {
    uint32_t size;
    AstralAgentFlags flags;
    AstralHandle model;
    AstralHandle prompt_cache;
    AstralHandle memory_index;
    AstralHandle toolset;
    uint32_t max_tokens;
    float temperature;
    uint32_t top_k;
    float top_p;
    uint8_t stream_enabled;
    uint8_t _padding0[3];
    uint32_t seed;
    AstralToolChoiceMode tool_choice_mode;
    uint32_t max_messages;
    uint32_t max_prompt_bytes;
    AstralAgentOverflowPolicy overflow_policy;
    uint32_t slot_affinity; // ASTRAL_AGENT_SLOT_AUTO or one-based executor slot
    AstralSpanU8 system_prompt;
    AstralSpanU8 summary;
    AstralSpanU8 memory_context;
} AstralAgentDesc;

typedef struct AstralAgentMessage {
    uint32_t size;
    AstralAgentRole role;
    AstralSpanU8 content;
} AstralAgentMessage;

typedef struct AstralAgentChatDesc {
    uint32_t size;
    AstralAgentChatFlags flags;
    AstralSpanU8 user_message;
} AstralAgentChatDesc;

typedef struct AstralAgentMemoryContextDesc {
    uint32_t size;
    uint32_t result_count;
    uint32_t chunk_count;
    uint32_t max_bytes;
    AstralSpanU8 document_text;
    AstralSpanU8 separator;
    const AstralChunkRange* chunks;
    const AstralMemorySearchResult* results;
} AstralAgentMemoryContextDesc;

typedef struct AstralAgentChatResult {
    uint32_t size;
    AstralSessionState state;
    uint32_t prompt_bytes;
    uint32_t history_messages;
    uint32_t prompt_tokens;
    uint32_t prompt_cache_reused_tokens;
    uint32_t prompt_cache_new_tokens;
    uint32_t prompt_cache_hits;
    uint32_t prompt_cache_misses;
    AstralErr last_error;
    double prompt_build_ms;
    uint64_t generated_tokens;
    double t_first_token_ms;
    double tok_per_s;
} AstralAgentChatResult;

#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
  ASTRAL_STATIC_ASSERT(sizeof(AstralRequestRef) == 24, "AstralRequestRef must be 24 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralRequestStatus) == 40, "AstralRequestStatus must be 40 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentDesc) == 136, "AstralAgentDesc must be 136 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentMessage) == 24, "AstralAgentMessage must be 24 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentChatDesc) == 24, "AstralAgentChatDesc must be 24 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentMemoryContextDesc) == 64, "AstralAgentMemoryContextDesc must be 64 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentChatResult) == 72, "AstralAgentChatResult must be 72 bytes on 64-bit");
#else
  ASTRAL_STATIC_ASSERT(sizeof(AstralRequestRef) == 24, "AstralRequestRef must be 24 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralRequestStatus) == 40, "AstralRequestStatus must be 40 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentDesc) == 108, "AstralAgentDesc must be 108 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentMessage) == 16, "AstralAgentMessage must be 16 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentChatDesc) == 16, "AstralAgentChatDesc must be 16 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentMemoryContextDesc) == 40, "AstralAgentMemoryContextDesc must be 40 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralAgentChatResult) == 72, "AstralAgentChatResult must be 72 bytes on 32-bit");
#endif

/**
 * Build request references for existing async owners.
 *
 * `AstralRequestRef` is a small value type for engine queues. It does not own
 * the underlying handle or embedding ticket.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_session(AstralHandle session, AstralRequestRef* out_request);
ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_conversation(AstralHandle conv, AstralRequestRef* out_request);
ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_agent_chat(AstralHandle agent, AstralRequestRef* out_request);
ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_embedding(
    AstralHandle emb,
    uint64_t ticket,
    AstralRequestRef* out_request
);
ASTRAL_API AstralErr ASTRAL_CALL astral_request_from_memory_search(
    AstralHandle cursor,
    AstralRequestRef* out_request
);

/**
 * Query, cancel, or wait for a request through one API shape.
 *
 * Embedding requests are ticketed. `astral_request_wait()` reports the current
 * embedding ticket state; collection still happens through `astral_embed_collect()`
 * because the caller owns the vector buffer.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_request_state(
    const AstralRequestRef* request,
    AstralRequestStatus* out_status
);
ASTRAL_API AstralErr ASTRAL_CALL astral_request_cancel(const AstralRequestRef* request);
ASTRAL_API AstralErr ASTRAL_CALL astral_request_wait(
    const AstralRequestRef* request,
    uint32_t timeout_ms,
    AstralRequestStatus* out_status
);

/**
 * Request cancellation for an in-flight session decode.
 * Thread-safety: Safe to call from any thread.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_cancel(AstralHandle session);

/**
 * Query current session state.
 * Thread-safety: Safe to call from any thread.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_state(AstralHandle session, AstralSessionState* out_state);

/**
 * Wait for session completion.
 *
 * @param timeout_ms 0 for non-blocking poll; otherwise timeout in milliseconds.
 * @return ASTRAL_OK if completed successfully,
 *         ASTRAL_E_CANCELED if canceled,
 *         or an error code if the decode failed.
 *         Returns ASTRAL_E_TIMEOUT if not completed by deadline.
 *
 * Thread-safety: Safe to call from any thread.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_wait(AstralHandle session, uint32_t timeout_ms);

/**
 * Reset a session for reuse.
 *
 * Clears prompt buffer, streaming ring, cancellation flag, and backend KV/cache state.
 *
 * Preconditions:
 * - Session must not be decoding (call cancel + wait first).
 * - Not thread-safe; must not be called concurrently with astral_stream_read().
 *
 * @param desc Optional: new session parameters. If provided, desc->model must match the session's model.
 *             NOTE: reset/reuse cannot grow internal scratch buffers; increasing top_k beyond the
 *             session's original capacity returns ASTRAL_E_INVALID.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_reset(AstralHandle session, const AstralSessionDesc* desc);

/**
 * Configure sampler controls for a session.
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 *
 * Thread-safety: Not thread-safe; single-threaded access per session.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_sampler(AstralHandle session, const AstralSamplerDesc* desc);

/**
 * Configure the "penalty prompt" token span used by sampling penalties.
 *
 * The provided token ids are counted once and added to the penalty counts for every generated token.
 * This is useful for penalizing repeats of a fixed system/prefix prompt without requiring callers
 * to re-tokenize or re-count per token.
 *
 * Notes:
 * - Passing `count == 0` clears the penalty prompt.
 * - Tokens outside [0, vocab_size) are ignored.
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_penalty_prompt_set_tokens(
    AstralHandle session,
    const int32_t* tokens,
    uint32_t count
);

/**
 * Clear all stop sequences for a session.
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 *
 * Thread-safety: Not thread-safe; single-threaded access per session.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_clear(AstralHandle session);

/**
 * Add a stop sequence (UTF-8). The text is tokenized once and matched by tokens.
 *
 * Stop sequences are not emitted to the UTF-8 stream; streaming may delay output by up to
 * the maximum stop sequence length to preserve this guarantee.
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 *
 * Thread-safety: Not thread-safe; single-threaded access per session.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_add_utf8(AstralHandle session, AstralSpanU8 utf8);

/**
 * Bulk set stop sequences (UTF-8).
 *
 * Equivalent to: `stop_clear()` then `stop_add_utf8()` for each entry, but avoids per-call overhead.
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_set_utf8(
    AstralHandle session,
    const AstralSpanU8* seqs,
    uint32_t count
);

/**
 * Configure per-token logprobs side-channel.
 *
 * When enabled, the decode thread publishes `AstralTokenMeta` events to the meta stream.
 *
 * Notes:
 * - `n_probs == 0` disables meta events.
 * - `n_probs` is clamped to `ASTRAL_LOGPROBS_MAX`.
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_logprobs(AstralHandle session, uint32_t n_probs);

/**
 * Read meta events from the side-channel stream.
 *
 * Return values:
 * - > 0: number of events written to out_events
 * - 0: end-of-stream (session is terminal and no buffered events remain)
 * - < 0: error code (e.g. ASTRAL_E_INVALID, ASTRAL_E_TIMEOUT)
 */
ASTRAL_API int32_t ASTRAL_CALL astral_stream_read_meta(
    AstralHandle session,
    AstralTokenMeta* out_events,
    uint32_t capacity,
    uint32_t timeout_ms
);

/**
 * Save/load session KV/state as bytes.
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_size(AstralHandle session, uint64_t* out_bytes);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_save(
    AstralHandle session,
    AstralMutSpanU8 out_buf,
    uint64_t* out_written
);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_load(AstralHandle session, AstralSpanU8 state_bytes);

/**
 * Load/unload a LoRA adapter.
 *
 * Adapter handles are model-scoped and can be attached to sessions via
 * `astral_session_adapters_add()`.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_adapter_load(
    AstralHandle model,
    const AstralAdapterDesc* desc,
    AstralHandle* out_adapter
);
ASTRAL_API AstralErr ASTRAL_CALL astral_model_adapter_info(AstralHandle adapter, AstralAdapterInfo* out_info);
ASTRAL_API AstralErr ASTRAL_CALL astral_model_adapter_path_copy(
    AstralHandle adapter,
    AstralMutSpanU8 out_path,
    uint32_t* out_len
);
ASTRAL_API void ASTRAL_CALL astral_model_adapter_release(AstralHandle adapter);

/**
 * Attach adapters to a session.
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_clear(AstralHandle session);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_add(AstralHandle session, AstralHandle adapter, float scale);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_count(AstralHandle session, uint32_t* out_count);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_get(
    AstralHandle session,
    uint32_t index,
    AstralHandle* out_adapter,
    float* out_scale
);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_set_scale(AstralHandle session, uint32_t index, float scale);

/**
 * Create a bounded token prompt cache.
 *
 * Prompt cache entries are setup-time objects used to reuse tokenized prompt
 * sections. Lookups do not allocate; callers provide the output token buffer
 * or request a cache-owned token view.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_create(const AstralPromptCacheDesc* desc, AstralHandle* out_cache);
ASTRAL_API void ASTRAL_CALL astral_prompt_cache_destroy(AstralHandle cache);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_clear(AstralHandle cache);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_stats(AstralHandle cache, AstralPromptCacheStats* out_stats);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_save_size(AstralHandle cache, uint32_t* out_bytes);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_save(AstralHandle cache, AstralMutSpanU8 out_bytes, uint32_t* out_len);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_load(
    const AstralPromptCacheDesc* desc,
    AstralSpanU8 bytes,
    AstralHandle* out_cache
);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_key_from_bytes(
    AstralHandle model,
    AstralPromptSectionKind section_kind,
    uint32_t generation,
    AstralSpanU8 bytes,
    AstralPromptCacheKey* out_key
);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_put_tokens(
    AstralHandle cache,
    const AstralPromptCacheKey* key,
    const int32_t* tokens,
    uint32_t token_count
);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_get_tokens(
    AstralHandle cache,
    const AstralPromptCacheKey* key,
    int32_t* out_tokens,
    uint32_t max_tokens,
    uint32_t* out_token_count
);
ASTRAL_API AstralErr ASTRAL_CALL astral_prompt_cache_get_token_view(
    AstralHandle cache,
    const AstralPromptCacheKey* key,
    const int32_t** out_tokens,
    uint32_t* out_token_count
);

/**
 * Configure grammar-constrained decoding (GBNF).
 *
 * Notes:
 * - `root` may be empty to use the default start symbol ("root").
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_grammar_gbnf(AstralHandle session, AstralSpanU8 gbnf, AstralSpanU8 root);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_grammar_json_schema(AstralHandle session, AstralSpanU8 json_schema);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_clear_grammar(AstralHandle session);

ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_create(const AstralToolsetDesc* desc, AstralHandle* out_toolset);
ASTRAL_API void ASTRAL_CALL astral_toolset_destroy(AstralHandle toolset);
ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_count(AstralHandle toolset, uint32_t* out_count);
ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_get(AstralHandle toolset, uint32_t index, AstralToolInfo* out_info);
ASTRAL_API AstralErr ASTRAL_CALL astral_toolset_parse_call(
    AstralHandle toolset,
    AstralSpanU8 generated_text,
    AstralToolCallResult* out_result
);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_toolset(
    AstralHandle session,
    AstralHandle toolset,
    AstralToolChoiceMode choice_mode
);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_clear_toolset(AstralHandle session);

/**
 * Set the session slot/sequence id (for providers that support parallel slots).
 *
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_slot(AstralHandle session, uint32_t slot_id);

/**
 * Create an inference session.
 * Thread-safety: Safe to call from multiple threads.
 *
 * @param desc Session configuration (must not be NULL)
 * @param out_session Output: session handle (must not be NULL; valid only if return is ASTRAL_OK)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if desc/out_session is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_create(const AstralSessionDesc* desc, AstralHandle* out_session);

/**
 * Destroy a session.
 * Thread-safety: Not thread-safe; must not be in use.
 *
 * @param session Session handle (from astral_session_create; must not be NULL)
 */
ASTRAL_API void ASTRAL_CALL astral_session_destroy(AstralHandle session);

/**
 * Feed a prompt chunk.
 * Call multiple times for long prompts, setting finalize=1 on the last chunk.
 * Thread-safety: Not thread-safe; single-threaded access per session.
 *
 * @param session Session handle (must not be NULL)
 * @param prompt_chunk UTF-8 prompt data (data may be NULL only if len is 0)
 * @param finalize 1 if this is the last chunk, 0 otherwise
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if session is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed(AstralHandle session, AstralSpanU8 prompt_chunk, uint8_t finalize);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed_tokens(
    AstralHandle session,
    const int32_t* tokens,
    uint32_t token_count,
    uint8_t finalize
);
ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_system_prompt(AstralHandle session, AstralSpanU8 system_prompt);

/**
 * Feed an image chunk into a session prompt.
 *
 * Notes:
 * - Media support must be initialized via astral_model_media_init().
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed_image(
    AstralHandle session,
    const AstralImageDesc* image,
    uint8_t finalize
);

/**
 * Feed an audio chunk into a session prompt.
 *
 * Notes:
 * - Media support must be initialized via astral_model_media_init().
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed_audio(
    AstralHandle session,
    const AstralAudioDesc* audio,
    uint8_t finalize
);

/**
 * Start decoding (non-blocking).
 * Submits work to thread pool; returns immediately.
 * Thread-safety: Not thread-safe; single-threaded access per session.
 *
 * @param session Session handle (must not be NULL)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if session is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_decode(AstralHandle session);

/**
 * Read tokens from stream.
 * Blocks up to `timeout_ms` if no data available.
 *
 * Thread-safety:
 * - Safe for a single consumer thread calling `astral_stream_read()` concurrently with the decode worker.
 * - Not safe to call `astral_stream_read()` from multiple threads concurrently.
 *
 * @param session Session handle (must not be NULL)
 * @param out_buf Output buffer for UTF-8 token data (data must not be NULL)
 * @param timeout_ms Timeout in milliseconds (0 = non-blocking)
 * @return Bytes written to out_buf (>= 0), or error code (< 0)
 *         0 indicates end-of-stream (session is terminal and no buffered data remains).
 *           Call `astral_session_wait()` to determine whether the session finished successfully,
 *           was canceled, or failed.
 *         ASTRAL_E_INVALID if session or out_buf.data is NULL
 *         ASTRAL_E_TIMEOUT if no data available within timeout
 */
ASTRAL_API int32_t ASTRAL_CALL astral_stream_read(AstralHandle session, AstralMutSpanU8 out_buf, uint32_t timeout_ms);

// ============================================================================
// Conversation (continuous batching)
// ============================================================================

/**
 * Configure the model-scoped executor used for continuous batching (v0.2+).
 *
 * Must be called before creating any conversations for this model. If the executor is
 * already created, returns ASTRAL_E_STATE.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_executor_configure(AstralHandle model, const AstralExecutorDesc* desc);

/**
 * Tune an already-created model executor (v0.2+).
 *
 * Thread-safety: Safe to call concurrently with conversations.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_executor_tune(AstralHandle model, const AstralExecutorTuning* tuning);

/**
 * Create a conversation (slot) for a model executor (v0.2+).
 *
 * Thread-safety: Not thread-safe; single-threaded access per conversation.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_create(const AstralConvDesc* desc, AstralHandle* out_conv);

/**
 * Destroy a conversation.
 * Thread-safety: Not thread-safe; must not be in use.
 */
ASTRAL_API void ASTRAL_CALL astral_conv_destroy(AstralHandle conv);

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed(AstralHandle conv, AstralSpanU8 prompt_chunk, uint8_t finalize);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed_tokens(
    AstralHandle conv,
    const int32_t* tokens,
    uint32_t token_count,
    uint8_t finalize
);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_system_prompt(AstralHandle conv, AstralSpanU8 system_prompt);

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed_image(
    AstralHandle conv,
    const AstralImageDesc* image,
    uint8_t finalize
);

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed_audio(
    AstralHandle conv,
    const AstralAudioDesc* audio,
    uint8_t finalize
);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_decode(AstralHandle conv);

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_cancel(AstralHandle conv);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_state(AstralHandle conv, AstralSessionState* out_state);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_wait(AstralHandle conv, uint32_t timeout_ms);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_reset(AstralHandle conv, const AstralConvDesc* desc);

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_sampler(AstralHandle conv, const AstralSamplerDesc* desc);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_penalty_prompt_set_tokens(
    AstralHandle conv, const int32_t* tokens, uint32_t count);

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_clear(AstralHandle conv);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_add_utf8(AstralHandle conv, AstralSpanU8 utf8);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_set_utf8(
    AstralHandle conv, const AstralSpanU8* seqs, uint32_t count);

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_logprobs(AstralHandle conv, uint32_t n_probs);

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_set_gbnf(
    AstralHandle conv, AstralSpanU8 gbnf, AstralSpanU8 root);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_set_json_schema(
    AstralHandle conv, AstralSpanU8 json_schema);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_clear(AstralHandle conv);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_toolset(
    AstralHandle conv,
    AstralHandle toolset,
    AstralToolChoiceMode choice_mode
);
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_clear_toolset(AstralHandle conv);

ASTRAL_API int32_t ASTRAL_CALL astral_conv_stream_read(AstralHandle conv, AstralMutSpanU8 out_buf, uint32_t timeout_ms);
ASTRAL_API int32_t ASTRAL_CALL astral_conv_stream_read_meta(
    AstralHandle conv, AstralTokenMeta* out_events, uint32_t capacity, uint32_t timeout_ms);

/**
 * Conversation statistics (v0.2+).
 *
 * Thread-safety: Safe to call concurrently with decoding/streaming.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stats(AstralHandle conv, AstralConvStats* out_stats);

// ============================================================================
// Agents
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_agent_create(const AstralAgentDesc* desc, AstralHandle* out_agent);
ASTRAL_API void ASTRAL_CALL astral_agent_destroy(AstralHandle agent);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_assigned_slot(AstralHandle agent, uint32_t* out_slot);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_release_slot(AstralHandle agent);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_set_system_prompt(AstralHandle agent, AstralSpanU8 system_prompt);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_system_prompt_size(AstralHandle agent, uint32_t* out_bytes);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_system_prompt(
    AstralHandle agent,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_set_summary(AstralHandle agent, AstralSpanU8 summary);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_summary_size(AstralHandle agent, uint32_t* out_bytes);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_summary(
    AstralHandle agent,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_set_memory_context(AstralHandle agent, AstralSpanU8 memory_context);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_set_memory_context_from_results(
    AstralHandle agent,
    const AstralAgentMemoryContextDesc* desc
);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_memory_context_size(AstralHandle agent, uint32_t* out_bytes);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_get_memory_context(
    AstralHandle agent,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_parse_tool_call(
    AstralHandle agent,
    AstralSpanU8 generated_text,
    AstralToolCallResult* out_result
);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_message_add(AstralHandle agent, const AstralAgentMessage* message);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_clear(AstralHandle agent);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_count(AstralHandle agent, uint32_t* out_count);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_save_size(AstralHandle agent, uint32_t* out_bytes);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_save(AstralHandle agent, AstralMutSpanU8 out_bytes, uint32_t* out_len);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_history_load(AstralHandle agent, AstralSpanU8 bytes);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_chat_enqueue(AstralHandle agent, const AstralAgentChatDesc* desc);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_chat_cancel(AstralHandle agent);
ASTRAL_API int32_t ASTRAL_CALL astral_agent_chat_stream_read(
    AstralHandle agent,
    AstralMutSpanU8 out_buf,
    uint32_t timeout_ms
);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_chat_tool_call_result(
    AstralHandle agent,
    AstralToolCallResult* out_result
);
ASTRAL_API AstralErr ASTRAL_CALL astral_agent_chat_result(AstralHandle agent, AstralAgentChatResult* out_result);

// ============================================================================
// Embeddings
// ============================================================================

/**
 * Create an embeddings handle.
 *
 * Thread-safety: Safe to call from multiple threads.
 *
 * @param model Model handle (must be loaded with embeddings_only=1; must not be NULL)
 * @param out_embedder Output: embedder handle (must not be NULL; valid only if return is ASTRAL_OK)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if model/out_embedder is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_embed_create(AstralHandle model, AstralHandle* out_embedder);

/**
 * Destroy an embeddings handle.
 * Thread-safety: Not thread-safe; must not be in use.
 *
 * @param emb Embedder handle (from astral_embed_create; must not be NULL)
 */
ASTRAL_API void ASTRAL_CALL astral_embed_destroy(AstralHandle emb);

/**
 * Enqueue text for embedding.
 * Non-blocking; returns a ticket to collect results later.
 * Thread-safety: Safe to call from multiple threads.
 *
 * @param emb Embedder handle (must not be NULL)
 * @param text UTF-8 input text (data may be NULL only if len is 0)
 * @param out_ticket Output: ticket ID for astral_embed_collect (must not be NULL)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if emb/out_ticket is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue(AstralHandle emb, AstralSpanU8 text, uint64_t* out_ticket);

/**
 * Enqueue image for embedding.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_image(
    AstralHandle emb,
    const AstralImageDesc* image,
    uint64_t* out_ticket
);

/**
 * Enqueue audio for embedding.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_audio(
    AstralHandle emb,
    const AstralAudioDesc* audio,
    uint64_t* out_ticket
);

/**
 * Enqueue multimodal input (text + optional image/audio) for embedding.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_multimodal(
    AstralHandle emb,
    AstralSpanU8 text,
    const AstralImageDesc* image,
    const AstralAudioDesc* audio,
    uint64_t* out_ticket
);

/**
 * Cancel a queued embedding ticket.
 * Releases the ticket's queue slot if the work has not started collecting yet.
 * Thread-safety: Safe to call from multiple threads.
 *
 * @param emb Embedder handle (must not be NULL)
 * @param ticket Ticket ID returned by an enqueue call
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID for stale/unknown tickets; ASTRAL_E_BUSY if already collecting
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_embed_cancel(AstralHandle emb, uint64_t ticket);

/**
 * Collect embedding vector for a ticket.
 * Blocks until embedding is ready.
 * Thread-safety: Safe to call from multiple threads.
 *
 * @param emb Embedder handle (must not be NULL)
 * @param ticket Ticket ID (from astral_embed_enqueue)
 * @param out_vector Output buffer for embedding vector (float32 array; data must not be NULL)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if emb/out_vector.data is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_embed_collect(AstralHandle emb, uint64_t ticket, AstralMutSpanU8 out_vector);

// ============================================================================
// Statistics
// ============================================================================

/**
 * Session statistics.
 */
typedef struct AstralStats {
    double t_init_ms;           /** Init time (model load) in milliseconds */
    double t_first_token_ms;    /** Time to first token in milliseconds */
    double tok_per_s;           /** Tokens per second (throughput) */
    uint64_t bytes_committed;   /** Committed memory in bytes */
    uint64_t bytes_reserved;    /** Reserved virtual memory in bytes */
} AstralStats;

/**
 * Get session statistics.
 * Thread-safety: Safe to call concurrently with other session operations.
 *
 * NOTE: Statistics are finalized when the session reaches `ASTRAL_SESSION_COMPLETED`.
 * If decoding is still in progress, this returns `ASTRAL_E_STATE`; call `astral_session_wait()` first.
 * Statistics are also available for `ASTRAL_SESSION_CANCELED` and `ASTRAL_SESSION_FAILED`.
 *
 * @param session Session handle (must not be NULL)
 * @param out_stats Output: statistics (must not be NULL)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if session/out_stats is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_stats(AstralHandle session, AstralStats* out_stats);

#ifdef __cplusplus
}
#endif
