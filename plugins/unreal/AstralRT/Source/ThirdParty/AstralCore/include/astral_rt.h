/**
 * astral_rt.h - Astral Runtime C ABI
 *
 * Stable C interface for high-performance LLM inference.
 * All functions return error codes; no exceptions.
 * All strings are UTF-8 spans (pointer + length).
 *
 * Version: 0.1.0
 * License: [TBD]
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
#else
  ASTRAL_STATIC_ASSERT(sizeof(AstralSpanU8) == 8, "AstralSpanU8 must be 8 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralMutSpanU8) == 8, "AstralMutSpanU8 must be 8 bytes on 32-bit");
#endif

/**
 * Opaque handle (model, session, embedder).
 *
 * v0.1 ABI hardening:
 * - Handles are 64-bit values encoding (type, slot index, generation).
 * - 0 is always invalid.
 */
typedef uint64_t AstralHandle;

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
    ASTRAL_E_CANCELED = -7, /** Operation canceled */
    ASTRAL_E_UNSUPPORTED = -8, /** Operation unsupported */
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
 * It does NOT guarantee the resource hasn't been freed (use-after-free detection is best-effort).
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
 * @param path Path to a shared library (UTF-8; not NUL-terminated)
 * @return ASTRAL_OK on success; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_backend_load_plugin(AstralSpanU8 path);

// ============================================================================
// Model
// ============================================================================

/**
 * Model configuration.
 */
typedef struct AstralModelDesc {
    AstralSpanU8 model_path;   /** Model identifier/path (UTF-8). May be empty for backends that don't load from file. */
    AstralSpanU8 backend_name; /** Optional backend override (e.g., "cpu", "mock"); empty = auto */
    uint32_t gpu_layers;       /** Number of layers to offload to GPU (0 = CPU only) */
    uint32_t n_ctx;            /** Context size (tokens) */
    uint32_t n_batch;          /** Batch size for prompt processing */
    uint32_t n_threads;        /** Threads for backend (0 = auto) */
    uint8_t embeddings_only;   /** Embeddings-only mode (1 = yes, 0 = no) */
#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
    uint8_t _padding0[7];
#else
    uint8_t _padding0[3];
#endif
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
    ASTRAL_CAP_LORA        = 1ull << 18,  // LoRA/adapters supported (not yet implemented in v0.1)
    ASTRAL_CAP_GRAMMAR     = 1ull << 19,  // Grammar-constrained decoding supported (not yet implemented in v0.1)
    ASTRAL_CAP_LOGPROBS    = 1ull << 20,  // Per-token metadata/logprobs supported (not yet implemented in v0.1)
    ASTRAL_CAP_KV_STATE    = 1ull << 21,  // Session KV save/load supported (not yet implemented in v0.1)
};

typedef struct AstralModelLimits {
    uint32_t vocab_size;
    uint32_t ctx_size;
    uint32_t max_batch;  // 0 if unknown/unreported
    uint32_t max_slots;  // 0 if not applicable
} AstralModelLimits;

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
 * Query model limits (best-effort; fields may be 0 if unknown).
 *
 * Thread-safety: Safe to call from multiple threads.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_limits(AstralHandle model, AstralModelLimits* out_limits);

/**
 * Get the model embedding dimension (number of floats per embedding vector).
 *
 * Thread-safety: Safe to call from multiple threads.
 *
 * @param model Model handle
 * @param out_dim Output: embedding dimension (must not be NULL)
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
 * Load a GGUF model.
 * Thread-safety: Safe to call from multiple threads.
 *
 * @param desc Model configuration (must not be NULL)
 * @param out_model Output: model handle (must not be NULL; valid only if return is ASTRAL_OK)
 * @return ASTRAL_OK on success; ASTRAL_E_INVALID if desc/out_model is NULL; error code on failure
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_model_load(const AstralModelDesc* desc, AstralHandle* out_model);

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

/**
 * Extended sampler controls (v0.2+).
 *
 * Notes:
 * - `size` must be set to sizeof(AstralSamplerDesc).
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
    uint32_t mirostat;          // 0 = disabled (not yet implemented in v0.1)
    float mirostat_tau;         // (not yet implemented in v0.1)
    float mirostat_eta;         // (not yet implemented in v0.1)
} AstralSamplerDesc;

// Compile-time layout validation for configs.
#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
  ASTRAL_STATIC_ASSERT(sizeof(AstralInit) == 64, "AstralInit must be 64 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralModelDesc) == 56, "AstralModelDesc must be 56 bytes on 64-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralSessionDesc) == 32, "AstralSessionDesc must be 32 bytes on 64-bit");
#else
  ASTRAL_STATIC_ASSERT(sizeof(AstralInit) == 48, "AstralInit must be 48 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralModelDesc) == 36, "AstralModelDesc must be 36 bytes on 32-bit");
  ASTRAL_STATIC_ASSERT(sizeof(AstralSessionDesc) == 32, "AstralSessionDesc must be 32 bytes on 32-bit");
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
 * Preconditions:
 * - Session must not be decoding (cancel + wait first).
 *
 * Thread-safety: Not thread-safe; single-threaded access per session.
 */
ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_add_utf8(AstralHandle session, AstralSpanU8 utf8);

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
