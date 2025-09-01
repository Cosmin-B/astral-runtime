/**
 * astral_backend.h - Backend provider contract (C ABI)
 *
 * This header defines the provider contract used internally by Astral and by
 * optional out-of-tree providers (plugins).
 *
 * Design goals:
 * - Provider-agnostic: core runtime never depends on provider-specific headers.
 * - Low overhead: dispatch is a single indirect call through an ops table.
 * - C-compatible: providers can be implemented in C or C++.
 */

#pragma once

#include "astral_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Read-only view of provider logits for sampling.
///
/// Contract:
/// - `logits` points to `vocab_size` float32 values for the "current" token position.
/// - The memory remains valid until the next provider call that mutates session state
///   (e.g., `session_accept()` / `session_feed()` / `session_destroy()`).
/// - Callers must not write to `logits`.
typedef struct AstralBackendLogitsView {
    const float* logits;
    uint32_t vocab_size;
} AstralBackendLogitsView;

/// Provider ops table.
///
/// Thread-safety notes:
/// - model_* and tokenize/detokenize must be thread-safe for a shared model_ctx.
/// - session_* are NOT thread-safe for the same session_ctx (single-session thread rule).
typedef struct AstralBackendOps {
    // Model lifetime
    void* (ASTRAL_CALL * model_load)(const AstralModelDesc* desc, AstralErr* out_err);
    void (ASTRAL_CALL * model_unload)(void* model_ctx);

    // Text processing
    AstralErr (ASTRAL_CALL * tokenize)(void* model_ctx, AstralSpanU8 text,
                                       int32_t* out_tokens, uint32_t max_tokens,
                                       uint8_t add_special, uint8_t parse_special,
                                       uint32_t* out_count);

    AstralErr (ASTRAL_CALL * detokenize)(void* model_ctx, const int32_t* tokens, uint32_t count,
                                         AstralMutSpanU8 out_text, uint32_t* out_len);

    // Metadata
    AstralErr (ASTRAL_CALL * model_info)(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size);
    AstralErr (ASTRAL_CALL * model_special_tokens)(void* model_ctx, int32_t* out_bos, int32_t* out_eos);
    AstralErr (ASTRAL_CALL * model_embedding_dim)(void* model_ctx, uint32_t* out_dim);

    // Session lifetime + decode primitives
    void* (ASTRAL_CALL * session_create)(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err);
    void (ASTRAL_CALL * session_destroy)(void* session_ctx);

    // Optional: reset backend session state (KV/cache) for reuse.
    // If null, Astral falls back to destroy + create on reset.
    AstralErr (ASTRAL_CALL * session_reset)(void* session_ctx);

    AstralErr (ASTRAL_CALL * session_feed)(void* session_ctx, const int32_t* tokens, uint32_t count);

    // Sampling support (zero-copy logits view + accept/advance).
    AstralErr (ASTRAL_CALL * session_logits)(void* session_ctx, AstralBackendLogitsView* out_view);
    AstralErr (ASTRAL_CALL * session_accept)(void* session_ctx, int32_t token);

    // Embeddings (optional)
    // Providers that do not support embeddings should set these to NULL and return ASTRAL_E_UNSUPPORTED via
    // the higher-level C ABI.
    void* (ASTRAL_CALL * embedder_create)(void* model_ctx, AstralErr* out_err);
    void (ASTRAL_CALL * embedder_destroy)(void* embedder_ctx);
    AstralErr (ASTRAL_CALL * embedder_reset)(void* embedder_ctx);
    AstralErr (ASTRAL_CALL * embedder_embed)(void* embedder_ctx,
                                             const int32_t* tokens,
                                             uint32_t count,
                                             float* out_vec,
                                             uint32_t vec_dim);
} AstralBackendOps;

/// Static backend provider descriptor.
typedef struct AstralBackendProvider {
    const char* name;                 // "cpu", "mock", ...
    const AstralBackendOps* ops;      // Required
    uint8_t supports_gpu;             // Selection hint
    uint32_t min_gpu_layers;          // Selection hint
} AstralBackendProvider;

#ifdef __cplusplus
} // extern "C"
#endif
