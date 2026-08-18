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

/// Batch token descriptor for provider-side multi-slot evaluation.
///
/// Contract:
/// - `slot_id` selects the logical KV/sequence slot (0..max_slots-1).
/// - `pos` is the absolute token position within that slot (0-based).
/// - If `want_logits != 0`, the provider must make logits available for this token
///   as part of the contiguous output set (see `session_batch_logits()`).
typedef struct AstralBackendBatchToken {
    int32_t token;
    uint32_t slot_id;
    uint32_t pos;
    uint8_t want_logits;
    uint8_t _padding0[3];
} AstralBackendBatchToken;

/// Provider ops table.
///
/// Required operations are `model_load`, `model_unload`, `tokenize`, `detokenize`,
/// `model_info`, `session_create`, `session_destroy`, `session_feed`,
/// `session_logits`, and `session_accept`. Plugin registration fails with
/// `ASTRAL_E_INVALID` if any required pointer is NULL. Every other operation is
/// optional. Set an unsupported optional operation to NULL. Astral reports
/// `ASTRAL_E_UNSUPPORTED` from the corresponding public API unless a comment below
/// specifies a fallback.
///
/// Lifetime and failure contract:
/// - `model_load` and each `*_create` function return a provider-owned context. On
///   success they return a non-NULL pointer and write `ASTRAL_OK` to `out_err`. On
///   failure they return NULL and write the failure code. Their matching unload or
///   destroy function must accept every successfully returned context exactly once.
/// - Model contexts remain live until `model_unload`. Session, embedder, and adapter
///   contexts must not outlive the model context from which they were created.
/// - Input spans and descriptor storage are borrowed for the duration of the call
///   unless a specific operation says otherwise. Providers must copy data they retain.
/// - On an `AstralErr` failure, output parameters contain no usable result unless the
///   operation's comment explicitly defines a sizing or partial-result contract.
/// - Provider functions must not throw exceptions across this C ABI.
///
/// Threading and reentrancy contract:
/// - `model_*`, `tokenize`, and `detokenize` must support concurrent calls that share
///   one `model_ctx`.
/// - `session_*` operations are not required to support concurrent calls on the same
///   `session_ctx`; Astral serializes provider work for a session. Providers must not
///   call back into Astral while servicing an operation.
/// - Operations run synchronously on the calling Astral thread and may block for
///   provider work. They must not retain caller-owned output buffers after returning.
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
    // Optional: initialize media (vision/audio) support for this model.
    AstralErr (ASTRAL_CALL * model_media_init)(void* model_ctx, const AstralModelMediaDesc* desc);
    // Optional: query media (vision/audio) info.
    AstralErr (ASTRAL_CALL * model_media_info)(void* model_ctx, AstralMediaInfo* out_info);

    // Session lifetime + decode primitives
    void* (ASTRAL_CALL * session_create)(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err);
    // Optional: create a session with an explicit max slot count (multi-seq context).
    // Providers that don't support slots should ignore max_slots or return ASTRAL_E_UNSUPPORTED.
    void* (ASTRAL_CALL * session_create_ex)(void* model_ctx, const AstralSessionDesc* desc, uint32_t max_slots, AstralErr* out_err);
    void (ASTRAL_CALL * session_destroy)(void* session_ctx);

    // Optional: reset backend session state (KV/cache) for reuse.
    // If null, Astral falls back to destroy + create on reset.
    AstralErr (ASTRAL_CALL * session_reset)(void* session_ctx);

    AstralErr (ASTRAL_CALL * session_feed)(void* session_ctx, const int32_t* tokens, uint32_t count);
    // Optional: feed image/audio chunks.
    AstralErr (ASTRAL_CALL * session_feed_image)(void* session_ctx, const AstralImageDesc* image, uint8_t finalize);
    AstralErr (ASTRAL_CALL * session_feed_audio)(void* session_ctx, const AstralAudioDesc* audio, uint8_t finalize);

    // Sampling support (zero-copy logits view + accept/advance).
    AstralErr (ASTRAL_CALL * session_logits)(void* session_ctx, AstralBackendLogitsView* out_view);
    AstralErr (ASTRAL_CALL * session_accept)(void* session_ctx, int32_t token);

    // --------------------------------------------------------------------
    // Continuous batching (optional, provider-specific support)
    // --------------------------------------------------------------------

    // Evaluate a mixed batch of tokens belonging to multiple slots.
    // Returns output logits for every token where `want_logits != 0`, stored contiguously
    // in token order (matching the order of `want_logits` tokens in the input batch).
    AstralErr (ASTRAL_CALL * session_batch_eval)(void* session_ctx,
                                                 const AstralBackendBatchToken* tokens,
                                                 uint32_t token_count,
                                                 uint32_t* out_output_count);

    // Get logits for the `output_index`th output produced by the last `session_batch_eval()`.
    AstralErr (ASTRAL_CALL * session_batch_logits)(void* session_ctx, uint32_t output_index, AstralBackendLogitsView* out_view);

    // Optional: reset/clear KV state for a single slot (sequence) for reuse.
    AstralErr (ASTRAL_CALL * session_slot_reset)(void* session_ctx, uint32_t slot_id);

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
    // Optional: multimodal embeddings.
    AstralErr (ASTRAL_CALL * embedder_embed_image)(void* embedder_ctx,
                                                   const AstralImageDesc* image,
                                                   float* out_vec,
                                                   uint32_t vec_dim);
    AstralErr (ASTRAL_CALL * embedder_embed_audio)(void* embedder_ctx,
                                                   const AstralAudioDesc* audio,
                                                   float* out_vec,
                                                   uint32_t vec_dim);
    AstralErr (ASTRAL_CALL * embedder_embed_multimodal)(void* embedder_ctx,
                                                        AstralSpanU8 text,
                                                        const AstralImageDesc* image,
                                                        const AstralAudioDesc* audio,
                                                        float* out_vec,
                                                        uint32_t vec_dim);

    // --------------------------------------------------------------------
    // Generation controls (optional, provider-specific support)
    // --------------------------------------------------------------------

    // Grammar (GBNF) support.
    // The provider owns the grammar object; it is associated with a single session.
    // Providers that don't support grammar should set these to NULL.
    AstralErr (ASTRAL_CALL * session_grammar_set_gbnf)(void* session_ctx, AstralSpanU8 gbnf, AstralSpanU8 root);
    // Optional: JSON schema grammar. Providers may compile this to an internal grammar representation.
    // Providers that don't support JSON schema should set this to NULL.
    AstralErr (ASTRAL_CALL * session_grammar_set_json_schema)(void* session_ctx, AstralSpanU8 json_schema);
    AstralErr (ASTRAL_CALL * session_grammar_clear)(void* session_ctx);
    // Apply grammar constraints to a candidate list (in-place).
    // `tokens[i]` corresponds to `logits[i]` (logits are mutable scratch; set -inf for disallowed).
    AstralErr (ASTRAL_CALL * session_apply_grammar)(void* session_ctx, uint32_t* tokens, float* logits, uint32_t count);

    // Slot-scoped grammar ops (for continuous batching executors).
    // If implemented, these must not depend on `session_set_slot()` and should apply to the given slot directly.
    AstralErr (ASTRAL_CALL * session_grammar_set_gbnf_for_slot)(void* session_ctx, uint32_t slot_id, AstralSpanU8 gbnf, AstralSpanU8 root);
    AstralErr (ASTRAL_CALL * session_grammar_set_json_schema_for_slot)(void* session_ctx, uint32_t slot_id, AstralSpanU8 json_schema);
    AstralErr (ASTRAL_CALL * session_grammar_clear_for_slot)(void* session_ctx, uint32_t slot_id);
    AstralErr (ASTRAL_CALL * session_apply_grammar_for_slot)(void* session_ctx, uint32_t slot_id, uint32_t* tokens, float* logits, uint32_t count);

    // KV/session state (save/load) support.
    // Providers should return ASTRAL_E_UNSUPPORTED if not implemented.
    AstralErr (ASTRAL_CALL * session_state_size)(void* session_ctx, uint64_t* out_bytes);
    AstralErr (ASTRAL_CALL * session_state_save)(void* session_ctx, uint8_t* out_bytes, uint64_t capacity, uint64_t* out_written);
    AstralErr (ASTRAL_CALL * session_state_load)(void* session_ctx, const uint8_t* bytes, uint64_t size);

    // LoRA/adapters support.
    // Adapter lifetime is model-scoped (the adapter references the model).
    void* (ASTRAL_CALL * model_adapter_load)(void* model_ctx, AstralSpanU8 path, AstralErr* out_err);
    void (ASTRAL_CALL * model_adapter_unload)(void* model_ctx, void* adapter_ctx);
    AstralErr (ASTRAL_CALL * session_adapter_clear)(void* session_ctx);
    AstralErr (ASTRAL_CALL * session_adapter_add)(void* session_ctx, void* adapter_ctx, float scale);

    // Slots (parallel prompts) support.
    AstralErr (ASTRAL_CALL * session_set_slot)(void* session_ctx, uint32_t slot_id);
    // Optional: query current slot position (n_past) for a given slot.
    AstralErr (ASTRAL_CALL * session_slot_pos)(void* session_ctx, uint32_t slot_id, uint32_t* out_pos);
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
