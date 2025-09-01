/**
 * cpu_backend.hpp - CPU Backend (llama.cpp Integration)
 *
 * Provider implementation for llama.cpp CPU inference.
 *
 * Notes:
 * - This file is internal to Astral; callers must not include llama headers.
 * - The backend registers a static `BackendProvider` with `BackendRegistry`.
 * - Sampling is owned by core inference (provider exposes logits view + accept).
 */

#pragma once

#include "../backend.hpp"

// Forward declarations (llama.cpp types)
struct llama_model;
struct llama_context;
struct llama_vocab;

namespace astral::backend {

/// CPU backend model context (opaque handle returned by model_load()).
struct CpuModel {
    llama_model* model; // Model weights
    const llama_vocab* vocab; // Cached vocab pointer (read-only)

    // Configuration (copied from AstralModelDesc)
    uint32_t n_ctx; // Effective context size (tokens)
    uint32_t n_batch;
    uint32_t n_threads;
    uint32_t n_threads_batch;
    uint32_t n_embd;
    uint8_t embeddings_only;

    // Metadata cache
    uint32_t vocab_size;
    int32_t token_bos;
    int32_t token_eos;
};

/// CPU backend session context (opaque handle returned by session_create()).
///
/// Contains llama.cpp session state (KV cache) only.
struct CpuSession {
    CpuModel* model;    // Back-reference (not owned)
    llama_context* ctx; // Per-session inference context (KV cache)

    uint32_t n_batch;
    bool has_logits; // True once prompt has been fed and logits are available.
};

/// CPU backend embedder context (opaque handle returned by embedder_create()).
struct CpuEmbedder {
    CpuModel* model;
    llama_context* ctx;
    uint8_t use_encode;
    uint8_t mean_pooling;
};

} // namespace astral::backend
