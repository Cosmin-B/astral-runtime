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
#include <atomic>
#include <vector>
#include <ggml-backend.h>

// Forward declarations (llama.cpp types)
struct llama_model;
struct llama_context;
struct llama_vocab;
struct llama_sampler;
#if ASTRAL_ENABLE_MTMD
struct mtmd_context;
#endif

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

    // GPU routing state for llama.cpp (optional; used in CUDA builds).
    std::vector<ggml_backend_dev_t> gpu_devices;
    std::vector<float> gpu_tensor_split;

#if ASTRAL_ENABLE_MTMD
    // Optional multimodal context (vision/audio)
    mtmd_context* mtmd;
    uint8_t media_initialized;
    uint8_t supports_vision;
    uint8_t supports_audio;
    uint8_t _padding0;
    uint32_t image_min_tokens;
    uint32_t image_max_tokens;
    int32_t audio_sample_rate;
    std::atomic_flag mtmd_lock = ATOMIC_FLAG_INIT;
#endif
};

/// CPU backend session context (opaque handle returned by session_create()).
///
/// Contains llama.cpp session state (KV cache) only.
struct CpuSession {
    CpuModel* model;    // Back-reference (not owned)
    llama_context* ctx; // Per-session inference context (KV cache)

    uint32_t n_batch;
    uint32_t n_slots;     // Number of supported slots (llama n_seq_max)
    uint32_t active_slot; // Currently selected slot

    uint32_t last_batch_outputs; // Output count from last session_batch_eval()
    uint32_t last_batch_token_count; // Token count from last session_batch_eval()

    // Per-slot state
    int32_t slot_pos[32];
    bool slot_has_logits[32];

    // Optional generation controls (per-slot; stateful)
    ::llama_sampler* grammar[32]; // nullptr if no grammar for this slot

    // Preallocated llama_batch buffers (no allocations in hot paths)
    int32_t* batch_token_storage;
    int32_t* batch_output_token_index;
    int32_t* batch_pos;
    int32_t* batch_n_seq_id;
    int32_t* batch_seq_id_storage;
    int32_t** batch_seq_id_ptrs;
    int8_t* batch_logits;
};

/// CPU backend embedder context (opaque handle returned by embedder_create()).
struct CpuEmbedder {
    CpuModel* model;
    llama_context* ctx;
    uint8_t use_encode;
    uint8_t mean_pooling;
};

} // namespace astral::backend
