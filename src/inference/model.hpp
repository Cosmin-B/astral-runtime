#pragma once

#include "../../include/astral_rt.h"
#include "../backend/backend.hpp"
#include "../core/handles.hpp"
#include <atomic>

namespace astral::inference {

/// Model handle structure.
///
/// Represents a loaded GGUF model with associated backend context.
///
/// Lifecycle:
/// - Created by model_load()
/// - Used by session_create()
/// - Destroyed by model_release()
///
/// Thread-safety: Read-only after initialization (safe to share across sessions)
struct Model {
    AstralHandle handle;                 // Public handle (type/index/generation)
    std::atomic<uint32_t> refcount;      // Strong refs (user + sessions)
    const backend::BackendProvider* backend; // Backend provider (CPU/CUDA/Metal/...)
    void* backend_model_ctx;             // Opaque backend model context
    AstralModelDesc desc;                // Model configuration (copy)
};

/// Load a GGUF model.
///
/// @param desc Model configuration (must not be NULL)
/// @param out_model Output: model handle (must not be NULL; valid only if return is ASTRAL_OK)
/// @return ASTRAL_OK on success, error code on failure
///
/// Error codes:
/// - ASTRAL_E_INVALID: desc or out_model is NULL
/// - ASTRAL_E_BACKEND: Backend initialization failed
/// - ASTRAL_E_NOMEM: Out of memory
///
/// Thread-safety: Safe to call from multiple threads
AstralErr model_load(const AstralModelDesc* desc, Model** out_model);

/// Release a model.
///
/// Frees backend resources and deallocates model handle.
/// All sessions using this model must be destroyed first.
///
/// @param model Model handle (may be NULL; no-op if NULL)
///
/// Thread-safety: Must not be in use by any session
void model_release(Model* model);

} // namespace astral::inference
