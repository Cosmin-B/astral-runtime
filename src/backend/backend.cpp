/**
 * backend.cpp - Backend Registry Implementation
 *
 * Singleton registry for managing backend providers.
 * Backends auto-register at static initialization time.
 *
 * Thread-safety:
 * - register_backend(): NOT thread-safe (call at startup only)
 * - select_backend()/get_backend(): Thread-safe (read-only after startup)
 *
 * Memory:
 * - Registry does not own backend instances
 * - Backends must outlive all contexts (typically static instances)
 */

#include "backend.hpp"
#include <cstring>

namespace astral::backend {

// Built-in backend providers (linked into astral_rt).
const BackendProvider* builtin_cpu_backend_provider();
const BackendProvider* builtin_mock_backend_provider();

BackendRegistry::BackendRegistry() {
    (void)register_backend(builtin_cpu_backend_provider());
    (void)register_backend(builtin_mock_backend_provider());
}

// ============================================================================
// BackendRegistry Implementation
// ============================================================================

BackendRegistry& BackendRegistry::instance() {
    // C++11 guarantees thread-safe static local initialization
    static BackendRegistry instance;
    return instance;
}

AstralErr BackendRegistry::register_backend(const BackendProvider* provider) {
    if (!provider || !provider->name || !provider->ops) {
        return ASTRAL_E_INVALID;
    }

    // Prevent duplicates by name.
    if (get_backend(provider->name) != nullptr) {
        return ASTRAL_E_BUSY;
    }

    if (backend_count_ >= kMaxBackends) {
        return ASTRAL_E_BUSY;
    }

    backends_[backend_count_++] = provider;
    return ASTRAL_OK;
}

const BackendProvider* BackendRegistry::select_backend(uint32_t gpu_layers) {
    // Selection logic:
    // - gpu_layers == 0: CPU backend
    // - gpu_layers > 0: GPU backend (CUDA > Metal > DirectML) if available; fallback to CPU
    
    if (gpu_layers == 0) {
        // CPU only; find CPU backend
        return get_backend("cpu");
    }

    // GPU requested; try GPU backends in preference order
    // Priority: CUDA > Metal > DirectML
    const BackendProvider* gpu_backend = get_backend("cuda");
    if (gpu_backend) return gpu_backend;

    gpu_backend = get_backend("metal");
    if (gpu_backend) return gpu_backend;

    gpu_backend = get_backend("directml");
    if (gpu_backend) return gpu_backend;

    // Fallback to CPU if no GPU backend available
    return get_backend("cpu");
}

const BackendProvider* BackendRegistry::get_backend(const char* name) {
    if (!name) return nullptr;

    // Linear search (small array, fast)
    for (size_t i = 0; i < backend_count_; ++i) {
        if (backends_[i] && backends_[i]->name && std::strcmp(backends_[i]->name, name) == 0) {
            return backends_[i];
        }
    }

    return nullptr;
}

} // namespace astral::backend
