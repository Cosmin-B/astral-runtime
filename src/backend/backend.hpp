#pragma once

#include "../../include/astral_backend.h"
#include <cstdint>
#include <cstddef>

namespace astral::backend {

using BackendLogitsView = ::AstralBackendLogitsView;
using BackendOps = ::AstralBackendOps;
using BackendProvider = ::AstralBackendProvider;

/// Backend registry for managing available backends.
///
/// Singleton registry that stores all registered backends.
/// Backends register themselves at static initialization time.
///
/// Thread-safety: register_backend() is not thread-safe (called at startup only).
/// select_backend() is thread-safe after initialization.
class BackendRegistry {
public:
    /// Get singleton instance.
    static BackendRegistry& instance();

    /// Register a backend provider.
    ///
    /// @param backend Backend to register (must remain valid for program lifetime)
    ///
    /// IMPORTANT: Not thread-safe. Call during static initialization only.
    AstralErr register_backend(const BackendProvider* backend);

    /// Select best backend for given configuration.
    ///
    /// @param gpu_layers Number of GPU layers requested (0 = CPU only)
    /// @return Best available backend, or nullptr if none available
    ///
    /// Selection logic:
    /// - gpu_layers == 0: Use CPU backend
    /// - gpu_layers > 0: Use first available GPU backend (CUDA > Metal > DirectML)
    /// - Fallback: CPU backend if no GPU backend available
    ///
    /// Thread-safety: Safe to call concurrently after initialization
    const BackendProvider* select_backend(uint32_t gpu_layers);

    /// Get backend by name.
    ///
    /// @param name Backend name (e.g., "cpu", "cuda", "metal")
    /// @return Backend provider, or nullptr if not found
    ///
    /// Thread-safety: Safe to call concurrently after initialization
    const BackendProvider* get_backend(const char* name);

private:
    BackendRegistry();
    ~BackendRegistry() = default;

    // Non-copyable, non-movable (singleton)
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    static constexpr size_t kMaxBackends = 8;
    const BackendProvider* backends_[kMaxBackends] = {nullptr};
    size_t backend_count_ = 0;
};

} // namespace astral::backend
