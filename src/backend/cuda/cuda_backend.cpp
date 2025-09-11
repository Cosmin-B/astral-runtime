/**
 * cuda_backend.cpp - CUDA backend provider descriptor
 *
 * v0: This backend reuses the existing llama.cpp-based implementation from the CPU backend,
 * but is only registered when Astral is built with CUDA enabled (GGML_CUDA=ON).
 *
 * Selection:
 * - BackendRegistry prefers "cuda" when gpu_layers > 0.
 * - The backend name can also be forced via AstralModelDesc.backend_name = "cuda".
 */

#include "../backend.hpp"

namespace astral::backend {

// Implemented in src/backend/cpu/cpu_backend.cpp.
const BackendProvider* builtin_cpu_backend_provider();

namespace {

#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
static BackendProvider kCudaBackendProvider = {
    /*name=*/"cuda",
    /*ops=*/nullptr,
    /*supports_gpu=*/true,
    /*min_gpu_layers=*/1,
};
#endif

} // namespace

const BackendProvider* builtin_cuda_backend_provider() {
#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
    const BackendProvider* cpu = builtin_cpu_backend_provider();
    if (cpu == nullptr || cpu->ops == nullptr) {
        return nullptr;
    }

    // Reuse the llama.cpp ops table; in CUDA builds llama.cpp will offload according to gpu_layers.
    kCudaBackendProvider.ops = cpu->ops;
    return &kCudaBackendProvider;
#else
    return nullptr;
#endif
}

} // namespace astral::backend
