#include "model.hpp"
#include "../backend/backend.hpp"
#include <cstring>
#include <new>

namespace astral::inference {

AstralErr model_load(const AstralModelDesc* desc, Model** out_model) {
    // Validate parameters
    if (desc == nullptr || out_model == nullptr) {
        return ASTRAL_E_INVALID;
    }

    // Select backend based on gpu_layers
    auto& registry = backend::BackendRegistry::instance();
    const backend::BackendProvider* backend = nullptr;
    if (desc->backend_name.data != nullptr && desc->backend_name.len > 0) {
        // Backend override is not a hot path; avoid heap allocation by bounding the name.
        char name_buf[64];
        if (desc->backend_name.len >= sizeof(name_buf)) {
            return ASTRAL_E_INVALID;
        }
        std::memcpy(name_buf, desc->backend_name.data, desc->backend_name.len);
        name_buf[desc->backend_name.len] = '\0';

        backend = registry.get_backend(name_buf);
    } else {
        backend = registry.select_backend(desc->gpu_layers);
    }
    if (backend == nullptr || backend->ops == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    // Initialize backend
    AstralErr err = ASTRAL_OK;
    void* backend_model_ctx = backend->ops->model_load(desc, &err);
    if (backend_model_ctx == nullptr) {
        return err != ASTRAL_OK ? err : ASTRAL_E_BACKEND;
    }

    // Allocate model handle
    Model* model = new (std::nothrow) Model;
    if (model == nullptr) {
        backend->ops->model_unload(backend_model_ctx);
        return ASTRAL_E_NOMEM;
    }

    // Initialize model
    model->handle = 0;
    model->refcount.store(1, std::memory_order_relaxed);
    model->backend = backend;
    model->backend_model_ctx = backend_model_ctx;
    model->desc = *desc;

    const AstralHandle handle = core::register_handle(core::HandleKind::Model, model);
    if (handle == 0) {
        backend->ops->model_unload(backend_model_ctx);
        delete model;
        return ASTRAL_E_BUSY;
    }
    model->handle = handle;

    *out_model = model;
    return ASTRAL_OK;
}

void model_release(Model* model) {
    if (model == nullptr) {
        return;
    }

    if (model->refcount.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }

    // Shutdown backend
    if (model->backend != nullptr && model->backend->ops != nullptr && model->backend_model_ctx != nullptr) {
        model->backend->ops->model_unload(model->backend_model_ctx);
    }

    core::unregister_handle(model->handle, core::HandleKind::Model);
    model->handle = 0;

    // Free model
    delete model;
}

} // namespace astral::inference
