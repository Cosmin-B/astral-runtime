#include "adapter.hpp"

#include "../core/handles.hpp"
#include "../core/error.hpp"

namespace astral::inference {

AstralErr adapter_load(Model* model, const AstralAdapterDesc* desc, Adapter** out_adapter) {
    if (model == nullptr || desc == nullptr || out_adapter == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (desc->size != sizeof(AstralAdapterDesc)) {
        return ASTRAL_E_INVALID;
    }

    if (desc->path.data == nullptr || desc->path.len == 0) {
        return ASTRAL_E_INVALID;
    }

    if (model->backend == nullptr || model->backend->ops == nullptr || model->backend->ops->model_adapter_load == nullptr ||
        model->backend->ops->model_adapter_unload == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    AstralErr backend_err = ASTRAL_OK;
    void* adapter_ctx = model->backend->ops->model_adapter_load(model->backend_model_ctx, desc->path, &backend_err);
    if (adapter_ctx == nullptr) {
        return backend_err != ASTRAL_OK ? backend_err : ASTRAL_E_BACKEND;
    }

    Adapter* a = new (std::nothrow) Adapter{};
    if (a == nullptr) {
        model->backend->ops->model_adapter_unload(model->backend_model_ctx, adapter_ctx);
        return ASTRAL_E_NOMEM;
    }

    // Hold model reference for adapter lifetime.
    model->refcount.fetch_add(1, std::memory_order_relaxed);

    a->handle = 0;
    a->model = model;
    a->backend_adapter_ctx = adapter_ctx;
    a->refcount.store(1, std::memory_order_relaxed);

    const AstralHandle h = core::register_handle(core::HandleKind::Adapter, a);
    if (h == 0) {
        model->backend->ops->model_adapter_unload(model->backend_model_ctx, adapter_ctx);
        model_release(model);
        delete a;
        return ASTRAL_E_BUSY;
    }
    a->handle = h;

    *out_adapter = a;
    return ASTRAL_OK;
}

void adapter_retain(Adapter* a) {
    if (a == nullptr) {
        return;
    }
    a->refcount.fetch_add(1, std::memory_order_relaxed);
}

void adapter_release(Adapter* a) {
    if (a == nullptr) {
        return;
    }

    const uint32_t prev = a->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev != 1) {
        return;
    }

    // Invalidate handle.
    core::unregister_handle(a->handle, core::HandleKind::Adapter);
    a->handle = 0;

    // Unload backend adapter.
    if (a->model != nullptr && a->model->backend != nullptr && a->model->backend->ops != nullptr &&
        a->model->backend->ops->model_adapter_unload != nullptr && a->backend_adapter_ctx != nullptr) {
        a->model->backend->ops->model_adapter_unload(a->model->backend_model_ctx, a->backend_adapter_ctx);
    }
    a->backend_adapter_ctx = nullptr;

    // Release model reference.
    model_release(a->model);
    a->model = nullptr;

    delete a;
}

} // namespace astral::inference

