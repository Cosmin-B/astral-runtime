#include "adapter.hpp"

#include "../core/handles.hpp"
#include "../core/error.hpp"
#include "../core/runtime_alloc.hpp"

#include <cstring>

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

    Adapter* a = core::runtime_new<Adapter>();
    if (a == nullptr) {
        model->backend->ops->model_adapter_unload(model->backend_model_ctx, adapter_ctx);
        return ASTRAL_E_NOMEM;
    }

    uint8_t* path_copy = static_cast<uint8_t*>(core::runtime_alloc(desc->path.len, 1));
    if (path_copy == nullptr) {
        model->backend->ops->model_adapter_unload(model->backend_model_ctx, adapter_ctx);
        core::runtime_delete(a);
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(path_copy, desc->path.data, desc->path.len);

    // Hold model reference for adapter lifetime.
    model->refcount.fetch_add(1, std::memory_order_relaxed);

    a->handle = 0;
    a->model = model;
    a->backend_adapter_ctx = adapter_ctx;
    a->path = path_copy;
    a->path_len = desc->path.len;
    a->refcount.store(1, std::memory_order_relaxed);

    const AstralHandle h = core::register_handle(core::HandleKind::Adapter, a);
    if (h == 0) {
        model->backend->ops->model_adapter_unload(model->backend_model_ctx, adapter_ctx);
        model_release(model);
        core::runtime_free(path_copy, desc->path.len, 1);
        core::runtime_delete(a);
        return ASTRAL_E_BUSY;
    }
    a->handle = h;

    *out_adapter = a;
    return ASTRAL_OK;
}

AstralErr adapter_info(Adapter* a, AstralAdapterInfo* out_info) {
    if (a == nullptr || out_info == nullptr || out_info->size != sizeof(AstralAdapterInfo)) {
        return ASTRAL_E_INVALID;
    }
    out_info->model = a->model != nullptr ? a->model->handle : 0;
    out_info->path_bytes = a->path_len;
    out_info->refcount = a->refcount.load(std::memory_order_acquire);
    return ASTRAL_OK;
}

AstralErr adapter_path_copy(Adapter* a, AstralMutSpanU8 out_path, uint32_t* out_len) {
    if (a == nullptr || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_len = a->path_len;
    if (out_path.data == nullptr || out_path.len < a->path_len) {
        return ASTRAL_E_NOMEM;
    }
    if (a->path_len != 0) {
        std::memcpy(out_path.data, a->path, a->path_len);
    }
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

    if (a->path != nullptr && a->path_len != 0) {
        core::runtime_free(a->path, a->path_len, 1);
    }
    a->path = nullptr;
    a->path_len = 0;

    // Release model reference.
    model_release(a->model);
    a->model = nullptr;

    core::runtime_delete(a);
}

} // namespace astral::inference
