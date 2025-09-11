#include "embedder.hpp"

#include "model.hpp"
#include "../core/runtime_alloc.hpp"
#include "../platform/atomics.h"

#include <atomic>
#include <cstdint>
#include <cstring>

namespace astral::inference {

constexpr uint32_t kMaxEmbedTokens = 2048;
constexpr uint32_t kMaxInflight = 8;
constexpr uint32_t kTicketIndexMask = 0xFFFFu;
constexpr uint32_t kTicketIndexBits = 16;

struct Embedder {
    enum class SlotState : uint32_t {
        Free = 0,
        Ready = 1,
        Running = 2,
    };

    struct Slot {
        std::atomic<uint32_t> state;
        std::atomic<uint64_t> ticket;
        std::atomic<int32_t> result;
        std::atomic_flag lock = ATOMIC_FLAG_INIT;
        uint32_t token_count;
        int32_t tokens[kMaxEmbedTokens];
    };

    AstralHandle handle;
    Model* model;
    uint32_t dim;

    const backend::BackendProvider* backend;
    const backend::BackendOps* ops;

    void* backend_ctx[kMaxInflight];
    float* vectors; // kMaxInflight * dim

    std::atomic<uint64_t> ticket_counter;
    std::atomic_flag free_lock = ATOMIC_FLAG_INIT;
    uint32_t free_count;
    uint32_t free_stack[kMaxInflight];

    Slot slots[kMaxInflight];
};

namespace {

static void lock(std::atomic_flag* f) {
    uint32_t spins = 0;
    while (f->test_and_set(std::memory_order_acquire)) {
        if (spins < 64) {
            astral::platform::cpu_pause();
        } else {
            astral::platform::cpu_wait_for_event();
        }
        if (spins < 1024) {
            ++spins;
        }
    }
}

static void unlock(std::atomic_flag* f) {
    f->clear(std::memory_order_release);
    astral::platform::cpu_signal_event();
}

static uint32_t ticket_index(uint64_t ticket) {
    return static_cast<uint32_t>(ticket & kTicketIndexMask);
}

} // namespace

AstralErr embedder_create(Model* model, AstralHandle* out_embedder) {
    if (model == nullptr || out_embedder == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (model->desc.embeddings_only == 0) {
        return ASTRAL_E_STATE;
    }

    if (model->backend == nullptr || model->backend->ops == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const backend::BackendOps* ops = model->backend->ops;
    if (ops->model_embedding_dim == nullptr || ops->embedder_create == nullptr || ops->embedder_destroy == nullptr ||
        ops->embedder_embed == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    uint32_t dim = 0;
    AstralErr derr = ops->model_embedding_dim(model->backend_model_ctx, &dim);
    if (derr != ASTRAL_OK || dim == 0) {
        return derr != ASTRAL_OK ? derr : ASTRAL_E_BACKEND;
    }

    auto* e = core::runtime_new<Embedder>();
    if (e == nullptr) {
        return ASTRAL_E_NOMEM;
    }

    e->model = model;
    e->model->refcount.fetch_add(1, std::memory_order_relaxed);
    e->backend = model->backend;
    e->ops = ops;
    e->dim = dim;
    e->vectors = nullptr;
    e->ticket_counter.store(1, std::memory_order_relaxed);
    e->free_count = 0;

    for (uint32_t i = 0; i < kMaxInflight; ++i) {
        e->backend_ctx[i] = nullptr;
        e->slots[i].state.store(static_cast<uint32_t>(Embedder::SlotState::Free), std::memory_order_relaxed);
        e->slots[i].ticket.store(0, std::memory_order_relaxed);
        e->slots[i].result.store(static_cast<int32_t>(ASTRAL_OK), std::memory_order_relaxed);
        e->slots[i].token_count = 0;
    }

    e->vectors = core::runtime_alloc_array<float>(static_cast<uint32_t>(kMaxInflight * dim));
    if (e->vectors == nullptr) {
        model->refcount.fetch_sub(1, std::memory_order_relaxed);
        core::runtime_delete(e);
        return ASTRAL_E_NOMEM;
    }

    for (uint32_t i = 0; i < kMaxInflight; ++i) {
        AstralErr be = ASTRAL_E_BACKEND;
        void* ctx = ops->embedder_create(model->backend_model_ctx, &be);
        if (ctx == nullptr || be != ASTRAL_OK) {
            for (uint32_t j = 0; j < i; ++j) {
                ops->embedder_destroy(e->backend_ctx[j]);
            }
            core::runtime_free_array(e->vectors, static_cast<uint32_t>(kMaxInflight * dim));
            model->refcount.fetch_sub(1, std::memory_order_relaxed);
            core::runtime_delete(e);
            return be != ASTRAL_OK ? be : ASTRAL_E_BACKEND;
        }
        e->backend_ctx[i] = ctx;
    }

    // Initialize free list (all slots free).
    e->free_count = kMaxInflight;
    for (uint32_t i = 0; i < kMaxInflight; ++i) {
        e->free_stack[i] = i;
    }

    e->handle = core::register_handle(core::HandleKind::Embedder, e);
    if (e->handle == 0) {
        for (uint32_t i = 0; i < kMaxInflight; ++i) {
            ops->embedder_destroy(e->backend_ctx[i]);
        }
        core::runtime_free_array(e->vectors, static_cast<uint32_t>(kMaxInflight * dim));
        model->refcount.fetch_sub(1, std::memory_order_relaxed);
        core::runtime_delete(e);
        return ASTRAL_E_NOMEM;
    }

    *out_embedder = e->handle;
    return ASTRAL_OK;
}

void embedder_destroy(Embedder* embedder) {
    auto* e = embedder;
    if (e == nullptr) {
        return;
    }

    core::unregister_handle(e->handle, core::HandleKind::Embedder);

    // Best-effort: wait for in-flight work to complete before freeing backend contexts.
    for (uint32_t i = 0; i < kMaxInflight; ++i) {
        uint32_t spins = 0;
        while (e->slots[i].state.load(std::memory_order_acquire) == static_cast<uint32_t>(Embedder::SlotState::Running)) {
            if (spins < 64) {
                astral::platform::cpu_pause();
            } else {
                astral::platform::cpu_wait_for_event();
            }
            if (spins < 1024) {
                ++spins;
            }
        }
    }

    if (e->ops && e->ops->embedder_destroy) {
        for (uint32_t i = 0; i < kMaxInflight; ++i) {
            if (e->backend_ctx[i] != nullptr) {
                e->ops->embedder_destroy(e->backend_ctx[i]);
                e->backend_ctx[i] = nullptr;
            }
        }
    }

    core::runtime_free_array(e->vectors, static_cast<uint32_t>(kMaxInflight * e->dim));
    e->vectors = nullptr;

    if (e->model) {
        e->model->refcount.fetch_sub(1, std::memory_order_relaxed);
        e->model = nullptr;
    }

    core::runtime_delete(e);
}

AstralErr embedder_enqueue(Embedder* embedder, AstralSpanU8 text, uint64_t* out_ticket) {
    auto* e = embedder;
    if (e == nullptr || out_ticket == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (e->model == nullptr || e->ops == nullptr) {
        return ASTRAL_E_STATE;
    }

    uint32_t idx = UINT32_MAX;
    lock(&e->free_lock);
    if (e->free_count > 0) {
        idx = e->free_stack[--e->free_count];
    }
    unlock(&e->free_lock);

    if (idx == UINT32_MAX || idx >= kMaxInflight) {
        return ASTRAL_E_BUSY;
    }

    Embedder::Slot& slot = e->slots[idx];

    uint64_t counter = e->ticket_counter.fetch_add(1, std::memory_order_relaxed);
    if (counter == 0) {
        counter = e->ticket_counter.fetch_add(1, std::memory_order_relaxed);
    }

    const uint64_t ticket = (counter << kTicketIndexBits) | static_cast<uint64_t>(idx);

    uint32_t token_count = 0;
    const AstralErr tok_err = e->ops->tokenize(
        e->model->backend_model_ctx,
        text,
        slot.tokens,
        kMaxEmbedTokens,
        /*add_special=*/1,
        /*parse_special=*/0,
        &token_count
    );

    if (tok_err != ASTRAL_OK || token_count == 0) {
        lock(&e->free_lock);
        e->free_stack[e->free_count++] = idx;
        unlock(&e->free_lock);
        return tok_err != ASTRAL_OK ? tok_err : ASTRAL_E_BACKEND;
    }

    slot.token_count = token_count;
    slot.result.store(static_cast<int32_t>(ASTRAL_OK), std::memory_order_relaxed);
    slot.ticket.store(ticket, std::memory_order_release);
    slot.state.store(static_cast<uint32_t>(Embedder::SlotState::Ready), std::memory_order_release);

    *out_ticket = ticket;
    return ASTRAL_OK;
}

AstralErr embedder_collect(Embedder* embedder, uint64_t ticket, AstralMutSpanU8 out_vector) {
    auto* e = embedder;
    if (e == nullptr || out_vector.data == nullptr) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t idx = ticket_index(ticket);
    if (idx >= kMaxInflight) {
        return ASTRAL_E_INVALID;
    }

    Embedder::Slot& slot = e->slots[idx];
    if (slot.ticket.load(std::memory_order_acquire) != ticket) {
        return ASTRAL_E_INVALID;
    }

    lock(&slot.lock);

    const uint32_t dim = e->dim;
    const uint64_t need_bytes = static_cast<uint64_t>(dim) * sizeof(float);
    if (out_vector.len < need_bytes) {
        unlock(&slot.lock);

        // Free slot before returning.
        slot.ticket.store(0, std::memory_order_release);
        slot.state.store(static_cast<uint32_t>(Embedder::SlotState::Free), std::memory_order_release);
        lock(&e->free_lock);
        e->free_stack[e->free_count++] = idx;
        unlock(&e->free_lock);
        return ASTRAL_E_NOMEM;
    }

    const uint32_t state = slot.state.load(std::memory_order_acquire);
    if (state != static_cast<uint32_t>(Embedder::SlotState::Ready)) {
        unlock(&slot.lock);
        return ASTRAL_E_STATE;
    }

    slot.state.store(static_cast<uint32_t>(Embedder::SlotState::Running), std::memory_order_release);

    AstralErr err = ASTRAL_E_BACKEND;
    float* dst = e->vectors ? (e->vectors + static_cast<size_t>(idx) * static_cast<size_t>(dim)) : nullptr;
    void* ctx = e->backend_ctx[idx];
    if (dst != nullptr && ctx != nullptr && e->ops && e->ops->embedder_embed) {
        if (e->ops->embedder_reset) {
            (void)e->ops->embedder_reset(ctx);
        }
        err = e->ops->embedder_embed(ctx, slot.tokens, slot.token_count, dst, dim);
        if (err == ASTRAL_OK) {
            std::memcpy(out_vector.data, dst, static_cast<size_t>(need_bytes));
        }
    } else {
        err = ASTRAL_E_UNSUPPORTED;
    }

    slot.result.store(static_cast<int32_t>(err), std::memory_order_release);
    unlock(&slot.lock);

    slot.ticket.store(0, std::memory_order_release);
    slot.state.store(static_cast<uint32_t>(Embedder::SlotState::Free), std::memory_order_release);
    lock(&e->free_lock);
    e->free_stack[e->free_count++] = idx;
    unlock(&e->free_lock);

    return err;
}

} // namespace astral::inference
