/**
 * api.cpp - C ABI exports for inference functions
 *
 * This file provides the C ABI layer for model and session operations.
 * All functions handle null checks and error propagation.
 */

#include "../../include/astral_rt.h"
#include "model.hpp"
#include "session.hpp"
#include "embedder.hpp"
#include "../core/error.hpp"
#include "../core/handles.hpp"

namespace {

inline void set_err_invalid(const char* what) {
    astral::core::set_last_errorf("Invalid parameter: %s", what ? what : "");
}

inline void set_err_unsupported(const char* what) {
    astral::core::set_last_errorf("Unsupported: %s", what ? what : "");
}

inline void set_err_code(AstralErr err) {
    astral::core::set_last_error_from_code(err);
}

} // namespace

extern "C" {

// ============================================================================
// Model API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_model_load(
    const AstralModelDesc* desc,
    AstralHandle* out_model
) {
    if (desc == nullptr || out_model == nullptr) {
        set_err_invalid("desc/out_model");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Model* model = nullptr;
    AstralErr err = astral::inference::model_load(desc, &model);
    if (err == ASTRAL_OK) {
        *out_model = model->handle;
    } else {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API void ASTRAL_CALL astral_model_release(AstralHandle model) {
    if (model == 0) {
        set_err_invalid("model");
        return;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return;
    }

    astral::inference::model_release(m);
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_info(AstralHandle model, AstralModelInfo* out_info) {
    if (model == 0 || out_info == nullptr) {
        set_err_invalid("model/out_info");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    uint32_t vocab_size = 0;
    uint32_t ctx_size = 0;
    AstralErr err = m->backend->ops->model_info(m->backend_model_ctx, &vocab_size, &ctx_size);
    if (err != ASTRAL_OK) {
        set_err_code(err);
        return err;
    }

    int32_t bos = -1;
    int32_t eos = -1;
    if (m->backend->ops->model_special_tokens != nullptr) {
        err = m->backend->ops->model_special_tokens(m->backend_model_ctx, &bos, &eos);
        if (err != ASTRAL_OK) {
            set_err_code(err);
            return err;
        }
    }

    out_info->vocab_size = vocab_size;
    out_info->ctx_size = ctx_size;
    out_info->token_bos = bos;
    out_info->token_eos = eos;
    return ASTRAL_OK;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_caps(AstralHandle model, AstralCaps* out_caps) {
    if (model == 0 || out_caps == nullptr) {
        set_err_invalid("model/out_caps");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    AstralCaps caps = 0;
    caps |= ASTRAL_CAP_SAMPLER_EXT;
    caps |= ASTRAL_CAP_STOP_SEQS;

    if (m->backend != nullptr && m->backend->ops != nullptr) {
        if (m->backend->ops->embedder_create != nullptr && m->backend->ops->embedder_embed != nullptr) {
            caps |= ASTRAL_CAP_EMBEDDINGS;
        }
        if (m->backend->supports_gpu) {
            caps |= ASTRAL_CAP_GPU_OFFLOAD;
        }
    }

    *out_caps = caps;
    return ASTRAL_OK;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_limits(AstralHandle model, AstralModelLimits* out_limits) {
    if (model == 0 || out_limits == nullptr) {
        set_err_invalid("model/out_limits");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    uint32_t vocab_size = 0;
    uint32_t ctx_size = 0;
    const AstralErr err = m->backend->ops->model_info(m->backend_model_ctx, &vocab_size, &ctx_size);
    if (err != ASTRAL_OK) {
        set_err_code(err);
        return err;
    }

    out_limits->vocab_size = vocab_size;
    out_limits->ctx_size = ctx_size;
    out_limits->max_batch = 0;
    out_limits->max_slots = 0;
    return ASTRAL_OK;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_embedding_dim(AstralHandle model, uint32_t* out_dim) {
    if (model == 0 || out_dim == nullptr) {
        set_err_invalid("model/out_dim");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const auto* backend = m->backend;
    if (backend == nullptr || backend->ops == nullptr || backend->ops->model_embedding_dim == nullptr) {
        *out_dim = 0;
        set_err_unsupported("embeddings");
        return ASTRAL_E_UNSUPPORTED;
    }

    const AstralErr err = backend->ops->model_embedding_dim(m->backend_model_ctx, out_dim);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_tokenize(
    AstralHandle model,
    AstralSpanU8 text,
    int32_t* out_tokens,
    uint32_t max_tokens,
    uint8_t add_special,
    uint8_t parse_special,
    uint32_t* out_count
) {
    if (model == 0 || out_tokens == nullptr || out_count == nullptr || max_tokens == 0) {
        set_err_invalid("model/out_tokens/out_count/max_tokens");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = m->backend->ops->tokenize(
        m->backend_model_ctx,
        text,
        out_tokens,
        max_tokens,
        add_special != 0,
        parse_special != 0,
        out_count
    );

    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_detokenize(
    AstralHandle model,
    const int32_t* tokens,
    uint32_t count,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
) {
    if (model == 0 || tokens == nullptr || out_text.data == nullptr || out_len == nullptr) {
        set_err_invalid("model/tokens/out_text/out_len");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = m->backend->ops->detokenize(m->backend_model_ctx, tokens, count, out_text, out_len);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

// ============================================================================
// Session API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_session_create(
    const AstralSessionDesc* desc,
    AstralHandle* out_session
) {
    if (desc == nullptr || out_session == nullptr) {
        set_err_invalid("desc/out_session");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Session* session = nullptr;
    AstralErr err = astral::inference::session_create(desc, &session);
    if (err == ASTRAL_OK) {
        *out_session = session->handle;
    } else {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API void ASTRAL_CALL astral_session_destroy(AstralHandle session) {
    if (session == 0) {
        set_err_invalid("session");
        return;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return;
    }

    astral::inference::session_destroy(s);
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed(
    AstralHandle session,
    AstralSpanU8 prompt_chunk,
    uint8_t finalize
) {
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_feed(
        s,
        prompt_chunk,
        finalize
    );
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_sampler(AstralHandle session, const AstralSamplerDesc* desc) {
    if (session == 0 || desc == nullptr) {
        set_err_invalid("session/desc");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_set_sampler(s, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_clear(AstralHandle session) {
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_stop_clear(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_add_utf8(AstralHandle session, AstralSpanU8 utf8) {
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_stop_add_utf8(s, utf8);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_decode(AstralHandle session) {
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_decode(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_cancel(AstralHandle session) {
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_cancel(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state(AstralHandle session, AstralSessionState* out_state) {
    if (session == 0 || out_state == nullptr) {
        set_err_invalid("session/out_state");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    astral::inference::SessionState state{};
    const AstralErr err = astral::inference::session_state(s, &state);
    if (err != ASTRAL_OK) {
        set_err_code(err);
        return err;
    }

    *out_state = static_cast<AstralSessionState>(state);
    return ASTRAL_OK;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_wait(AstralHandle session, uint32_t timeout_ms) {
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_wait(s, timeout_ms);
    if (err != ASTRAL_OK && err != ASTRAL_E_TIMEOUT) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_reset(AstralHandle session, const AstralSessionDesc* desc) {
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_reset(s, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API int32_t ASTRAL_CALL astral_stream_read(
    AstralHandle session,
    AstralMutSpanU8 out_buf,
    uint32_t timeout_ms
) {
    if (session == 0) {
        set_err_invalid("session");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const int32_t result = astral::inference::stream_read(s, out_buf, timeout_ms);

    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }

    return result;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stats(
    AstralHandle session,
    AstralStats* out_stats
) {
    if (session == 0 || out_stats == nullptr) {
        set_err_invalid("session/out_stats");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_stats(
        s,
        out_stats
    );

    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

// ============================================================================
// Embeddings API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_create(
    AstralHandle model,
    AstralHandle* out_embedder
) {
    if (model == 0 || out_embedder == nullptr) {
        set_err_invalid("model/out_embedder");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_create(m, out_embedder);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API void ASTRAL_CALL astral_embed_destroy(AstralHandle emb) {
    if (emb == 0) {
        set_err_invalid("emb");
        return;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return;
    }

    astral::inference::embedder_destroy(e);
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue(
    AstralHandle emb,
    AstralSpanU8 text,
    uint64_t* out_ticket
) {
    if (emb == 0 || out_ticket == nullptr) {
        set_err_invalid("emb/out_ticket");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_enqueue(e, text, out_ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_collect(
    AstralHandle emb,
    uint64_t ticket,
    AstralMutSpanU8 out_vector
) {
    if (emb == 0 || out_vector.data == nullptr) {
        set_err_invalid("emb/out_vector");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_collect(e, ticket, out_vector);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
}

} // extern "C"
