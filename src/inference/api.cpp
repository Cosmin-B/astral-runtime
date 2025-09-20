/**
 * api.cpp - C ABI exports for inference functions
 *
 * This file provides the C ABI layer for model and session operations.
 * All functions handle null checks and error propagation.
 */

#include "../../include/astral_rt.h"
#include "model.hpp"
#include "session.hpp"
#include "executor.hpp"
#include "conversation_runtime.hpp"
#include "embedder.hpp"
#include "adapter.hpp"
#include "../core/error.hpp"
#include "../core/abi_guard.hpp"
#include "../core/handles.hpp"
#include "../core/runtime_state.hpp"
#include "../core/model_sources.hpp"
#include "../core/model_load_config.hpp"

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

inline AstralErr model_load_impl(const AstralModelDesc* desc, AstralHandle* out_model) {
    if (desc == nullptr || out_model == nullptr) {
        set_err_invalid("desc/out_model");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralModelDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    AstralModelDesc local = *desc;

    if (desc->source_kind == ASTRAL_MODEL_SOURCE_PATH) {
        if (desc->model_path.data == nullptr || desc->model_path.len == 0) {
            set_err_invalid("desc.model_path");
            return ASTRAL_E_INVALID;
        }
        astral::inference::Model* model = nullptr;
        astral::core::ModelLoadConfigScope scope(desc);
        const AstralErr err = astral::inference::model_load(&local, &model);
        if (err == ASTRAL_OK) {
            *out_model = model->handle;
        } else {
            set_err_code(err);
        }
        return err;
    }

    if (desc->source_kind == ASTRAL_MODEL_SOURCE_MEMORY) {
        if (desc->model_bytes.data == nullptr || desc->model_bytes.len == 0) {
            set_err_invalid("desc.model_bytes");
            return ASTRAL_E_INVALID;
        }

        astral::core::ModelSource src{};
        src.kind = ASTRAL_MODEL_SOURCE_MEMORY;
        src.bytes = desc->model_bytes;

        char token[64]{};
        uint64_t id = 0;
        const AstralErr reg = astral::core::model_source_register(src, &id, token, sizeof(token));
        if (reg != ASTRAL_OK) {
            set_err_code(reg);
            return reg;
        }

        local.model_path.data = reinterpret_cast<const uint8_t*>(token);
        local.model_path.len = static_cast<uint32_t>(std::strlen(token));
        local.source_kind = ASTRAL_MODEL_SOURCE_PATH;

        astral::inference::Model* model = nullptr;
        astral::core::ModelLoadConfigScope scope(desc);
        const AstralErr err = astral::inference::model_load(&local, &model);
        if (astral::core::model_source_present(id)) {
            astral::core::model_source_release(id);
        }
        if (err == ASTRAL_OK) {
            *out_model = model->handle;
        } else {
            set_err_code(err);
        }
        return err;
    }

    if (desc->source_kind == ASTRAL_MODEL_SOURCE_IO) {
        if (desc->io.size == nullptr || desc->io.read_at == nullptr) {
            set_err_invalid("desc.io.size/read_at");
            return ASTRAL_E_INVALID;
        }

        astral::core::ModelSource src{};
        src.kind = ASTRAL_MODEL_SOURCE_IO;
        src.io = desc->io;

        char token[64]{};
        uint64_t id = 0;
        const AstralErr reg = astral::core::model_source_register(src, &id, token, sizeof(token));
        if (reg != ASTRAL_OK) {
            set_err_code(reg);
            return reg;
        }

        local.model_path.data = reinterpret_cast<const uint8_t*>(token);
        local.model_path.len = static_cast<uint32_t>(std::strlen(token));
        local.source_kind = ASTRAL_MODEL_SOURCE_PATH;

        astral::inference::Model* model = nullptr;
        astral::core::ModelLoadConfigScope scope(desc);
        const AstralErr err = astral::inference::model_load(&local, &model);
        if (astral::core::model_source_present(id)) {
            astral::core::model_source_release(id);
        }
        if (err == ASTRAL_OK) {
            *out_model = model->handle;
        } else {
            set_err_code(err);
        }
        return err;
    }

    set_err_invalid("desc.source_kind");
    return ASTRAL_E_INVALID;
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
    ASTRAL_ABI_TRY_BEGIN
    return model_load_impl(desc, out_model);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_load2(
    const AstralModelDesc* desc,
    AstralHandle* out_model
) {
    ASTRAL_ABI_TRY_BEGIN
    return model_load_impl(desc, out_model);
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_model_release(AstralHandle model) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_adapter_load(
    AstralHandle model,
    const AstralAdapterDesc* desc,
    AstralHandle* out_adapter
) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || desc == nullptr || out_adapter == nullptr) {
        set_err_invalid("model/desc/out_adapter");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Adapter* a = nullptr;
    const AstralErr err = astral::inference::adapter_load(m, desc, &a);
    if (err == ASTRAL_OK) {
        *out_adapter = a->handle;
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_model_adapter_release(AstralHandle adapter) {
    ASTRAL_ABI_TRY_BEGIN
    if (adapter == 0) {
        set_err_invalid("adapter");
        return;
    }

    auto* a =
        static_cast<astral::inference::Adapter*>(astral::core::lookup_handle(adapter, astral::core::HandleKind::Adapter));
    if (a == nullptr) {
        set_err_invalid("adapter (invalid handle)");
        return;
    }

    astral::inference::adapter_release(a);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_info(AstralHandle model, AstralModelInfo* out_info) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_caps(AstralHandle model, AstralCaps* out_caps) {
    ASTRAL_ABI_TRY_BEGIN
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
    caps |= ASTRAL_CAP_LOGPROBS;

    if (m->backend != nullptr && m->backend->ops != nullptr) {
        if (m->backend->ops->embedder_create != nullptr && m->backend->ops->embedder_embed != nullptr) {
            caps |= ASTRAL_CAP_EMBEDDINGS;
        }
        if (m->backend->supports_gpu) {
            caps |= ASTRAL_CAP_GPU_OFFLOAD;
        }

        if (m->backend->ops->session_state_size != nullptr && m->backend->ops->session_state_save != nullptr &&
            m->backend->ops->session_state_load != nullptr) {
            caps |= ASTRAL_CAP_KV_STATE;
        }

        if (m->backend->ops->model_adapter_load != nullptr && m->backend->ops->model_adapter_unload != nullptr &&
            m->backend->ops->session_adapter_clear != nullptr && m->backend->ops->session_adapter_add != nullptr) {
            caps |= ASTRAL_CAP_LORA;
        }

        if (m->backend->ops->session_grammar_set_gbnf != nullptr && m->backend->ops->session_grammar_clear != nullptr &&
            m->backend->ops->session_apply_grammar != nullptr) {
            caps |= ASTRAL_CAP_GRAMMAR;
            caps |= ASTRAL_CAP_GRAMMAR_GBNF;
        }

        if (m->backend->ops->session_grammar_set_json_schema != nullptr && m->backend->ops->session_grammar_clear != nullptr &&
            m->backend->ops->session_apply_grammar != nullptr) {
            caps |= ASTRAL_CAP_GRAMMAR;
            caps |= ASTRAL_CAP_GRAMMAR_JSON_SCHEMA;
        }

        if (m->backend->ops->session_set_slot != nullptr) {
            caps |= ASTRAL_CAP_SLOTS;
        }

        if (m->backend->ops->model_media_info != nullptr) {
            AstralMediaInfo info{};
            info.size = sizeof(AstralMediaInfo);
            if (m->backend->ops->model_media_info(m->backend_model_ctx, &info) == ASTRAL_OK) {
                if (info.supports_image != 0) {
                    caps |= ASTRAL_CAP_IMAGE;
                }
                if (info.supports_audio != 0) {
                    caps |= ASTRAL_CAP_AUDIO;
                }
            }
        }

        if ((caps & (ASTRAL_CAP_IMAGE | ASTRAL_CAP_AUDIO)) != 0) {
            if (m->backend->ops->embedder_embed_multimodal != nullptr ||
                m->backend->ops->embedder_embed_image != nullptr ||
                m->backend->ops->embedder_embed_audio != nullptr) {
                caps |= ASTRAL_CAP_MM_EMBEDDINGS;
            }
        }
    }

    *out_caps = caps;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_limits(AstralHandle model, AstralModelLimits* out_limits) {
    ASTRAL_ABI_TRY_BEGIN
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
    out_limits->max_batch = m->desc.n_batch;
    out_limits->max_slots =
        (m->backend && m->backend->ops &&
         m->backend->ops->session_create_ex != nullptr &&
         m->backend->ops->session_batch_eval != nullptr &&
         m->backend->ops->session_batch_logits != nullptr)
            ? astral::inference::ModelExecutor::kMaxSlotsHard
            : 1u;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_media_init(AstralHandle model, const AstralModelMediaDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || desc == nullptr) {
        set_err_invalid("model/desc");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralModelMediaDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const auto* backend = m->backend;
    if (backend == nullptr || backend->ops == nullptr || backend->ops->model_media_init == nullptr) {
        set_err_unsupported("media init");
        return ASTRAL_E_UNSUPPORTED;
    }

    const AstralErr err = backend->ops->model_media_init(m->backend_model_ctx, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_media_info(AstralHandle model, AstralMediaInfo* out_info) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || out_info == nullptr) {
        set_err_invalid("model/out_info");
        return ASTRAL_E_INVALID;
    }
    if (out_info->size != sizeof(AstralMediaInfo)) {
        set_err_invalid("out_info.size");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const auto* backend = m->backend;
    if (backend == nullptr || backend->ops == nullptr || backend->ops->model_media_info == nullptr) {
        set_err_unsupported("media info");
        return ASTRAL_E_UNSUPPORTED;
    }

    const AstralErr err = backend->ops->model_media_info(m->backend_model_ctx, out_info);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_executor_configure(AstralHandle model, const AstralExecutorDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || desc == nullptr) {
        set_err_invalid("model/desc");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralExecutorDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    if (m->executor.load(std::memory_order_acquire) != nullptr) {
        set_err_code(ASTRAL_E_STATE);
        return ASTRAL_E_STATE;
    }

    m->executor_desc = *desc;
    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_executor_tune(AstralHandle model, const AstralExecutorTuning* tuning) {
    ASTRAL_ABI_TRY_BEGIN
    if (model == 0 || tuning == nullptr) {
        set_err_invalid("model/tuning");
        return ASTRAL_E_INVALID;
    }
    if (tuning->size != sizeof(AstralExecutorTuning)) {
        set_err_invalid("tuning.size");
        return ASTRAL_E_INVALID;
    }

    auto* m = static_cast<astral::inference::Model*>(astral::core::lookup_handle(model, astral::core::HandleKind::Model));
    if (m == nullptr) {
        set_err_invalid("model (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    auto* ex = m->executor.load(std::memory_order_acquire);
    if (ex == nullptr) {
        set_err_code(ASTRAL_E_STATE);
        return ASTRAL_E_STATE;
    }

    if (tuning->max_prompt_tokens_per_slot_tick != 0) {
        ex->max_prompt_tokens_per_slot_per_tick.store(tuning->max_prompt_tokens_per_slot_tick, std::memory_order_relaxed);
    }

    return ASTRAL_OK;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_model_embedding_dim(AstralHandle model, uint32_t* out_dim) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
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
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_detokenize(
    AstralHandle model,
    const int32_t* tokens,
    uint32_t count,
    AstralMutSpanU8 out_text,
    uint32_t* out_len
) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

// ============================================================================
// Session API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_session_create(
    const AstralSessionDesc* desc,
    AstralHandle* out_session
) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_session_destroy(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed(
    AstralHandle session,
    AstralSpanU8 prompt_chunk,
    uint8_t finalize
) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed_image(
    AstralHandle session,
    const AstralImageDesc* image,
    uint8_t finalize
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || image == nullptr) {
        set_err_invalid("session/image");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_feed_image(s, image, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_feed_audio(
    AstralHandle session,
    const AstralAudioDesc* audio,
    uint8_t finalize
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || audio == nullptr) {
        set_err_invalid("session/audio");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_feed_audio(s, audio, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_sampler(AstralHandle session, const AstralSamplerDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_penalty_prompt_set_tokens(
    AstralHandle session,
    const int32_t* tokens,
    uint32_t count
) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_penalty_prompt_set_tokens(s, tokens, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_clear(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_add_utf8(AstralHandle session, AstralSpanU8 utf8) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stop_set_utf8(
    AstralHandle session,
    const AstralSpanU8* seqs,
    uint32_t count
) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_stop_set_utf8(s, seqs, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_logprobs(AstralHandle session, uint32_t n_probs) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_set_logprobs(s, n_probs);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_stream_read_meta(
    AstralHandle session,
    AstralTokenMeta* out_events,
    uint32_t capacity,
    uint32_t timeout_ms
) {
    ASTRAL_ABI_TRY_BEGIN
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

    const int32_t result = astral::inference::stream_read_meta(s, out_events, capacity, timeout_ms);
    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }
    return result;
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_size(AstralHandle session, uint64_t* out_bytes) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || out_bytes == nullptr) {
        set_err_invalid("session/out_bytes");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_state_size(s, out_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_save(
    AstralHandle session,
    AstralMutSpanU8 out_buf,
    uint64_t* out_written
) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || out_written == nullptr) {
        set_err_invalid("session/out_written");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_state_save(s, out_buf, out_written);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state_load(AstralHandle session, AstralSpanU8 state_bytes) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_state_load(s, state_bytes);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_clear(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_adapters_clear(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_adapters_add(AstralHandle session, AstralHandle adapter, float scale) {
    ASTRAL_ABI_TRY_BEGIN
    if (session == 0 || adapter == 0) {
        set_err_invalid("session/adapter");
        return ASTRAL_E_INVALID;
    }

    auto* s =
        static_cast<astral::inference::Session*>(astral::core::lookup_handle(session, astral::core::HandleKind::Session));
    if (s == nullptr) {
        set_err_invalid("session (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::session_adapters_add(s, adapter, scale);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_grammar_gbnf(AstralHandle session, AstralSpanU8 gbnf, AstralSpanU8 root) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_set_grammar_gbnf(s, gbnf, root);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_grammar_json_schema(AstralHandle session, AstralSpanU8 json_schema) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_set_grammar_json_schema(s, json_schema);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_clear_grammar(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_clear_grammar(s);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_set_slot(AstralHandle session, uint32_t slot_id) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::session_set_slot(s, slot_id);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_decode(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_cancel(AstralHandle session) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_state(AstralHandle session, AstralSessionState* out_state) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_wait(AstralHandle session, uint32_t timeout_ms) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_reset(AstralHandle session, const AstralSessionDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_stream_read(
    AstralHandle session,
    AstralMutSpanU8 out_buf,
    uint32_t timeout_ms
) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_session_stats(
    AstralHandle session,
    AstralStats* out_stats
) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

// ============================================================================
// Conversation API (continuous batching)
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_create(const AstralConvDesc* desc, AstralHandle* out_conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (desc == nullptr || out_conv == nullptr) {
        set_err_invalid("desc/out_conv");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralConvDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    astral::inference::Conversation* conv = nullptr;
    const AstralErr err = astral::inference::conv_create(desc, &conv);
    if (err == ASTRAL_OK) {
        *out_conv = conv->handle;
    } else {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_conv_destroy(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return;
    }

    astral::inference::conv_destroy(c);
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed(AstralHandle conv, AstralSpanU8 prompt_chunk, uint8_t finalize) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_feed(c, prompt_chunk, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed_image(AstralHandle conv, const AstralImageDesc* image, uint8_t finalize) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || image == nullptr) {
        set_err_invalid("conv/image");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_feed_image(c, image, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_feed_audio(AstralHandle conv, const AstralAudioDesc* audio, uint8_t finalize) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || audio == nullptr) {
        set_err_invalid("conv/audio");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_feed_audio(c, audio, finalize);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_decode(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_decode(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_cancel(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_cancel(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_state(AstralHandle conv, AstralSessionState* out_state) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || out_state == nullptr) {
        set_err_invalid("conv/out_state");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_state(c, out_state);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_wait(AstralHandle conv, uint32_t timeout_ms) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_wait(c, timeout_ms);
    if (err != ASTRAL_OK && err != ASTRAL_E_TIMEOUT) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_reset(AstralHandle conv, const AstralConvDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || desc == nullptr) {
        set_err_invalid("conv/desc");
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralConvDesc)) {
        set_err_invalid("desc.size");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_reset(c, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_sampler(AstralHandle conv, const AstralSamplerDesc* desc) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || desc == nullptr) {
        set_err_invalid("conv/desc");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_set_sampler(c, desc);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_penalty_prompt_set_tokens(
    AstralHandle conv, const int32_t* tokens, uint32_t count) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_penalty_prompt_set_tokens(c, tokens, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_clear(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_stop_clear(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_add_utf8(AstralHandle conv, AstralSpanU8 utf8) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_stop_add_utf8(c, utf8);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stop_set_utf8(
    AstralHandle conv, const AstralSpanU8* seqs, uint32_t count) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_stop_set_utf8(c, seqs, count);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_set_logprobs(AstralHandle conv, uint32_t n_probs) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_set_logprobs(c, n_probs);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_set_gbnf(AstralHandle conv, AstralSpanU8 gbnf, AstralSpanU8 root) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_grammar_set_gbnf(c, gbnf, root);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_set_json_schema(AstralHandle conv, AstralSpanU8 json_schema) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_grammar_set_json_schema(c, json_schema);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_grammar_clear(AstralHandle conv) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_grammar_clear(c);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_conv_stream_read(AstralHandle conv, AstralMutSpanU8 out_buf, uint32_t timeout_ms) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const int32_t result = astral::inference::conv_stream_read(c, out_buf, timeout_ms);
    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }
    return result;
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API int32_t ASTRAL_CALL astral_conv_stream_read_meta(
    AstralHandle conv, AstralTokenMeta* out_events, uint32_t capacity, uint32_t timeout_ms) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0) {
        set_err_invalid("conv");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const int32_t result = astral::inference::conv_stream_read_meta(c, out_events, capacity, timeout_ms);
    if (result < 0 && result != ASTRAL_E_TIMEOUT) {
        set_err_code(static_cast<AstralErr>(result));
    }
    return result;
    ASTRAL_ABI_CATCH_END_I32(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_conv_stats(AstralHandle conv, AstralConvStats* out_stats) {
    ASTRAL_ABI_TRY_BEGIN
    if (conv == 0 || out_stats == nullptr) {
        set_err_invalid("conv/out_stats");
        return ASTRAL_E_INVALID;
    }

    auto* c = static_cast<astral::inference::Conversation*>(
        astral::core::lookup_handle(conv, astral::core::HandleKind::Conversation));
    if (c == nullptr) {
        set_err_invalid("conv (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::conv_stats(c, out_stats);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

// ============================================================================
// Embeddings API
// ============================================================================

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_create(
    AstralHandle model,
    AstralHandle* out_embedder
) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API void ASTRAL_CALL astral_embed_destroy(AstralHandle emb) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue(
    AstralHandle emb,
    AstralSpanU8 text,
    uint64_t* out_ticket
) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_image(
    AstralHandle emb,
    const AstralImageDesc* image,
    uint64_t* out_ticket
) {
    ASTRAL_ABI_TRY_BEGIN
    if (emb == 0 || image == nullptr || out_ticket == nullptr) {
        set_err_invalid("emb/image/out_ticket");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_enqueue_image(e, image, out_ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_audio(
    AstralHandle emb,
    const AstralAudioDesc* audio,
    uint64_t* out_ticket
) {
    ASTRAL_ABI_TRY_BEGIN
    if (emb == 0 || audio == nullptr || out_ticket == nullptr) {
        set_err_invalid("emb/audio/out_ticket");
        return ASTRAL_E_INVALID;
    }

    auto* e =
        static_cast<astral::inference::Embedder*>(astral::core::lookup_handle(emb, astral::core::HandleKind::Embedder));
    if (e == nullptr) {
        set_err_invalid("emb (invalid handle)");
        return ASTRAL_E_INVALID;
    }

    const AstralErr err = astral::inference::embedder_enqueue_audio(e, audio, out_ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_enqueue_multimodal(
    AstralHandle emb,
    AstralSpanU8 text,
    const AstralImageDesc* image,
    const AstralAudioDesc* audio,
    uint64_t* out_ticket
) {
    ASTRAL_ABI_TRY_BEGIN
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

    const AstralErr err = astral::inference::embedder_enqueue_multimodal(e, text, image, audio, out_ticket);
    if (err != ASTRAL_OK) {
        set_err_code(err);
    }
    return err;
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

ASTRAL_API AstralErr ASTRAL_CALL astral_embed_collect(
    AstralHandle emb,
    uint64_t ticket,
    AstralMutSpanU8 out_vector
) {
    ASTRAL_ABI_TRY_BEGIN
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
    ASTRAL_ABI_CATCH_END_ERR(ASTRAL_E_BACKEND)
}

} // extern "C"
