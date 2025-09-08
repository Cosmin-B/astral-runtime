/**
 * sample_backend_plugin.cpp - Sample out-of-tree backend provider (plugin)
 *
 * Purpose:
 * - Demonstrate the backend plugin interface and validate provider swapping.
 * - Keep runtime dispatch cost identical to built-ins (ops-table indirection).
 *
 * This provider is deterministic and intentionally small:
 * - Tokenization: 1 byte -> 1 token (0..255), optional BOS.
 * - Detokenization: token 0..255 -> byte, BOS/EOS -> empty.
 * - Logits: argmax points to the next byte in a fixed message, then EOS.
 */

#include "astral_backend_plugin.h"

#include <new>
#include <cstring>

namespace {

struct SampleModel {
    uint32_t vocab_size;
    uint32_t ctx_size;
    uint32_t emb_dim;
    int32_t token_bos;
    int32_t token_eos;
    bool infinite;
};

struct SampleSession {
    SampleModel* model;
    uint32_t step;
    bool has_logits;
    float* logits;
};

struct SampleEmbedder {
    SampleModel* model;
};

constexpr uint32_t kVocabSize = 260;
constexpr int32_t kTokenBos = 256;
constexpr int32_t kTokenEos = 257;
constexpr uint32_t kEmbDim = 8;

constexpr const char* kMessage = "sample-backend";

inline bool span_equals_ascii(AstralSpanU8 span, const char* lit) {
    if (lit == nullptr) {
        return false;
    }
    const size_t lit_len = std::strlen(lit);
    if (span.data == nullptr || span.len != lit_len) {
        return false;
    }
    return std::memcmp(span.data, lit, lit_len) == 0;
}

void* ASTRAL_CALL sample_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    SampleModel* model = new (std::nothrow) SampleModel{};
    if (model == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    model->vocab_size = kVocabSize;
    model->ctx_size = desc->n_ctx != 0 ? desc->n_ctx : 2048;
    model->emb_dim = kEmbDim;
    model->token_bos = kTokenBos;
    model->infinite = span_equals_ascii(desc->model_path, "infinite");
    model->token_eos = model->infinite ? -1 : kTokenEos;

    *out_err = ASTRAL_OK;
    return model;
}

void ASTRAL_CALL sample_model_unload(void* model_ctx) {
    delete static_cast<SampleModel*>(model_ctx);
}

AstralErr ASTRAL_CALL sample_tokenize(void* model_ctx, AstralSpanU8 text,
                                      int32_t* out_tokens, uint32_t max_tokens,
                                      uint8_t add_special, uint8_t parse_special,
                                      uint32_t* out_count) {
    (void)parse_special;

    if (model_ctx == nullptr || out_tokens == nullptr || out_count == nullptr) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t needed = text.len + (add_special ? 1u : 0u);
    if (needed > max_tokens) {
        return ASTRAL_E_NOMEM;
    }

    uint32_t n = 0;
    if (add_special) {
        out_tokens[n++] = kTokenBos;
    }

    for (uint32_t i = 0; i < text.len; ++i) {
        out_tokens[n++] = static_cast<int32_t>(text.data ? text.data[i] : 0);
    }

    *out_count = n;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL sample_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count,
                                        AstralMutSpanU8 out_text, uint32_t* out_len) {
    if (model_ctx == nullptr || tokens == nullptr || out_text.data == nullptr || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t t = tokens[i];
        if (t == kTokenBos || t == kTokenEos) {
            continue;
        }

        if (offset >= out_text.len) {
            return ASTRAL_E_NOMEM;
        }

        const uint8_t b = (t >= 0 && t <= 255) ? static_cast<uint8_t>(t) : static_cast<uint8_t>('?');
        out_text.data[offset++] = b;
    }

    *out_len = offset;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL sample_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    if (model_ctx == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SampleModel* model = static_cast<SampleModel*>(model_ctx);
    *out_vocab_size = model->vocab_size;
    *out_ctx_size = model->ctx_size;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL sample_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    if (model_ctx == nullptr || out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SampleModel* model = static_cast<SampleModel*>(model_ctx);
    *out_bos = model->token_bos;
    *out_eos = model->token_eos;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL sample_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
    if (model_ctx == nullptr || out_dim == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SampleModel* model = static_cast<SampleModel*>(model_ctx);
    *out_dim = model->emb_dim;
    return ASTRAL_OK;
}

void* ASTRAL_CALL sample_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
    (void)desc;

    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    SampleModel* model = static_cast<SampleModel*>(model_ctx);

    SampleSession* session = new (std::nothrow) SampleSession{};
    if (session == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    session->logits = new (std::nothrow) float[model->vocab_size];
    if (session->logits == nullptr) {
        delete session;
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    session->model = model;
    session->step = 0;
    session->has_logits = false;

    *out_err = ASTRAL_OK;
    return session;
}

void ASTRAL_CALL sample_session_destroy(void* session_ctx) {
    if (session_ctx == nullptr) {
        return;
    }

    SampleSession* session = static_cast<SampleSession*>(session_ctx);
    delete[] session->logits;
    delete session;
}

AstralErr ASTRAL_CALL sample_session_reset(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SampleSession* session = static_cast<SampleSession*>(session_ctx);
    session->step = 0;
    session->has_logits = false;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL sample_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    (void)tokens;
    (void)count;

    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SampleSession* session = static_cast<SampleSession*>(session_ctx);
    session->step = 0;
    session->has_logits = true;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL sample_session_logits(void* session_ctx, AstralBackendLogitsView* out_view) {
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SampleSession* session = static_cast<SampleSession*>(session_ctx);
    if (!session->has_logits) {
        return ASTRAL_E_STATE;
    }

    const uint32_t vocab_size = session->model ? session->model->vocab_size : 0;
    if (vocab_size == 0 || session->logits == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    std::memset(session->logits, 0, sizeof(float) * vocab_size);

    int32_t next = kTokenEos;
    if (session->model && session->model->infinite) {
        next = static_cast<int32_t>(static_cast<uint8_t>('x'));
    } else {
        const size_t msg_len = std::strlen(kMessage);
        if (session->step < msg_len) {
            next = static_cast<int32_t>(static_cast<uint8_t>(kMessage[session->step]));
        }
    }

    if (next >= 0 && static_cast<uint32_t>(next) < vocab_size) {
        session->logits[next] = 1.0f;
    }

    out_view->logits = session->logits;
    out_view->vocab_size = vocab_size;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL sample_session_accept(void* session_ctx, int32_t token) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SampleSession* session = static_cast<SampleSession*>(session_ctx);
    if (!session->has_logits) {
        return ASTRAL_E_STATE;
    }

    if (token == kTokenEos && !(session->model && session->model->infinite)) {
        session->has_logits = false;
        return ASTRAL_OK;
    }

    session->step += 1;
    session->has_logits = true;
    return ASTRAL_OK;
}

void* ASTRAL_CALL sample_embedder_create(void* model_ctx, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    SampleEmbedder* e = new (std::nothrow) SampleEmbedder{};
    if (e == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    e->model = static_cast<SampleModel*>(model_ctx);
    *out_err = ASTRAL_OK;
    return e;
}

void ASTRAL_CALL sample_embedder_destroy(void* embedder_ctx) {
    delete static_cast<SampleEmbedder*>(embedder_ctx);
}

AstralErr ASTRAL_CALL sample_embedder_reset(void* embedder_ctx) {
    if (embedder_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL sample_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec, uint32_t vec_dim) {
    if (embedder_ctx == nullptr || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SampleEmbedder* e = static_cast<SampleEmbedder*>(embedder_ctx);
    if (e->model == nullptr || e->model->emb_dim == 0) {
        return ASTRAL_E_BACKEND;
    }

    if (tokens == nullptr && count > 0) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t dim = e->model->emb_dim;
    if (vec_dim < dim) {
        return ASTRAL_E_NOMEM;
    }

    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t t = tokens[i];
        sum += static_cast<uint32_t>(t >= 0 ? t : -t);
    }
    for (uint32_t i = 0; i < dim; ++i) {
        out_vec[i] = static_cast<float>(sum + i);
    }
    return ASTRAL_OK;
}

static const AstralBackendOps kOps = {
    .model_load = sample_model_load,
    .model_unload = sample_model_unload,

    .tokenize = sample_tokenize,
    .detokenize = sample_detokenize,

    .model_info = sample_model_info,
    .model_special_tokens = sample_model_special_tokens,
    .model_embedding_dim = sample_model_embedding_dim,

    .session_create = sample_session_create,
    .session_destroy = sample_session_destroy,
    .session_reset = sample_session_reset,
    .session_feed = sample_session_feed,
    .session_logits = sample_session_logits,
    .session_accept = sample_session_accept,

    .embedder_create = sample_embedder_create,
    .embedder_destroy = sample_embedder_destroy,
    .embedder_reset = sample_embedder_reset,
    .embedder_embed = sample_embedder_embed,

    .session_grammar_set_gbnf = nullptr,
    .session_grammar_set_json_schema = nullptr,
    .session_grammar_clear = nullptr,
    .session_apply_grammar = nullptr,

    .session_state_size = nullptr,
    .session_state_save = nullptr,
    .session_state_load = nullptr,

    .model_adapter_load = nullptr,
    .model_adapter_unload = nullptr,
    .session_adapter_clear = nullptr,
    .session_adapter_add = nullptr,

    .session_set_slot = nullptr,
};

static const AstralBackendProvider kProvider = {
    /*name=*/"sample",
    /*ops=*/&kOps,
    /*supports_gpu=*/0,
    /*min_gpu_layers=*/0,
};

} // namespace

extern "C" ASTRAL_BACKEND_PLUGIN_EXPORT const AstralBackendProvider* ASTRAL_CALL astral_backend_plugin_provider_v0() {
    return &kProvider;
}
