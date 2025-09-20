/**
 * toy_backend_plugin.cpp - "toy" backend provider (plugin)
 *
 * Purpose:
 * - Provide a non-llama provider for contract + performance validation.
 * - Keep ops-table dispatch identical to built-ins/plugins.
 * - Support embeddings with configurable embedding dimension so we can benchmark
 *   output vector sizes (e.g., 256/512/1024 floats).
 *
 * Config (via AstralModelDesc.model_path; ASCII, optional):
 * - "dim=<N>" sets embedding dimension (floats). Default: 256.
 * - "infinite" makes EOS unavailable (generation runs until max_tokens).
 */

#include "astral_backend_plugin.h"

#include <new>
#include <cstdint>
#include <cstring>

namespace {

struct ToyModel {
    uint32_t vocab_size;
    uint32_t ctx_size;
    uint32_t emb_dim;
    int32_t token_bos;
    int32_t token_eos;
    bool infinite;
};

struct ToySession {
    ToyModel* model;
    uint32_t step;
    int32_t last_token;
    bool has_logits;
    float* logits;
};

struct ToyEmbedder {
    ToyModel* model;
};

constexpr uint32_t kVocabSize = 260;
constexpr int32_t kTokenBos = 256;
constexpr int32_t kTokenEos = 257;

static inline bool span_equals_ascii(AstralSpanU8 span, const char* lit) {
    if (lit == nullptr) {
        return false;
    }
    const size_t lit_len = std::strlen(lit);
    if (span.data == nullptr || span.len != lit_len) {
        return false;
    }
    return std::memcmp(span.data, lit, lit_len) == 0;
}

static inline uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static uint32_t parse_dim_from_model_path(AstralSpanU8 s, uint32_t fallback) {
    if (s.data == nullptr || s.len == 0) {
        return fallback;
    }

    const char* p = reinterpret_cast<const char*>(s.data);
    const char* end = p + s.len;

    for (; p + 4 < end; ++p) {
        if (p[0] == 'd' && p[1] == 'i' && p[2] == 'm' && p[3] == '=') {
            p += 4;
            uint32_t v = 0;
            bool any = false;
            while (p < end && *p >= '0' && *p <= '9') {
                any = true;
                v = v * 10u + static_cast<uint32_t>(*p - '0');
                ++p;
            }
            if (!any) {
                return fallback;
            }
            return clamp_u32(v, 1u, 8192u);
        }
    }

    return fallback;
}

void* ASTRAL_CALL toy_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    ToyModel* model = new (std::nothrow) ToyModel{};
    if (model == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    model->vocab_size = kVocabSize;
    model->ctx_size = desc->n_ctx != 0 ? desc->n_ctx : 2048;
    model->emb_dim = parse_dim_from_model_path(desc->model_path, 256u);
    model->token_bos = kTokenBos;
    model->infinite = span_equals_ascii(desc->model_path, "infinite");
    model->token_eos = model->infinite ? -1 : kTokenEos;

    *out_err = ASTRAL_OK;
    return model;
}

void ASTRAL_CALL toy_model_unload(void* model_ctx) {
    delete static_cast<ToyModel*>(model_ctx);
}

AstralErr ASTRAL_CALL toy_tokenize(void* model_ctx, AstralSpanU8 text,
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

AstralErr ASTRAL_CALL toy_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count,
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

AstralErr ASTRAL_CALL toy_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    if (model_ctx == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ToyModel* model = static_cast<ToyModel*>(model_ctx);
    *out_vocab_size = model->vocab_size;
    *out_ctx_size = model->ctx_size;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL toy_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    if (model_ctx == nullptr || out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ToyModel* model = static_cast<ToyModel*>(model_ctx);
    *out_bos = model->token_bos;
    *out_eos = model->token_eos;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL toy_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
    if (model_ctx == nullptr || out_dim == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ToyModel* model = static_cast<ToyModel*>(model_ctx);
    *out_dim = model->emb_dim;
    return ASTRAL_OK;
}

void* ASTRAL_CALL toy_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
    (void)desc;

    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    ToyModel* model = static_cast<ToyModel*>(model_ctx);

    ToySession* session = new (std::nothrow) ToySession{};
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
    session->last_token = kTokenBos;
    session->has_logits = false;

    *out_err = ASTRAL_OK;
    return session;
}

void ASTRAL_CALL toy_session_destroy(void* session_ctx) {
    if (session_ctx == nullptr) {
        return;
    }

    ToySession* session = static_cast<ToySession*>(session_ctx);
    delete[] session->logits;
    delete session;
}

AstralErr ASTRAL_CALL toy_session_reset(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ToySession* session = static_cast<ToySession*>(session_ctx);
    session->step = 0;
    session->last_token = kTokenBos;
    session->has_logits = false;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL toy_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (tokens == nullptr && count > 0) {
        return ASTRAL_E_INVALID;
    }

    ToySession* session = static_cast<ToySession*>(session_ctx);
    if (count > 0) {
        session->last_token = tokens[count - 1u];
        session->has_logits = true;
    }
    return ASTRAL_OK;
}

static inline uint32_t next_token_u32(uint32_t x) {
    // Simple LCG step (deterministic, fast, no heap).
    return x * 1664525u + 1013904223u;
}

AstralErr ASTRAL_CALL toy_session_logits(void* session_ctx, AstralBackendLogitsView* out_view) {
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ToySession* session = static_cast<ToySession*>(session_ctx);
    if (!session->has_logits || session->model == nullptr) {
        return ASTRAL_E_STATE;
    }

    const uint32_t vocab = session->model->vocab_size;
    float* logits = session->logits;
    if (logits == nullptr || vocab == 0) {
        return ASTRAL_E_BACKEND;
    }

    // Compute a deterministic "next token" distribution.
    const uint32_t seed = static_cast<uint32_t>(session->last_token) ^ (session->step * 0x9E3779B9u);
    const uint32_t r = next_token_u32(seed);
    const uint32_t next = r % 256u;

    for (uint32_t i = 0; i < vocab; ++i) {
        logits[i] = -1.0e9f;
    }
    logits[next] = 0.0f;

    out_view->logits = logits;
    out_view->vocab_size = vocab;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL toy_session_accept(void* session_ctx, int32_t token) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ToySession* session = static_cast<ToySession*>(session_ctx);
    if (!session->has_logits) {
        return ASTRAL_E_STATE;
    }

    session->last_token = token;
    session->step += 1;
    session->has_logits = true;
    return ASTRAL_OK;
}

void* ASTRAL_CALL toy_embedder_create(void* model_ctx, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    ToyModel* model = static_cast<ToyModel*>(model_ctx);
    ToyEmbedder* emb = new (std::nothrow) ToyEmbedder{};
    if (emb == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    emb->model = model;
    *out_err = ASTRAL_OK;
    return emb;
}

void ASTRAL_CALL toy_embedder_destroy(void* embedder_ctx) {
    delete static_cast<ToyEmbedder*>(embedder_ctx);
}

AstralErr ASTRAL_CALL toy_embedder_reset(void* embedder_ctx) {
    if (embedder_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL toy_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec,
                                         uint32_t vec_dim) {
    if (embedder_ctx == nullptr || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (tokens == nullptr && count > 0) {
        return ASTRAL_E_INVALID;
    }

    ToyEmbedder* emb = static_cast<ToyEmbedder*>(embedder_ctx);
    if (emb->model == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const uint32_t dim = emb->model->emb_dim;
    if (vec_dim < dim) {
        return ASTRAL_E_NOMEM;
    }

    // Deterministic "embedding": scale with dim and input length (useful for benchmarking).
    float acc = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        acc += static_cast<float>(tokens[i]);
    }

    for (uint32_t i = 0; i < dim; ++i) {
        out_vec[i] = acc + static_cast<float>(i) * 0.001f;
    }

    return ASTRAL_OK;
}

static const AstralBackendOps kOps = {
    .model_load = toy_model_load,
    .model_unload = toy_model_unload,

    .tokenize = toy_tokenize,
    .detokenize = toy_detokenize,

    .model_info = toy_model_info,
    .model_special_tokens = toy_model_special_tokens,
    .model_embedding_dim = toy_model_embedding_dim,
    .model_media_init = nullptr,
    .model_media_info = nullptr,

    .session_create = toy_session_create,
    .session_create_ex = nullptr,
    .session_destroy = toy_session_destroy,
    .session_reset = toy_session_reset,
    .session_feed = toy_session_feed,
    .session_feed_image = nullptr,
    .session_feed_audio = nullptr,
    .session_logits = toy_session_logits,
    .session_accept = toy_session_accept,

    .session_batch_eval = nullptr,
    .session_batch_logits = nullptr,
    .session_slot_reset = nullptr,

    .embedder_create = toy_embedder_create,
    .embedder_destroy = toy_embedder_destroy,
    .embedder_reset = toy_embedder_reset,
    .embedder_embed = toy_embedder_embed,
    .embedder_embed_image = nullptr,
    .embedder_embed_audio = nullptr,
    .embedder_embed_multimodal = nullptr,

    .session_grammar_set_gbnf = nullptr,
    .session_grammar_set_json_schema = nullptr,
    .session_grammar_clear = nullptr,
    .session_apply_grammar = nullptr,

    .session_grammar_set_gbnf_for_slot = nullptr,
    .session_grammar_set_json_schema_for_slot = nullptr,
    .session_grammar_clear_for_slot = nullptr,
    .session_apply_grammar_for_slot = nullptr,

    .session_state_size = nullptr,
    .session_state_save = nullptr,
    .session_state_load = nullptr,

    .model_adapter_load = nullptr,
    .model_adapter_unload = nullptr,
    .session_adapter_clear = nullptr,
    .session_adapter_add = nullptr,

    .session_set_slot = nullptr,
    .session_slot_pos = nullptr,
};

static const AstralBackendProvider kProvider = {
    /*name=*/"toy",
    /*ops=*/&kOps,
    /*supports_gpu=*/0,
    /*min_gpu_layers=*/0,
};

} // namespace

extern "C" {

ASTRAL_BACKEND_PLUGIN_EXPORT const AstralBackendProvider* ASTRAL_CALL astral_backend_plugin_provider_v0(void) {
    return &kProvider;
}

} // extern "C"
