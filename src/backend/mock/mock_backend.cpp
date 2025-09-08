/**
 * mock_backend.cpp - Mock backend provider (no llama dependency)
 *
 * Purpose:
 * - Validate Astral's provider-agnostic backend contract.
 * - Exercise the core decode + sampler loop without relying on llama.cpp.
 *
 * This backend is intentionally tiny and deterministic:
 * - Tokenization: 1 byte -> 1 token (0..255), optional BOS.
 * - Detokenization: token 0..255 -> byte, BOS/EOS -> empty.
 * - Logits: argmax points to the next byte in a fixed message, then EOS.
 */

#include "../backend.hpp"

#include <new>
#include <cstring>
#include <limits>

namespace astral::backend {

namespace {

struct MockModel {
    uint32_t vocab_size;
    uint32_t ctx_size;
    uint32_t emb_dim;
    int32_t token_bos;
    int32_t token_eos;
    bool infinite;
    bool sampler_mode;
};

struct MockSession {
    MockModel* model;
    uint32_t step;
    uint32_t slot_id;
    uint32_t adapter_count;
    uint32_t grammar_enabled;
    uint32_t grammar_allow_token; // 0..255, or 0xFFFFFFFF for none
    bool has_logits;
    float* logits;
};

struct MockEmbedder {
    MockModel* model;
};

constexpr uint32_t kMockVocabSize = 260;
constexpr int32_t kMockTokenBos = 256;
constexpr int32_t kMockTokenEos = 257;
constexpr uint32_t kMockEmbDim = 8;

constexpr const char* kMockMessage = "mock-backend";

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

void* mock_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }

    if (desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    MockModel* model = new (std::nothrow) MockModel{};
    if (model == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    model->vocab_size = kMockVocabSize;
    model->ctx_size = desc->n_ctx != 0 ? desc->n_ctx : 2048;
    model->emb_dim = kMockEmbDim;
    model->token_bos = kMockTokenBos;
    model->infinite = span_equals_ascii(desc->model_path, "infinite");
    model->sampler_mode = span_equals_ascii(desc->model_path, "sampler");
    model->token_eos = (model->infinite || model->sampler_mode) ? -1 : kMockTokenEos;

    *out_err = ASTRAL_OK;
    return model;
}

void mock_model_unload(void* model_ctx) {
    delete static_cast<MockModel*>(model_ctx);
}

AstralErr mock_tokenize(void* model_ctx, AstralSpanU8 text,
                        int32_t* out_tokens, uint32_t max_tokens,
                        uint8_t add_special, uint8_t parse_special,
                        uint32_t* out_count) {
    (void)parse_special;

    if (model_ctx == nullptr || out_tokens == nullptr || out_count == nullptr) {
        return ASTRAL_E_INVALID;
    }

    uint32_t needed = text.len + (add_special != 0 ? 1u : 0u);
    if (needed > max_tokens) {
        return ASTRAL_E_NOMEM;
    }

    uint32_t n = 0;
    if (add_special != 0) {
        out_tokens[n++] = kMockTokenBos;
    }

    for (uint32_t i = 0; i < text.len; ++i) {
        out_tokens[n++] = static_cast<int32_t>(text.data ? text.data[i] : 0);
    }

    *out_count = n;
    return ASTRAL_OK;
}

AstralErr mock_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count,
                          AstralMutSpanU8 out_text, uint32_t* out_len) {
    if (model_ctx == nullptr || tokens == nullptr || out_text.data == nullptr || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t t = tokens[i];
        if (t == kMockTokenBos || t == kMockTokenEos) {
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

AstralErr mock_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    if (model_ctx == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockModel* model = static_cast<MockModel*>(model_ctx);
    *out_vocab_size = model->vocab_size;
    *out_ctx_size = model->ctx_size;
    return ASTRAL_OK;
}

AstralErr mock_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    if (model_ctx == nullptr || out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockModel* model = static_cast<MockModel*>(model_ctx);
    *out_bos = model->token_bos;
    *out_eos = model->token_eos;
    return ASTRAL_OK;
}

AstralErr mock_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
    if (model_ctx == nullptr || out_dim == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockModel* model = static_cast<MockModel*>(model_ctx);
    *out_dim = model->emb_dim;
    return ASTRAL_OK;
}

void* mock_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }

    if (model_ctx == nullptr || desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    MockModel* model = static_cast<MockModel*>(model_ctx);

    MockSession* session = new (std::nothrow) MockSession{};
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
    session->slot_id = 0;
    session->adapter_count = 0;
    session->grammar_enabled = 0;
    session->grammar_allow_token = 0xFFFFFFFFu;
    session->has_logits = false;

    *out_err = ASTRAL_OK;
    return session;
}

void mock_session_destroy(void* session_ctx) {
    if (session_ctx == nullptr) {
        return;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    delete[] session->logits;
    delete session;
}

AstralErr mock_session_reset(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    session->step = 0;
    session->adapter_count = 0;
    session->has_logits = false;
    return ASTRAL_OK;
}

AstralErr mock_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    (void)tokens;
    (void)count;

    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    session->step = 0;
    session->has_logits = true;
    return ASTRAL_OK;
}

AstralErr mock_session_logits(void* session_ctx, BackendLogitsView* out_view) {
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (!session->has_logits || session->model == nullptr || session->logits == nullptr) {
        return ASTRAL_E_STATE;
    }

    // Default all logits low (small vocab; ok for tests).
    for (uint32_t i = 0; i < session->model->vocab_size; ++i) {
        session->logits[i] = -1000.0f;
    }

    if (session->model != nullptr && session->model->sampler_mode) {
        // Deterministic 2-token distribution for sampler tests.
        const uint32_t a = static_cast<uint32_t>('a');
        const uint32_t b = static_cast<uint32_t>('b');
        if (a < session->model->vocab_size) session->logits[a] = 10.0f;
        if (b < session->model->vocab_size) session->logits[b] = 10.0f;
    } else {
        const size_t msg_len = std::strlen(kMockMessage);
        int32_t next = kMockTokenEos;
        if (msg_len > 0 && session->model != nullptr && session->model->infinite) {
            next = static_cast<int32_t>(static_cast<uint8_t>(kMockMessage[session->step % msg_len]));
        } else if (session->step < msg_len) {
            next = static_cast<int32_t>(static_cast<uint8_t>(kMockMessage[session->step]));
        }

        if (next >= 0 && static_cast<uint32_t>(next) < session->model->vocab_size) {
            session->logits[static_cast<uint32_t>(next)] = 10.0f;
        }
    }

    out_view->logits = session->logits;
    out_view->vocab_size = session->model->vocab_size;
    return ASTRAL_OK;
}

AstralErr mock_session_accept(void* session_ctx, int32_t token) {
    (void)token;

    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (!session->has_logits) {
        return ASTRAL_E_STATE;
    }

    session->step++;
    return ASTRAL_OK;
}

// -----------------------------------------------------------------------------
// Optional generation controls
// -----------------------------------------------------------------------------

struct MockAdapter {
    uint32_t id;
};

static inline bool parse_u32_dec(AstralSpanU8 s, uint32_t* out) {
    if (out == nullptr || s.data == nullptr || s.len == 0) {
        return false;
    }
    uint32_t v = 0;
    for (uint32_t i = 0; i < s.len; ++i) {
        const uint8_t c = s.data[i];
        if (c < '0' || c > '9') {
            return false;
        }
        const uint32_t d = static_cast<uint32_t>(c - '0');
        const uint32_t nv = v * 10u + d;
        if (nv < v) {
            return false;
        }
        v = nv;
    }
    *out = v;
    return true;
}

AstralErr mock_session_grammar_set_gbnf(void* session_ctx, AstralSpanU8 gbnf, AstralSpanU8 root) {
    (void)root;
    if (session_ctx == nullptr || gbnf.data == nullptr || gbnf.len == 0) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);

    // Mock grammar format:
    // - "allow=<single ascii char>"
    // - "allow_byte=<0..255>"
    if (gbnf.len >= 6 && std::memcmp(gbnf.data, "allow=", 6) == 0) {
        if (gbnf.len != 7) {
            return ASTRAL_E_INVALID;
        }
        const uint8_t ch = gbnf.data[6];
        s->grammar_allow_token = static_cast<uint32_t>(ch);
        s->grammar_enabled = 1;
        return ASTRAL_OK;
    }

    if (gbnf.len >= 11 && std::memcmp(gbnf.data, "allow_byte=", 11) == 0) {
        AstralSpanU8 num{};
        num.data = gbnf.data + 11;
        num.len = gbnf.len - 11;
        uint32_t v = 0;
        if (!parse_u32_dec(num, &v) || v > 255u) {
            return ASTRAL_E_INVALID;
        }
        s->grammar_allow_token = v;
        s->grammar_enabled = 1;
        return ASTRAL_OK;
    }

    return ASTRAL_E_INVALID;
}

[[maybe_unused]] AstralErr mock_session_grammar_set_json_schema(void* session_ctx, AstralSpanU8 json_schema) {
    (void)session_ctx;
    (void)json_schema;
    return ASTRAL_E_UNSUPPORTED;
}

AstralErr mock_session_grammar_clear(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    s->grammar_enabled = 0;
    s->grammar_allow_token = 0xFFFFFFFFu;
    return ASTRAL_OK;
}

AstralErr mock_session_apply_grammar(void* session_ctx, uint32_t* tokens, float* logits, uint32_t count) {
    if (session_ctx == nullptr || tokens == nullptr || logits == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    if (s->grammar_enabled == 0 || s->grammar_allow_token > 255u) {
        return ASTRAL_OK;
    }
    const uint32_t allowed = s->grammar_allow_token;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < count; ++i) {
        if (tokens[i] != allowed) {
            logits[i] = neg_inf;
        }
    }
    return ASTRAL_OK;
}

AstralErr mock_session_state_size(void* session_ctx, uint64_t* out_bytes) {
    if (session_ctx == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_bytes = 16;
    return ASTRAL_OK;
}

AstralErr mock_session_state_save(void* session_ctx, uint8_t* out_bytes, uint64_t capacity, uint64_t* out_written) {
    if (session_ctx == nullptr || out_bytes == nullptr || out_written == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (capacity < 16) {
        return ASTRAL_E_NOMEM;
    }
    const MockSession* s = static_cast<const MockSession*>(session_ctx);
    std::memcpy(out_bytes + 0, &s->step, 4);
    std::memcpy(out_bytes + 4, &s->slot_id, 4);
    std::memcpy(out_bytes + 8, &s->grammar_allow_token, 4);
    std::memcpy(out_bytes + 12, &s->adapter_count, 4);
    *out_written = 16;
    return ASTRAL_OK;
}

AstralErr mock_session_state_load(void* session_ctx, const uint8_t* bytes, uint64_t size) {
    if (session_ctx == nullptr || bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (size < 16) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    std::memcpy(&s->step, bytes + 0, 4);
    std::memcpy(&s->slot_id, bytes + 4, 4);
    std::memcpy(&s->grammar_allow_token, bytes + 8, 4);
    std::memcpy(&s->adapter_count, bytes + 12, 4);
    s->grammar_enabled = (s->grammar_allow_token <= 255u) ? 1u : 0u;
    s->has_logits = true;
    return ASTRAL_OK;
}

void* mock_model_adapter_load(void* model_ctx, AstralSpanU8 path, AstralErr* out_err) {
    (void)model_ctx;
    if (out_err == nullptr) {
        return nullptr;
    }
    if (path.data == nullptr || path.len == 0) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < path.len; ++i) {
        h ^= static_cast<uint32_t>(path.data[i]);
        h *= 16777619u;
    }

    MockAdapter* a = new (std::nothrow) MockAdapter{};
    if (a == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    a->id = h;
    *out_err = ASTRAL_OK;
    return a;
}

void mock_model_adapter_unload(void* model_ctx, void* adapter_ctx) {
    (void)model_ctx;
    delete static_cast<MockAdapter*>(adapter_ctx);
}

AstralErr mock_session_adapter_clear(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    s->adapter_count = 0;
    return ASTRAL_OK;
}

AstralErr mock_session_adapter_add(void* session_ctx, void* adapter_ctx, float scale) {
    (void)scale;
    if (session_ctx == nullptr || adapter_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    s->adapter_count++;
    return ASTRAL_OK;
}

AstralErr mock_session_set_slot(void* session_ctx, uint32_t slot_id) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    s->slot_id = slot_id;
    return ASTRAL_OK;
}

void* mock_embedder_create(void* model_ctx, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    MockEmbedder* e = new (std::nothrow) MockEmbedder{};
    if (e == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    e->model = static_cast<MockModel*>(model_ctx);
    *out_err = ASTRAL_OK;
    return e;
}

void mock_embedder_destroy(void* embedder_ctx) {
    delete static_cast<MockEmbedder*>(embedder_ctx);
}

AstralErr mock_embedder_reset(void* embedder_ctx) {
    if (embedder_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    return ASTRAL_OK;
}

AstralErr mock_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec, uint32_t vec_dim) {
    if (embedder_ctx == nullptr || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockEmbedder* e = static_cast<MockEmbedder*>(embedder_ctx);
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

static const BackendOps kMockBackendOps = {
    .model_load = mock_model_load,
    .model_unload = mock_model_unload,

    .tokenize = mock_tokenize,
    .detokenize = mock_detokenize,

    .model_info = mock_model_info,
    .model_special_tokens = mock_model_special_tokens,
    .model_embedding_dim = mock_model_embedding_dim,

    .session_create = mock_session_create,
    .session_destroy = mock_session_destroy,
    .session_reset = mock_session_reset,
    .session_feed = mock_session_feed,
    .session_logits = mock_session_logits,
    .session_accept = mock_session_accept,

    .embedder_create = mock_embedder_create,
    .embedder_destroy = mock_embedder_destroy,
    .embedder_reset = mock_embedder_reset,
    .embedder_embed = mock_embedder_embed,

    .session_grammar_set_gbnf = mock_session_grammar_set_gbnf,
    .session_grammar_set_json_schema = nullptr,
    .session_grammar_clear = mock_session_grammar_clear,
    .session_apply_grammar = mock_session_apply_grammar,

    .session_state_size = mock_session_state_size,
    .session_state_save = mock_session_state_save,
    .session_state_load = mock_session_state_load,

    .model_adapter_load = mock_model_adapter_load,
    .model_adapter_unload = mock_model_adapter_unload,
    .session_adapter_clear = mock_session_adapter_clear,
    .session_adapter_add = mock_session_adapter_add,

    .session_set_slot = mock_session_set_slot,
};

static const BackendProvider kMockBackendProvider = {
    /*name=*/"mock",
    /*ops=*/&kMockBackendOps,
    /*supports_gpu=*/false,
    /*min_gpu_layers=*/0,
};

} // namespace

const BackendProvider* builtin_mock_backend_provider() {
    return &kMockBackendProvider;
}

} // namespace astral::backend
