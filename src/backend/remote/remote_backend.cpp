#include "../backend.hpp"
#include "../../core/runtime_alloc.hpp"
#include "../../utils/trace.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <cpp-httplib/httplib.h>

namespace astral::backend {
namespace {

constexpr uint32_t kRemoteVocabSize = 258;
constexpr int32_t kRemoteTokenBos = 256;
constexpr int32_t kRemoteTokenEos = 257;
constexpr uint32_t kRemoteDefaultCtx = 4096;
constexpr uint32_t kRemoteDefaultEmbeddingDim = 8;
constexpr uint32_t kRemoteMaxPromptBytes = 8192;
constexpr uint32_t kRemoteMaxOutputBytes = 8192;
constexpr uint32_t kRemoteMaxUrlBytes = 512;
constexpr uint32_t kRemoteMaxApiKeyBytes = 256;
constexpr uint32_t kRemoteTimeoutSeconds = 5;
constexpr uint32_t kRemoteMaxAttempts = 2;
constexpr float kRemoteSelectedLogit = 1000.0f;
constexpr float kRemoteSuppressedLogit = -1000.0f;

struct RemoteModel {
    char base_url[kRemoteMaxUrlBytes];
    char api_key[kRemoteMaxApiKeyBytes];
    uint32_t ctx_size;
    uint32_t embedding_dim;
};

struct RemoteSession {
    RemoteModel* model;
    uint8_t prompt[kRemoteMaxPromptBytes];
    uint8_t output[kRemoteMaxOutputBytes];
    float logits[kRemoteVocabSize];
    uint32_t prompt_len;
    uint32_t output_len;
    uint32_t output_pos;
    bool fetched;
};

struct RemoteEmbedder {
    RemoteModel* model;
};

static bool span_copy_nt(AstralSpanU8 span, char* dst, uint32_t cap) {
    if (dst == nullptr || cap == 0 || span.data == nullptr || span.len == 0 || span.len >= cap) {
        return false;
    }
    std::memcpy(dst, span.data, span.len);
    dst[span.len] = '\0';
    return true;
}

static bool is_http_url(const char* url) {
    return std::strncmp(url, "http://", 7) == 0 || std::strncmp(url, "https://", 8) == 0;
}

static bool is_https_url(const char* url) {
    return std::strncmp(url, "https://", 8) == 0;
}

static bool remote_tls_available() {
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    return true;
#else
    return false;
#endif
}

static httplib::Headers remote_headers(const RemoteModel* model) {
    httplib::Headers headers;
    if (model != nullptr && model->api_key[0] != '\0') {
        headers.emplace("Authorization", std::string("Bearer ") + model->api_key);
    }
    return headers;
}

static AstralErr http_status_to_err(int status) {
    if (status == 200) {
        return ASTRAL_OK;
    }
    if (status == 404) {
        return ASTRAL_E_NOT_FOUND;
    }
    if (status == 408 || status == 504) {
        return ASTRAL_E_TIMEOUT;
    }
    if (status == 501 || status == 405) {
        return ASTRAL_E_UNSUPPORTED;
    }
    return ASTRAL_E_BACKEND;
}

static bool http_status_retryable(int status) {
    return status == 408 || status == 429 || status == 500 || status == 502 || status == 503 || status == 504;
}

static AstralErr post_remote(const RemoteModel* model, const char* path, const char* body, uint32_t body_len, std::string* out_body) {
    ASTRAL_ZONE_N("astral.remote.transport_wait");
    if (model == nullptr || path == nullptr || out_body == nullptr) {
        return ASTRAL_E_INVALID;
    }

    httplib::Client client(model->base_url);
    client.set_connection_timeout(kRemoteTimeoutSeconds, 0);
    client.set_read_timeout(kRemoteTimeoutSeconds, 0);
    client.set_write_timeout(kRemoteTimeoutSeconds, 0);

    const std::string payload(body != nullptr ? body : "", body_len);
    AstralErr last_err = ASTRAL_E_TIMEOUT;
    for (uint32_t attempt = 0; attempt < kRemoteMaxAttempts; ++attempt) {
        auto result = client.Post(path, remote_headers(model), payload, "text/plain");
        if (!result) {
            last_err = ASTRAL_E_TIMEOUT;
            continue;
        }
        const AstralErr status_err = http_status_to_err(result->status);
        if (status_err == ASTRAL_OK) {
            *out_body = result->body;
            return ASTRAL_OK;
        }
        last_err = status_err;
        if (!http_status_retryable(result->status)) {
            return last_err;
        }
    }
    return last_err;
}

static AstralErr get_remote(const RemoteModel* model, const char* path) {
    ASTRAL_ZONE_N("astral.remote.transport_wait");
    if (model == nullptr || path == nullptr) {
        return ASTRAL_E_INVALID;
    }

    httplib::Client client(model->base_url);
    client.set_connection_timeout(kRemoteTimeoutSeconds, 0);
    client.set_read_timeout(kRemoteTimeoutSeconds, 0);
    client.set_write_timeout(kRemoteTimeoutSeconds, 0);

    AstralErr last_err = ASTRAL_E_TIMEOUT;
    for (uint32_t attempt = 0; attempt < kRemoteMaxAttempts; ++attempt) {
        auto result = client.Get(path, remote_headers(model));
        if (!result) {
            last_err = ASTRAL_E_TIMEOUT;
            continue;
        }
        last_err = http_status_to_err(result->status);
        if (last_err == ASTRAL_OK || !http_status_retryable(result->status)) {
            return last_err;
        }
    }
    return last_err;
}

static uint32_t parse_token_list(const char* text, uint32_t len, int32_t* out_tokens, uint32_t max_tokens) {
    uint32_t count = 0;
    const char* p = text;
    const char* end = text + len;
    while (p < end) {
        while (p < end && !std::isdigit(static_cast<unsigned char>(*p)) && *p != '-') {
            ++p;
        }
        if (p >= end) {
            break;
        }
        char* next = nullptr;
        const long v = std::strtol(p, &next, 10);
        if (next == p) {
            break;
        }
        if (out_tokens != nullptr && count < max_tokens) {
            out_tokens[count] = static_cast<int32_t>(v);
        }
        ++count;
        p = next;
    }
    return count;
}

static uint32_t parse_float_list(const char* text, uint32_t len, float* out_vec, uint32_t max_values) {
    uint32_t count = 0;
    const char* p = text;
    const char* end = text + len;
    while (p < end) {
        while (p < end && !std::isdigit(static_cast<unsigned char>(*p)) && *p != '-' && *p != '+') {
            ++p;
        }
        if (p >= end) {
            break;
        }
        char* next = nullptr;
        const float v = std::strtof(p, &next);
        if (next == p) {
            break;
        }
        if (out_vec != nullptr && count < max_values) {
            out_vec[count] = v;
        }
        ++count;
        p = next;
    }
    return count;
}

static uint32_t copy_response_text(const std::string& response, uint8_t* out, uint32_t cap) {
    const uint32_t n = static_cast<uint32_t>(std::min<std::size_t>(response.size(), cap));
    if (n != 0) {
        std::memcpy(out, response.data(), n);
    }
    return n;
}

void* remote_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    RemoteModel* model = core::runtime_new<RemoteModel>();
    if (model == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    std::memset(model, 0, sizeof(*model));

    if (!span_copy_nt(desc->model_path, model->base_url, sizeof(model->base_url)) || !is_http_url(model->base_url)) {
        core::runtime_delete(model);
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }
    if (is_https_url(model->base_url) && !remote_tls_available()) {
        core::runtime_delete(model);
        *out_err = ASTRAL_E_UNSUPPORTED;
        return nullptr;
    }

    if (desc->model_bytes.data != nullptr && desc->model_bytes.len != 0) {
        if (!span_copy_nt(desc->model_bytes, model->api_key, sizeof(model->api_key))) {
            core::runtime_delete(model);
            *out_err = ASTRAL_E_INVALID;
            return nullptr;
        }
    }

    model->ctx_size = desc->n_ctx != 0 ? desc->n_ctx : kRemoteDefaultCtx;
    model->embedding_dim = kRemoteDefaultEmbeddingDim;

    const AstralErr health = get_remote(model, "/health");
    if (health != ASTRAL_OK) {
        core::runtime_delete(model);
        *out_err = health;
        return nullptr;
    }

    *out_err = ASTRAL_OK;
    return model;
}

void remote_model_unload(void* model_ctx) {
    core::runtime_delete(static_cast<RemoteModel*>(model_ctx));
}

AstralErr remote_tokenize(void* model_ctx, AstralSpanU8 text, int32_t* out_tokens, uint32_t max_tokens,
                          uint8_t add_special, uint8_t parse_special, uint32_t* out_count) {
    (void)parse_special;
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr || out_count == nullptr || (text.len != 0 && text.data == nullptr)) {
        return ASTRAL_E_INVALID;
    }

    std::string body;
    AstralErr err = post_remote(model, "/tokenize", reinterpret_cast<const char*>(text.data), text.len, &body);
    if (err == ASTRAL_E_NOT_FOUND || err == ASTRAL_E_UNSUPPORTED) {
        const uint32_t needed = text.len + (add_special != 0 ? 1u : 0u);
        *out_count = needed;
        if (out_tokens == nullptr && max_tokens == 0) {
            return ASTRAL_OK;
        }
        if (needed > max_tokens) {
            return ASTRAL_E_NOMEM;
        }
        uint32_t n = 0;
        if (add_special != 0) {
            out_tokens[n++] = kRemoteTokenBos;
        }
        for (uint32_t i = 0; i < text.len; ++i) {
            out_tokens[n++] = static_cast<int32_t>(text.data[i]);
        }
        return ASTRAL_OK;
    }
    if (err != ASTRAL_OK) {
        return err;
    }

    const uint32_t remote_count = parse_token_list(body.data(), static_cast<uint32_t>(body.size()), out_tokens, max_tokens);
    const uint32_t needed = remote_count + (add_special != 0 ? 1u : 0u);
    *out_count = needed;
    if (out_tokens == nullptr && max_tokens == 0) {
        return ASTRAL_OK;
    }
    if (needed > max_tokens) {
        return ASTRAL_E_NOMEM;
    }
    if (add_special != 0) {
        for (uint32_t i = remote_count; i > 0; --i) {
            out_tokens[i] = out_tokens[i - 1u];
        }
        out_tokens[0] = kRemoteTokenBos;
    }
    return ASTRAL_OK;
}

AstralErr remote_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count, AstralMutSpanU8 out_text, uint32_t* out_len) {
    (void)model_ctx;
    if (out_len == nullptr || (count != 0 && tokens == nullptr)) {
        return ASTRAL_E_INVALID;
    }
    uint32_t needed = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t token = tokens[i];
        needed += (token >= 0 && token < 256) ? 1u : 0u;
    }
    *out_len = needed;
    if (out_text.data == nullptr && out_text.len == 0) {
        return ASTRAL_OK;
    }
    if (out_text.data == nullptr || out_text.len < needed) {
        return ASTRAL_E_NOMEM;
    }
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t token = tokens[i];
        if (token >= 0 && token < 256) {
            out_text.data[n++] = static_cast<uint8_t>(token);
        }
    }
    return ASTRAL_OK;
}

AstralErr remote_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_vocab_size = kRemoteVocabSize;
    *out_ctx_size = model->ctx_size;
    return ASTRAL_OK;
}

AstralErr remote_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    (void)model_ctx;
    if (out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_bos = kRemoteTokenBos;
    *out_eos = kRemoteTokenEos;
    return ASTRAL_OK;
}

AstralErr remote_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr || out_dim == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_dim = model->embedding_dim;
    return ASTRAL_OK;
}

void* remote_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
    (void)desc;
    if (out_err == nullptr) {
        return nullptr;
    }
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }
    RemoteSession* session = core::runtime_new<RemoteSession>();
    if (session == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    std::memset(session, 0, sizeof(*session));
    session->model = model;
    *out_err = ASTRAL_OK;
    return session;
}

void remote_session_destroy(void* session_ctx) {
    core::runtime_delete(static_cast<RemoteSession*>(session_ctx));
}

AstralErr remote_session_reset(void* session_ctx) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    session->prompt_len = 0;
    session->output_len = 0;
    session->output_pos = 0;
    session->fetched = false;
    return ASTRAL_OK;
}

AstralErr remote_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr || (count != 0 && tokens == nullptr)) {
        return ASTRAL_E_INVALID;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t token = tokens[i];
        if (token >= 0 && token < 256) {
            if (session->prompt_len >= sizeof(session->prompt)) {
                return ASTRAL_E_NOMEM;
            }
            session->prompt[session->prompt_len++] = static_cast<uint8_t>(token);
        }
    }
    return ASTRAL_OK;
}

AstralErr remote_session_logits(void* session_ctx, AstralBackendLogitsView* out_view) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (!session->fetched) {
        std::string body;
        const AstralErr err = post_remote(session->model, "/completion",
                                          reinterpret_cast<const char*>(session->prompt),
                                          session->prompt_len,
                                          &body);
        if (err != ASTRAL_OK) {
            return err;
        }
        session->output_len = copy_response_text(body, session->output, sizeof(session->output));
        session->output_pos = 0;
        session->fetched = true;
    }

    for (uint32_t i = 0; i < kRemoteVocabSize; ++i) {
        session->logits[i] = kRemoteSuppressedLogit;
    }
    const uint32_t next = session->output_pos < session->output_len
        ? static_cast<uint32_t>(session->output[session->output_pos])
        : static_cast<uint32_t>(kRemoteTokenEos);
    session->logits[next] = kRemoteSelectedLogit;

    out_view->logits = session->logits;
    out_view->vocab_size = kRemoteVocabSize;
    return ASTRAL_OK;
}

AstralErr remote_session_accept(void* session_ctx, int32_t token) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (token >= 0 && token < 256 && session->output_pos < session->output_len) {
        ++session->output_pos;
    }
    return ASTRAL_OK;
}

void* remote_embedder_create(void* model_ctx, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }
    RemoteEmbedder* embedder = core::runtime_new<RemoteEmbedder>();
    if (embedder == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    embedder->model = model;
    *out_err = ASTRAL_OK;
    return embedder;
}

void remote_embedder_destroy(void* embedder_ctx) {
    core::runtime_delete(static_cast<RemoteEmbedder*>(embedder_ctx));
}

AstralErr remote_embedder_reset(void* embedder_ctx) {
    return embedder_ctx != nullptr ? ASTRAL_OK : ASTRAL_E_INVALID;
}

AstralErr remote_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec, uint32_t vec_dim) {
    RemoteEmbedder* embedder = static_cast<RemoteEmbedder*>(embedder_ctx);
    if (embedder == nullptr || out_vec == nullptr || (count != 0 && tokens == nullptr)) {
        return ASTRAL_E_INVALID;
    }

    char text[kRemoteMaxPromptBytes];
    uint32_t text_len = 0;
    for (uint32_t i = 0; i < count && text_len < sizeof(text); ++i) {
        const int32_t token = tokens[i];
        if (token >= 0 && token < 256) {
            text[text_len++] = static_cast<char>(token);
        }
    }

    std::string body;
    const AstralErr err = post_remote(embedder->model, "/embeddings", text, text_len, &body);
    if (err != ASTRAL_OK) {
        return err;
    }
    for (uint32_t i = 0; i < vec_dim; ++i) {
        out_vec[i] = 0.0f;
    }
    const uint32_t parsed = parse_float_list(body.data(), static_cast<uint32_t>(body.size()), out_vec, vec_dim);
    return parsed != 0 ? ASTRAL_OK : ASTRAL_E_BACKEND;
}

const AstralBackendOps kRemoteOps = {
    .model_load = remote_model_load,
    .model_unload = remote_model_unload,
    .tokenize = remote_tokenize,
    .detokenize = remote_detokenize,
    .model_info = remote_model_info,
    .model_special_tokens = remote_model_special_tokens,
    .model_embedding_dim = remote_model_embedding_dim,
    .model_media_init = nullptr,
    .model_media_info = nullptr,
    .session_create = remote_session_create,
    .session_create_ex = nullptr,
    .session_destroy = remote_session_destroy,
    .session_reset = remote_session_reset,
    .session_feed = remote_session_feed,
    .session_feed_image = nullptr,
    .session_feed_audio = nullptr,
    .session_logits = remote_session_logits,
    .session_accept = remote_session_accept,
    .session_batch_eval = nullptr,
    .session_batch_logits = nullptr,
    .session_slot_reset = nullptr,
    .embedder_create = remote_embedder_create,
    .embedder_destroy = remote_embedder_destroy,
    .embedder_reset = remote_embedder_reset,
    .embedder_embed = remote_embedder_embed,
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

const AstralBackendProvider kRemoteProvider = {
    .name = "remote",
    .ops = &kRemoteOps,
    .supports_gpu = 0,
    .min_gpu_layers = 0,
};

} // namespace

const BackendProvider* builtin_remote_backend_provider() {
    return &kRemoteProvider;
}

} // namespace astral::backend
