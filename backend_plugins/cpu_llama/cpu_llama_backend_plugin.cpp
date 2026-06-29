/**
 * cpu_llama_backend_plugin.cpp - GGUF-capable CPU backend provider (plugin)
 *
 * Purpose:
 * - Provide a second "real" provider that can load and run GGUF models via the
 *   runtime plugin interface, validating provider swap without changing call sites.
 *
 * Notes:
 * - Provider dispatch remains a single indirect call through `AstralBackendOps`.
 * - Hot paths (feed/logits/accept/stream) must not allocate.
 */

#include "astral_backend_plugin.h"

#include <llama.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>
#include <thread>

namespace {

std::atomic<uint32_t> g_llama_backend_refs{0};

inline void llama_backend_ref() {
    if (g_llama_backend_refs.fetch_add(1, std::memory_order_acq_rel) == 0) {
        llama_backend_init();
    }
}

inline void llama_backend_unref() {
    if (g_llama_backend_refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        llama_backend_free();
    }
}

inline uint32_t clamp_threads(uint32_t v, uint32_t fallback) {
    return v > 0 ? v : fallback;
}

inline uint32_t default_threads_fallback() {
    uint32_t hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 4;
}

struct CpuModel {
    llama_model* model;
    const llama_vocab* vocab;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_threads;
    uint32_t n_threads_batch;
    uint32_t n_embd;
    uint32_t vocab_size;
    int32_t token_bos;
    int32_t token_eos;
};

struct CpuSession {
    CpuModel* model;
    llama_context* ctx;
    uint32_t n_batch;
    bool has_logits;
};

struct CpuEmbedder {
    CpuModel* model;
    llama_context* ctx;
    uint8_t use_encode;
    uint8_t mean_pooling;
};

void* ASTRAL_CALL cpu_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }
    if (desc->model_path.data == nullptr || desc->model_path.len == 0) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    llama_backend_ref();

    const size_t path_len = static_cast<size_t>(desc->model_path.len);
    char* path = new (std::nothrow) char[path_len + 1];
    if (path == nullptr) {
        llama_backend_unref();
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    std::memcpy(path, desc->model_path.data, path_len);
    path[path_len] = '\0';

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = static_cast<int32_t>(desc->gpu_layers);
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    llama_model* model = llama_model_load_from_file(path, model_params);
    delete[] path;

    if (model == nullptr) {
        llama_backend_unref();
        *out_err = ASTRAL_E_BACKEND;
        return nullptr;
    }

    CpuModel* cpu_model = new (std::nothrow) CpuModel{};
    if (cpu_model == nullptr) {
        llama_model_free(model);
        llama_backend_unref();
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    cpu_model->model = model;
    cpu_model->vocab = llama_model_get_vocab(model);
    if (cpu_model->vocab == nullptr) {
        llama_model_free(model);
        delete cpu_model;
        llama_backend_unref();
        *out_err = ASTRAL_E_BACKEND;
        return nullptr;
    }

    cpu_model->n_ctx = desc->n_ctx;
    if (cpu_model->n_ctx == 0) {
        const int32_t n_ctx_train = llama_model_n_ctx_train(model);
        if (n_ctx_train > 0) {
            cpu_model->n_ctx = static_cast<uint32_t>(n_ctx_train);
        }
    }
    if (cpu_model->n_ctx == 0) {
        cpu_model->n_ctx = 2048;
    }

    cpu_model->n_batch = desc->n_batch > 0 ? desc->n_batch : 512;
    if (cpu_model->n_batch > cpu_model->n_ctx) {
        cpu_model->n_batch = cpu_model->n_ctx;
    }
    cpu_model->n_threads = desc->n_threads;
    cpu_model->n_threads_batch = desc->n_threads;

    cpu_model->vocab_size = static_cast<uint32_t>(llama_vocab_n_tokens(cpu_model->vocab));
    cpu_model->token_bos = static_cast<int32_t>(llama_vocab_bos(cpu_model->vocab));
    cpu_model->token_eos = static_cast<int32_t>(llama_vocab_eos(cpu_model->vocab));
    const int32_t n_embd = llama_model_n_embd(model);
    cpu_model->n_embd = n_embd > 0 ? static_cast<uint32_t>(n_embd) : 0;

    *out_err = ASTRAL_OK;
    return cpu_model;
}

void ASTRAL_CALL cpu_model_unload(void* model_ctx) {
    if (model_ctx == nullptr) {
        return;
    }
    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    if (model->model) {
        llama_model_free(model->model);
    }
    delete model;
    llama_backend_unref();
}

AstralErr ASTRAL_CALL cpu_tokenize(void* model_ctx, AstralSpanU8 text,
                                   int32_t* out_tokens, uint32_t max_tokens,
                                   uint8_t add_special, uint8_t parse_special,
                                   uint32_t* out_count) {
    if (model_ctx == nullptr || out_count == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    const llama_vocab* vocab = model->vocab;
    if (vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const char* input = text.data ? reinterpret_cast<const char*>(text.data) : "";
    const int32_t input_len = static_cast<int32_t>(text.len);

    const int32_t n_tokens = llama_tokenize(
        vocab,
        input,
        input_len,
        reinterpret_cast<llama_token*>(out_tokens),
        static_cast<int32_t>(max_tokens),
        add_special != 0,
        parse_special != 0
    );

    if (n_tokens < 0) {
        *out_count = static_cast<uint32_t>(-n_tokens);
        return out_tokens == nullptr ? ASTRAL_OK : ASTRAL_E_NOMEM;
    }

    *out_count = static_cast<uint32_t>(n_tokens);
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL cpu_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count,
                                     AstralMutSpanU8 out_text, uint32_t* out_len) {
    if (model_ctx == nullptr || (tokens == nullptr && count != 0) || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    const llama_vocab* vocab = model->vocab;
    if (vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t space = out_text.data != nullptr ? out_text.len - offset : 0;
        if (out_text.data != nullptr && space == 0) {
            *out_len = offset;
            return ASTRAL_E_NOMEM;
        }

        char count_buf[1]{};
        char* piece_out = out_text.data != nullptr ? reinterpret_cast<char*>(out_text.data + offset) : count_buf;
        const int32_t written = llama_token_to_piece(
            vocab,
            static_cast<llama_token>(tokens[i]),
            piece_out,
            static_cast<int32_t>(space),
            0,
            false
        );

        if (written < 0) {
            offset += static_cast<uint32_t>(-written);
            if (out_text.data != nullptr) {
                *out_len = offset;
                return ASTRAL_E_NOMEM;
            }
            continue;
        }

        offset += static_cast<uint32_t>(written);
    }

    *out_len = offset;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL cpu_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    if (model_ctx == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    *out_vocab_size = model->vocab_size;
    *out_ctx_size = model->n_ctx;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL cpu_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    if (model_ctx == nullptr || out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    *out_bos = model->token_bos;
    *out_eos = model->token_eos;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL cpu_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
    if (model_ctx == nullptr || out_dim == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    if (model->n_embd == 0) {
        return ASTRAL_E_BACKEND;
    }

    *out_dim = model->n_embd;
    return ASTRAL_OK;
}

void* ASTRAL_CALL cpu_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr || desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = model->n_ctx;
    ctx_params.n_batch = model->n_batch;
    ctx_params.n_ubatch = ctx_params.n_batch;
    const uint32_t threads_default = default_threads_fallback();
    ctx_params.n_threads = static_cast<int32_t>(clamp_threads(model->n_threads, threads_default));
    ctx_params.n_threads_batch =
        static_cast<int32_t>(clamp_threads(model->n_threads_batch, static_cast<uint32_t>(ctx_params.n_threads)));
    ctx_params.n_seq_max = 1;

    llama_context* ctx = llama_init_from_model(model->model, ctx_params);
    if (ctx == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    CpuSession* session = new (std::nothrow) CpuSession{};
    if (session == nullptr) {
        llama_free(ctx);
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    session->model = model;
    session->ctx = ctx;
    session->n_batch = model->n_batch;
    session->has_logits = false;

    *out_err = ASTRAL_OK;
    return session;
}

void ASTRAL_CALL cpu_session_destroy(void* session_ctx) {
    if (session_ctx == nullptr) {
        return;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx) {
        llama_free(session->ctx);
    }
    delete session;
}

AstralErr ASTRAL_CALL cpu_session_reset(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    llama_memory_t mem = llama_get_memory(session->ctx);
    if (mem == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    llama_memory_clear(mem, true);
    session->has_logits = false;
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL cpu_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (count > 0 && tokens == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    const uint32_t batch_size = session->n_batch > 0 ? session->n_batch : count;
    for (uint32_t i = 0; i < count; i += batch_size) {
        const uint32_t n = (count - i) < batch_size ? (count - i) : batch_size;

        llama_token* chunk = const_cast<llama_token*>(
            reinterpret_cast<const llama_token*>(tokens + i)
        );

        const llama_batch batch = llama_batch_get_one(chunk, static_cast<int32_t>(n));
        const int32_t result = llama_decode(session->ctx, batch);
        if (result != 0) {
            return ASTRAL_E_BACKEND;
        }
    }

    if (count > 0) {
        session->has_logits = true;
    }

    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL cpu_session_logits(void* session_ctx, AstralBackendLogitsView* out_view) {
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (!session->has_logits) {
        return ASTRAL_E_STATE;
    }

    float* logits = llama_get_logits_ith(session->ctx, -1);
    if (logits == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    out_view->logits = logits;
    out_view->vocab_size = session->model ? session->model->vocab_size : 0;
    if (out_view->vocab_size == 0) {
        return ASTRAL_E_BACKEND;
    }

    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL cpu_session_accept(void* session_ctx, int32_t token) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (!session->has_logits) {
        return ASTRAL_E_STATE;
    }

    llama_token t = static_cast<llama_token>(token);
    llama_batch batch = llama_batch_get_one(&t, 1);
    const int32_t result = llama_decode(session->ctx, batch);
    if (result != 0) {
        return ASTRAL_E_BACKEND;
    }

    session->has_logits = true;
    return ASTRAL_OK;
}

void* ASTRAL_CALL cpu_embedder_create(void* model_ctx, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = model->n_ctx;
    ctx_params.n_batch = model->n_batch;
    ctx_params.n_ubatch = ctx_params.n_batch;
    const uint32_t threads_default = default_threads_fallback();
    ctx_params.n_threads = static_cast<int32_t>(clamp_threads(model->n_threads, threads_default));
    ctx_params.n_threads_batch =
        static_cast<int32_t>(clamp_threads(model->n_threads_batch, static_cast<uint32_t>(ctx_params.n_threads)));
    ctx_params.n_seq_max = 1;

    ctx_params.embeddings = true;
    ctx_params.no_perf = true;
    // Keep pooling off and do pooling in Astral to avoid allocator churn in pooled paths.
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (llama_model_has_encoder(model->model)) {
        ctx_params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    } else {
        ctx_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    llama_context* ctx = llama_init_from_model(model->model, ctx_params);
    if (ctx == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    llama_set_embeddings(ctx, true);

    CpuEmbedder* emb = new (std::nothrow) CpuEmbedder{};
    if (emb == nullptr) {
        llama_free(ctx);
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    emb->model = model;
    emb->ctx = ctx;
    emb->use_encode = llama_model_has_encoder(model->model) ? 1 : 0;
    emb->mean_pooling = emb->use_encode;
    *out_err = ASTRAL_OK;
    return emb;
}

void ASTRAL_CALL cpu_embedder_destroy(void* embedder_ctx) {
    if (embedder_ctx == nullptr) {
        return;
    }

    CpuEmbedder* emb = static_cast<CpuEmbedder*>(embedder_ctx);
    if (emb->ctx) {
        llama_free(emb->ctx);
    }
    delete emb;
}

AstralErr ASTRAL_CALL cpu_embedder_reset(void* embedder_ctx) {
    if (embedder_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuEmbedder* emb = static_cast<CpuEmbedder*>(embedder_ctx);
    if (emb->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    llama_memory_t mem = llama_get_memory(emb->ctx);
    if (mem == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    llama_memory_clear(mem, /*data=*/true);
    return ASTRAL_OK;
}

AstralErr ASTRAL_CALL cpu_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec, uint32_t vec_dim) {
    if (embedder_ctx == nullptr || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuEmbedder* emb = static_cast<CpuEmbedder*>(embedder_ctx);
    if (emb->ctx == nullptr || emb->model == nullptr || emb->model->model == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    if (tokens == nullptr && count > 0) {
        return ASTRAL_E_INVALID;
    }

    if (vec_dim < emb->model->n_embd) {
        return ASTRAL_E_NOMEM;
    }

    (void)cpu_embedder_reset(embedder_ctx);

    if (count == 0) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t dim = emb->model->n_embd;
    const bool use_encode = emb->use_encode != 0;
    const bool mean_pooling = emb->mean_pooling != 0;

    if (mean_pooling) {
        std::memset(out_vec, 0, static_cast<size_t>(dim) * sizeof(float));
    }

    const uint32_t batch_size = emb->model->n_batch > 0 ? emb->model->n_batch : count;
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; i += batch_size) {
        const uint32_t n = (count - i) < batch_size ? (count - i) : batch_size;
        llama_token* chunk = const_cast<llama_token*>(reinterpret_cast<const llama_token*>(tokens + i));
        const llama_batch batch = llama_batch_get_one(chunk, static_cast<int32_t>(n));
        const int32_t rc = use_encode ? llama_encode(emb->ctx, batch) : llama_decode(emb->ctx, batch);
        if (rc != 0) {
            return ASTRAL_E_BACKEND;
        }

        float* base = llama_get_embeddings(emb->ctx);
        if (base == nullptr) {
            return ASTRAL_E_BACKEND;
        }

        if (mean_pooling) {
            for (uint32_t t = 0; t < n; ++t) {
                const float* row = base + static_cast<size_t>(t) * static_cast<size_t>(dim);
                for (uint32_t k = 0; k < dim; ++k) {
                    out_vec[k] += row[k];
                }
            }
            total += n;
        } else {
            const float* last = base + static_cast<size_t>(n - 1u) * static_cast<size_t>(dim);
            std::memcpy(out_vec, last, static_cast<size_t>(dim) * sizeof(float));
        }
    }

    if (mean_pooling) {
        if (total == 0) {
            return ASTRAL_E_BACKEND;
        }
        const float inv = 1.0f / static_cast<float>(total);
        for (uint32_t k = 0; k < dim; ++k) {
            out_vec[k] *= inv;
        }
    }

    return ASTRAL_OK;
}

static const AstralBackendOps kOps = [] {
  AstralBackendOps ops{};
  ops.model_load = cpu_model_load;
  ops.model_unload = cpu_model_unload;
  ops.tokenize = cpu_tokenize;
  ops.detokenize = cpu_detokenize;
  ops.model_info = cpu_model_info;
  ops.model_special_tokens = cpu_model_special_tokens;
  ops.model_embedding_dim = cpu_model_embedding_dim;
  ops.session_create = cpu_session_create;
  ops.session_destroy = cpu_session_destroy;
  ops.session_reset = cpu_session_reset;
  ops.session_feed = cpu_session_feed;
  ops.session_logits = cpu_session_logits;
  ops.session_accept = cpu_session_accept;
  ops.embedder_create = cpu_embedder_create;
  ops.embedder_destroy = cpu_embedder_destroy;
  ops.embedder_reset = cpu_embedder_reset;
  ops.embedder_embed = cpu_embedder_embed;
  return ops;
}();

static const AstralBackendProvider kProvider = {
    /*name=*/"cpu_llama",
    /*ops=*/&kOps,
    /*supports_gpu=*/0,
    /*min_gpu_layers=*/0,
};

} // namespace

extern "C" ASTRAL_BACKEND_PLUGIN_EXPORT const AstralBackendProvider* ASTRAL_CALL astral_backend_plugin_provider_v0() {
    return &kProvider;
}
