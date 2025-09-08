/**
 * cpu_backend.cpp - CPU Backend Implementation
 *
 * llama.cpp integration for CPU inference.
 *
 * Model lifetime:
 * - model_load(): loads GGUF weights into a CpuModel (shareable)
 * - session_create(): creates a CpuSession (llama_context) per Astral session
 *
 * Hot-path constraints:
 * - No allocations in session_feed()/session_accept()/session_logits()
 * - Avoid llama_batch_init/free (heap); use llama_batch_get_one() (no alloc)
 */

#include "cpu_backend.hpp"

#include "../../core/runtime_state.hpp"
#include <llama.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <new>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>
#include "json-schema-to-grammar.h"

#if defined(_MSC_VER)
#include <malloc.h>
#define ASTRAL_ALLOCA _alloca
#else
#include <alloca.h>
#define ASTRAL_ALLOCA alloca
#endif

namespace astral::backend {

namespace {

constexpr uint32_t kCpuMaxSlots = 32;

std::atomic<uint32_t> g_llama_backend_refs{0};
std::atomic<int> g_llama_log_threshold{GGML_LOG_LEVEL_WARN};

void llama_log_callback(ggml_log_level level, const char* text, void*) {
    const int threshold = g_llama_log_threshold.load(std::memory_order_relaxed);
    if (threshold < 0) {
        return;
    }
    if (static_cast<int>(level) >= threshold) {
        if (text != nullptr) {
            std::fputs(text, stderr);
        }
    }
}

void llama_log_init_from_env() {
    const char* v = std::getenv("ASTRAL_LLAMA_LOG");
    if (v == nullptr || v[0] == '\0') {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_WARN, std::memory_order_relaxed);
        return;
    }

    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "none") == 0) {
        g_llama_log_threshold.store(-1, std::memory_order_relaxed);
        return;
    }
    if (std::strcmp(v, "error") == 0) {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_ERROR, std::memory_order_relaxed);
        return;
    }
    if (std::strcmp(v, "warn") == 0) {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_WARN, std::memory_order_relaxed);
        return;
    }
    if (std::strcmp(v, "info") == 0) {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_INFO, std::memory_order_relaxed);
        return;
    }
    if (std::strcmp(v, "debug") == 0) {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_DEBUG, std::memory_order_relaxed);
        return;
    }

    // Default: keep logs quiet by allowing only warnings/errors.
    g_llama_log_threshold.store(GGML_LOG_LEVEL_WARN, std::memory_order_relaxed);
}

inline void llama_backend_ref() {
    if (g_llama_backend_refs.fetch_add(1, std::memory_order_acq_rel) == 0) {
        llama_log_init_from_env();
        llama_log_set(llama_log_callback, nullptr);
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
    if (astral::core::runtime_initialized()) {
        const uint32_t rt = astral::core::runtime_thread_count();
        if (rt > 0) {
            return rt;
        }
    }
    uint32_t hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 4;
}

inline uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

uint32_t cpu_default_slots_from_env() {
    // Slots are a session-level setting; parsing env vars here is not a hot path.
    const char* v = std::getenv("ASTRAL_LLAMA_MAX_SLOTS");
    if (v == nullptr || v[0] == '\0') {
        return 1;
    }

    uint64_t n = 0;
    for (size_t i = 0; v[i] != '\0'; ++i) {
        const char c = v[i];
        if (c < '0' || c > '9') {
            return 1;
        }
        n = n * 10u + static_cast<uint64_t>(c - '0');
        if (n > 1000000u) {
            return 1;
        }
    }

    return clamp_u32(static_cast<uint32_t>(n), 1u, kCpuMaxSlots);
}

void* cpu_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
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

    // Initialize llama.cpp backend (reference-counted; best-effort, not a hot path).
    llama_backend_ref();

    // llama_model_load_from_file expects a NUL-terminated path.
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
    // Prefer mmap-backed weights for faster startup and lower RSS (when supported by the OS).
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
    cpu_model->n_threads = desc->n_threads; // 0 means default
    cpu_model->n_threads_batch = desc->n_threads;
    cpu_model->embeddings_only = desc->embeddings_only;

    cpu_model->vocab_size = static_cast<uint32_t>(llama_vocab_n_tokens(cpu_model->vocab));
    cpu_model->token_bos = static_cast<int32_t>(llama_vocab_bos(cpu_model->vocab));
    cpu_model->token_eos = static_cast<int32_t>(llama_vocab_eos(cpu_model->vocab));
    const int32_t n_embd = llama_model_n_embd(model);
    cpu_model->n_embd = n_embd > 0 ? static_cast<uint32_t>(n_embd) : 0;

    *out_err = ASTRAL_OK;
    return cpu_model;
}

void cpu_model_unload(void* model_ctx) {
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

AstralErr cpu_tokenize(void* model_ctx, AstralSpanU8 text,
                       int32_t* out_tokens, uint32_t max_tokens,
                       uint8_t add_special, uint8_t parse_special,
                       uint32_t* out_count) {
    if (model_ctx == nullptr || out_tokens == nullptr || out_count == nullptr) {
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
        // Buffer too small; llama.cpp returns negative required size.
        return ASTRAL_E_BACKEND;
    }

    *out_count = static_cast<uint32_t>(n_tokens);
    return ASTRAL_OK;
}

AstralErr cpu_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count,
                         AstralMutSpanU8 out_text, uint32_t* out_len) {
    if (model_ctx == nullptr || tokens == nullptr || out_text.data == nullptr || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    const llama_vocab* vocab = model->vocab;
    if (vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t space = out_text.len - offset;
        if (space == 0) {
            return ASTRAL_E_BACKEND;
        }

        const int32_t written = llama_token_to_piece(
            vocab,
            static_cast<llama_token>(tokens[i]),
            reinterpret_cast<char*>(out_text.data + offset),
            static_cast<int32_t>(space),
            0,     // lstrip (don't strip leading space)
            false  // special (don't render special tokens)
        );

        if (written < 0) {
            // Output buffer too small for this token piece.
            return ASTRAL_E_BACKEND;
        }

        offset += static_cast<uint32_t>(written);
    }

    *out_len = offset;
    return ASTRAL_OK;
}

AstralErr cpu_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    if (model_ctx == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    *out_vocab_size = model->vocab_size;
    *out_ctx_size = model->n_ctx;
    return ASTRAL_OK;
}

AstralErr cpu_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    if (model_ctx == nullptr || out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    *out_bos = model->token_bos;
    *out_eos = model->token_eos;
    return ASTRAL_OK;
}

AstralErr cpu_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
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

void* cpu_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
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
    ctx_params.n_seq_max = static_cast<int32_t>(cpu_default_slots_from_env());

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
    session->n_slots = static_cast<uint32_t>(ctx_params.n_seq_max);
    session->active_slot = 0;

    for (uint32_t i = 0; i < kCpuMaxSlots; ++i) {
        session->slot_pos[i] = 0;
        session->slot_has_logits[i] = false;
        session->grammar[i] = nullptr;
    }

    // Preallocate batch scratch.
    const uint32_t cap = session->n_batch > 0 ? session->n_batch : 1;
    session->batch_pos = new (std::nothrow) int32_t[cap];
    session->batch_n_seq_id = new (std::nothrow) int32_t[cap];
    session->batch_seq_id_storage = new (std::nothrow) int32_t[cap];
    session->batch_seq_id_ptrs = new (std::nothrow) int32_t*[cap];
    session->batch_logits = new (std::nothrow) int8_t[cap];

    if (session->batch_pos == nullptr || session->batch_n_seq_id == nullptr || session->batch_seq_id_storage == nullptr ||
        session->batch_seq_id_ptrs == nullptr || session->batch_logits == nullptr) {
        delete[] session->batch_pos;
        delete[] session->batch_n_seq_id;
        delete[] session->batch_seq_id_storage;
        delete[] session->batch_seq_id_ptrs;
        delete[] session->batch_logits;
        delete session;
        llama_free(ctx);
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    for (uint32_t i = 0; i < cap; ++i) {
        session->batch_seq_id_ptrs[i] = &session->batch_seq_id_storage[i];
    }

    *out_err = ASTRAL_OK;
    return session;
}

void cpu_session_destroy(void* session_ctx) {
    if (session_ctx == nullptr) {
        return;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    for (uint32_t i = 0; i < kCpuMaxSlots; ++i) {
        if (session->grammar[i] != nullptr) {
            llama_sampler_free(session->grammar[i]);
            session->grammar[i] = nullptr;
        }
    }
    delete[] session->batch_pos;
    delete[] session->batch_n_seq_id;
    delete[] session->batch_seq_id_storage;
    delete[] session->batch_seq_id_ptrs;
    delete[] session->batch_logits;
    if (session->ctx) {
        llama_free(session->ctx);
    }
    delete session;
}

AstralErr cpu_session_reset(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    // Clear KV/memory for all sequences (provider-side state).
    // This keeps the context allocated and avoids session re-creation overhead.
    llama_memory_t mem = llama_get_memory(session->ctx);
    if (mem == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    llama_memory_clear(mem, /*data=*/true);
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        session->slot_pos[i] = 0;
        session->slot_has_logits[i] = false;
        if (session->grammar[i] != nullptr) {
            llama_sampler_reset(session->grammar[i]);
        }
    }
    return ASTRAL_OK;
}

AstralErr cpu_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (count > 0 && tokens == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);

    if (session->active_slot >= session->n_slots) {
        return ASTRAL_E_STATE;
    }

    const uint32_t batch_cap = session->n_batch > 0 ? session->n_batch : count;
    const uint32_t batch_size = batch_cap > 0 ? batch_cap : count;
    int32_t slot_pos = session->slot_pos[session->active_slot];
    for (uint32_t i = 0; i < count; i += batch_size) {
        const uint32_t n = (count - i) < batch_size ? (count - i) : batch_size;

        llama_token* chunk = const_cast<llama_token*>(
            reinterpret_cast<const llama_token*>(tokens + i)
        );

        llama_batch batch{};
        batch.n_tokens = static_cast<int32_t>(n);
        batch.token = chunk;
        batch.embd = nullptr;
        batch.pos = reinterpret_cast<llama_pos*>(session->batch_pos);
        batch.n_seq_id = session->batch_n_seq_id;
        batch.seq_id = reinterpret_cast<llama_seq_id**>(session->batch_seq_id_ptrs);
        batch.logits = session->batch_logits;

        for (uint32_t j = 0; j < n; ++j) {
            session->batch_pos[j] = slot_pos + static_cast<int32_t>(j);
            session->batch_n_seq_id[j] = 1;
            session->batch_seq_id_storage[j] = static_cast<int32_t>(session->active_slot);
            session->batch_logits[j] = 0;
        }
        session->batch_logits[n - 1u] = 1;

        const int32_t result = llama_decode(session->ctx, batch);
        if (result != 0) {
            return ASTRAL_E_BACKEND;
        }

        slot_pos += static_cast<int32_t>(n);
    }

    if (count > 0) {
        session->slot_pos[session->active_slot] = slot_pos;
        session->slot_has_logits[session->active_slot] = true;
    }

    return ASTRAL_OK;
}

AstralErr cpu_session_logits(void* session_ctx, BackendLogitsView* out_view) {
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->active_slot >= session->n_slots) {
        return ASTRAL_E_STATE;
    }
    if (!session->slot_has_logits[session->active_slot]) {
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

AstralErr cpu_session_accept(void* session_ctx, int32_t token) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->active_slot >= session->n_slots) {
        return ASTRAL_E_STATE;
    }
    if (!session->slot_has_logits[session->active_slot]) {
        return ASTRAL_E_STATE;
    }

    llama_token t = static_cast<llama_token>(token);
    llama_batch batch{};
    batch.n_tokens = 1;
    batch.token = &t;
    batch.embd = nullptr;
    batch.pos = reinterpret_cast<llama_pos*>(session->batch_pos);
    batch.n_seq_id = session->batch_n_seq_id;
    batch.seq_id = reinterpret_cast<llama_seq_id**>(session->batch_seq_id_ptrs);
    batch.logits = session->batch_logits;

    session->batch_pos[0] = session->slot_pos[session->active_slot];
    session->batch_n_seq_id[0] = 1;
    session->batch_seq_id_storage[0] = static_cast<int32_t>(session->active_slot);
    session->batch_logits[0] = 1;

    const int32_t result = llama_decode(session->ctx, batch);
    if (result != 0) {
        return ASTRAL_E_BACKEND;
    }

    if (session->grammar[session->active_slot] != nullptr) {
        llama_sampler_accept(session->grammar[session->active_slot], t);
    }

    session->slot_pos[session->active_slot] += 1;
    session->slot_has_logits[session->active_slot] = true;
    return ASTRAL_OK;
}

// -----------------------------------------------------------------------------
// Optional generation controls
// -----------------------------------------------------------------------------

AstralErr cpu_session_grammar_set_gbnf(void* session_ctx, AstralSpanU8 gbnf, AstralSpanU8 root) {
    if (session_ctx == nullptr || gbnf.data == nullptr || gbnf.len == 0) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->model == nullptr || session->model->vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    // llama.cpp expects NUL-terminated strings.
    const size_t g_len = static_cast<size_t>(gbnf.len);
    char* g = new (std::nothrow) char[g_len + 1];
    if (g == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(g, gbnf.data, g_len);
    g[g_len] = '\0';

    const size_t r_len = static_cast<size_t>(root.len);
    char* r = nullptr;
    if (root.data != nullptr && root.len > 0) {
        r = new (std::nothrow) char[r_len + 1];
        if (r == nullptr) {
            delete[] g;
            return ASTRAL_E_NOMEM;
        }
        std::memcpy(r, root.data, r_len);
        r[r_len] = '\0';
    }

    llama_sampler* new_grammars[kCpuMaxSlots]{};
    const char* root_name = r != nullptr ? r : "root";
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        new_grammars[i] = llama_sampler_init_grammar(session->model->vocab, g, root_name);
        if (new_grammars[i] == nullptr) {
            for (uint32_t j = 0; j < i; ++j) {
                llama_sampler_free(new_grammars[j]);
            }
            delete[] g;
            delete[] r;
            return ASTRAL_E_INVALID;
        }
    }

    delete[] g;
    delete[] r;

    for (uint32_t i = 0; i < session->n_slots; ++i) {
        if (session->grammar[i] != nullptr) {
            llama_sampler_free(session->grammar[i]);
        }
        session->grammar[i] = new_grammars[i];
    }
    return ASTRAL_OK;
}

[[maybe_unused]] AstralErr cpu_session_grammar_set_json_schema(void* session_ctx, AstralSpanU8 json_schema) {
    if (session_ctx == nullptr || json_schema.data == nullptr || json_schema.len == 0) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->model == nullptr || session->model->vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const char* begin = reinterpret_cast<const char*>(json_schema.data);
    const char* end = begin + json_schema.len;
    std::string schema_text(begin, end);

    nlohmann::ordered_json schema = nlohmann::ordered_json::parse(schema_text, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (schema.is_discarded()) {
        return ASTRAL_E_INVALID;
    }

    std::string gbnf;
    try {
        gbnf = json_schema_to_grammar(schema, /*force_gbnf=*/true);
    } catch (...) {
        return ASTRAL_E_INVALID;
    }

    AstralSpanU8 g{};
    g.data = reinterpret_cast<const uint8_t*>(gbnf.data());
    g.len = static_cast<uint32_t>(gbnf.size());

    AstralSpanU8 root{};
    return cpu_session_grammar_set_gbnf(session_ctx, g, root);
}

AstralErr cpu_session_grammar_clear(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        if (session->grammar[i] != nullptr) {
            llama_sampler_free(session->grammar[i]);
            session->grammar[i] = nullptr;
        }
    }
    return ASTRAL_OK;
}

AstralErr cpu_session_apply_grammar(void* session_ctx, uint32_t* tokens, float* logits, uint32_t count) {
    if (session_ctx == nullptr || tokens == nullptr || logits == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (count == 0 || session->active_slot >= session->n_slots) {
        return ASTRAL_OK;
    }
    llama_sampler* smpl = session->grammar[session->active_slot];
    if (smpl == nullptr) {
        return ASTRAL_OK;
    }

    llama_token_data* data =
        static_cast<llama_token_data*>(ASTRAL_ALLOCA(static_cast<size_t>(count) * sizeof(llama_token_data)));

    for (uint32_t i = 0; i < count; ++i) {
        data[i].id = static_cast<llama_token>(tokens[i]);
        data[i].logit = logits[i];
        data[i].p = 0.0f;
    }

    llama_token_data_array arr{};
    arr.data = data;
    arr.size = count;
    arr.selected = -1;
    arr.sorted = false;

    llama_sampler_apply(smpl, &arr);

    const float neg_inf = -std::numeric_limits<float>::infinity();
    const size_t n = arr.size < static_cast<size_t>(count) ? arr.size : static_cast<size_t>(count);
    for (size_t i = 0; i < n; ++i) {
        tokens[i] = static_cast<uint32_t>(arr.data[i].id);
        logits[i] = arr.data[i].logit;
    }
    for (size_t i = n; i < count; ++i) {
        logits[i] = neg_inf;
    }

    return ASTRAL_OK;
}

AstralErr cpu_session_set_slot(void* session_ctx, uint32_t slot_id) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (slot_id >= session->n_slots) {
        return ASTRAL_E_INVALID;
    }
    session->active_slot = slot_id;
    return ASTRAL_OK;
}

AstralErr cpu_session_state_size(void* session_ctx, uint64_t* out_bytes) {
    if (session_ctx == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    *out_bytes = static_cast<uint64_t>(llama_state_get_size(session->ctx));
    return ASTRAL_OK;
}

AstralErr cpu_session_state_save(void* session_ctx, uint8_t* out_bytes, uint64_t capacity, uint64_t* out_written) {
    if (session_ctx == nullptr || out_bytes == nullptr || out_written == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    const size_t need = llama_state_get_size(session->ctx);
    if (capacity < need) {
        return ASTRAL_E_NOMEM;
    }
    const size_t wrote = llama_state_get_data(session->ctx, out_bytes, static_cast<size_t>(capacity));
    if (wrote == 0) {
        return ASTRAL_E_BACKEND;
    }
    *out_written = static_cast<uint64_t>(wrote);
    return ASTRAL_OK;
}

AstralErr cpu_session_state_load(void* session_ctx, const uint8_t* bytes, uint64_t size) {
    if (session_ctx == nullptr || bytes == nullptr || size == 0) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    const size_t read = llama_state_set_data(session->ctx, bytes, static_cast<size_t>(size));
    if (read != static_cast<size_t>(size)) {
        return ASTRAL_E_INVALID;
    }

    // Reconstruct per-slot token positions from the KV cache.
    llama_memory_t mem = llama_get_memory(session->ctx);
    if (mem != nullptr) {
        for (uint32_t i = 0; i < session->n_slots; ++i) {
            const llama_pos pmax = llama_memory_seq_pos_max(mem, static_cast<llama_seq_id>(i));
            session->slot_pos[i] = pmax >= 0 ? (static_cast<int32_t>(pmax) + 1) : 0;
            session->slot_has_logits[i] = false;
        }
    }

    // Logits are only defined for the last decoded sequence. Keep the previous active slot
    // and mark it "logits available" best-effort.
    if (session->active_slot < session->n_slots) {
        session->slot_has_logits[session->active_slot] = session->slot_pos[session->active_slot] > 0;
    }

    // Grammar state is not serialized in llama_state_*; reset samplers.
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        if (session->grammar[i] != nullptr) {
            llama_sampler_reset(session->grammar[i]);
        }
    }
    return ASTRAL_OK;
}

void* cpu_model_adapter_load(void* model_ctx, AstralSpanU8 path, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr || path.data == nullptr || path.len == 0) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    const size_t path_len = static_cast<size_t>(path.len);
    char* p = new (std::nothrow) char[path_len + 1];
    if (p == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    std::memcpy(p, path.data, path_len);
    p[path_len] = '\0';

    llama_adapter_lora* a = llama_adapter_lora_init(model->model, p);
    delete[] p;

    if (a == nullptr) {
        *out_err = ASTRAL_E_BACKEND;
        return nullptr;
    }
    *out_err = ASTRAL_OK;
    return a;
}

void cpu_model_adapter_unload(void* model_ctx, void* adapter_ctx) {
    (void)model_ctx;
    if (adapter_ctx == nullptr) {
        return;
    }
    llama_adapter_lora_free(static_cast<llama_adapter_lora*>(adapter_ctx));
}

AstralErr cpu_session_adapter_clear(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    llama_clear_adapter_lora(session->ctx);
    return ASTRAL_OK;
}

AstralErr cpu_session_adapter_add(void* session_ctx, void* adapter_ctx, float scale) {
    if (session_ctx == nullptr || adapter_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const int32_t rc = llama_set_adapter_lora(session->ctx, static_cast<llama_adapter_lora*>(adapter_ctx), scale);
    return rc == 0 ? ASTRAL_OK : ASTRAL_E_BACKEND;
}

void* cpu_embedder_create(void* model_ctx, AstralErr* out_err) {
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
    // NOTE: llama.cpp clears pooled-embedding caches using a hash-map each call, which
    // can allocate. Keep pooling off and do any pooling in Astral without heap use.
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

void cpu_embedder_destroy(void* embedder_ctx) {
    if (embedder_ctx == nullptr) {
        return;
    }

    CpuEmbedder* emb = static_cast<CpuEmbedder*>(embedder_ctx);
    if (emb->ctx) {
        llama_free(emb->ctx);
    }
    delete emb;
}

AstralErr cpu_embedder_reset(void* embedder_ctx) {
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

AstralErr cpu_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec, uint32_t vec_dim) {
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

    // Ensure a clean KV/memory for this request.
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

static const BackendOps kCpuBackendOps = {
    .model_load = cpu_model_load,
    .model_unload = cpu_model_unload,

    .tokenize = cpu_tokenize,
    .detokenize = cpu_detokenize,

    .model_info = cpu_model_info,
    .model_special_tokens = cpu_model_special_tokens,
    .model_embedding_dim = cpu_model_embedding_dim,

    .session_create = cpu_session_create,
    .session_destroy = cpu_session_destroy,
    .session_reset = cpu_session_reset,
    .session_feed = cpu_session_feed,
    .session_logits = cpu_session_logits,
    .session_accept = cpu_session_accept,

    .embedder_create = cpu_embedder_create,
    .embedder_destroy = cpu_embedder_destroy,
    .embedder_reset = cpu_embedder_reset,
    .embedder_embed = cpu_embedder_embed,

    .session_grammar_set_gbnf = cpu_session_grammar_set_gbnf,
    .session_grammar_set_json_schema = cpu_session_grammar_set_json_schema,
    .session_grammar_clear = cpu_session_grammar_clear,
    .session_apply_grammar = cpu_session_apply_grammar,

    .session_state_size = cpu_session_state_size,
    .session_state_save = cpu_session_state_save,
    .session_state_load = cpu_session_state_load,

    .model_adapter_load = cpu_model_adapter_load,
    .model_adapter_unload = cpu_model_adapter_unload,
    .session_adapter_clear = cpu_session_adapter_clear,
    .session_adapter_add = cpu_session_adapter_add,

    .session_set_slot = cpu_session_set_slot,
};

static const BackendProvider kCpuBackendProvider = {
    /*name=*/"cpu",
    /*ops=*/&kCpuBackendOps,
    /*supports_gpu=*/false,
    /*min_gpu_layers=*/0,
};

} // namespace

const BackendProvider* builtin_cpu_backend_provider() {
    return &kCpuBackendProvider;
}

} // namespace astral::backend
