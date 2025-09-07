#include "session.hpp"
#include "../core/work_queue.hpp"
#include "../core/runtime_state.hpp"
#include "../platform/vm.h"
#include "../platform/atomics.h"
#include "../platform/time.h"
#include <cstring>
#include <new>
#include <cstdlib>

namespace astral::inference {

void decode_loop(Session* session);

Session::Session(Model* model_,
                 void* allocator_memory_,
                 size_t allocator_capacity_,
                 const AstralSessionDesc& desc_) noexcept
    : handle(0)
    , model(model_)
    , backend_session_ctx(nullptr)
    , desc(desc_)
    , allocator(allocator_memory_, allocator_capacity_)
    , allocator_memory(allocator_memory_)
    , allocator_capacity(allocator_capacity_)
    , token_ring()
    , pending_utf8{}
    , pending_len(0)
    , pending_off(0)
    , prompt_tokens(nullptr)
    , prompt_count(0)
    , prompt_capacity(0)
    , state(SessionState::Idle)
    , cancel_requested(false)
    , final_err(ASTRAL_OK)
    , vocab_size(0)
    , ctx_size(0)
    , indices_buffer(nullptr)
    , sample_ids(nullptr)
    , sample_logits(nullptr)
    , sample_capacity(0)
    , sampler_cfg{}
    , rng_state(0)
    , token_counts(nullptr)
    , recent_tokens(nullptr)
    , recent_capacity(0)
    , recent_pos(0)
    , recent_size(0)
    , token_nl(-1)
    , total_tokens(0)
    , t_start_ticks(0)
    , t_first_token_ticks(0)
    , t_end_ticks(0)
    , n_past(0)
    , token_bos(-1)
    , token_eos(-1)
    , stop_seq_count(0)
    , stop_seq_lens{}
    , stop_seq_tokens{}
    , stop_max_len(0) {}

namespace {

inline uint64_t get_ticks() {
    return platform::ticks_now();
}

/// Default allocator capacity per session (2MB).
/// Sufficient for:
/// - Prompt tokens: ~8K tokens = 32KB
/// - Logits buffer: 32K vocab * 4 bytes = 128KB
/// - Indices buffer: 32K vocab * 4 bytes = 128KB
/// - Temp buffers: ~256KB
/// - Headroom: ~1.5MB
constexpr size_t kDefaultAllocatorCapacity = 2 * 1024 * 1024;

void decode_work(void* user) {
    decode_loop(static_cast<Session*>(user));
}

inline uint32_t clamp_repeat_last_n(int32_t v, uint32_t ctx_size) {
    if (v == 0) {
        return 0;
    }
    if (v < 0) {
        return ctx_size;
    }
    return static_cast<uint32_t>(v);
}

inline bool sampler_needs_counts(const SamplerConfig& cfg) {
    if (cfg.repeat_last_n == 0) {
        return false;
    }
    if (cfg.repeat_penalty != 1.0f) return true;
    if (cfg.presence_penalty != 0.0f) return true;
    if (cfg.frequency_penalty != 0.0f) return true;
    return false;
}

AstralErr ensure_penalty_state(Session* session) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (!sampler_needs_counts(session->sampler_cfg)) {
        return ASTRAL_OK;
    }

    const uint32_t ctx_size = session->ctx_size != 0 ? session->ctx_size : 2048;
    const uint32_t repeat_last_n = clamp_repeat_last_n(session->sampler_cfg.repeat_last_n, ctx_size);
    if (repeat_last_n == 0) {
        return ASTRAL_OK;
    }

    if (session->token_counts == nullptr) {
        session->token_counts = static_cast<uint16_t*>(
            session->allocator.alloc(session->vocab_size * sizeof(uint16_t), alignof(uint16_t))
        );
        if (session->token_counts == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        std::memset(session->token_counts, 0, session->vocab_size * sizeof(uint16_t));
    }

    if (session->recent_capacity != repeat_last_n || session->recent_tokens == nullptr) {
        session->recent_tokens = static_cast<uint32_t*>(
            session->allocator.alloc(static_cast<size_t>(repeat_last_n) * sizeof(uint32_t), alignof(uint32_t))
        );
        if (session->recent_tokens == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        session->recent_capacity = repeat_last_n;
        session->recent_pos = 0;
        session->recent_size = 0;
    }

    if (session->token_nl < 0 && session->model != nullptr && session->model->backend != nullptr &&
        session->model->backend->ops != nullptr) {
        // Best-effort: resolve a single-token "\\n".
        int32_t tok = -1;
        uint32_t out_count = 0;
        const uint8_t nl = static_cast<uint8_t>('\n');
        AstralSpanU8 text{};
        text.data = &nl;
        text.len = 1;
        const AstralErr err = session->model->backend->ops->tokenize(
            session->model->backend_model_ctx, text, &tok, 1, 0, 0, &out_count
        );
        if (err == ASTRAL_OK && out_count == 1) {
            session->token_nl = tok;
        } else {
            session->token_nl = -1;
        }
    }

    return ASTRAL_OK;
}

inline void penalty_state_clear(Session* session) {
    if (session == nullptr || session->token_counts == nullptr) {
        return;
    }
    std::memset(session->token_counts, 0, session->vocab_size * sizeof(uint16_t));
    session->recent_pos = 0;
    session->recent_size = 0;
}

inline void penalty_state_push(Session* session, uint32_t token_id) {
    if (session == nullptr || session->token_counts == nullptr || session->recent_tokens == nullptr ||
        session->recent_capacity == 0) {
        return;
    }
    if (token_id >= session->vocab_size) {
        return;
    }

    if (session->recent_size == session->recent_capacity) {
        const uint32_t old = session->recent_tokens[session->recent_pos];
        if (old < session->vocab_size) {
            uint16_t& c = session->token_counts[old];
            if (c > 0) {
                --c;
            }
        }
    } else {
        ++session->recent_size;
    }

    session->recent_tokens[session->recent_pos] = token_id;
    session->recent_pos = (session->recent_pos + 1u) % session->recent_capacity;

    uint16_t& c = session->token_counts[token_id];
    if (c != 0xFFFFu) {
        ++c;
    }
}

inline void* session_alloc_mem() {
#if defined(__cpp_aligned_new)
    return ::operator new(sizeof(Session), std::align_val_t(alignof(Session)), std::nothrow);
#else
    void* p = nullptr;
    if (::posix_memalign(&p, alignof(Session), sizeof(Session)) != 0) {
        return nullptr;
    }
    return p;
#endif
}

inline void session_free_mem(void* p) noexcept {
    if (p == nullptr) {
        return;
    }
#if defined(__cpp_aligned_new)
    ::operator delete(p, std::align_val_t(alignof(Session)));
#else
    std::free(p);
#endif
}

} // namespace

AstralErr session_create(const AstralSessionDesc* desc, Session** out_session) {
    // Validate parameters
    if (desc == nullptr || out_session == nullptr) {
        return ASTRAL_E_INVALID;
    }

    // Validate model handle
    Model* model = static_cast<Model*>(core::lookup_handle(desc->model, core::HandleKind::Model));
    if (model == nullptr) {
        return ASTRAL_E_INVALID;
    }
    model->refcount.fetch_add(1, std::memory_order_relaxed);

    // Allocate per-session memory (reserve + commit)
    // This will be used by FrameAllocator
    constexpr size_t allocator_capacity = kDefaultAllocatorCapacity;
    void* allocator_memory = platform::vm_reserve(allocator_capacity);
    if (allocator_memory == nullptr) {
        return ASTRAL_E_NOMEM;
    }

    // Pre-commit all memory ( avoid page faults in hot path)
    platform::vm_commit(allocator_memory, allocator_capacity);
    if (::astral::core::runtime_hugepages_enabled()) {
        // Best-effort: ignore failure.
        platform::vm_try_hugepages(allocator_memory, allocator_capacity);
    }

    // Allocate session struct on heap
    // Use malloc instead of new to avoid default constructor issues
    void* session_mem = session_alloc_mem();
    if (session_mem == nullptr) {
        platform::vm_release(allocator_memory, allocator_capacity);
        return ASTRAL_E_NOMEM;
    }

    // Construct session object (starts lifetime; avoids UB in release builds).
    Session* session = new (session_mem) Session(model, allocator_memory, allocator_capacity, *desc);

    // Allocate prompt buffer (initial capacity: 8K tokens)
    session->prompt_capacity = 8192;
    session->prompt_tokens = static_cast<int32_t*>(
        session->allocator.alloc(session->prompt_capacity * sizeof(int32_t), alignof(int32_t))
    );
    session->prompt_count = 0;

    if (session->prompt_tokens == nullptr) {
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        session->~Session();
        session_free_mem(session_mem);
        return ASTRAL_E_NOMEM;
    }

    // Get model info for allocating buffers
    uint32_t vocab_size = 0;
    uint32_t ctx_size = 0;
    AstralErr info_err =
        model->backend->ops->model_info(model->backend_model_ctx, &vocab_size, &ctx_size);
    if (info_err != ASTRAL_OK || vocab_size == 0) {
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        session->~Session();
        session_free_mem(session_mem);
        return info_err != ASTRAL_OK ? info_err : ASTRAL_E_BACKEND;
    }

    // Create backend session context (KV cache + sampler).
    AstralErr backend_err = ASTRAL_OK;
    void* backend_session_ctx =
        model->backend->ops->session_create(model->backend_model_ctx, desc, &backend_err);
    if (backend_session_ctx == nullptr) {
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        session->~Session();
        session_free_mem(session_mem);
        return backend_err != ASTRAL_OK ? backend_err : ASTRAL_E_BACKEND;
    }
    session->backend_session_ctx = backend_session_ctx;

    session->vocab_size = vocab_size;
    session->ctx_size = ctx_size;

    // Cache special tokens (used for stop conditions).
    int32_t token_bos = -1;
    int32_t token_eos = -1;
    if (model->backend->ops->model_special_tokens) {
        const AstralErr tok_err =
            model->backend->ops->model_special_tokens(model->backend_model_ctx, &token_bos, &token_eos);
        if (tok_err != ASTRAL_OK) {
            model->backend->ops->session_destroy(session->backend_session_ctx);
            session->backend_session_ctx = nullptr;
            model_release(model);
            platform::vm_release(allocator_memory, allocator_capacity);
            session->~Session();
            session_free_mem(session_mem);
            return tok_err;
        }
    }
    session->token_bos = token_bos;
    session->token_eos = token_eos;

    // Allocate sampling scratch buffers.
    // indices_buffer is used for full-vocab top-p fallback (rare; only when top_k == 0 and top_p < 1).
    session->indices_buffer = static_cast<uint32_t*>(
        session->allocator.alloc(vocab_size * sizeof(uint32_t), alignof(uint32_t))
    );

    // Candidate buffers for fast top-k/top-p within a bounded set.
    uint32_t sample_capacity = 0;
    if (desc->top_k > 0) {
        sample_capacity = desc->top_k;
    } else if (desc->top_p > 0.0f && desc->top_p < 1.0f) {
        // For top-p without explicit top-k, keep a reasonable candidate window for fast path.
        // If this is insufficient to reach p, sampling falls back to full-vocab sorting via indices_buffer.
        sample_capacity = 2048;
    }

    if (sample_capacity > vocab_size) {
        sample_capacity = vocab_size;
    }

    session->sample_capacity = sample_capacity;
    session->sample_ids = sample_capacity > 0 ? static_cast<uint32_t*>(
                              session->allocator.alloc(sample_capacity * sizeof(uint32_t), alignof(uint32_t)))
                                              : nullptr;
    session->sample_logits = sample_capacity > 0 ? static_cast<float*>(
                                 session->allocator.alloc(sample_capacity * sizeof(float), alignof(float)))
                                                 : nullptr;

    // Penalty state (allocated lazily when needed; never in steady-state decode).
    session->token_counts = nullptr;
    session->recent_tokens = nullptr;
    session->recent_capacity = 0;
    session->recent_pos = 0;
    session->recent_size = 0;
    session->token_nl = -1;

    // Stop sequences (none by default).
    session->stop_seq_count = 0;
    std::memset(session->stop_seq_lens, 0, sizeof(session->stop_seq_lens));
    std::memset(session->stop_seq_tokens, 0, sizeof(session->stop_seq_tokens));
    session->stop_max_len = 0;

    if (session->indices_buffer == nullptr || (sample_capacity > 0 && (session->sample_ids == nullptr ||
                                                                      session->sample_logits == nullptr))) {
        if (session->backend_session_ctx != nullptr && model->backend != nullptr) {
            model->backend->ops->session_destroy(session->backend_session_ctx);
            session->backend_session_ctx = nullptr;
        }
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        session->~Session();
        session_free_mem(session_mem);
        return ASTRAL_E_NOMEM;
    }

    // Initialize sampler config
    session->sampler_cfg.temperature = desc->temperature;
    session->sampler_cfg.top_k = desc->top_k;
    session->sampler_cfg.top_p = desc->top_p;
    session->sampler_cfg.min_p = 0.0f;
    session->sampler_cfg.typical_p = 1.0f;
    session->sampler_cfg.repeat_penalty = 1.0f;
    session->sampler_cfg.repeat_last_n = 0;
    session->sampler_cfg.penalize_nl = 0;
    session->sampler_cfg.presence_penalty = 0.0f;
    session->sampler_cfg.frequency_penalty = 0.0f;
    session->sampler_cfg.seed =
        desc->seed != 0 ? desc->seed : static_cast<uint32_t>(get_ticks() & 0xFFFFFFFF);
    session->rng_state = session->sampler_cfg.seed;

    const AstralHandle handle = core::register_handle(core::HandleKind::Session, session);
    if (handle == 0) {
        model->backend->ops->session_destroy(session->backend_session_ctx);
        session->backend_session_ctx = nullptr;
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        session->~Session();
        session_free_mem(session_mem);
        return ASTRAL_E_BUSY;
    }
    session->handle = handle;

    *out_session = session;
    return ASTRAL_OK;
}

void session_destroy(Session* session) {
    if (session == nullptr) {
        return;
    }

    // Invalidate handle early so concurrent callers fail fast.
    core::unregister_handle(session->handle, core::HandleKind::Session);
    session->handle = 0;

    // Wait for decode to complete (if in progress)
    // NOTE: v0.1 uses a simple wait loop with WFE/SEV style hints.
    SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        // Request cancellation
        session->cancel_requested.store(true, std::memory_order_release);

        uint32_t spins = 0;
        while (true) {
            state = session->state.load(std::memory_order_acquire);
            if (state != SessionState::Decoding) {
                break;
            }
            if (spins < 64) {
                platform::cpu_pause();
            } else {
                platform::cpu_wait_for_event();
            }
            if (spins < 1024) {
                ++spins;
            }
        }
    }

    // Release allocator memory
    if (session->allocator_memory != nullptr) {
        platform::vm_release(session->allocator_memory, session->allocator_capacity);
    }

    // Destroy backend session context
    if (session->backend_session_ctx != nullptr && session->model != nullptr && session->model->backend != nullptr) {
        session->model->backend->ops->session_destroy(session->backend_session_ctx);
        session->backend_session_ctx = nullptr;
    }

    // Release model reference (session holds one ref).
    model_release(session->model);
    session->model = nullptr;

    // Destroy and free session memory.
    session->~Session();
    session_free_mem(session);
}

AstralErr session_feed(Session* session, AstralSpanU8 prompt_chunk, uint8_t finalize) {
    // Validate session
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }

    // Check state (must be Idle or FeedingPrompt)
    SessionState state = session->state.load(std::memory_order_acquire);

    if (state != SessionState::Idle && state != SessionState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    // Transition to FeedingPrompt on first feed
    if (state == SessionState::Idle) {
        session->state.store(SessionState::FeedingPrompt, std::memory_order_release);
    }

    // Tokenize prompt chunk.
    // IMPORTANT: only add special tokens (e.g., BOS) on the first chunk.
    const bool should_tokenize = (prompt_chunk.len > 0 && prompt_chunk.data != nullptr) ||
                                 (finalize != 0 && session->prompt_count == 0);

    if (should_tokenize) {
        uint32_t space_available = session->prompt_capacity - session->prompt_count;
        uint32_t n_tokens = 0;

        AstralErr err = session->model->backend->ops->tokenize(
            session->model->backend_model_ctx,
            prompt_chunk,
            session->prompt_tokens + session->prompt_count,
            space_available,
            /*add_special=*/session->prompt_count == 0,
            /*parse_special=*/false,
            &n_tokens
        );

        if (err != ASTRAL_OK) {
            // Tokenization failed
            return err;
        }

        if (n_tokens == 0) {
            // No tokens produced (empty chunk)
            return ASTRAL_OK;
        }

        session->prompt_count += n_tokens;

        // Check if buffer is full
        if (session->prompt_count >= session->prompt_capacity) {
            return ASTRAL_E_NOMEM;
        }
    }

    // If finalized, stay in FeedingPrompt state (waiting for decode)
    // Caller must call session_decode() next

    return ASTRAL_OK;
}

AstralErr session_decode(Session* session) {
    // Validate session
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }

    // Check state (must be FeedingPrompt)
    SessionState state = session->state.load(std::memory_order_acquire);
    if (state != SessionState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    // Transition to Decoding
    session->state.store(SessionState::Decoding, std::memory_order_release);

    // Record start time
    session->t_start_ticks = get_ticks();
    session->t_end_ticks = 0;

    // Submit decode work to runtime worker pool.
    const AstralErr submit_err = ::astral::core::submit_work(decode_work, session);
    if (submit_err != ASTRAL_OK) {
        session->state.store(SessionState::FeedingPrompt, std::memory_order_release);
        return submit_err;
    }

    return ASTRAL_OK; // async
}

AstralErr session_cancel(Session* session) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }

    session->cancel_requested.store(true, std::memory_order_release);
    platform::cpu_signal_event();
    return ASTRAL_OK;
}

AstralErr session_state(Session* session, SessionState* out_state) {
    if (session == nullptr || out_state == nullptr) {
        return ASTRAL_E_INVALID;
    }

    *out_state = session->state.load(std::memory_order_acquire);
    return ASTRAL_OK;
}

AstralErr session_wait(Session* session, uint32_t timeout_ms) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }

    SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Completed) return ASTRAL_OK;
    if (state == SessionState::Canceled) return ASTRAL_E_CANCELED;
    if (state == SessionState::Failed) {
        const int32_t final_err = session->final_err.load(std::memory_order_acquire);
        return final_err != 0 ? static_cast<AstralErr>(final_err) : ASTRAL_E_BACKEND;
    }

    if (timeout_ms == 0) {
        return ASTRAL_E_TIMEOUT;
    }

    const uint64_t start_ticks = get_ticks();
    const uint64_t timeout_ns = static_cast<uint64_t>(timeout_ms) * 1000000ULL;
    const uint64_t timeout_ticks = platform::ticks_from_ns(timeout_ns);
    const uint64_t deadline_ticks = start_ticks + timeout_ticks;
    const uint32_t check_mask =
        timeout_ms <= 1u   ? 31u :
        timeout_ms <= 10u  ? 255u :
        timeout_ms <= 100u ? 4095u :
                             65535u;
    uint32_t spins = 0;

    while (true) {
        state = session->state.load(std::memory_order_acquire);
        if (state == SessionState::Completed) return ASTRAL_OK;
        if (state == SessionState::Canceled) return ASTRAL_E_CANCELED;
        if (state == SessionState::Failed) {
            const int32_t final_err = session->final_err.load(std::memory_order_acquire);
            return final_err != 0 ? static_cast<AstralErr>(final_err) : ASTRAL_E_BACKEND;
        }

        // Check timeout (tick source is platform-dependent; on x86 we use TSC, avoiding syscalls).
        if ((spins & check_mask) == 0u) {
            const uint64_t now_ticks = get_ticks();
            if (now_ticks >= deadline_ticks) {
                return ASTRAL_E_TIMEOUT;
            }
        }

        if (spins < 64) {
            platform::cpu_pause();
        } else {
            platform::cpu_wait_for_event();
        }
        if (spins < 1024) {
            ++spins;
        }
    }
}

AstralErr session_reset(Session* session, const AstralSessionDesc* new_desc) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }

    // Must not reset while decoding. Caller should cancel + wait first.
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }

    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }

    const backend::BackendOps* ops = session->model->backend->ops;

    // Update session parameters (optional).
    if (new_desc != nullptr) {
        if (new_desc->model == 0 || new_desc->model != session->model->handle) {
            return ASTRAL_E_INVALID;
        }

        // Reset/reuse cannot grow scratch buffers; require top_k <= existing capacity when enabled.
        if (new_desc->top_k > 0 && new_desc->top_k > session->sample_capacity) {
            return ASTRAL_E_INVALID;
        }

        session->desc = *new_desc;
    }

    // Re-seed RNG. If seed==0, auto-seed each reset.
    session->sampler_cfg.temperature = session->desc.temperature;
    session->sampler_cfg.top_k = session->desc.top_k;
    session->sampler_cfg.top_p = session->desc.top_p;
    session->sampler_cfg.seed =
        session->desc.seed != 0 ? session->desc.seed : static_cast<uint32_t>(get_ticks() & 0xFFFFFFFF);
    session->rng_state = session->sampler_cfg.seed;

    // Clear core state (not thread-safe; caller must ensure no concurrent stream reads).
    session->cancel_requested.store(false, std::memory_order_release);
    session->final_err.store(ASTRAL_OK, std::memory_order_release);
    session->prompt_count = 0;
    session->token_ring.reset();
    session->pending_len = 0;
    session->pending_off = 0;

    session->total_tokens = 0;
    session->t_start_ticks = 0;
    session->t_first_token_ticks = 0;
    session->t_end_ticks = 0;
    session->n_past = 0;

    penalty_state_clear(session);

    // Reset backend session state (KV/cache). Fall back to destroy+create if unsupported.
    if (session->backend_session_ctx == nullptr) {
        return ASTRAL_E_STATE;
    }

    AstralErr err = ASTRAL_E_UNSUPPORTED;
    if (ops->session_reset != nullptr) {
        err = ops->session_reset(session->backend_session_ctx);
    }

    if (err == ASTRAL_E_UNSUPPORTED) {
        ops->session_destroy(session->backend_session_ctx);
        session->backend_session_ctx = nullptr;

        AstralErr create_err = ASTRAL_OK;
        session->backend_session_ctx =
            ops->session_create(session->model->backend_model_ctx, &session->desc, &create_err);
        if (session->backend_session_ctx == nullptr) {
            return create_err != ASTRAL_OK ? create_err : ASTRAL_E_BACKEND;
        }
    } else if (err != ASTRAL_OK) {
        return err;
    }

    // Back to Idle for the next prompt.
    session->state.store(SessionState::Idle, std::memory_order_release);
    platform::cpu_signal_event();
    return ASTRAL_OK;
}

AstralErr session_set_sampler(Session* session, const AstralSamplerDesc* desc) {
    if (session == nullptr || desc == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralSamplerDesc)) {
        return ASTRAL_E_INVALID;
    }

    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }

    // Keep scratch capacity invariant.
    if (desc->top_k > 0 && desc->top_k > session->sample_capacity) {
        return ASTRAL_E_INVALID;
    }

    session->sampler_cfg.temperature = desc->temperature;
    session->sampler_cfg.top_k = desc->top_k;
    session->sampler_cfg.top_p = desc->top_p;
    session->sampler_cfg.min_p = desc->min_p;
    session->sampler_cfg.typical_p = desc->typical_p;
    session->sampler_cfg.repeat_penalty = desc->repeat_penalty;
    session->sampler_cfg.repeat_last_n = desc->repeat_last_n;
    session->sampler_cfg.penalize_nl = desc->penalize_nl;
    session->sampler_cfg.presence_penalty = desc->presence_penalty;
    session->sampler_cfg.frequency_penalty = desc->frequency_penalty;

    // mirostat is not implemented in v0.1; ignore for now (kept in ABI for forward compat).
    (void)desc->mirostat;
    (void)desc->mirostat_tau;
    (void)desc->mirostat_eta;

    return ensure_penalty_state(session);
}

AstralErr session_stop_clear(Session* session) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }

    session->stop_seq_count = 0;
    std::memset(session->stop_seq_lens, 0, sizeof(session->stop_seq_lens));
    std::memset(session->stop_seq_tokens, 0, sizeof(session->stop_seq_tokens));
    session->stop_max_len = 0;
    return ASTRAL_OK;
}

AstralErr session_stop_add_utf8(Session* session, AstralSpanU8 utf8) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (utf8.data == nullptr || utf8.len == 0) {
        return ASTRAL_E_INVALID;
    }
    if (session->stop_seq_count >= 16) {
        return ASTRAL_E_NOMEM;
    }

    int32_t tokens[32];
    uint32_t out_count = 0;
    const AstralErr err = session->model->backend->ops->tokenize(
        session->model->backend_model_ctx, utf8, tokens, 32, 0, 0, &out_count
    );
    if (err != ASTRAL_OK) {
        return err;
    }
    if (out_count == 0 || out_count > 32) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t idx = session->stop_seq_count++;
    session->stop_seq_lens[idx] = static_cast<uint8_t>(out_count);
    for (uint32_t i = 0; i < out_count; ++i) {
        session->stop_seq_tokens[idx][i] = tokens[i];
    }
    if (out_count > session->stop_max_len) {
        session->stop_max_len = out_count;
    }
    return ASTRAL_OK;
}

AstralErr session_stats(Session* session, AstralStats* out_stats) {
    if (session == nullptr || out_stats == nullptr) {
        return ASTRAL_E_INVALID;
    }

    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state != SessionState::Completed && state != SessionState::Canceled && state != SessionState::Failed) {
        return ASTRAL_E_STATE;
    }

    const uint64_t t_start_ticks = session->t_start_ticks;
    if (t_start_ticks == 0) {
        return ASTRAL_E_STATE;
    }

    const uint64_t end_ticks = session->t_end_ticks != 0 ? session->t_end_ticks : get_ticks();

    const uint64_t elapsed_ticks = end_ticks > t_start_ticks ? (end_ticks - t_start_ticks) : 0;
    const uint64_t elapsed_ns = platform::ticks_to_ns(elapsed_ticks);
    const double elapsed_s = elapsed_ns > 0 ? static_cast<double>(elapsed_ns) / 1e9 : 0.0;

    double ttft_ms = 0.0;
    if (session->t_first_token_ticks > 0 && session->t_first_token_ticks >= t_start_ticks) {
        const uint64_t ttft_ticks = session->t_first_token_ticks - t_start_ticks;
        ttft_ms = static_cast<double>(platform::ticks_to_ns(ttft_ticks)) / 1e6;
    }

    const uint64_t total_tokens = session->total_tokens;
    const double tok_per_s = (elapsed_s > 0.0 && total_tokens > 0) ? (static_cast<double>(total_tokens) / elapsed_s)
                                                                   : 0.0;

    out_stats->t_init_ms = 0.0;
    out_stats->t_first_token_ms = ttft_ms;
    out_stats->tok_per_s = tok_per_s;

    uint64_t bytes_committed = 0;
    uint64_t bytes_reserved = 0;
    core::runtime_memory_stats(&bytes_committed, &bytes_reserved);
    out_stats->bytes_committed = bytes_committed;
    out_stats->bytes_reserved = bytes_reserved;

    return ASTRAL_OK;
}

void decode_loop(Session* session) {
    if (session == nullptr) {
        return;
    }

    Model* model = session->model;
    const backend::BackendProvider* backend = model->backend;
    const backend::BackendOps* ops = backend->ops;
    void* backend_session_ctx = session->backend_session_ctx;

    uint32_t max_tokens = session->desc.max_tokens;

    SessionState final_state = SessionState::Completed;
    AstralErr final_err = ASTRAL_OK;

    // Feed prompt tokens
    AstralErr err = ops->session_feed(
        backend_session_ctx,
        session->prompt_tokens,
        session->prompt_count
    );

    if (err != ASTRAL_OK) {
        // Feed failed; transition to Failed
        final_state = SessionState::Failed;
        final_err = err;
    } else {
        // Initialize penalty state from prompt once (not a hot path).
        if (ensure_penalty_state(session) == ASTRAL_OK) {
            penalty_state_clear(session);
            for (uint32_t i = 0; i < session->prompt_count; ++i) {
                const int32_t t = session->prompt_tokens[i];
                if (t >= 0) {
                    penalty_state_push(session, static_cast<uint32_t>(t));
                }
            }
        }

        // Stop tail (token ids only; small fixed buffer).
        int32_t stop_tail[32];
        uint32_t stop_tail_len = 0;

        // Streaming hold buffer for stop suppression (enabled only if stop sequences are configured).
        constexpr uint32_t kHoldCap = 64;
        concurrency::StreamToken hold[kHoldCap];
        uint32_t hold_head = 0;
        uint32_t hold_count = 0;
        const uint32_t stop_hold =
            (session->stop_max_len > 0 && session->stop_max_len <= 32) ? session->stop_max_len : 0;

        auto flush_one = [&](const concurrency::StreamToken& tok) -> bool {
            bool pushed = false;
            uint32_t spins = 0;
            while (!pushed) {
                if (session->token_ring.push(tok)) {
                    pushed = true;
                    break;
                }

                if (spins < 64) {
                    platform::cpu_pause();
                } else {
                    platform::cpu_wait_for_event();
                }
                if (spins < 1024) {
                    ++spins;
                }

                if (session->cancel_requested.load(std::memory_order_acquire)) {
                    return false;
                }
            }
            return true;
        };

    auto hold_push = [&](const concurrency::StreamToken& tok) -> bool {
        if (stop_hold == 0) {
            return flush_one(tok);
        }

        // Append at tail.
        const uint32_t tail = (hold_head + hold_count) % kHoldCap;
        hold[tail] = tok;
        if (hold_count < kHoldCap) {
            ++hold_count;
        } else {
            // Should not happen given stop_hold <= 32, but keep safe behavior.
            if (!flush_one(hold[hold_head])) {
                return false;
            }
            hold_head = (hold_head + 1) % kHoldCap;
        }

        // Keep at most stop_hold tokens buffered (flush oldest).
        while (hold_count > stop_hold) {
            if (!flush_one(hold[hold_head])) {
                return false;
            }
            hold_head = (hold_head + 1) % kHoldCap;
            --hold_count;
        }
        return true;
    };

    auto hold_flush_all = [&]() -> bool {
        while (hold_count > 0) {
            if (!flush_one(hold[hold_head])) {
                return false;
            }
            hold_head = (hold_head + 1) % kHoldCap;
            --hold_count;
        }
        return true;
    };

    auto stop_check = [&](int32_t token, uint32_t* out_match_len) -> bool {
        if (session->stop_seq_count == 0 || token < 0) {
            return false;
        }

        if (stop_tail_len < 32) {
            stop_tail[stop_tail_len++] = token;
        } else {
            std::memmove(stop_tail, stop_tail + 1, (32 - 1) * sizeof(int32_t));
            stop_tail[31] = token;
        }

        for (uint32_t i = 0; i < session->stop_seq_count; ++i) {
            const uint32_t len = session->stop_seq_lens[i];
            if (len == 0 || len > stop_tail_len) {
                continue;
            }
            bool match = true;
            for (uint32_t j = 0; j < len; ++j) {
                if (stop_tail[stop_tail_len - len + j] != session->stop_seq_tokens[i][j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                if (out_match_len) {
                    *out_match_len = len;
                }
                return true;
            }
        }
        return false;
    };

    // Generation loop
    for (uint32_t i = 0; i < max_tokens; ++i) {
        // Check for cancellation
        if (session->cancel_requested.load(std::memory_order_acquire)) {
            final_state = SessionState::Canceled;
            final_err = ASTRAL_E_CANCELED;
            break;
        }

        backend::BackendLogitsView logits_view{};
        err = ops->session_logits(backend_session_ctx, &logits_view);

        if (err != ASTRAL_OK) {
            final_state = SessionState::Failed;
            final_err = err;
            break;
        }

        const int32_t next_token = static_cast<int32_t>(sample_token(
            logits_view.logits,
            logits_view.vocab_size,
            session->sampler_cfg,
            &session->rng_state,
            session->token_counts,
            session->token_nl,
            session->sample_ids,
            session->sample_logits,
            session->sample_capacity,
            session->indices_buffer
        ));

        err = ops->session_accept(backend_session_ctx, next_token);
        if (err != ASTRAL_OK) {
            final_state = SessionState::Failed;
            final_err = err;
            break;
        }

        if (next_token >= 0) {
            penalty_state_push(session, static_cast<uint32_t>(next_token));
        }

        if (session->token_eos >= 0 && next_token == session->token_eos) {
            break;
        }

        // Update statistics (for both streaming and non-streaming sessions).
        session->total_tokens++;
        if (session->total_tokens == 1) {
            session->t_first_token_ticks = get_ticks();
        }

        uint32_t stop_len = 0;
        const bool stop_hit = stop_check(next_token, &stop_len);

        if (session->desc.stream_enabled) {
            // Detokenize to UTF-8
            concurrency::StreamToken stream_token;
            stream_token.token_id = static_cast<uint32_t>(next_token);

            AstralMutSpanU8 out_buf;
            out_buf.data = stream_token.utf8_data;
            out_buf.len = sizeof(stream_token.utf8_data);

            uint32_t utf8_len = 0;
            err = ops->detokenize(
                model->backend_model_ctx,
                &next_token,
                1,
                out_buf,
                &utf8_len
            );

            if (err != ASTRAL_OK) {
                // Detokenization failed
                final_state = SessionState::Failed;
                final_err = err;
                break;
            }

            stream_token.utf8_len = static_cast<uint16_t>(utf8_len);

            if (!hold_push(stream_token)) {
                final_state = SessionState::Canceled;
                final_err = ASTRAL_E_CANCELED;
                break;
            }
        }

        if (stop_hit) {
            if (session->desc.stream_enabled && stop_hold > 0 && stop_len > 0) {
                // Drop stop tokens from the end of the hold buffer.
                if (stop_len >= hold_count) {
                    hold_count = 0;
                } else {
                    hold_count -= stop_len;
                }
                if (!hold_flush_all()) {
                    final_state = SessionState::Canceled;
                    final_err = ASTRAL_E_CANCELED;
                }
            }
            break;
        }
    }

    // Flush any remaining buffered tokens (normal completion/EOS/max_tokens).
    if (final_state == SessionState::Completed && session->desc.stream_enabled && hold_count > 0) {
        if (!hold_flush_all()) {
            final_state = SessionState::Canceled;
            final_err = ASTRAL_E_CANCELED;
        }
    }
    } // feed_ok

    // Transition to terminal state.
    session->t_end_ticks = get_ticks();
    session->final_err.store(static_cast<int32_t>(final_err), std::memory_order_release);
    session->state.store(final_state, std::memory_order_release);
    platform::cpu_signal_event();
}

int32_t stream_read(Session* session, AstralMutSpanU8 out_buf, uint32_t timeout_ms) {
    // Validate session
    if (session == nullptr || out_buf.data == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (out_buf.len == 0) {
        return ASTRAL_E_INVALID;
    }

    // Consume any remainder from a previous partial-token read.
    if (session->pending_len > session->pending_off) {
        const uint32_t avail = static_cast<uint32_t>(session->pending_len - session->pending_off);
        const uint32_t n = (avail > out_buf.len) ? out_buf.len : avail;
        std::memcpy(out_buf.data, session->pending_utf8 + session->pending_off, n);
        session->pending_off = static_cast<uint8_t>(session->pending_off + n);
        if (session->pending_off >= session->pending_len) {
            session->pending_len = 0;
            session->pending_off = 0;
        }
        return static_cast<int32_t>(n);
    }

    // Try to read from token ring (fast path).
    concurrency::StreamToken token{};
    if (session->token_ring.pop(&token)) {
        uint8_t* dst = out_buf.data;
        uint32_t remaining = out_buf.len;
        uint32_t total = 0;

        for (;;) {
            const uint32_t tok_len = token.utf8_len;
            if (tok_len == 0) {
                // Skip empty tokens (e.g. BOS/EOS detokenize to empty). Returning 0 is reserved for EOF.
            } else if (tok_len <= remaining) {
                std::memcpy(dst, token.utf8_data, tok_len);
                dst += tok_len;
                remaining -= tok_len;
                total += tok_len;
            } else {
                // Partial token: copy what we can and store the remainder.
                std::memcpy(dst, token.utf8_data, remaining);
                std::memcpy(session->pending_utf8, token.utf8_data, tok_len);
                session->pending_len = static_cast<uint8_t>(tok_len);
                session->pending_off = static_cast<uint8_t>(remaining);
                total += remaining;
                return static_cast<int32_t>(total);
            }

            if (remaining == 0) {
                break;
            }
            if (!session->token_ring.pop(&token)) {
                break;
            }
        }

        if (total > 0) {
            return static_cast<int32_t>(total);
        }
        // Only empty tokens were available; fall through to the regular empty-ring behavior.
    }

    // Ring is empty; check if decoding is complete
    SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Completed || state == SessionState::Canceled || state == SessionState::Failed) {
        // No more data will be produced
        return 0;
    }

    // Wait for data (if timeout > 0)
    if (timeout_ms > 0) {
        const uint64_t start_ticks = get_ticks();
        const uint64_t timeout_ns = static_cast<uint64_t>(timeout_ms) * 1000000ULL;
        const uint64_t timeout_ticks = platform::ticks_from_ns(timeout_ns);
        const uint64_t deadline_ticks = start_ticks + timeout_ticks;
        const uint32_t check_mask =
            timeout_ms <= 1u   ? 31u :
            timeout_ms <= 10u  ? 255u :
            timeout_ms <= 100u ? 4095u :
                                 65535u;
        uint32_t spins = 0;

        while (true) {
            // Try to pop again
            if (session->token_ring.pop(&token)) {
                uint8_t* dst = out_buf.data;
                uint32_t remaining = out_buf.len;
                uint32_t total = 0;

                for (;;) {
                    const uint32_t tok_len = token.utf8_len;
                    if (tok_len == 0) {
                        // Skip empty tokens (e.g. BOS/EOS detokenize to empty). Returning 0 is reserved for EOF.
                    } else if (tok_len <= remaining) {
                        std::memcpy(dst, token.utf8_data, tok_len);
                        dst += tok_len;
                        remaining -= tok_len;
                        total += tok_len;
                    } else {
                        std::memcpy(dst, token.utf8_data, remaining);
                        std::memcpy(session->pending_utf8, token.utf8_data, tok_len);
                        session->pending_len = static_cast<uint8_t>(tok_len);
                        session->pending_off = static_cast<uint8_t>(remaining);
                        total += remaining;
                        return static_cast<int32_t>(total);
                    }

                    if (remaining == 0) {
                        break;
                    }
                    if (!session->token_ring.pop(&token)) {
                        break;
                    }
                }

                if (total > 0) {
                    return static_cast<int32_t>(total);
                }
                // Only empty tokens were available; continue waiting.
                continue;
            }

            // Check if completed
            state = session->state.load(std::memory_order_acquire);
            if (state == SessionState::Completed || state == SessionState::Canceled || state == SessionState::Failed) {
                return 0;
            }

            // Check timeout (tick source is platform-dependent; on x86 we use TSC, avoiding syscalls).
            if ((spins & check_mask) == 0u) {
                const uint64_t now_ticks = get_ticks();
                if (now_ticks >= deadline_ticks) {
                    return ASTRAL_E_TIMEOUT;
                }
            }

            // Wait for producer to push (SPSC ring signals on empty->non-empty).
            if (spins < 64) {
                platform::cpu_pause();
            } else {
                platform::cpu_wait_for_event();
                // On non-ARM platforms `cpu_wait_for_event()` is a light hint (not a true event wait).
                // Keep this path syscall-free: avoid `sched_yield()` here and rely on pause/backoff.
            }
            if (spins < 1024) {
                ++spins;
            }
        }
    }

    // Non-blocking and no data available
    return ASTRAL_E_TIMEOUT;
}

} // namespace astral::inference
