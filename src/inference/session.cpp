#include "session.hpp"
#include "../core/work_queue.hpp"
#include "../core/runtime_state.hpp"
#include "../platform/vm.h"
#include "../platform/atomics.h"
#include <cstring>
#include <new>
#include <chrono>

namespace astral::inference {

void decode_loop(Session* session);

namespace {

/// Get current time in nanoseconds (monotonic clock).
uint64_t get_time_ns() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
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

    // Allocate session struct on heap
    // Use malloc instead of new to avoid default constructor issues
    void* session_mem = ::operator new(sizeof(Session), std::nothrow);
    if (session_mem == nullptr) {
        platform::vm_release(allocator_memory, allocator_capacity);
        return ASTRAL_E_NOMEM;
    }

    // Construct Session using placement new
    // Initialize all members explicitly
    Session* session = static_cast<Session*>(session_mem);

    // Store model reference and config
    session->handle = 0;
    session->model = model;
    session->backend_session_ctx = nullptr;
    session->desc = *desc;

    // Store allocator memory info
    session->allocator_memory = allocator_memory;
    session->allocator_capacity = allocator_capacity;

    // Initialize frame allocator using placement new
    new (&session->allocator) memory::FrameAllocator(
        allocator_memory,
        allocator_capacity
    );

    // Initialize token ring using placement new
    new (&session->token_ring) concurrency::SpscRing<concurrency::StreamToken, 256>();

    // Allocate prompt buffer (initial capacity: 8K tokens)
    session->prompt_capacity = 8192;
    session->prompt_tokens = static_cast<int32_t*>(
        session->allocator.alloc(session->prompt_capacity * sizeof(int32_t), alignof(int32_t))
    );
    session->prompt_count = 0;

    if (session->prompt_tokens == nullptr) {
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        ::operator delete(session_mem);
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
        ::operator delete(session_mem);
        return info_err != ASTRAL_OK ? info_err : ASTRAL_E_BACKEND;
    }

    // Create backend session context (KV cache + sampler).
    AstralErr backend_err = ASTRAL_OK;
    void* backend_session_ctx =
        model->backend->ops->session_create(model->backend_model_ctx, desc, &backend_err);
    if (backend_session_ctx == nullptr) {
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        ::operator delete(session_mem);
        return backend_err != ASTRAL_OK ? backend_err : ASTRAL_E_BACKEND;
    }
    session->backend_session_ctx = backend_session_ctx;

    session->vocab_size = vocab_size;

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
            ::operator delete(session_mem);
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

    if (session->indices_buffer == nullptr || (sample_capacity > 0 && (session->sample_ids == nullptr ||
                                                                      session->sample_logits == nullptr))) {
        if (session->backend_session_ctx != nullptr && model->backend != nullptr) {
            model->backend->ops->session_destroy(session->backend_session_ctx);
            session->backend_session_ctx = nullptr;
        }
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        ::operator delete(session_mem);
        return ASTRAL_E_NOMEM;
    }

    // Initialize sampler config
    session->sampler_cfg.temperature = desc->temperature;
    session->sampler_cfg.top_k = desc->top_k;
    session->sampler_cfg.top_p = desc->top_p;
    session->sampler_cfg.seed =
        desc->seed != 0 ? desc->seed : static_cast<uint32_t>(get_time_ns() & 0xFFFFFFFF);
    session->rng_state = session->sampler_cfg.seed;

    // Initialize state atomics using placement new
    new (&session->state) std::atomic<SessionState>(SessionState::Idle);
    new (&session->cancel_requested) std::atomic<bool>(false);
    new (&session->final_err) std::atomic<int32_t>(ASTRAL_OK);

    // Initialize statistics
    session->total_tokens = 0;
    session->t_start_ns = 0;
    session->t_first_token_ns = 0;
    session->t_end_ns = 0;
    session->n_past = 0;

    const AstralHandle handle = core::register_handle(core::HandleKind::Session, session);
    if (handle == 0) {
        model->backend->ops->session_destroy(session->backend_session_ctx);
        session->backend_session_ctx = nullptr;
        model_release(model);
        platform::vm_release(allocator_memory, allocator_capacity);
        ::operator delete(session_mem);
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

    // Manually destroy non-trivial members (atomics, allocator, ring)
    session->state.~atomic();
    session->cancel_requested.~atomic();
    session->final_err.~atomic();
    session->allocator.~FrameAllocator();
    session->token_ring.~SpscRing();

    // Release model reference (session holds one ref).
    model_release(session->model);
    session->model = nullptr;

    // Free session memory
    ::operator delete(session);
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
    session->t_start_ns = get_time_ns();
    session->t_end_ns = 0;

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

    const uint64_t start_ns = get_time_ns();
    const uint64_t timeout_ns = static_cast<uint64_t>(timeout_ms) * 1000000ULL;
    uint32_t spins = 0;

    while (true) {
        state = session->state.load(std::memory_order_acquire);
        if (state == SessionState::Completed) return ASTRAL_OK;
        if (state == SessionState::Canceled) return ASTRAL_E_CANCELED;
        if (state == SessionState::Failed) {
            const int32_t final_err = session->final_err.load(std::memory_order_acquire);
            return final_err != 0 ? static_cast<AstralErr>(final_err) : ASTRAL_E_BACKEND;
        }

        const uint64_t now_ns = get_time_ns();
        if (now_ns - start_ns >= timeout_ns) {
            return ASTRAL_E_TIMEOUT;
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
        session->desc.seed != 0 ? session->desc.seed : static_cast<uint32_t>(get_time_ns() & 0xFFFFFFFF);
    session->rng_state = session->sampler_cfg.seed;

    // Clear core state (not thread-safe; caller must ensure no concurrent stream reads).
    session->cancel_requested.store(false, std::memory_order_release);
    session->final_err.store(ASTRAL_OK, std::memory_order_release);
    session->prompt_count = 0;
    session->token_ring.reset();

    session->total_tokens = 0;
    session->t_start_ns = 0;
    session->t_first_token_ns = 0;
    session->t_end_ns = 0;
    session->n_past = 0;

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

AstralErr session_stats(Session* session, AstralStats* out_stats) {
    if (session == nullptr || out_stats == nullptr) {
        return ASTRAL_E_INVALID;
    }

    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state != SessionState::Completed && state != SessionState::Canceled && state != SessionState::Failed) {
        return ASTRAL_E_STATE;
    }

    const uint64_t t_start_ns = session->t_start_ns;
    if (t_start_ns == 0) {
        return ASTRAL_E_STATE;
    }

    const uint64_t end_ns = session->t_end_ns != 0 ? session->t_end_ns : get_time_ns();

    const uint64_t elapsed_ns = end_ns > t_start_ns ? (end_ns - t_start_ns) : 0;
    const double elapsed_s = elapsed_ns > 0 ? static_cast<double>(elapsed_ns) / 1e9 : 0.0;

    double ttft_ms = 0.0;
    if (session->t_first_token_ns > 0 && session->t_first_token_ns >= t_start_ns) {
        ttft_ms = static_cast<double>(session->t_first_token_ns - t_start_ns) / 1e6;
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
        goto finish;
    }

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

        if (session->token_eos >= 0 && next_token == session->token_eos) {
            break;
        }

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

        // Push to token ring (with backpressure)
        bool pushed = false;
        uint32_t spins = 0;
        while (!pushed) {
            if (session->token_ring.push(stream_token)) {
                pushed = true;
                break;
            }

            // Ring is full; wait for consumer to drain.
            // Use a short spin, then WFE on ARM for low overhead.
            if (spins < 64) {
                platform::cpu_pause();
            } else {
                platform::cpu_wait_for_event();
            }
            if (spins < 1024) {
                ++spins;
            }

            // Respect cancellation even if consumer isn't draining.
            if (session->cancel_requested.load(std::memory_order_acquire)) {
                final_state = SessionState::Canceled;
                final_err = ASTRAL_E_CANCELED;
                break;
            }
        }

        if (final_state == SessionState::Canceled) {
            break;
        }

        if (!pushed) {
            // Ring still full after retries; abort
            final_state = SessionState::Failed;
            final_err = ASTRAL_E_BACKEND;
            break;
        }

        // Update statistics
        session->total_tokens++;
        if (session->total_tokens == 1) {
            session->t_first_token_ns = get_time_ns();
        }
    }

finish:
    // Transition to terminal state.
    session->t_end_ns = get_time_ns();
    session->final_err.store(static_cast<int32_t>(final_err), std::memory_order_release);
    session->state.store(final_state, std::memory_order_release);
    platform::cpu_signal_event();
}

int32_t stream_read(Session* session, AstralMutSpanU8 out_buf, uint32_t timeout_ms) {
    // Validate session
    if (session == nullptr || out_buf.data == nullptr) {
        return ASTRAL_E_INVALID;
    }

    // Try to read from token ring
    concurrency::StreamToken token;
    bool success = session->token_ring.pop(&token);

    if (success) {
        // Copy UTF-8 data to output buffer
        uint32_t bytes_to_copy = token.utf8_len;
        if (bytes_to_copy > out_buf.len) {
            bytes_to_copy = out_buf.len;
        }

        std::memcpy(out_buf.data, token.utf8_data, bytes_to_copy);
        return static_cast<int32_t>(bytes_to_copy);
    }

    // Ring is empty; check if decoding is complete
    SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Completed || state == SessionState::Canceled || state == SessionState::Failed) {
        // No more data will be produced
        return 0;
    }

    // Wait for data (if timeout > 0)
    if (timeout_ms > 0) {
        uint64_t start_ns = get_time_ns();
        uint64_t timeout_ns = static_cast<uint64_t>(timeout_ms) * 1000000ULL;
        uint32_t spins = 0;

        while (true) {
            // Try to pop again
            if (session->token_ring.pop(&token)) {
                uint32_t bytes_to_copy = token.utf8_len;
                if (bytes_to_copy > out_buf.len) {
                    bytes_to_copy = out_buf.len;
                }

                std::memcpy(out_buf.data, token.utf8_data, bytes_to_copy);
                return static_cast<int32_t>(bytes_to_copy);
            }

            // Check if completed
            state = session->state.load(std::memory_order_acquire);
            if (state == SessionState::Completed || state == SessionState::Canceled || state == SessionState::Failed) {
                return 0;
            }

            // Check timeout
            uint64_t now_ns = get_time_ns();
            if (now_ns - start_ns >= timeout_ns) {
                return ASTRAL_E_TIMEOUT;
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
