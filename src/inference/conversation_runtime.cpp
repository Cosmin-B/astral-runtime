#include "conversation_runtime.hpp"

#include "../backend/backend.hpp"
#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"
#include "../core/runtime_state.hpp"
#include "../platform/atomics.h"
#include "../platform/time.h"

#include "executor.hpp"
#include "model.hpp"

#include <cstring>
#include <cstdlib>

namespace astral::inference {

namespace {

inline uint64_t get_ticks() {
    return platform::ticks_now();
}

struct ScopedAtomicFlagGuard {
    std::atomic_flag* flag = nullptr;
    bool locked = false;

    explicit ScopedAtomicFlagGuard(std::atomic_flag* f) : flag(f) {
        if (flag != nullptr) {
            locked = !flag->test_and_set(std::memory_order_acquire);
        }
    }

    ScopedAtomicFlagGuard(const ScopedAtomicFlagGuard&) = delete;
    ScopedAtomicFlagGuard& operator=(const ScopedAtomicFlagGuard&) = delete;

    ~ScopedAtomicFlagGuard() {
        if (locked && flag != nullptr) {
            flag->clear(std::memory_order_release);
        }
    }
};

constexpr size_t kDefaultAllocatorCapacity = 2 * 1024 * 1024;

inline void lock_flag(std::atomic_flag& f) {
    uint32_t spins = 0;
    while (f.test_and_set(std::memory_order_acquire)) {
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

inline void unlock_flag(std::atomic_flag& f) {
    f.clear(std::memory_order_release);
    platform::cpu_signal_event();
}

inline uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline void conv_prompt_chunks_clear(Conversation* conv) {
    if (conv == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < conv->prompt_chunk_count && i < kMaxPromptChunks; ++i) {
        prompt_chunk_reset(conv->prompt_chunks[i]);
    }
    conv->prompt_chunk_count = 0;
    conv->prompt_chunk_index = 0;
    conv->prompt_chunk_token_off = 0;
}

inline AstralErr conv_push_text_chunk(Conversation* conv, uint32_t token_start, uint32_t token_count, uint8_t finalize) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (token_count == 0) {
        return ASTRAL_OK;
    }
    if (conv->prompt_chunk_count >= kMaxPromptChunks) {
        return ASTRAL_E_NOMEM;
    }

    PromptChunk& chunk = conv->prompt_chunks[conv->prompt_chunk_count++];
    chunk = PromptChunk{};
    chunk.kind = PromptChunkKind::Text;
    chunk.finalize = finalize;
    chunk.token_start = token_start;
    chunk.token_count = token_count;
    return ASTRAL_OK;
}

inline AstralErr conv_push_media_chunk(Conversation* conv,
                                       const AstralImageDesc* image,
                                       const AstralAudioDesc* audio,
                                       uint8_t finalize) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (conv->prompt_chunk_count >= kMaxPromptChunks) {
        return ASTRAL_E_NOMEM;
    }
    if ((image == nullptr) == (audio == nullptr)) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t idx = conv->prompt_chunk_count;
    PromptChunk& chunk = conv->prompt_chunks[idx];
    chunk = PromptChunk{};
    chunk.finalize = finalize;

    if (image != nullptr) {
        if (image->size != sizeof(AstralImageDesc) || image->pixels.data == nullptr || image->pixels.len == 0 ||
            image->width == 0 || image->height == 0) {
            return ASTRAL_E_INVALID;
        }
        const size_t bytes = static_cast<size_t>(image->pixels.len);
        uint8_t* buf = static_cast<uint8_t*>(core::runtime_alloc(bytes, 1));
        if (buf == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        std::memcpy(buf, image->pixels.data, bytes);
        chunk.kind = PromptChunkKind::Image;
        chunk.image = *image;
        chunk.image.pixels.data = buf;
        chunk.image.pixels.len = image->pixels.len;
        chunk.owned_buffer = buf;
        chunk.owned_bytes = image->pixels.len;
        chunk.owned_align = 1;
    } else if (audio != nullptr) {
        if (audio->size != sizeof(AstralAudioDesc) || audio->samples.data == nullptr || audio->samples.len == 0 ||
            audio->channels == 0 || audio->sample_rate == 0 || audio->frame_count == 0) {
            return ASTRAL_E_INVALID;
        }
        const size_t bytes = static_cast<size_t>(audio->samples.len);
        uint8_t* buf = static_cast<uint8_t*>(core::runtime_alloc(bytes, 1));
        if (buf == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        std::memcpy(buf, audio->samples.data, bytes);
        chunk.kind = PromptChunkKind::Audio;
        chunk.audio = *audio;
        chunk.audio.samples.data = buf;
        chunk.audio.samples.len = audio->samples.len;
        chunk.owned_buffer = buf;
        chunk.owned_bytes = audio->samples.len;
        chunk.owned_align = 1;
    }

    conv->prompt_chunk_count = idx + 1u;
    return ASTRAL_OK;
}

inline uint32_t parse_u32_env(const char* key, uint32_t fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    uint64_t n = 0;
    for (size_t i = 0; v[i] != '\0'; ++i) {
        const char c = v[i];
        if (c < '0' || c > '9') {
            return fallback;
        }
        n = n * 10u + static_cast<uint64_t>(c - '0');
        if (n > 1000000u) {
            return fallback;
        }
    }
    return static_cast<uint32_t>(n);
}

ModelExecutor* ensure_executor(Model* model) {
    if (model == nullptr) {
        return nullptr;
    }

    ModelExecutor* ex = model->executor.load(std::memory_order_acquire);
    if (ex != nullptr) {
        return ex;
    }

    lock_flag(model->executor_lock);
    ex = model->executor.load(std::memory_order_acquire);
    if (ex == nullptr) {
        auto* created = core::runtime_new<ModelExecutor>(model);
        if (created != nullptr) {
            const AstralExecutorDesc cfg = model->executor_desc;
            const uint32_t max_slots = cfg.max_slots != 0 ? cfg.max_slots : 1;
            const uint32_t max_batch = cfg.max_batch_tokens;

            created->max_slots = clamp_u32(max_slots, 1u, ModelExecutor::kMaxSlotsHard);
            const uint32_t model_batch = model->desc.n_batch > 0 ? model->desc.n_batch : 256;
            created->max_batch_tokens =
                clamp_u32(max_batch != 0 ? max_batch : model_batch, 1u, model_batch);

            // Scheduling knob: prompt ingestion cap per slot per tick.
            const uint32_t prompt_cap =
                parse_u32_env("ASTRAL_EXEC_MAX_PROMPT_TOKENS_PER_SLOT_TICK", 8u);
            created->max_prompt_tokens_per_slot_per_tick.store(clamp_u32(prompt_cap, 1u, 1024u),
                                                              std::memory_order_relaxed);

            // Requires provider support.
            if (model->backend == nullptr || model->backend->ops == nullptr ||
                model->backend->ops->session_batch_eval == nullptr || model->backend->ops->session_batch_logits == nullptr) {
                core::runtime_delete(created);
                created = nullptr;
            } else {
                executor_start(created);
                if (!created->running.load(std::memory_order_acquire) || created->backend_session_ctx == nullptr) {
                    executor_stop_and_join(created);
                    core::runtime_delete(created);
                    created = nullptr;
                }
            }
        }
        model->executor.store(created, std::memory_order_release);
        ex = created;
    }
    unlock_flag(model->executor_lock);
    return ex;
}

} // namespace

AstralErr conv_create(const AstralConvDesc* desc, Conversation** out_conv) {
    if (desc == nullptr || out_conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralConvDesc)) {
        return ASTRAL_E_INVALID;
    }
    if (desc->model == 0) {
        return ASTRAL_E_INVALID;
    }

    auto* model = static_cast<Model*>(core::lookup_handle(desc->model, core::HandleKind::Model));
    if (model == nullptr) {
        return ASTRAL_E_INVALID;
    }

    // Ensure executor exists first (also validates backend batching support).
    ModelExecutor* ex = ensure_executor(model);
    if (ex == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    // Acquire allocator backing.
    size_t allocator_capacity = 0;
    void* allocator_memory = ::astral::core::runtime_session_scratch_acquire(
        kDefaultAllocatorCapacity, alignof(std::max_align_t), &allocator_capacity
    );
    if (allocator_memory == nullptr || allocator_capacity == 0) {
        return ASTRAL_E_NOMEM;
    }

    Conversation* conv = core::runtime_new<Conversation>(model, allocator_memory, allocator_capacity, *desc);
    if (conv == nullptr) {
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
        return ASTRAL_E_NOMEM;
    }

    // Model holds conversations via refcount.
    model->refcount.fetch_add(1, std::memory_order_relaxed);

    // Initialize prompt buffer (fixed).
    conv->prompt_capacity = 8192;
    conv->prompt_tokens = static_cast<int32_t*>(
        conv->allocator.alloc(conv->prompt_capacity * sizeof(int32_t), alignof(int32_t))
    );
    if (conv->prompt_tokens == nullptr) {
        model_release(model);
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
        core::runtime_delete(conv);
        return ASTRAL_E_NOMEM;
    }

    // Cache model info.
    uint32_t vocab_size = 0;
    uint32_t ctx_size = 0;
    const AstralErr info_err = model->backend->ops->model_info(model->backend_model_ctx, &vocab_size, &ctx_size);
    if (info_err != ASTRAL_OK || vocab_size == 0) {
        model_release(model);
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
        core::runtime_delete(conv);
        return info_err != ASTRAL_OK ? info_err : ASTRAL_E_BACKEND;
    }
    conv->vocab_size = vocab_size;
    conv->ctx_size = ctx_size;

    // Cache special tokens.
    int32_t token_bos = -1;
    int32_t token_eos = -1;
    if (model->backend->ops->model_special_tokens) {
        const AstralErr tok_err = model->backend->ops->model_special_tokens(model->backend_model_ctx, &token_bos, &token_eos);
        if (tok_err != ASTRAL_OK) {
            model_release(model);
            ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
            core::runtime_delete(conv);
            return tok_err;
        }
    }
    conv->token_bos = token_bos;
    conv->token_eos = token_eos;

    // Allocate per-conversation bounded candidate buffers.
    uint32_t sample_capacity = 0;
    if (desc->top_k > 0) {
        sample_capacity = desc->top_k;
    } else if (desc->top_p > 0.0f && desc->top_p < 1.0f) {
        sample_capacity = 2048;
    }
    if (sample_capacity > vocab_size) {
        sample_capacity = vocab_size;
    }
    conv->sample_capacity = sample_capacity;
    conv->sample_ids = sample_capacity > 0 ? static_cast<uint32_t*>(
        conv->allocator.alloc(sample_capacity * sizeof(uint32_t), alignof(uint32_t))
    ) : nullptr;
    conv->sample_logits = sample_capacity > 0 ? static_cast<float*>(
        conv->allocator.alloc(sample_capacity * sizeof(float), alignof(float))
    ) : nullptr;
    if (sample_capacity > 0 && (conv->sample_ids == nullptr || conv->sample_logits == nullptr)) {
        model_release(model);
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
        core::runtime_delete(conv);
        return ASTRAL_E_NOMEM;
    }

    // Init sampler defaults.
    conv->sampler_cfg.temperature = desc->temperature;
    conv->sampler_cfg.top_k = desc->top_k;
    conv->sampler_cfg.top_p = desc->top_p;
    conv->sampler_cfg.min_p = 0.0f;
    conv->sampler_cfg.typical_p = 1.0f;
    conv->sampler_cfg.repeat_penalty = 1.0f;
    conv->sampler_cfg.repeat_last_n = 0;
    conv->sampler_cfg.penalize_nl = 0;
    conv->sampler_cfg.presence_penalty = 0.0f;
    conv->sampler_cfg.frequency_penalty = 0.0f;
    conv->sampler_cfg.seed = desc->seed != 0 ? desc->seed : static_cast<uint32_t>(get_ticks() & 0xFFFFFFFFu);
    conv->rng_state = conv->sampler_cfg.seed;
    conv->mirostat_mu = 0.0f;

    // Allocate slot by scanning under executor lock (not a hot path).
    lock_flag(model->executor_lock);
    uint32_t sid = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < ex->max_slots; ++i) {
        if (ex->slots[i].load(std::memory_order_relaxed) == nullptr) {
            ex->slots[i].store(conv, std::memory_order_release);
            sid = i;
            break;
        }
    }
    unlock_flag(model->executor_lock);

    if (sid == 0xFFFFFFFFu) {
        model_release(model);
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
        core::runtime_delete(conv);
        return ASTRAL_E_NOMEM;
    }
    conv->slot_id.store(sid, std::memory_order_release);

    const AstralHandle handle = core::register_handle(core::HandleKind::Conversation, conv);
    if (handle == 0) {
        ex->slots[sid].store(nullptr, std::memory_order_release);
        conv->slot_id.store(0xFFFFFFFFu, std::memory_order_release);
        model_release(model);
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
        core::runtime_delete(conv);
        return ASTRAL_E_BUSY;
    }
    conv->handle = handle;

    *out_conv = conv;
    platform::cpu_signal_event();
    return ASTRAL_OK;
}

void conv_destroy(Conversation* conv) {
    if (conv == nullptr) {
        return;
    }

    core::unregister_handle(conv->handle, core::HandleKind::Conversation);
    conv->handle = 0;

    // Cancel and wait if decoding.
    ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        conv->cancel_requested.store(true, std::memory_order_release);
        platform::cpu_signal_event();
        uint32_t spins = 0;
        while (true) {
            st = conv->state.load(std::memory_order_acquire);
            if (st != ConvState::Decoding) {
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

    Model* model = conv->model;
    ModelExecutor* ex = model ? model->executor.load(std::memory_order_acquire) : nullptr;
    const uint32_t sid = conv->slot_id.load(std::memory_order_acquire);

    if (ex != nullptr && sid < ex->max_slots) {
        ex->slots[sid].store(nullptr, std::memory_order_release);
        platform::cpu_signal_event();

        // Wait for executor to drop all in-flight references.
        uint32_t spins = 0;
        while (conv->exec_refs.load(std::memory_order_acquire) != 0) {
            if (spins < 64) {
                platform::cpu_pause();
            } else {
                platform::cpu_wait_for_event();
            }
            if (spins < 1024) {
                ++spins;
            }
        }

        // Clear provider-owned slot state when the backend exposes slot reset.
        if (model && model->backend && model->backend->ops && model->backend->ops->session_slot_reset &&
            ex->backend_session_ctx) {
            (void)model->backend->ops->session_slot_reset(ex->backend_session_ctx, sid);
        }
    }

    conv_prompt_chunks_clear(conv);

    // Release allocator memory.
    if (conv->allocator_memory != nullptr) {
        ::astral::core::runtime_session_scratch_release(conv->allocator_memory, conv->allocator_capacity);
    }

    if (conv->grammar_gbnf != nullptr) {
        ::astral::core::runtime_free(conv->grammar_gbnf, conv->grammar_gbnf_len, 1);
        conv->grammar_gbnf = nullptr;
        conv->grammar_gbnf_len = 0;
    }
    if (conv->grammar_root != nullptr) {
        ::astral::core::runtime_free(conv->grammar_root, conv->grammar_root_len, 1);
        conv->grammar_root = nullptr;
        conv->grammar_root_len = 0;
    }
    if (conv->grammar_json != nullptr) {
        ::astral::core::runtime_free(conv->grammar_json, conv->grammar_json_len, 1);
        conv->grammar_json = nullptr;
        conv->grammar_json_len = 0;
    }

    // Release model ref (conversation holds one ref).
    if (model) {
        model_release(model);
        conv->model = nullptr;
    }

    core::runtime_delete(conv);
}

AstralErr conv_feed(Conversation* conv, AstralSpanU8 prompt_chunk, uint8_t finalize) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ConvState state = conv->state.load(std::memory_order_acquire);
    if (state != ConvState::Idle && state != ConvState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    if (state == ConvState::Idle) {
        conv->state.store(ConvState::FeedingPrompt, std::memory_order_release);
    }

    const bool should_tokenize =
        (prompt_chunk.len > 0 && prompt_chunk.data != nullptr) || (finalize != 0 && conv->prompt_count == 0);

    if (should_tokenize) {
        if (conv->prompt_chunk_count >= kMaxPromptChunks) {
            return ASTRAL_E_NOMEM;
        }
        const uint32_t space = conv->prompt_capacity - conv->prompt_count;
        uint32_t n_tokens = 0;
        const uint32_t token_start = conv->prompt_count;

        const AstralErr err = conv->model->backend->ops->tokenize(
            conv->model->backend_model_ctx,
            prompt_chunk,
            conv->prompt_tokens + conv->prompt_count,
            space,
            /*add_special=*/conv->prompt_count == 0,
            /*parse_special=*/false,
            &n_tokens
        );
        if (err != ASTRAL_OK) {
            return err;
        }
        if (n_tokens == 0) {
            return ASTRAL_OK;
        }

        conv->prompt_count += n_tokens;
        if (conv->prompt_count >= conv->prompt_capacity) {
            return ASTRAL_E_NOMEM;
        }

        const AstralErr chunk_err = conv_push_text_chunk(conv, token_start, n_tokens, finalize);
        if (chunk_err != ASTRAL_OK) {
            return chunk_err;
        }
    }

    return ASTRAL_OK;
}

AstralErr conv_feed_image(Conversation* conv, const AstralImageDesc* image, uint8_t finalize) {
    if (conv == nullptr || image == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ConvState state = conv->state.load(std::memory_order_acquire);
    if (state != ConvState::Idle && state != ConvState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    if (state == ConvState::Idle) {
        conv->state.store(ConvState::FeedingPrompt, std::memory_order_release);
    }

    if (conv->model == nullptr || conv->model->backend == nullptr || conv->model->backend->ops == nullptr ||
        conv->model->backend->ops->session_feed_image == nullptr ||
        conv->model->backend->ops->session_slot_pos == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    return conv_push_media_chunk(conv, image, nullptr, finalize);
}

AstralErr conv_feed_audio(Conversation* conv, const AstralAudioDesc* audio, uint8_t finalize) {
    if (conv == nullptr || audio == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ConvState state = conv->state.load(std::memory_order_acquire);
    if (state != ConvState::Idle && state != ConvState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    if (state == ConvState::Idle) {
        conv->state.store(ConvState::FeedingPrompt, std::memory_order_release);
    }

    if (conv->model == nullptr || conv->model->backend == nullptr || conv->model->backend->ops == nullptr ||
        conv->model->backend->ops->session_feed_audio == nullptr ||
        conv->model->backend->ops->session_slot_pos == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    return conv_push_media_chunk(conv, nullptr, audio, finalize);
}

AstralErr conv_decode(Conversation* conv) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }

    const ConvState state = conv->state.load(std::memory_order_acquire);
    if (state != ConvState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    if (conv->prompt_count == 0 && conv->prompt_chunk_count == 0) {
        return ASTRAL_E_INVALID;
    }

    // Reset per-run state.
    conv->cancel_requested.store(false, std::memory_order_release);
    conv->final_err.store(ASTRAL_OK, std::memory_order_release);
    conv->token_ring.reset();
    conv->meta_ring.reset();
    conv->pending_len = 0;
    conv->pending_off = 0;
    conv->prompt_off = 0;
    conv->prompt_chunk_index = 0;
    conv->prompt_chunk_token_off = 0;
    conv->total_tokens = 0;
    conv->t_start_ticks = get_ticks();
    conv->t_first_token_ticks = 0;
    conv->t_end_ticks = 0;
    conv->n_past = 0;
    conv->stop_tail_len = 0;
    conv->hold_head = 0;
    conv->hold_count = 0;
    conv->pending_token_valid = 0;
    conv->pending_emit_valid = 0;
    conv->mirostat_mu = 0.0f;

    // Slot reset must be performed by the executor thread (backend session ctx is not thread-safe).
    conv->needs_slot_reset = 1;
    conv->grammar_applied = 0;

    conv->state.store(ConvState::Decoding, std::memory_order_release);
    platform::cpu_signal_event();
    return ASTRAL_OK;
}

AstralErr conv_cancel(Conversation* conv) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    conv->cancel_requested.store(true, std::memory_order_release);
    platform::cpu_signal_event();
    return ASTRAL_OK;
}

AstralErr conv_state(Conversation* conv, AstralSessionState* out_state) {
    if (conv == nullptr || out_state == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_state = conv_state_to_public(conv->state.load(std::memory_order_acquire));
    return ASTRAL_OK;
}

AstralErr conv_wait(Conversation* conv, uint32_t timeout_ms) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }

    ConvState state = conv->state.load(std::memory_order_acquire);
    if (state == ConvState::Completed) return ASTRAL_OK;
    if (state == ConvState::Canceled) return ASTRAL_E_CANCELED;
    if (state == ConvState::Failed) {
        const int32_t e = conv->final_err.load(std::memory_order_acquire);
        return e != 0 ? static_cast<AstralErr>(e) : ASTRAL_E_BACKEND;
    }

    if (timeout_ms == 0) {
        return ASTRAL_E_TIMEOUT;
    }

    const uint64_t start_ticks = get_ticks();
    const uint64_t timeout_ns = static_cast<uint64_t>(timeout_ms) * 1000000ULL;
    const uint64_t timeout_ticks = platform::ticks_from_ns(timeout_ns);
    const uint64_t deadline = start_ticks + timeout_ticks;
    const uint32_t check_mask =
        timeout_ms <= 1u   ? 31u :
        timeout_ms <= 10u  ? 255u :
        timeout_ms <= 100u ? 4095u :
                             65535u;
    uint32_t spins = 0;

    while (true) {
        state = conv->state.load(std::memory_order_acquire);
        if (state == ConvState::Completed) return ASTRAL_OK;
        if (state == ConvState::Canceled) return ASTRAL_E_CANCELED;
        if (state == ConvState::Failed) {
            const int32_t e = conv->final_err.load(std::memory_order_acquire);
            return e != 0 ? static_cast<AstralErr>(e) : ASTRAL_E_BACKEND;
        }

        if ((spins & check_mask) == 0u) {
            if (get_ticks() >= deadline) {
                return ASTRAL_E_TIMEOUT;
            }
        }

        if (spins < 64) {
            platform::cpu_pause();
        } else {
            platform::cpu_wait_for_event();
        }
        ++spins;
    }
}

AstralErr conv_reset(Conversation* conv, const AstralConvDesc* desc) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }

    const ConvState state = conv->state.load(std::memory_order_acquire);
    if (state == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }

    ScopedAtomicFlagGuard stream_guard(&conv->stream_read_lock);
    if (!stream_guard.locked) {
        return ASTRAL_E_STATE;
    }
    ScopedAtomicFlagGuard meta_guard(&conv->meta_read_lock);
    if (!meta_guard.locked) {
        return ASTRAL_E_STATE;
    }

    if (desc != nullptr) {
        if (desc->size != sizeof(AstralConvDesc) || desc->model != conv->desc.model) {
            return ASTRAL_E_INVALID;
        }
        if (desc->top_k > 0 && desc->top_k > conv->sample_capacity) {
            return ASTRAL_E_INVALID;
        }
        conv->desc = *desc;
    }

    conv->sampler_cfg.temperature = conv->desc.temperature;
    conv->sampler_cfg.top_k = conv->desc.top_k;
    conv->sampler_cfg.top_p = conv->desc.top_p;
    conv->sampler_cfg.seed =
        conv->desc.seed != 0 ? conv->desc.seed : static_cast<uint32_t>(get_ticks() & 0xFFFFFFFFu);
    conv->rng_state = conv->sampler_cfg.seed;
    conv->mirostat_mu = 0.0f;

    conv->cancel_requested.store(false, std::memory_order_release);
    conv->final_err.store(ASTRAL_OK, std::memory_order_release);
    conv->prompt_count = 0;
    conv->prompt_off = 0;
    conv_prompt_chunks_clear(conv);
    conv->token_ring.reset();
    conv->meta_ring.reset();
    conv->pending_len = 0;
    conv->pending_off = 0;
    conv->total_tokens = 0;
    conv->t_start_ticks = 0;
    conv->t_first_token_ticks = 0;
    conv->t_end_ticks = 0;
    conv->n_past = 0;
    conv->stop_tail_len = 0;
    conv->hold_head = 0;
    conv->hold_count = 0;
    conv->pending_token_valid = 0;
    conv->pending_emit_valid = 0;
    conv->needs_slot_reset = 1;
    conv->grammar_applied = 0;

    conv->state.store(ConvState::Idle, std::memory_order_release);
    platform::cpu_signal_event();
    return ASTRAL_OK;
}

AstralErr conv_set_sampler(Conversation* conv, const AstralSamplerDesc* desc) {
    if (conv == nullptr || desc == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralSamplerDesc)) {
        return ASTRAL_E_INVALID;
    }
    const ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (desc->top_k > 0 && desc->top_k > conv->sample_capacity) {
        return ASTRAL_E_INVALID;
    }

    conv->sampler_cfg.temperature = desc->temperature;
    conv->sampler_cfg.top_k = desc->top_k;
    conv->sampler_cfg.top_p = desc->top_p;
    conv->sampler_cfg.min_p = desc->min_p;
    conv->sampler_cfg.typical_p = desc->typical_p;
    conv->sampler_cfg.repeat_penalty = desc->repeat_penalty;
    conv->sampler_cfg.repeat_last_n = desc->repeat_last_n;
    conv->sampler_cfg.penalize_nl = desc->penalize_nl;
    conv->sampler_cfg.presence_penalty = desc->presence_penalty;
    conv->sampler_cfg.frequency_penalty = desc->frequency_penalty;
    conv->sampler_cfg.mirostat = desc->mirostat;
    conv->sampler_cfg.mirostat_tau = desc->mirostat_tau;
    conv->sampler_cfg.mirostat_eta = desc->mirostat_eta;

    return ASTRAL_OK;
}

AstralErr conv_penalty_prompt_set_tokens(Conversation* conv, const int32_t* tokens, uint32_t count) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (count > 0 && tokens == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (count == 0) {
        conv->token_counts_base = nullptr;
        return ASTRAL_OK;
    }

    if (conv->token_counts_base == nullptr) {
        conv->token_counts_base = static_cast<uint16_t*>(
            conv->allocator.alloc(conv->vocab_size * sizeof(uint16_t), alignof(uint16_t))
        );
        if (conv->token_counts_base == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        std::memset(conv->token_counts_base, 0, conv->vocab_size * sizeof(uint16_t));
    } else {
        std::memset(conv->token_counts_base, 0, conv->vocab_size * sizeof(uint16_t));
    }

    for (uint32_t i = 0; i < count; ++i) {
        const int32_t t = tokens[i];
        if (t >= 0 && static_cast<uint32_t>(t) < conv->vocab_size) {
            uint16_t& c = conv->token_counts_base[static_cast<uint32_t>(t)];
            if (c != 0xFFFFu) {
                ++c;
            }
        }
    }

    return ASTRAL_OK;
}

AstralErr conv_stop_clear(Conversation* conv) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }
    conv->stop_seq_count = 0;
    std::memset(conv->stop_seq_lens, 0, sizeof(conv->stop_seq_lens));
    std::memset(conv->stop_seq_tokens, 0, sizeof(conv->stop_seq_tokens));
    conv->stop_max_len = 0;
    return ASTRAL_OK;
}

AstralErr conv_stop_add_utf8(Conversation* conv, AstralSpanU8 utf8) {
    if (conv == nullptr || conv->model == nullptr || conv->model->backend == nullptr || conv->model->backend->ops == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (conv->stop_seq_count >= 16) {
        return ASTRAL_E_NOMEM;
    }

    int32_t tokens[32];
    uint32_t out_count = 0;
    const AstralErr err = conv->model->backend->ops->tokenize(
        conv->model->backend_model_ctx, utf8, tokens, 32, /*add_special=*/false, /*parse_special=*/false, &out_count
    );
    if (err != ASTRAL_OK) {
        return err;
    }
    if (out_count == 0 || out_count > 32) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t idx = conv->stop_seq_count++;
    conv->stop_seq_lens[idx] = static_cast<uint8_t>(out_count);
    for (uint32_t i = 0; i < out_count; ++i) {
        conv->stop_seq_tokens[idx][i] = tokens[i];
    }
    if (out_count > conv->stop_max_len) {
        conv->stop_max_len = out_count;
    }
    return ASTRAL_OK;
}

AstralErr conv_stop_set_utf8(Conversation* conv, const AstralSpanU8* seqs, uint32_t count) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const AstralErr err = conv_stop_clear(conv);
    if (err != ASTRAL_OK) {
        return err;
    }
    if (count > 16) {
        return ASTRAL_E_NOMEM;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const AstralErr e = conv_stop_add_utf8(conv, seqs[i]);
        if (e != ASTRAL_OK) {
            return e;
        }
    }
    return ASTRAL_OK;
}

AstralErr conv_set_logprobs(Conversation* conv, uint32_t n_probs) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (n_probs > ASTRAL_LOGPROBS_MAX) {
        n_probs = ASTRAL_LOGPROBS_MAX;
    }
    conv->logprobs_n = n_probs;
    return ASTRAL_OK;
}

static uint8_t* copy_span_malloc(AstralSpanU8 s, uint32_t* out_len) {
    if (out_len == nullptr) {
        return nullptr;
    }
    *out_len = 0;
    if (s.data == nullptr || s.len == 0) {
        return nullptr;
    }
    uint8_t* p = static_cast<uint8_t*>(::astral::core::runtime_alloc(s.len, 1));
    if (p == nullptr) {
        return nullptr;
    }
    std::memcpy(p, s.data, s.len);
    *out_len = s.len;
    return p;
}

AstralErr conv_grammar_set_gbnf(Conversation* conv, AstralSpanU8 gbnf, AstralSpanU8 root) {
    if (conv == nullptr || gbnf.data == nullptr || gbnf.len == 0) {
        return ASTRAL_E_INVALID;
    }

    const ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }

    const backend::BackendOps* ops = (conv->model && conv->model->backend) ? conv->model->backend->ops : nullptr;
    if (ops == nullptr || ops->session_apply_grammar_for_slot == nullptr || ops->session_grammar_set_gbnf_for_slot == nullptr ||
        ops->session_grammar_clear_for_slot == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    if (conv->grammar_gbnf != nullptr) {
        ::astral::core::runtime_free(conv->grammar_gbnf, conv->grammar_gbnf_len, 1);
        conv->grammar_gbnf = nullptr;
        conv->grammar_gbnf_len = 0;
    }
    if (conv->grammar_root != nullptr) {
        ::astral::core::runtime_free(conv->grammar_root, conv->grammar_root_len, 1);
        conv->grammar_root = nullptr;
        conv->grammar_root_len = 0;
    }
    if (conv->grammar_json != nullptr) {
        ::astral::core::runtime_free(conv->grammar_json, conv->grammar_json_len, 1);
        conv->grammar_json = nullptr;
        conv->grammar_json_len = 0;
    }

    uint32_t gbnf_len = 0;
    uint8_t* gbnf_copy = copy_span_malloc(gbnf, &gbnf_len);
    if (gbnf_copy == nullptr) {
        return ASTRAL_E_NOMEM;
    }

    uint32_t root_len = 0;
    uint8_t* root_copy = nullptr;
    if (root.data != nullptr && root.len > 0) {
        root_copy = copy_span_malloc(root, &root_len);
        if (root_copy == nullptr) {
            ::astral::core::runtime_free(gbnf_copy, gbnf_len, 1);
            return ASTRAL_E_NOMEM;
        }
    }

    conv->grammar_gbnf = gbnf_copy;
    conv->grammar_gbnf_len = gbnf_len;
    conv->grammar_root = root_copy;
    conv->grammar_root_len = root_len;
    conv->grammar_kind = 1;
    conv->grammar_dirty = 1;
    conv->grammar_applied = 0;

    platform::cpu_signal_event();
    return ASTRAL_OK;
}

AstralErr conv_grammar_set_json_schema(Conversation* conv, AstralSpanU8 json_schema) {
    if (conv == nullptr || json_schema.data == nullptr || json_schema.len == 0) {
        return ASTRAL_E_INVALID;
    }

    const ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }

    const backend::BackendOps* ops = (conv->model && conv->model->backend) ? conv->model->backend->ops : nullptr;
    if (ops == nullptr || ops->session_apply_grammar_for_slot == nullptr || ops->session_grammar_set_json_schema_for_slot == nullptr ||
        ops->session_grammar_clear_for_slot == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    if (conv->grammar_gbnf != nullptr) {
        ::astral::core::runtime_free(conv->grammar_gbnf, conv->grammar_gbnf_len, 1);
        conv->grammar_gbnf = nullptr;
        conv->grammar_gbnf_len = 0;
    }
    if (conv->grammar_root != nullptr) {
        ::astral::core::runtime_free(conv->grammar_root, conv->grammar_root_len, 1);
        conv->grammar_root = nullptr;
        conv->grammar_root_len = 0;
    }
    if (conv->grammar_json != nullptr) {
        ::astral::core::runtime_free(conv->grammar_json, conv->grammar_json_len, 1);
        conv->grammar_json = nullptr;
        conv->grammar_json_len = 0;
    }

    uint32_t schema_len = 0;
    uint8_t* schema_copy = copy_span_malloc(json_schema, &schema_len);
    if (schema_copy == nullptr) {
        return ASTRAL_E_NOMEM;
    }

    conv->grammar_json = schema_copy;
    conv->grammar_json_len = schema_len;
    conv->grammar_kind = 2;
    conv->grammar_dirty = 1;
    conv->grammar_applied = 0;

    platform::cpu_signal_event();
    return ASTRAL_OK;
}

AstralErr conv_grammar_clear(Conversation* conv) {
    if (conv == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const ConvState st = conv->state.load(std::memory_order_acquire);
    if (st == ConvState::Decoding) {
        return ASTRAL_E_STATE;
    }

    if (conv->grammar_gbnf != nullptr) {
        ::astral::core::runtime_free(conv->grammar_gbnf, conv->grammar_gbnf_len, 1);
        conv->grammar_gbnf = nullptr;
        conv->grammar_gbnf_len = 0;
    }
    if (conv->grammar_root != nullptr) {
        ::astral::core::runtime_free(conv->grammar_root, conv->grammar_root_len, 1);
        conv->grammar_root = nullptr;
        conv->grammar_root_len = 0;
    }
    if (conv->grammar_json != nullptr) {
        ::astral::core::runtime_free(conv->grammar_json, conv->grammar_json_len, 1);
        conv->grammar_json = nullptr;
        conv->grammar_json_len = 0;
    }

    conv->grammar_kind = 0;
    conv->grammar_dirty = 1;
    conv->grammar_applied = 0;
    platform::cpu_signal_event();
    return ASTRAL_OK;
}

AstralErr conv_stats(Conversation* conv, AstralConvStats* out_stats) {
    if (conv == nullptr || out_stats == nullptr) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t sid = conv->slot_id.load(std::memory_order_acquire);
    out_stats->slot_id = sid;
    out_stats->prompt_tokens = conv->prompt_count;
    out_stats->kv_tokens = conv->n_past;
    out_stats->_padding0 = 0;
    out_stats->generated_tokens = conv->total_tokens;

    // Best-effort timing:
    // - first token time is only meaningful after at least one token is produced
    // - throughput is only meaningful once we have a time window
    out_stats->t_first_token_ms = 0.0;
    out_stats->tok_per_s = 0.0;

    const ConvState st = conv->state.load(std::memory_order_acquire);
    (void)st;

    if (conv->t_start_ticks != 0 && conv->t_first_token_ticks != 0) {
        const uint64_t dt = conv->t_first_token_ticks - conv->t_start_ticks;
        out_stats->t_first_token_ms = static_cast<double>(platform::ticks_to_ns(dt)) / 1.0e6;
    }

    // Throughput:
    // - Use end time if terminal, otherwise use "now".
    if (conv->t_start_ticks != 0 && conv->total_tokens > 0) {
        const uint64_t end_ticks =
            (st == ConvState::Completed || st == ConvState::Canceled || st == ConvState::Failed)
                ? conv->t_end_ticks
                : get_ticks();
        if (end_ticks > conv->t_start_ticks) {
            const double dt_s = static_cast<double>(platform::ticks_to_ns(end_ticks - conv->t_start_ticks)) / 1.0e9;
            if (dt_s > 0.0) {
                out_stats->tok_per_s = static_cast<double>(conv->total_tokens) / dt_s;
            }
        }
    }

    return ASTRAL_OK;
}

int32_t conv_stream_read(Conversation* conv, AstralMutSpanU8 out_buf, uint32_t timeout_ms) {
    if (conv == nullptr || out_buf.data == nullptr || out_buf.len == 0) {
        return ASTRAL_E_INVALID;
    }

    ScopedAtomicFlagGuard guard(&conv->stream_read_lock);
    if (!guard.locked) {
        return ASTRAL_E_STATE;
    }

    if (conv->pending_len > conv->pending_off) {
        const uint32_t avail = static_cast<uint32_t>(conv->pending_len - conv->pending_off);
        const uint32_t n = (avail > out_buf.len) ? out_buf.len : avail;
        std::memcpy(out_buf.data, conv->pending_utf8 + conv->pending_off, n);
        conv->pending_off = static_cast<uint8_t>(conv->pending_off + n);
        if (conv->pending_off >= conv->pending_len) {
            conv->pending_len = 0;
            conv->pending_off = 0;
        }
        return static_cast<int32_t>(n);
    }

    const uint64_t start_ticks = get_ticks();
    const uint64_t timeout_ticks = platform::ticks_from_ns(static_cast<uint64_t>(timeout_ms) * 1000000ULL);
    const uint64_t deadline = start_ticks + timeout_ticks;
    uint32_t spins = 0;

    while (true) {
        concurrency::StreamToken tok{};
        if (conv->token_ring.pop(&tok)) {
            const uint32_t n = (tok.utf8_len > out_buf.len) ? out_buf.len : tok.utf8_len;
            std::memcpy(out_buf.data, tok.utf8_data, n);
            if (n < tok.utf8_len) {
                std::memcpy(conv->pending_utf8, tok.utf8_data, tok.utf8_len);
                conv->pending_len = static_cast<uint8_t>(tok.utf8_len);
                conv->pending_off = static_cast<uint8_t>(n);
            }
            platform::cpu_signal_event();
            return static_cast<int32_t>(n);
        }

        const ConvState st = conv->state.load(std::memory_order_acquire);
        if ((st == ConvState::Completed || st == ConvState::Canceled || st == ConvState::Failed) &&
            conv->pending_emit_valid == 0 && conv->hold_count == 0) {
            return 0;
        }

        if (timeout_ms == 0) {
            return ASTRAL_E_TIMEOUT;
        }

        if (get_ticks() >= deadline) {
            return ASTRAL_E_TIMEOUT;
        }

        if (spins < 64) {
            platform::cpu_pause();
        } else {
            platform::cpu_wait_for_event();
        }
        ++spins;
    }
}

int32_t conv_stream_read_meta(Conversation* conv, AstralTokenMeta* out_events, uint32_t capacity, uint32_t timeout_ms) {
    if (conv == nullptr || out_events == nullptr || capacity == 0) {
        return ASTRAL_E_INVALID;
    }

    ScopedAtomicFlagGuard guard(&conv->meta_read_lock);
    if (!guard.locked) {
        return ASTRAL_E_STATE;
    }

    AstralTokenMeta ev{};
    if (conv->meta_ring.pop(&ev)) {
        out_events[0] = ev;
        uint32_t n = 1;
        while (n < capacity && conv->meta_ring.pop(&ev)) {
            out_events[n++] = ev;
        }
        platform::cpu_signal_event();
        return static_cast<int32_t>(n);
    }

    if (timeout_ms == 0) {
        return ASTRAL_E_TIMEOUT;
    }

    const uint64_t start_ticks = get_ticks();
    const uint64_t timeout_ticks = platform::ticks_from_ns(static_cast<uint64_t>(timeout_ms) * 1000000ULL);
    const uint64_t deadline = start_ticks + timeout_ticks;
    uint32_t spins = 0;

    while (get_ticks() < deadline) {
        if (conv->meta_ring.pop(&ev)) {
            out_events[0] = ev;
            uint32_t n = 1;
            while (n < capacity && conv->meta_ring.pop(&ev)) {
                out_events[n++] = ev;
            }
            platform::cpu_signal_event();
            return static_cast<int32_t>(n);
        }

        if (spins < 64) {
            platform::cpu_pause();
        } else {
            platform::cpu_wait_for_event();
        }
        ++spins;
    }

    return ASTRAL_E_TIMEOUT;
}

} // namespace astral::inference
