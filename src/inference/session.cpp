#include "session.hpp"
#include "../core/work_queue.hpp"
#include "../core/runtime_state.hpp"
#include "../core/runtime_alloc.hpp"
#include "../platform/atomics.h"
#include "../platform/time.h"
#include "../utils/trace.hpp"
#include <cstring>
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
    , prompt_chunk_count(0)
    , prompt_chunk_index(0)
    , prompt_chunk_token_off(0)
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
    , mirostat_mu(0.0f)
    , logprobs_n(0)
    , meta_ring()
    , token_counts(nullptr)
    , token_counts_base(nullptr)
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
    , stop_max_len(0)
    , adapter_count(0)
    , toolset(nullptr)
    , tool_choice_mode(ASTRAL_TOOL_CHOICE_AUTO)
    , slot_id(0)
    , worker_id(0) {}

namespace {

constexpr uint32_t kStreamReadBatchCap = 16;

inline uint64_t get_ticks() {
    return platform::ticks_now();
}

inline uint32_t max_stream_read_batch(uint32_t remaining) {
    uint32_t count = remaining / concurrency::kStreamTokenUtf8Capacity;
    if (count == 0) {
        count = 1;
    }
    return count < kStreamReadBatchCap ? count : kStreamReadBatchCap;
}

struct SessionStreamDrainConsumer {
  Session* session;
  uint8_t*& dst;
  uint32_t& remaining;
  uint32_t& total;
  bool& partial;

  void operator()(const concurrency::StreamToken& token) const {
    const uint32_t tok_len = token.utf8_len;
    if (tok_len == 0) {
      return;
    }
    if (tok_len <= remaining) {
      std::memcpy(dst, token.utf8_data, tok_len);
      dst += tok_len;
      remaining -= tok_len;
      total += tok_len;
      return;
    }

    const uint32_t copied = remaining;
    std::memcpy(dst, token.utf8_data, copied);
    std::memcpy(session->pending_utf8, token.utf8_data, tok_len);
    session->pending_len = static_cast<uint8_t>(tok_len);
    session->pending_off = static_cast<uint8_t>(copied);
    total += copied;
    remaining = 0;
    partial = true;
  }
};

int32_t session_stream_drain(Session* session, AstralMutSpanU8 out_buf) {
    uint8_t* dst = out_buf.data;
    uint32_t remaining = out_buf.len;
    uint32_t total = 0;

    while (remaining != 0) {
      bool partial = false;
      SessionStreamDrainConsumer consumer{session, dst, remaining, total, partial};
      const size_t consumed =
          session->token_ring.consume_batch(max_stream_read_batch(remaining), consumer);
      if (consumed == 0) {
        break;
      }
      if (partial) {
        return static_cast<int32_t>(total);
      }
    }

    return static_cast<int32_t>(total);
}

inline bool session_stream_flush_one(Session* session, const concurrency::StreamToken& tok) {
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
}

inline bool session_hold_flush_all(Session* session,
                                   concurrency::StreamToken* hold,
                                   uint32_t* hold_head,
                                   uint32_t* hold_count,
                                   uint32_t hold_cap) {
    while (*hold_count > 0) {
        if (!session_stream_flush_one(session, hold[*hold_head])) {
            return false;
        }
        *hold_head = (*hold_head + 1) % hold_cap;
        --(*hold_count);
    }
    return true;
}

inline bool session_hold_push(Session* session,
                              const concurrency::StreamToken& tok,
                              concurrency::StreamToken* hold,
                              uint32_t* hold_head,
                              uint32_t* hold_count,
                              uint32_t hold_cap,
                              uint32_t stop_hold) {
    if (stop_hold == 0) {
        return session_stream_flush_one(session, tok);
    }

    const uint32_t tail = (*hold_head + *hold_count) % hold_cap;
    hold[tail] = tok;
    ++(*hold_count);

    while (*hold_count > stop_hold) {
        if (!session_stream_flush_one(session, hold[*hold_head])) {
            return false;
        }
        *hold_head = (*hold_head + 1) % hold_cap;
        --(*hold_count);
    }
    return true;
}

inline bool session_stop_check(Session* session,
                               int32_t token,
                               int32_t* stop_tail,
                               uint32_t* stop_tail_len,
                               uint32_t* out_match_len) {
    if (session->stop_seq_count == 0 || token < 0) {
        return false;
    }

    if (*stop_tail_len < 32) {
        stop_tail[(*stop_tail_len)++] = token;
    } else {
        std::memmove(stop_tail, stop_tail + 1, (32 - 1) * sizeof(int32_t));
        stop_tail[31] = token;
    }

    for (uint32_t i = 0; i < session->stop_seq_count; ++i) {
        const uint32_t len = session->stop_seq_lens[i];
        if (len == 0 || len > *stop_tail_len) {
            continue;
        }
        bool match = true;
        for (uint32_t j = 0; j < len; ++j) {
            if (stop_tail[*stop_tail_len - len + j] != session->stop_seq_tokens[i][j]) {
                match = false;
                break;
            }
        }
        if (match) {
            if (out_match_len != nullptr) {
                *out_match_len = len;
            }
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Session state serialization wrapper
//
// llama_state_* only serializes the provider state. For deterministic continuation and parity
// checks we also need to capture Astral-owned sampling state (rng_state, sampler knobs).
//
// This wrapper is backward compatible: if the magic is not present, we assume the payload is the
// legacy provider-only format and forward it to the backend.
// ---------------------------------------------------------------------------

static constexpr uint32_t kStateMagic = 0x41535452u; // 'ASTR'
static constexpr uint16_t kStateVersion = 1;

struct SessionStateHeaderV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t backend_bytes;

    uint32_t rng_state;
    uint32_t logprobs_n;
    uint32_t slot_id;
    uint32_t _pad0;

    AstralSamplerDesc sampler;
    float mirostat_mu;
    uint32_t _pad1[3];
};

static_assert(sizeof(SessionStateHeaderV1) <= 256, "SessionStateHeaderV1 unexpectedly large");
static_assert((sizeof(SessionStateHeaderV1) % 8) == 0, "SessionStateHeaderV1 must be 8-byte aligned");

static bool session_state_parse_header(AstralSpanU8 bytes, SessionStateHeaderV1* out, const uint8_t** out_backend) {
    if (out == nullptr || out_backend == nullptr) {
        return false;
    }
    *out_backend = nullptr;
    if (bytes.data == nullptr || bytes.len < sizeof(SessionStateHeaderV1)) {
        return false;
    }

    SessionStateHeaderV1 h{};
    std::memcpy(&h, bytes.data, sizeof(h));
    if (h.magic != kStateMagic || h.version != kStateVersion) {
        return false;
    }
    if (h.header_bytes != sizeof(SessionStateHeaderV1)) {
        return false;
    }
    const uint64_t total = static_cast<uint64_t>(bytes.len);
    if (h.backend_bytes == 0 || (static_cast<uint64_t>(h.header_bytes) + h.backend_bytes) > total) {
        return false;
    }

    *out = h;
    *out_backend = bytes.data + h.header_bytes;
    return true;
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

/// Default allocator capacity per session (2MB).
/// Sufficient for:
/// - Prompt tokens: ~8K tokens = 32KB
/// - Logits buffer: 32K vocab * 4 bytes = 128KB
/// - Indices buffer: 32K vocab * 4 bytes = 128KB
/// - Temp buffers: ~256KB
/// - Headroom: ~1.5MB
constexpr size_t kDefaultAllocatorCapacity = 2 * 1024 * 1024;

void decode_work(void* user) {
    ASTRAL_ZONE_N("astral.decode_work");
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

    if (repeat_last_n != 0) {
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
    }

    if (session->sampler_cfg.penalize_nl != 0 && session->token_nl < 0 && session->model != nullptr &&
        session->model->backend != nullptr && session->model->backend->ops != nullptr) {
        // Resolve a single-token "\\n" when the backend tokenizer exposes one.
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
    return ::astral::core::runtime_alloc(sizeof(Session), alignof(Session));
}

inline void session_free_mem(void* p) noexcept {
    ::astral::core::runtime_free(p, sizeof(Session), alignof(Session));
}

inline void session_prompt_chunks_clear(Session* session) {
    if (session == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < session->prompt_chunk_count && i < kMaxPromptChunks; ++i) {
        prompt_chunk_reset(session->prompt_chunks[i]);
    }
    session->prompt_chunk_count = 0;
    session->prompt_chunk_index = 0;
    session->prompt_chunk_token_off = 0;
}

inline void session_adapter_refs_clear(Session* session) {
    for (uint32_t i = 0; i < session->adapter_count; ++i) {
        Adapter* adapter = session->adapter_refs[i];
        if (adapter != nullptr) {
            adapter_release(adapter);
        }
    }
    session->adapter_count = 0;
}

inline bool image_desc_valid(const AstralImageDesc* image) {
    return image != nullptr && image->size == sizeof(AstralImageDesc) && image->pixels.data != nullptr &&
           image->pixels.len != 0 && image->width != 0 && image->height != 0;
}

inline bool audio_desc_valid(const AstralAudioDesc* audio) {
    return audio != nullptr && audio->size == sizeof(AstralAudioDesc) && audio->samples.data != nullptr &&
           audio->samples.len != 0 && audio->channels != 0 && audio->sample_rate != 0 && audio->frame_count != 0;
}

inline AstralErr session_push_text_chunk(Session* session, uint32_t token_start, uint32_t token_count, uint8_t finalize) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (token_count == 0) {
        return ASTRAL_OK;
    }
    if (session->prompt_chunk_count >= kMaxPromptChunks) {
        return ASTRAL_E_NOMEM;
    }
    PromptChunk& chunk = session->prompt_chunks[session->prompt_chunk_count++];
    chunk.kind = PromptChunkKind::Text;
    chunk.finalize = finalize;
    chunk.token_start = token_start;
    chunk.token_count = token_count;
    chunk.owned_buffer = nullptr;
    chunk.owned_bytes = 0;
    chunk.owned_align = 1;
    return ASTRAL_OK;
}

inline AstralErr session_push_media_chunk(Session* session,
                                          const AstralImageDesc* image,
                                          const AstralAudioDesc* audio,
                                          uint8_t finalize) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (session->prompt_chunk_count >= kMaxPromptChunks) {
        return ASTRAL_E_NOMEM;
    }

    if ((image == nullptr) == (audio == nullptr)) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t idx = session->prompt_chunk_count;
    PromptChunk& chunk = session->prompt_chunks[idx];
    chunk = PromptChunk{};
    chunk.finalize = finalize;

    if (image != nullptr) {
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

    session->prompt_chunk_count = idx + 1u;
    return ASTRAL_OK;
}

inline AstralErr session_feed_prompt_chunks(Session* session, const backend::BackendOps* ops) {
    if (session == nullptr || ops == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (session->prompt_chunk_count == 0) {
        return ops->session_feed(session->backend_session_ctx, session->prompt_tokens, session->prompt_count);
    }

    for (uint32_t i = 0; i < session->prompt_chunk_count; ++i) {
        PromptChunk& chunk = session->prompt_chunks[i];
        switch (chunk.kind) {
            case PromptChunkKind::Text: {
                const uint32_t count = chunk.token_count;
                if (count == 0) {
                    break;
                }
                const int32_t* tokens = session->prompt_tokens + chunk.token_start;
                const AstralErr err = ops->session_feed(session->backend_session_ctx, tokens, count);
                if (err != ASTRAL_OK) {
                    return err;
                }
                break;
            }
            case PromptChunkKind::Image: {
                if (ops->session_feed_image == nullptr) {
                    return ASTRAL_E_UNSUPPORTED;
                }
                const AstralErr err = ops->session_feed_image(session->backend_session_ctx, &chunk.image, chunk.finalize);
                prompt_chunk_release(chunk);
                if (err != ASTRAL_OK) {
                    return err;
                }
                break;
            }
            case PromptChunkKind::Audio: {
                if (ops->session_feed_audio == nullptr) {
                    return ASTRAL_E_UNSUPPORTED;
                }
                const AstralErr err = ops->session_feed_audio(session->backend_session_ctx, &chunk.audio, chunk.finalize);
                prompt_chunk_release(chunk);
                if (err != ASTRAL_OK) {
                    return err;
                }
                break;
            }
        }
    }

    return ASTRAL_OK;
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

    // Allocate per-session scratch backing memory (not a hot path).
    size_t allocator_capacity = kDefaultAllocatorCapacity;
    void* allocator_memory =
        ::astral::core::runtime_session_scratch_acquire(kDefaultAllocatorCapacity, 64, &allocator_capacity);
    if (allocator_memory == nullptr) {
        return ASTRAL_E_NOMEM;
    }

    // Allocate session struct on heap
    // Use malloc instead of new to avoid default constructor issues
    void* session_mem = session_alloc_mem();
    if (session_mem == nullptr) {
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
        return ASTRAL_E_NOMEM;
    }

    // Construct session object (starts lifetime; avoids UB in release builds).
    Session* session = new (session_mem) Session(model, allocator_memory, allocator_capacity, *desc);
    session->worker_id = ::astral::core::runtime_assign_worker_id();

    // Allocate prompt buffer (initial capacity: 8K tokens)
    session->prompt_capacity = 8192;
    session->prompt_tokens = static_cast<int32_t*>(
        session->allocator.alloc(session->prompt_capacity * sizeof(int32_t), alignof(int32_t))
    );
    session->prompt_count = 0;

    if (session->prompt_tokens == nullptr) {
        model_release(model);
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
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
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
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
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
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
            ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
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
    session->token_counts_base = nullptr;
    session->recent_tokens = nullptr;
    session->recent_capacity = 0;
    session->recent_pos = 0;
    session->recent_size = 0;
    session->token_nl = -1;

    // Stop sequences (none by default).
    session->stop_seq_count = 0;
    session->stop_max_len = 0;

    // Logprobs meta stream disabled by default.
    session->logprobs_n = 0;
    session->mirostat_mu = 0.0f;
    session->meta_ring.reset();

    // Adapters.
    session->adapter_count = 0;

    // Slot id.
    session->slot_id = 0;

    if (session->indices_buffer == nullptr || (sample_capacity > 0 && (session->sample_ids == nullptr ||
                                                                      session->sample_logits == nullptr))) {
        if (session->backend_session_ctx != nullptr && model->backend != nullptr) {
            model->backend->ops->session_destroy(session->backend_session_ctx);
            session->backend_session_ctx = nullptr;
        }
        model_release(model);
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
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
        ::astral::core::runtime_session_scratch_release(allocator_memory, allocator_capacity);
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

    session_prompt_chunks_clear(session);

    if (session->toolset != nullptr) {
        toolset_release(session->toolset);
        session->toolset = nullptr;
    }

    session_adapter_refs_clear(session);

    // Release allocator memory
    if (session->allocator_memory != nullptr) {
        ::astral::core::runtime_session_scratch_release(session->allocator_memory, session->allocator_capacity);
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
        if (session->prompt_chunk_count >= kMaxPromptChunks) {
            return ASTRAL_E_NOMEM;
        }
        uint32_t space_available = session->prompt_capacity - session->prompt_count;
        uint32_t n_tokens = 0;
        const uint32_t token_start = session->prompt_count;

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

        const AstralErr chunk_err = session_push_text_chunk(session, token_start, n_tokens, finalize);
        if (chunk_err != ASTRAL_OK) {
            return chunk_err;
        }
    }

    // If finalized, stay in FeedingPrompt state (waiting for decode)
    // Caller must call session_decode() next

    return ASTRAL_OK;
}

AstralErr session_feed_tokens(Session* session, const int32_t* tokens, uint32_t token_count, uint8_t finalize) {
    if (session == nullptr || (token_count != 0 && tokens == nullptr)) {
        return ASTRAL_E_INVALID;
    }

    SessionState state = session->state.load(std::memory_order_acquire);
    if (state != SessionState::Idle && state != SessionState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    if (state == SessionState::Idle) {
        session->state.store(SessionState::FeedingPrompt, std::memory_order_release);
    }

    if (token_count == 0) {
        return ASTRAL_OK;
    }
    if (session->prompt_chunk_count >= kMaxPromptChunks) {
        return ASTRAL_E_NOMEM;
    }
    const uint32_t space = session->prompt_capacity - session->prompt_count;
    if (token_count >= space) {
        return ASTRAL_E_NOMEM;
    }

    const uint32_t token_start = session->prompt_count;
    std::memcpy(session->prompt_tokens + session->prompt_count, tokens, static_cast<size_t>(token_count) * sizeof(int32_t));
    session->prompt_count += token_count;
    return session_push_text_chunk(session, token_start, token_count, finalize);
}

AstralErr session_feed_image(Session* session, const AstralImageDesc* image, uint8_t finalize) {
    if (session == nullptr || !image_desc_valid(image)) {
        return ASTRAL_E_INVALID;
    }

    SessionState state = session->state.load(std::memory_order_acquire);
    if (state != SessionState::Idle && state != SessionState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    if (session->model == nullptr || session->model->backend == nullptr ||
        session->model->backend->ops == nullptr || session->model->backend->ops->session_feed_image == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    if (state == SessionState::Idle) {
        session->state.store(SessionState::FeedingPrompt, std::memory_order_release);
    }

    return session_push_media_chunk(session, image, nullptr, finalize);
}

AstralErr session_feed_audio(Session* session, const AstralAudioDesc* audio, uint8_t finalize) {
    if (session == nullptr || !audio_desc_valid(audio)) {
        return ASTRAL_E_INVALID;
    }

    SessionState state = session->state.load(std::memory_order_acquire);
    if (state != SessionState::Idle && state != SessionState::FeedingPrompt) {
        return ASTRAL_E_STATE;
    }

    if (session->model == nullptr || session->model->backend == nullptr ||
        session->model->backend->ops == nullptr || session->model->backend->ops->session_feed_audio == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    if (state == SessionState::Idle) {
        session->state.store(SessionState::FeedingPrompt, std::memory_order_release);
    }

    return session_push_media_chunk(session, nullptr, audio, finalize);
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
    const AstralErr submit_err = ::astral::core::submit_work_affine(session->worker_id, decode_work, session);
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
        ++spins;
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

    // Not safe to reset concurrently with stream readers; fail fast instead of racing on `pending_*` buffers.
    ScopedAtomicFlagGuard stream_guard(&session->stream_read_lock);
    if (!stream_guard.locked) {
        return ASTRAL_E_STATE;
    }
    ScopedAtomicFlagGuard meta_guard(&session->meta_read_lock);
    if (!meta_guard.locked) {
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

    // Update sampler parameters that are still part of the legacy SessionDesc.
    // Extended sampler controls are configured via `session_set_sampler()` and are preserved across reset.
    session->sampler_cfg.temperature = session->desc.temperature;
    session->sampler_cfg.top_k = session->desc.top_k;
    session->sampler_cfg.top_p = session->desc.top_p;

    // Re-seed RNG. If seed==0, auto-seed each reset.
    session->sampler_cfg.seed =
        session->desc.seed != 0 ? session->desc.seed : static_cast<uint32_t>(get_ticks() & 0xFFFFFFFF);
    session->rng_state = session->sampler_cfg.seed;
    session->mirostat_mu = 0.0f;

    // Clear core state (not thread-safe; caller must ensure no concurrent stream reads).
    session->cancel_requested.store(false, std::memory_order_release);
    session->final_err.store(ASTRAL_OK, std::memory_order_release);
    session->prompt_count = 0;
    session_prompt_chunks_clear(session);
    session->token_ring.reset();
    session->meta_ring.reset();
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

    // Re-apply session-scoped generation controls that are provider-owned.
    if (ops->session_adapter_clear != nullptr && ops->session_adapter_add != nullptr) {
        (void)ops->session_adapter_clear(session->backend_session_ctx);
        for (uint32_t i = 0; i < session->adapter_count; ++i) {
            Adapter* a = session->adapter_refs[i];
            if (a != nullptr && a->backend_adapter_ctx != nullptr) {
                (void)ops->session_adapter_add(session->backend_session_ctx, a->backend_adapter_ctx, session->adapter_scales[i]);
            }
        }
    }

    if (ops->session_set_slot != nullptr && session->slot_id != 0) {
        (void)ops->session_set_slot(session->backend_session_ctx, session->slot_id);
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

    session->sampler_cfg.mirostat = desc->mirostat;
    session->sampler_cfg.mirostat_tau = desc->mirostat_tau;
    session->sampler_cfg.mirostat_eta = desc->mirostat_eta;

    // Keep legacy session desc knobs in sync so reset/reuse preserves the expected values.
    session->desc.temperature = desc->temperature;
    session->desc.top_k = desc->top_k;
    session->desc.top_p = desc->top_p;

    return ensure_penalty_state(session);
}

AstralErr session_penalty_prompt_set_tokens(Session* session, const int32_t* tokens, uint32_t count) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (count > 0 && tokens == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (session->token_counts_base == nullptr) {
        session->token_counts_base = static_cast<uint16_t*>(
            session->allocator.alloc(session->vocab_size * sizeof(uint16_t), alignof(uint16_t))
        );
        if (session->token_counts_base == nullptr) {
            return ASTRAL_E_NOMEM;
        }
    }

    std::memset(session->token_counts_base, 0, session->vocab_size * sizeof(uint16_t));

    for (uint32_t i = 0; i < count; ++i) {
        const int32_t t = tokens[i];
        if (t < 0) {
            continue;
        }
        const uint32_t id = static_cast<uint32_t>(t);
        if (id >= session->vocab_size) {
            continue;
        }
        uint16_t& c = session->token_counts_base[id];
        if (c != 0xFFFFu) {
            c = static_cast<uint16_t>(c + 1u);
        }
    }

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

AstralErr session_stop_set_utf8(Session* session, const AstralSpanU8* seqs, uint32_t count) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (count > 0 && seqs == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (count > 16) {
        return ASTRAL_E_NOMEM;
    }

    AstralErr err = session_stop_clear(session);
    if (err != ASTRAL_OK) {
        return err;
    }
    for (uint32_t i = 0; i < count; ++i) {
        err = session_stop_add_utf8(session, seqs[i]);
        if (err != ASTRAL_OK) {
            return err;
        }
    }
    return ASTRAL_OK;
}

AstralErr session_set_logprobs(Session* session, uint32_t n_probs) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }

    if (n_probs > ASTRAL_LOGPROBS_MAX) {
        n_probs = ASTRAL_LOGPROBS_MAX;
    }
    session->logprobs_n = n_probs;
    session->meta_ring.reset();
    return ASTRAL_OK;
}

AstralErr session_state_size(Session* session, uint64_t* out_bytes) {
    if (session == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_state_size == nullptr || ops->session_state_save == nullptr || ops->session_state_load == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }
    uint64_t backend_bytes = 0;
    const AstralErr err = ops->session_state_size(session->backend_session_ctx, &backend_bytes);
    if (err != ASTRAL_OK) {
        return err;
    }
    *out_bytes = static_cast<uint64_t>(sizeof(SessionStateHeaderV1)) + backend_bytes;
    return ASTRAL_OK;
}

AstralErr session_state_save(Session* session, AstralMutSpanU8 out_buf, uint64_t* out_written) {
    if (session == nullptr || out_written == nullptr || out_buf.data == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_state_size == nullptr || ops->session_state_save == nullptr || ops->session_state_load == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }
    uint64_t backend_bytes = 0;
    AstralErr err = ops->session_state_size(session->backend_session_ctx, &backend_bytes);
    if (err != ASTRAL_OK) {
        return err;
    }

    const uint64_t need = static_cast<uint64_t>(sizeof(SessionStateHeaderV1)) + backend_bytes;
    if (static_cast<uint64_t>(out_buf.len) < need) {
        return ASTRAL_E_NOMEM;
    }

    SessionStateHeaderV1 h{};
    h.magic = kStateMagic;
    h.version = kStateVersion;
    h.header_bytes = static_cast<uint16_t>(sizeof(SessionStateHeaderV1));
    h.backend_bytes = backend_bytes;
    h.rng_state = session->rng_state;
    h.logprobs_n = session->logprobs_n;
    h.slot_id = session->slot_id;
    h.sampler.size = sizeof(AstralSamplerDesc);
    h.sampler.temperature = session->sampler_cfg.temperature;
    h.sampler.top_k = session->sampler_cfg.top_k;
    h.sampler.top_p = session->sampler_cfg.top_p;
    h.sampler.min_p = session->sampler_cfg.min_p;
    h.sampler.typical_p = session->sampler_cfg.typical_p;
    h.sampler.repeat_penalty = session->sampler_cfg.repeat_penalty;
    h.sampler.repeat_last_n = session->sampler_cfg.repeat_last_n;
    h.sampler.penalize_nl = session->sampler_cfg.penalize_nl;
    h.sampler.presence_penalty = session->sampler_cfg.presence_penalty;
    h.sampler.frequency_penalty = session->sampler_cfg.frequency_penalty;
    h.sampler.mirostat = session->sampler_cfg.mirostat;
    h.sampler.mirostat_tau = session->sampler_cfg.mirostat_tau;
    h.sampler.mirostat_eta = session->sampler_cfg.mirostat_eta;
    h.mirostat_mu = session->mirostat_mu;

    std::memcpy(out_buf.data, &h, sizeof(h));

    uint64_t backend_written = 0;
    err = ops->session_state_save(
        session->backend_session_ctx,
        out_buf.data + sizeof(SessionStateHeaderV1),
        static_cast<uint64_t>(out_buf.len) - sizeof(SessionStateHeaderV1),
        &backend_written
    );
    if (err != ASTRAL_OK) {
        return err;
    }

    *out_written = static_cast<uint64_t>(sizeof(SessionStateHeaderV1)) + backend_written;
    return ASTRAL_OK;
}

AstralErr session_state_load(Session* session, AstralSpanU8 state_bytes) {
    if (session == nullptr || state_bytes.data == nullptr || state_bytes.len == 0) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_state_size == nullptr || ops->session_state_save == nullptr || ops->session_state_load == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    SessionStateHeaderV1 h{};
    const uint8_t* backend = nullptr;
    if (session_state_parse_header(state_bytes, &h, &backend)) {
        if (h.sampler.size == sizeof(AstralSamplerDesc)) {
            // Keep scratch capacity invariant.
            if (h.sampler.top_k > 0 && h.sampler.top_k > session->sample_capacity) {
                return ASTRAL_E_INVALID;
            }

            session->sampler_cfg.temperature = h.sampler.temperature;
            session->sampler_cfg.top_k = h.sampler.top_k;
            session->sampler_cfg.top_p = h.sampler.top_p;
            session->sampler_cfg.min_p = h.sampler.min_p;
            session->sampler_cfg.typical_p = h.sampler.typical_p;
            session->sampler_cfg.repeat_penalty = h.sampler.repeat_penalty;
            session->sampler_cfg.repeat_last_n = h.sampler.repeat_last_n;
            session->sampler_cfg.penalize_nl = h.sampler.penalize_nl;
            session->sampler_cfg.presence_penalty = h.sampler.presence_penalty;
            session->sampler_cfg.frequency_penalty = h.sampler.frequency_penalty;
            session->sampler_cfg.mirostat = h.sampler.mirostat;
            session->sampler_cfg.mirostat_tau = h.sampler.mirostat_tau;
            session->sampler_cfg.mirostat_eta = h.sampler.mirostat_eta;

            // Keep legacy SessionDesc knobs in sync.
            session->desc.temperature = h.sampler.temperature;
            session->desc.top_k = h.sampler.top_k;
            session->desc.top_p = h.sampler.top_p;
        }

        session->rng_state = h.rng_state;
        session->mirostat_mu = h.mirostat_mu;

        session->logprobs_n = h.logprobs_n > ASTRAL_LOGPROBS_MAX ? ASTRAL_LOGPROBS_MAX : h.logprobs_n;
        session->meta_ring.reset();

        // Slot is provider-owned; apply before loading state so providers can validate if needed.
        session->slot_id = h.slot_id;
        if (ops->session_set_slot != nullptr && h.slot_id != 0) {
            (void)ops->session_set_slot(session->backend_session_ctx, h.slot_id);
        }

        return ops->session_state_load(session->backend_session_ctx, backend, h.backend_bytes);
    }

    // Legacy provider-only format.
    return ops->session_state_load(session->backend_session_ctx, state_bytes.data, state_bytes.len);
}

AstralErr session_adapters_clear(Session* session) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_adapter_clear == nullptr || ops->session_adapter_add == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    const AstralErr err = ops->session_adapter_clear(session->backend_session_ctx);
    if (err != ASTRAL_OK) {
        return err;
    }

    session_adapter_refs_clear(session);
    return ASTRAL_OK;
}

AstralErr session_adapters_add(Session* session, AstralHandle adapter, float scale) {
    if (session == nullptr || adapter == 0) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->adapter_count >= ASTRAL_SESSION_ADAPTERS_MAX) {
        return ASTRAL_E_NOMEM;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_adapter_clear == nullptr || ops->session_adapter_add == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    auto* a = static_cast<Adapter*>(core::lookup_handle(adapter, core::HandleKind::Adapter));
    if (a == nullptr || a->model != session->model || a->backend_adapter_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    adapter_retain(a);

    const AstralErr err = ops->session_adapter_add(session->backend_session_ctx, a->backend_adapter_ctx, scale);
    if (err != ASTRAL_OK) {
        adapter_release(a);
        return err;
    }

    const uint32_t idx = session->adapter_count++;
    session->adapter_handles[idx] = adapter;
    session->adapter_refs[idx] = a;
    session->adapter_scales[idx] = scale;
    return ASTRAL_OK;
}

AstralErr session_adapters_count(Session* session, uint32_t* out_count) {
    if (session == nullptr || out_count == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_count = session->adapter_count;
    return ASTRAL_OK;
}

AstralErr session_adapters_get(Session* session, uint32_t index, AstralHandle* out_adapter, float* out_scale) {
    if (session == nullptr || out_adapter == nullptr || out_scale == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (index >= session->adapter_count) {
        return ASTRAL_E_NOT_FOUND;
    }
    *out_adapter = session->adapter_handles[index];
    *out_scale = session->adapter_scales[index];
    return ASTRAL_OK;
}

AstralErr session_adapters_set_scale(Session* session, uint32_t index, float scale) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (index >= session->adapter_count) {
        return ASTRAL_E_NOT_FOUND;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_adapter_clear == nullptr || ops->session_adapter_add == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    AstralErr err = ops->session_adapter_clear(session->backend_session_ctx);
    if (err != ASTRAL_OK) {
        return err;
    }

    for (uint32_t i = 0; i < session->adapter_count; ++i) {
        Adapter* a = session->adapter_refs[i];
        if (a == nullptr || a->model != session->model || a->backend_adapter_ctx == nullptr) {
            return ASTRAL_E_INVALID;
        }
        const float next_scale = i == index ? scale : session->adapter_scales[i];
        err = ops->session_adapter_add(session->backend_session_ctx, a->backend_adapter_ctx, next_scale);
        if (err != ASTRAL_OK) {
            return err;
        }
    }

    session->adapter_scales[index] = scale;
    return ASTRAL_OK;
}

AstralErr session_set_grammar_gbnf(Session* session, AstralSpanU8 gbnf, AstralSpanU8 root) {
    if (session == nullptr || gbnf.data == nullptr || gbnf.len == 0) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_grammar_set_gbnf == nullptr || ops->session_grammar_clear == nullptr ||
        ops->session_apply_grammar == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    if (root.data == nullptr || root.len == 0) {
        static constexpr uint8_t kRoot[] = {'r','o','o','t'};
        root.data = kRoot;
        root.len = 4;
    }
    return ops->session_grammar_set_gbnf(session->backend_session_ctx, gbnf, root);
}

AstralErr session_set_grammar_json_schema(Session* session, AstralSpanU8 json_schema) {
    if (session == nullptr || json_schema.data == nullptr || json_schema.len == 0) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_grammar_set_json_schema == nullptr || ops->session_grammar_clear == nullptr ||
        ops->session_apply_grammar == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }
    return ops->session_grammar_set_json_schema(session->backend_session_ctx, json_schema);
}

AstralErr session_clear_grammar(Session* session) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_grammar_clear == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }
    return ops->session_grammar_clear(session->backend_session_ctx);
}

AstralErr session_set_toolset(Session* session, Toolset* toolset, AstralToolChoiceMode choice_mode) {
    if (session == nullptr || toolset == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (choice_mode != ASTRAL_TOOL_CHOICE_AUTO && choice_mode != ASTRAL_TOOL_CHOICE_REQUIRED &&
        choice_mode != ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }

    toolset_retain(toolset);
    Toolset* old = session->toolset;
    session->toolset = toolset;
    session->tool_choice_mode = choice_mode;
    if (old != nullptr) {
        toolset_release(old);
    }
    return ASTRAL_OK;
}

AstralErr session_clear_toolset(Session* session) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    Toolset* old = session->toolset;
    session->toolset = nullptr;
    session->tool_choice_mode = ASTRAL_TOOL_CHOICE_AUTO;
    if (old != nullptr) {
        toolset_release(old);
    }
    return ASTRAL_OK;
}

AstralErr session_set_slot(Session* session, uint32_t slot_id) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Decoding) {
        return ASTRAL_E_STATE;
    }
    if (session->model == nullptr || session->model->backend == nullptr || session->model->backend->ops == nullptr) {
        return ASTRAL_E_STATE;
    }
    const backend::BackendOps* ops = session->model->backend->ops;
    if (ops->session_set_slot == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }
    const AstralErr err = ops->session_set_slot(session->backend_session_ctx, slot_id);
    if (err == ASTRAL_OK) {
        session->slot_id = slot_id;
    }
    return err;
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
    ASTRAL_ZONE_N("astral.decode_loop");
    if (session == nullptr) {
        return;
    }

    // Affinity hygiene: decode must run on the session's assigned worker when threading is enabled.
#if ASTRAL_ENABLE_THREADS
    if (core::runtime_on_worker_thread() && core::runtime_worker_id() != session->worker_id) {
        session->t_end_ticks = get_ticks();
        session->final_err.store(static_cast<int32_t>(ASTRAL_E_STATE), std::memory_order_release);
        session->state.store(SessionState::Failed, std::memory_order_release);
        platform::cpu_signal_event();
        return;
    }
    core::runtime_worker_scratch_reset();
#endif

    Model* model = session->model;
    const backend::BackendProvider* backend = model->backend;
    const backend::BackendOps* ops = backend->ops;
    void* backend_session_ctx = session->backend_session_ctx;

    uint32_t max_tokens = session->desc.max_tokens;

    SessionState final_state = SessionState::Completed;
    AstralErr final_err = ASTRAL_OK;

    // Feed prompt tokens
    AstralErr err = ASTRAL_OK;
    {
        ASTRAL_ZONE_N("astral.session_feed");
        err = session_feed_prompt_chunks(session, ops);
    }

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

        {
            ASTRAL_ZONE_N("astral.generation_loop");
            for (uint32_t i = 0; i < max_tokens; ++i) {
                if (session->cancel_requested.load(std::memory_order_acquire)) {
                    final_state = SessionState::Canceled;
                    final_err = ASTRAL_E_CANCELED;
                    break;
                }

                backend::BackendLogitsView logits_view{};
                ASTRAL_ZONE_MICRO_N("astral.backend.session_logits");
                err = ops->session_logits(backend_session_ctx, &logits_view);
                if (err != ASTRAL_OK) {
                    final_state = SessionState::Failed;
                    final_err = err;
                    break;
                }

                SamplerMeta meta{};
                SamplerMeta* meta_ptr = (session->logprobs_n > 0) ? &meta : nullptr;
                const int32_t next_token = static_cast<int32_t>(sample_token(
                    logits_view.logits,
                    logits_view.vocab_size,
                    session->sampler_cfg,
                    &session->rng_state,
                    session->token_counts,
                    session->token_counts_base,
                    session->token_nl,
                    &session->mirostat_mu,
                    session->logprobs_n,
                    meta_ptr,
                    session->sample_ids,
                    session->sample_logits,
                    session->sample_capacity,
                    backend_session_ctx,
                    ops->session_apply_grammar,
                    session->indices_buffer));

                ASTRAL_ZONE_MICRO_N("astral.backend.session_accept");
                err = ops->session_accept(backend_session_ctx, next_token);
                if (err != ASTRAL_OK) {
                    final_state = SessionState::Failed;
                    final_err = err;
                    break;
                }

                if (next_token >= 0) {
                    penalty_state_push(session, static_cast<uint32_t>(next_token));
                }

                if (meta_ptr != nullptr) {
                    AstralTokenMeta ev{};
                    ev.token_id = meta_ptr->token_id;
                    ev.logprob = meta_ptr->logprob;
                    ev.top_n = meta_ptr->top_n;
                    for (uint32_t j = 0; j < meta_ptr->top_n && j < ASTRAL_LOGPROBS_MAX; ++j) {
                        ev.top_token_ids[j] = meta_ptr->top_ids[j];
                        ev.top_logprobs[j] = meta_ptr->top_logprobs[j];
                    }
                    (void)session->meta_ring.push(ev);
                }

                if (session->token_eos >= 0 && next_token == session->token_eos) {
                    break;
                }

                session->total_tokens++;
                if (session->total_tokens == 1) {
                    session->t_first_token_ticks = get_ticks();
                }

                uint32_t stop_len = 0;
                const bool stop_hit = session_stop_check(session, next_token, stop_tail, &stop_tail_len, &stop_len);

                if (session->desc.stream_enabled) {
                    concurrency::StreamToken stream_token;
                    stream_token.token_id = static_cast<uint32_t>(next_token);

                    AstralMutSpanU8 out_buf;
                    out_buf.data = stream_token.utf8_data;
                    out_buf.len = sizeof(stream_token.utf8_data);

                    uint32_t utf8_len = 0;
                    ASTRAL_ZONE_MICRO_N("astral.backend.detokenize");
                    err = ops->detokenize(model->backend_model_ctx, &next_token, 1, out_buf, &utf8_len);
                    if (err != ASTRAL_OK) {
                        final_state = SessionState::Failed;
                        final_err = err;
                        break;
                    }

                    stream_token.utf8_len = static_cast<uint16_t>(utf8_len);

                    if (!session_hold_push(session, stream_token, hold, &hold_head, &hold_count, kHoldCap, stop_hold)) {
                        final_state = SessionState::Canceled;
                        final_err = ASTRAL_E_CANCELED;
                        break;
                    }
                }

                if (stop_hit) {
                    if (session->desc.stream_enabled && stop_hold > 0 && stop_len > 0) {
                        if (stop_len >= hold_count) {
                            hold_count = 0;
                        } else {
                            hold_count -= stop_len;
                        }
                        if (!session_hold_flush_all(session, hold, &hold_head, &hold_count, kHoldCap)) {
                            final_state = SessionState::Canceled;
                            final_err = ASTRAL_E_CANCELED;
                        }
                    }
                    break;
                }
            }
        }

        if (final_state == SessionState::Completed && session->desc.stream_enabled && hold_count > 0) {
            if (!session_hold_flush_all(session, hold, &hold_head, &hold_count, kHoldCap)) {
                final_state = SessionState::Canceled;
                final_err = ASTRAL_E_CANCELED;
            }
        }
    } // feed_ok

    // Transition to terminal state.
    session->t_end_ticks = get_ticks();

    // Emit terminal session timing plots when Tracy is enabled.
#if ASTRAL_ENABLE_TRACY
    if (session->t_end_ticks > session->t_start_ticks) {
        const uint64_t dt_ticks = session->t_end_ticks - session->t_start_ticks;
        const uint64_t dt_ns = platform::ticks_to_ns(dt_ticks);
        if (dt_ns > 0) {
            const float dt_s = static_cast<float>(static_cast<double>(dt_ns) * 1e-9);
            ASTRAL_PLOT("astral.tokens_total", static_cast<double>(session->total_tokens));
            ASTRAL_PLOT("astral.tokens_per_s",
                        dt_s > 0.0f ? static_cast<double>(static_cast<float>(session->total_tokens) / dt_s) : 0.0);
        }
    }
    if (session->t_first_token_ticks > session->t_start_ticks) {
        const uint64_t ttft_ns = platform::ticks_to_ns(session->t_first_token_ticks - session->t_start_ticks);
        ASTRAL_PLOT("astral.ttft_ms", static_cast<double>(ttft_ns) / 1e6);
    }
#endif

    session->final_err.store(static_cast<int32_t>(final_err), std::memory_order_release);
    session->state.store(final_state, std::memory_order_release);
    platform::cpu_signal_event();
}

int32_t stream_read(Session* session, AstralMutSpanU8 out_buf, uint32_t timeout_ms) {
    ASTRAL_ZONE_N("astral.stream_read");
    // Validate session
    if (session == nullptr || out_buf.data == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (out_buf.len == 0) {
        return ASTRAL_E_INVALID;
    }

    // Single-consumer enforcement (callback-safe / fail-fast).
    ScopedAtomicFlagGuard guard(&session->stream_read_lock);
    if (!guard.locked) {
        return ASTRAL_E_STATE;
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

    int32_t drained = session_stream_drain(session, out_buf);
    if (drained > 0) {
        return drained;
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
            drained = session_stream_drain(session, out_buf);
            if (drained > 0) {
                return drained;
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
            ++spins;
        }
    }

    // Non-blocking and no data available
    return ASTRAL_E_TIMEOUT;
}

int32_t stream_read_meta(Session* session, AstralTokenMeta* out_events, uint32_t capacity, uint32_t timeout_ms) {
    if (session == nullptr || out_events == nullptr || capacity == 0) {
        return ASTRAL_E_INVALID;
    }

    ScopedAtomicFlagGuard guard(&session->meta_read_lock);
    if (!guard.locked) {
        return ASTRAL_E_STATE;
    }

    size_t n = session->meta_ring.pop_batch(out_events, capacity);
    if (n != 0) {
      return static_cast<int32_t>(n);
    }

    SessionState state = session->state.load(std::memory_order_acquire);
    if (state == SessionState::Completed || state == SessionState::Canceled || state == SessionState::Failed) {
        return 0;
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
      n = session->meta_ring.pop_batch(out_events, capacity);
      if (n != 0) {
        return static_cast<int32_t>(n);
      }

        state = session->state.load(std::memory_order_acquire);
        if (state == SessionState::Completed || state == SessionState::Canceled || state == SessionState::Failed) {
            return 0;
        }

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
        ++spins;
    }
}

} // namespace astral::inference
