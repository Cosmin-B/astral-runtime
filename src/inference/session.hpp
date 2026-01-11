#pragma once

#include "../../include/astral_rt.h"
#include "../memory/frame_allocator.hpp"
#include "../concurrency/spsc_ring.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "adapter.hpp"
#include "tooling.hpp"
#include "prompt_chunks.hpp"
#include <atomic>
#include <cstdint>

namespace astral::inference {

/// Session state machine.
///
/// State transitions:
/// - Idle -> FeedingPrompt: astral_session_feed() called
/// - FeedingPrompt -> FeedingPrompt: astral_session_feed(finalize=0)
/// - FeedingPrompt -> Decoding: astral_session_feed(finalize=1) + astral_session_decode()
/// - Decoding -> Completed: Max tokens reached or EOS token generated
/// - Completed -> Idle: Session reset/reuse
///
/// Thread-safety: Atomic state transitions (safe to read from multiple threads)
enum class SessionState : uint32_t {
    Idle = 0,
    FeedingPrompt = 1,
    Decoding = 2,
    Completed = 3, // Finished successfully
    Canceled = 4,  // Finished due to cancel
    Failed = 5,    // Finished due to error
};

/// Inference session handle.
///
/// Represents a single inference session (one prompt + generation).
///
/// Memory management:
/// - allocator: Per-session FrameAllocator for prompt tokens, logits, sampling buffers
/// - token_ring: SPSC ring for streaming tokens to consumer
///
/// Lifecycle:
/// 1. Create: session_create()
/// 2. Feed prompt: session_feed() (may be called multiple times)
/// 3. Decode: session_decode() (submits work to thread pool)
/// 4. Stream: stream_read() (consumer polls for tokens)
/// 5. Destroy: session_destroy()
///
/// Thread-safety:
/// - Single-threaded API: feed/decode must be called from same thread
/// - stream_read() may be called from different thread (consumer thread)
/// - Internal state is atomic where needed
struct Session {
    Session(Model* model_,
            void* allocator_memory_,
            size_t allocator_capacity_,
            const AstralSessionDesc& desc_) noexcept;

    AstralHandle handle; // Public handle (type/index/generation)

    // Model reference
    Model* model;
    void* backend_session_ctx; // Opaque backend session context

    // Session configuration
    AstralSessionDesc desc;

    // Per-session allocator (pre-committed memory)
    memory::FrameAllocator allocator;
    void* allocator_memory;      // Base address for allocator (owned by session)
    size_t allocator_capacity;   // Total capacity in bytes

    // Token streaming ring (decode thread -> consumer)
    concurrency::SpscRing<concurrency::StreamToken, 256> token_ring;
    // If `astral_stream_read()` is called with a small buffer, a single token can be split
    // across multiple reads without losing bytes.
    uint8_t pending_utf8[32];
    uint8_t pending_len;
    uint8_t pending_off;

    // Prompt buffer (tokenized prompt)
    int32_t* prompt_tokens;
    uint32_t prompt_count;
    uint32_t prompt_capacity;
    PromptChunk prompt_chunks[kMaxPromptChunks];
    uint32_t prompt_chunk_count;
    uint32_t prompt_chunk_index;
    uint32_t prompt_chunk_token_off;

    // Decoding state
    std::atomic<SessionState> state;
    std::atomic<bool> cancel_requested;
    std::atomic<int32_t> final_err; // Valid when in a terminal state

    // Sampling buffers (allocated from allocator)
    uint32_t vocab_size;         // Cached vocabulary size
    uint32_t ctx_size;           // Cached context size (tokens)
    uint32_t* indices_buffer;    // Size: vocab_size (used for full-vocab top-p fallback)
    uint32_t* sample_ids;        // Size: sample_capacity (top-k candidates)
    float* sample_logits;        // Size: sample_capacity (top-k candidate logits / weights)
    uint32_t sample_capacity;    // Candidate capacity for top-k sampling
    SamplerConfig sampler_cfg;
    uint32_t rng_state;          // Mutable RNG state for sampling
    float mirostat_mu;           // Mirostat mutable state (0 if unused)

    // Logprobs side-channel (optional).
    uint32_t logprobs_n;         // 0 = disabled
    concurrency::SpscRing<AstralTokenMeta, 256> meta_ring;

    // Sampling penalties state (allocated lazily; not a hot-path allocation)
    uint16_t* token_counts;      // Size: vocab_size (counts within repeat_last_n window + optional base)
    uint16_t* token_counts_base; // Size: vocab_size (fixed counts, e.g. penalty_prompt tokens)
    uint32_t* recent_tokens;     // Size: recent_capacity (ring of last N tokens)
    uint32_t recent_capacity;    // Effective repeat_last_n
    uint32_t recent_pos;         // Ring write position
    uint32_t recent_size;        // Number of valid entries (<= recent_capacity)
    int32_t token_nl;            // Token id for "\\n" if resolvable, else -1

    // Statistics
    uint64_t total_tokens;       // Total tokens generated
    uint64_t t_start_ticks;       // Start time (ticks)
    uint64_t t_first_token_ticks; // Time to first token (ticks)
    uint64_t t_end_ticks;         // End time (ticks; 0 if still running)

    // Current generation position
    uint32_t n_past;             // Number of tokens processed so far

    // Special tokens (cached from backend)
    int32_t token_bos;
    int32_t token_eos;

    // Stop sequences (tokenized; fixed-size storage for embedded-friendliness).
    uint32_t stop_seq_count;
    uint8_t stop_seq_lens[16];
    int32_t stop_seq_tokens[16][32];
    uint32_t stop_max_len; // max(stop_seq_lens)

    // Adapters (fixed-size list; session-scoped).
    uint32_t adapter_count;
    AstralHandle adapter_handles[ASTRAL_SESSION_ADAPTERS_MAX];
    Adapter* adapter_refs[ASTRAL_SESSION_ADAPTERS_MAX];
    float adapter_scales[ASTRAL_SESSION_ADAPTERS_MAX];

    // Structured output toolset (setup-time only; decode reads provider grammar state).
    Toolset* toolset;
    AstralToolChoiceMode tool_choice_mode;

    // Slot id (for providers that support parallel slots).
    uint32_t slot_id;

    // Worker affinity (one session ↔ one worker thread).
    uint32_t worker_id;

    // Streaming consumer guards.
    // `SpscRing` is single-consumer; these flags make misuse fail fast instead of racing.
    std::atomic_flag stream_read_lock = ATOMIC_FLAG_INIT;
    std::atomic_flag meta_read_lock = ATOMIC_FLAG_INIT;
};

/// Create an inference session.
///
/// @param desc Session configuration (must not be NULL)
/// @param out_session Output: session handle (must not be NULL)
/// @return ASTRAL_OK on success, error code on failure
///
/// Error codes:
/// - ASTRAL_E_INVALID: desc or out_session is NULL, or invalid model handle
/// - ASTRAL_E_NOMEM: Out of memory
///
/// Thread-safety: Safe to call from multiple threads
AstralErr session_create(const AstralSessionDesc* desc, Session** out_session);

/// Destroy a session.
///
/// Frees all session resources (allocator, token ring, buffers).
/// Session must not be in use (no active decode).
///
/// @param session Session handle (may be NULL; no-op if NULL)
///
/// Thread-safety: Must not be in use
void session_destroy(Session* session);

/// Feed a prompt chunk.
///
/// Tokenizes UTF-8 text and appends to prompt buffer.
/// Call multiple times for long prompts, setting finalize=1 on last chunk.
///
/// @param session Session handle (must not be NULL)
/// @param prompt_chunk UTF-8 prompt data (data may be NULL if len is 0)
/// @param finalize 1 if this is the last chunk, 0 otherwise
/// @return ASTRAL_OK on success, error code on failure
///
/// Error codes:
/// - ASTRAL_E_INVALID: session is NULL
/// - ASTRAL_E_STATE: Session not in Idle or FeedingPrompt state
/// - ASTRAL_E_NOMEM: Prompt buffer full (too many tokens)
///
/// Thread-safety: Not thread-safe (single-threaded access per session)
AstralErr session_feed(Session* session, AstralSpanU8 prompt_chunk, uint8_t finalize);
AstralErr session_feed_tokens(Session* session, const int32_t* tokens, uint32_t token_count, uint8_t finalize);
/// Feed an image prompt chunk.
AstralErr session_feed_image(Session* session, const AstralImageDesc* image, uint8_t finalize);
/// Feed an audio prompt chunk.
AstralErr session_feed_audio(Session* session, const AstralAudioDesc* audio, uint8_t finalize);

/// Start decoding (non-blocking).
///
/// Submits decode work to thread pool; returns immediately.
/// Decode thread will:
/// 1. Process prompt tokens in batches (n_batch)
/// 2. Generate tokens via decode -> sample -> detokenize loop
/// 3. Write tokens to token_ring
/// 4. Stop when max_tokens reached or EOS token generated
///
/// @param session Session handle (must not be NULL)
/// @return ASTRAL_OK on success, error code on failure
///
/// Error codes:
/// - ASTRAL_E_INVALID: session is NULL
/// - ASTRAL_E_STATE: Session not in FeedingPrompt state (must feed prompt first)
///
/// Thread-safety: Not thread-safe (single-threaded access per session)
AstralErr session_decode(Session* session);

/// Request cancellation for an in-flight decode.
///
/// Thread-safety: Safe to call from any thread.
AstralErr session_cancel(Session* session);

/// Query current session state (atomic read).
///
/// Thread-safety: Safe to call from any thread.
AstralErr session_state(Session* session, SessionState* out_state);

/// Wait for session completion.
///
/// @param timeout_ms 0 for non-blocking poll; otherwise timeout in milliseconds.
/// @return ASTRAL_OK if completed, ASTRAL_E_TIMEOUT if deadline exceeded.
///
/// Thread-safety: Safe to call from any thread.
AstralErr session_wait(Session* session, uint32_t timeout_ms);

/// Populate session statistics.
///
/// Thread-safety: Safe to call from any thread.
AstralErr session_stats(Session* session, AstralStats* out_stats);

/// Reset a session for reuse.
///
/// Clears prompt buffer, streaming ring, cancellation flag, and backend KV/cache.
///
/// Preconditions:
/// - Session must not be decoding (call cancel + wait first).
/// - Not thread-safe; must not be called concurrently with stream_read().
///
/// @param session Session handle (must not be NULL)
/// @param new_desc Optional new session parameters (may be NULL to keep current parameters).
///                If provided, new_desc->model must match the session model handle.
/// @return ASTRAL_OK on success, error code on failure
AstralErr session_reset(Session* session, const AstralSessionDesc* new_desc);

/// Configure sampler controls for a session (extended v0.2 surface).
AstralErr session_set_sampler(Session* session, const AstralSamplerDesc* desc);

/// Configure penalty prompt (token span) used by penalties.
AstralErr session_penalty_prompt_set_tokens(Session* session, const int32_t* tokens, uint32_t count);

/// Stop sequences: clear all.
AstralErr session_stop_clear(Session* session);

/// Stop sequences: add UTF-8 text (tokenized once).
AstralErr session_stop_add_utf8(Session* session, AstralSpanU8 utf8);
/// Stop sequences: bulk set (clear + add each).
AstralErr session_stop_set_utf8(Session* session, const AstralSpanU8* seqs, uint32_t count);

/// Configure logprobs side-channel (0 disables; clamped).
AstralErr session_set_logprobs(Session* session, uint32_t n_probs);

/// Read meta events from the side-channel stream.
int32_t stream_read_meta(Session* session, AstralTokenMeta* out_events, uint32_t capacity, uint32_t timeout_ms);

/// Session state size/save/load (KV cache).
AstralErr session_state_size(Session* session, uint64_t* out_bytes);
AstralErr session_state_save(Session* session, AstralMutSpanU8 out_buf, uint64_t* out_written);
AstralErr session_state_load(Session* session, AstralSpanU8 state_bytes);

/// Attach adapters to a session.
AstralErr session_adapters_clear(Session* session);
AstralErr session_adapters_add(Session* session, AstralHandle adapter, float scale);
AstralErr session_adapters_count(Session* session, uint32_t* out_count);
AstralErr session_adapters_get(Session* session, uint32_t index, AstralHandle* out_adapter, float* out_scale);
AstralErr session_adapters_set_scale(Session* session, uint32_t index, float scale);

/// Configure grammar-constrained decoding.
AstralErr session_set_grammar_gbnf(Session* session, AstralSpanU8 gbnf, AstralSpanU8 root);
AstralErr session_set_grammar_json_schema(Session* session, AstralSpanU8 json_schema);
AstralErr session_clear_grammar(Session* session);
AstralErr session_set_toolset(Session* session, Toolset* toolset, AstralToolChoiceMode choice_mode);
AstralErr session_clear_toolset(Session* session);

/// Set the session slot/sequence id.
AstralErr session_set_slot(Session* session, uint32_t slot_id);

/// Read tokens from stream.
///
/// Polls token_ring for available tokens, blocks up to timeout_ms if empty.
///
/// @param session Session handle (must not be NULL)
/// @param out_buf Output buffer for UTF-8 token data (data must not be NULL)
/// @param timeout_ms Timeout in milliseconds (0 = non-blocking)
/// @return Bytes written to out_buf (>= 0), or error code (< 0)
///
/// Return values:
/// - > 0: Bytes written to out_buf
/// - 0: End-of-stream (session is terminal and no buffered data remains)
/// - ASTRAL_E_INVALID: session or out_buf.data is NULL
/// - ASTRAL_E_TIMEOUT: No data available within timeout (including timeout_ms == 0 poll)
///
/// Thread-safety:
/// - Safe for a single consumer thread calling stream_read concurrently with the decode worker.
/// - Not safe to call from multiple consumer threads concurrently.
int32_t stream_read(Session* session, AstralMutSpanU8 out_buf, uint32_t timeout_ms);

/// Decode loop (internal function, called by thread pool worker).
///
/// Runs the full decode loop:
/// 1. Feed prompt tokens in batches
/// 2. Generate tokens via decode -> sample -> detokenize
/// 3. Write to token_ring
/// 4. Repeat until max_tokens or EOS
///
/// @param session Session handle (must not be NULL)
///
/// Thread-safety: Must be called from single worker thread
void decode_loop(Session* session);

} // namespace astral::inference
