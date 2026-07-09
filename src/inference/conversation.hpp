#pragma once

#include "../../include/astral_rt.h"

#include "../concurrency/spsc_ring.hpp"
#include "../memory/frame_allocator.hpp"

#include "model.hpp"
#include "sampler.hpp"
#include "prompt_chunks.hpp"
#include "tooling.hpp"

#include <atomic>
#include <cstdint>

namespace astral::inference {

enum class ConvState : uint32_t {
    Idle = 0,
    FeedingPrompt = 1,
    Decoding = 2,
    Completed = 3,
    Canceled = 4,
    Failed = 5,
};

struct alignas(64) PublishedConversationStats {
  std::atomic<uint32_t> prompt_tokens{0};
  std::atomic<uint32_t> kv_tokens{0};
  std::atomic<uint64_t> generated_tokens{0};
  std::atomic<uint64_t> t_start_ticks{0};
  std::atomic<uint64_t> t_first_token_ticks{0};
  std::atomic<uint64_t> t_end_ticks{0};
};

struct Conversation {
    Conversation(Model* model_,
                 void* allocator_memory_,
                 size_t allocator_capacity_,
                 const AstralConvDesc& desc_) noexcept;
    ~Conversation() noexcept;

    AstralHandle handle;

    Model* model;
    AstralConvDesc desc;

    // Assigned by the model executor (slot id / seq id).
    std::atomic<uint32_t> slot_id;
    std::atomic<bool> epoch_reclaim_ready{false};

    memory::FrameAllocator allocator;
    void* allocator_memory;
    size_t allocator_capacity;

    // Stream (bytes-first) + meta side-channel.
    concurrency::SpscRing<concurrency::StreamToken, 256> token_ring;
    uint8_t pending_utf8[32];
    uint8_t pending_len;
    uint8_t pending_off;

    std::atomic_flag stream_read_lock = ATOMIC_FLAG_INIT;
    std::atomic_flag meta_read_lock = ATOMIC_FLAG_INIT;

    // Prompt buffer (tokenized prompt).
    int32_t* prompt_tokens;
    uint32_t prompt_count;
    uint32_t prompt_capacity;
    uint32_t prompt_off; // how many prompt tokens have been evaluated
    PromptChunk prompt_chunks[kMaxPromptChunks];
    uint32_t prompt_chunk_count;
    uint32_t prompt_chunk_index;
    uint32_t prompt_chunk_token_off;

    std::atomic<ConvState> state;
    std::atomic<bool> cancel_requested;
    std::atomic<int32_t> final_err;

    uint32_t vocab_size;
    uint32_t ctx_size;

    // Sampling scratch (bounded; per-conversation).
    uint32_t* sample_ids;
    float* sample_logits;
    uint32_t sample_capacity;
    SamplerConfig sampler_cfg;
    uint32_t rng_state;
    float mirostat_mu;

    // Logprobs side-channel (optional).
    uint32_t logprobs_n;
    concurrency::SpscRing<AstralTokenMeta, 256> meta_ring;

    // Penalties.
    uint16_t* token_counts;
    uint16_t* token_counts_base;
    uint32_t* recent_tokens;
    uint32_t recent_capacity;
    uint32_t recent_pos;
    uint32_t recent_size;
    int32_t token_nl;

    // Statistics.
    uint64_t total_tokens;
    uint64_t t_start_ticks;
    uint64_t t_first_token_ticks;
    uint64_t t_end_ticks;

    // Observer-facing snapshot. The executor remains the sole owner of the working counters above.
    PublishedConversationStats published_stats;

    // Position within slot.
    uint32_t n_past;

    int32_t token_bos;
    int32_t token_eos;

    // Stop sequences.
    uint32_t stop_seq_count;
    uint8_t stop_seq_lens[16];
    int32_t stop_seq_tokens[16][32];
    uint32_t stop_max_len;

    // Stop matching tail (persistent; avoids reallocation).
    int32_t stop_tail[32];
    uint32_t stop_tail_len;

    // Stop suppression hold buffer (persistent).
    int32_t hold[64];
    uint32_t hold_head;
    uint32_t hold_count;

    // Pipeline token:
    // - `pending_token_valid`: token to be evaluated in the next backend batch.
    // - `pending_emit_valid`: token sampled, but not yet emitted to stream due to backpressure.
    int32_t pending_token;
    uint8_t pending_token_valid;
    int32_t pending_emit_token;
    uint8_t pending_emit_valid;
    uint8_t* pending_emit_spill{nullptr};
    uint32_t pending_emit_spill_capacity{0};
    uint32_t pending_emit_spill_len{0};
    uint32_t pending_emit_spill_off{0};
    uint8_t completion_pending{0};

    // Executor-side control flags (set by control thread, consumed by executor thread).
    // These must not require any backend calls from the control thread.
    uint8_t needs_slot_reset;

    // Grammar binding (per-slot, provider-owned).
    // When `grammar_dirty != 0`, the executor applies the current config to the provider slot
    // before advancing decoding.
    uint8_t grammar_kind; // 0=none, 1=GBNF, 2=JSON schema
    uint8_t grammar_dirty;
    uint8_t grammar_applied;
    uint8_t _padding1;

    uint8_t* grammar_gbnf;
    uint32_t grammar_gbnf_len;
    uint8_t* grammar_root;
    uint32_t grammar_root_len;
    uint8_t* grammar_json;
    uint32_t grammar_json_len;

    // Structured output toolset (setup-time only; executor applies provider grammar separately).
    Toolset* toolset;
    AstralToolChoiceMode tool_choice_mode;

    // Cached token emission to avoid repeated detokenize when under backpressure.
    concurrency::StreamToken pending_emit_stream;
};

inline AstralSessionState conv_state_to_public(ConvState s) {
    switch (s) {
        case ConvState::Idle: return ASTRAL_SESSION_IDLE;
        case ConvState::FeedingPrompt: return ASTRAL_SESSION_FEEDING_PROMPT;
        case ConvState::Decoding: return ASTRAL_SESSION_DECODING;
        case ConvState::Completed: return ASTRAL_SESSION_COMPLETED;
        case ConvState::Canceled: return ASTRAL_SESSION_CANCELED;
        case ConvState::Failed: return ASTRAL_SESSION_FAILED;
    }
    return ASTRAL_SESSION_FAILED;
}

} // namespace astral::inference
