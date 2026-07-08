#include "conversation.hpp"

#include "../core/runtime_alloc.hpp"
#include "../platform/time.h"

#include <cstring>

namespace astral::inference {

Conversation::Conversation(Model* model_,
                           void* allocator_memory_,
                           size_t allocator_capacity_,
                           const AstralConvDesc& desc_) noexcept
    : handle(0)
    , model(model_)
    , desc(desc_)
    , slot_id(0xFFFFFFFFu)
    , exec_refs(0)
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
    , prompt_off(0)
    , prompt_chunks{}
    , prompt_chunk_count(0)
    , prompt_chunk_index(0)
    , prompt_chunk_token_off(0)
    , state(ConvState::Idle)
    , cancel_requested(false)
    , final_err(ASTRAL_OK)
    , vocab_size(0)
    , ctx_size(0)
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
    , stop_seq_lens{}
    , stop_seq_tokens{}
    , stop_max_len(0)
    , stop_tail{}
    , stop_tail_len(0)
    , hold{}
    , hold_head(0)
    , hold_count(0)
    , pending_token(0)
    , pending_token_valid(0)
    , pending_emit_token(0)
    , pending_emit_valid(0)
    , needs_slot_reset(0)
    , grammar_kind(0)
    , grammar_dirty(0)
    , grammar_applied(0)
    , _padding1(0)
    , grammar_gbnf(nullptr)
    , grammar_gbnf_len(0)
    , grammar_root(nullptr)
    , grammar_root_len(0)
    , grammar_json(nullptr)
    , grammar_json_len(0)
    , toolset(nullptr)
    , tool_choice_mode(ASTRAL_TOOL_CHOICE_AUTO)
    , pending_emit_stream{} {}

Conversation::~Conversation() noexcept {
  if (pending_emit_spill != nullptr) {
    core::runtime_free_array(pending_emit_spill, pending_emit_spill_capacity);
  }
}

} // namespace astral::inference
