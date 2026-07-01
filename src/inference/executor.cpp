#include "executor.hpp"

#include "../core/runtime_alloc.hpp"
#include "../core/runtime_state.hpp"
#include "../core/work_queue.hpp"
#include "../platform/atomics.h"
#include "../platform/time.h"

#include <cstring>

namespace astral::inference {

namespace {

inline uint64_t ticks_now() {
  return astral::platform::ticks_now();
}

inline void wait_hint(uint32_t& spins) {
  if (spins < 64) {
    astral::platform::cpu_pause();
  } else {
    astral::platform::cpu_wait_for_event();
  }
  if (spins < 1024) {
    ++spins;
  }
}

inline void signal() {
  astral::platform::cpu_signal_event();
}

inline void slot_table_lock(ModelExecutor* ex) {
  uint32_t spins = 0;
  while (ex->model->executor_lock.test_and_set(std::memory_order_acquire)) {
    wait_hint(spins);
  }
}

inline void slot_table_unlock(ModelExecutor* ex) {
  ex->model->executor_lock.clear(std::memory_order_release);
  signal();
}

inline uint32_t next_mask_slot(uint32_t mask) {
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<uint32_t>(__builtin_ctz(mask));
#else
  uint32_t sid = 0;
  while ((mask & 1u) == 0u) {
    mask >>= 1u;
    ++sid;
  }
  return sid;
#endif
}

struct SlotSnapshot {
  ModelExecutor* ex = nullptr;
  Conversation* slots[ModelExecutor::kMaxSlotsHard]{};
  uint32_t active_mask = 0;

  SlotSnapshot() = default;
  SlotSnapshot(const SlotSnapshot&) = delete;
  SlotSnapshot& operator=(const SlotSnapshot&) = delete;

  ~SlotSnapshot() { release(); }

  void acquire(ModelExecutor* executor) {
    release();
    if (executor == nullptr || executor->model == nullptr) {
      return;
    }

    ex = executor;
    slot_table_lock(ex);
    uint32_t mask = ex->active_slot_mask;
    active_mask = mask;
    while (mask != 0u) {
      const uint32_t sid = next_mask_slot(mask);
      mask &= mask - 1u;
      Conversation* conv = ex->slots[sid].load(std::memory_order_relaxed);
      slots[sid] = conv;
      if (conv != nullptr) {
        conv->exec_refs.fetch_add(1, std::memory_order_acq_rel);
      }
    }
    slot_table_unlock(ex);
  }

  void release() {
    if (ex == nullptr) {
      return;
    }

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    bool wake_waiter = false;
#endif
    uint32_t mask = active_mask;
    active_mask = 0;
    while (mask != 0u) {
      const uint32_t sid = next_mask_slot(mask);
      mask &= mask - 1u;
      Conversation* conv = slots[sid];
      slots[sid] = nullptr;
      if (conv != nullptr) {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
        const uint32_t refs = conv->exec_refs.fetch_sub(1, std::memory_order_acq_rel);
        wake_waiter = wake_waiter || refs == 1u;
#else
        conv->exec_refs.fetch_sub(1, std::memory_order_acq_rel);
#endif
      }
    }
    ex = nullptr;
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    if (wake_waiter) {
      signal();
    }
#endif
  }
};

bool snapshot_has_work(const SlotSnapshot& snapshot) {
  uint32_t mask = snapshot.active_mask;
  while (mask != 0u) {
    const uint32_t sid = next_mask_slot(mask);
    mask &= mask - 1u;
    Conversation* conv = snapshot.slots[sid];
    if (conv != nullptr && (conv->state.load(std::memory_order_acquire) == ConvState::Decoding ||
                            conv->pending_emit_valid != 0)) {
      return true;
    }
  }
  return false;
}

struct GrammarApplyCtx {
  void* session_ctx;
  const backend::BackendOps* ops;
  uint32_t slot_id;
};

AstralErr ASTRAL_CALL apply_grammar_for_slot(void* grammar_ctx, uint32_t* tokens, float* logits,
                                             uint32_t count) {
  if (grammar_ctx == nullptr) {
    return ASTRAL_E_INVALID;
  }
  const GrammarApplyCtx* ctx = static_cast<const GrammarApplyCtx*>(grammar_ctx);
  if (ctx->ops == nullptr || ctx->ops->session_apply_grammar_for_slot == nullptr) {
    return ASTRAL_E_UNSUPPORTED;
  }
  return ctx->ops->session_apply_grammar_for_slot(ctx->session_ctx, ctx->slot_id, tokens, logits,
                                                  count);
}

AstralErr ensure_penalty_state(Conversation* conv) {
  if (conv == nullptr) {
    return ASTRAL_E_INVALID;
  }

  const bool needs_counts = (conv->sampler_cfg.repeat_penalty != 1.0f) ||
                            (conv->sampler_cfg.presence_penalty != 0.0f) ||
                            (conv->sampler_cfg.frequency_penalty != 0.0f);

  if (!needs_counts) {
    return ASTRAL_OK;
  }

  const uint32_t ctx_size = conv->ctx_size != 0 ? conv->ctx_size : 2048;
  const int32_t v = conv->sampler_cfg.repeat_last_n;
  const uint32_t repeat_last_n = v == 0 ? 0u : (v < 0 ? ctx_size : static_cast<uint32_t>(v));

  if (repeat_last_n != 0) {
    if (conv->token_counts == nullptr) {
      conv->token_counts = static_cast<uint16_t*>(
          conv->allocator.alloc(conv->vocab_size * sizeof(uint16_t), alignof(uint16_t)));
      if (conv->token_counts == nullptr) {
        return ASTRAL_E_NOMEM;
      }
      std::memset(conv->token_counts, 0, conv->vocab_size * sizeof(uint16_t));
    }

    if (conv->recent_capacity != repeat_last_n || conv->recent_tokens == nullptr) {
      conv->recent_tokens = static_cast<uint32_t*>(conv->allocator.alloc(
          static_cast<size_t>(repeat_last_n) * sizeof(uint32_t), alignof(uint32_t)));
      if (conv->recent_tokens == nullptr) {
        return ASTRAL_E_NOMEM;
      }
      conv->recent_capacity = repeat_last_n;
      conv->recent_pos = 0;
      conv->recent_size = 0;
    }
  }

  return ASTRAL_OK;
}

inline void penalty_state_clear(Conversation* conv) {
  if (conv == nullptr || conv->token_counts == nullptr) {
    return;
  }
  std::memset(conv->token_counts, 0, conv->vocab_size * sizeof(uint16_t));
  conv->recent_pos = 0;
  conv->recent_size = 0;
}

inline void penalty_state_push(Conversation* conv, uint32_t token_id) {
  if (conv == nullptr || conv->token_counts == nullptr || conv->recent_tokens == nullptr ||
      conv->recent_capacity == 0) {
    return;
  }
  if (token_id >= conv->vocab_size) {
    return;
  }

  if (conv->recent_size == conv->recent_capacity) {
    const uint32_t old = conv->recent_tokens[conv->recent_pos];
    if (old < conv->vocab_size) {
      uint16_t& c = conv->token_counts[old];
      if (c > 0) {
        --c;
      }
    }
  } else {
    ++conv->recent_size;
  }

  conv->recent_tokens[conv->recent_pos] = token_id;
  conv->recent_pos = (conv->recent_pos + 1u) % conv->recent_capacity;

  uint16_t& c = conv->token_counts[token_id];
  if (c != 0xFFFFu) {
    ++c;
  }
}

inline bool stop_check(Conversation* conv, int32_t token, uint32_t* out_match_len) {
  if (conv == nullptr || conv->stop_seq_count == 0 || token < 0) {
    return false;
  }

  if (conv->stop_tail_len < 32) {
    conv->stop_tail[conv->stop_tail_len++] = token;
  } else {
    std::memmove(conv->stop_tail, conv->stop_tail + 1, (32 - 1) * sizeof(int32_t));
    conv->stop_tail[31] = token;
  }

  for (uint32_t i = 0; i < conv->stop_seq_count; ++i) {
    const uint32_t len = conv->stop_seq_lens[i];
    if (len == 0 || len > conv->stop_tail_len) {
      continue;
    }
    bool match = true;
    for (uint32_t j = 0; j < len; ++j) {
      if (conv->stop_tail[conv->stop_tail_len - len + j] != conv->stop_seq_tokens[i][j]) {
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
}

inline bool stream_flush_one(Conversation* conv, const concurrency::StreamToken& tok) {
  if (conv == nullptr) {
    return false;
  }
  if (conv->token_ring.push(tok)) {
    signal();
    return true;
  }
  return false;
}

inline bool stream_hold_push(Conversation* conv, const concurrency::StreamToken& tok,
                             uint32_t stop_hold) {
  if (conv == nullptr) {
    return false;
  }

  if (stop_hold == 0) {
    return stream_flush_one(conv, tok);
  }

  const uint32_t tail = (conv->hold_head + conv->hold_count) % 64u;
  conv->hold[tail] = tok;
  if (conv->hold_count < 64u) {
    ++conv->hold_count;
  } else {
    if (!stream_flush_one(conv, conv->hold[conv->hold_head])) {
      return false;
    }
    conv->hold_head = (conv->hold_head + 1u) % 64u;
  }

  while (conv->hold_count > stop_hold) {
    if (!stream_flush_one(conv, conv->hold[conv->hold_head])) {
      return false;
    }
    conv->hold_head = (conv->hold_head + 1u) % 64u;
    --conv->hold_count;
  }

  return true;
}

inline bool stream_hold_flush_all(Conversation* conv) {
  if (conv == nullptr) {
    return false;
  }
  while (conv->hold_count > 0) {
    if (!stream_flush_one(conv, conv->hold[conv->hold_head])) {
      return false;
    }
    conv->hold_head = (conv->hold_head + 1u) % 64u;
    --conv->hold_count;
  }
  return true;
}

inline PromptChunk* conv_current_chunk(Conversation* conv) {
  if (conv == nullptr || conv->prompt_chunk_index >= conv->prompt_chunk_count) {
    return nullptr;
  }
  return &conv->prompt_chunks[conv->prompt_chunk_index];
}

inline bool conv_prompt_complete(const Conversation* conv) {
  return conv == nullptr || conv->prompt_chunk_index >= conv->prompt_chunk_count;
}

inline void process_logits_for_conv(ModelExecutor* ex, const backend::BackendOps* ops,
                                    Conversation* conv, uint32_t sid,
                                    const backend::BackendLogitsView& view) {
  if (ex == nullptr || ops == nullptr || conv == nullptr || view.logits == nullptr ||
      view.vocab_size == 0) {
    if (conv != nullptr) {
      conv->t_end_ticks = ticks_now();
      conv->final_err.store(ASTRAL_E_BACKEND, std::memory_order_release);
      conv->state.store(ConvState::Failed, std::memory_order_release);
      signal();
    }
    return;
  }

  // Seed repetition penalties from the prompt when the token-count table is available.
  if (ensure_penalty_state(conv) == ASTRAL_OK && conv->token_counts != nullptr &&
      conv->total_tokens == 0 && conv->prompt_count > 0) {
    penalty_state_clear(conv);
    for (uint32_t j = 0; j < conv->prompt_count; ++j) {
      const int32_t t = conv->prompt_tokens[j];
      if (t >= 0) {
        penalty_state_push(conv, static_cast<uint32_t>(t));
      }
    }
  }

  SamplerMeta meta{};
  SamplerMeta* meta_ptr = conv->logprobs_n > 0 ? &meta : nullptr;

  GrammarApplyCtx grammar_ctx{};
  void* grammar_ctx_ptr = nullptr;
  AstralErr(ASTRAL_CALL * grammar_apply_fn)(void*, uint32_t*, float*, uint32_t) = nullptr;
  if (conv->grammar_kind != 0 && conv->grammar_applied != 0) {
    if (ops->session_apply_grammar_for_slot != nullptr) {
      grammar_ctx.session_ctx = ex->backend_session_ctx;
      grammar_ctx.ops = ops;
      grammar_ctx.slot_id = sid;
      grammar_ctx_ptr = &grammar_ctx;
      grammar_apply_fn = apply_grammar_for_slot;
    }
  }

  const uint32_t next = sample_token(
      view.logits, static_cast<size_t>(view.vocab_size), conv->sampler_cfg, &conv->rng_state,
      conv->token_counts, conv->token_counts_base, conv->token_nl, &conv->mirostat_mu,
      conv->logprobs_n, meta_ptr, conv->sample_ids, conv->sample_logits, conv->sample_capacity,
      grammar_ctx_ptr, grammar_apply_fn, ex->indices_buffer);

  const int32_t next_token = static_cast<int32_t>(next);

  conv->total_tokens++;
  if (conv->total_tokens == 1) {
    conv->t_first_token_ticks = ticks_now();
  }

  uint32_t stop_len = 0;
  const bool stop_hit = stop_check(conv, next_token, &stop_len);

  if (next_token >= 0) {
    penalty_state_push(conv, static_cast<uint32_t>(next_token));
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
    (void)conv->meta_ring.push(ev);
  }

  conv->pending_emit_token = next_token;
  if (conv->desc.stream_enabled != 0) {
    concurrency::StreamToken st{};
    st.token_id = static_cast<uint32_t>(next_token);
    AstralMutSpanU8 out{};
    out.data = st.utf8_data;
    out.len = sizeof(st.utf8_data);
    uint32_t utf8_len = 0;
    const AstralErr det_err =
        ops->detokenize(ex->model->backend_model_ctx, &next_token, 1, out, &utf8_len);
    if (det_err != ASTRAL_OK) {
      conv->t_end_ticks = ticks_now();
      conv->final_err.store(det_err, std::memory_order_release);
      conv->state.store(ConvState::Failed, std::memory_order_release);
      signal();
      return;
    }
    st.utf8_len = static_cast<uint16_t>(utf8_len);

    const uint32_t stop_hold =
        (conv->stop_max_len > 0 && conv->stop_max_len <= 32) ? conv->stop_max_len : 0;
    if (stream_hold_push(conv, st, stop_hold)) {
      conv->pending_emit_valid = 0;
    } else {
      conv->pending_emit_stream = st;
      conv->pending_emit_valid = 1;
    }
  } else {
    conv->pending_emit_valid = 0;
  }

  if (!stop_hit && conv->token_eos >= 0 && next_token == conv->token_eos) {
    conv->pending_token_valid = 0;
  } else if (conv->total_tokens >= conv->desc.max_tokens) {
    conv->pending_token_valid = 0;
  } else if (!stop_hit) {
    conv->pending_token = next_token;
    conv->pending_token_valid = 1;
  } else {
    conv->pending_token_valid = 0;
  }

  if (stop_hit) {
    if (conv->desc.stream_enabled != 0) {
      if (stop_len >= conv->hold_count) {
        conv->hold_count = 0;
      } else {
        conv->hold_count -= stop_len;
      }
      (void)stream_hold_flush_all(conv);
    }
    conv->t_end_ticks = ticks_now();
    conv->final_err.store(ASTRAL_OK, std::memory_order_release);
    conv->state.store(ConvState::Completed, std::memory_order_release);
    signal();
  } else if (conv->total_tokens >= conv->desc.max_tokens ||
             (conv->token_eos >= 0 && next_token == conv->token_eos)) {
    if (conv->desc.stream_enabled != 0) {
      (void)stream_hold_flush_all(conv);
    }
    conv->t_end_ticks = ticks_now();
    conv->final_err.store(ASTRAL_OK, std::memory_order_release);
    conv->state.store(ConvState::Completed, std::memory_order_release);
    signal();
  }
}

void executor_thread(ModelExecutor* ex) {
  ASTRAL_ZONE_N("astral.executor.thread");
  if (ex == nullptr || ex->model == nullptr || ex->model->backend == nullptr ||
      ex->model->backend->ops == nullptr) {
    return;
  }

  auto* model = ex->model;
  const auto* ops = model->backend->ops;

  if (ops->session_batch_eval == nullptr || ops->session_batch_logits == nullptr) {
    return;
  }

  const uint32_t max_prompt_tokens_per_slot_per_tick =
      ex->max_prompt_tokens_per_slot_per_tick.load(std::memory_order_relaxed);
  uint32_t rr = 0;

  while (ex->running.load(std::memory_order_acquire)) {
    SlotSnapshot snapshot;
    snapshot.acquire(ex);
    bool any_work = snapshot_has_work(snapshot);

    if (!any_work) {
      uint32_t spins = 0;
      while (ex->running.load(std::memory_order_acquire) && !any_work) {
        snapshot.release();
        wait_hint(spins);
        snapshot.acquire(ex);
        any_work = snapshot_has_work(snapshot);
      }
      if (!any_work) {
        continue;
      }
    }

    // 1) Backpressure recovery: attempt to emit one token per slot (non-blocking).
    uint32_t mask = snapshot.active_mask;
    while (mask != 0u) {
      const uint32_t sid = next_mask_slot(mask);
      mask &= mask - 1u;
      Conversation* conv = snapshot.slots[sid];
      if (conv == nullptr)
        continue;

      if (conv->pending_emit_valid != 0) {
        if (conv->desc.stream_enabled != 0) {
          const uint32_t stop_hold =
              (conv->stop_max_len > 0 && conv->stop_max_len <= 32) ? conv->stop_max_len : 0;
          if (stream_hold_push(conv, conv->pending_emit_stream, stop_hold)) {
            conv->pending_emit_valid = 0;
            signal();
          }
        } else {
          conv->pending_emit_valid = 0;
          signal();
        }
      }
    }

    // 1b) Apply per-slot reset/grammar before advancing decoding.
    mask = snapshot.active_mask;
    while (mask != 0u) {
      const uint32_t sid = next_mask_slot(mask);
      mask &= mask - 1u;
      Conversation* conv = snapshot.slots[sid];
      if (conv == nullptr)
        continue;

      if (conv->state.load(std::memory_order_acquire) != ConvState::Decoding) {
        continue;
      }

      if (conv->needs_slot_reset != 0 && ops->session_slot_reset != nullptr) {
        (void)ops->session_slot_reset(ex->backend_session_ctx, sid);
        conv->needs_slot_reset = 0;
        conv->n_past = 0;
        conv->prompt_off = 0;
        conv->pending_token_valid = 0;
        conv->pending_emit_valid = 0;
        conv->stop_tail_len = 0;
        conv->hold_head = 0;
        conv->hold_count = 0;
      }

      if (conv->grammar_dirty != 0) {
        AstralErr gerr = ASTRAL_OK;
        if (ops->session_grammar_clear_for_slot != nullptr) {
          (void)ops->session_grammar_clear_for_slot(ex->backend_session_ctx, sid);
        }

        if (conv->grammar_kind == 1 && ops->session_grammar_set_gbnf_for_slot != nullptr &&
            conv->grammar_gbnf != nullptr && conv->grammar_gbnf_len > 0) {
          AstralSpanU8 g{};
          g.data = conv->grammar_gbnf;
          g.len = conv->grammar_gbnf_len;
          AstralSpanU8 root{};
          root.data = conv->grammar_root;
          root.len = conv->grammar_root_len;
          gerr = ops->session_grammar_set_gbnf_for_slot(ex->backend_session_ctx, sid, g, root);
        } else if (conv->grammar_kind == 2 &&
                   ops->session_grammar_set_json_schema_for_slot != nullptr &&
                   conv->grammar_json != nullptr && conv->grammar_json_len > 0) {
          AstralSpanU8 schema{};
          schema.data = conv->grammar_json;
          schema.len = conv->grammar_json_len;
          gerr =
              ops->session_grammar_set_json_schema_for_slot(ex->backend_session_ctx, sid, schema);
        }

        if (gerr != ASTRAL_OK) {
          conv->t_end_ticks = ticks_now();
          conv->final_err.store(gerr, std::memory_order_release);
          conv->state.store(ConvState::Failed, std::memory_order_release);
          signal();
        } else {
          conv->grammar_dirty = 0;
          conv->grammar_applied = 1;
        }
      }
    }

    // 1c) Feed media prompt chunks (image/audio) outside batch eval.
    mask = snapshot.active_mask;
    while (mask != 0u) {
      const uint32_t sid = next_mask_slot(mask);
      mask &= mask - 1u;
      Conversation* conv = snapshot.slots[sid];
      if (conv == nullptr)
        continue;

      if (conv->state.load(std::memory_order_acquire) != ConvState::Decoding ||
          conv->pending_emit_valid != 0) {
        continue;
      }

      if (conv->cancel_requested.load(std::memory_order_acquire)) {
        conv->t_end_ticks = ticks_now();
        conv->final_err.store(ASTRAL_E_CANCELED, std::memory_order_release);
        conv->state.store(ConvState::Canceled, std::memory_order_release);
        signal();
        continue;
      }

      PromptChunk* chunk = conv_current_chunk(conv);
      if (chunk == nullptr || chunk->kind == PromptChunkKind::Text) {
        continue;
      }

      if (ops->session_set_slot == nullptr || ops->session_slot_pos == nullptr ||
          ops->session_logits == nullptr) {
        conv->t_end_ticks = ticks_now();
        conv->final_err.store(ASTRAL_E_UNSUPPORTED, std::memory_order_release);
        conv->state.store(ConvState::Failed, std::memory_order_release);
        signal();
        continue;
      }

      const AstralErr slot_err = ops->session_set_slot(ex->backend_session_ctx, sid);
      if (slot_err != ASTRAL_OK) {
        conv->t_end_ticks = ticks_now();
        conv->final_err.store(slot_err, std::memory_order_release);
        conv->state.store(ConvState::Failed, std::memory_order_release);
        signal();
        continue;
      }

      AstralErr feed_err = ASTRAL_E_UNSUPPORTED;
      if (chunk->kind == PromptChunkKind::Image) {
        if (ops->session_feed_image != nullptr) {
          feed_err =
              ops->session_feed_image(ex->backend_session_ctx, &chunk->image, chunk->finalize);
        }
      } else if (chunk->kind == PromptChunkKind::Audio) {
        if (ops->session_feed_audio != nullptr) {
          feed_err =
              ops->session_feed_audio(ex->backend_session_ctx, &chunk->audio, chunk->finalize);
        }
      }

      if (feed_err != ASTRAL_OK) {
        conv->t_end_ticks = ticks_now();
        conv->final_err.store(feed_err, std::memory_order_release);
        conv->state.store(ConvState::Failed, std::memory_order_release);
        signal();
        continue;
      }

      uint32_t slot_pos = 0;
      const AstralErr pos_err = ops->session_slot_pos(ex->backend_session_ctx, sid, &slot_pos);
      if (pos_err != ASTRAL_OK) {
        conv->t_end_ticks = ticks_now();
        conv->final_err.store(pos_err, std::memory_order_release);
        conv->state.store(ConvState::Failed, std::memory_order_release);
        signal();
        continue;
      }
      conv->n_past = slot_pos;

      prompt_chunk_reset(*chunk);
      conv->prompt_chunk_index += 1u;
      conv->prompt_chunk_token_off = 0;
      any_work = true;

      if (conv_prompt_complete(conv) && conv->pending_token_valid == 0) {
        backend::BackendLogitsView view{};
        const AstralErr log_err = ops->session_logits(ex->backend_session_ctx, &view);
        if (log_err != ASTRAL_OK || view.logits == nullptr || view.vocab_size == 0) {
          conv->t_end_ticks = ticks_now();
          conv->final_err.store(log_err != ASTRAL_OK ? log_err : ASTRAL_E_BACKEND,
                                std::memory_order_release);
          conv->state.store(ConvState::Failed, std::memory_order_release);
          signal();
          continue;
        }
        process_logits_for_conv(ex, ops, conv, sid, view);
      }
    }

    // 2) Build a continuous batching step.
    uint32_t batch_count = 0;
    const uint32_t cap = ex->max_batch_tokens;

    // Decode stepping: at most 1 token per slot (round-robin).
    const uint32_t before_rr_mask = rr == 0u ? 0u : ((1u << rr) - 1u);
    for (uint32_t pass = 0; pass < 2u && batch_count < cap; ++pass) {
      mask = pass == 0u ? (snapshot.active_mask & ~before_rr_mask)
                        : (snapshot.active_mask & before_rr_mask);
      while (mask != 0u && batch_count < cap) {
        const uint32_t sid = next_mask_slot(mask);
        mask &= mask - 1u;
        Conversation* conv = snapshot.slots[sid];
        if (conv == nullptr)
          continue;

        if (conv->state.load(std::memory_order_acquire) != ConvState::Decoding) {
          continue;
        }

        if (conv->cancel_requested.load(std::memory_order_acquire)) {
          conv->t_end_ticks = ticks_now();
          conv->final_err.store(ASTRAL_E_CANCELED, std::memory_order_release);
          conv->state.store(ConvState::Canceled, std::memory_order_release);
          signal();
          continue;
        }

        if (conv->pending_emit_valid != 0) {
          continue;
        }

        if (conv->pending_token_valid != 0) {
          ex->batch_tokens[batch_count].token = conv->pending_token;
          ex->batch_tokens[batch_count].slot_id = sid;
          ex->batch_tokens[batch_count].pos = conv->n_past;
          ex->batch_tokens[batch_count].want_logits = 1;
          ++batch_count;
        }
      }
    }

    // Prompt ingestion (use remaining capacity).
    for (uint32_t pass = 0; pass < 2u && batch_count < cap; ++pass) {
      mask = pass == 0u ? (snapshot.active_mask & ~before_rr_mask)
                        : (snapshot.active_mask & before_rr_mask);
      while (mask != 0u && batch_count < cap) {
        const uint32_t sid = next_mask_slot(mask);
        mask &= mask - 1u;
        Conversation* conv = snapshot.slots[sid];
        if (conv == nullptr)
          continue;

        if (conv->state.load(std::memory_order_acquire) != ConvState::Decoding ||
            conv->pending_emit_valid != 0) {
          continue;
        }

        PromptChunk* chunk = conv_current_chunk(conv);
        if (chunk == nullptr || chunk->kind != PromptChunkKind::Text) {
          continue;
        }

        if (conv->prompt_chunk_token_off >= chunk->token_count) {
          conv->prompt_chunk_index += 1u;
          conv->prompt_chunk_token_off = 0;
          continue;
        }

        const uint32_t remaining = chunk->token_count - conv->prompt_chunk_token_off;
        const uint32_t slot_cap = (cap - batch_count) < max_prompt_tokens_per_slot_per_tick
                                      ? (cap - batch_count)
                                      : max_prompt_tokens_per_slot_per_tick;
        const uint32_t take = remaining < slot_cap ? remaining : slot_cap;
        for (uint32_t i = 0; i < take; ++i) {
          const uint32_t idx = chunk->token_start + conv->prompt_chunk_token_off + i;
          const int32_t tok = conv->prompt_tokens[idx];
          ex->batch_tokens[batch_count].token = tok;
          ex->batch_tokens[batch_count].slot_id = sid;
          ex->batch_tokens[batch_count].pos = conv->n_past + i;
          ex->batch_tokens[batch_count].want_logits = 0;
          ++batch_count;
        }

        if (conv->prompt_chunk_token_off + take == chunk->token_count &&
            conv->prompt_chunk_index + 1u >= conv->prompt_chunk_count &&
            conv->pending_token_valid == 0 && take > 0) {
          ex->batch_tokens[batch_count - 1u].want_logits = 1;
        }
      }
    }

    if (batch_count == 0) {
      rr = (rr + 1u) % ex->max_slots;
      continue;
    }

    uint32_t out_count = 0;
    const AstralErr eval_err =
        ops->session_batch_eval(ex->backend_session_ctx, ex->batch_tokens, batch_count, &out_count);

    if (eval_err != ASTRAL_OK) {
      // Fail all decoding conversations.
      mask = snapshot.active_mask;
      while (mask != 0u) {
        const uint32_t sid = next_mask_slot(mask);
        mask &= mask - 1u;
        Conversation* conv = snapshot.slots[sid];
        if (conv == nullptr)
          continue;
        if (conv->state.load(std::memory_order_acquire) != ConvState::Decoding) {
          continue;
        }
        conv->t_end_ticks = ticks_now();
        conv->final_err.store(eval_err, std::memory_order_release);
        conv->state.store(ConvState::Failed, std::memory_order_release);
      }
      signal();
      rr = (rr + 1u) % ex->max_slots;
      continue;
    }

    // Update per-slot positions and prompt offsets.
    for (uint32_t i = 0; i < batch_count; ++i) {
      const uint32_t sid = ex->batch_tokens[i].slot_id;
      Conversation* conv = snapshot.slots[sid];
      if (conv == nullptr)
        continue;

      if (conv->state.load(std::memory_order_acquire) != ConvState::Decoding) {
        continue;
      }

      if (conv->prompt_off < conv->prompt_count) {
        const uint32_t off = conv->prompt_off;
        if (off < conv->prompt_count && conv->prompt_tokens[off] == ex->batch_tokens[i].token) {
          conv->prompt_off = off + 1u;
          if (conv->prompt_chunk_index < conv->prompt_chunk_count) {
            PromptChunk& chunk = conv->prompt_chunks[conv->prompt_chunk_index];
            if (chunk.kind == PromptChunkKind::Text) {
              if (conv->prompt_chunk_token_off < chunk.token_count) {
                conv->prompt_chunk_token_off += 1u;
              }
              if (conv->prompt_chunk_token_off >= chunk.token_count) {
                conv->prompt_chunk_index += 1u;
                conv->prompt_chunk_token_off = 0;
              }
            }
          }
        }
      } else if (conv->pending_token_valid != 0) {
        conv->pending_token_valid = 0;
      }

      conv->n_past += 1;
    }

    // Build output mapping by scanning want_logits in batch order.
    uint32_t out_expected = 0;
    for (uint32_t i = 0; i < batch_count; ++i) {
      if (ex->batch_tokens[i].want_logits != 0) {
        ex->batch_output_slots[out_expected++] = ex->batch_tokens[i].slot_id;
      }
    }
    if (out_count > out_expected) {
      out_count = out_expected;
    }

    for (uint32_t oi = 0; oi < out_count; ++oi) {
      const uint32_t sid = ex->batch_output_slots[oi];
      Conversation* conv = snapshot.slots[sid];
      if (conv == nullptr)
        continue;

      if (conv->state.load(std::memory_order_acquire) != ConvState::Decoding ||
          conv->pending_emit_valid != 0) {
        continue;
      }

      backend::BackendLogitsView view{};
      const AstralErr log_err = ops->session_batch_logits(ex->backend_session_ctx, oi, &view);
      if (log_err != ASTRAL_OK || view.logits == nullptr || view.vocab_size == 0) {
        conv->t_end_ticks = ticks_now();
        conv->final_err.store(log_err != ASTRAL_OK ? log_err : ASTRAL_E_BACKEND,
                              std::memory_order_release);
        conv->state.store(ConvState::Failed, std::memory_order_release);
        signal();
        continue;
      }

      process_logits_for_conv(ex, ops, conv, sid, view);
    }

    rr = (rr + 1u) % ex->max_slots;
    signal();
  }
}

} // namespace

ModelExecutor::ModelExecutor(Model* model_) noexcept : model(model_) {
  active_slot_mask = 0;
  for (uint32_t i = 0; i < kMaxSlotsHard; ++i) {
    slots[i].store(nullptr, std::memory_order_relaxed);
  }
}

void executor_work_item(void* user) {
  auto* ex = static_cast<ModelExecutor*>(user);
  if (ex == nullptr) {
    return;
  }
  ex->started.store(true, std::memory_order_release);
  executor_thread(ex);
  ex->finished.store(true, std::memory_order_release);
  signal();
}

void executor_start(ModelExecutor* ex) {
  if (ex == nullptr || ex->model == nullptr || ex->model->backend == nullptr ||
      ex->model->backend->ops == nullptr) {
    return;
  }

  ex->started.store(false, std::memory_order_release);
  ex->finished.store(true, std::memory_order_release); // until successfully started

  auto* model = ex->model;
  const auto* ops = model->backend->ops;

  uint32_t vocab_size = 0;
  uint32_t ctx_size = 0;
  const AstralErr info_err = ops->model_info(model->backend_model_ctx, &vocab_size, &ctx_size);
  if (info_err != ASTRAL_OK || vocab_size == 0) {
    ex->running.store(false, std::memory_order_release);
    return;
  }

  ex->vocab_size = vocab_size;
  ex->ctx_size = ctx_size;

  ex->indices_buffer = static_cast<uint32_t*>(
      core::runtime_alloc(static_cast<size_t>(vocab_size) * sizeof(uint32_t), alignof(uint32_t)));
  ex->batch_tokens = static_cast<AstralBackendBatchToken*>(core::runtime_alloc(
      static_cast<size_t>(ex->max_batch_tokens) * sizeof(AstralBackendBatchToken),
      alignof(AstralBackendBatchToken)));
  ex->batch_output_slots = static_cast<uint32_t*>(core::runtime_alloc(
      static_cast<size_t>(ex->max_batch_tokens) * sizeof(uint32_t), alignof(uint32_t)));

  if (ex->indices_buffer == nullptr || ex->batch_tokens == nullptr ||
      ex->batch_output_slots == nullptr) {
    ex->running.store(false, std::memory_order_release);
    return;
  }

  AstralSessionDesc sd{};
  sd.model = model->handle;
  sd.max_tokens = 1;
  sd.temperature = 0.0f;
  sd.top_k = 0;
  sd.top_p = 1.0f;
  sd.stream_enabled = 0;
  sd.seed = 1;

  AstralErr create_err = ASTRAL_OK;
  if (ops->session_create_ex != nullptr) {
    ex->backend_session_ctx =
        ops->session_create_ex(model->backend_model_ctx, &sd, ex->max_slots, &create_err);
  } else {
    ex->backend_session_ctx = ops->session_create(model->backend_model_ctx, &sd, &create_err);
  }

  if (ex->backend_session_ctx == nullptr) {
    ex->running.store(false, std::memory_order_release);
    return;
  }

  // Use the runtime worker pool to run the executor loop (one worker per model).
  if (!core::runtime_initialized()) {
    ex->running.store(false, std::memory_order_release);
    return;
  }

  const uint32_t cfg_worker = ex->model->executor_desc.worker_hint;
  ex->worker_id = cfg_worker != 0 ? cfg_worker : core::runtime_assign_worker_id();

  ex->running.store(true, std::memory_order_release);
  ex->finished.store(false, std::memory_order_release);

  const AstralErr submit_err = core::submit_work_affine(ex->worker_id, executor_work_item, ex);
  if (submit_err != ASTRAL_OK) {
    ex->running.store(false, std::memory_order_release);
    ex->finished.store(true, std::memory_order_release);
    return;
  }
}

void executor_stop_and_join(ModelExecutor* ex) {
  if (ex == nullptr) {
    return;
  }

  ex->running.store(false, std::memory_order_release);
  signal();

  uint32_t spins = 0;
  while (!ex->finished.load(std::memory_order_acquire)) {
    if (spins < 64) {
      astral::platform::cpu_pause();
    } else {
      astral::platform::cpu_wait_for_event();
    }
    if (spins < 1024) {
      ++spins;
    }
  }

  if (ex->model && ex->model->backend && ex->model->backend->ops && ex->backend_session_ctx) {
    ex->model->backend->ops->session_destroy(ex->backend_session_ctx);
    ex->backend_session_ctx = nullptr;
  }

  if (ex->indices_buffer) {
    core::runtime_free(ex->indices_buffer, static_cast<size_t>(ex->vocab_size) * sizeof(uint32_t),
                       alignof(uint32_t));
    ex->indices_buffer = nullptr;
  }
  if (ex->batch_tokens) {
    core::runtime_free(ex->batch_tokens,
                       static_cast<size_t>(ex->max_batch_tokens) * sizeof(AstralBackendBatchToken),
                       alignof(AstralBackendBatchToken));
    ex->batch_tokens = nullptr;
  }
  if (ex->batch_output_slots) {
    core::runtime_free(ex->batch_output_slots,
                       static_cast<size_t>(ex->max_batch_tokens) * sizeof(uint32_t),
                       alignof(uint32_t));
    ex->batch_output_slots = nullptr;
  }
}

} // namespace astral::inference
