#pragma once

#include "../../include/astral_rt.h"
#include <cstddef>
#include <cstdint>

namespace astral::inference {

/// Sampler configuration (from AstralSessionDesc).
///
/// This is configuration-only. Mutable RNG state lives in the session.
struct SamplerConfig {
    float temperature; // 0.0 = greedy
    uint32_t top_k;    // 0 = disabled
    float top_p;       // <=0 or >=1 = disabled
    float min_p;       // 0.0 = disabled
    float typical_p;   // 1.0 = disabled
    float repeat_penalty;    // 1.0 = disabled
    int32_t repeat_last_n;   // 0 = disabled, -1 = ctx size (treated as clamp)
    uint8_t penalize_nl;     // 1 = penalize newline token
    uint8_t _padding0[3];
    float presence_penalty;  // 0.0 = disabled
    float frequency_penalty; // 0.0 = disabled
    uint32_t mirostat;       // 0 = disabled, 1 = mirostat, 2 = mirostat v2
    float mirostat_tau;      // target surprise
    float mirostat_eta;      // learning rate
    uint32_t seed;     // Initial seed (captured for diagnostics; session owns rng_state)
};

struct SamplerMeta {
    uint32_t token_id;
    float logprob; // 0 for greedy
    uint32_t top_n;
    uint32_t top_ids[ASTRAL_LOGPROBS_MAX];
    float top_logprobs[ASTRAL_LOGPROBS_MAX];
};

/// Sample a token from a provider logits view.
///
/// Constraints:
/// - Must not allocate.
/// - Must not write to `logits` (provider-owned).
///
/// Scratch:
/// - `scratch_ids`/`scratch_vals` are used for the fast top-k path and must have
///   capacity >= max(top_k, internal_candidate_cap). May be null when not needed.
/// - `indices_buffer` must be `vocab_size` entries when full-vocab top-p fallback is used.
///
/// Returns the sampled token id in [0, vocab_size).
uint32_t sample_token(const float* logits,
                      size_t vocab_size,
                      const SamplerConfig& cfg,
                      uint32_t* rng_state,
                      const uint16_t* token_counts_recent,
                      const uint16_t* token_counts_base,
                      int32_t token_nl,
                      float* mirostat_mu_in_out,
                      uint32_t logprobs_n,
                      SamplerMeta* out_meta,
                      uint32_t* scratch_ids,
                      float* scratch_vals,
                      uint32_t scratch_capacity,
                      void* grammar_ctx,
                      AstralErr (ASTRAL_CALL * grammar_apply)(void* grammar_ctx,
                                                             uint32_t* tokens,
                                                             float* logits,
                                                             uint32_t count),
                      uint32_t* indices_buffer);

} // namespace astral::inference
