#pragma once

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
    uint32_t seed;     // Initial seed (captured for diagnostics; session owns rng_state)
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
                      uint32_t* scratch_ids,
                      float* scratch_vals,
                      uint32_t scratch_capacity,
                      uint32_t* indices_buffer);

} // namespace astral::inference

