#include "sampler.hpp"

#include <cmath>
#include <limits>

namespace astral::inference {

namespace {

inline uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline float uniform01(uint32_t* state) {
    // Produce a float in [0, 1).
    *state = xorshift32(*state);
    return static_cast<float>(*state) * (1.0f / 4294967296.0f);
}

inline float sanitize_top_p(float p) {
    return (p > 0.0f && p < 1.0f) ? p : 1.0f;
}

inline float sanitize_min_p(float p) {
    return (p > 0.0f && p < 1.0f) ? p : 0.0f;
}

inline float sanitize_typical_p(float p) {
    return (p > 0.0f && p < 1.0f) ? p : 1.0f;
}

inline float sanitize_repeat_penalty(float p) {
    return (p > 0.0f) ? p : 1.0f;
}

inline float apply_repeat_penalty(float logit, float penalty) {
    if (penalty == 1.0f) {
        return logit;
    }
    // Common convention: negative logits are multiplied, positive are divided.
    return logit < 0.0f ? (logit * penalty) : (logit / penalty);
}

inline float apply_presence_frequency_penalties(float logit, uint16_t count, float presence, float frequency) {
    if (count == 0) {
        return logit;
    }
    return logit - presence - (frequency * static_cast<float>(count));
}

inline uint32_t argmax(const float* logits, size_t n) {
    size_t best = 0;
    float best_val = logits[0];
    for (size_t i = 1; i < n; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return static_cast<uint32_t>(best);
}

inline void heap_swap_u32(uint32_t* ids, size_t a, size_t b) {
    const uint32_t tmp = ids[a];
    ids[a] = ids[b];
    ids[b] = tmp;
}

// Max-heap by logits[ids[i]].
inline void heap_sift_down_max(uint32_t* ids, size_t idx, size_t size, const float* logits) {
    for (;;) {
        const size_t left = (idx << 1) + 1;
        if (left >= size) {
            return;
        }

        const size_t right = left + 1;
        size_t best = left;
        if (right < size && logits[ids[right]] > logits[ids[left]]) {
            best = right;
        }

        if (logits[ids[idx]] >= logits[ids[best]]) {
            return;
        }

        heap_swap_u32(ids, idx, best);
        idx = best;
    }
}

inline void heap_build_max(uint32_t* ids, size_t size, const float* logits) {
    if (size <= 1) {
        return;
    }

    for (size_t i = size / 2; i-- > 0;) {
        heap_sift_down_max(ids, i, size, logits);
    }
}

inline uint32_t heap_pop_max(uint32_t* ids, size_t* size_in_out, const float* logits) {
    size_t size = *size_in_out;
    if (size == 0) {
        return 0;
    }

    heap_swap_u32(ids, 0, size - 1);
    const uint32_t id = ids[size - 1];
    --size;
    heap_sift_down_max(ids, 0, size, logits);
    *size_in_out = size;
    return id;
}

inline void heap_swap(uint32_t* ids, float* vals, size_t a, size_t b) {
    const uint32_t id_tmp = ids[a];
    ids[a] = ids[b];
    ids[b] = id_tmp;

    const float v_tmp = vals[a];
    vals[a] = vals[b];
    vals[b] = v_tmp;
}

// Min-heap by vals.
inline void heap_sift_up(uint32_t* ids, float* vals, size_t idx) {
    while (idx > 0) {
        const size_t parent = (idx - 1) >> 1;
        if (vals[parent] <= vals[idx]) {
            break;
        }
        heap_swap(ids, vals, parent, idx);
        idx = parent;
    }
}

inline void heap_sift_down(uint32_t* ids, float* vals, size_t idx, size_t size) {
    for (;;) {
        const size_t left = (idx << 1) + 1;
        if (left >= size) {
            return;
        }

        const size_t right = left + 1;
        size_t smallest = left;
        if (right < size && vals[right] < vals[left]) {
            smallest = right;
        }

        if (vals[idx] <= vals[smallest]) {
            return;
        }

        heap_swap(ids, vals, idx, smallest);
        idx = smallest;
    }
}

inline uint32_t sample_from_weights_sorted_desc(const uint32_t* ids,
                                                float* weights_in_out,
                                                size_t count,
                                                float top_p,
                                                uint32_t* rng_state) {
    // weights_in_out contains unnormalized weights; this function:
    // - finds nucleus cutoff (by cumulative mass in sorted order)
    // - samples from the truncated distribution
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(weights_in_out[i]);
    }

    if (!(sum > 0.0)) {
        // Degenerate distribution; fall back to best token.
        return ids[0];
    }

    const float p = sanitize_top_p(top_p);
    const double threshold = static_cast<double>(p) * sum;

    double nucleus_sum = sum;
    size_t cutoff = count;
    if (p < 1.0f) {
        double cum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            cum += static_cast<double>(weights_in_out[i]);
            if (cum >= threshold) {
                cutoff = i + 1;
                nucleus_sum = cum;
                break;
            }
        }
    }

    const double target = static_cast<double>(uniform01(rng_state)) * nucleus_sum;
    double cum = 0.0;
    for (size_t i = 0; i < cutoff; ++i) {
        cum += static_cast<double>(weights_in_out[i]);
        if (cum >= target) {
            return ids[i];
        }
    }

    return ids[cutoff > 0 ? cutoff - 1 : 0];
}

inline uint32_t sample_from_weights_unsorted(const uint32_t* ids, const float* weights, size_t count, uint32_t* rng) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(weights[i]);
    }
    if (!(sum > 0.0)) {
        return ids[0];
    }

    const double target = static_cast<double>(uniform01(rng)) * sum;
    double cum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        cum += static_cast<double>(weights[i]);
        if (cum >= target) {
            return ids[i];
        }
    }
    return ids[count - 1];
}

} // namespace

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
                      uint32_t* indices_buffer) {
    if (logits == nullptr || vocab_size == 0 || rng_state == nullptr) {
        return 0;
    }

    if (out_meta != nullptr) {
        out_meta->token_id = 0;
        out_meta->logprob = 0.0f;
        out_meta->top_n = 0;
        for (uint32_t i = 0; i < ASTRAL_LOGPROBS_MAX; ++i) {
            out_meta->top_ids[i] = 0;
            out_meta->top_logprobs[i] = 0.0f;
        }
    }

    const float repeat_penalty = sanitize_repeat_penalty(cfg.repeat_penalty);
    const float presence_penalty = cfg.presence_penalty;
    const float frequency_penalty = cfg.frequency_penalty;
    const bool penalize_nl = (cfg.penalize_nl != 0) && (token_nl >= 0);
    const bool use_penalties = (token_counts_recent != nullptr || token_counts_base != nullptr) &&
                               ((repeat_penalty != 1.0f) || (presence_penalty != 0.0f) || (frequency_penalty != 0.0f));

    auto adjusted_logit = [&](size_t token_id, float base) -> float {
        if (!use_penalties) {
            return base;
        }
        if (!penalize_nl && static_cast<int32_t>(token_id) == token_nl) {
            // NL excluded from penalties unless explicitly enabled.
            return base;
        }
        uint16_t count = 0;
        if (token_id < vocab_size) {
            if (token_counts_recent != nullptr) {
                count = static_cast<uint16_t>(count + token_counts_recent[token_id]);
            }
            if (token_counts_base != nullptr) {
                count = static_cast<uint16_t>(count + token_counts_base[token_id]);
            }
        }
        float v = base;
        if (count > 0) {
            if (repeat_penalty != 1.0f) {
                v = apply_repeat_penalty(v, repeat_penalty);
            }
            if (presence_penalty != 0.0f || frequency_penalty != 0.0f) {
                v = apply_presence_frequency_penalties(v, count, presence_penalty, frequency_penalty);
            }
        }
        return v;
    };

    // Greedy path.
    if (cfg.temperature <= 0.0f) {
        uint32_t best = 0;
        float best_val = -std::numeric_limits<float>::infinity();

        if (grammar_ctx != nullptr && grammar_apply != nullptr) {
            constexpr size_t kChunk = 256;
            uint32_t ids[kChunk];
            float vals[kChunk];

            for (size_t base = 0; base < vocab_size; base += kChunk) {
                const size_t n = (vocab_size - base) < kChunk ? (vocab_size - base) : kChunk;
                for (size_t i = 0; i < n; ++i) {
                    const size_t tid = base + i;
                    ids[i] = static_cast<uint32_t>(tid);
                    vals[i] = adjusted_logit(tid, logits[tid]);
                }

                (void)grammar_apply(grammar_ctx, ids, vals, static_cast<uint32_t>(n));

                for (size_t i = 0; i < n; ++i) {
                    const float v = vals[i];
                    if (v > best_val) {
                        best_val = v;
                        best = ids[i];
                    }
                }
            }
        } else {
            best = 0;
            best_val = adjusted_logit(0, logits[0]);
            for (size_t i = 1; i < vocab_size; ++i) {
                const float v = adjusted_logit(i, logits[i]);
                if (v > best_val) {
                    best_val = v;
                    best = static_cast<uint32_t>(i);
                }
            }
        }

        if (out_meta != nullptr && logprobs_n > 0) {
            out_meta->token_id = best;
            out_meta->logprob = 0.0f;
            out_meta->top_n = 1;
            out_meta->top_ids[0] = best;
            out_meta->top_logprobs[0] = 0.0f;
        }
        return best;
    }

    // Mirostat sampling ignores top-k/top-p/typical/min_p and uses a self-adjusting truncation.
    if (cfg.mirostat != 0) {
        if (scratch_ids == nullptr || scratch_vals == nullptr || scratch_capacity == 0) {
            return argmax(logits, vocab_size);
        }

        float mu = (mirostat_mu_in_out != nullptr) ? *mirostat_mu_in_out : (2.0f * cfg.mirostat_tau);
        const float tau = (cfg.mirostat_tau > 0.0f) ? cfg.mirostat_tau : 5.0f;
        const float eta = (cfg.mirostat_eta > 0.0f) ? cfg.mirostat_eta : 0.1f;
        if (!(mu > 0.0f)) {
            mu = 2.0f * tau;
        }

        const float inv_temp = 1.0f / cfg.temperature;

        // Keep top-N candidates, where N is the available scratch capacity.
        size_t heap_size = 0;
        const size_t k = scratch_capacity < vocab_size ? static_cast<size_t>(scratch_capacity) : vocab_size;

        for (size_t i = 0; i < vocab_size; ++i) {
            const float v = adjusted_logit(i, logits[i]);

            if (heap_size < k) {
                scratch_ids[heap_size] = static_cast<uint32_t>(i);
                scratch_vals[heap_size] = v;
                heap_sift_up(scratch_ids, scratch_vals, heap_size);
                ++heap_size;
                continue;
            }

            if (v > scratch_vals[0]) {
                scratch_ids[0] = static_cast<uint32_t>(i);
                scratch_vals[0] = v;
                heap_sift_down(scratch_ids, scratch_vals, 0, heap_size);
            }
        }

        // Heap-sort to get candidates sorted by logit (descending).
        for (size_t end = heap_size; end > 1; --end) {
            heap_swap(scratch_ids, scratch_vals, 0, end - 1);
            heap_sift_down(scratch_ids, scratch_vals, 0, end - 1);
        }

        // Apply grammar constraints at candidate level (best-effort).
        if (grammar_ctx != nullptr && grammar_apply != nullptr && heap_size > 0) {
            (void)grammar_apply(grammar_ctx, scratch_ids, scratch_vals, static_cast<uint32_t>(heap_size));
        }

        // Convert to unnormalized weights.
        // NOTE: grammar filtering can invalidate the top candidate; recompute the max over finite values.
        float max_logit = -std::numeric_limits<float>::infinity();
        bool any_finite = false;
        for (size_t i = 0; i < heap_size; ++i) {
            const float v = scratch_vals[i];
            if (!std::isfinite(v)) {
                continue;
            }
            if (!any_finite || v > max_logit) {
                max_logit = v;
                any_finite = true;
            }
        }

        if (!any_finite) {
            // All candidates filtered out; fall back to grammar-aware argmax.
            if (grammar_ctx != nullptr && grammar_apply != nullptr) {
                uint32_t best = 0;
                float best_val = -std::numeric_limits<float>::infinity();

                constexpr size_t kChunk = 256;
                uint32_t ids[kChunk];
                float vals[kChunk];

                for (size_t base = 0; base < vocab_size; base += kChunk) {
                    const size_t n = (vocab_size - base) < kChunk ? (vocab_size - base) : kChunk;
                    for (size_t i = 0; i < n; ++i) {
                        const size_t tid = base + i;
                        ids[i] = static_cast<uint32_t>(tid);
                        vals[i] = adjusted_logit(tid, logits[tid]);
                    }

                    (void)grammar_apply(grammar_ctx, ids, vals, static_cast<uint32_t>(n));

                    for (size_t i = 0; i < n; ++i) {
                        const float v = vals[i];
                        if (v > best_val) {
                            best_val = v;
                            best = ids[i];
                        }
                    }
                }

                return best;
            }

            return scratch_ids[0];
        }

        const float max_scaled = max_logit * inv_temp;
        for (size_t i = 0; i < heap_size; ++i) {
            scratch_vals[i] = static_cast<float>(std::exp(scratch_vals[i] * inv_temp - max_scaled));
        }

        size_t count = heap_size;

        auto softmax_sum = [&](size_t n) -> double {
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i) {
                sum += static_cast<double>(scratch_vals[i]);
            }
            return sum;
        };

        double sum = softmax_sum(count);
        if (!(sum > 0.0)) {
            return scratch_ids[0];
        }

        // Normalize p in-place into a temporary stack array for the computations that require it.
        float pbuf[ASTRAL_LOGPROBS_MAX + 256];
        const size_t pcap = sizeof(pbuf) / sizeof(pbuf[0]);
        const size_t pcount = (count < pcap) ? count : pcap;
        for (size_t i = 0; i < pcount; ++i) {
            pbuf[i] = static_cast<float>(static_cast<double>(scratch_vals[i]) / sum);
        }

        if (cfg.mirostat == 1) {
            // Estimate s_hat using the most probable m tokens (m=100 in the paper; clamp to available).
            const int32_t m = 100;
            const size_t m_eff = (static_cast<size_t>(m) - 1u < count) ? static_cast<size_t>(m) - 1u : (count > 0 ? count - 1 : 0);
            float sum_ti_bi = 0.0f;
            float sum_ti_sq = 0.0f;
            for (size_t i = 0; i < m_eff; ++i) {
                const float p0 = (i < pcount) ? pbuf[i] : static_cast<float>(static_cast<double>(scratch_vals[i]) / sum);
                const float p1 = ((i + 1) < pcount) ? pbuf[i + 1] : static_cast<float>(static_cast<double>(scratch_vals[i + 1]) / sum);
                if (!(p0 > 0.0f) || !(p1 > 0.0f)) {
                    continue;
                }
                const float t_i = std::log(static_cast<float>(i + 2) / static_cast<float>(i + 1));
                const float b_i = std::log(p0 / p1);
                sum_ti_bi += t_i * b_i;
                sum_ti_sq += t_i * t_i;
            }

            const float s_hat = (sum_ti_sq > 0.0f) ? (sum_ti_bi / sum_ti_sq) : 1.0f;
            const float epsilon_hat = s_hat - 1.0f;
            const double denom = 1.0 - std::pow(static_cast<double>(vocab_size), -static_cast<double>(epsilon_hat));
            const double k_d = (denom != 0.0)
                                   ? std::pow((static_cast<double>(epsilon_hat) * std::pow(2.0, static_cast<double>(mu))) / denom,
                                              1.0 / static_cast<double>(s_hat))
                                   : 1.0;

            size_t k_keep = static_cast<size_t>(k_d < 1.0 ? 1.0 : k_d);
            if (k_keep > count) {
                k_keep = count;
            }
            sum = softmax_sum(k_keep);
            if (!(sum > 0.0)) {
                return scratch_ids[0];
            }

            const double target = static_cast<double>(uniform01(rng_state)) * sum;
            double cum = 0.0;
            size_t chosen_idx = 0;
            for (size_t i = 0; i < k_keep; ++i) {
                cum += static_cast<double>(scratch_vals[i]);
                if (cum >= target) {
                    chosen_idx = i;
                    break;
                }
            }

            const float p = static_cast<float>(static_cast<double>(scratch_vals[chosen_idx]) / sum);
            const float logprob = (p > 0.0f) ? std::log(p) : -std::numeric_limits<float>::infinity();
            const float observed_surprise = (p > 0.0f) ? (-std::log2(p)) : 0.0f;
            const float e = observed_surprise - tau;
            mu = mu - eta * e;
            if (mirostat_mu_in_out != nullptr) {
                *mirostat_mu_in_out = mu;
            }

            if (out_meta != nullptr && logprobs_n > 0) {
                const uint32_t topn = (logprobs_n > ASTRAL_LOGPROBS_MAX) ? ASTRAL_LOGPROBS_MAX : logprobs_n;
                out_meta->token_id = scratch_ids[chosen_idx];
                out_meta->logprob = logprob;
                out_meta->top_n = (topn < k_keep) ? topn : static_cast<uint32_t>(k_keep);
                for (uint32_t i = 0; i < out_meta->top_n; ++i) {
                    const float pi = static_cast<float>(static_cast<double>(scratch_vals[i]) / sum);
                    out_meta->top_ids[i] = scratch_ids[i];
                    out_meta->top_logprobs[i] = (pi > 0.0f) ? std::log(pi) : -std::numeric_limits<float>::infinity();
                }
            }

            return scratch_ids[chosen_idx];
        }

        // Mirostat v2: truncate by surprise > mu.
        const float p_threshold = std::pow(2.0f, -mu);
        size_t keep = 0;
        for (; keep < count; ++keep) {
            const float p = (keep < pcount) ? pbuf[keep] : static_cast<float>(static_cast<double>(scratch_vals[keep]) / sum);
            if (!(p >= p_threshold)) {
                break;
            }
        }
        if (keep == 0) {
            keep = 1;
        }

        sum = softmax_sum(keep);
        if (!(sum > 0.0)) {
            return scratch_ids[0];
        }

        const double target = static_cast<double>(uniform01(rng_state)) * sum;
        double cum = 0.0;
        size_t chosen_idx = 0;
        for (size_t i = 0; i < keep; ++i) {
            cum += static_cast<double>(scratch_vals[i]);
            if (cum >= target) {
                chosen_idx = i;
                break;
            }
        }

        const float p = static_cast<float>(static_cast<double>(scratch_vals[chosen_idx]) / sum);
        const float logprob = (p > 0.0f) ? std::log(p) : -std::numeric_limits<float>::infinity();
        const float observed_surprise = (p > 0.0f) ? (-std::log2(p)) : 0.0f;
        const float e = observed_surprise - tau;
        mu = mu - eta * e;
        if (mirostat_mu_in_out != nullptr) {
            *mirostat_mu_in_out = mu;
        }

        if (out_meta != nullptr && logprobs_n > 0) {
            const uint32_t topn = (logprobs_n > ASTRAL_LOGPROBS_MAX) ? ASTRAL_LOGPROBS_MAX : logprobs_n;
            out_meta->token_id = scratch_ids[chosen_idx];
            out_meta->logprob = logprob;
            out_meta->top_n = (topn < keep) ? topn : static_cast<uint32_t>(keep);
            for (uint32_t i = 0; i < out_meta->top_n; ++i) {
                const float pi = static_cast<float>(static_cast<double>(scratch_vals[i]) / sum);
                out_meta->top_ids[i] = scratch_ids[i];
                out_meta->top_logprobs[i] = (pi > 0.0f) ? std::log(pi) : -std::numeric_limits<float>::infinity();
            }
        }

        return scratch_ids[chosen_idx];
    }

    const float top_p = sanitize_top_p(cfg.top_p);
    const float min_p = sanitize_min_p(cfg.min_p);
    const float typical_p = sanitize_typical_p(cfg.typical_p);

    uint32_t top_k = cfg.top_k;
    if (top_k > 0 && top_k > vocab_size) {
        top_k = static_cast<uint32_t>(vocab_size);
    }

    // If grammar is enabled and top_p is configured without an explicit top_k, prefer a bounded candidate
    // window so the grammar can be applied (otherwise we'd need a full-vocab grammar filter).
    if (top_k == 0 && top_p < 1.0f && grammar_ctx != nullptr && grammar_apply != nullptr && scratch_ids != nullptr &&
        scratch_vals != nullptr && scratch_capacity > 0) {
        top_k = scratch_capacity < vocab_size ? scratch_capacity : static_cast<uint32_t>(vocab_size);
    }

    // Exact full-vocab nucleus sampling fallback when top_k is disabled.
    // Preserves semantics without copying logits.
    if (top_k == 0 && top_p < 1.0f) {
        if (indices_buffer == nullptr) {
            return argmax(logits, vocab_size);
        }

        const float inv_temp = 1.0f / cfg.temperature;

        // Compute max and full partition function (no storage).
        float max_scaled = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < vocab_size; ++i) {
            const float v = adjusted_logit(i, logits[i]) * inv_temp;
            if (v > max_scaled) {
                max_scaled = v;
            }
        }

        double sum_all = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            sum_all += static_cast<double>(std::exp(adjusted_logit(i, logits[i]) * inv_temp - max_scaled));
        }

        if (!(sum_all > 0.0)) {
            return argmax(logits, vocab_size);
        }

        for (size_t i = 0; i < vocab_size; ++i) {
            indices_buffer[i] = static_cast<uint32_t>(i);
        }

        // Build a max-heap of all indices (O(n)), then pop until we reach nucleus mass.
        size_t heap_size = vocab_size;
        heap_build_max(indices_buffer, heap_size, logits);

        const double threshold = static_cast<double>(top_p) * sum_all;

        double nucleus_sum = 0.0;
        size_t selected = 0;
        while (heap_size > 0) {
            const uint32_t id = heap_pop_max(indices_buffer, &heap_size, logits);
            nucleus_sum += static_cast<double>(std::exp(adjusted_logit(id, logits[id]) * inv_temp - max_scaled));
            ++selected;
            if (nucleus_sum >= threshold) {
                break;
            }
        }

        if (selected == 0) {
            return argmax(logits, vocab_size);
        }

        const double target = static_cast<double>(uniform01(rng_state)) * nucleus_sum;
        double cum = 0.0;

        // Popped ids are stored at the end of indices_buffer:
        // indices_buffer[vocab_size - 1] is the first (largest) popped token.
        for (size_t i = 0; i < selected; ++i) {
            const uint32_t id = indices_buffer[vocab_size - 1 - i];
            cum += static_cast<double>(std::exp(adjusted_logit(id, logits[id]) * inv_temp - max_scaled));
            if (cum >= target) {
                return id;
            }
        }

        return indices_buffer[vocab_size - selected];
    }

    // Full distribution sampling without top-k/top-p (two-pass, no storage).
    if (top_k == 0 && top_p >= 1.0f) {
        const float inv_temp = 1.0f / cfg.temperature;

        // 1) Compute log-sum-exp in a single pass (optionally grammar-filtered).
        double sumexp = 0.0;
        float max_scaled = -std::numeric_limits<float>::infinity();
        bool any = false;

        if (grammar_ctx != nullptr && grammar_apply != nullptr) {
            constexpr size_t kChunk = 256;
            uint32_t ids[kChunk];
            float vals[kChunk];

            for (size_t base = 0; base < vocab_size; base += kChunk) {
                const size_t n = (vocab_size - base) < kChunk ? (vocab_size - base) : kChunk;
                for (size_t i = 0; i < n; ++i) {
                    const size_t tid = base + i;
                    ids[i] = static_cast<uint32_t>(tid);
                    vals[i] = adjusted_logit(tid, logits[tid]);
                }
                (void)grammar_apply(grammar_ctx, ids, vals, static_cast<uint32_t>(n));

                for (size_t i = 0; i < n; ++i) {
                    const float v = vals[i];
                    if (!std::isfinite(v)) {
                        continue;
                    }
                    const float s = v * inv_temp;
                    if (!any) {
                        max_scaled = s;
                        sumexp = 1.0;
                        any = true;
                        continue;
                    }
                    if (s > max_scaled) {
                        sumexp = sumexp * std::exp(static_cast<double>(max_scaled - s)) + 1.0;
                        max_scaled = s;
                    } else {
                        sumexp += std::exp(static_cast<double>(s - max_scaled));
                    }
                }
            }
        } else {
            for (size_t i = 0; i < vocab_size; ++i) {
                const float s = adjusted_logit(i, logits[i]) * inv_temp;
                if (!any) {
                    max_scaled = s;
                    sumexp = 1.0;
                    any = true;
                    continue;
                }
                if (s > max_scaled) {
                    sumexp = sumexp * std::exp(static_cast<double>(max_scaled - s)) + 1.0;
                    max_scaled = s;
                } else {
                    sumexp += std::exp(static_cast<double>(s - max_scaled));
                }
            }
        }

        if (!any || !(sumexp > 0.0)) {
            return argmax(logits, vocab_size);
        }

        const double target = static_cast<double>(uniform01(rng_state)) * sumexp;

        // 2) Sample by cumulative mass using the same max.
        double cum = 0.0;
        uint32_t last = 0;

        if (grammar_ctx != nullptr && grammar_apply != nullptr) {
            constexpr size_t kChunk = 256;
            uint32_t ids[kChunk];
            float vals[kChunk];

            for (size_t base = 0; base < vocab_size; base += kChunk) {
                const size_t n = (vocab_size - base) < kChunk ? (vocab_size - base) : kChunk;
                for (size_t i = 0; i < n; ++i) {
                    const size_t tid = base + i;
                    ids[i] = static_cast<uint32_t>(tid);
                    vals[i] = adjusted_logit(tid, logits[tid]);
                }
                (void)grammar_apply(grammar_ctx, ids, vals, static_cast<uint32_t>(n));

                for (size_t i = 0; i < n; ++i) {
                    const float v = vals[i];
                    if (!std::isfinite(v)) {
                        continue;
                    }
                    const double w = std::exp(static_cast<double>(v * inv_temp - max_scaled));
                    cum += w;
                    last = ids[i];
                    if (cum >= target) {
                        return ids[i];
                    }
                }
            }
        } else {
            for (size_t i = 0; i < vocab_size; ++i) {
                const double w = std::exp(static_cast<double>(adjusted_logit(i, logits[i]) * inv_temp - max_scaled));
                cum += w;
                if (cum >= target) {
                    return static_cast<uint32_t>(i);
                }
                last = static_cast<uint32_t>(i);
            }
        }

        return last;
    }

    // Fast top-k path (optionally followed by nucleus within top-k).
    if (scratch_ids == nullptr || scratch_vals == nullptr || scratch_capacity < top_k || top_k == 0) {
        // No scratch available; fall back to greedy.
        return argmax(logits, vocab_size);
    }

    const float inv_temp = 1.0f / cfg.temperature;

    size_t heap_size = 0;
    const size_t k = static_cast<size_t>(top_k);

    for (size_t i = 0; i < vocab_size; ++i) {
        const float v = adjusted_logit(i, logits[i]);

        if (heap_size < k) {
            scratch_ids[heap_size] = static_cast<uint32_t>(i);
            scratch_vals[heap_size] = v;
            heap_sift_up(scratch_ids, scratch_vals, heap_size);
            ++heap_size;
            continue;
        }

        // Keep only the top-k logits.
        if (v > scratch_vals[0]) {
            scratch_ids[0] = static_cast<uint32_t>(i);
            scratch_vals[0] = v;
            heap_sift_down(scratch_ids, scratch_vals, 0, heap_size);
        }
    }

    // Heap-sort the min-heap to get candidates sorted by logit (descending).
    for (size_t end = heap_size; end > 1; --end) {
        heap_swap(scratch_ids, scratch_vals, 0, end - 1);
        heap_sift_down(scratch_ids, scratch_vals, 0, end - 1);
    }

    // Apply grammar constraints at candidate level (best-effort).
    if (grammar_ctx != nullptr && grammar_apply != nullptr && heap_size > 0) {
        (void)grammar_apply(grammar_ctx, scratch_ids, scratch_vals, static_cast<uint32_t>(heap_size));
    }

    // Convert candidate logits to unnormalized weights (softmax over candidates).
    const float max_scaled = scratch_vals[0] * inv_temp;
    for (size_t i = 0; i < heap_size; ++i) {
        scratch_vals[i] = static_cast<float>(std::exp(scratch_vals[i] * inv_temp - max_scaled));
    }

    size_t count = heap_size;

    // min_p filter within candidate set: p_i >= min_p * p_max  <=> w_i >= min_p * w_max.
    if (min_p > 0.0f && count > 0) {
        const float w_max = scratch_vals[0];
        const float cutoff = min_p * w_max;
        size_t kept = 0;
        for (size_t i = 0; i < count; ++i) {
            if (scratch_vals[i] >= cutoff) {
                scratch_ids[kept] = scratch_ids[i];
                scratch_vals[kept] = scratch_vals[i];
                ++kept;
            }
        }
        if (kept > 0) {
            count = kept;
        }
    }

    // typical sampling within the candidate set. For now we treat typical_p as mutually exclusive with top_p.
    // (This keeps the implementation simple and avoids extra sorting passes in v0.1.)
    if (typical_p < 1.0f && count > 1 && count <= 256) {
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            sum += static_cast<double>(scratch_vals[i]);
        }
        if (sum > 0.0) {
            const double inv_sum = 1.0 / sum;
            double entropy = 0.0;
            for (size_t i = 0; i < count; ++i) {
                const double p = static_cast<double>(scratch_vals[i]) * inv_sum;
                if (p > 0.0) {
                    entropy += -p * std::log(p);
                }
            }

            float dist[256];
            for (size_t i = 0; i < count; ++i) {
                const double p = static_cast<double>(scratch_vals[i]) * inv_sum;
                const double neg_logp = (p > 0.0) ? -std::log(p) : 1e30;
                dist[i] = static_cast<float>(std::fabs(neg_logp - entropy));
            }

            // Insertion sort by dist ascending (count <= 256).
            for (size_t i = 1; i < count; ++i) {
                const float d = dist[i];
                const uint32_t id = scratch_ids[i];
                const float w = scratch_vals[i];
                size_t j = i;
                while (j > 0 && dist[j - 1] > d) {
                    dist[j] = dist[j - 1];
                    scratch_ids[j] = scratch_ids[j - 1];
                    scratch_vals[j] = scratch_vals[j - 1];
                    --j;
                }
                dist[j] = d;
                scratch_ids[j] = id;
                scratch_vals[j] = w;
            }

            double cum_p = 0.0;
            size_t cutoff = count;
            for (size_t i = 0; i < count; ++i) {
                cum_p += static_cast<double>(scratch_vals[i]) * inv_sum;
                if (cum_p >= static_cast<double>(typical_p)) {
                    cutoff = i + 1;
                    break;
                }
            }

            const uint32_t chosen = sample_from_weights_unsorted(scratch_ids, scratch_vals, cutoff, rng_state);

            if (out_meta != nullptr && logprobs_n > 0) {
                double sum_sub = 0.0;
                for (size_t i = 0; i < cutoff; ++i) {
                    sum_sub += static_cast<double>(scratch_vals[i]);
                }
                const uint32_t topn = (logprobs_n > ASTRAL_LOGPROBS_MAX) ? ASTRAL_LOGPROBS_MAX : logprobs_n;

                out_meta->token_id = chosen;
                out_meta->top_n = 0;
                if (sum_sub > 0.0) {
                    // Find chosen logprob.
                    for (size_t i = 0; i < cutoff; ++i) {
                        if (scratch_ids[i] == chosen) {
                            const float p = static_cast<float>(static_cast<double>(scratch_vals[i]) / sum_sub);
                            out_meta->logprob = (p > 0.0f) ? std::log(p) : -std::numeric_limits<float>::infinity();
                            break;
                        }
                    }

                    // Top-N by probability within the subset (selection without extra sorting).
                    // O(cutoff * topn), cutoff <= 256.
                    for (uint32_t j = 0; j < topn; ++j) {
                        size_t best_i = cutoff;
                        double best_w = -1.0;
                        for (size_t i = 0; i < cutoff; ++i) {
                            const uint32_t id = scratch_ids[i];
                            // Skip ids already picked.
                            bool used = false;
                            for (uint32_t k = 0; k < out_meta->top_n; ++k) {
                                if (out_meta->top_ids[k] == id) {
                                    used = true;
                                    break;
                                }
                            }
                            if (used) {
                                continue;
                            }

                            const double w = static_cast<double>(scratch_vals[i]);
                            if (w > best_w) {
                                best_w = w;
                                best_i = i;
                            }
                        }
                        if (best_i == cutoff || best_w <= 0.0) {
                            break;
                        }
                        const float p = static_cast<float>(best_w / sum_sub);
                        out_meta->top_ids[out_meta->top_n] = scratch_ids[best_i];
                        out_meta->top_logprobs[out_meta->top_n] =
                            (p > 0.0f) ? std::log(p) : -std::numeric_limits<float>::infinity();
                        out_meta->top_n += 1;
                    }
                }
            }

            return chosen;
        }
    }

    if (out_meta == nullptr || logprobs_n == 0) {
        return sample_from_weights_sorted_desc(scratch_ids, scratch_vals, count, top_p, rng_state);
    }

    // Top-p sampling with meta: compute nucleus cutoff and renormalized logprobs.
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(scratch_vals[i]);
    }
    if (!(sum > 0.0)) {
        out_meta->token_id = scratch_ids[0];
        out_meta->logprob = 0.0f;
        out_meta->top_n = 1;
        out_meta->top_ids[0] = scratch_ids[0];
        out_meta->top_logprobs[0] = 0.0f;
        return scratch_ids[0];
    }

    const float p = sanitize_top_p(top_p);
    const double threshold = static_cast<double>(p) * sum;

    double nucleus_sum = sum;
    size_t cutoff = count;
    if (p < 1.0f) {
        double cum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            cum += static_cast<double>(scratch_vals[i]);
            if (cum >= threshold) {
                cutoff = i + 1;
                nucleus_sum = cum;
                break;
            }
        }
    }

    const double target = static_cast<double>(uniform01(rng_state)) * nucleus_sum;
    double cum = 0.0;
    size_t chosen_idx = 0;
    for (size_t i = 0; i < cutoff; ++i) {
        cum += static_cast<double>(scratch_vals[i]);
        if (cum >= target) {
            chosen_idx = i;
            break;
        }
    }

    const float chosen_p = static_cast<float>(static_cast<double>(scratch_vals[chosen_idx]) / nucleus_sum);
    out_meta->token_id = scratch_ids[chosen_idx];
    out_meta->logprob = (chosen_p > 0.0f) ? std::log(chosen_p) : -std::numeric_limits<float>::infinity();

    const uint32_t topn = (logprobs_n > ASTRAL_LOGPROBS_MAX) ? ASTRAL_LOGPROBS_MAX : logprobs_n;
    out_meta->top_n = (topn < cutoff) ? topn : static_cast<uint32_t>(cutoff);
    for (uint32_t i = 0; i < out_meta->top_n; ++i) {
        const float pi = static_cast<float>(static_cast<double>(scratch_vals[i]) / nucleus_sum);
        out_meta->top_ids[i] = scratch_ids[i];
        out_meta->top_logprobs[i] = (pi > 0.0f) ? std::log(pi) : -std::numeric_limits<float>::infinity();
    }

    return out_meta->token_id;
}

} // namespace astral::inference
