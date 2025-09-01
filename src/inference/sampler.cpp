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

} // namespace

uint32_t sample_token(const float* logits,
                      size_t vocab_size,
                      const SamplerConfig& cfg,
                      uint32_t* rng_state,
                      uint32_t* scratch_ids,
                      float* scratch_vals,
                      uint32_t scratch_capacity,
                      uint32_t* indices_buffer) {
    if (logits == nullptr || vocab_size == 0 || rng_state == nullptr) {
        return 0;
    }

    // Greedy path.
    if (cfg.temperature <= 0.0f) {
        return argmax(logits, vocab_size);
    }

    const float top_p = sanitize_top_p(cfg.top_p);

    uint32_t top_k = cfg.top_k;
    if (top_k > 0 && top_k > vocab_size) {
        top_k = static_cast<uint32_t>(vocab_size);
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
            const float v = logits[i] * inv_temp;
            if (v > max_scaled) {
                max_scaled = v;
            }
        }

        double sum_all = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            sum_all += static_cast<double>(std::exp(logits[i] * inv_temp - max_scaled));
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
            nucleus_sum += static_cast<double>(std::exp(logits[id] * inv_temp - max_scaled));
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
            cum += static_cast<double>(std::exp(logits[id] * inv_temp - max_scaled));
            if (cum >= target) {
                return id;
            }
        }

        return indices_buffer[vocab_size - selected];
    }

    // Full distribution sampling without top-k/top-p (two-pass, no storage).
    if (top_k == 0 && top_p >= 1.0f) {
        const float inv_temp = 1.0f / cfg.temperature;

        float max_scaled = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < vocab_size; ++i) {
            const float v = logits[i] * inv_temp;
            if (v > max_scaled) {
                max_scaled = v;
            }
        }

        double sum = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            sum += static_cast<double>(std::exp(logits[i] * inv_temp - max_scaled));
        }
        if (!(sum > 0.0)) {
            return argmax(logits, vocab_size);
        }

        const double target = static_cast<double>(uniform01(rng_state)) * sum;
        double cum = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            cum += static_cast<double>(std::exp(logits[i] * inv_temp - max_scaled));
            if (cum >= target) {
                return static_cast<uint32_t>(i);
            }
        }

        return static_cast<uint32_t>(vocab_size - 1);
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
        const float v = logits[i];

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

    // Convert candidate logits to unnormalized weights (softmax over candidates).
    const float max_scaled = scratch_vals[0] * inv_temp;
    for (size_t i = 0; i < heap_size; ++i) {
        scratch_vals[i] = static_cast<float>(std::exp(scratch_vals[i] * inv_temp - max_scaled));
    }

    return sample_from_weights_sorted_desc(scratch_ids, scratch_vals, heap_size, top_p, rng_state);
}

} // namespace astral::inference
