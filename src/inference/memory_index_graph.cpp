#include "memory_index_internal.hpp"

#include "../core/runtime_alloc.hpp"
#include "../core/runtime_state.hpp"
#include "../core/work_queue.hpp"
#include "../platform/atomics.h"

#include <atomic>
#include <cstdint>
#include <cstring>

namespace astral::inference {

namespace {

constexpr uint32_t kGraphMinSearch = 4;
constexpr uint32_t kGraphCandidateReserveMultiplier = 4;
constexpr uint32_t kGraphLongLinkCount = 0;
constexpr uint32_t kGraphNeighborPrefetchDistance = 2;
constexpr uint32_t kGraphF32RerankMinCandidates = 256;
constexpr uint32_t kGraphF32RerankTopKMultiplier = 32;
constexpr uint32_t kGraphMaxNeighbors = 64;
constexpr uint32_t kGraphBaseNeighborMultiplier = 2;
constexpr uint32_t kGraphMaxBaseNeighbors = kGraphMaxNeighbors * kGraphBaseNeighborMultiplier;
constexpr uint64_t kBytesPerKiB = 1024;
constexpr uint64_t kBytesPerMiB = kBytesPerKiB * kBytesPerKiB;
constexpr uint64_t kGraphCompactExactSearchMaxBytes = 16 * kBytesPerMiB;
constexpr uint16_t kGraphVisitGenerationStart = 1;
constexpr uint32_t kMemorySearchBatchParallelMinQueries = 8;

struct MemoryGraphSearchBatchJob {
  MemoryIndex* index;
  const AstralMemorySearchDesc* desc;
  const float* queries;
  AstralMemorySearchResult* out_results;
  uint32_t* out_counts;
  uint32_t begin;
  uint32_t end;
  std::atomic<uint32_t>* remaining;
  GraphSearchScratch* scratch;
};

inline void memory_parallel_job_complete(std::atomic<uint32_t>* remaining) {
  if (remaining->fetch_sub(1, std::memory_order_release) == 1u) {
    platform::cpu_signal_event();
  }
}

inline uint32_t graph_search_for_query(const MemoryIndex* index,
                                       const AstralMemorySearchDesc* desc) {
  uint32_t requested = desc->graph_search;
  if (requested == 0) {
    requested = index->graph_query_search_capacity;
  }
  if (requested < kGraphMinSearch) {
    requested = kGraphMinSearch;
  }
  return requested < index->graph_scratch_capacity ? requested : index->graph_scratch_capacity;
}

inline uint32_t graph_candidate_search_capacity(const MemoryIndex* index,
                                                uint32_t search_capacity) {
  uint32_t requested = search_capacity;
  if (search_capacity <= kU32Max / kGraphCandidateReserveMultiplier) {
    requested = search_capacity * kGraphCandidateReserveMultiplier;
  }
  return requested < index->graph_candidate_capacity ? requested : index->graph_candidate_capacity;
}

inline bool compact_graph_exact_search_preferred(const MemoryIndex* index) {
  return compact_storage(index) &&
         static_cast<uint64_t>(index->count) * static_cast<uint64_t>(index->dim) <=
             kGraphCompactExactSearchMaxBytes;
}


inline bool graph_candidate_better(const MemoryIndex* index, float candidate_score,
                                   uint32_t candidate_slot, float existing_score,
                                   uint32_t existing_slot) {
  return candidate_score > existing_score ||
         (candidate_score == existing_score &&
          index->slots[candidate_slot].record.key < index->slots[existing_slot].record.key);
}

inline bool graph_candidate_worse(const MemoryIndex* index, float candidate_score,
                                  uint32_t candidate_slot, float existing_score,
                                  uint32_t existing_slot) {
  return candidate_score < existing_score ||
         (candidate_score == existing_score &&
          index->slots[candidate_slot].record.key > index->slots[existing_slot].record.key);
}


enum class CompactByteQueryStorage : uint8_t {
  q8,
  e5m2,
};

template <CompactByteQueryStorage Storage>
inline float score_slot_compact_query(MemoryIndex* index, const int8_t* query, int32_t query_sum,
                                      float query_scale, uint32_t slot, float cosine_query_scale) {
  const int8_t* q8 = q8_vector_at(index, slot);
  const float scale = index->compact_score_scales[slot];
  if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    if constexpr (Storage == CompactByteQueryStorage::e5m2) {
      return index->e5m2_kernels->dot_e5m2(q8, query, index->dim) * scale * query_scale;
    } else {
      return dot_q8_q8_query_aligned(q8, query, index->dim, query_sum) * scale * query_scale;
    }
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    if constexpr (Storage == CompactByteQueryStorage::e5m2) {
      return index->e5m2_kernels->dot_e5m2(q8, query, index->dim) * scale * query_scale *
             cosine_query_scale;
    } else {
      return dot_q8_q8_query_aligned(q8, query, index->dim, query_sum) * scale * query_scale *
             cosine_query_scale;
    }
  }
  if constexpr (Storage == CompactByteQueryStorage::e5m2) {
    return index->e5m2_kernels->l2_e5m2(q8, compact_value_scale(index, index->q8_scales[slot]),
                                        query, query_scale, index->dim);
  } else {
    return l2_score_q8_q8(q8, compact_value_scale(index, index->q8_scales[slot]), query,
                          query_scale, index->dim);
  }
}

inline float score_slot_compact_query_i16(MemoryIndex* index, const int16_t* query,
                                          float query_scale, uint32_t slot,
                                          float cosine_query_scale) {
  const int16_t* v = i16_vector_at(index, slot);
  const float scale = index->compact_score_scales[slot];
  if (index->metric == ASTRAL_MEMORY_METRIC_DOT) {
    const float dot = index->i16_vectors_aligned != 0 ? dot_i16_i16_aligned(v, query, index->dim)
                                                      : dot_i16_i16(v, query, index->dim);
    return dot * scale * query_scale;
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    const float dot = index->i16_vectors_aligned != 0 ? dot_i16_i16_aligned(v, query, index->dim)
                                                      : dot_i16_i16(v, query, index->dim);
    return dot * scale * query_scale * cosine_query_scale;
  }
  return l2_score_i16_i16(v, compact_value_scale(index, index->q8_scales[slot]), query, query_scale,
                          index->dim);
}

inline float score_pair(MemoryIndex* index, uint32_t a, uint32_t b) {
  ++index->graph_build_score_evals;
  if (f8_f32_rerank_storage(index)) {
    const float* va = vector_at(index, a);
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return dot_f32(va, vector_at(index, b), index->dim);
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
      return l2_score_f32(va, vector_at(index, b), index->dim);
    }
    return dot_f32(va, vector_at(index, b), index->dim);
  }
  if (compact_storage(index)) {
    const float scale_a = index->compact_score_scales[a];
    const float scale_b = index->compact_score_scales[b];
    if (i16_storage(index)) {
      const int16_t* va = i16_vector_at(index, a);
      const int16_t* vb = i16_vector_at(index, b);
      if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
        const float dot = index->i16_vectors_aligned != 0 ? dot_i16_i16_aligned(va, vb, index->dim)
                                                          : dot_i16_i16(va, vb, index->dim);
        return dot * scale_a * scale_b;
      }
      if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
        return l2_score_i16_i16(va, compact_value_scale(index, index->q8_scales[a]), vb,
                                compact_value_scale(index, index->q8_scales[b]), index->dim);
      }
      const float dot = index->i16_vectors_aligned != 0 ? dot_i16_i16_aligned(va, vb, index->dim)
                                                        : dot_i16_i16(va, vb, index->dim);
      return dot * scale_a * scale_b;
    }
    const int8_t* va = q8_vector_at(index, a);
    const int8_t* vb = q8_vector_at(index, b);
    if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return (e5m2_storage(index)
                  ? index->e5m2_kernels->dot_e5m2(va, vb, index->dim)
                  : dot_q8_q8_query_aligned(va, vb, index->dim, index->compact_vector_sums[b])) *
             scale_a * scale_b;
    }
    if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
      return e5m2_storage(index)
                 ? index->e5m2_kernels->l2_e5m2(va, compact_value_scale(index, index->q8_scales[a]),
                                                vb, compact_value_scale(index, index->q8_scales[b]),
                                                index->dim)
                 : l2_score_q8_q8(va, compact_value_scale(index, index->q8_scales[a]), vb,
                                  compact_value_scale(index, index->q8_scales[b]), index->dim);
    }
    return (e5m2_storage(index)
                ? index->e5m2_kernels->dot_e5m2(va, vb, index->dim)
                : dot_q8_q8_query_aligned(va, vb, index->dim, index->compact_vector_sums[b])) *
           scale_a * scale_b;
  }
  const float* va = vector_at(index, a);
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return dot_f32(va, vector_at(index, b), index->dim);
  }
  if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
    return l2_score_f32(va, vector_at(index, b), index->dim);
  }
  return dot_f32(va, vector_at(index, b), index->dim);
}

void graph_begin_visit(MemoryIndex* index);
void graph_add_candidate(MemoryIndex* index, uint32_t capacity, uint32_t slot, float score,
                         uint32_t* candidate_count);
bool graph_pop_candidate(MemoryIndex* index, uint32_t* candidate_count, uint32_t* out_slot,
                         float* out_score);
inline void graph_mark_visited(MemoryIndex* index, uint32_t slot);
inline bool graph_was_visited(const MemoryIndex* index, uint32_t slot);
void graph_query_begin_visit(const MemoryIndex* index, GraphSearchScratch* scratch);
void graph_query_add_candidate(MemoryIndex* index, GraphSearchScratch* scratch, uint32_t capacity,
                               uint32_t slot, float score, uint32_t* candidate_count);
bool graph_query_pop_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                               uint32_t* candidate_count, uint32_t* out_slot, float* out_score);
inline void graph_query_mark_visited(GraphSearchScratch* scratch, uint32_t slot);
inline bool graph_query_was_visited(const GraphSearchScratch* scratch, uint32_t slot);
void insert_graph_build_candidate(MemoryIndex* index, uint32_t capacity, uint32_t* filled,
                                  uint32_t slot, float score);
bool insert_graph_query_top_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                                      uint32_t capacity, uint32_t* filled, uint32_t slot,
                                      float score);
bool graph_neighbor_diverse(MemoryIndex* index, uint32_t candidate_slot, float candidate_score,
                            const uint32_t* neighbors, uint32_t count);
bool graph_neighbor_diverse_except(MemoryIndex* index, uint32_t candidate_slot,
                                   float candidate_score, const uint32_t* neighbors, uint32_t count,
                                   uint32_t except_slot);

void refine_graph_neighbor_list(MemoryIndex* index, uint32_t owner_slot, uint32_t candidate_slot,
                                float candidate_score, uint32_t level) {
  if (owner_slot == candidate_slot) {
    return;
  }

  uint32_t* neighbors = graph_neighbors_at_level(index, owner_slot, level);
  uint8_t& count = graph_neighbor_count_ref(index, owner_slot, level);
  const uint32_t capacity = graph_neighbor_capacity_at_level(index, level);
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == candidate_slot) {
      return;
    }
  }
  if (count < capacity) {
    neighbors[count] = candidate_slot;
    ++count;
    return;
  }

  uint32_t weakest_pos = 0;
  uint32_t weakest_slot = neighbors[weakest_pos];
  float weakest_score = score_pair(index, owner_slot, weakest_slot);
  for (uint32_t i = 1; i < count; ++i) {
    const uint32_t neighbor = neighbors[i];
    const float score = score_pair(index, owner_slot, neighbor);
    if (graph_candidate_worse(index, score, neighbor, weakest_score, weakest_slot)) {
      weakest_pos = i;
      weakest_score = score;
      weakest_slot = neighbor;
    }
  }
  if (!graph_candidate_better(index, candidate_score, candidate_slot, weakest_score,
                              weakest_slot)) {
    return;
  }
  if (!graph_neighbor_diverse_except(index, candidate_slot, candidate_score, neighbors, count,
                                     weakest_slot)) {
    return;
  }
  neighbors[weakest_pos] = candidate_slot;
}

void force_graph_neighbor(MemoryIndex* index, uint32_t owner_slot, uint32_t neighbor_slot,
                          uint32_t position, uint32_t level) {
  if (owner_slot == neighbor_slot || position >= graph_neighbor_capacity_at_level(index, level)) {
    return;
  }
  uint32_t* neighbors = graph_neighbors_at_level(index, owner_slot, level);
  uint8_t& count = graph_neighbor_count_ref(index, owner_slot, level);
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == neighbor_slot) {
      return;
    }
  }
  if (position >= count) {
    neighbors[count] = neighbor_slot;
    ++count;
    return;
  }
  neighbors[position] = neighbor_slot;
}

void insert_graph_build_candidate(MemoryIndex* index, uint32_t capacity, uint32_t* filled,
                                  uint32_t slot, float score) {
  uint32_t pos = *filled;
  if (pos < capacity) {
    ++(*filled);
  } else if (score <= index->graph_scratch_scores[capacity - 1u]) {
    return;
  } else {
    pos = capacity - 1u;
  }

  while (pos > 0 && score > index->graph_scratch_scores[pos - 1u]) {
    index->graph_scratch_scores[pos] = index->graph_scratch_scores[pos - 1u];
    index->graph_scratch_slots[pos] = index->graph_scratch_slots[pos - 1u];
    --pos;
  }
  index->graph_scratch_scores[pos] = score;
  index->graph_scratch_slots[pos] = slot;
}

bool graph_neighbor_selected(const uint32_t* neighbors, uint32_t count, uint32_t slot) {
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == slot) {
      return true;
    }
  }
  return false;
}

bool graph_neighbor_diverse(MemoryIndex* index, uint32_t candidate_slot, float candidate_score,
                            const uint32_t* neighbors, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    if (score_pair(index, candidate_slot, neighbors[i]) > candidate_score) {
      return false;
    }
  }
  return true;
}

bool graph_neighbor_diverse_except(MemoryIndex* index, uint32_t candidate_slot,
                                   float candidate_score, const uint32_t* neighbors, uint32_t count,
                                   uint32_t except_slot) {
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] == except_slot) {
      continue;
    }
    if (score_pair(index, candidate_slot, neighbors[i]) > candidate_score) {
      return false;
    }
  }
  return true;
}

void graph_select_neighbors(MemoryIndex* index, uint32_t owner_slot, uint32_t level,
                            uint32_t candidate_count, uint32_t* neighbors, float* selected_scores,
                            uint32_t* out_count, uint32_t selection_capacity) {
  (void)level;
  uint32_t selected = 0;
  const uint32_t capacity = selection_capacity;
  for (uint32_t i = 0; i < candidate_count && selected < capacity; ++i) {
    const uint32_t candidate = index->graph_scratch_slots[i];
    const float candidate_score = index->graph_scratch_scores[i];
    if (candidate == owner_slot) {
      continue;
    }
    if (graph_neighbor_diverse(index, candidate, candidate_score, neighbors, selected)) {
      neighbors[selected] = candidate;
      selected_scores[selected] = candidate_score;
      ++selected;
    }
  }

  for (uint32_t i = 0; i < candidate_count && selected < capacity; ++i) {
    const uint32_t candidate = index->graph_scratch_slots[i];
    if (candidate == owner_slot || graph_neighbor_selected(neighbors, selected, candidate)) {
      continue;
    }
    neighbors[selected] = candidate;
    selected_scores[selected] = index->graph_scratch_scores[i];
    ++selected;
  }
  *out_count = selected;
}

bool insert_graph_query_top_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                                      uint32_t capacity, uint32_t* filled, uint32_t slot,
                                      float score) {
  uint32_t count = *filled;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, scratch->top_scores[parent], scratch->top_slots[parent],
                                  score, slot)) {
        break;
      }
      scratch->top_scores[pos] = scratch->top_scores[parent];
      scratch->top_slots[pos] = scratch->top_slots[parent];
      pos = parent;
    }
    scratch->top_scores[pos] = score;
    scratch->top_slots[pos] = slot;
    *filled = count + 1u;
    return true;
  }

  if (!graph_candidate_better(index, score, slot, scratch->top_scores[0], scratch->top_slots[0])) {
    return false;
  }

  uint32_t pos = 0;
  for (;;) {
    const uint32_t left = (pos << 1u) + 1u;
    if (left >= count) {
      break;
    }
    const uint32_t right = left + 1u;
    uint32_t child = left;
    if (right < count &&
        graph_candidate_worse(index, scratch->top_scores[right], scratch->top_slots[right],
                              scratch->top_scores[left], scratch->top_slots[left])) {
      child = right;
    }
    if (!graph_candidate_worse(index, scratch->top_scores[child], scratch->top_slots[child], score,
                               slot)) {
      break;
    }
    scratch->top_scores[pos] = scratch->top_scores[child];
    scratch->top_slots[pos] = scratch->top_slots[child];
    pos = child;
  }
  scratch->top_scores[pos] = score;
  scratch->top_slots[pos] = slot;
  return true;
}

void remove_graph_query_worst_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                                        uint32_t* filled) {
  uint32_t count = *filled;
  if (count == 0) {
    return;
  }

  --count;
  if (count != 0) {
    const uint32_t slot = scratch->top_slots[count];
    const float score = scratch->top_scores[count];
    uint32_t pos = 0;
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count &&
          graph_candidate_worse(index, scratch->top_scores[right], scratch->top_slots[right],
                                scratch->top_scores[left], scratch->top_slots[left])) {
        child = right;
      }
      if (!graph_candidate_worse(index, scratch->top_scores[child], scratch->top_slots[child],
                                 score, slot)) {
        break;
      }
      scratch->top_scores[pos] = scratch->top_scores[child];
      scratch->top_slots[pos] = scratch->top_slots[child];
      pos = child;
    }
    scratch->top_scores[pos] = score;
    scratch->top_slots[pos] = slot;
  }
  *filled = count;
}

uint32_t graph_f32_rerank_capacity(uint32_t top_k, uint32_t top_count) {
  uint32_t requested = kGraphF32RerankMinCandidates;
  if (top_k <= kU32Max / kGraphF32RerankTopKMultiplier) {
    const uint32_t top_k_capacity = top_k * kGraphF32RerankTopKMultiplier;
    if (top_k_capacity > requested) {
      requested = top_k_capacity;
    }
  }
  return requested < top_count ? requested : top_count;
}

void trim_graph_query_top_candidates(MemoryIndex* index, GraphSearchScratch* scratch,
                                     uint32_t* top_count, uint32_t target_count) {
  while (*top_count > target_count) {
    remove_graph_query_worst_candidate(index, scratch, top_count);
  }
}

void graph_collect_neighbors_exact(MemoryIndex* index, uint32_t slot, uint32_t level,
                                   uint32_t* filled) {
  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t other = active_slot_at(index, active_pos);
    if (other == slot || index->slots[other].occupied == 0 || index->graph_levels[other] < level) {
      continue;
    }
    ++index->graph_build_candidate_visits;
    insert_graph_build_candidate(index, index->graph_scratch_capacity, filled, other,
                                 score_pair(index, slot, other));
  }
}

void graph_search_layer_pair(MemoryIndex* index, uint32_t slot, uint32_t level, uint32_t entry,
                             uint32_t capacity, uint32_t* filled) {
  if (entry == kU32Max || entry == slot || index->slots[entry].occupied == 0 ||
      index->graph_levels[entry] < level) {
    return;
  }

  graph_begin_visit(index);
  uint32_t candidate_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_mark_visited(index, entry);
  const float entry_score = score_pair(index, slot, entry);
  insert_graph_build_candidate(index, capacity, filled, entry, entry_score);
  graph_add_candidate(index, candidate_capacity, entry, entry_score, &candidate_count);

  while (candidate_count != 0) {
    uint32_t current = kU32Max;
    float current_score = kWorstScore;
    if (!graph_pop_candidate(index, &candidate_count, &current, &current_score)) {
      break;
    }
    if (*filled == capacity && !graph_candidate_better(index, current_score, current,
                                                       index->graph_scratch_scores[capacity - 1u],
                                                       index->graph_scratch_slots[capacity - 1u])) {
      break;
    }

    ++index->graph_build_candidate_visits;

    const uint32_t* neighbors = graph_neighbors_at_level(index, current, level);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, current, level);
    for (uint32_t i = 0; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (neighbor == slot || graph_was_visited(index, neighbor) ||
          index->slots[neighbor].occupied == 0 || index->graph_levels[neighbor] < level) {
        continue;
      }
      graph_mark_visited(index, neighbor);
      const float score = score_pair(index, slot, neighbor);
      if (*filled < capacity ||
          graph_candidate_better(index, score, neighbor, index->graph_scratch_scores[capacity - 1u],
                                 index->graph_scratch_slots[capacity - 1u])) {
        graph_add_candidate(index, candidate_capacity, neighbor, score, &candidate_count);
        insert_graph_build_candidate(index, capacity, filled, neighbor, score);
      }
    }
  }
}

uint32_t graph_greedy_closest_pair(MemoryIndex* index, uint32_t slot, uint32_t entry,
                                   uint32_t begin_level, uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score = score_pair(index, slot, closest);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = graph_neighbors_at_level(index, closest, level);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = neighbors[i];
        if (index->slots[candidate].occupied == 0 || index->graph_levels[candidate] < level) {
          continue;
        }
        const float score = score_pair(index, slot, candidate);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

void graph_search_layer_query(MemoryIndex* index, GraphSearchScratch* scratch, const float* query,
                              float query_scale, uint32_t entry, uint32_t capacity,
                              uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_query_mark_visited(scratch, entry);
  const float entry_score = score_slot(index, query, entry, query_scale);
  insert_graph_query_top_candidate(index, scratch, capacity, &top_count, entry, entry_score);
  graph_query_add_candidate(index, scratch, candidate_capacity, entry, entry_score,
                            &candidate_count);

  while (candidate_count != 0) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!graph_query_pop_candidate(index, scratch, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == capacity &&
        !graph_candidate_better(index, slot_score, slot, scratch->top_scores[0],
                                scratch->top_slots[0])) {
      break;
    }

    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
    const uint32_t prefetch_limit = neighbor_count > kGraphNeighborPrefetchDistance
                                        ? neighbor_count - kGraphNeighborPrefetchDistance
                                        : 0u;
    uint32_t i = 0;
    for (; i < prefetch_limit; ++i) {
      const uint32_t neighbor = neighbors[i];
      prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score = score_slot(index, query, neighbor, query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
    for (; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score = score_slot(index, query, neighbor, query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
  }

  *out_top_count = top_count;
}

template <CompactByteQueryStorage Storage>
void graph_search_layer_compact_query(MemoryIndex* index, GraphSearchScratch* scratch,
                                      const int8_t* query, int32_t query_sum, float query_scale,
                                      float cosine_query_scale, uint32_t entry, uint32_t capacity,
                                      uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_query_mark_visited(scratch, entry);
  const float entry_score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                              entry, cosine_query_scale);
  insert_graph_query_top_candidate(index, scratch, capacity, &top_count, entry, entry_score);
  graph_query_add_candidate(index, scratch, candidate_capacity, entry, entry_score,
                            &candidate_count);

  while (candidate_count != 0) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!graph_query_pop_candidate(index, scratch, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == capacity &&
        !graph_candidate_better(index, slot_score, slot, scratch->top_scores[0],
                                scratch->top_slots[0])) {
      break;
    }

    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
    const uint32_t prefetch_limit = neighbor_count > kGraphNeighborPrefetchDistance
                                        ? neighbor_count - kGraphNeighborPrefetchDistance
                                        : 0u;
    uint32_t i = 0;
    for (; i < prefetch_limit; ++i) {
      const uint32_t neighbor = neighbors[i];
      prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                            neighbor, cosine_query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
    for (; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                            neighbor, cosine_query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
  }

  *out_top_count = top_count;
}

void graph_search_layer_compact_query_i16(MemoryIndex* index, GraphSearchScratch* scratch,
                                          const int16_t* query, float query_scale,
                                          float cosine_query_scale, uint32_t entry,
                                          uint32_t capacity, uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity = graph_candidate_search_capacity(index, capacity);
  graph_query_mark_visited(scratch, entry);
  const float entry_score =
      score_slot_compact_query_i16(index, query, query_scale, entry, cosine_query_scale);
  insert_graph_query_top_candidate(index, scratch, capacity, &top_count, entry, entry_score);
  graph_query_add_candidate(index, scratch, candidate_capacity, entry, entry_score,
                            &candidate_count);

  while (candidate_count != 0) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!graph_query_pop_candidate(index, scratch, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == capacity &&
        !graph_candidate_better(index, slot_score, slot, scratch->top_scores[0],
                                scratch->top_slots[0])) {
      break;
    }

    const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
    const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
    const uint32_t prefetch_limit = neighbor_count > kGraphNeighborPrefetchDistance
                                        ? neighbor_count - kGraphNeighborPrefetchDistance
                                        : 0u;
    uint32_t i = 0;
    for (; i < prefetch_limit; ++i) {
      const uint32_t neighbor = neighbors[i];
      prefetch_slot_vector(index, neighbors[i + kGraphNeighborPrefetchDistance]);
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score =
          score_slot_compact_query_i16(index, query, query_scale, neighbor, cosine_query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
    for (; i < neighbor_count; ++i) {
      const uint32_t neighbor = neighbors[i];
      if (graph_query_was_visited(scratch, neighbor)) {
        continue;
      }
      graph_query_mark_visited(scratch, neighbor);
      const float score =
          score_slot_compact_query_i16(index, query, query_scale, neighbor, cosine_query_scale);
      if (insert_graph_query_top_candidate(index, scratch, capacity, &top_count, neighbor, score)) {
        graph_query_add_candidate(index, scratch, candidate_capacity, neighbor, score,
                                  &candidate_count);
      }
    }
  }

  *out_top_count = top_count;
}

uint32_t graph_greedy_closest_query(MemoryIndex* index, const float* query, float query_scale,
                                    uint32_t entry, uint32_t begin_level, uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score = score_slot(index, query, closest, query_scale);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = graph_neighbors_at_level(index, closest, level);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = neighbors[i];
        if (index->slots[candidate].occupied == 0 || index->graph_levels[candidate] < level) {
          continue;
        }
        const float score = score_slot(index, query, candidate, query_scale);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

uint32_t graph_greedy_closest_compact_query_i16(MemoryIndex* index, const int16_t* query,
                                                float query_scale, float cosine_query_scale,
                                                uint32_t entry, uint32_t begin_level,
                                                uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score =
      score_slot_compact_query_i16(index, query, query_scale, closest, cosine_query_scale);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = graph_neighbors_at_level(index, closest, level);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = neighbors[i];
        if (index->slots[candidate].occupied == 0 || index->graph_levels[candidate] < level) {
          continue;
        }
        const float score =
            score_slot_compact_query_i16(index, query, query_scale, candidate, cosine_query_scale);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

template <CompactByteQueryStorage Storage>
uint32_t graph_greedy_closest_compact_query(MemoryIndex* index, const int8_t* query,
                                            int32_t query_sum, float query_scale,
                                            float cosine_query_scale, uint32_t entry,
                                            uint32_t begin_level, uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                          closest, cosine_query_scale);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = graph_neighbors_at_level(index, closest, level);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = neighbors[i];
        if (index->slots[candidate].occupied == 0 || index->graph_levels[candidate] < level) {
          continue;
        }
        const float score = score_slot_compact_query<Storage>(index, query, query_sum, query_scale,
                                                              candidate, cosine_query_scale);
        if (score > closest_score) {
          closest = candidate;
          closest_score = score;
          changed = true;
        }
      }
    }
  }
  return closest;
}

void graph_connect_slot_impl(MemoryIndex* index, uint32_t slot) {
  if (!graph_enabled(index) || index->count <= 1u) {
    index->graph_entry_slot = slot;
    index->graph_max_level = graph_enabled(index) ? index->graph_levels[slot] : 0;
    return;
  }

  uint32_t entry = index->graph_entry_slot;
  const uint32_t slot_level = index->graph_levels[slot];
  if (entry == kU32Max || index->slots[entry].occupied == 0) {
    entry = active_slot_at(index, 0);
  }
  if (index->graph_max_level > slot_level) {
    entry = graph_greedy_closest_pair(index, slot, entry, index->graph_max_level, slot_level);
  }

  uint32_t level = slot_level < index->graph_max_level ? slot_level : index->graph_max_level;
  for (;;) {
    uint32_t candidate_count = 0;
    if (index->count <= index->graph_search_capacity) {
      graph_collect_neighbors_exact(index, slot, level, &candidate_count);
    } else {
      graph_search_layer_pair(index, slot, level, entry, index->graph_search_capacity,
                              &candidate_count);
    }

    const uint32_t next_entry = candidate_count != 0 ? index->graph_scratch_slots[0] : entry;
    uint32_t* neighbors = graph_neighbors_at_level(index, slot, level);
    uint32_t filled = 0;
    const uint32_t level_capacity = graph_neighbor_capacity_at_level(index, level);
    const uint32_t outgoing_capacity = graph_outgoing_capacity_at_level(index, level);
    float selected_scores[kGraphMaxBaseNeighbors];
    graph_select_neighbors(index, slot, level, candidate_count, neighbors, selected_scores, &filled,
                           outgoing_capacity);
    graph_neighbor_count_ref(index, slot, level) = static_cast<uint8_t>(filled);
    for (uint32_t i = 0; i < filled; ++i) {
      refine_graph_neighbor_list(index, neighbors[i], slot, selected_scores[i], level);
    }
    if (level == 0 && kGraphLongLinkCount != 0 &&
        level_capacity == index->graph_neighbor_capacity && index->count > level_capacity &&
        level_capacity > kGraphLongLinkCount) {
      const uint32_t long_links =
          kGraphLongLinkCount < level_capacity ? kGraphLongLinkCount : level_capacity;
      const uint32_t stride = index->count / long_links;
      const uint32_t first_long_pos = level_capacity - long_links;
      for (uint32_t i = 0; i < long_links; ++i) {
        const uint32_t linked = active_slot_at(index, i * stride);
        force_graph_neighbor(index, slot, linked, first_long_pos + i, level);
        force_graph_neighbor(index, linked, slot, first_long_pos + i, level);
      }
    }
    if (candidate_count != 0) {
      entry = next_entry;
    }
    if (level == 0) {
      break;
    }
    --level;
  }
  if (slot_level > index->graph_max_level) {
    index->graph_max_level = slot_level;
    index->graph_entry_slot = slot;
  }
}

void graph_rebuild_impl(MemoryIndex* index) {
  if (!graph_enabled(index)) {
    return;
  }
  std::memset(index->graph_neighbor_counts, 0,
              sizeof(uint8_t) * index->capacity * index->graph_level_capacity);
  index->graph_entry_slot = kU32Max;
  index->graph_max_level = 0;
  index->graph_build_score_evals = 0;
  index->graph_build_candidate_visits = 0;
  const uint32_t final_count = index->count;
  index->count = 0;
  for (uint32_t active_pos = 0; active_pos < final_count; ++active_pos) {
    ++index->count;
    graph_connect_slot_impl(index, active_slot_at(index, active_pos));
  }
  index->count = final_count;
}

inline void graph_mark_visited(MemoryIndex* index, uint32_t slot) {
  index->graph_visited[slot] = index->graph_visit_generation;
}

inline bool graph_was_visited(const MemoryIndex* index, uint32_t slot) {
  return index->graph_visited[slot] == index->graph_visit_generation;
}

void graph_begin_visit(MemoryIndex* index) {
  ++index->graph_visit_generation;
  index->graph_candidate_worst_valid = 0;
  if (index->graph_visit_generation == 0) {
    std::memset(index->graph_visited, 0, sizeof(uint16_t) * index->capacity);
    index->graph_visit_generation = kGraphVisitGenerationStart;
  }
}

void graph_refresh_worst_candidate(MemoryIndex* index, uint32_t count) {
  uint32_t worst_pos = 0;
  float worst_score = index->graph_candidate_scores[0];
  for (uint32_t i = 1; i < count; ++i) {
    if (graph_candidate_worse(index, index->graph_candidate_scores[i], index->graph_candidates[i],
                              worst_score, index->graph_candidates[worst_pos])) {
      worst_score = index->graph_candidate_scores[i];
      worst_pos = i;
    }
  }
  index->graph_candidate_worst_pos = worst_pos;
  index->graph_candidate_worst_slot = index->graph_candidates[worst_pos];
  index->graph_candidate_worst_score = worst_score;
  index->graph_candidate_worst_valid = 1;
}

void graph_add_candidate(MemoryIndex* index, uint32_t capacity, uint32_t slot, float score,
                         uint32_t* candidate_count) {
  uint32_t count = *candidate_count;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, score, slot, index->graph_candidate_scores[parent],
                                  index->graph_candidates[parent])) {
        break;
      }
      index->graph_candidates[pos] = index->graph_candidates[parent];
      index->graph_candidate_scores[pos] = index->graph_candidate_scores[parent];
      pos = parent;
    }
    index->graph_candidates[pos] = slot;
    index->graph_candidate_scores[pos] = score;
    *candidate_count = count + 1u;
    return;
  }

  if (index->graph_candidate_worst_valid == 0) {
    graph_refresh_worst_candidate(index, count);
  }
  if (graph_candidate_better(index, score, slot, index->graph_candidate_worst_score,
                             index->graph_candidate_worst_slot)) {
    uint32_t pos = index->graph_candidate_worst_pos;
    bool moved_up = false;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, score, slot, index->graph_candidate_scores[parent],
                                  index->graph_candidates[parent])) {
        break;
      }
      index->graph_candidates[pos] = index->graph_candidates[parent];
      index->graph_candidate_scores[pos] = index->graph_candidate_scores[parent];
      pos = parent;
      moved_up = true;
    }
    if (!moved_up) {
      for (;;) {
        const uint32_t left = (pos << 1u) + 1u;
        if (left >= count) {
          break;
        }
        const uint32_t right = left + 1u;
        uint32_t child = left;
        if (right < count && graph_candidate_better(index, index->graph_candidate_scores[right],
                                                    index->graph_candidates[right],
                                                    index->graph_candidate_scores[left],
                                                    index->graph_candidates[left])) {
          child = right;
        }
        if (!graph_candidate_better(index, index->graph_candidate_scores[child],
                                    index->graph_candidates[child], score, slot)) {
          break;
        }
        index->graph_candidates[pos] = index->graph_candidates[child];
        index->graph_candidate_scores[pos] = index->graph_candidate_scores[child];
        pos = child;
      }
    }
    index->graph_candidates[pos] = slot;
    index->graph_candidate_scores[pos] = score;
    index->graph_candidate_worst_valid = 0;
  }
}

bool graph_pop_candidate(MemoryIndex* index, uint32_t* candidate_count, uint32_t* out_slot,
                         float* out_score) {
  uint32_t count = *candidate_count;
  if (count == 0) {
    return false;
  }

  *out_slot = index->graph_candidates[0];
  *out_score = index->graph_candidate_scores[0];
  --count;
  if (count != 0) {
    const uint32_t slot = index->graph_candidates[count];
    const float score = index->graph_candidate_scores[count];
    uint32_t pos = 0;
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count && graph_candidate_better(index, index->graph_candidate_scores[right],
                                                  index->graph_candidates[right],
                                                  index->graph_candidate_scores[left],
                                                  index->graph_candidates[left])) {
        child = right;
      }
      if (!graph_candidate_better(index, index->graph_candidate_scores[child],
                                  index->graph_candidates[child], score, slot)) {
        break;
      }
      index->graph_candidates[pos] = index->graph_candidates[child];
      index->graph_candidate_scores[pos] = index->graph_candidate_scores[child];
      pos = child;
    }
    index->graph_candidates[pos] = slot;
    index->graph_candidate_scores[pos] = score;
  }
  *candidate_count = count;
  index->graph_candidate_worst_valid = 0;
  return true;
}

inline void graph_query_mark_visited(GraphSearchScratch* scratch, uint32_t slot) {
  scratch->visited[slot] = scratch->visit_generation;
}

inline bool graph_query_was_visited(const GraphSearchScratch* scratch, uint32_t slot) {
  return scratch->visited[slot] == scratch->visit_generation;
}

void graph_query_begin_visit(const MemoryIndex* index, GraphSearchScratch* scratch) {
  ++scratch->visit_generation;
  scratch->candidate_worst_valid = 0;
  if (scratch->visit_generation == 0) {
    std::memset(scratch->visited, 0, sizeof(uint16_t) * index->capacity);
    scratch->visit_generation = kGraphVisitGenerationStart;
  }
}

void graph_query_refresh_worst_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                                         uint32_t count) {
  uint32_t worst_pos = 0;
  float worst_score = scratch->candidate_scores[0];
  for (uint32_t i = 1; i < count; ++i) {
    if (graph_candidate_worse(index, scratch->candidate_scores[i], scratch->candidates[i],
                              worst_score, scratch->candidates[worst_pos])) {
      worst_score = scratch->candidate_scores[i];
      worst_pos = i;
    }
  }
  scratch->candidate_worst_pos = worst_pos;
  scratch->candidate_worst_slot = scratch->candidates[worst_pos];
  scratch->candidate_worst_score = worst_score;
  scratch->candidate_worst_valid = 1;
}

void graph_query_add_candidate(MemoryIndex* index, GraphSearchScratch* scratch, uint32_t capacity,
                               uint32_t slot, float score, uint32_t* candidate_count) {
  uint32_t count = *candidate_count;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, score, slot, scratch->candidate_scores[parent],
                                  scratch->candidates[parent])) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[parent];
      scratch->candidate_scores[pos] = scratch->candidate_scores[parent];
      pos = parent;
    }
    scratch->candidates[pos] = slot;
    scratch->candidate_scores[pos] = score;
    *candidate_count = count + 1u;
    return;
  }

  if (scratch->candidate_worst_valid == 0) {
    graph_query_refresh_worst_candidate(index, scratch, count);
  }
  if (graph_candidate_better(index, score, slot, scratch->candidate_worst_score,
                             scratch->candidate_worst_slot)) {
    const uint32_t worst_pos = scratch->candidate_worst_pos;
    uint32_t pos = worst_pos;
    bool moved_up = false;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!graph_candidate_better(index, score, slot, scratch->candidate_scores[parent],
                                  scratch->candidates[parent])) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[parent];
      scratch->candidate_scores[pos] = scratch->candidate_scores[parent];
      pos = parent;
      moved_up = true;
    }
    if (!moved_up) {
      for (;;) {
        const uint32_t left = (pos << 1u) + 1u;
        if (left >= count) {
          break;
        }
        const uint32_t right = left + 1u;
        uint32_t child = left;
        if (right < count &&
            graph_candidate_better(index, scratch->candidate_scores[right],
                                   scratch->candidates[right], scratch->candidate_scores[left],
                                   scratch->candidates[left])) {
          child = right;
        }
        if (!graph_candidate_better(index, scratch->candidate_scores[child],
                                    scratch->candidates[child], score, slot)) {
          break;
        }
        scratch->candidates[pos] = scratch->candidates[child];
        scratch->candidate_scores[pos] = scratch->candidate_scores[child];
        pos = child;
      }
    }
    scratch->candidates[pos] = slot;
    scratch->candidate_scores[pos] = score;
    scratch->candidate_worst_valid = 0;
  }
}

bool graph_query_pop_candidate(MemoryIndex* index, GraphSearchScratch* scratch,
                               uint32_t* candidate_count, uint32_t* out_slot, float* out_score) {
  uint32_t count = *candidate_count;
  if (count == 0) {
    return false;
  }

  *out_slot = scratch->candidates[0];
  *out_score = scratch->candidate_scores[0];
  --count;
  if (count != 0) {
    const uint32_t slot = scratch->candidates[count];
    const float score = scratch->candidate_scores[count];
    uint32_t pos = 0;
    for (;;) {
      const uint32_t left = (pos << 1u) + 1u;
      if (left >= count) {
        break;
      }
      const uint32_t right = left + 1u;
      uint32_t child = left;
      if (right < count && graph_candidate_better(
                               index, scratch->candidate_scores[right], scratch->candidates[right],
                               scratch->candidate_scores[left], scratch->candidates[left])) {
        child = right;
      }
      if (!graph_candidate_better(index, scratch->candidate_scores[child],
                                  scratch->candidates[child], score, slot)) {
        break;
      }
      scratch->candidates[pos] = scratch->candidates[child];
      scratch->candidate_scores[pos] = scratch->candidate_scores[child];
      pos = child;
    }
    scratch->candidates[pos] = slot;
    scratch->candidate_scores[pos] = score;
  }
  *candidate_count = count;
  scratch->candidate_worst_valid = 0;
  return true;
}

void memory_search_graph_with_scratch(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                      const float* query, AstralMemorySearchResult* out_results,
                                      uint32_t* out_count, GraphSearchScratch* scratch) {
  if (!graph_enabled(index) || index->count == 0 || desc->group_id != ASTRAL_MEMORY_GROUP_ANY ||
      index->graph_entry_slot == kU32Max || index->slots[index->graph_entry_slot].occupied == 0) {
    memory_search_flat_fallback(index, desc, query, out_results, out_count);
    return;
  }
  if (compact_graph_exact_search_preferred(index)) {
    memory_search_flat_fallback(index, desc, query, out_results, out_count);
    return;
  }

  alignas(kVectorStorageAlign) float normalized_query[kMaxDim];
  if (!compact_storage(index) && index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    normalize_f32_vector(normalized_query, query, index->dim);
    query = normalized_query;
  }
  const float query_scale = compact_storage(index) && index->metric == ASTRAL_MEMORY_METRIC_COSINE
                                ? cosine_scale(query, index->dim)
                                : 1.0f;
  graph_query_begin_visit(index, scratch);
  if (scratch->visited == index->graph_visited) {
    index->graph_visit_generation = scratch->visit_generation;
  }
  const uint32_t search_capacity = graph_search_for_query(index, desc);

  uint32_t filled = 0;
  uint32_t top_count = 0;
  if (compact_storage(index) && i16_storage(index)) {
    alignas(kVectorStorageAlign) int16_t compact_query[kMaxDim];
    float compact_query_scale = 1.0f;
    quantize_e3m2_vector(compact_query, &compact_query_scale, query, index->dim);
    compact_query_scale *= kE3M2InvScale;
    const uint32_t entry = index->graph_max_level != 0
                               ? graph_greedy_closest_compact_query_i16(
                                     index, compact_query, compact_query_scale, query_scale,
                                     index->graph_entry_slot, index->graph_max_level, 0)
                               : index->graph_entry_slot;
    graph_search_layer_compact_query_i16(index, scratch, compact_query, compact_query_scale,
                                         query_scale, entry, search_capacity, &top_count);
  } else if (compact_storage(index) && !i16_storage(index)) {
    alignas(kVectorStorageAlign) int8_t compact_query[kMaxDim];
    float compact_query_scale = 1.0f;
    float compact_cosine_query_scale = query_scale;
    quantize_compact_query(index, compact_query, &compact_query_scale, query);
    if (e5m2_storage(index)) {
      const uint32_t entry =
          index->graph_max_level != 0
              ? graph_greedy_closest_compact_query<CompactByteQueryStorage::e5m2>(
                    index, compact_query, 0, compact_query_scale, compact_cosine_query_scale,
                    index->graph_entry_slot, index->graph_max_level, 0)
              : index->graph_entry_slot;
      graph_search_layer_compact_query<CompactByteQueryStorage::e5m2>(
          index, scratch, compact_query, 0, compact_query_scale, compact_cosine_query_scale, entry,
          search_capacity, &top_count);
    } else {
      const int32_t compact_query_sum = sum_i8(compact_query, index->dim);
      const uint32_t entry =
          index->graph_max_level != 0
              ? graph_greedy_closest_compact_query<CompactByteQueryStorage::q8>(
                    index, compact_query, compact_query_sum, compact_query_scale,
                    compact_cosine_query_scale, index->graph_entry_slot, index->graph_max_level, 0)
              : index->graph_entry_slot;
      graph_search_layer_compact_query<CompactByteQueryStorage::q8>(
          index, scratch, compact_query, compact_query_sum, compact_query_scale,
          compact_cosine_query_scale, entry, search_capacity, &top_count);
    }
  } else {
    const uint32_t entry =
        index->graph_max_level != 0
            ? graph_greedy_closest_query(index, query, query_scale, index->graph_entry_slot,
                                         index->graph_max_level, 0)
            : index->graph_entry_slot;
    graph_search_layer_query(index, scratch, query, query_scale, entry, search_capacity,
                             &top_count);
  }

  if (f32_rerank_storage(index)) {
    const uint32_t rerank_capacity = graph_f32_rerank_capacity(desc->top_k, top_count);
    trim_graph_query_top_candidates(index, scratch, &top_count, rerank_capacity);
  }

  for (uint32_t i = 0; i < top_count; ++i) {
    const uint32_t slot = scratch->top_slots[i];
    const MemorySlot& s = index->slots[slot];
    const float score = compact_storage(index) ? score_slot(index, query, slot, query_scale)
                                               : scratch->top_scores[i];
    AstralMemorySearchResult candidate{};
    fill_result(&candidate, s, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  if (compact_storage(index) && !f32_rerank_storage(index)) {
    for (uint32_t i = 0; i < top_count; ++i) {
      const uint32_t slot = scratch->top_slots[i];
      const uint32_t* neighbors = graph_neighbors_at_level(index, slot, 0);
      const uint32_t neighbor_count = graph_neighbor_count_at_level(index, slot, 0);
      for (uint32_t neighbor_i = 0; neighbor_i < neighbor_count; ++neighbor_i) {
        const uint32_t neighbor = neighbors[neighbor_i];
        if (graph_query_was_visited(scratch, neighbor)) {
          continue;
        }
        graph_query_mark_visited(scratch, neighbor);
        const MemorySlot& s = index->slots[neighbor];
        const float score = score_slot(index, query, neighbor, query_scale);
        AstralMemorySearchResult candidate{};
        fill_result(&candidate, s, score);
        insert_result(out_results, desc->top_k, &filled, candidate);
      }
    }
  }
  *out_count = filled;
}

void memory_search_graph_impl(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                         AstralMemorySearchResult* out_results, uint32_t* out_count) {
  GraphSearchScratch scratch{index->graph_candidates,
                             index->graph_candidate_scores,
                             index->graph_scratch_slots,
                             index->graph_scratch_scores,
                             index->graph_visited,
                             index->graph_visit_generation,
                             0u,
                             kU32Max,
                             kWorstScore,
                             0u};
  memory_search_graph_with_scratch(index, desc, query, out_results, out_count, &scratch);
}

void graph_search_scratch_free_impl(const MemoryIndex* index, GraphSearchScratch* scratch) {
  if (scratch->candidates != nullptr) {
    core::runtime_free_array(scratch->candidates, index->graph_candidate_capacity);
    scratch->candidates = nullptr;
  }
  if (scratch->candidate_scores != nullptr) {
    core::runtime_free_array(scratch->candidate_scores, index->graph_candidate_capacity);
    scratch->candidate_scores = nullptr;
  }
  if (scratch->top_slots != nullptr) {
    core::runtime_free_array(scratch->top_slots, index->graph_scratch_capacity);
    scratch->top_slots = nullptr;
  }
  if (scratch->top_scores != nullptr) {
    core::runtime_free_array(scratch->top_scores, index->graph_scratch_capacity);
    scratch->top_scores = nullptr;
  }
  if (scratch->visited != nullptr) {
    core::runtime_free_array(scratch->visited, index->capacity);
    scratch->visited = nullptr;
  }
  scratch->visit_generation = 0;
}

bool graph_search_scratch_alloc_impl(const MemoryIndex* index, GraphSearchScratch* scratch) {
  *scratch = GraphSearchScratch{};
  scratch->candidates = core::runtime_alloc_array<uint32_t>(index->graph_candidate_capacity);
  scratch->candidate_scores = core::runtime_alloc_array<float>(index->graph_candidate_capacity);
  scratch->top_slots = core::runtime_alloc_array<uint32_t>(index->graph_scratch_capacity);
  scratch->top_scores = core::runtime_alloc_array<float>(index->graph_scratch_capacity);
  scratch->visited = core::runtime_alloc_array<uint16_t>(index->capacity);
  if (scratch->candidates == nullptr || scratch->candidate_scores == nullptr ||
      scratch->top_slots == nullptr || scratch->top_scores == nullptr ||
      scratch->visited == nullptr) {
    graph_search_scratch_free_impl(index, scratch);
    return false;
  }
  std::memset(scratch->visited, 0, sizeof(uint16_t) * index->capacity);
  return true;
}

void memory_search_graph_batch_work(void* user) {
  MemoryGraphSearchBatchJob* job = static_cast<MemoryGraphSearchBatchJob*>(user);
  for (uint32_t i = job->begin; i < job->end; ++i) {
    uint32_t count = 0;
    const float* query = job->queries + static_cast<size_t>(i) * job->index->dim;
    AstralMemorySearchResult* results =
        job->out_results + static_cast<size_t>(i) * job->desc->top_k;
    memory_search_graph_with_scratch(job->index, job->desc, query, results, &count, job->scratch);
    job->out_counts[i] = count;
  }
  memory_parallel_job_complete(job->remaining);
}

bool memory_search_graph_batch_parallel_impl(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                        const float* queries, uint32_t query_count,
                                        AstralMemorySearchResult* out_results,
                                        uint32_t* out_counts) {
  if (!graph_enabled(index) || desc->group_id != ASTRAL_MEMORY_GROUP_ANY ||
      query_count < kMemorySearchBatchParallelMinQueries || !core::runtime_initialized()) {
    return false;
  }

  const uint32_t runtime_threads = core::runtime_thread_count();
  const bool on_worker = core::runtime_on_worker_thread();
  if (runtime_threads < 2u) {
    return false;
  }

  uint32_t worker_count = on_worker ? runtime_threads - 1u : runtime_threads;
  if (worker_count > kMemorySearchBatchParallelMaxWorkers) {
    worker_count = kMemorySearchBatchParallelMaxWorkers;
  }
  if (worker_count < 2u) {
    return false;
  }
  if (worker_count > query_count) {
    worker_count = query_count;
  }

  MemoryGraphSearchBatchJob jobs[kMemorySearchBatchParallelMaxWorkers];
  GraphSearchScratch fallback_scratch[kMemorySearchBatchParallelMaxWorkers];
  bool claimed_index_scratch = false;
  if (index->graph_batch_scratch_count >= worker_count &&
      index->graph_batch_scratch_claimed.exchange(1u, std::memory_order_acquire) == 0u) {
    claimed_index_scratch = true;
    for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
      jobs[worker_i].scratch = &index->graph_batch_scratch[worker_i];
    }
  } else {
    uint32_t allocated = 0;
    for (; allocated < worker_count; ++allocated) {
      if (!graph_search_scratch_alloc_impl(index, &fallback_scratch[allocated])) {
        for (uint32_t i = 0; i < allocated; ++i) {
          graph_search_scratch_free_impl(index, &fallback_scratch[i]);
        }
        return false;
      }
      jobs[allocated].scratch = &fallback_scratch[allocated];
    }
  }

  std::atomic<uint32_t> remaining(worker_count);
  const uint32_t current_worker = on_worker ? core::runtime_worker_id() : kU32Max;
  for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
    const uint32_t begin =
        static_cast<uint32_t>((static_cast<uint64_t>(query_count) * worker_i) / worker_count);
    const uint32_t end = static_cast<uint32_t>(
        (static_cast<uint64_t>(query_count) * (worker_i + 1u)) / worker_count);
    jobs[worker_i].index = index;
    jobs[worker_i].desc = desc;
    jobs[worker_i].queries = queries;
    jobs[worker_i].out_results = out_results;
    jobs[worker_i].out_counts = out_counts;
    jobs[worker_i].begin = begin;
    jobs[worker_i].end = end;
    jobs[worker_i].remaining = &remaining;

    uint32_t target_worker = worker_i;
    if (on_worker && target_worker >= current_worker) {
      ++target_worker;
    }
    const AstralErr err =
        core::submit_work_affine(target_worker, memory_search_graph_batch_work, &jobs[worker_i]);
    if (err != ASTRAL_OK) {
      memory_search_graph_batch_work(&jobs[worker_i]);
    }
  }

  while (remaining.load(std::memory_order_acquire) != 0) {
    astral::platform::cpu_wait_for_event();
  }

  for (uint32_t worker_i = 0; worker_i < worker_count; ++worker_i) {
    if (!claimed_index_scratch) {
      graph_search_scratch_free_impl(index, jobs[worker_i].scratch);
    }
  }
  if (claimed_index_scratch) {
    index->graph_batch_scratch_claimed.store(0u, std::memory_order_release);
  }
  return true;
}

} // namespace

void graph_connect_slot(MemoryIndex* index, uint32_t slot) {
  graph_connect_slot_impl(index, slot);
}

void graph_rebuild(MemoryIndex* index) {
  graph_rebuild_impl(index);
}

void memory_search_graph(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                         const float* query, AstralMemorySearchResult* out_results,
                         uint32_t* out_count) {
  memory_search_graph_impl(index, desc, query, out_results, out_count);
}

void graph_search_scratch_free(const MemoryIndex* index, GraphSearchScratch* scratch) {
  graph_search_scratch_free_impl(index, scratch);
}

bool graph_search_scratch_alloc(const MemoryIndex* index, GraphSearchScratch* scratch) {
  return graph_search_scratch_alloc_impl(index, scratch);
}

bool memory_search_graph_batch_parallel(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                                        const float* queries, uint32_t query_count,
                                        AstralMemorySearchResult* out_results,
                                        uint32_t* out_counts) {
  return memory_search_graph_batch_parallel_impl(index, desc, queries, query_count, out_results,
                                                 out_counts);
}

} // namespace astral::inference
