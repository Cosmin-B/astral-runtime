#include "memory_index.hpp"

#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"

#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace astral::inference {

namespace {

constexpr uint32_t kMaxDim = 8192;
constexpr uint32_t kSaveMagic = 0x414D454Du;
constexpr uint32_t kSaveVersion = 1;
constexpr uint32_t kU32Max = 0xFFFFFFFFu;
constexpr uint32_t kNoResults = 0;

struct MemorySlot {
  AstralMemoryRecord record;
  float norm;
  uint8_t occupied;
};

struct SaveHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t dim;
  uint32_t count;
  uint32_t metric;
  uint32_t index_kind;
  uint32_t _reserved0;
  uint32_t _reserved1;
};

inline bool metric_valid(AstralMemoryMetric metric) {
  return metric == ASTRAL_MEMORY_METRIC_DOT || metric == ASTRAL_MEMORY_METRIC_COSINE ||
         metric == ASTRAL_MEMORY_METRIC_L2;
}

inline bool desc_valid(const AstralMemoryIndexDesc* desc) {
  return desc != nullptr && desc->size == sizeof(AstralMemoryIndexDesc) && desc->dim != 0 &&
         desc->dim <= kMaxDim && desc->capacity != 0 &&
         desc->index_kind == ASTRAL_MEMORY_INDEX_FLAT && metric_valid(desc->metric);
}

float dot_f32(const float* a, const float* b, uint32_t dim) {
#if defined(__AVX2__)
  __m256 acc = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8u <= dim; i += 8u) {
    const __m256 av = _mm256_loadu_ps(a + i);
    const __m256 bv = _mm256_loadu_ps(b + i);
    acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#else
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4u <= dim; i += 4u) {
    sum0 += a[i] * b[i];
    sum1 += a[i + 1u] * b[i + 1u];
    sum2 += a[i + 2u] * b[i + 2u];
    sum3 += a[i + 3u] * b[i + 3u];
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

float l2_score_f32(const float* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return -sum;
}

inline float vector_norm(const float* v, uint32_t dim) {
  const float sq = dot_f32(v, v, dim);
  return sq > 0.0f ? std::sqrt(sq) : 0.0f;
}

} // namespace

struct MemoryIndex {
  AstralHandle handle;
  uint32_t dim;
  uint32_t capacity;
  uint32_t count;
  AstralMemoryMetric metric;
  MemorySlot* slots;
  float* vectors;
};

struct MemorySearchCursor {
  AstralHandle handle;
  uint32_t capacity;
  uint32_t count;
  uint32_t offset;
  AstralMemorySearchResult* results;
};

AstralHandle memory_handle(MemoryIndex* index) {
  return index != nullptr ? index->handle : 0;
}

AstralHandle memory_search_cursor_handle(MemorySearchCursor* cursor) {
  return cursor != nullptr ? cursor->handle : 0;
}

namespace {

inline float* vector_at(MemoryIndex* index, uint32_t slot) {
  return index->vectors + static_cast<size_t>(slot) * index->dim;
}

inline const float* vector_at(const MemoryIndex* index, uint32_t slot) {
  return index->vectors + static_cast<size_t>(slot) * index->dim;
}

uint32_t find_slot_by_key(const MemoryIndex* index, uint64_t key) {
  for (uint32_t i = 0; i < index->capacity; ++i) {
    if (index->slots[i].occupied != 0 && index->slots[i].record.key == key) {
      return i;
    }
  }
  return kU32Max;
}

uint32_t find_free_slot(const MemoryIndex* index) {
  for (uint32_t i = 0; i < index->capacity; ++i) {
    if (index->slots[i].occupied == 0) {
      return i;
    }
  }
  return kU32Max;
}

float score_vector(const MemoryIndex* index, const float* query, float query_norm, uint32_t slot) {
  const float* v = vector_at(index, slot);
  if (index->metric == ASTRAL_MEMORY_METRIC_L2) {
    return l2_score_f32(query, v, index->dim);
  }
  const float dot = dot_f32(query, v, index->dim);
  if (index->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    const float denom = query_norm * index->slots[slot].norm;
    return denom > 0.0f ? dot / denom : 0.0f;
  }
  return dot;
}

void insert_result(AstralMemorySearchResult* results, uint32_t top_k, uint32_t* filled,
                   const AstralMemorySearchResult& candidate) {
  uint32_t pos = *filled;
  if (pos < top_k) {
    ++(*filled);
  } else if (top_k != 0 && candidate.score <= results[top_k - 1u].score) {
    return;
  } else {
    pos = top_k - 1u;
  }

  while (pos > 0 && candidate.score > results[pos - 1u].score) {
    results[pos] = results[pos - 1u];
    --pos;
  }
  results[pos] = candidate;
}

void destroy_allocations(MemoryIndex* index) {
  if (index->slots != nullptr) {
    core::runtime_free_array(index->slots, index->capacity);
    index->slots = nullptr;
  }
  if (index->vectors != nullptr) {
    core::runtime_free_array(index->vectors, index->capacity * index->dim);
    index->vectors = nullptr;
  }
}

void destroy_search_cursor(MemorySearchCursor* cursor) {
  if (cursor == nullptr) {
    return;
  }
  if (cursor->results != nullptr) {
    core::runtime_free_array(cursor->results, cursor->capacity);
    cursor->results = nullptr;
  }
  core::runtime_delete(cursor);
}

} // namespace

AstralErr memory_create(const AstralMemoryIndexDesc* desc, MemoryIndex** out_index) {
  if (!desc_valid(desc) || out_index == nullptr) {
    return ASTRAL_E_INVALID;
  }
  if (desc->capacity > kU32Max / desc->dim) {
    return ASTRAL_E_NOMEM;
  }

  MemoryIndex* index = core::runtime_new<MemoryIndex>();
  if (index == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  index->handle = 0;
  index->dim = desc->dim;
  index->capacity = desc->capacity;
  index->count = 0;
  index->metric = desc->metric;
  index->slots = core::runtime_alloc_array<MemorySlot>(desc->capacity);
  index->vectors = core::runtime_alloc_array<float>(desc->capacity * desc->dim);
  if (index->slots == nullptr || index->vectors == nullptr) {
    destroy_allocations(index);
    core::runtime_delete(index);
    return ASTRAL_E_NOMEM;
  }

  std::memset(index->slots, 0, sizeof(MemorySlot) * desc->capacity);
  const AstralHandle handle = core::register_handle(core::HandleKind::MemoryIndex, index);
  if (handle == 0) {
    destroy_allocations(index);
    core::runtime_delete(index);
    return ASTRAL_E_BUSY;
  }

  index->handle = handle;
  *out_index = index;
  return ASTRAL_OK;
}

void memory_destroy(MemoryIndex* index) {
  if (index == nullptr) {
    return;
  }
  core::unregister_handle(index->handle, core::HandleKind::MemoryIndex);
  destroy_allocations(index);
  core::runtime_delete(index);
}

AstralErr memory_count(MemoryIndex* index, uint32_t* out_count) {
  if (index == nullptr || out_count == nullptr) {
    return ASTRAL_E_INVALID;
  }
  *out_count = index->count;
  return ASTRAL_OK;
}

AstralErr memory_clear(MemoryIndex* index) {
  if (index == nullptr) {
    return ASTRAL_E_INVALID;
  }
  std::memset(index->slots, 0, sizeof(MemorySlot) * index->capacity);
  index->count = 0;
  return ASTRAL_OK;
}

AstralErr memory_add_batch(MemoryIndex* index, const AstralMemoryRecord* records,
                           const float* vectors, uint32_t count) {
  if (index == nullptr || records == nullptr || vectors == nullptr || count == 0) {
    return ASTRAL_E_INVALID;
  }

  for (uint32_t i = 0; i < count; ++i) {
    if (records[i].size != sizeof(AstralMemoryRecord) || records[i].key == 0) {
      return ASTRAL_E_INVALID;
    }
    uint32_t slot = find_slot_by_key(index, records[i].key);
    if (slot == kU32Max) {
      slot = find_free_slot(index);
      if (slot == kU32Max) {
        return ASTRAL_E_NOMEM;
      }
      index->slots[slot].occupied = 1;
      ++index->count;
    }
    index->slots[slot].record = records[i];
    float* dst = vector_at(index, slot);
    const float* src = vectors + static_cast<size_t>(i) * index->dim;
    std::memcpy(dst, src, sizeof(float) * index->dim);
    index->slots[slot].norm = vector_norm(dst, index->dim);
  }
  return ASTRAL_OK;
}

AstralErr memory_remove(MemoryIndex* index, uint64_t key) {
  if (index == nullptr || key == 0) {
    return ASTRAL_E_INVALID;
  }
  const uint32_t slot = find_slot_by_key(index, key);
  if (slot == kU32Max) {
    return ASTRAL_E_NOT_FOUND;
  }
  index->slots[slot] = MemorySlot{};
  --index->count;
  return ASTRAL_OK;
}

AstralErr memory_search(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                        AstralMemorySearchResult* out_results, uint32_t max_results,
                        uint32_t* out_count) {
  if (index == nullptr || desc == nullptr || desc->size != sizeof(AstralMemorySearchDesc) ||
      query == nullptr || out_count == nullptr || desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }
  if (max_results < desc->top_k || out_results == nullptr) {
    return ASTRAL_E_NOMEM;
  }

  uint32_t filled = 0;
  const float query_norm =
      index->metric == ASTRAL_MEMORY_METRIC_COSINE ? vector_norm(query, index->dim) : 0.0f;
  for (uint32_t slot = 0; slot < index->capacity; ++slot) {
    const MemorySlot& s = index->slots[slot];
    if (s.occupied == 0) {
      continue;
    }
    if (desc->group_id != ASTRAL_MEMORY_GROUP_ANY && s.record.group_id != desc->group_id) {
      continue;
    }

    AstralMemorySearchResult candidate{};
    candidate.size = sizeof(AstralMemorySearchResult);
    candidate.group_id = s.record.group_id;
    candidate.key = s.record.key;
    candidate.document_id = s.record.document_id;
    candidate.chunk_id = s.record.chunk_id;
    candidate.flags = s.record.flags;
    candidate.score = score_vector(index, query, query_norm, slot);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }

  *out_count = filled;
  return ASTRAL_OK;
}

AstralErr memory_search_begin(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* query, MemorySearchCursor** out_cursor) {
  if (index == nullptr || desc == nullptr || desc->size != sizeof(AstralMemorySearchDesc) ||
      query == nullptr || out_cursor == nullptr || desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }

  const uint32_t capacity = desc->top_k < index->count ? desc->top_k : index->count;
  MemorySearchCursor* cursor = core::runtime_new<MemorySearchCursor>();
  if (cursor == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  cursor->handle = 0;
  cursor->capacity = capacity;
  cursor->count = kNoResults;
  cursor->offset = 0;
  cursor->results = core::runtime_alloc_array<AstralMemorySearchResult>(capacity);
  if (capacity != kNoResults && cursor->results == nullptr) {
    destroy_search_cursor(cursor);
    return ASTRAL_E_NOMEM;
  }

  if (capacity != kNoResults) {
    AstralMemorySearchDesc bounded_desc = *desc;
    bounded_desc.top_k = capacity;
    uint32_t result_count = 0;
    const AstralErr search_err =
        memory_search(index, &bounded_desc, query, cursor->results, capacity, &result_count);
    if (search_err != ASTRAL_OK) {
      destroy_search_cursor(cursor);
      return search_err;
    }
    cursor->count = result_count;
  }

  const AstralHandle handle = core::register_handle(core::HandleKind::MemorySearch, cursor);
  if (handle == 0) {
    destroy_search_cursor(cursor);
    return ASTRAL_E_BUSY;
  }

  cursor->handle = handle;
  *out_cursor = cursor;
  return ASTRAL_OK;
}

AstralErr memory_search_fetch(MemorySearchCursor* cursor, AstralMemorySearchResult* out_results,
                              uint32_t max_results, uint32_t* out_count) {
  if (cursor == nullptr || out_count == nullptr || (max_results != 0 && out_results == nullptr)) {
    return ASTRAL_E_INVALID;
  }
  if (max_results == 0 || cursor->offset >= cursor->count) {
    *out_count = kNoResults;
    return ASTRAL_OK;
  }

  const uint32_t remaining = cursor->count - cursor->offset;
  const uint32_t to_copy = remaining < max_results ? remaining : max_results;
  std::memcpy(out_results, cursor->results + cursor->offset,
              sizeof(AstralMemorySearchResult) * to_copy);
  cursor->offset += to_copy;
  *out_count = to_copy;
  return ASTRAL_OK;
}

void memory_search_end(MemorySearchCursor* cursor) {
  if (cursor == nullptr) {
    return;
  }
  core::unregister_handle(cursor->handle, core::HandleKind::MemorySearch);
  destroy_search_cursor(cursor);
}

AstralErr memory_save_size(MemoryIndex* index, uint64_t* out_bytes) {
  if (index == nullptr || out_bytes == nullptr) {
    return ASTRAL_E_INVALID;
  }
  *out_bytes = sizeof(SaveHeader) +
               static_cast<uint64_t>(index->count) * sizeof(AstralMemoryRecord) +
               static_cast<uint64_t>(index->count) * index->dim * sizeof(float);
  return ASTRAL_OK;
}

AstralErr memory_save(MemoryIndex* index, AstralMutSpanU8 out_bytes, uint64_t* out_written) {
  if (index == nullptr || out_written == nullptr) {
    return ASTRAL_E_INVALID;
  }

  uint64_t need = 0;
  AstralErr err = memory_save_size(index, &need);
  if (err != ASTRAL_OK) {
    return err;
  }
  *out_written = need;
  if (out_bytes.data == nullptr || out_bytes.len < need) {
    return ASTRAL_E_NOMEM;
  }

  SaveHeader header{};
  header.magic = kSaveMagic;
  header.version = kSaveVersion;
  header.dim = index->dim;
  header.count = index->count;
  header.metric = index->metric;
  header.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

  uint8_t* cursor = out_bytes.data;
  std::memcpy(cursor, &header, sizeof(header));
  cursor += sizeof(header);

  for (uint32_t i = 0; i < index->capacity; ++i) {
    if (index->slots[i].occupied == 0) {
      continue;
    }
    std::memcpy(cursor, &index->slots[i].record, sizeof(AstralMemoryRecord));
    cursor += sizeof(AstralMemoryRecord);
    std::memcpy(cursor, vector_at(index, i), sizeof(float) * index->dim);
    cursor += sizeof(float) * index->dim;
  }
  return ASTRAL_OK;
}

AstralErr memory_load(const AstralMemoryIndexDesc* desc, AstralSpanU8 bytes,
                      MemoryIndex** out_index) {
  if (!desc_valid(desc) || bytes.data == nullptr || bytes.len < sizeof(SaveHeader) ||
      out_index == nullptr) {
    return ASTRAL_E_INVALID;
  }

  SaveHeader header{};
  std::memcpy(&header, bytes.data, sizeof(header));
  if (header.magic != kSaveMagic || header.version != kSaveVersion || header.dim != desc->dim ||
      header.metric != desc->metric || header.index_kind != desc->index_kind ||
      header.count > desc->capacity) {
    return ASTRAL_E_INVALID;
  }

  const uint64_t need = sizeof(SaveHeader) +
                        static_cast<uint64_t>(header.count) * sizeof(AstralMemoryRecord) +
                        static_cast<uint64_t>(header.count) * header.dim * sizeof(float);
  if (bytes.len < need) {
    return ASTRAL_E_INVALID;
  }

  MemoryIndex* index = nullptr;
  AstralErr err = memory_create(desc, &index);
  if (err != ASTRAL_OK) {
    return err;
  }

  const uint8_t* cursor = bytes.data + sizeof(SaveHeader);
  for (uint32_t i = 0; i < header.count; ++i) {
    AstralMemoryRecord record{};
    std::memcpy(&record, cursor, sizeof(record));
    cursor += sizeof(record);
    err = memory_add_batch(index, &record, reinterpret_cast<const float*>(cursor), 1);
    if (err != ASTRAL_OK) {
      memory_destroy(index);
      return err;
    }
    cursor += sizeof(float) * header.dim;
  }

  *out_index = index;
  return ASTRAL_OK;
}

} // namespace astral::inference
