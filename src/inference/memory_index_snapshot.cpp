#include "memory_index_internal.hpp"

#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"
#include "../platform/file_map.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace astral::inference {

struct SnapshotGraphLayout {
  SaveGraphHeader header;
  uint64_t levels_offset;
  uint64_t counts_offset;
  uint64_t neighbors_offset;
  uint32_t candidate_capacity;
  uint32_t scratch_capacity;
};

struct MemorySnapshotView {
  AstralHandle handle;
  platform::ReadOnlyFileMap map;
  AstralMemorySnapshotInfo info;
  SnapshotGraphLayout graph;
  GraphSearchScratch graph_scratch;
  uint8_t graph_ready;
  uint8_t e5m2_clamp_non_finite;
};

namespace {

constexpr uint32_t kGraphMaxNeighbors = 64;
constexpr uint32_t kGraphBaseNeighborMultiplier = 2;
constexpr uint32_t kGraphMaxBaseNeighbors = kGraphMaxNeighbors * kGraphBaseNeighborMultiplier;
constexpr uint32_t kGraphMaxLevels = 16;
constexpr uint32_t kGraphMinSearch = 4;
constexpr uint32_t kGraphCandidateReserveMultiplier = 4;
constexpr uint32_t kGraphNeighborPrefetchDistance = 2;
constexpr uint32_t kGraphF32RerankMinCandidates = 256;
constexpr uint32_t kGraphF32RerankTopKMultiplier = 32;
constexpr uint16_t kGraphVisitGenerationStart = 1;
constexpr uint64_t kBytesPerKiB = 1024;
constexpr uint64_t kBytesPerMiB = kBytesPerKiB * kBytesPerKiB;
constexpr uint64_t kGraphCompactExactSearchMaxBytes = 16 * kBytesPerMiB;
constexpr uint32_t kFlatQ8PrefetchDistance = 4;

inline bool metric_valid(AstralMemoryMetric metric) {
  return metric == ASTRAL_MEMORY_METRIC_DOT || metric == ASTRAL_MEMORY_METRIC_COSINE ||
         metric == ASTRAL_MEMORY_METRIC_L2;
}

inline bool index_kind_valid(AstralMemoryIndexKind kind) {
  return kind == ASTRAL_MEMORY_INDEX_FLAT || kind == ASTRAL_MEMORY_INDEX_GRAPH;
}

inline bool storage_kind_valid(AstralMemoryStorageKind kind) {
  return kind == ASTRAL_MEMORY_STORAGE_F32 || kind == ASTRAL_MEMORY_STORAGE_Q8 ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 || kind == ASTRAL_MEMORY_STORAGE_F8_E5M2 ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E3M2 || kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E2M3_F32_RERANK ||
         kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK;
}

inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  const uint64_t mask = alignment - 1u;
  return (value + mask) & ~mask;
}

inline uint64_t save_payload_align(uint32_t version) {
  return version >= kSaveVersionAlignedVectorData ? static_cast<uint64_t>(kVectorStorageAlign) : 1u;
}


inline uint32_t save_graph_count_bytes_impl(uint32_t version) {
  return version >= kSaveVersionCompactGraphCounts ? sizeof(uint8_t) : sizeof(uint32_t);
}

bool memory_save_layout_impl(uint32_t version, uint32_t dim, uint32_t count,
                        AstralMemoryStorageKind storage, uint32_t index_kind,
                        uint32_t graph_base_neighbors, uint32_t graph_neighbors,
                        uint32_t graph_levels, SaveLayout* out_layout);

inline uint64_t memory_save_byte_count_impl(const MemoryIndex* index) {
  SaveLayout layout{};
  const bool has_graph =
      index->index_kind == ASTRAL_MEMORY_INDEX_GRAPH && index->graph_neighbor_capacity != 0;
  const uint32_t graph_base_neighbors = has_graph ? index->graph_base_neighbor_capacity : 0u;
  const uint32_t graph_neighbors = has_graph ? index->graph_neighbor_capacity : 0u;
  const uint32_t graph_levels = has_graph ? index->graph_level_capacity : 0u;
  if (!memory_save_layout_impl(kSaveVersion, index->dim, index->count, index->storage_kind,
                          index->index_kind, graph_base_neighbors, graph_neighbors, graph_levels,
                          &layout)) {
    return 0;
  }
  return layout.total_bytes;
}

bool memory_save_layout_impl(uint32_t version, uint32_t dim, uint32_t count,
                        AstralMemoryStorageKind storage, uint32_t index_kind,
                        uint32_t graph_base_neighbors, uint32_t graph_neighbors,
                        uint32_t graph_levels, SaveLayout* out_layout) {
  if (out_layout == nullptr || !storage_kind_valid(storage)) {
    return false;
  }
  SaveLayout layout{};
  const bool compact = compact_storage_kind(storage);
  const uint64_t compact_element_bytes =
      i16_storage_kind(storage) ? sizeof(int16_t) : sizeof(int8_t);
  layout.record_offset = sizeof(SaveHeader);
  layout.record_stride = sizeof(AstralMemoryRecord);
  if (version >= kSaveVersionModernLayout) {
    uint64_t cursor =
        layout.record_offset + static_cast<uint64_t>(count) * sizeof(AstralMemoryRecord);
    const uint64_t payload_align = save_payload_align(version);
    if (compact) {
      layout.scale_offset = cursor;
      layout.scale_stride = sizeof(float);
      cursor += static_cast<uint64_t>(count) * sizeof(float);
      if (version >= kSaveVersionCompactScoreScales) {
        layout.compact_score_scale_offset = cursor;
        layout.compact_score_scale_stride = sizeof(float);
        cursor += static_cast<uint64_t>(count) * sizeof(float);
      }
      cursor = align_up_u64(cursor, payload_align);
      layout.vector_offset = cursor;
      layout.vector_stride = static_cast<uint64_t>(dim) * compact_element_bytes;
      cursor += static_cast<uint64_t>(count) * layout.vector_stride;
      if (f32_rerank_storage_kind(storage)) {
        cursor = align_up_u64(cursor, payload_align);
        layout.rerank_vector_offset = cursor;
        layout.rerank_vector_stride = static_cast<uint64_t>(dim) * sizeof(float);
        cursor += static_cast<uint64_t>(count) * layout.rerank_vector_stride;
      }
    } else {
      cursor = align_up_u64(cursor, payload_align);
      layout.vector_offset = cursor;
      layout.vector_stride = static_cast<uint64_t>(dim) * sizeof(float);
      cursor += static_cast<uint64_t>(count) * layout.vector_stride;
    }
    cursor = align_up_u64(cursor, payload_align);
    layout.graph_offset = cursor;
  } else {
    layout.scale_offset = compact ? layout.record_offset + sizeof(AstralMemoryRecord) : 0;
    layout.scale_stride =
        compact ? sizeof(AstralMemoryRecord) + sizeof(float) + static_cast<uint64_t>(dim) : 0;
    layout.vector_offset =
        layout.record_offset + sizeof(AstralMemoryRecord) + (compact ? sizeof(float) : 0);
    layout.vector_stride =
        sizeof(AstralMemoryRecord) +
        (compact ? sizeof(float) + static_cast<uint64_t>(dim) * compact_element_bytes
                 : static_cast<uint64_t>(dim) * sizeof(float));
    layout.graph_offset =
        layout.record_offset + static_cast<uint64_t>(count) * layout.vector_stride;
  }
  if (index_kind == ASTRAL_MEMORY_INDEX_GRAPH && graph_neighbors != 0 && graph_levels != 0) {
    const uint64_t graph_header_bytes =
        version >= kSaveVersionGraphQuerySearch ? sizeof(SaveGraphHeader)
        : version >= kSaveVersionLegacyLayout   ? sizeof(SaveGraphHeaderV6)
                                                : sizeof(SaveGraphHeaderV3);
    const uint64_t neighbor_slots =
        static_cast<uint64_t>(count) * graph_base_neighbors +
        static_cast<uint64_t>(count) * (graph_levels - 1u) * graph_neighbors;
    layout.graph_bytes =
        graph_header_bytes + static_cast<uint64_t>(count) * sizeof(uint8_t) +
        static_cast<uint64_t>(count) * graph_levels * save_graph_count_bytes_impl(version) +
        neighbor_slots * sizeof(uint32_t);
  }
  layout.total_bytes = layout.graph_offset + layout.graph_bytes;
  *out_layout = layout;
  return true;
}

inline uint32_t save_graph_header_bytes_impl(uint32_t version) {
  return version >= kSaveVersionGraphQuerySearch ? sizeof(SaveGraphHeader)
         : version >= kSaveVersionLegacyLayout   ? sizeof(SaveGraphHeaderV6)
                                                 : sizeof(SaveGraphHeaderV3);
}

inline void memory_save_skip_to(uint8_t** cursor, uint8_t* target) {
  if (*cursor < target) {
    std::memset(*cursor, 0, static_cast<size_t>(target - *cursor));
    *cursor = target;
  }
}

void read_save_graph_header_impl(uint32_t version, const uint8_t* bytes, SaveGraphHeader* out_header) {
  *out_header = {};
  if (version >= kSaveVersionGraphQuerySearch) {
    std::memcpy(out_header, bytes, sizeof(SaveGraphHeader));
    return;
  }
  if (version >= kSaveVersionLegacyLayout) {
    SaveGraphHeaderV6 header_v6{};
    std::memcpy(&header_v6, bytes, sizeof(header_v6));
    out_header->flags = header_v6.flags;
    out_header->neighbor_capacity = header_v6.neighbor_capacity;
    out_header->base_neighbor_capacity = header_v6.base_neighbor_capacity;
    out_header->search_capacity = header_v6.search_capacity;
    out_header->query_search_capacity = header_v6.search_capacity;
    out_header->level_capacity = header_v6.level_capacity;
    out_header->max_level = header_v6.max_level;
    out_header->entry_active_pos = header_v6.entry_active_pos;
    return;
  }
  SaveGraphHeaderV3 header_v3{};
  std::memcpy(&header_v3, bytes, sizeof(header_v3));
  out_header->flags = header_v3.flags;
  out_header->neighbor_capacity = header_v3.neighbor_capacity;
  out_header->base_neighbor_capacity = header_v3.neighbor_capacity;
  out_header->search_capacity = header_v3.search_capacity;
  out_header->query_search_capacity = header_v3.search_capacity;
  out_header->level_capacity = header_v3.level_capacity;
  out_header->max_level = header_v3.max_level;
  out_header->entry_active_pos = header_v3.entry_active_pos;
}

inline bool snapshot_compact_graph_exact_search_preferred(const AstralMemorySnapshotInfo* info) {
  return compact_storage_kind(info->storage_kind) &&
         static_cast<uint64_t>(info->count) * static_cast<uint64_t>(info->dim) <=
             kGraphCompactExactSearchMaxBytes;
}

inline uint32_t snapshot_graph_f32_rerank_capacity(uint32_t top_k, uint32_t top_count) {
  uint32_t requested = kGraphF32RerankMinCandidates;
  if (top_k <= kU32Max / kGraphF32RerankTopKMultiplier) {
    const uint32_t top_k_capacity = top_k * kGraphF32RerankTopKMultiplier;
    if (top_k_capacity > requested) {
      requested = top_k_capacity;
    }
  }
  return requested < top_count ? requested : top_count;
}


void snapshot_graph_scratch_free(const AstralMemorySnapshotInfo* info,
                                 const SnapshotGraphLayout* graph, GraphSearchScratch* scratch) {
  if (scratch->candidates != nullptr) {
    core::runtime_free_array(scratch->candidates, graph->candidate_capacity);
    scratch->candidates = nullptr;
  }
  if (scratch->candidate_scores != nullptr) {
    core::runtime_free_array(scratch->candidate_scores, graph->candidate_capacity);
    scratch->candidate_scores = nullptr;
  }
  if (scratch->top_slots != nullptr) {
    core::runtime_free_array(scratch->top_slots, graph->scratch_capacity);
    scratch->top_slots = nullptr;
  }
  if (scratch->top_scores != nullptr) {
    core::runtime_free_array(scratch->top_scores, graph->scratch_capacity);
    scratch->top_scores = nullptr;
  }
  if (scratch->visited != nullptr) {
    core::runtime_free_array(scratch->visited, info->count);
    scratch->visited = nullptr;
  }
  scratch->visit_generation = 0;
}

bool snapshot_graph_scratch_alloc(const AstralMemorySnapshotInfo* info,
                                  const SnapshotGraphLayout* graph, GraphSearchScratch* scratch) {
  *scratch = GraphSearchScratch{};
  scratch->candidates = core::runtime_alloc_array<uint32_t>(graph->candidate_capacity);
  scratch->candidate_scores = core::runtime_alloc_array<float>(graph->candidate_capacity);
  scratch->top_slots = core::runtime_alloc_array<uint32_t>(graph->scratch_capacity);
  scratch->top_scores = core::runtime_alloc_array<float>(graph->scratch_capacity);
  scratch->visited = core::runtime_alloc_array<uint16_t>(info->count);
  if (scratch->candidates == nullptr || scratch->candidate_scores == nullptr ||
      scratch->top_slots == nullptr || scratch->top_scores == nullptr ||
      scratch->visited == nullptr) {
    snapshot_graph_scratch_free(info, graph, scratch);
    return false;
  }
  std::memset(scratch->visited, 0, sizeof(uint16_t) * info->count);
  return true;
}



} // namespace

uint64_t memory_save_byte_count(const MemoryIndex* index) {
  return memory_save_byte_count_impl(index);
}

uint32_t save_graph_count_bytes(uint32_t version) {
  return save_graph_count_bytes_impl(version);
}

bool memory_save_layout(uint32_t version, uint32_t dim, uint32_t count,
                        AstralMemoryStorageKind storage, uint32_t index_kind,
                        uint32_t graph_base_neighbors, uint32_t graph_neighbors,
                        uint32_t graph_levels, SaveLayout* out_layout) {
  return memory_save_layout_impl(version, dim, count, storage, index_kind,
                                 graph_base_neighbors, graph_neighbors, graph_levels,
                                 out_layout);
}

uint32_t save_graph_header_bytes(uint32_t version) {
  return save_graph_header_bytes_impl(version);
}

void read_save_graph_header(uint32_t version, const uint8_t* bytes,
                            SaveGraphHeader* out_header) {
  read_save_graph_header_impl(version, bytes, out_header);
}

AstralHandle memory_snapshot_view_handle(MemorySnapshotView* view) {
  return view != nullptr ? view->handle : 0;
}

AstralErr memory_save_size(MemoryIndex* index, uint64_t* out_bytes) {
  if (index == nullptr || out_bytes == nullptr) {
    return ASTRAL_E_INVALID;
  }
  *out_bytes = memory_save_byte_count_impl(index);
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
  SaveLayout layout{};
  if (!memory_save_layout_impl(kSaveVersion, index->dim, index->count, index->storage_kind,
                          index->index_kind,
                          graph_enabled(index) ? index->graph_base_neighbor_capacity : 0u,
                          graph_enabled(index) ? index->graph_neighbor_capacity : 0u,
                          graph_enabled(index) ? index->graph_level_capacity : 0u, &layout)) {
    return ASTRAL_E_INVALID;
  }

  SaveHeader header{};
  header.magic = kSaveMagic;
  header.version = kSaveVersion;
  header.dim = index->dim;
  header.count = index->count;
  header.metric = index->metric;
  header.index_kind = index->index_kind;
  header._reserved0 = index->storage_kind;
  header._reserved1 = graph_enabled(index) ? kSaveGraphTopologyFlag : 0;

  uint8_t* cursor = out_bytes.data;
  std::memcpy(cursor, &header, sizeof(header));
  cursor += sizeof(header);

  for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
    const uint32_t slot = active_slot_at(index, active_pos);
    std::memcpy(cursor, &index->slots[slot].record, sizeof(AstralMemoryRecord));
    cursor += sizeof(AstralMemoryRecord);
  }
  if (compact_storage(index)) {
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const float scale = index->q8_scales[slot];
      std::memcpy(cursor, &scale, sizeof(float));
      cursor += sizeof(float);
    }
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const float scale = index->compact_score_scales[slot];
      std::memcpy(cursor, &scale, sizeof(float));
      cursor += sizeof(float);
    }
    memory_save_skip_to(&cursor, out_bytes.data + layout.vector_offset);
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      const size_t bytes =
          static_cast<size_t>(index->dim) * (i16_storage(index) ? sizeof(int16_t) : sizeof(int8_t));
      std::memcpy(cursor,
                  i16_storage(index) ? static_cast<const void*>(i16_vector_at(index, slot))
                                     : static_cast<const void*>(q8_vector_at(index, slot)),
                  bytes);
      cursor += bytes;
    }
    if (f32_rerank_storage(index)) {
      memory_save_skip_to(&cursor, out_bytes.data + layout.rerank_vector_offset);
      for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
        const uint32_t slot = active_slot_at(index, active_pos);
        std::memcpy(cursor, vector_at(index, slot), sizeof(float) * index->dim);
        cursor += sizeof(float) * index->dim;
      }
    }
  } else {
    memory_save_skip_to(&cursor, out_bytes.data + layout.vector_offset);
    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      std::memcpy(cursor, vector_at(index, slot), sizeof(float) * index->dim);
      cursor += sizeof(float) * index->dim;
    }
  }

  memory_save_skip_to(&cursor, out_bytes.data + layout.graph_offset);
  if (graph_enabled(index)) {
    SaveGraphHeader graph_header{};
    graph_header.flags = kSaveGraphTopologyFlag;
    graph_header.neighbor_capacity = index->graph_neighbor_capacity;
    graph_header.base_neighbor_capacity = index->graph_base_neighbor_capacity;
    graph_header.search_capacity = index->graph_search_capacity;
    graph_header.query_search_capacity = index->graph_query_search_capacity;
    graph_header.level_capacity = index->graph_level_capacity;
    graph_header.max_level = index->graph_max_level;
    graph_header.entry_active_pos = index->graph_entry_slot != kU32Max
                                        ? index->slots[index->graph_entry_slot].active_pos
                                        : kU32Max;
    std::memcpy(cursor, &graph_header, sizeof(graph_header));
    cursor += sizeof(graph_header);

    for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
      const uint32_t slot = active_slot_at(index, active_pos);
      *cursor = index->graph_levels[slot];
      ++cursor;
    }
    for (uint32_t level = 0; level < index->graph_level_capacity; ++level) {
      for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
        const uint32_t slot = active_slot_at(index, active_pos);
        const uint8_t count =
            static_cast<uint8_t>(graph_neighbor_count_at_level(index, slot, level));
        std::memcpy(cursor, &count, sizeof(count));
        cursor += sizeof(count);
      }
    }
    for (uint32_t level = 0; level < index->graph_level_capacity; ++level) {
      for (uint32_t active_pos = 0; active_pos < index->count; ++active_pos) {
        const uint32_t slot = active_slot_at(index, active_pos);
        const uint32_t* neighbors = graph_neighbors_at_level(index, slot, level);
        const uint32_t capacity = graph_neighbor_capacity_at_level(index, level);
        for (uint32_t i = 0; i < capacity; ++i) {
          const uint32_t neighbor = neighbors[i];
          const uint32_t neighbor_pos = neighbor != kU32Max && neighbor < index->capacity &&
                                                index->slots[neighbor].occupied != 0
                                            ? index->slots[neighbor].active_pos
                                            : kU32Max;
          std::memcpy(cursor, &neighbor_pos, sizeof(neighbor_pos));
          cursor += sizeof(neighbor_pos);
        }
      }
    }
  }
  return ASTRAL_OK;
}

namespace {

struct SnapshotBytes {
  const uint8_t* data;
  uint64_t len;
};

AstralErr memory_snapshot_info_bytes(SnapshotBytes bytes, AstralMemorySnapshotInfo* out_info) {
  if (bytes.data == nullptr || bytes.len < sizeof(SaveHeader) || out_info == nullptr ||
      out_info->size != sizeof(AstralMemorySnapshotInfo)) {
    return ASTRAL_E_INVALID;
  }

  SaveHeader header{};
  std::memcpy(&header, bytes.data, sizeof(header));
  const bool version_valid =
      header.version == kSaveVersionF32 || header.version == kSaveVersionCompactStorage ||
      header.version == kSaveVersionGraphTopology || header.version == kSaveVersionLegacyLayout ||
      header.version == kSaveVersionModernLayout ||
      header.version == kSaveVersionNormalizedF32Cosine ||
      header.version == kSaveVersionGraphQuerySearch ||
      header.version == kSaveVersionCompactScoreScales ||
      header.version == kSaveVersionCompactGraphCounts || header.version == kSaveVersion;
  AstralMemoryStorageKind storage = static_cast<AstralMemoryStorageKind>(ASTRAL_MEMORY_STORAGE_F32);
  if (header.version >= kSaveVersionCompactStorage) {
    storage = static_cast<AstralMemoryStorageKind>(header._reserved0);
  }
  if (header.magic != kSaveMagic || !version_valid || header.dim == 0 || header.dim > kMaxDim ||
      !metric_valid(static_cast<AstralMemoryMetric>(header.metric)) ||
      !index_kind_valid(static_cast<AstralMemoryIndexKind>(header.index_kind)) ||
      !storage_kind_valid(storage)) {
    return ASTRAL_E_INVALID;
  }

  SaveLayout layout{};
  if (!memory_save_layout_impl(header.version, header.dim, header.count, storage, header.index_kind, 0,
                          0, 0, &layout) ||
      bytes.len < layout.graph_offset) {
    return ASTRAL_E_INVALID;
  }

  if (header.index_kind == ASTRAL_MEMORY_INDEX_GRAPH &&
      header._reserved1 == kSaveGraphTopologyFlag) {
    const uint32_t graph_header_bytes = save_graph_header_bytes_impl(header.version);
    if (bytes.len < layout.graph_offset + graph_header_bytes) {
      return ASTRAL_E_INVALID;
    }
    SaveGraphHeader graph_header{};
    read_save_graph_header_impl(header.version, bytes.data + layout.graph_offset, &graph_header);
    if (graph_header.flags != kSaveGraphTopologyFlag || graph_header.neighbor_capacity == 0 ||
        graph_header.neighbor_capacity > kGraphMaxNeighbors ||
        graph_header.base_neighbor_capacity == 0 ||
        graph_header.base_neighbor_capacity > kGraphMaxBaseNeighbors ||
        graph_header.base_neighbor_capacity < graph_header.neighbor_capacity ||
        graph_header.query_search_capacity == 0 || graph_header.level_capacity == 0 ||
        graph_header.level_capacity > kGraphMaxLevels) {
      return ASTRAL_E_INVALID;
    }
    if (!memory_save_layout_impl(header.version, header.dim, header.count, storage, header.index_kind,
                            graph_header.base_neighbor_capacity, graph_header.neighbor_capacity,
                            graph_header.level_capacity, &layout) ||
        bytes.len < layout.total_bytes) {
      return ASTRAL_E_INVALID;
    }
  } else if (bytes.len < layout.total_bytes) {
    return ASTRAL_E_INVALID;
  }

  out_info->version = header.version;
  out_info->dim = header.dim;
  out_info->count = header.count;
  out_info->metric = static_cast<AstralMemoryMetric>(header.metric);
  out_info->index_kind = static_cast<AstralMemoryIndexKind>(header.index_kind);
  out_info->storage_kind = storage;
  out_info->flags = header._reserved1;
  out_info->record_offset = layout.record_offset;
  out_info->record_stride = layout.record_stride;
  out_info->vector_offset = layout.vector_offset;
  out_info->vector_stride = layout.vector_stride;
  out_info->scale_offset = layout.scale_offset;
  out_info->scale_stride = layout.scale_stride;
  out_info->graph_offset = layout.graph_offset;
  out_info->graph_bytes = layout.graph_bytes;
  out_info->total_bytes = layout.total_bytes;
  return ASTRAL_OK;
}

bool snapshot_graph_layout(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                           SnapshotGraphLayout* out_graph) {
  if (bytes.data == nullptr || info == nullptr || out_graph == nullptr ||
      info->index_kind != ASTRAL_MEMORY_INDEX_GRAPH || info->flags != kSaveGraphTopologyFlag ||
      info->graph_bytes == 0) {
    return false;
  }
  const uint32_t graph_header_bytes = save_graph_header_bytes_impl(info->version);
  if (bytes.len < info->graph_offset + graph_header_bytes) {
    return false;
  }

  SaveGraphHeader graph_header{};
  read_save_graph_header_impl(info->version, bytes.data + info->graph_offset, &graph_header);
  if (graph_header.flags != kSaveGraphTopologyFlag || graph_header.neighbor_capacity == 0 ||
      graph_header.neighbor_capacity > kGraphMaxNeighbors ||
      graph_header.base_neighbor_capacity == 0 ||
      graph_header.base_neighbor_capacity > kGraphMaxBaseNeighbors ||
      graph_header.base_neighbor_capacity < graph_header.neighbor_capacity ||
      graph_header.query_search_capacity == 0 || graph_header.level_capacity == 0 ||
      graph_header.level_capacity > kGraphMaxLevels ||
      graph_header.max_level >= graph_header.level_capacity ||
      graph_header.entry_active_pos >= info->count) {
    return false;
  }

  const uint64_t level_count = static_cast<uint64_t>(info->count) * graph_header.level_capacity;
  const uint64_t neighbor_slots =
      static_cast<uint64_t>(info->count) * graph_header.base_neighbor_capacity +
      static_cast<uint64_t>(info->count) * (graph_header.level_capacity - 1u) *
          graph_header.neighbor_capacity;
  const uint32_t graph_count_bytes = save_graph_count_bytes_impl(info->version);
  const uint64_t graph_bytes = graph_header_bytes + static_cast<uint64_t>(info->count) +
                               level_count * graph_count_bytes + neighbor_slots * sizeof(uint32_t);
  if (bytes.len < info->graph_offset + graph_bytes) {
    return false;
  }

  SnapshotGraphLayout graph{};
  graph.header = graph_header;
  graph.levels_offset = info->graph_offset + graph_header_bytes;
  graph.counts_offset = graph.levels_offset + static_cast<uint64_t>(info->count);
  graph.neighbors_offset = graph.counts_offset + level_count * graph_count_bytes;
  for (uint32_t active_pos = 0; active_pos < info->count; ++active_pos) {
    if (bytes.data[graph.levels_offset + active_pos] >= graph_header.level_capacity) {
      return false;
    }
  }
  for (uint32_t level = 0; level < graph_header.level_capacity; ++level) {
    const uint32_t level_capacity =
        level == 0 ? graph_header.base_neighbor_capacity : graph_header.neighbor_capacity;
    for (uint32_t active_pos = 0; active_pos < info->count; ++active_pos) {
      uint32_t count = 0;
      const uint8_t* count_ptr =
          bytes.data + graph.counts_offset +
          (static_cast<uint64_t>(level) * info->count + active_pos) * graph_count_bytes;
      if (graph_count_bytes == sizeof(uint8_t)) {
        count = *count_ptr;
      } else {
        std::memcpy(&count, count_ptr, sizeof(count));
      }
      if (count > level_capacity) {
        return false;
      }
      const uint64_t neighbor_base =
          level == 0
              ? graph.neighbors_offset + static_cast<uint64_t>(active_pos) *
                                             graph_header.base_neighbor_capacity * sizeof(uint32_t)
              : graph.neighbors_offset +
                    (static_cast<uint64_t>(info->count) * graph_header.base_neighbor_capacity +
                     (static_cast<uint64_t>(level - 1u) * info->count + active_pos) *
                         graph_header.neighbor_capacity) *
                        sizeof(uint32_t);
      for (uint32_t neighbor_i = 0; neighbor_i < level_capacity; ++neighbor_i) {
        uint32_t neighbor = kU32Max;
        std::memcpy(&neighbor,
                    bytes.data + neighbor_base +
                        static_cast<uint64_t>(neighbor_i) * sizeof(uint32_t),
                    sizeof(neighbor));
        if (neighbor != kU32Max && neighbor >= info->count) {
          return false;
        }
      }
    }
  }
  uint32_t scratch_capacity = graph_header.query_search_capacity > graph_header.search_capacity
                                  ? graph_header.query_search_capacity
                                  : graph_header.search_capacity;
  if (scratch_capacity < kGraphMinSearch) {
    scratch_capacity = kGraphMinSearch;
  }
  if (scratch_capacity > info->count) {
    scratch_capacity = info->count;
  }
  uint32_t candidate_capacity = scratch_capacity;
  if (scratch_capacity <= kU32Max / kGraphCandidateReserveMultiplier) {
    candidate_capacity = scratch_capacity * kGraphCandidateReserveMultiplier;
  }
  if (candidate_capacity > info->count) {
    candidate_capacity = info->count;
  }
  graph.scratch_capacity = scratch_capacity;
  graph.candidate_capacity = candidate_capacity;
  *out_graph = graph;
  return true;
}

inline uint64_t snapshot_graph_level_offset(const AstralMemorySnapshotInfo* info,
                                            const SnapshotGraphLayout* graph, uint32_t active_pos,
                                            uint32_t level) {
  if (level == 0) {
    return graph->neighbors_offset + static_cast<uint64_t>(active_pos) *
                                         graph->header.base_neighbor_capacity * sizeof(uint32_t);
  }
  return graph->neighbors_offset +
         (static_cast<uint64_t>(info->count) * graph->header.base_neighbor_capacity +
          (static_cast<uint64_t>(level - 1u) * info->count + active_pos) *
              graph->header.neighbor_capacity) *
             sizeof(uint32_t);
}

inline uint8_t snapshot_graph_level(const uint8_t* bytes, const SnapshotGraphLayout* graph,
                                    uint32_t active_pos) {
  return bytes[graph->levels_offset + active_pos];
}

template <bool CompactCounts>
inline uint32_t
snapshot_graph_neighbor_count(const uint8_t* bytes, const AstralMemorySnapshotInfo* info,
                              const SnapshotGraphLayout* graph, uint32_t slot, uint32_t level) {
  const uint64_t ordinal = static_cast<uint64_t>(level) * info->count + slot;
  if constexpr (CompactCounts) {
    return bytes[graph->counts_offset + ordinal];
  } else {
    uint32_t count = 0;
    std::memcpy(&count, bytes + graph->counts_offset + ordinal * sizeof(uint32_t), sizeof(count));
    return count;
  }
}

inline uint32_t snapshot_graph_neighbor_at_base(const uint8_t* bytes, uint64_t neighbor_base,
                                                uint32_t neighbor_i) {
  uint32_t neighbor = kU32Max;
  std::memcpy(&neighbor,
              bytes + neighbor_base + static_cast<uint64_t>(neighbor_i) * sizeof(uint32_t),
              sizeof(neighbor));
  return neighbor;
}

struct SnapshotPreparedQuery {
  alignas(kVectorStorageAlign) float query[kMaxDim];
  alignas(kVectorStorageAlign) int8_t compact_query[kMaxDim];
  alignas(kVectorStorageAlign) int16_t compact_query_i16[kMaxDim];
  uint64_t score_vector_offset;
  uint64_t score_vector_stride;
  uint64_t compact_score_scale_offset;
  uint64_t compact_score_scale_stride;
  const E5m2Kernels* e5m2_kernels;
  float query_scale;
  float compact_query_scale;
  int32_t compact_query_sum;
  uint8_t compact;
  uint8_t use_f6;
  uint8_t use_f6_i16;
  uint8_t use_f8;
  uint8_t compact_i8_vectors_aligned;
  uint8_t compact_i16_vectors_aligned;
  uint8_t normalized_f32_cosine;
  uint8_t e5m2_clamp_non_finite;
};

void snapshot_prepare_query(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                            const float* query, uint8_t e5m2_clamp_non_finite,
                            SnapshotPreparedQuery* prepared) {
  *prepared = {};
  prepared->e5m2_kernels = select_e5m2_kernels();
  prepared->e5m2_clamp_non_finite = e5m2_clamp_non_finite;
  prepared->score_vector_offset = info->vector_offset;
  prepared->score_vector_stride = info->vector_stride;
  if (f32_rerank_storage_kind(info->storage_kind)) {
    SaveLayout layout{};
    if (memory_save_layout_impl(info->version, info->dim, info->count, info->storage_kind,
                           info->index_kind, 0, 0, 0, &layout)) {
      prepared->score_vector_offset = layout.rerank_vector_offset;
      prepared->score_vector_stride = layout.rerank_vector_stride;
    }
  }
  if (info->version >= kSaveVersionCompactScoreScales && compact_storage_kind(info->storage_kind)) {
    SaveLayout layout{};
    if (memory_save_layout_impl(info->version, info->dim, info->count, info->storage_kind,
                           info->index_kind, 0, 0, 0, &layout)) {
      prepared->compact_score_scale_offset = layout.compact_score_scale_offset;
      prepared->compact_score_scale_stride = layout.compact_score_scale_stride;
    }
  }
  prepared->compact = compact_storage_kind(info->storage_kind) ? 1u : 0u;
  prepared->normalized_f32_cosine = !prepared->compact &&
                                            info->metric == ASTRAL_MEMORY_METRIC_COSINE &&
                                            info->version >= kSaveVersionNormalizedF32Cosine
                                        ? 1u
                                        : 0u;
  if (prepared->normalized_f32_cosine != 0) {
    normalize_f32_vector(prepared->query, query, info->dim);
  } else {
    std::memcpy(prepared->query, query, sizeof(float) * info->dim);
  }
  prepared->query_scale =
      info->metric == ASTRAL_MEMORY_METRIC_COSINE && prepared->normalized_f32_cosine == 0
          ? cosine_scale(prepared->query, info->dim)
          : 1.0f;
  prepared->compact_query_scale = 1.0f;
  prepared->use_f6 = (info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3 ||
                      info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E2M3_F32_RERANK)
                         ? 1u
                         : 0u;
  prepared->use_f6_i16 = (info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E3M2 ||
                          info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK)
                             ? 1u
                             : 0u;
  prepared->use_f8 = (info->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2 ||
                      info->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK)
                         ? 1u
                         : 0u;
  if (!i16_storage_kind(info->storage_kind) && compact_storage_kind(info->storage_kind) &&
      info->version >= kSaveVersionAlignedVectorData &&
      (info->vector_stride & static_cast<uint64_t>(kVectorStorageAlign - 1u)) == 0) {
    const uintptr_t vector_address = reinterpret_cast<uintptr_t>(bytes.data + info->vector_offset);
    prepared->compact_i8_vectors_aligned =
        (vector_address & static_cast<uintptr_t>(kVectorStorageAlign - 1u)) == 0 ? 1u : 0u;
  }
  if (i16_storage_kind(info->storage_kind) && info->version >= kSaveVersionAlignedVectorData &&
      (info->vector_stride & static_cast<uint64_t>(kVectorStorageAlign - 1u)) == 0) {
    const uintptr_t vector_address = reinterpret_cast<uintptr_t>(bytes.data + info->vector_offset);
    prepared->compact_i16_vectors_aligned =
        (vector_address & static_cast<uintptr_t>(kVectorStorageAlign - 1u)) == 0 ? 1u : 0u;
  }
  if (info->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK) {
    quantize_q8_vector(prepared->compact_query, &prepared->compact_query_scale, prepared->query,
                       info->dim);
    prepared->compact_query_sum = sum_i8(prepared->compact_query, info->dim);
  } else if (prepared->use_f6 != 0) {
    quantize_e2m3_vector(prepared->compact_query, &prepared->compact_query_scale, prepared->query,
                         info->dim);
    prepared->compact_query_scale *= kE2M3InvScale;
    prepared->compact_query_sum = sum_i8(prepared->compact_query, info->dim);
  } else if (prepared->use_f6_i16 != 0) {
    quantize_e3m2_vector(prepared->compact_query_i16, &prepared->compact_query_scale,
                         prepared->query, info->dim);
    prepared->compact_query_scale *= kE3M2InvScale;
  } else if (prepared->use_f8 != 0) {
    quantize_e5m2_vector(prepared->compact_query, &prepared->compact_query_scale, prepared->query,
                         info->dim);
  }
}

inline const uint8_t* snapshot_record_ptr(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                          uint32_t active_pos) {
  return bytes.data + info->record_offset + static_cast<uint64_t>(active_pos) * info->record_stride;
}

inline const uint8_t* snapshot_vector_ptr(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                          uint32_t active_pos) {
  return bytes.data + info->vector_offset + static_cast<uint64_t>(active_pos) * info->vector_stride;
}

bool snapshot_e5m2_requires_clamp(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info) {
  if (info->storage_kind != ASTRAL_MEMORY_STORAGE_F8_E5M2 &&
      info->storage_kind != ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK) {
    return false;
  }

  constexpr uint64_t kExponentMask = 0x7c7c7c7c7c7c7c7cull;
  constexpr uint64_t kByteOnes = 0x0101010101010101ull;
  constexpr uint64_t kByteHighBits = 0x8080808080808080ull;
  for (uint32_t active_pos = 0; active_pos < info->count; ++active_pos) {
    const uint8_t* vector = snapshot_vector_ptr(bytes, info, active_pos);
    uint32_t dim_i = 0;
    for (; dim_i + sizeof(uint64_t) <= info->dim; dim_i += sizeof(uint64_t)) {
      uint64_t packed = 0;
      std::memcpy(&packed, vector + dim_i, sizeof(packed));
      const uint64_t exponent_delta = (packed & kExponentMask) ^ kExponentMask;
      if (((exponent_delta - kByteOnes) & ~exponent_delta & kByteHighBits) != 0) {
        return true;
      }
    }
    for (; dim_i < info->dim; ++dim_i) {
      if ((vector[dim_i] & 0x7cu) == 0x7cu) {
        return true;
      }
    }
  }
  return false;
}

inline const uint8_t* snapshot_score_vector_ptr(SnapshotBytes bytes,
                                                const SnapshotPreparedQuery* prepared,
                                                uint32_t active_pos) {
  return bytes.data + prepared->score_vector_offset +
         static_cast<uint64_t>(active_pos) * prepared->score_vector_stride;
}

inline float snapshot_stored_scale(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                   uint32_t active_pos) {
  float stored_scale = 1.0f;
  std::memcpy(&stored_scale,
              bytes.data + info->scale_offset +
                  static_cast<uint64_t>(active_pos) * info->scale_stride,
              sizeof(stored_scale));
  return stored_scale;
}

inline float snapshot_compact_score_scale(SnapshotBytes bytes,
                                          const SnapshotPreparedQuery* prepared,
                                          uint32_t active_pos) {
  float scale = 1.0f;
  std::memcpy(&scale,
              bytes.data + prepared->compact_score_scale_offset +
                  static_cast<uint64_t>(active_pos) * prepared->compact_score_scale_stride,
              sizeof(scale));
  return scale;
}

float snapshot_score_active(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                            const SnapshotPreparedQuery* prepared, uint32_t active_pos);

float snapshot_graph_score_active(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                  const SnapshotPreparedQuery* prepared, uint32_t active_pos) {
  if (info->storage_kind == ASTRAL_MEMORY_STORAGE_F6_E3M2_F32_RERANK) {
    const float stored_scale = snapshot_stored_scale(bytes, info, active_pos);
    const int16_t* vector =
        reinterpret_cast<const int16_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    const float scale = prepared->compact_score_scale_stride != 0
                            ? snapshot_compact_score_scale(bytes, prepared, active_pos)
                            : compact_value_scale_kind(info->storage_kind, stored_scale);
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      const float dot = prepared->compact_i16_vectors_aligned != 0
                            ? dot_i16_i16_aligned(vector, prepared->compact_query_i16, info->dim)
                            : dot_i16_i16(vector, prepared->compact_query_i16, info->dim);
      return dot * scale * prepared->compact_query_scale;
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      const float dot = prepared->compact_i16_vectors_aligned != 0
                            ? dot_i16_i16_aligned(vector, prepared->compact_query_i16, info->dim)
                            : dot_i16_i16(vector, prepared->compact_query_i16, info->dim);
      return dot * scale * prepared->compact_query_scale * prepared->query_scale;
    }
    return l2_score_i16_i16(vector, scale, prepared->compact_query_i16,
                            prepared->compact_query_scale, info->dim);
  }
  if (info->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK) {
    const float stored_scale = snapshot_stored_scale(bytes, info, active_pos);
    const int8_t* vector =
        reinterpret_cast<const int8_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    const float scale = prepared->compact_score_scale_stride != 0
                            ? snapshot_compact_score_scale(bytes, prepared, active_pos)
                            : compact_value_scale_kind(info->storage_kind, stored_scale);
    const float dot = prepared->compact_i8_vectors_aligned != 0
                          ? dot_q8_q8_query_aligned(vector, prepared->compact_query, info->dim,
                                                    prepared->compact_query_sum)
                          : dot_q8_q8_query(vector, prepared->compact_query, info->dim,
                                            prepared->compact_query_sum);
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return dot * scale * prepared->compact_query_scale;
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      const float vector_scale = prepared->compact_score_scale_stride == 0
                                     ? cosine_scale_q8(vector, stored_scale, info->dim)
                                     : 1.0f;
      return dot * scale * prepared->compact_query_scale * prepared->query_scale * vector_scale;
    }
    return l2_score_q8_q8(vector, scale, prepared->compact_query, prepared->compact_query_scale,
                          info->dim);
  }
  if (info->storage_kind == ASTRAL_MEMORY_STORAGE_F8_E5M2_F32_RERANK) {
    const float stored_scale = snapshot_stored_scale(bytes, info, active_pos);
    const int8_t* vector =
        reinterpret_cast<const int8_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    const float scale = prepared->compact_score_scale_stride != 0
                            ? snapshot_compact_score_scale(bytes, prepared, active_pos)
                            : compact_value_scale_kind(info->storage_kind, stored_scale);
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      const float dot =
          prepared->e5m2_clamp_non_finite != 0
              ? prepared->e5m2_kernels->dot_e5m2_clamped_a(vector, prepared->compact_query,
                                                           info->dim)
              : prepared->e5m2_kernels->dot_e5m2(vector, prepared->compact_query, info->dim);
      return dot * scale * prepared->compact_query_scale;
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      const float dot =
          prepared->e5m2_clamp_non_finite != 0
              ? prepared->e5m2_kernels->dot_e5m2_clamped_a(vector, prepared->compact_query,
                                                           info->dim)
              : prepared->e5m2_kernels->dot_e5m2(vector, prepared->compact_query, info->dim);
      return dot * scale * prepared->compact_query_scale * prepared->query_scale;
    }
    return prepared->e5m2_clamp_non_finite != 0
               ? prepared->e5m2_kernels->l2_e5m2_clamped_a(vector, scale, prepared->compact_query,
                                                           prepared->compact_query_scale, info->dim)
               : prepared->e5m2_kernels->l2_e5m2(vector, scale, prepared->compact_query,
                                                 prepared->compact_query_scale, info->dim);
  }
  return snapshot_score_active(bytes, info, prepared, active_pos);
}

float snapshot_score_active(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                            const SnapshotPreparedQuery* prepared, uint32_t active_pos) {
  if (f32_rerank_storage_kind(info->storage_kind)) {
    const float* vector =
        reinterpret_cast<const float*>(snapshot_score_vector_ptr(bytes, prepared, active_pos));
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return dot_f32(prepared->query, vector, info->dim);
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      return dot_f32(prepared->query, vector, info->dim) * prepared->query_scale;
    }
    return l2_score_f32(prepared->query, vector, info->dim);
  }
  if (prepared->compact != 0) {
    const float stored_scale = snapshot_stored_scale(bytes, info, active_pos);
    const float scale = compact_value_scale_kind(info->storage_kind, stored_scale);
    const int8_t* vector =
        reinterpret_cast<const int8_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    const int16_t* vector_i16 =
        reinterpret_cast<const int16_t*>(snapshot_vector_ptr(bytes, info, active_pos));
    if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
      return prepared->use_f6 != 0
                 ? (prepared->compact_i8_vectors_aligned != 0
                        ? dot_q8_q8_query_aligned(vector, prepared->compact_query, info->dim,
                                                  prepared->compact_query_sum)
                        : dot_q8_q8_query(vector, prepared->compact_query, info->dim,
                                          prepared->compact_query_sum)) *
                       scale * prepared->compact_query_scale
             : prepared->use_f6_i16 != 0
                 ? (prepared->compact_i16_vectors_aligned != 0
                        ? dot_i16_i16_aligned(vector_i16, prepared->compact_query_i16, info->dim)
                        : dot_i16_i16(vector_i16, prepared->compact_query_i16, info->dim)) *
                       scale * prepared->compact_query_scale
             : prepared->use_f8 != 0 ? (prepared->e5m2_clamp_non_finite != 0
                                            ? prepared->e5m2_kernels->dot_e5m2_clamped_a(
                                                  vector, prepared->compact_query, info->dim)
                                            : prepared->e5m2_kernels->dot_e5m2(
                                                  vector, prepared->compact_query, info->dim)) *
                                           scale * prepared->compact_query_scale
                                     : dot_q8_f32(vector, prepared->query, info->dim) * scale;
    }
    if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
      if (prepared->use_f6 != 0) {
        const float dot = prepared->compact_i8_vectors_aligned != 0
                              ? dot_q8_q8_query_aligned(vector, prepared->compact_query, info->dim,
                                                        prepared->compact_query_sum)
                              : dot_q8_q8_query(vector, prepared->compact_query, info->dim,
                                                prepared->compact_query_sum);
        return dot * scale * prepared->compact_query_scale * prepared->query_scale;
      }
      if (prepared->use_f6_i16 != 0) {
        const float dot =
            prepared->compact_i16_vectors_aligned != 0
                ? dot_i16_i16_aligned(vector_i16, prepared->compact_query_i16, info->dim)
                : dot_i16_i16(vector_i16, prepared->compact_query_i16, info->dim);
        return dot * scale * prepared->compact_query_scale * prepared->query_scale;
      }
      if (prepared->use_f8 != 0) {
        const float dot =
            prepared->e5m2_clamp_non_finite != 0
                ? prepared->e5m2_kernels->dot_e5m2_clamped_a(vector, prepared->compact_query,
                                                             info->dim)
                : prepared->e5m2_kernels->dot_e5m2(vector, prepared->compact_query, info->dim);
        return dot * scale * prepared->compact_query_scale * prepared->query_scale;
      }
      const float vector_scale = cosine_scale_q8(vector, stored_scale, info->dim);
      return dot_q8_f32(vector, prepared->query, info->dim) * scale * prepared->query_scale *
             vector_scale;
    }
    return prepared->use_f6 != 0 ? l2_score_q8_q8(vector, scale, prepared->compact_query,
                                                  prepared->compact_query_scale, info->dim)
           : prepared->use_f6_i16 != 0
               ? l2_score_i16_i16(vector_i16, scale, prepared->compact_query_i16,
                                  prepared->compact_query_scale, info->dim)
           : prepared->use_f8 != 0
               ? (prepared->e5m2_clamp_non_finite != 0
                      ? prepared->e5m2_kernels->l2_e5m2_clamped_a(
                            vector, scale, prepared->compact_query, prepared->compact_query_scale,
                            info->dim)
                      : prepared->e5m2_kernels->l2_e5m2(vector, scale, prepared->compact_query,
                                                        prepared->compact_query_scale, info->dim))
               : l2_score_q8_f32(vector, scale, prepared->query, info->dim);
  }

  const float* vector =
      reinterpret_cast<const float*>(snapshot_vector_ptr(bytes, info, active_pos));
  if (info->metric == ASTRAL_MEMORY_METRIC_DOT) {
    return dot_f32(prepared->query, vector, info->dim);
  }
  if (info->metric == ASTRAL_MEMORY_METRIC_COSINE) {
    return prepared->normalized_f32_cosine != 0
               ? dot_f32(prepared->query, vector, info->dim)
               : dot_f32(prepared->query, vector, info->dim) * prepared->query_scale *
                     cosine_scale(vector, info->dim);
  }
  return l2_score_f32(prepared->query, vector, info->dim);
}

bool snapshot_graph_better(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                           float candidate_score, uint32_t candidate_slot, float existing_score,
                           uint32_t existing_slot) {
  if (candidate_score != existing_score) {
    return candidate_score > existing_score;
  }
  AstralMemoryRecord candidate{};
  AstralMemoryRecord existing{};
  std::memcpy(&candidate, snapshot_record_ptr(bytes, info, candidate_slot), sizeof(candidate));
  std::memcpy(&existing, snapshot_record_ptr(bytes, info, existing_slot), sizeof(existing));
  return candidate.key < existing.key;
}

inline bool snapshot_graph_worse(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                 float candidate_score, uint32_t candidate_slot,
                                 float existing_score, uint32_t existing_slot) {
  return snapshot_graph_better(bytes, info, existing_score, existing_slot, candidate_score,
                               candidate_slot);
}

bool snapshot_graph_insert_top(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                               GraphSearchScratch* scratch, uint32_t capacity, uint32_t* filled,
                               uint32_t slot, float score) {
  uint32_t count = *filled;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!snapshot_graph_better(bytes, info, scratch->top_scores[parent],
                                 scratch->top_slots[parent], score, slot)) {
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

  if (!snapshot_graph_better(bytes, info, score, slot, scratch->top_scores[0],
                             scratch->top_slots[0])) {
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
        snapshot_graph_worse(bytes, info, scratch->top_scores[right], scratch->top_slots[right],
                             scratch->top_scores[left], scratch->top_slots[left])) {
      child = right;
    }
    if (!snapshot_graph_worse(bytes, info, scratch->top_scores[child], scratch->top_slots[child],
                              score, slot)) {
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

void snapshot_graph_add_candidate(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                  GraphSearchScratch* scratch, uint32_t capacity, uint32_t slot,
                                  float score, uint32_t* candidate_count) {
  uint32_t count = *candidate_count;
  if (count < capacity) {
    uint32_t pos = count;
    while (pos > 0) {
      const uint32_t parent = (pos - 1u) >> 1u;
      if (!snapshot_graph_better(bytes, info, score, slot, scratch->candidate_scores[parent],
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
    uint32_t worst_pos = 0;
    float worst_score = scratch->candidate_scores[0];
    for (uint32_t i = 1; i < count; ++i) {
      if (snapshot_graph_worse(bytes, info, scratch->candidate_scores[i], scratch->candidates[i],
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
  if (!snapshot_graph_better(bytes, info, score, slot, scratch->candidate_worst_score,
                             scratch->candidate_worst_slot)) {
    return;
  }

  uint32_t pos = scratch->candidate_worst_pos;
  bool moved_up = false;
  while (pos > 0) {
    const uint32_t parent = (pos - 1u) >> 1u;
    if (!snapshot_graph_better(bytes, info, score, slot, scratch->candidate_scores[parent],
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
          snapshot_graph_better(bytes, info, scratch->candidate_scores[right],
                                scratch->candidates[right], scratch->candidate_scores[left],
                                scratch->candidates[left])) {
        child = right;
      }
      if (!snapshot_graph_better(bytes, info, scratch->candidate_scores[child],
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

bool snapshot_graph_pop_candidate(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                  GraphSearchScratch* scratch, uint32_t* candidate_count,
                                  uint32_t* out_slot, float* out_score) {
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
      if (right < count &&
          snapshot_graph_better(bytes, info, scratch->candidate_scores[right],
                                scratch->candidates[right], scratch->candidate_scores[left],
                                scratch->candidates[left])) {
        child = right;
      }
      if (!snapshot_graph_better(bytes, info, scratch->candidate_scores[child],
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

void remove_snapshot_graph_worst_top(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                     GraphSearchScratch* scratch, uint32_t* top_count) {
  uint32_t count = *top_count;
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
          snapshot_graph_worse(bytes, info, scratch->top_scores[right], scratch->top_slots[right],
                               scratch->top_scores[left], scratch->top_slots[left])) {
        child = right;
      }
      if (!snapshot_graph_worse(bytes, info, scratch->top_scores[child], scratch->top_slots[child],
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
  *top_count = count;
}

void trim_snapshot_graph_top_candidates(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                        GraphSearchScratch* scratch, uint32_t* top_count,
                                        uint32_t target_count) {
  while (*top_count > target_count) {
    remove_snapshot_graph_worst_top(bytes, info, scratch, top_count);
  }
}

void snapshot_graph_begin_visit(const AstralMemorySnapshotInfo* info, GraphSearchScratch* scratch) {
  ++scratch->visit_generation;
  scratch->candidate_worst_valid = 0;
  if (scratch->visit_generation == 0) {
    std::memset(scratch->visited, 0, sizeof(uint16_t) * info->count);
    scratch->visit_generation = kGraphVisitGenerationStart;
  }
}

inline void snapshot_graph_mark_visited(GraphSearchScratch* scratch, uint32_t active_pos) {
  scratch->visited[active_pos] = scratch->visit_generation;
}

inline bool snapshot_graph_was_visited(const GraphSearchScratch* scratch, uint32_t active_pos) {
  return scratch->visited[active_pos] == scratch->visit_generation;
}

inline uint32_t snapshot_graph_search_capacity(const MemorySnapshotView* view,
                                               const AstralMemorySearchDesc* desc) {
  uint32_t requested =
      desc->graph_search != 0 ? desc->graph_search : view->graph.header.query_search_capacity;
  if (requested < kGraphMinSearch) {
    requested = kGraphMinSearch;
  }
  if (requested > view->graph.scratch_capacity) {
    requested = view->graph.scratch_capacity;
  }
  return requested;
}

inline uint32_t snapshot_graph_candidate_search_capacity(const SnapshotGraphLayout* graph,
                                                         uint32_t search_capacity) {
  uint32_t requested = search_capacity;
  if (search_capacity <= kU32Max / kGraphCandidateReserveMultiplier) {
    requested = search_capacity * kGraphCandidateReserveMultiplier;
  }
  return requested < graph->candidate_capacity ? requested : graph->candidate_capacity;
}

template <bool CompactCounts>
uint32_t snapshot_graph_greedy_closest(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                       const SnapshotGraphLayout* graph,
                                       const SnapshotPreparedQuery* prepared, uint32_t entry,
                                       uint32_t begin_level, uint32_t end_level) {
  uint32_t closest = entry;
  float closest_score = snapshot_graph_score_active(bytes, info, prepared, closest);
  for (uint32_t level = begin_level; level > end_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t neighbor_count =
          snapshot_graph_neighbor_count<CompactCounts>(bytes.data, info, graph, closest, level);
      const uint64_t neighbor_base = snapshot_graph_level_offset(info, graph, closest, level);
      for (uint32_t i = 0; i < neighbor_count; ++i) {
        const uint32_t candidate = snapshot_graph_neighbor_at_base(bytes.data, neighbor_base, i);
        if (candidate == kU32Max || candidate >= info->count ||
            snapshot_graph_level(bytes.data, graph, candidate) < level) {
          continue;
        }
        const float score = snapshot_graph_score_active(bytes, info, prepared, candidate);
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

template <bool CompactCounts>
void snapshot_graph_search_layer(SnapshotBytes bytes, const AstralMemorySnapshotInfo* info,
                                 const SnapshotGraphLayout* graph,
                                 const SnapshotPreparedQuery* prepared, GraphSearchScratch* scratch,
                                 uint32_t entry, uint32_t search_capacity,
                                 uint32_t* out_top_count) {
  uint32_t candidate_count = 0;
  uint32_t top_count = 0;
  const uint32_t candidate_capacity =
      snapshot_graph_candidate_search_capacity(graph, search_capacity);
  snapshot_graph_mark_visited(scratch, entry);
  const float entry_score = snapshot_graph_score_active(bytes, info, prepared, entry);
  snapshot_graph_insert_top(bytes, info, scratch, search_capacity, &top_count, entry, entry_score);
  snapshot_graph_add_candidate(bytes, info, scratch, candidate_capacity, entry, entry_score,
                               &candidate_count);

  while (candidate_count != 0) {
    uint32_t slot = kU32Max;
    float slot_score = kWorstScore;
    if (!snapshot_graph_pop_candidate(bytes, info, scratch, &candidate_count, &slot, &slot_score)) {
      break;
    }
    if (top_count == search_capacity &&
        !snapshot_graph_better(bytes, info, slot_score, slot, scratch->top_scores[0],
                               scratch->top_slots[0])) {
      break;
    }
    const uint32_t neighbor_count =
        snapshot_graph_neighbor_count<CompactCounts>(bytes.data, info, graph, slot, 0);
    const uint64_t neighbor_base = snapshot_graph_level_offset(info, graph, slot, 0);
    const uint32_t prefetch_limit = neighbor_count > kGraphNeighborPrefetchDistance
                                        ? neighbor_count - kGraphNeighborPrefetchDistance
                                        : 0u;
    uint32_t i = 0;
    for (; i < prefetch_limit; ++i) {
      const uint32_t neighbor = snapshot_graph_neighbor_at_base(bytes.data, neighbor_base, i);
#if defined(__GNUC__) || defined(__clang__)
      const uint32_t prefetch_neighbor = snapshot_graph_neighbor_at_base(
          bytes.data, neighbor_base, i + kGraphNeighborPrefetchDistance);
      __builtin_prefetch(info->storage_kind == ASTRAL_MEMORY_STORAGE_Q8_F32_RERANK
                             ? snapshot_vector_ptr(bytes, info, prefetch_neighbor)
                             : snapshot_score_vector_ptr(bytes, prepared, prefetch_neighbor),
                         0, 1);
#endif
      if (snapshot_graph_was_visited(scratch, neighbor)) {
        continue;
      }
      snapshot_graph_mark_visited(scratch, neighbor);
      const float score = snapshot_graph_score_active(bytes, info, prepared, neighbor);
      if (snapshot_graph_insert_top(bytes, info, scratch, search_capacity, &top_count, neighbor,
                                    score)) {
        snapshot_graph_add_candidate(bytes, info, scratch, candidate_capacity, neighbor, score,
                                     &candidate_count);
      }
    }
    for (; i < neighbor_count; ++i) {
      const uint32_t neighbor = snapshot_graph_neighbor_at_base(bytes.data, neighbor_base, i);
      if (snapshot_graph_was_visited(scratch, neighbor)) {
        continue;
      }
      snapshot_graph_mark_visited(scratch, neighbor);
      const float score = snapshot_graph_score_active(bytes, info, prepared, neighbor);
      if (snapshot_graph_insert_top(bytes, info, scratch, search_capacity, &top_count, neighbor,
                                    score)) {
        snapshot_graph_add_candidate(bytes, info, scratch, candidate_capacity, neighbor, score,
                                     &candidate_count);
      }
    }
  }
  *out_top_count = top_count;
}

AstralErr memory_snapshot_search_with_info(SnapshotBytes bytes,
                                           const AstralMemorySnapshotInfo* info,
                                           const AstralMemorySearchDesc* desc, const float* query,
                                           AstralMemorySearchResult* out_results,
                                           uint32_t max_results, uint32_t* out_count,
                                           uint8_t e5m2_clamp_non_finite);

AstralErr memory_snapshot_view_search_graph(MemorySnapshotView* view,
                                            const AstralMemorySearchDesc* desc, const float* query,
                                            AstralMemorySearchResult* out_results,
                                            uint32_t max_results, uint32_t* out_count) {
  if (desc == nullptr || desc->size != sizeof(AstralMemorySearchDesc) || query == nullptr ||
      out_count == nullptr || desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }
  if (view->graph_ready == 0 || desc->group_id != ASTRAL_MEMORY_GROUP_ANY) {
    return memory_snapshot_search_with_info(SnapshotBytes{view->map.data, view->map.size},
                                            &view->info, desc, query, out_results, max_results,
                                            out_count, view->e5m2_clamp_non_finite);
  }
  if (snapshot_compact_graph_exact_search_preferred(&view->info)) {
    return memory_snapshot_search_with_info(SnapshotBytes{view->map.data, view->map.size},
                                            &view->info, desc, query, out_results, max_results,
                                            out_count, view->e5m2_clamp_non_finite);
  }
  if (out_results == nullptr || max_results < desc->top_k) {
    return ASTRAL_E_NOMEM;
  }
  if (!f32_values_finite(query, view->info.dim)) {
    return ASTRAL_E_INVALID;
  }
  SnapshotBytes bytes{view->map.data, view->map.size};
  SnapshotPreparedQuery prepared{};
  snapshot_prepare_query(bytes, &view->info, query, view->e5m2_clamp_non_finite, &prepared);
  GraphSearchScratch* scratch = &view->graph_scratch;
  snapshot_graph_begin_visit(&view->info, scratch);
  const uint32_t search_capacity = snapshot_graph_search_capacity(view, desc);
  uint32_t top_count = 0;
  if (save_graph_count_bytes_impl(view->info.version) == sizeof(uint8_t)) {
    const uint32_t entry =
        view->graph.header.max_level != 0
            ? snapshot_graph_greedy_closest<true>(bytes, &view->info, &view->graph, &prepared,
                                                  view->graph.header.entry_active_pos,
                                                  view->graph.header.max_level, 0)
            : view->graph.header.entry_active_pos;
    snapshot_graph_search_layer<true>(bytes, &view->info, &view->graph, &prepared, scratch, entry,
                                      search_capacity, &top_count);
  } else {
    const uint32_t entry =
        view->graph.header.max_level != 0
            ? snapshot_graph_greedy_closest<false>(bytes, &view->info, &view->graph, &prepared,
                                                   view->graph.header.entry_active_pos,
                                                   view->graph.header.max_level, 0)
            : view->graph.header.entry_active_pos;
    snapshot_graph_search_layer<false>(bytes, &view->info, &view->graph, &prepared, scratch, entry,
                                       search_capacity, &top_count);
  }
  if (f32_rerank_storage_kind(view->info.storage_kind)) {
    const uint32_t rerank_capacity = snapshot_graph_f32_rerank_capacity(desc->top_k, top_count);
    trim_snapshot_graph_top_candidates(bytes, &view->info, scratch, &top_count, rerank_capacity);
  }

  uint32_t filled = 0;
  for (uint32_t i = 0; i < top_count; ++i) {
    AstralMemoryRecord record{};
    std::memcpy(&record, snapshot_record_ptr(bytes, &view->info, scratch->top_slots[i]),
                sizeof(record));
    if (record.size != sizeof(AstralMemoryRecord) || record.key == 0) {
      return ASTRAL_E_INVALID;
    }
    MemorySlot slot{};
    slot.record = record;
    AstralMemorySearchResult candidate{};
    const float score =
        f32_rerank_storage_kind(view->info.storage_kind)
            ? snapshot_score_active(bytes, &view->info, &prepared, scratch->top_slots[i])
            : scratch->top_scores[i];
    fill_result(&candidate, slot, score);
    insert_result(out_results, desc->top_k, &filled, candidate);
  }
  *out_count = filled;
  return ASTRAL_OK;
}

AstralErr memory_snapshot_search_with_info(SnapshotBytes bytes,
                                           const AstralMemorySnapshotInfo* info,
                                           const AstralMemorySearchDesc* desc, const float* query,
                                           AstralMemorySearchResult* out_results,
                                           uint32_t max_results, uint32_t* out_count,
                                           uint8_t e5m2_clamp_non_finite) {
  if (bytes.data == nullptr || info == nullptr || desc == nullptr ||
      desc->size != sizeof(AstralMemorySearchDesc) || query == nullptr || out_count == nullptr ||
      desc->top_k == 0) {
    return ASTRAL_E_INVALID;
  }
  if (out_results == nullptr || max_results < desc->top_k) {
    return ASTRAL_E_NOMEM;
  }
  if (!f32_values_finite(query, info->dim)) {
    return ASTRAL_E_INVALID;
  }

  SnapshotPreparedQuery prepared{};
  snapshot_prepare_query(bytes, info, query, e5m2_clamp_non_finite, &prepared);
  uint32_t filled = 0;
  for (uint32_t i = 0; i < info->count; ++i) {
#if defined(__GNUC__) || defined(__clang__)
    if (prepared.compact != 0 && i + kFlatQ8PrefetchDistance < info->count) {
      const uint64_t prefetch_i = static_cast<uint64_t>(i + kFlatQ8PrefetchDistance);
      __builtin_prefetch(bytes.data + info->scale_offset + prefetch_i * info->scale_stride, 0, 1);
      __builtin_prefetch(
          snapshot_score_vector_ptr(bytes, &prepared, static_cast<uint32_t>(prefetch_i)), 0, 1);
      __builtin_prefetch(bytes.data + info->record_offset + prefetch_i * info->record_stride, 0, 1);
    }
#endif
    AstralMemoryRecord record{};
    std::memcpy(&record, snapshot_record_ptr(bytes, info, i), sizeof(record));
    if (record.size != sizeof(AstralMemoryRecord) || record.key == 0) {
      return ASTRAL_E_INVALID;
    }
    if (desc->group_id != ASTRAL_MEMORY_GROUP_ANY && record.group_id != desc->group_id) {
      continue;
    }

    MemorySlot slot{};
    slot.record = record;
    AstralMemorySearchResult candidate{};
    fill_result(&candidate, slot, snapshot_score_active(bytes, info, &prepared, i));
    insert_result(out_results, desc->top_k, &filled, candidate);
  }

  *out_count = filled;
  return ASTRAL_OK;
}

AstralErr memory_snapshot_search_bytes(SnapshotBytes bytes, const AstralMemorySearchDesc* desc,
                                       const float* query, AstralMemorySearchResult* out_results,
                                       uint32_t max_results, uint32_t* out_count) {
  AstralMemorySnapshotInfo info{};
  info.size = sizeof(AstralMemorySnapshotInfo);
  AstralErr err = memory_snapshot_info_bytes(bytes, &info);
  if (err != ASTRAL_OK) {
    return err;
  }

  return memory_snapshot_search_with_info(bytes, &info, desc, query, out_results, max_results,
                                          out_count, 1u);
}

} // namespace

AstralErr memory_snapshot_info(AstralSpanU8 bytes, AstralMemorySnapshotInfo* out_info) {
  return memory_snapshot_info_bytes(SnapshotBytes{bytes.data, bytes.len}, out_info);
}

AstralErr memory_snapshot_search(AstralSpanU8 bytes, const AstralMemorySearchDesc* desc,
                                 const float* query, AstralMemorySearchResult* out_results,
                                 uint32_t max_results, uint32_t* out_count) {
  return memory_snapshot_search_bytes(SnapshotBytes{bytes.data, bytes.len}, desc, query,
                                      out_results, max_results, out_count);
}

AstralErr memory_snapshot_map(AstralSpanU8 path, MemorySnapshotView** out_view,
                              AstralMemorySnapshotInfo* out_info) {
  if (path.data == nullptr || path.len == 0 || out_view == nullptr || out_info == nullptr ||
      out_info->size != sizeof(AstralMemorySnapshotInfo)) {
    return ASTRAL_E_INVALID;
  }
  *out_view = nullptr;

  std::string path_string(reinterpret_cast<const char*>(path.data), path.len);
  auto* view = core::runtime_new<MemorySnapshotView>();
  if (view == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  *view = {};
  if (!platform::file_map_readonly(path_string.c_str(), &view->map)) {
    core::runtime_delete(view);
    return ASTRAL_E_NOT_FOUND;
  }

  AstralMemorySnapshotInfo info{};
  info.size = sizeof(AstralMemorySnapshotInfo);
  AstralErr err = memory_snapshot_info_bytes(SnapshotBytes{view->map.data, view->map.size}, &info);
  if (err != ASTRAL_OK) {
    platform::file_unmap_readonly(&view->map);
    core::runtime_delete(view);
    return err;
  }
  SnapshotBytes view_bytes{view->map.data, view->map.size};
  view->graph_ready = 0;
  view->e5m2_clamp_non_finite = snapshot_e5m2_requires_clamp(view_bytes, &info) ? 1u : 0u;
  if (snapshot_graph_layout(view_bytes, &info, &view->graph) &&
      snapshot_graph_scratch_alloc(&info, &view->graph, &view->graph_scratch)) {
    view->graph_ready = 1;
  }

  const AstralHandle handle = core::register_handle(core::HandleKind::MemorySnapshot, view);
  if (handle == 0) {
    if (view->graph_ready != 0) {
      snapshot_graph_scratch_free(&info, &view->graph, &view->graph_scratch);
    }
    platform::file_unmap_readonly(&view->map);
    core::runtime_delete(view);
    return ASTRAL_E_NOMEM;
  }
  view->handle = handle;
  view->info = info;
  *out_info = info;
  *out_view = view;
  return ASTRAL_OK;
}

void memory_snapshot_unmap(MemorySnapshotView* view) {
  if (view == nullptr) {
    return;
  }
  core::unregister_handle(view->handle, core::HandleKind::MemorySnapshot);
  if (view->graph_ready != 0) {
    snapshot_graph_scratch_free(&view->info, &view->graph, &view->graph_scratch);
  }
  platform::file_unmap_readonly(&view->map);
  core::runtime_delete(view);
}

AstralErr memory_snapshot_view_info(MemorySnapshotView* view, AstralMemorySnapshotInfo* out_info) {
  if (view == nullptr || out_info == nullptr ||
      out_info->size != sizeof(AstralMemorySnapshotInfo)) {
    return ASTRAL_E_INVALID;
  }
  *out_info = view->info;
  return ASTRAL_OK;
}

AstralErr memory_snapshot_view_search(MemorySnapshotView* view, const AstralMemorySearchDesc* desc,
                                      const float* query, AstralMemorySearchResult* out_results,
                                      uint32_t max_results, uint32_t* out_count) {
  if (view == nullptr) {
    return ASTRAL_E_INVALID;
  }
  if (view->info.index_kind == ASTRAL_MEMORY_INDEX_GRAPH) {
    return memory_snapshot_view_search_graph(view, desc, query, out_results, max_results,
                                             out_count);
  }
  return memory_snapshot_search_with_info(SnapshotBytes{view->map.data, view->map.size},
                                          &view->info, desc, query, out_results, max_results,
                                          out_count, view->e5m2_clamp_non_finite);
}

} // namespace astral::inference
