#pragma once

#include "../../include/astral_rt.h"

namespace astral::inference {

struct MemoryIndex;
struct MemorySearchCursor;
struct MemorySnapshotView;

AstralHandle memory_handle(MemoryIndex* index);
AstralHandle memory_search_cursor_handle(MemorySearchCursor* cursor);
AstralHandle memory_snapshot_view_handle(MemorySnapshotView* view);
AstralErr memory_create(const AstralMemoryIndexDesc* desc, MemoryIndex** out_index);
void memory_destroy(MemoryIndex* index);
AstralErr memory_count(MemoryIndex* index, uint32_t* out_count);
AstralErr memory_stats(MemoryIndex* index, AstralMemoryStats* out_stats);
AstralErr memory_clear(MemoryIndex* index);
AstralErr memory_get_record(MemoryIndex* index, uint64_t key, AstralMemoryRecord* out_record);
AstralErr memory_update_record(MemoryIndex* index, uint64_t key, const AstralMemoryRecord* record);
AstralErr memory_add_batch(MemoryIndex* index, const AstralMemoryRecord* records,
                           const float* vectors, uint32_t count);
AstralErr memory_remove(MemoryIndex* index, uint64_t key);
AstralErr memory_search(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                        AstralMemorySearchResult* out_results, uint32_t max_results,
                        uint32_t* out_count);
AstralErr memory_search_batch(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* queries, uint32_t query_count,
                              AstralMemorySearchResult* out_results, uint32_t max_results,
                              uint32_t* out_counts);
AstralErr memory_search_begin(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* query, MemorySearchCursor** out_cursor);
AstralErr memory_search_fetch(MemorySearchCursor* cursor, AstralMemorySearchResult* out_results,
                              uint32_t max_results, uint32_t* out_count);
AstralErr memory_search_cancel(MemorySearchCursor* cursor);
AstralErr memory_search_cursor_status(MemorySearchCursor* cursor, AstralRequestState* out_state,
                                      uint32_t* out_remaining);
AstralErr memory_search_cursor_remaining(MemorySearchCursor* cursor, uint32_t* out_remaining);
void memory_search_end(MemorySearchCursor* cursor);
AstralErr memory_save_size(MemoryIndex* index, uint64_t* out_bytes);
AstralErr memory_save(MemoryIndex* index, AstralMutSpanU8 out_bytes, uint64_t* out_written);
AstralErr memory_snapshot_info(AstralSpanU8 bytes, AstralMemorySnapshotInfo* out_info);
AstralErr memory_snapshot_search(AstralSpanU8 bytes, const AstralMemorySearchDesc* desc,
                                 const float* query, AstralMemorySearchResult* out_results,
                                 uint32_t max_results, uint32_t* out_count);
AstralErr memory_snapshot_map(AstralSpanU8 path, MemorySnapshotView** out_view,
                              AstralMemorySnapshotInfo* out_info);
void memory_snapshot_unmap(MemorySnapshotView* view);
AstralErr memory_snapshot_view_info(MemorySnapshotView* view, AstralMemorySnapshotInfo* out_info);
AstralErr memory_snapshot_view_search(MemorySnapshotView* view, const AstralMemorySearchDesc* desc,
                                      const float* query, AstralMemorySearchResult* out_results,
                                      uint32_t max_results, uint32_t* out_count);
AstralErr memory_load(const AstralMemoryIndexDesc* desc, AstralSpanU8 bytes,
                      MemoryIndex** out_index);

} // namespace astral::inference
