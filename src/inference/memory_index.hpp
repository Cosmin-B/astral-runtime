#pragma once

#include "../../include/astral_rt.h"

namespace astral::inference {

struct MemoryIndex;
struct MemorySearchCursor;

AstralHandle memory_handle(MemoryIndex* index);
AstralHandle memory_search_cursor_handle(MemorySearchCursor* cursor);
AstralErr memory_create(const AstralMemoryIndexDesc* desc, MemoryIndex** out_index);
void memory_destroy(MemoryIndex* index);
AstralErr memory_count(MemoryIndex* index, uint32_t* out_count);
AstralErr memory_clear(MemoryIndex* index);
AstralErr memory_add_batch(MemoryIndex* index, const AstralMemoryRecord* records,
                           const float* vectors, uint32_t count);
AstralErr memory_remove(MemoryIndex* index, uint64_t key);
AstralErr memory_search(MemoryIndex* index, const AstralMemorySearchDesc* desc, const float* query,
                        AstralMemorySearchResult* out_results, uint32_t max_results,
                        uint32_t* out_count);
AstralErr memory_search_begin(MemoryIndex* index, const AstralMemorySearchDesc* desc,
                              const float* query, MemorySearchCursor** out_cursor);
AstralErr memory_search_fetch(MemorySearchCursor* cursor, AstralMemorySearchResult* out_results,
                              uint32_t max_results, uint32_t* out_count);
void memory_search_end(MemorySearchCursor* cursor);
AstralErr memory_save_size(MemoryIndex* index, uint64_t* out_bytes);
AstralErr memory_save(MemoryIndex* index, AstralMutSpanU8 out_bytes, uint64_t* out_written);
AstralErr memory_load(const AstralMemoryIndexDesc* desc, AstralSpanU8 bytes,
                      MemoryIndex** out_index);

} // namespace astral::inference
