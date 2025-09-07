#pragma once

#include <cstdint>

namespace astral::core {

// Read-only runtime state accessors (not hot paths).
bool runtime_initialized();
uint32_t runtime_thread_count();
void runtime_memory_stats(uint64_t* out_committed, uint64_t* out_reserved);
bool runtime_hugepages_enabled();

} // namespace astral::core
