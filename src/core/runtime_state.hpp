#pragma once

#include <cstddef>
#include <cstdint>

namespace astral::core {

// Read-only runtime state accessors (not hot paths).
bool runtime_initialized();
uint32_t runtime_thread_count();
uint32_t runtime_assign_worker_id();
void runtime_memory_stats(uint64_t* out_committed, uint64_t* out_reserved);
bool runtime_hugepages_enabled();
bool runtime_uses_arena();

// Runtime-owned allocation helpers (not hot paths).
//
// Intended for allocations during init/model/session creation, plugin loading, etc.
// In arena mode, these can be served from the runtime arena heap for determinism.
void* runtime_alloc(size_t size, size_t align);
void runtime_free(void* ptr, size_t size, size_t align);

// Worker scratch (per worker thread). Intended for transient helpers on worker threads.
// Not for decode hot paths unless explicitly designed to be bump-only and resettable.
bool runtime_on_worker_thread();
uint32_t runtime_worker_id();
void* runtime_worker_scratch_alloc(size_t size, size_t align);
void runtime_worker_scratch_reset();

// Allocate/release per-session scratch backing memory.
// These are not hot paths; they run during session create/destroy only.
void* runtime_session_scratch_acquire(size_t min_size, size_t alignment, size_t* out_size);
void runtime_session_scratch_release(void* ptr, size_t size);

} // namespace astral::core
