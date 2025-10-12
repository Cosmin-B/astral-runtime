#pragma once

#include "../../include/astral_rt.h"

namespace astral::core {

using WorkFn = void (*)(void* user);

// Submit work to Astral's internal worker pool.
// Returns:
// - ASTRAL_OK on success
// - ASTRAL_E_STATE if runtime not initialized
// - ASTRAL_E_INVALID if fn is null
// Note: Current implementation is blocking on saturation (no try/timeout path yet).
AstralErr submit_work(WorkFn fn, void* user);

// Submit work to a specific worker thread (affinity).
//
// This is used to enforce "one session ↔ one worker thread" so per-session scratch/allocators
// are not touched concurrently by multiple workers.
//
// `worker_id` is normalized with modulo thread_count before enqueue.
AstralErr submit_work_affine(uint32_t worker_id, WorkFn fn, void* user);

} // namespace astral::core
