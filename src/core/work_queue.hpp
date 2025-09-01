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

} // namespace astral::core
