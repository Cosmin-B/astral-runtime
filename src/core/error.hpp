#pragma once

#include "../../include/astral_rt.h"

namespace astral::core {

// Thread-local last error context (UTF-8, NUL-terminated).
// - Set only on error paths (avoid overhead on fast paths).
// - Returned pointer remains valid until the next Astral call on the same thread.
void set_last_error(const char* msg);
void set_last_errorf(const char* fmt, ...);
void set_last_error_from_code(AstralErr err);
void clear_last_error();
const char* last_error();

} // namespace astral::core
