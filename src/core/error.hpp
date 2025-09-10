#pragma once

#include "../../include/astral_rt.h"

namespace astral::core {

// Error scope used to make `astral_last_error()` callback-safe.
//
// Design:
// - `astral_last_error()` returns a pointer to a per-thread buffer.
// - Each C ABI entry point pushes an error scope so nested Astral calls (e.g. from callbacks)
//   use a different buffer slot and do not clobber the outer error string.
// - After the outermost call returns, the last error remains available via slot 0 until the
//   next Astral call on the same thread.
class ErrorScope {
public:
    ErrorScope() noexcept;
    ~ErrorScope() noexcept;

    ErrorScope(const ErrorScope&) = delete;
    ErrorScope& operator=(const ErrorScope&) = delete;
    ErrorScope(ErrorScope&&) = delete;
    ErrorScope& operator=(ErrorScope&&) = delete;
};

// Thread-local last error context (UTF-8, NUL-terminated).
// - Set only on error paths (avoid overhead on fast paths).
// - Returned pointer remains valid until the next Astral call on the same thread.
void set_last_error(const char* msg);
void set_last_errorf(const char* fmt, ...);
void set_last_error_from_code(AstralErr err);
void clear_last_error();
const char* last_error();

} // namespace astral::core
