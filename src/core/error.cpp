/**
 * error.cpp - Error handling and validation
 *
 * Provides error string conversion and handle validation.
 *
 * Design:
 * - Static error strings (no allocation)
 * - Fast error code to string conversion
 * - Handle validation via magic number check
 *
 * Thread Safety: All functions are thread-safe (read-only data).
 *
 * Influenced by POSIX strerror() design.
 */

#include "../../include/astral_rt.h"
#include "error.hpp"
#include "handles.hpp"

#include <cstdarg>
#include <cstdio>

// ============================================================================
// Error String Conversion
// ============================================================================

extern "C" {

ASTRAL_API void ASTRAL_CALL astral_version(uint32_t* major, uint32_t* minor, uint32_t* patch) {
    if (major) {
        *major = ASTRAL_VERSION_MAJOR;
    }
    if (minor) {
        *minor = ASTRAL_VERSION_MINOR;
    }
    if (patch) {
        *patch = ASTRAL_VERSION_PATCH;
    }
}

ASTRAL_API const char* ASTRAL_CALL astral_error_string(AstralErr err) {
    // Return static error string for error code
    // No allocation, no thread-local state
    switch (err) {
        case ASTRAL_OK:
            return "Success";
        case ASTRAL_E_INVALID:
            return "Invalid parameter";
        case ASTRAL_E_NOMEM:
            return "Out of memory";
        case ASTRAL_E_BUSY:
            return "Resource busy";
        case ASTRAL_E_TIMEOUT:
            return "Operation timed out";
        case ASTRAL_E_STATE:
            return "Invalid state";
        case ASTRAL_E_BACKEND:
            return "Backend error";
        case ASTRAL_E_CANCELED:
            return "Canceled";
        case ASTRAL_E_UNSUPPORTED:
            return "Unsupported";
        default:
            return "Unknown error";
    }
}

} // extern "C"

namespace astral::core {

namespace {

thread_local char g_last_error[512] = {0};

} // namespace

void set_last_error(const char* msg) {
    if (msg == nullptr) {
        g_last_error[0] = '\0';
        return;
    }
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
}

void set_last_errorf(const char* fmt, ...) {
    if (fmt == nullptr) {
        g_last_error[0] = '\0';
        return;
    }

    va_list args;
    va_start(args, fmt);
    std::vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
}

void set_last_error_from_code(AstralErr err) {
    set_last_error(astral_error_string(err));
}

void clear_last_error() {
    g_last_error[0] = '\0';
}

const char* last_error() {
    return g_last_error;
}

} // namespace astral::core

extern "C" {

ASTRAL_API const char* ASTRAL_CALL astral_last_error(void) {
    return astral::core::last_error();
}

ASTRAL_API void ASTRAL_CALL astral_clear_last_error(void) {
    astral::core::clear_last_error();
}

ASTRAL_API int ASTRAL_CALL astral_handle_valid(AstralHandle handle) {
    return astral::core::handle_valid(handle) ? 1 : 0;
}

} // extern "C"
