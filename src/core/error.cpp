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
#include "../utils/string_builder.hpp"
#include "abi_guard.hpp"
#include "error.hpp"
#include "handles.hpp"

#include <cstdint>
#include <cstring>

// ============================================================================
// Error String Conversion
// ============================================================================

extern "C" {

ASTRAL_API void ASTRAL_CALL astral_version(uint32_t* major, uint32_t* minor, uint32_t* patch) {
    ASTRAL_ABI_TRY_BEGIN
    if (major) {
        *major = ASTRAL_VERSION_MAJOR;
    }
    if (minor) {
        *minor = ASTRAL_VERSION_MINOR;
    }
    if (patch) {
        *patch = ASTRAL_VERSION_PATCH;
    }
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API const char* ASTRAL_CALL astral_error_string(AstralErr err) {
    ASTRAL_ABI_TRY_BEGIN
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
        case ASTRAL_E_NOT_FOUND:
            return "Not found";
        default:
            return "Unknown error";
    }
    ASTRAL_ABI_CATCH_END_CSTR()
}

} // extern "C"

namespace astral::core {

namespace {

constexpr uint32_t kErrorSlots = 8;
constexpr size_t kErrorMsgBytes = 512;

struct ErrorTLS {
    uint32_t depth; // 0 outside Astral; 1+ inside a C ABI call; nested calls bump this
    char slots[kErrorSlots][kErrorMsgBytes];
};

thread_local ErrorTLS g_error_tls = {0, {{0}}};

static inline uint32_t current_slot_index() noexcept {
    // Slot 0 holds the last error for the most recent top-level call.
    // Slot 1.. are used for nested calls (callbacks calling back into Astral).
    const uint32_t d = g_error_tls.depth;
    return d == 0 ? 0u : (d - 1u);
}

static inline char* current_slot() noexcept {
    return g_error_tls.slots[current_slot_index()];
}

} // namespace

ErrorScope::ErrorScope() noexcept {
    // Bounded nesting: if we overflow, we saturate at the last slot.
    if (g_error_tls.depth < kErrorSlots) {
        ++g_error_tls.depth;
    }
    current_slot()[0] = '\0';
}

ErrorScope::~ErrorScope() noexcept {
    if (g_error_tls.depth > 0) {
        --g_error_tls.depth;
    }
}

void set_last_error(const char* msg) {
    if (msg == nullptr) {
        current_slot()[0] = '\0';
        return;
    }

    const utf8::Span text = utf8::Span::from_cstr(msg);
    const uint32_t copy_bytes =
        text.len < kErrorMsgBytes ? text.len : static_cast<uint32_t>(kErrorMsgBytes - 1u);
    std::memcpy(current_slot(), text.data, copy_bytes);
    current_slot()[copy_bytes] = '\0';
}

void set_last_error_parts(const char* prefix, const char* detail) {
  utf8::StackStringBuilder<kErrorMsgBytes - 1u> builder;
  if (prefix != nullptr) {
    builder.append(utf8::Span::from_cstr(prefix));
  }
  if (detail != nullptr) {
    builder.append(utf8::Span::from_cstr(detail));
  }
  set_last_error(builder.c_str());
}

void set_last_error_from_code(AstralErr err) {
    set_last_error(astral_error_string(err));
}

void clear_last_error() {
    current_slot()[0] = '\0';
}

const char* last_error() {
    return current_slot();
}

} // namespace astral::core

extern "C" {

ASTRAL_API const char* ASTRAL_CALL astral_last_error(void) {
    ASTRAL_ABI_TRY_BEGIN
    return astral::core::last_error();
    ASTRAL_ABI_CATCH_END_CSTR()
}

ASTRAL_API void ASTRAL_CALL astral_clear_last_error(void) {
    ASTRAL_ABI_TRY_BEGIN
    astral::core::clear_last_error();
    ASTRAL_ABI_CATCH_END_VOID()
}

ASTRAL_API int ASTRAL_CALL astral_handle_valid(AstralHandle handle) {
    ASTRAL_ABI_TRY_BEGIN
    return astral::core::handle_valid(handle) ? 1 : 0;
    ASTRAL_ABI_CATCH_END_I32(0)
}

} // extern "C"
