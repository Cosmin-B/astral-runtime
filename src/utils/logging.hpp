/**
 * logging.hpp - Non-blocking logging system
 *
 * Thread-safe logging with callback dispatch.
 * Uses thread-local buffers to avoid allocations.
 *
 * Design:
 * - Thread-local buffer per thread (4KB)
 * - Format message into TLS buffer
 * - Dispatch to callback (if set)
 * - Drop logs if the callback exceeds the 10ms logging budget
 * - Never block on logging
 *
 * Thread Safety: Safe to call from multiple threads.
 */

#pragma once

#include <cstdint>
#include <cstdarg>

namespace astral::logging {

/**
 * Log levels (matches AstralLog* constants from C ABI).
 */
enum class Level : uint8_t {
    Error = 0,  // ASTRAL_LOG_ERROR
    Warn = 1,   // ASTRAL_LOG_WARN
    Info = 2,   // ASTRAL_LOG_INFO
    Debug = 3,  // ASTRAL_LOG_DEBUG
    Trace = 4   // ASTRAL_LOG_TRACE
};

/**
 * Log callback signature (matches AstralLogFn from C ABI).
 *
 * @param user User data (from set_callback)
 * @param level Log level (0-4)
 * @param msg UTF-8 log message (not NUL-terminated)
 * @param len Message length in bytes
 *
 *  Callback MUST be non-blocking.
 * If callback takes >10ms, logs will be dropped.
 */
using LogCallback = void (*)(void* user, int level, const uint8_t* msg, uint32_t len);

/**
 * Set global log callback.
 *
 * @param cb Callback function (or nullptr to disable logging)
 * @param user User data passed to callback
 *
 * Thread-safety: Safe to call, but not atomic with log() calls.
 * Set callback once during initialization, not during hot path.
 */
void set_callback(LogCallback cb, void* user);

/**
 * Set minimum log level.
 * Logs below this level are ignored.
 *
 * @param level Minimum level to log
 *
 * Thread-safety: Safe to call, but not atomic with log() calls.
 */
void set_min_level(Level level);

/**
 * Log formatted message.
 *
 * @param level Log level
 * @param fmt printf-style format string (must be NUL-terminated)
 * @param ... Format arguments
 *
 * Thread-safety: Safe to call from multiple threads.
 * Performance: ~100-500ns per call (format + callback dispatch).
 *
 * Behavior:
 * - Formats into thread-local buffer (4KB)
 * - Truncates if message exceeds buffer
 * - Calls callback if set
 * - Silently drops logs if callback is null
 */
void log(Level level, const char* fmt, ...);

/**
 * Log formatted message (va_list variant).
 *
 * @param level Log level
 * @param fmt Format string
 * @param args Format arguments
 */
void logv(Level level, const char* fmt, va_list args);

// ============================================================================
// Convenience Wrappers
// ============================================================================

/**
 * Log error message.
 */
inline void error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logv(Level::Error, fmt, args);
    va_end(args);
}

/**
 * Log warning message.
 */
inline void warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logv(Level::Warn, fmt, args);
    va_end(args);
}

/**
 * Log info message.
 */
inline void info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logv(Level::Info, fmt, args);
    va_end(args);
}

/**
 * Log debug message.
 */
inline void debug(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logv(Level::Debug, fmt, args);
    va_end(args);
}

/**
 * Log trace message.
 */
inline void trace(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logv(Level::Trace, fmt, args);
    va_end(args);
}

} // namespace astral::logging
