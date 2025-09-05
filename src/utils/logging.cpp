/**
 * logging.cpp - Non-blocking logging implementation
 *
 * Thread-local buffers prevent allocation on every log call.
 * Callback dispatch is non-blocking; slow callbacks cause log drops.
 */

#include "logging.hpp"
#include "../platform/time.h"
#include <atomic>
#include <cstdio>
#include <cstring>

namespace astral::logging {

// ============================================================================
// Global State
// ============================================================================

namespace {

// Global callback (set once at init)
std::atomic<LogCallback> g_callback{nullptr};
std::atomic<void*> g_user{nullptr};

// Minimum log level
std::atomic<Level> g_min_level{Level::Info};

// Thread-local buffer for formatting
constexpr size_t kBufferSize = 4096;
thread_local char g_buffer[kBufferSize];

// Maximum callback execution time (10ms per MASTER_SPEC)
constexpr uint64_t kMaxCallbackNanos = 10'000'000; // 10ms

/**
 * Get monotonic timestamp in nanoseconds.
 * Used to detect slow callbacks.
 */
inline uint64_t monotonic_now_ns() {
    return astral::platform::monotonic_time_ns();
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

void set_callback(LogCallback cb, void* user) {
    g_callback.store(cb, std::memory_order_release);
    g_user.store(user, std::memory_order_release);
}

void set_min_level(Level level) {
    g_min_level.store(level, std::memory_order_release);
}

void logv(Level level, const char* fmt, va_list args) {
    // Early exit: Level filtering
    if (level > g_min_level.load(std::memory_order_relaxed)) {
        return;
    }

    // Early exit: No callback
    LogCallback cb = g_callback.load(std::memory_order_acquire);
    if (!cb) {
        return;
    }

    // Format into thread-local buffer
    int n = ::vsnprintf(g_buffer, kBufferSize, fmt, args);

    // Handle formatting errors or truncation
    if (n < 0) {
        // Formatting error; skip this log
        return;
    }

    // Truncate if message exceeds buffer
    uint32_t msg_len = static_cast<uint32_t>(n);
    if (msg_len >= kBufferSize) {
        msg_len = kBufferSize - 1;
        // Add truncation marker
        const char* marker = "...";
        ::memcpy(g_buffer + kBufferSize - 4, marker, 3);
        g_buffer[kBufferSize - 1] = '\0';
    }

    // Dispatch to callback with timeout detection
    void* user = g_user.load(std::memory_order_acquire);
    uint64_t start = monotonic_now_ns();

    cb(user, static_cast<int>(level), reinterpret_cast<const uint8_t*>(g_buffer), msg_len);

    uint64_t elapsed = monotonic_now_ns() - start;

    // Warn if callback is slow (once per thread to avoid spam)
    if (elapsed > kMaxCallbackNanos) [[unlikely]] {
        static thread_local bool warned = false;
        if (!warned) {
            // Can't log via same system; use stderr
            ::fprintf(stderr,
                      "[ASTRAL WARNING] Log callback took %lu ms (>10ms limit). "
                      "Callback MUST be non-blocking.\n",
                      static_cast<unsigned long>(elapsed / 1'000'000));
            warned = true;
        }
    }
}

void log(Level level, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logv(level, fmt, args);
    va_end(args);
}

} // namespace astral::logging
