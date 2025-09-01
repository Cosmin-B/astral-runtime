/**
 * string_builder.hpp - Append-only UTF-8 string builder
 *
 * StringBuilder for constructing UTF-8 strings without allocations.
 * Uses FrameAllocator for backing storage; all memory reclaimed on allocator reset.
 *
 * Design:
 * - Append-only (no remove/insert operations)
 * - Zero-copy freeze() to snapshot current content
 * - All allocations from provided FrameAllocator
 * - Grows exponentially when space exhausted
 *
 * Thread Safety: NOT thread-safe (same as FrameAllocator).
 */

#pragma once

#include "utf8.hpp"
#include "../memory/frame_allocator.hpp"
#include <cstdint>
#include <cstdio>

namespace astral::utf8 {

/**
 * Append-only string builder over FrameAllocator.
 *
 * Usage:
 * ```cpp
 * FrameAllocator alloc(memory, capacity);
 * StringBuilder sb(alloc);
 *
 * sb.append(Span::from_cstr("Token: "));
 * sb.append_u32(42);
 * sb.append(Span::from_cstr(" (score: "));
 * sb.append_f32(0.95f, 2);
 * sb.append(Span::from_cstr(")"));
 *
 * Span result = sb.freeze(); // "Token: 42 (score: 0.95)"
 * ```
 *
 * NOT thread-safe: Use one per thread.
 */
class StringBuilder {
public:
    /**
     * Create string builder backed by frame allocator.
     *
     * @param alloc Frame allocator for backing storage (must outlive this object)
     * @param initial_capacity Initial buffer size (default 256 bytes)
     */
    explicit StringBuilder(memory::FrameAllocator& alloc, uint32_t initial_capacity = 256)
        : alloc_(alloc), buffer_(nullptr), capacity_(0), len_(0) {
        reserve(initial_capacity);
    }

    /**
     * Append UTF-8 span.
     *
     * @param utf8 UTF-8 data to append
     *
     * Thread-safety: NOT thread-safe.
     * Performance: ~10-20ns per append (memcpy + length update).
     */
    void append(Span utf8) {
        if (utf8.len == 0) return;
        ensure_space(utf8.len);
        ::memcpy(buffer_ + len_, utf8.data, utf8.len);
        len_ += utf8.len;
    }

    /**
     * Append unsigned 32-bit integer as UTF-8 decimal.
     *
     * @param val Integer value to append
     *
     * Example: append_u32(42) → "42"
     */
    void append_u32(uint32_t val) {
        // Max digits: 10 (4294967295)
        char tmp[16];
        int n = ::snprintf(tmp, sizeof(tmp), "%u", val);
        if (n > 0) {
            append(Span(reinterpret_cast<const uint8_t*>(tmp), static_cast<uint32_t>(n)));
        }
    }

    /**
     * Append signed 32-bit integer as UTF-8 decimal.
     *
     * @param val Integer value to append
     */
    void append_i32(int32_t val) {
        char tmp[16];
        int n = ::snprintf(tmp, sizeof(tmp), "%d", val);
        if (n > 0) {
            append(Span(reinterpret_cast<const uint8_t*>(tmp), static_cast<uint32_t>(n)));
        }
    }

    /**
     * Append 32-bit float as UTF-8 decimal.
     *
     * @param val Float value to append
     * @param decimals Number of decimal places (default 2)
     *
     * Example: append_f32(3.14159f, 2) → "3.14"
     */
    void append_f32(float val, uint32_t decimals = 2) {
        char tmp[32];
        int n = ::snprintf(tmp, sizeof(tmp), "%.*f", static_cast<int>(decimals), val);
        if (n > 0) {
            append(Span(reinterpret_cast<const uint8_t*>(tmp), static_cast<uint32_t>(n)));
        }
    }

    /**
     * Append single ASCII character.
     *
     * @param c ASCII character (0-127)
     *
     * Precondition: c must be ASCII (0-127).
     */
    void append_char(char c) {
        ensure_space(1);
        buffer_[len_++] = static_cast<uint8_t>(c);
    }

    /**
     * Snapshot current content as immutable span (zero-copy).
     *
     * @return Span covering current buffer content
     *
     * Returned span is valid until next append() or reset().
     * No memory allocation; just returns pointer + length.
     *
     * Thread-safety: NOT thread-safe.
     * Performance: ~1-2ns (return struct).
     */
    Span freeze() const {
        return Span(buffer_, len_);
    }

    /**
     * Reset builder to empty state.
     * Invalidates all previous freeze() results.
     *
     * Thread-safety: NOT thread-safe.
     * Performance: ~1-2ns (set length to 0).
     */
    void reset() {
        len_ = 0;
    }

    /**
     * Get current length in bytes.
     */
    uint32_t length() const { return len_; }

    /**
     * Get current capacity in bytes.
     */
    uint32_t capacity() const { return capacity_; }

    /**
     * Check if builder is empty.
     */
    bool empty() const { return len_ == 0; }

private:
    /**
     * Ensure buffer has at least `additional` bytes free.
     * Grows buffer if needed.
     */
    void ensure_space(uint32_t additional) {
        if (len_ + additional <= capacity_) [[likely]] {
            return;
        }
        grow(len_ + additional);
    }

    /**
     * Reserve initial capacity.
     */
    void reserve(uint32_t new_capacity) {
        if (new_capacity <= capacity_) return;
        uint8_t* new_buffer = static_cast<uint8_t*>(alloc_.alloc(new_capacity, alignof(uint8_t)));
        if (!new_buffer) return; // Out of memory

        if (buffer_ && len_ > 0) {
            ::memcpy(new_buffer, buffer_, len_);
        }
        buffer_ = new_buffer;
        capacity_ = new_capacity;
    }

    /**
     * Grow buffer to accommodate at least `required` bytes.
     * Doubles capacity each time.
     */
    void grow(uint32_t required) {
        // Double until we can fit required
        uint32_t new_capacity = capacity_ > 0 ? capacity_ : 256;
        while (new_capacity < required) {
            new_capacity *= 2;
        }
        reserve(new_capacity);
    }

    memory::FrameAllocator& alloc_;
    uint8_t* buffer_;
    uint32_t capacity_;
    uint32_t len_;
};

} // namespace astral::utf8
