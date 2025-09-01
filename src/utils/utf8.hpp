/**
 * utf8.hpp - UTF-8 validation and utilities
 *
 * Core UTF-8 operations with SIMD-accelerated validation.
 * All strings are immutable spans (pointer + length); NO NUL termination assumed.
 *
 * Validation rules:
 * - Reject invalid sequences (out-of-range bytes, incomplete sequences)
 * - Reject overlong encodings (security: canonical representation only)
 * - Reject surrogate pairs (U+D800-U+DFFF are invalid UTF-8)
 * - Reject code points > U+10FFFF (Unicode maximum)
 *
 * Performance:
 * - SIMD path: 1-2 GB/s on modern hardware (AVX2/NEON)
 * - Scalar fallback: 200-400 MB/s
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace astral::utf8 {

// ============================================================================
// Span: Immutable UTF-8 string view
// ============================================================================

/**
 * Immutable UTF-8 byte span.
 * Matches AstralSpanU8 layout from C ABI.
 *
 * NO NUL terminator assumed; use len for all operations.
 */
struct Span {
    const uint8_t* data;
    uint32_t len;

    constexpr Span() : data(nullptr), len(0) {}
    constexpr Span(const uint8_t* d, uint32_t l) : data(d), len(l) {}

    /**
     * Compare two spans for exact byte equality.
     * @return true if both spans have identical content
     */
    bool equals(Span other) const {
        if (len != other.len) return false;
        if (data == other.data) return true;
        return ::memcmp(data, other.data, len) == 0;
    }

    /**
     * Extract a slice [start, end).
     * @param start Starting byte offset (inclusive)
     * @param end Ending byte offset (exclusive)
     * @return New span covering [start, end)
     *
     * Preconditions: start <= end <= len
     */
    Span slice(uint32_t start, uint32_t end) const {
        // Clamp to valid range
        if (start > len) start = len;
        if (end > len) end = len;
        if (start > end) start = end;
        return Span(data + start, end - start);
    }

    /**
     * Create span from NUL-terminated C string.
     * Scans for NUL terminator; use sparingly (prefer explicit length).
     *
     * @param str NUL-terminated string
     * @return Span covering [str, NUL)
     */
    static Span from_cstr(const char* str) {
        if (!str) return Span();
        const char* end = str;
        while (*end) ++end;
        return Span(reinterpret_cast<const uint8_t*>(str),
                    static_cast<uint32_t>(end - str));
    }

    /**
     * Check if span is empty.
     */
    bool empty() const { return len == 0; }

    /**
     * Access byte at index (no bounds checking in release).
     */
    uint8_t operator[](uint32_t i) const { return data[i]; }
};

// ============================================================================
// UTF-8 Validation (Scalar)
// ============================================================================

/**
 * Validate UTF-8 encoding (scalar implementation).
 *
 * Checks:
 * - All bytes in valid ranges
 * - Complete multi-byte sequences
 * - No overlong encodings
 * - No surrogate pairs (U+D800-U+DFFF)
 * - No code points > U+10FFFF
 *
 * @param input UTF-8 byte sequence
 * @return true if valid UTF-8; false otherwise
 *
 * Thread-safety: Safe to call from multiple threads.
 * Performance: ~200-400 MB/s (scalar loop)
 */
bool validate(Span input);

// ============================================================================
// UTF-8 Code Point Counting
// ============================================================================

/**
 * Count Unicode code points (not bytes).
 *
 * Counts the number of UTF-8 code points by identifying continuation bytes.
 * Continuation bytes have pattern 10xxxxxx; all others are start bytes.
 *
 * Precondition: input must be valid UTF-8 (call validate() first)
 *
 * @param input Valid UTF-8 byte sequence
 * @return Number of code points
 *
 * Performance: ~1-2 GB/s (SIMD-accelerated on supported platforms)
 */
uint32_t count_codepoints(Span input);

// ============================================================================
// SIMD-Accelerated Validation
// ============================================================================

/**
 * Validate UTF-8 encoding with SIMD acceleration.
 *
 * Uses AVX2 (x86-64) or NEON (ARM) when available; falls back to scalar.
 * Runtime detection of CPU features; no recompilation needed.
 *
 * @param input UTF-8 byte sequence
 * @return true if valid UTF-8; false otherwise
 *
 * Thread-safety: Safe to call from multiple threads.
 * Performance:
 * - AVX2:  1-2 GB/s
 * - NEON:  800 MB/s - 1.5 GB/s
 * - Scalar: 200-400 MB/s (automatic fallback)
 */
bool validate_simd(Span input);

// ============================================================================
// Internal Helpers (exposed for testing)
// ============================================================================

namespace detail {

/**
 * Detect if current CPU supports AVX2.
 * Cached after first call (no overhead).
 */
bool has_avx2();

/**
 * Detect if current CPU supports NEON.
 * Cached after first call (no overhead).
 */
bool has_neon();

} // namespace detail

} // namespace astral::utf8
