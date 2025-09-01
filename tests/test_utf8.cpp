/**
 * test_utf8.cpp - UTF-8 utilities tests
 *
 * Tests for UTF-8 span operations, validation, and code point counting.
 * Validates: equals, slice, from_cstr, validation, SIMD equivalence.
 */

#include "test_framework.hpp"
#include "../src/utils/utf8.hpp"

#include <cstring>

using namespace astral::utf8;

//
// Span Operations Tests
//

TEST(span_equals_identical) {
    const char* str = "Hello, World!";
    Span s1 = Span::from_cstr(str);
    Span s2 = Span::from_cstr(str);

    ASSERT_TRUE(s1.equals(s2));
}

TEST(span_equals_different_content) {
    Span s1 = Span::from_cstr("Hello");
    Span s2 = Span::from_cstr("World");

    ASSERT_FALSE(s1.equals(s2));
}

TEST(span_equals_different_length) {
    Span s1 = Span::from_cstr("Hello");
    Span s2 = Span::from_cstr("Hello, World!");

    ASSERT_FALSE(s1.equals(s2));
}

TEST(span_equals_empty) {
    Span s1;
    Span s2;

    ASSERT_TRUE(s1.equals(s2));
}

TEST(span_slice_basic) {
    Span s = Span::from_cstr("Hello, World!");
    Span slice = s.slice(7, 12); // "World"

    Span expected = Span::from_cstr("World");
    ASSERT_TRUE(slice.equals(expected));
}

TEST(span_slice_full) {
    Span s = Span::from_cstr("Test");
    Span slice = s.slice(0, s.len);

    ASSERT_TRUE(s.equals(slice));
}

TEST(span_slice_empty) {
    Span s = Span::from_cstr("Test");
    Span slice = s.slice(2, 2);

    ASSERT_EQ(slice.len, 0);
}

TEST(span_slice_clamp) {
    Span s = Span::from_cstr("Test");

    // Out of bounds should clamp
    Span slice = s.slice(0, 1000);
    ASSERT_EQ(slice.len, s.len);

    // Invalid range should return empty
    Span slice2 = s.slice(10, 5);
    ASSERT_EQ(slice2.len, 0);
}

TEST(span_from_cstr_basic) {
    const char* str = "Hello";
    Span s = Span::from_cstr(str);

    ASSERT_EQ(s.len, 5);
    ASSERT_EQ(memcmp(s.data, str, 5), 0);
}

TEST(span_from_cstr_empty) {
    const char* str = "";
    Span s = Span::from_cstr(str);

    ASSERT_EQ(s.len, 0);
}

TEST(span_from_cstr_nullptr) {
    Span s = Span::from_cstr(nullptr);

    ASSERT_EQ(s.len, 0);
    ASSERT_EQ(s.data, nullptr);
}

//
// UTF-8 Validation Tests
//

TEST(validate_utf8_ascii) {
    const char* str = "Hello, World!";
    Span s = Span::from_cstr(str);

    bool valid = validate(s);
    ASSERT_TRUE(valid);
}

TEST(validate_utf8_multibyte) {
    // UTF-8 encoded string: "Hello 世界" (Chinese)
    const uint8_t data[] = {
        'H', 'e', 'l', 'l', 'o', ' ',
        0xE4, 0xB8, 0x96, // 世 (U+4E16)
        0xE7, 0x95, 0x8C  // 界 (U+754C)
    };
    Span s(data, sizeof(data));

    bool valid = validate(s);
    ASSERT_TRUE(valid);
}

TEST(validate_utf8_invalid_sequence) {
    // Invalid UTF-8: continuation byte without lead byte
    const uint8_t data[] = {0x80, 0x80};
    Span s(data, sizeof(data));

    bool valid = validate(s);
    ASSERT_FALSE(valid);
}

TEST(validate_utf8_overlong_encoding) {
    // Overlong encoding of '/' (U+002F)
    // Valid: 0x2F
    // Overlong (2-byte): 0xC0 0xAF (INVALID - security risk)
    const uint8_t data[] = {0xC0, 0xAF};
    Span s(data, sizeof(data));

    bool valid = validate(s);
    ASSERT_FALSE(valid);
}

TEST(validate_utf8_surrogate_pair) {
    // UTF-16 surrogate pair encoded in UTF-8 (INVALID)
    // U+D800 (high surrogate): 0xED 0xA0 0x80
    const uint8_t data[] = {0xED, 0xA0, 0x80};
    Span s(data, sizeof(data));

    bool valid = validate(s);
    ASSERT_FALSE(valid);
}

TEST(validate_utf8_max_codepoint) {
    // U+10FFFF (maximum valid Unicode code point)
    // UTF-8: 0xF4 0x8F 0xBF 0xBF
    const uint8_t data[] = {0xF4, 0x8F, 0xBF, 0xBF};
    Span s(data, sizeof(data));

    bool valid = validate(s);
    ASSERT_TRUE(valid);
}

TEST(validate_utf8_above_max_codepoint) {
    // U+110000 (above maximum, INVALID)
    // UTF-8: 0xF4 0x90 0x80 0x80
    const uint8_t data[] = {0xF4, 0x90, 0x80, 0x80};
    Span s(data, sizeof(data));

    bool valid = validate(s);
    ASSERT_FALSE(valid);
}

TEST(validate_utf8_incomplete_sequence) {
    // Incomplete 3-byte sequence
    const uint8_t data[] = {0xE4, 0xB8}; // Missing final byte
    Span s(data, sizeof(data));

    bool valid = validate(s);
    ASSERT_FALSE(valid);
}

TEST(validate_utf8_empty) {
    Span s;

    bool valid = validate(s);
    ASSERT_TRUE(valid); // Empty string is valid UTF-8
}

//
// Code Point Counting Tests
//

TEST(count_codepoints_ascii) {
    Span s = Span::from_cstr("Hello");

    size_t count = count_codepoints(s);
    ASSERT_EQ(count, 5);
}

TEST(count_codepoints_multibyte) {
    // "Hello 世界" = 8 code points (6 ASCII + 2 Chinese)
    const uint8_t data[] = {
        'H', 'e', 'l', 'l', 'o', ' ',
        0xE4, 0xB8, 0x96, // 世
        0xE7, 0x95, 0x8C  // 界
    };
    Span s(data, sizeof(data));

    size_t count = count_codepoints(s);
    ASSERT_EQ(count, 8);
}

TEST(count_codepoints_emoji) {
    // "😀" (U+1F600) = 1 code point, 4 bytes
    const uint8_t data[] = {0xF0, 0x9F, 0x98, 0x80};
    Span s(data, sizeof(data));

    size_t count = count_codepoints(s);
    ASSERT_EQ(count, 1);
}

TEST(count_codepoints_empty) {
    Span s;

    size_t count = count_codepoints(s);
    ASSERT_EQ(count, 0);
}

//
// SIMD Validation Tests
//

TEST(validate_utf8_simd_matches_scalar) {
    // Test that SIMD and scalar implementations agree
    const char* test_strings[] = {
        "Hello, World!",
        "UTF-8: ©®™",
        "Chinese: 你好世界",
        "Japanese: こんにちは",
        "Emoji: 😀🎉🚀",
        "Mixed: Hello世界😀",
        "",
    };

    for (const char* str : test_strings) {
        Span s = Span::from_cstr(str);

        // Both should give same result
        bool simd_result = validate_simd(s);
        bool scalar_result = validate(s);

        ASSERT_EQ(simd_result, scalar_result);
    }
}

TEST(validate_utf8_large_input) {
    // Test with large input (>256 bytes to trigger SIMD path)
    constexpr size_t kSize = 1024;
    uint8_t data[kSize];

    // Fill with valid ASCII
    for (size_t i = 0; i < kSize; ++i) {
        data[i] = 'A' + (i % 26);
    }

    Span s(data, kSize);
    bool valid = validate(s);
    ASSERT_TRUE(valid);
}
