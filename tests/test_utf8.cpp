/**
 * test_utf8.cpp - UTF-8 utilities tests
 *
 * Tests for UTF-8 span operations, validation, and code point counting.
 * Validates: equals, slice, from_cstr, validation, SIMD equivalence.
 */

#include "../src/utils/string_builder.hpp"
#include "../src/utils/utf8.hpp"
#include "test_framework.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>

using namespace astral::utf8;

namespace {

struct TrackingStringAllocator {
  static thread_local uint32_t allocations;
  static thread_local uint32_t frees;
  static thread_local bool fail;

  static void* allocate(size_t size, size_t) noexcept {
    ++allocations;
    return fail ? nullptr : std::malloc(size);
  }

  static void deallocate(void* ptr, size_t, size_t) noexcept {
    ++frees;
    std::free(ptr);
  }

  static void reset() noexcept {
    allocations = 0;
    frees = 0;
    fail = false;
  }
};

thread_local uint32_t TrackingStringAllocator::allocations = 0;
thread_local uint32_t TrackingStringAllocator::frees = 0;
thread_local bool TrackingStringAllocator::fail = false;

} // namespace

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

TEST(count_codepoints_simd_width_ascii) {
    const uint8_t data[16] = {
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    };
    Span s(data, sizeof(data));

    size_t count = count_codepoints(s);
    ASSERT_EQ(count, 16);
}

//
// Inline String Builder Tests
//

TEST(string_builder_stays_inline_within_capacity) {
  TrackingStringAllocator::reset();
  StringBuilder<16, TrackingStringAllocator> builder;

  ASSERT_TRUE(builder.append_literal("hello"));
  ASSERT_TRUE(builder.append_char(' '));
  ASSERT_TRUE(builder.append_u32(42));

  ASSERT_FALSE(builder.spilled());
  ASSERT_FALSE(builder.truncated());
  ASSERT_EQ(TrackingStringAllocator::allocations, 0u);
  ASSERT_EQ(builder.length(), 8u);
  ASSERT_EQ(std::strcmp(builder.c_str(), "hello 42"), 0);
  ASSERT_EQ(builder.c_str()[builder.length()], '\0');
}

TEST(string_builder_spills_through_policy_allocator) {
  TrackingStringAllocator::reset();
  {
    StringBuilder<4, TrackingStringAllocator> builder;

    ASSERT_TRUE(builder.append(Span::from_cstr("abcdefgh")));
    ASSERT_TRUE(builder.spilled());
    ASSERT_FALSE(builder.truncated());
    ASSERT_EQ(TrackingStringAllocator::allocations, 1u);
    ASSERT_EQ(std::strcmp(builder.c_str(), "abcdefgh"), 0);

    ASSERT_TRUE(builder.append_literal("ijklmnopq"));
    ASSERT_EQ(TrackingStringAllocator::allocations, 2u);
    ASSERT_EQ(TrackingStringAllocator::frees, 1u);
    ASSERT_EQ(std::strcmp(builder.c_str(), "abcdefghijklmnopq"), 0);

    const uint32_t capacity = builder.capacity();
    builder.reset();
    ASSERT_TRUE(builder.empty());
    ASSERT_TRUE(builder.spilled());
    ASSERT_EQ(builder.capacity(), capacity);
    ASSERT_TRUE(builder.append(Span::from_cstr("reuse")));
    ASSERT_EQ(TrackingStringAllocator::allocations, 2u);
  }
  ASSERT_EQ(TrackingStringAllocator::frees, 2u);
}

TEST(string_builder_default_spill_uses_runtime_allocator) {
  StringBuilder<4> builder;

  ASSERT_TRUE(builder.append_literal("runtime"));
  ASSERT_TRUE(builder.spilled());
  ASSERT_FALSE(builder.truncated());
  ASSERT_EQ(std::strcmp(builder.c_str(), "runtime"), 0);
}

TEST(string_builder_truncates_without_spilling_when_requested) {
  TrackingStringAllocator::reset();
  StackStringBuilder<5, TrackingStringAllocator> builder;

  ASSERT_FALSE(builder.append(Span::from_cstr("abcdefgh")));
  ASSERT_TRUE(builder.truncated());
  ASSERT_FALSE(builder.spilled());
  ASSERT_EQ(TrackingStringAllocator::allocations, 0u);
  ASSERT_EQ(builder.length(), 5u);
  ASSERT_EQ(std::strcmp(builder.c_str(), "abcde"), 0);
  ASSERT_EQ(builder.c_str()[builder.length()], '\0');
}

TEST(string_builder_truncates_when_spill_allocation_fails) {
  TrackingStringAllocator::reset();
  TrackingStringAllocator::fail = true;
  StringBuilder<5, TrackingStringAllocator> builder;

  ASSERT_FALSE(builder.append(Span::from_cstr("abcdefgh")));
  ASSERT_TRUE(builder.truncated());
  ASSERT_FALSE(builder.spilled());
  ASSERT_EQ(TrackingStringAllocator::allocations, 1u);
  ASSERT_EQ(std::strcmp(builder.c_str(), "abcde"), 0);
}

TEST(string_builder_appends_integer_boundaries_without_printf) {
  StringBuilder<96, TrackingStringAllocator> builder;

  ASSERT_TRUE(builder.append_i32((std::numeric_limits<int32_t>::min)()));
  ASSERT_TRUE(builder.append_char(' '));
  ASSERT_TRUE(builder.append_u32((std::numeric_limits<uint32_t>::max)()));
  ASSERT_TRUE(builder.append_char(' '));
  ASSERT_TRUE(builder.append_i64((std::numeric_limits<int64_t>::min)()));
  ASSERT_TRUE(builder.append_char(' '));
  ASSERT_TRUE(builder.append_u64((std::numeric_limits<uint64_t>::max)()));

  ASSERT_EQ(std::strcmp(builder.c_str(),
                        "-2147483648 4294967295 -9223372036854775808 18446744073709551615"),
            0);
}

TEST(string_builder_appends_fixed_float_without_printf) {
  StringBuilder<32, TrackingStringAllocator> builder;

  ASSERT_TRUE(builder.append_f32(3.14159f, 3));
  ASSERT_TRUE(builder.append_char(' '));
  ASSERT_TRUE(builder.append_f32(-0.5f, 2));

  ASSERT_EQ(std::strcmp(builder.c_str(), "3.142 -0.50"), 0);
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
