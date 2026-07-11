/**
 * test_utils.cpp - Quick compilation and functionality test
 *
 * This is a minimal test to verify the utility implementations compile
 * and have basic functionality working.
 */

#include "utf8.hpp"
#include "string_builder.hpp"
#include "logging.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace astral;

struct TestStringAllocator {
  static void* allocate(size_t size, size_t) noexcept { return malloc(size); }
  static void deallocate(void* ptr, size_t, size_t) noexcept { free(ptr); }
};

using TestStringBuilder = utf8::StringBuilder<256, TestStringAllocator>;

// ============================================================================
// Test Helpers
// ============================================================================

static int g_test_count = 0;
static int g_test_passed = 0;

#define TEST(name) \
    static void test_##name(); \
    static void run_test_##name() { \
        g_test_count++; \
        printf("Running test: %s\n", #name); \
        test_##name(); \
        g_test_passed++; \
        printf("  PASSED\n"); \
    } \
    static void test_##name()

#define ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "ASSERTION FAILED: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            exit(1); \
        } \
    } while (0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_TRUE(x) ASSERT((x))
#define ASSERT_FALSE(x) ASSERT(!(x))

// ============================================================================
// UTF-8 Tests
// ============================================================================

TEST(utf8_span_equals) {
    utf8::Span s1(reinterpret_cast<const uint8_t*>("hello"), 5);
    utf8::Span s2(reinterpret_cast<const uint8_t*>("hello"), 5);
    utf8::Span s3(reinterpret_cast<const uint8_t*>("world"), 5);
    utf8::Span s4(reinterpret_cast<const uint8_t*>("helloworld"), 5); // Same prefix

    ASSERT_TRUE(s1.equals(s2));
    ASSERT_FALSE(s1.equals(s3));
    ASSERT_TRUE(s1.equals(s4)); // Same length and data
}

TEST(utf8_span_slice) {
    utf8::Span s = utf8::Span::from_cstr("hello world");
    utf8::Span hello = s.slice(0, 5);
    utf8::Span world = s.slice(6, 11);

    ASSERT_EQ(hello.len, 5);
    ASSERT_EQ(world.len, 5);
    ASSERT_TRUE(hello.equals(utf8::Span::from_cstr("hello")));
    ASSERT_TRUE(world.equals(utf8::Span::from_cstr("world")));
}

TEST(utf8_validate_ascii) {
    utf8::Span ascii = utf8::Span::from_cstr("Hello, World!");
    ASSERT_TRUE(utf8::validate(ascii));
}

TEST(utf8_validate_multibyte) {
    // "Hello 世界" in UTF-8
    const uint8_t data[] = {
        'H', 'e', 'l', 'l', 'o', ' ',
        0xE4, 0xB8, 0x96,  // 世 (U+4E16)
        0xE7, 0x95, 0x8C   // 界 (U+754C)
    };
    utf8::Span s(data, sizeof(data));
    ASSERT_TRUE(utf8::validate(s));
}

TEST(utf8_validate_invalid_continuation) {
    // Missing continuation byte
    const uint8_t data[] = {'H', 'i', 0xE4, 'X'};
    utf8::Span s(data, sizeof(data));
    ASSERT_FALSE(utf8::validate(s));
}

TEST(utf8_validate_overlong) {
    // Overlong encoding of 'A' (U+0041)
    const uint8_t data[] = {0xC1, 0x81};
    utf8::Span s(data, sizeof(data));
    ASSERT_FALSE(utf8::validate(s));
}

TEST(utf8_validate_surrogate) {
    // Surrogate pair U+D800 (invalid in UTF-8)
    const uint8_t data[] = {0xED, 0xA0, 0x80};
    utf8::Span s(data, sizeof(data));
    ASSERT_FALSE(utf8::validate(s));
}

TEST(utf8_count_codepoints_ascii) {
    utf8::Span s = utf8::Span::from_cstr("Hello");
    ASSERT_EQ(utf8::count_codepoints(s), 5);
}

TEST(utf8_count_codepoints_multibyte) {
    // "Hi 世" (4 code points: H, i, space, 世)
    const uint8_t data[] = {'H', 'i', ' ', 0xE4, 0xB8, 0x96};
    utf8::Span s(data, sizeof(data));
    ASSERT_EQ(utf8::count_codepoints(s), 4);
}

// ============================================================================
// StringBuilder Tests
// ============================================================================

TEST(string_builder_basic) {
  TestStringBuilder sb;

  sb.append(utf8::Span::from_cstr("Hello"));
  utf8::Span result = sb.freeze();

  ASSERT_EQ(result.len, 5);
  ASSERT_TRUE(result.equals(utf8::Span::from_cstr("Hello")));
}

TEST(string_builder_append_multiple) {
  TestStringBuilder sb;

  sb.append(utf8::Span::from_cstr("Hello"));
  sb.append(utf8::Span::from_cstr(" "));
  sb.append(utf8::Span::from_cstr("World"));

  utf8::Span result = sb.freeze();
  ASSERT_EQ(result.len, 11);
  ASSERT_TRUE(result.equals(utf8::Span::from_cstr("Hello World")));
}

TEST(string_builder_append_u32) {
  TestStringBuilder sb;

  sb.append(utf8::Span::from_cstr("Token: "));
  sb.append_u32(42);

  utf8::Span result = sb.freeze();
  ASSERT_TRUE(result.equals(utf8::Span::from_cstr("Token: 42")));
}

TEST(string_builder_append_f32) {
  TestStringBuilder sb;

  sb.append(utf8::Span::from_cstr("Score: "));
  sb.append_f32(0.95f, 2);

  utf8::Span result = sb.freeze();
  ASSERT_TRUE(result.equals(utf8::Span::from_cstr("Score: 0.95")));
}

TEST(string_builder_reset) {
  TestStringBuilder sb;

  sb.append(utf8::Span::from_cstr("First"));
  ASSERT_EQ(sb.length(), 5);

  sb.reset();
  ASSERT_EQ(sb.length(), 0);
  ASSERT_TRUE(sb.empty());

  sb.append(utf8::Span::from_cstr("Second"));
  utf8::Span result = sb.freeze();
  ASSERT_TRUE(result.equals(utf8::Span::from_cstr("Second")));
}

// ============================================================================
// Logging Tests
// ============================================================================

static int g_log_call_count = 0;
static int g_last_log_level = -1;
static char g_last_log_msg[256];

void test_log_callback(void* /* user */, int level, const uint8_t* msg, uint32_t len) {
    g_log_call_count++;
    g_last_log_level = level;
    uint32_t copy_len = len < sizeof(g_last_log_msg) - 1 ? len : sizeof(g_last_log_msg) - 1;
    memcpy(g_last_log_msg, msg, copy_len);
    g_last_log_msg[copy_len] = '\0';
}

TEST(logging_basic) {
    g_log_call_count = 0;
    logging::set_callback(test_log_callback, nullptr);

    logging::log(logging::Level::Info, "Test message");

    ASSERT_EQ(g_log_call_count, 1);
    ASSERT_EQ(g_last_log_level, static_cast<int>(logging::Level::Info));
    ASSERT_EQ(strcmp(g_last_log_msg, "Test message"), 0);
}

TEST(logging_formatted) {
    g_log_call_count = 0;
    logging::set_callback(test_log_callback, nullptr);

    logging::log(logging::Level::Warn, "Value: %d", 42);

    ASSERT_EQ(g_log_call_count, 1);
    ASSERT_EQ(strcmp(g_last_log_msg, "Value: 42"), 0);
}

TEST(logging_level_filter) {
    g_log_call_count = 0;
    logging::set_callback(test_log_callback, nullptr);
    logging::set_min_level(logging::Level::Warn);

    // Should be logged (Warn >= Warn)
    logging::log(logging::Level::Warn, "Warning");
    ASSERT_EQ(g_log_call_count, 1);

    // Should be filtered (Info < Warn)
    logging::log(logging::Level::Info, "Info");
    ASSERT_EQ(g_log_call_count, 1); // No change

    // Reset to default
    logging::set_min_level(logging::Level::Info);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== Astral Utils Test Suite ===\n\n");

    // UTF-8 tests
    run_test_utf8_span_equals();
    run_test_utf8_span_slice();
    run_test_utf8_validate_ascii();
    run_test_utf8_validate_multibyte();
    run_test_utf8_validate_invalid_continuation();
    run_test_utf8_validate_overlong();
    run_test_utf8_validate_surrogate();
    run_test_utf8_count_codepoints_ascii();
    run_test_utf8_count_codepoints_multibyte();

    // StringBuilder tests
    run_test_string_builder_basic();
    run_test_string_builder_append_multiple();
    run_test_string_builder_append_u32();
    run_test_string_builder_append_f32();
    run_test_string_builder_reset();

    // Logging tests
    run_test_logging_basic();
    run_test_logging_formatted();
    run_test_logging_level_filter();

    printf("\n=== Results ===\n");
    printf("Passed: %d/%d\n", g_test_passed, g_test_count);

    return (g_test_passed == g_test_count) ? 0 : 1;
}
