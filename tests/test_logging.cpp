/**
 * test_logging.cpp - Logging tests
 *
 * Tests for logging subsystem.
 * Validates: basic logging, level filtering, callback invocation, thread-local buffers.
 */

#include "test_framework.hpp"
#include "../src/utils/logging.hpp"

#include <atomic>
#include <cstring>

using namespace astral::logging;

// Test callback data
struct LogCallbackData {
    std::atomic<int> call_count{0};
    int last_level{-1};
    char last_message[256];
};

void test_log_callback(void* user, int level, const uint8_t* msg, uint32_t len) {
    auto* data = static_cast<LogCallbackData*>(user);
    data->call_count.fetch_add(1, std::memory_order_relaxed);
    data->last_level = level;

    size_t copy_len = len < sizeof(data->last_message) - 1 ? len : sizeof(data->last_message) - 1;
    memcpy(data->last_message, msg, copy_len);
    data->last_message[copy_len] = '\0';
}

//
// Basic Logging Tests
//

TEST(logging_basic) {
    LogCallbackData data;
    set_callback(test_log_callback, &data);
    set_min_level(Level::Info);

    info("Test message");

    ASSERT_EQ(data.call_count.load(), 1);
    ASSERT_EQ(data.last_level, static_cast<int>(Level::Info));
}

TEST(logging_all_levels) {
    LogCallbackData data;
    set_callback(test_log_callback, &data);
    set_min_level(Level::Trace); // Enable all levels

    error("Error");
    ASSERT_EQ(data.last_level, static_cast<int>(Level::Error));

    warn("Warning");
    ASSERT_EQ(data.last_level, static_cast<int>(Level::Warn));

    info("Info");
    ASSERT_EQ(data.last_level, static_cast<int>(Level::Info));

    debug("Debug");
    ASSERT_EQ(data.last_level, static_cast<int>(Level::Debug));

    trace("Trace");
    ASSERT_EQ(data.last_level, static_cast<int>(Level::Trace));

    ASSERT_EQ(data.call_count.load(), 5);
}

TEST(logging_level_filtering) {
    LogCallbackData data;
    set_callback(test_log_callback, &data);
    set_min_level(Level::Warn); // Only WARN and ERROR

    error("Error");
    warn("Warning");
    info("Info");      // Should be filtered
    debug("Debug");    // Should be filtered
    trace("Trace");    // Should be filtered

    // Only ERROR and WARN should have been logged
    ASSERT_EQ(data.call_count.load(), 2);
}

TEST(logging_format_string) {
    LogCallbackData data;
    set_callback(test_log_callback, &data);
    set_min_level(Level::Info);

    info("Value: %d, String: %s", 42, "test");

    ASSERT_EQ(data.call_count.load(), 1);
    ASSERT_TRUE(strstr(data.last_message, "42") != nullptr);
    ASSERT_TRUE(strstr(data.last_message, "test") != nullptr);
}

TEST(logging_null_callback) {
    // Should not crash with null callback
    set_callback(nullptr, nullptr);
    set_min_level(Level::Info);

    info("This should not crash");

    // No assertion - just verify no crash
}

TEST(logging_empty_message) {
    LogCallbackData data;
    set_callback(test_log_callback, &data);
    set_min_level(Level::Info);

    info("");

    ASSERT_EQ(data.call_count.load(), 1);
}

TEST(logging_long_message) {
    LogCallbackData data;
    set_callback(test_log_callback, &data);
    set_min_level(Level::Info);

    // Very long message (should be truncated)
    char long_msg[2048];
    memset(long_msg, 'A', sizeof(long_msg) - 1);
    long_msg[sizeof(long_msg) - 1] = '\0';

    info("%s", long_msg);

    ASSERT_EQ(data.call_count.load(), 1);
    // Message should be truncated but not crash
}

TEST(logging_thread_local_buffer) {
    LogCallbackData data;
    set_callback(test_log_callback, &data);
    set_min_level(Level::Info);

    // Multiple logs should use thread-local buffer
    for (int i = 0; i < 10; ++i) {
        info("Message %d", i);
    }

    ASSERT_EQ(data.call_count.load(), 10);
}
