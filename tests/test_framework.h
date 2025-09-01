/**
 * test_framework.h - Minimal test framework for Astral
 *
 * No external dependencies (no Google Test, no Catch2).
 * Simple macro-based test registration.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

// Test macros
#define TEST(name) \
    static void test_##name(); \
    static TestRegistrar reg_##name(#name, test_##name); \
    static void test_##name()

#define ASSERT_EQ(a, b) \
    do { \
        auto val_a = (a); \
        auto val_b = (b); \
        if (val_a != val_b) { \
            fprintf(stderr, "[FAIL] %s:%d: ASSERT_EQ failed: %s != %s\n", \
                    __FILE__, __LINE__, #a, #b); \
            g_test_failed = true; \
            return; \
        } \
    } while (0)

#define ASSERT_NE(a, b) \
    do { \
        auto val_a = (a); \
        auto val_b = (b); \
        if (val_a == val_b) { \
            fprintf(stderr, "[FAIL] %s:%d: ASSERT_NE failed: %s == %s\n", \
                    __FILE__, __LINE__, #a, #b); \
            g_test_failed = true; \
            return; \
        } \
    } while (0)

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "[FAIL] %s:%d: ASSERT_TRUE failed: %s is false\n", \
                    __FILE__, __LINE__, #cond); \
            g_test_failed = true; \
            return; \
        } \
    } while (0)

#define ASSERT_FALSE(cond) \
    do { \
        if (cond) { \
            fprintf(stderr, "[FAIL] %s:%d: ASSERT_FALSE failed: %s is true\n", \
                    __FILE__, __LINE__, #cond); \
            g_test_failed = true; \
            return; \
        } \
    } while (0)

#define ASSERT_GT(a, b) \
    do { \
        auto val_a = (a); \
        auto val_b = (b); \
        if (!(val_a > val_b)) { \
            fprintf(stderr, "[FAIL] %s:%d: ASSERT_GT failed: %s <= %s\n", \
                    __FILE__, __LINE__, #a, #b); \
            g_test_failed = true; \
            return; \
        } \
    } while (0)

#define ASSERT_GE(a, b) \
    do { \
        auto val_a = (a); \
        auto val_b = (b); \
        if (!(val_a >= val_b)) { \
            fprintf(stderr, "[FAIL] %s:%d: ASSERT_GE failed: %s < %s\n", \
                    __FILE__, __LINE__, #a, #b); \
            g_test_failed = true; \
            return; \
        } \
    } while (0)

#define ASSERT_LT(a, b) \
    do { \
        auto val_a = (a); \
        auto val_b = (b); \
        if (!(val_a < val_b)) { \
            fprintf(stderr, "[FAIL] %s:%d: ASSERT_LT failed: %s >= %s\n", \
                    __FILE__, __LINE__, #a, #b); \
            g_test_failed = true; \
            return; \
        } \
    } while (0)

#define ASSERT_LE(a, b) \
    do { \
        auto val_a = (a); \
        auto val_b = (b); \
        if (!(val_a <= val_b)) { \
            fprintf(stderr, "[FAIL] %s:%d: ASSERT_LE failed: %s > %s\n", \
                    __FILE__, __LINE__, #a, #b); \
            g_test_failed = true; \
            return; \
        } \
    } while (0)

// Global test state
extern bool g_test_failed;

// Test registration
struct TestCase {
    const char* name;
    std::function<void()> func;
};

class TestRegistrar {
public:
    TestRegistrar(const char* name, void (*func)()) {
        get_tests().push_back({name, func});
    }

    static std::vector<TestCase>& get_tests();
};

// Test runner
int run_all_tests();
