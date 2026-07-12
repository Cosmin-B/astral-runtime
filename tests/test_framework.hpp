/**
 * test_framework.hpp - Minimal test framework for Astral
 *
 * Lightweight test infrastructure with no external dependencies.
 * Follows CODING_STANDARDS.md guidelines.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace astral::testing {

// Test registry for automatic test registration
class TestRegistry {
public:
  using TestFn = void (*)();

  struct TestCase {
    const char* name;
    TestFn fn;
  };

    struct TestSkipped {
        const char* file;
        int line;
        char message[512];

        TestSkipped(const char* f, int l, const char* msg) : file(f), line(l) {
            snprintf(message, sizeof(message), "%s", msg);
        }
    };

    struct TestFailure {
        const char* file;
        int line;
        char message[512];

        TestFailure(const char* f, int l, const char* msg) : file(f), line(l) {
            snprintf(message, sizeof(message), "%s", msg);
        }
    };

    static TestRegistry& instance() {
        static TestRegistry registry;
        return registry;
    }

    void register_test(const char* name, TestFn fn) {
        tests_.push_back({name, fn});
    }

    int run_all() {
        int passed = 0;
        int failed = 0;
        int skipped = 0;

        printf("\n=== Running %zu tests ===\n\n", tests_.size());

        for (const auto& test : tests_) {
            printf("[ RUN      ] %s\n", test.name);

            try {
                test.fn();
                printf("[       OK ] %s\n", test.name);
                ++passed;
            } catch (const TestSkipped& e) {
                printf("[  SKIPPED ] %s\n", test.name);
                printf("    %s:%d: %s\n", e.file, e.line, e.message);
                ++skipped;
            } catch (const TestFailure& e) {
                printf("[  FAILED  ] %s\n", test.name);
                printf("    %s:%d: %s\n", e.file, e.line, e.message);
                ++failed;
            } catch (...) {
                printf("[  FAILED  ] %s (unexpected exception)\n", test.name);
                ++failed;
            }
        }

        printf("\n=== Test Results ===\n");
        printf("[  PASSED  ] %d tests\n", passed);
        if (skipped > 0) {
            printf("[ SKIPPED  ] %d tests\n", skipped);
        }
        if (failed > 0) {
            printf("[  FAILED  ] %d tests\n", failed);
        }
        printf("===================\n\n");

        return failed;
    }

private:
    std::vector<TestCase> tests_;
};

// Test registration helper
struct TestRegistrar {
    TestRegistrar(const char* name, TestRegistry::TestFn fn) {
        TestRegistry::instance().register_test(name, fn);
    }
};

// Assertion failure handler
[[noreturn]] inline void test_fail(const char* file, int line, const char* expr) {
    throw TestRegistry::TestFailure(file, line, expr);
}

// Assertion failure handler with custom message
[[noreturn]] inline void test_fail_msg(const char* file, int line, const char* msg) {
    throw TestRegistry::TestFailure(file, line, msg);
}

// Explicitly report fixture/probe coverage that is not available in this run.
[[noreturn]] inline void test_skip_msg(const char* file, int line, const char* msg) {
    throw TestRegistry::TestSkipped(file, line, msg);
}

} // namespace astral::testing

// Test macros
#define TEST(name) \
    void test_##name(); \
    static ::astral::testing::TestRegistrar reg_##name(#name, test_##name); \
    void test_##name()

#define SKIP_TEST(msg) \
    do { \
        ::astral::testing::test_skip_msg(__FILE__, __LINE__, (msg)); \
    } while (0)

#define ASSERT_EQ(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (_a != _b) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "%s != %s (expected: %s, got: %s)", #a, #b, #b, #a); \
            ::astral::testing::test_fail(__FILE__, __LINE__, _msg); \
        } \
    } while (0)

#define ASSERT_NE(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (_a == _b) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "%s == %s (expected different values)", #a, #b); \
            ::astral::testing::test_fail(__FILE__, __LINE__, _msg); \
        } \
    } while (0)

#define ASSERT_TRUE(x) \
    do { \
        if (!(x)) { \
            ::astral::testing::test_fail(__FILE__, __LINE__, "Expected true: " #x); \
        } \
    } while (0)

#define ASSERT_FALSE(x) \
    do { \
        if (x) { \
            ::astral::testing::test_fail(__FILE__, __LINE__, "Expected false: " #x); \
        } \
    } while (0)

#define ASSERT_NULL(x) \
    do { \
        if ((x) != nullptr) { \
            ::astral::testing::test_fail(__FILE__, __LINE__, "Expected nullptr: " #x); \
        } \
    } while (0)

#define ASSERT_NOT_NULL(x) \
    do { \
        if ((x) == nullptr) { \
            ::astral::testing::test_fail(__FILE__, __LINE__, "Expected non-nullptr: " #x); \
        } \
    } while (0)

#define ASSERT_LT(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (!(_a < _b)) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "%s >= %s (expected %s < %s)", #a, #b, #a, #b); \
            ::astral::testing::test_fail(__FILE__, __LINE__, _msg); \
        } \
    } while (0)

#define ASSERT_LE(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (!(_a <= _b)) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "%s > %s (expected %s <= %s)", #a, #b, #a, #b); \
            ::astral::testing::test_fail(__FILE__, __LINE__, _msg); \
        } \
    } while (0)

#define ASSERT_GT(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (!(_a > _b)) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "%s <= %s (expected %s > %s)", #a, #b, #a, #b); \
            ::astral::testing::test_fail(__FILE__, __LINE__, _msg); \
        } \
    } while (0)

#define ASSERT_GE(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (!(_a >= _b)) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "%s < %s (expected %s >= %s)", #a, #b, #a, #b); \
            ::astral::testing::test_fail(__FILE__, __LINE__, _msg); \
        } \
    } while (0)
