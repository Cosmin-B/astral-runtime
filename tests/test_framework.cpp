/**
 * test_framework.cpp - Minimal test framework implementation
 */

#include "test_framework.h"
#include <cstdio>

// Global test state
bool g_test_failed = false;

std::vector<TestCase>& TestRegistrar::get_tests() {
    static std::vector<TestCase> tests;
    return tests;
}

int run_all_tests() {
    auto& tests = TestRegistrar::get_tests();

    if (tests.empty()) {
        fprintf(stderr, "No tests registered\n");
        return 1;
    }

    int passed = 0;
    int failed = 0;

    fprintf(stdout, "Running %zu tests...\n", tests.size());
    fprintf(stdout, "================================================================================\n");

    for (const auto& test : tests) {
        g_test_failed = false;

        fprintf(stdout, "[ RUN      ] %s\n", test.name);
        test.func();

        if (g_test_failed) {
            fprintf(stdout, "[  FAILED  ] %s\n", test.name);
            failed++;
        } else {
            fprintf(stdout, "[       OK ] %s\n", test.name);
            passed++;
        }
    }

    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "Tests: %d total, %d passed, %d failed\n",
            passed + failed, passed, failed);

    return (failed == 0) ? 0 : 1;
}
