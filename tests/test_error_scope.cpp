/**
 * test_error_scope.cpp - last-error stack/ring behavior
 */

#include "test_framework.hpp"

#include "../src/core/error.hpp"

#include <cstring>

TEST(error_scope_nested_preserves_outer) {
    using namespace astral::core;

    clear_last_error();

    {
        ErrorScope outer;
        set_last_error("outer");
        ASSERT_EQ(std::strcmp(last_error(), "outer"), 0);

        {
            ErrorScope inner;
            set_last_error("inner");
            ASSERT_EQ(std::strcmp(last_error(), "inner"), 0);
        }

        ASSERT_EQ(std::strcmp(last_error(), "outer"), 0);
    }

    // After the outermost call returns, slot 0 remains the last-error buffer.
    ASSERT_EQ(std::strcmp(last_error(), "outer"), 0);
}

