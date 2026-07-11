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

TEST(error_scope_builds_typed_error_parts) {
  using namespace astral::core;

  set_last_error_parts("Invalid parameter: ", "model");
  ASSERT_EQ(std::strcmp(last_error(), "Invalid parameter: model"), 0);
}

TEST(error_scope_truncates_long_error_parts) {
  using namespace astral::core;

  char detail[600];
  std::memset(detail, 'x', sizeof(detail) - 1u);
  detail[sizeof(detail) - 1u] = '\0';

  set_last_error_parts("Backend error: ", detail);
  ASSERT_EQ(std::strlen(last_error()), 511u);
  ASSERT_EQ(std::memcmp(last_error(), "Backend error: ", 15u), 0);
  ASSERT_EQ(last_error()[511], '\0');
}
