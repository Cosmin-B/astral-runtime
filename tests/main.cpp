/**
 * main.cpp - Test runner entry point
 */

#include "test_framework.hpp"

int main() {
    return astral::testing::TestRegistry::instance().run_all();
}
