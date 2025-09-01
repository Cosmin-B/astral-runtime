# Astral Utils - UTF-8, logging, and string utilities

# UTF-8 utilities
add_library(astral_utf8 STATIC
    utf8.cpp
)

target_include_directories(astral_utf8 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(astral_utf8 PRIVATE cxx_std_20)

# Enable SIMD optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(astral_utf8 PRIVATE -mavx2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    target_compile_options(astral_utf8 PRIVATE -march=armv8-a)
endif()

# Logging utilities
add_library(astral_logging STATIC
    logging.cpp
)

target_include_directories(astral_logging PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(astral_logging PRIVATE cxx_std_20)

# Test executable (if testing enabled)
if(ASTRAL_BUILD_TESTS OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_executable(test_utils
        test_utils.cpp
    )

    target_link_libraries(test_utils PRIVATE
        astral_utf8
        astral_logging
    )

    target_include_directories(test_utils PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    target_compile_features(test_utils PRIVATE cxx_std_20)

    # Enable warnings
    if(MSVC)
        target_compile_options(test_utils PRIVATE /W4)
    else()
        target_compile_options(test_utils PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # Add test
    add_test(NAME utils_test COMMAND test_utils)
endif()
