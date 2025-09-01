add_library(astral_backend_toy_plugin SHARED
    toy_backend_plugin.cpp
)

target_include_directories(astral_backend_toy_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(astral_backend_toy_plugin PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

# Keep the plugin next to the tests so it can be loaded without path gymnastics.
set_target_properties(astral_backend_toy_plugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

