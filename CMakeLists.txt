cmake_minimum_required(VERSION 3.20)
project(AstralRT VERSION 0.1.0 LANGUAGES CXX)

# C++20 required, no extensions
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(ASTRAL_BUILD_TESTS "Build tests" ON)
option(ASTRAL_BUILD_BENCHMARKS "Build benchmarks" ON)
option(ASTRAL_BUILD_EXAMPLES "Build examples" OFF)
option(ASTRAL_BUILD_UNITY_PLUGIN "Build Unity plugin" OFF)
option(ASTRAL_BUILD_UNREAL_PLUGIN "Build Unreal plugin" OFF)

# Some shared-library consumers (Unity plugin, backend provider plugins) require
# all static deps to be built as PIC.
if((ASTRAL_BUILD_UNITY_PLUGIN AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS") OR ASTRAL_BUILD_TESTS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(ASTRAL_BUILD_UNITY_PLUGIN AND ASTRAL_BUILD_UNREAL_PLUGIN)
    message(FATAL_ERROR "ASTRAL_BUILD_UNITY_PLUGIN and ASTRAL_BUILD_UNREAL_PLUGIN cannot both be ON in the same build")
endif()

# Include compiler flags module
include(cmake/CompilerFlags.cmake)

# Add backend subdirectory (builds llama.cpp integration)
add_subdirectory(src/backend)

# Collect all source files from subdirectories (excluding test files)
file(GLOB_RECURSE ASTRAL_SOURCES_ALL
    "src/core/*.cpp"
    "src/memory/*.cpp"
    "src/concurrency/*.cpp"
    "src/platform/*.cpp"
    "src/backend/*.cpp"
    "src/inference/*.cpp"
    "src/utils/*.cpp"
)

# Filter out test files (test_*.cpp)
set(ASTRAL_SOURCES)
foreach(source_file ${ASTRAL_SOURCES_ALL})
    get_filename_component(filename ${source_file} NAME)
    if(NOT filename MATCHES "^test_")
        list(APPEND ASTRAL_SOURCES ${source_file})
    endif()
endforeach()

# Core library.
# - Default: static lib for apps and Unreal packaging.
# - Unity desktop/mobile plugin: shared lib (iOS remains static).
set(ASTRAL_RT_LIB_TYPE STATIC)
if(ASTRAL_BUILD_UNITY_PLUGIN AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(ASTRAL_RT_LIB_TYPE SHARED)
endif()

if(ASTRAL_SOURCES)
    add_library(astral_rt ${ASTRAL_RT_LIB_TYPE} ${ASTRAL_SOURCES})
else()
    # Fallback: Create library with at least one source file
    message(WARNING "No source files found, creating minimal library")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp "// Placeholder\n")
    add_library(astral_rt ${ASTRAL_RT_LIB_TYPE} ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp)
endif()

# Export symbols when building a DLL on Windows.
if(ASTRAL_RT_LIB_TYPE STREQUAL "SHARED")
    target_compile_definitions(astral_rt PRIVATE ASTRAL_BUILD_DLL)
endif()

# Public include directory
target_include_directories(astral_rt PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Private include directory for internal headers
target_include_directories(astral_rt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # llama.cpp includes (needed by the CPU backend and JSON schema -> GBNF support)
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/vendor
)

# Link llama.cpp library (built by backend subdirectory)
target_link_libraries(astral_rt PRIVATE llama common)

# Compiler warnings as errors
target_compile_options(astral_rt PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

# Apply platform-specific compiler flags
apply_compiler_flags(astral_rt)

# Consumers of `astral_rt` need these when linking static builds on Linux.
if(UNIX AND NOT APPLE)
    target_link_libraries(astral_rt PUBLIC pthread dl)
endif()

# Tests
if(ASTRAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(backend_plugins)
    add_subdirectory(tests)
endif()

# Benchmarks
if(ASTRAL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Examples
if(ASTRAL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Unity plugin
if(ASTRAL_BUILD_UNITY_PLUGIN)
    add_subdirectory(plugins/unity)
endif()

# Unreal plugin
if(ASTRAL_BUILD_UNREAL_PLUGIN)
    add_subdirectory(plugins/unreal)
endif()

# Installation rules
# Note: llama is not included in export (built as subdirectory, not installed)
install(TARGETS astral_rt
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/astral_rt.h DESTINATION include)
install(FILES include/astral_backend.h DESTINATION include)
install(FILES include/astral_backend_plugin.h DESTINATION include)
