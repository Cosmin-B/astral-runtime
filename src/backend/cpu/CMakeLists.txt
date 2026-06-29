# CPU Backend CMakeLists.txt
#
# Builds CPU backend using llama.cpp.
# This is the default backend when gpu_layers=0.

# CPU backend sources
set(CPU_BACKEND_SOURCES
    cpu_backend.cpp
)

# CPU backend headers
set(CPU_BACKEND_HEADERS
    cpu_backend.hpp
)

# Create CPU backend library (object library, linked into astral_rt)
add_library(cpu_backend OBJECT ${CPU_BACKEND_SOURCES})

# Include directories
target_include_directories(cpu_backend PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/vendor
)

target_include_directories(cpu_backend PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Link against llama.cpp
target_link_libraries(cpu_backend PUBLIC llama::llama ${ASTRAL_LLAMA_COMMON_TARGET})

# Compiler flags (same as parent)
target_compile_options(cpu_backend PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

if(MSVC)
    target_compile_definitions(cpu_backend PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Add CPU backend to parent library
# Note: Main CMakeLists.txt already includes backend/*.cpp via glob
# This file is for explicit linking and configuration only
