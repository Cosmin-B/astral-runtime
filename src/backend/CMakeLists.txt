# Backend subsystem CMakeLists.txt
#
# Manages backend implementations and llama.cpp integration.
#
# llama.cpp Integration Strategy:
# 1. Try to find pre-built llama.cpp via find_package()
# 2. If not found, build llama.cpp as ExternalProject
# 3. Link CPU backend against llama library
#
# Future GPU backends (CUDA, Metal, DirectML) will be added here.

# Find or build llama.cpp.
#
# Preferred layout (submodule): `astral/external/llama.cpp`
# Fallback layout (sibling): `../llama.cpp`
get_filename_component(_ASTRAL_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(_LLAMA_CPP_DIR_DEFAULT "${_ASTRAL_ROOT_DIR}/external/llama.cpp")
if(NOT EXISTS "${_LLAMA_CPP_DIR_DEFAULT}/CMakeLists.txt")
    set(_LLAMA_CPP_DIR_DEFAULT "${_ASTRAL_ROOT_DIR}/../llama.cpp")
endif()

set(LLAMA_CPP_DIR "${_LLAMA_CPP_DIR_DEFAULT}" CACHE PATH "llama.cpp source directory")
unset(_LLAMA_CPP_DIR_DEFAULT)
unset(_ASTRAL_ROOT_DIR)

# Check if llama.cpp directory exists
if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(STATUS "Found llama.cpp at: ${LLAMA_CPP_DIR}")

    # Some shared-library consumers (Unity plugin, backend provider plugins) can end up
    # picking static OpenMP runtimes on Linux, which breaks shared linking (TLS relocations).
    if((ASTRAL_BUILD_UNITY_PLUGIN AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS") OR ASTRAL_BUILD_TESTS)
        set(GGML_OPENMP OFF CACHE BOOL "ggml: disable OpenMP for shared consumers/tests" FORCE)
    endif()
    
    # Add llama.cpp as subdirectory
    # This builds llama.cpp as part of Astral build
    # Configure llama.cpp build options
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
    set(GGML_STATIC ON CACHE BOOL "Build static library" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama.cpp tests" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable llama.cpp examples" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable llama.cpp server" FORCE)
    
    add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
    
    # Alias for consistency
    add_library(llama::llama ALIAS llama)
else()
    message(FATAL_ERROR "llama.cpp not found at: ${LLAMA_CPP_DIR}\n"
                        "Please clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp ${LLAMA_CPP_DIR}")
endif()

# CPU backend subdirectory
add_subdirectory(cpu)

# Future GPU backends (v0.2+)
# add_subdirectory(cuda)
# add_subdirectory(metal)
# add_subdirectory(directml)
