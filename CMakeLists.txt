cmake_minimum_required(VERSION 3.20)
project(AstralRT VERSION 0.1.0 LANGUAGES C CXX)

# C++17 required, no extensions.
#
# Unreal 5.4 supports C++17 and C++20; Astral keeps the native core at C++17 so
# engine integrations can consume the stable C ABI without pulling newer C++
# requirements through private implementation headers.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(ASTRAL_BUILD_TESTS "Build tests" ON)
option(ASTRAL_BUILD_TSAN_TESTS "Build ThreadSanitizer test variants" OFF)
option(ASTRAL_BUILD_BENCHMARKS "Build benchmarks" ON)
option(ASTRAL_BUILD_EXAMPLES "Build examples" OFF)
option(ASTRAL_BUILD_UNITY_PLUGIN "Build Unity plugin" OFF)
option(ASTRAL_BUILD_UNREAL_PLUGIN "Build Unreal plugin" OFF)
option(ASTRAL_UNITY_STATIC_CXX_RUNTIME "Link Linux Unity plugin builds with a static C++ runtime" ON)
option(ASTRAL_NO_EXCEPTIONS "Build Astral sources without C++ exceptions (experimental; embedded)" OFF)
option(ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR "Enable JSON schema -> GBNF grammar support (requires exceptions from llama.cpp helpers)" ON)
option(ASTRAL_ENABLE_DYNAMIC_BACKENDS "Enable runtime-loadable backend providers (dlopen/LoadLibrary)" ON)
option(ASTRAL_ENABLE_THREADS "Enable Astral internal worker threads" ON)
option(ASTRAL_NO_THROW_ABI "Catch and translate exceptions at the C ABI boundary (recommended)" ON)
option(ASTRAL_ENABLE_VIRTUAL_MEMORY "Enable platform virtual memory primitives (mmap/VirtualAlloc)" ON)
option(ASTRAL_CPU_MEMORY_SOURCE_MMAP "CPU backend: materialize MEMORY/IO sources to temp files so llama.cpp can use mmap" ON)
option(ASTRAL_ENABLE_MTMD "Enable multimodal (vision/audio) support via llama.cpp libmtmd" OFF)
option(ASTRAL_ENABLE_TRACY "Enable Tracy profiler integration (profiling presets only; requires external/tracy submodule)" OFF)
option(ASTRAL_ENABLE_TRACY_MICRO "Enable Tracy micro-instrumentation (higher overhead; profiling builds only)" OFF)
option(ASTRAL_TRACY_ON_DEMAND "Tracy: on-demand mode (do not capture until connected)" ON)
option(ASTRAL_TRACY_ONLY_LOCALHOST "Tracy: only accept localhost connections" ON)
option(ASTRAL_TRACY_NO_BROADCAST "Tracy: disable discovery broadcast" ON)
option(ASTRAL_TRACY_CALLSTACK "Tracy: collect callstacks (higher overhead)" OFF)

# GPU backends (optional; keep CPU-only builds dependency-free).
option(ASTRAL_ENABLE_CUDA "Enable CUDA backend provider (requires CUDA toolkit; builds llama.cpp with GGML_CUDA=ON)" OFF)
option(ASTRAL_CUDA_FORCE_CUBLAS "CUDA: force cuBLAS over mmq kernels (llama.cpp GGML_CUDA_FORCE_CUBLAS)" OFF)
option(ASTRAL_CUDA_FORCE_MMQ "CUDA: force mmq kernels over cuBLAS (llama.cpp GGML_CUDA_FORCE_MMQ)" OFF)
set(ASTRAL_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA arch list for fatbin builds (CMAKE_CUDA_ARCHITECTURES)")

# Artifacts (desktop v0.1: ship both static + shared).
option(ASTRAL_BUILD_STATIC_LIB "Build static astral runtime library" ON)
option(ASTRAL_BUILD_SHARED_LIB "Build shared astral runtime library" OFF)

if(ASTRAL_NO_EXCEPTIONS AND ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR)
    message(WARNING "ASTRAL_NO_EXCEPTIONS=ON disables JSON schema grammar support; forcing ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR=OFF")
    set(ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR OFF CACHE BOOL "Enable JSON schema -> GBNF grammar support (requires exceptions from llama.cpp helpers)" FORCE)
endif()

# Unity desktop plugin requires a shared library. iOS requires a static library.
if(ASTRAL_BUILD_UNITY_PLUGIN AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(ASTRAL_BUILD_SHARED_LIB ON CACHE BOOL "Build shared astral runtime library" FORCE)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "iOS" AND ASTRAL_BUILD_SHARED_LIB)
    message(WARNING "iOS does not support the shared-library runtime artifact; forcing ASTRAL_BUILD_SHARED_LIB=OFF")
    set(ASTRAL_BUILD_SHARED_LIB OFF CACHE BOOL "Build shared astral runtime library" FORCE)
endif()

# Internal build products (tests/benches/examples) expect a static library by default.
if((ASTRAL_BUILD_TESTS OR ASTRAL_BUILD_BENCHMARKS OR ASTRAL_BUILD_EXAMPLES OR ASTRAL_BUILD_UNREAL_PLUGIN) AND NOT ASTRAL_BUILD_STATIC_LIB)
    set(ASTRAL_BUILD_STATIC_LIB ON CACHE BOOL "Build static astral runtime library" FORCE)
endif()

# Some shared-library consumers (Unity plugin, backend provider plugins) require
# all static deps to be built as PIC.
if(ASTRAL_BUILD_SHARED_LIB OR (ASTRAL_BUILD_UNITY_PLUGIN AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS") OR ASTRAL_BUILD_UNREAL_PLUGIN OR ASTRAL_BUILD_TESTS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(ASTRAL_BUILD_UNITY_PLUGIN AND ASTRAL_BUILD_UNREAL_PLUGIN)
    message(FATAL_ERROR "ASTRAL_BUILD_UNITY_PLUGIN and ASTRAL_BUILD_UNREAL_PLUGIN cannot both be ON in the same build")
endif()

if(ASTRAL_BUILD_UNREAL_PLUGIN)
    add_compile_definitions(JSON_HAS_CPP_17)
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

if(NOT ASTRAL_ENABLE_VIRTUAL_MEMORY)
    list(FILTER ASTRAL_SOURCES EXCLUDE REGEX ".*/src/platform/vm_.*\\.cpp$")
    if(ASTRAL_CPU_MEMORY_SOURCE_MMAP)
        message(WARNING "ASTRAL_ENABLE_VIRTUAL_MEMORY=OFF disables temp-file mmap model loading for MEMORY/IO sources; forcing ASTRAL_CPU_MEMORY_SOURCE_MMAP=OFF")
        set(ASTRAL_CPU_MEMORY_SOURCE_MMAP OFF CACHE BOOL "CPU backend: materialize MEMORY/IO sources to temp files so llama.cpp can use mmap" FORCE)
    endif()
elseif(WIN32)
    list(FILTER ASTRAL_SOURCES EXCLUDE REGEX ".*/src/platform/vm_(linux|macos)\\.cpp$")
elseif(APPLE)
    list(FILTER ASTRAL_SOURCES EXCLUDE REGEX ".*/src/platform/vm_(linux|win32)\\.cpp$")
else()
    list(FILTER ASTRAL_SOURCES EXCLUDE REGEX ".*/src/platform/vm_(macos|win32)\\.cpp$")
endif()

if(WIN32)
    list(FILTER ASTRAL_SOURCES EXCLUDE REGEX ".*/src/platform/thread_posix\\.cpp$")
else()
    list(FILTER ASTRAL_SOURCES EXCLUDE REGEX ".*/src/platform/thread_win32\\.cpp$")
endif()

# Mobile platforms: avoid temp-file-based mmap paths by default (sandboxed filesystems, non-standard temp locations).
if((CMAKE_SYSTEM_NAME STREQUAL "iOS" OR ANDROID) AND ASTRAL_CPU_MEMORY_SOURCE_MMAP)
    message(WARNING "ASTRAL_CPU_MEMORY_SOURCE_MMAP=ON is not supported on ${CMAKE_SYSTEM_NAME}; forcing OFF")
    set(ASTRAL_CPU_MEMORY_SOURCE_MMAP OFF CACHE BOOL "CPU backend: materialize MEMORY/IO sources to temp files so llama.cpp can use mmap" FORCE)
endif()

# Tracy integration (optional; must remain zero-cost when OFF).
set(_ASTRAL_TRACY_INCLUDE "")
if(ASTRAL_ENABLE_TRACY)
    set(_ASTRAL_TRACY_CLIENT "${CMAKE_CURRENT_SOURCE_DIR}/external/tracy/public/TracyClient.cpp")
    set(_ASTRAL_TRACY_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/external/tracy/public")
    if(NOT EXISTS "${_ASTRAL_TRACY_CLIENT}")
        message(FATAL_ERROR
            "ASTRAL_ENABLE_TRACY=ON but Tracy submodule is missing.\n"
            "Initialize it with:\n"
            "  git submodule update --init --recursive external/tracy\n"
            "Or add it (recommended):\n"
            "  git submodule add https://github.com/wolfpld/tracy.git external/tracy"
        )
    endif()

    add_library(astral_tracy_client STATIC "${_ASTRAL_TRACY_CLIENT}")
    target_include_directories(astral_tracy_client PUBLIC "${_ASTRAL_TRACY_INCLUDE}")
    target_compile_definitions(astral_tracy_client PUBLIC TRACY_ENABLE=1)

    if(ASTRAL_TRACY_ON_DEMAND)
        target_compile_definitions(astral_tracy_client PUBLIC TRACY_ON_DEMAND=1)
    endif()
    if(ASTRAL_TRACY_ONLY_LOCALHOST)
        target_compile_definitions(astral_tracy_client PUBLIC TRACY_ONLY_LOCALHOST=1)
    endif()
    if(ASTRAL_TRACY_NO_BROADCAST)
        target_compile_definitions(astral_tracy_client PUBLIC TRACY_NO_BROADCAST=1)
    endif()
    if(ASTRAL_TRACY_CALLSTACK)
        target_compile_definitions(astral_tracy_client PUBLIC TRACY_CALLSTACK=1)
    endif()

    # Third-party: don't fail the build on Tracy warnings.
    target_compile_options(astral_tracy_client PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-w>
        $<$<CXX_COMPILER_ID:MSVC>:/w>
    )

    find_package(Threads REQUIRED)
    target_link_libraries(astral_tracy_client PRIVATE Threads::Threads)
    if(WIN32)
        target_link_libraries(astral_tracy_client PRIVATE ws2_32)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(astral_tracy_client PRIVATE dl)
    endif()
endif()

# Runtime libraries.
if(NOT ASTRAL_SOURCES)
    message(WARNING "No source files found, creating minimal library")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp "// Placeholder\n")
    set(ASTRAL_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/empty.cpp")
endif()

function(astral_configure_runtime_target tgt)
    target_include_directories(${tgt} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )

    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/common
        ${LLAMA_CPP_DIR}/vendor
        $<$<BOOL:${ASTRAL_ENABLE_MTMD}>:${LLAMA_CPP_DIR}/tools/mtmd>
    )

    target_link_libraries(${tgt} PRIVATE llama ${ASTRAL_LLAMA_COMMON_TARGET})
    if(TARGET cpp-httplib)
        # remote_backend.cpp includes httplib types directly. Link the target so
        # ABI-affecting definitions such as CPPHTTPLIB_OPENSSL_SUPPORT propagate.
        target_link_libraries(${tgt} PRIVATE cpp-httplib)
    endif()
    if(ASTRAL_ENABLE_MTMD)
        target_link_libraries(${tgt} PRIVATE mtmd)
    endif()

    target_compile_options(${tgt} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror -Wno-unused-parameter>
        # C4324 reports the intentional tail padding created by cache-line-aligned queue cursors.
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /wd4324>
    )

    apply_compiler_flags(${tgt})

    if(MSVC)
        target_compile_definitions(${tgt} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()

    if(ASTRAL_NO_EXCEPTIONS)
        if(MSVC)
            target_compile_options(${tgt} PRIVATE /EHs-c-)
        else()
            target_compile_options(${tgt} PRIVATE -fno-exceptions)
        endif()
    endif()

    if(ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR)
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR=1)
    else()
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR=0)
    endif()

    if(ASTRAL_ENABLE_MTMD)
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_MTMD=1)
    else()
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_MTMD=0)
    endif()

    if(ASTRAL_ENABLE_DYNAMIC_BACKENDS)
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_DYNAMIC_BACKENDS=1)
    else()
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_DYNAMIC_BACKENDS=0)
    endif()

    if(ASTRAL_ENABLE_VIRTUAL_MEMORY)
        target_compile_definitions(${tgt} PUBLIC ASTRAL_ENABLE_VIRTUAL_MEMORY=1)
    else()
        target_compile_definitions(${tgt} PUBLIC ASTRAL_ENABLE_VIRTUAL_MEMORY=0)
    endif()

    if(ASTRAL_ENABLE_CUDA)
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_CUDA=1)
    else()
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_CUDA=0)
    endif()

    if(ASTRAL_ENABLE_THREADS)
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_THREADS=1)
    else()
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_THREADS=0)
    endif()

    if(ASTRAL_NO_THROW_ABI)
        target_compile_definitions(${tgt} PRIVATE ASTRAL_NO_THROW_ABI=1)
    else()
        target_compile_definitions(${tgt} PRIVATE ASTRAL_NO_THROW_ABI=0)
    endif()

    if(ASTRAL_ENABLE_TRACY)
        target_link_libraries(${tgt} PRIVATE astral_tracy_client)
        target_include_directories(${tgt} PRIVATE "${_ASTRAL_TRACY_INCLUDE}")
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_TRACY=1)
        if(ASTRAL_ENABLE_TRACY_MICRO)
            target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_TRACY_MICRO=1)
        else()
            target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_TRACY_MICRO=0)
        endif()
        target_compile_definitions(${tgt} PRIVATE TRACY_ENABLE=1)
    else()
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_TRACY=0)
        target_compile_definitions(${tgt} PRIVATE ASTRAL_ENABLE_TRACY_MICRO=0)
    endif()

    if(ASTRAL_CPU_MEMORY_SOURCE_MMAP)
        target_compile_definitions(${tgt} PRIVATE ASTRAL_CPU_MEMORY_SOURCE_MMAP=1)
    else()
        target_compile_definitions(${tgt} PRIVATE ASTRAL_CPU_MEMORY_SOURCE_MMAP=0)
    endif()
endfunction()

if(ASTRAL_BUILD_STATIC_LIB)
    add_library(astral_rt STATIC ${ASTRAL_SOURCES})
    astral_configure_runtime_target(astral_rt)
endif()

if(ASTRAL_BUILD_SHARED_LIB)
    add_library(astral_rt_shared SHARED ${ASTRAL_SOURCES})
    astral_configure_runtime_target(astral_rt_shared)

    set_target_properties(astral_rt_shared PROPERTIES OUTPUT_NAME "astral_rt")
    set_target_properties(astral_rt_shared PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        C_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )

    if(UNIX AND NOT APPLE)
        set(_ASTRAL_EXPORTS_MAP "${CMAKE_CURRENT_BINARY_DIR}/astral_rt_exports.map")
        file(WRITE "${_ASTRAL_EXPORTS_MAP}" "{\n  global:\n    astral_*;\n  local:\n    *;\n};\n")
        target_link_options(astral_rt_shared PRIVATE
            "LINKER:--version-script=${_ASTRAL_EXPORTS_MAP}"
            "LINKER:--exclude-libs,ALL"
        )
        if(ASTRAL_BUILD_UNITY_PLUGIN AND ASTRAL_UNITY_STATIC_CXX_RUNTIME)
            target_link_options(astral_rt_shared PRIVATE
                $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-static-libstdc++>
                $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-static-libgcc>
            )
        endif()
    endif()

    if(WIN32)
        target_compile_definitions(astral_rt_shared PRIVATE ASTRAL_BUILD_DLL)
        target_compile_definitions(astral_rt_shared INTERFACE ASTRAL_USE_DLL)
    endif()
endif()

if(WIN32 AND TARGET astral_rt AND TARGET astral_rt_shared)
    set_target_properties(astral_rt PROPERTIES OUTPUT_NAME "astral_rt_static")
endif()

# Keep backend object library (if built) consistent with the main library flags/defs.
if(TARGET cpu_backend)
    if(ASTRAL_NO_EXCEPTIONS)
        if(MSVC)
            target_compile_options(cpu_backend PRIVATE /EHs-c-)
        else()
            target_compile_options(cpu_backend PRIVATE -fno-exceptions)
        endif()
    endif()

    if(ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR)
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR=1)
    else()
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR=0)
    endif()

    if(ASTRAL_ENABLE_THREADS)
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_THREADS=1)
    else()
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_THREADS=0)
    endif()

    if(ASTRAL_ENABLE_VIRTUAL_MEMORY)
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_VIRTUAL_MEMORY=1)
    else()
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_VIRTUAL_MEMORY=0)
    endif()

    if(ASTRAL_ENABLE_CUDA)
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_CUDA=1)
    else()
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_CUDA=0)
    endif()

    if(ASTRAL_NO_THROW_ABI)
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_NO_THROW_ABI=1)
    else()
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_NO_THROW_ABI=0)
    endif()

    if(ASTRAL_ENABLE_TRACY)
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_TRACY=1)
        if(ASTRAL_ENABLE_TRACY_MICRO)
            target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_TRACY_MICRO=1)
        else()
            target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_TRACY_MICRO=0)
        endif()
        target_include_directories(cpu_backend PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/external/tracy/public")
        target_compile_definitions(cpu_backend PRIVATE TRACY_ENABLE=1)
    else()
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_TRACY=0)
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_ENABLE_TRACY_MICRO=0)
    endif()

    if(ASTRAL_CPU_MEMORY_SOURCE_MMAP)
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_CPU_MEMORY_SOURCE_MMAP=1)
    else()
        target_compile_definitions(cpu_backend PRIVATE ASTRAL_CPU_MEMORY_SOURCE_MMAP=0)
    endif()
endif()

# Consumers of `astral_rt` need these when linking static builds on Linux.
if(UNIX AND NOT APPLE)
    if(TARGET astral_rt)
        target_link_libraries(astral_rt PUBLIC pthread)
        if(ASTRAL_ENABLE_DYNAMIC_BACKENDS)
            target_link_libraries(astral_rt PUBLIC dl)
        endif()
    endif()

    if(TARGET astral_rt_shared)
        target_link_libraries(astral_rt_shared PRIVATE pthread)
        if(ASTRAL_ENABLE_DYNAMIC_BACKENDS)
            target_link_libraries(astral_rt_shared PRIVATE dl)
        endif()
    endif()
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
if(TARGET astral_rt)
    install(TARGETS astral_rt
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

if(TARGET astral_rt_shared)
    install(TARGETS astral_rt_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(FILES include/astral_rt.h DESTINATION include)
install(FILES include/astral_backend.h DESTINATION include)
install(FILES include/astral_backend_plugin.h DESTINATION include)

# Packaging (optional)
set(CPACK_PACKAGE_NAME "astral")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_GENERATOR "ZIP")
include(CPack)
