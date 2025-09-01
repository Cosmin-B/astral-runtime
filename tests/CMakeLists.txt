# Astral Runtime Test Suite
# Comprehensive unit tests for all subsystems

cmake_minimum_required(VERSION 3.20)

# Enable testing
enable_testing()

# Test framework (static library with implementation)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_framework.cpp)
    add_library(astral_test_framework STATIC
        test_framework.cpp
    )
    target_include_directories(astral_test_framework PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    message(STATUS "Test framework library configured")
else()
    # Fallback: header-only interface
    add_library(astral_test_framework INTERFACE)
    target_include_directories(astral_test_framework INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# List of test files
set(TEST_FILES
    test_platform
    test_memory
    test_concurrency
    test_utf8
    test_logging
    test_core
    test_backend
    test_inference
    test_embeddings
    test_provider_harness
)

# Create test executables and register with CTest
foreach(test_file ${TEST_FILES})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${test_file}.cpp)
        # Create executable
        add_executable(${test_file}
            ${test_file}.cpp
            main.cpp
        )

        # Link dependencies
        target_link_libraries(${test_file} PRIVATE
            astral_rt
            astral_test_framework
        )

        # Set C++20 standard
        set_target_properties(${test_file} PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )

        # Include directories
        target_include_directories(${test_file} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
        )

        # Platform-specific settings
        if(UNIX AND NOT APPLE)
            # Linux: link pthread for std::thread
            target_link_libraries(${test_file} PRIVATE pthread dl)
        endif()

        # Add to CTest
        add_test(NAME ${test_file} COMMAND ${test_file})

        # Set test properties
        set_tests_properties(${test_file} PROPERTIES
            TIMEOUT 30
            LABELS "unit"
        )

        message(STATUS "Configured test: ${test_file}")
    endif()
endforeach()

# Backend plugin test wiring (sample plugin).
if(TARGET test_provider_harness AND TARGET astral_backend_sample_plugin)
    add_dependencies(test_provider_harness astral_backend_sample_plugin)
    target_compile_definitions(test_provider_harness PRIVATE
        ASTRAL_SAMPLE_BACKEND_PLUGIN_PATH="$<TARGET_FILE:astral_backend_sample_plugin>"
    )
endif()

# Backend plugin test wiring (GGUF-capable CPU llama plugin).
if(TARGET test_provider_harness AND TARGET astral_backend_cpu_llama_plugin)
    add_dependencies(test_provider_harness astral_backend_cpu_llama_plugin)
    target_compile_definitions(test_provider_harness PRIVATE
        ASTRAL_CPU_LLAMA_BACKEND_PLUGIN_PATH="$<TARGET_FILE:astral_backend_cpu_llama_plugin>"
    )
endif()

# Backend plugin test wiring (toy provider plugin).
if(TARGET test_provider_harness AND TARGET astral_backend_toy_plugin)
    add_dependencies(test_provider_harness astral_backend_toy_plugin)
    target_compile_definitions(test_provider_harness PRIVATE
        ASTRAL_TOY_BACKEND_PLUGIN_PATH="$<TARGET_FILE:astral_backend_toy_plugin>"
    )
endif()

# Thread Sanitizer tests (optional, for concurrency tests)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    find_program(ASTRAL_SETARCH_EXECUTABLE setarch)
    foreach(test_file test_concurrency test_memory)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${test_file}.cpp)
            add_executable(${test_file}_tsan
                ${test_file}.cpp
                main.cpp
            )

            target_link_libraries(${test_file}_tsan PRIVATE
                astral_rt
                astral_test_framework
            )

            target_compile_options(${test_file}_tsan PRIVATE
                -fsanitize=thread
                -g
            )

            target_link_options(${test_file}_tsan PRIVATE
                -fsanitize=thread
            )

            set_target_properties(${test_file}_tsan PROPERTIES
                CXX_STANDARD 20
                CXX_STANDARD_REQUIRED ON
            )

            target_include_directories(${test_file}_tsan PRIVATE
                ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/src
            )

            if(UNIX AND NOT APPLE)
                target_link_libraries(${test_file}_tsan PRIVATE pthread dl)
            endif()

            if(UNIX AND NOT APPLE AND ASTRAL_SETARCH_EXECUTABLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
                # ThreadSanitizer has known issues with high-entropy ASLR on some kernels; disable ASLR for the test.
                add_test(NAME ${test_file}_tsan
                         COMMAND ${ASTRAL_SETARCH_EXECUTABLE} x86_64 -R $<TARGET_FILE:${test_file}_tsan>)
            else()
                add_test(NAME ${test_file}_tsan COMMAND $<TARGET_FILE:${test_file}_tsan>)
            endif()
            set_tests_properties(${test_file}_tsan PROPERTIES
                LABELS "tsan;sanitizer"
                TIMEOUT 60
            )

            message(STATUS "Configured TSan test: ${test_file}_tsan")
        endif()
    endforeach()
endif()

# Integration test executable (if exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_integration.cpp)
    add_executable(test_integration
        main.cpp
        test_integration.cpp
    )

    target_link_libraries(test_integration PRIVATE
        astral_rt
        astral_test_framework
    )

    target_include_directories(test_integration PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    add_test(NAME test_integration COMMAND test_integration)
    set_tests_properties(test_integration PROPERTIES
        TIMEOUT 300
        LABELS "integration"
    )

    message(STATUS "Configured test: test_integration")
endif()

# Source-level validation gates (provider-agnostic hard rules)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_source_scans.cmake)
    add_test(
        NAME gate_source_scans
        COMMAND ${CMAKE_COMMAND}
            -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_source_scans.cmake
    )
    set_tests_properties(gate_source_scans PROPERTIES
        TIMEOUT 30
        LABELS "gate"
    )
    message(STATUS "Configured test: gate_source_scans")
endif()

# Print test summary
message(STATUS "========================================")
message(STATUS "Astral Test Suite Configuration")
message(STATUS "========================================")
message(STATUS "Unit tests: ${TEST_FILES}")
message(STATUS "")
message(STATUS "Run tests with:")
message(STATUS "  All tests:         ctest")
message(STATUS "  Specific test:     ctest -R test_platform")
message(STATUS "  Verbose output:    ctest -V")
message(STATUS "  Parallel:          ctest -j8")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(STATUS "  Thread Sanitizer:  ctest -L tsan")
endif()
message(STATUS "========================================")

# Gate: no allocations in steady-state decode/stream (best-effort portable).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gate_allocations.cpp)
    add_executable(gate_allocations
        gate_allocations.cpp
        main.cpp
    )

    target_link_libraries(gate_allocations PRIVATE
        astral_rt
        astral_test_framework
    )

    target_include_directories(gate_allocations PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    if(UNIX AND NOT APPLE)
        target_link_libraries(gate_allocations PRIVATE pthread dl)
        target_link_options(gate_allocations PRIVATE
            "-Wl,--wrap=malloc"
            "-Wl,--wrap=calloc"
            "-Wl,--wrap=realloc"
            "-Wl,--wrap=free"
            "-Wl,--wrap=posix_memalign"
        )
        target_compile_definitions(gate_allocations PRIVATE ASTRAL_ALLOC_WRAP=1)
    elseif(APPLE)
        target_link_options(gate_allocations PRIVATE
            "-Wl,-wrap,malloc"
            "-Wl,-wrap,calloc"
            "-Wl,-wrap,realloc"
            "-Wl,-wrap,free"
            "-Wl,-wrap,posix_memalign"
        )
        target_compile_definitions(gate_allocations PRIVATE ASTRAL_ALLOC_WRAP=1)
    endif()

    add_test(NAME gate_allocations COMMAND gate_allocations)
    set_tests_properties(gate_allocations PROPERTIES
        TIMEOUT 60
        LABELS "gate"
    )
    message(STATUS "Configured test: gate_allocations")
endif()

# Gate: no steady-state I/O syscalls in decode/stream (best-effort portable).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gate_io_syscalls.cpp)
    add_executable(gate_io_syscalls
        gate_io_syscalls.cpp
        main.cpp
    )

    target_link_libraries(gate_io_syscalls PRIVATE
        astral_rt
        astral_test_framework
    )

    target_include_directories(gate_io_syscalls PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    if(UNIX AND NOT APPLE)
        target_link_libraries(gate_io_syscalls PRIVATE pthread dl)
        target_link_options(gate_io_syscalls PRIVATE
            "-Wl,--wrap=read"
            "-Wl,--wrap=write"
            "-Wl,--wrap=pread"
            "-Wl,--wrap=pwrite"
            "-Wl,--wrap=lseek"
            "-Wl,--wrap=close"
            "-Wl,--wrap=mmap"
            "-Wl,--wrap=munmap"
            "-Wl,--wrap=mprotect"
            "-Wl,--wrap=madvise"
        )
        target_compile_definitions(gate_io_syscalls PRIVATE ASTRAL_IO_WRAP=1)
    elseif(APPLE)
        target_link_options(gate_io_syscalls PRIVATE
            "-Wl,-wrap,read"
            "-Wl,-wrap,write"
            "-Wl,-wrap,pread"
            "-Wl,-wrap,pwrite"
            "-Wl,-wrap,lseek"
            "-Wl,-wrap,close"
            "-Wl,-wrap,mmap"
            "-Wl,-wrap,munmap"
            "-Wl,-wrap,mprotect"
            "-Wl,-wrap,madvise"
        )
        target_compile_definitions(gate_io_syscalls PRIVATE ASTRAL_IO_WRAP=1)
    endif()

    add_test(NAME gate_io_syscalls COMMAND gate_io_syscalls)
    set_tests_properties(gate_io_syscalls PROPERTIES
        TIMEOUT 60
        LABELS "gate"
    )
    message(STATUS "Configured test: gate_io_syscalls")
endif()

# Gate: RSS cap (embedded validation, Linux-only enforcement).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gate_rss_cap.cpp)
    add_executable(gate_rss_cap
        gate_rss_cap.cpp
        main.cpp
    )

    target_link_libraries(gate_rss_cap PRIVATE
        astral_rt
        astral_test_framework
    )

    target_include_directories(gate_rss_cap PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    if(UNIX AND NOT APPLE)
        target_link_libraries(gate_rss_cap PRIVATE pthread dl)
    endif()

    add_test(NAME gate_rss_cap COMMAND gate_rss_cap)
    set_tests_properties(gate_rss_cap PROPERTIES
        TIMEOUT 60
        LABELS "gate"
    )
    message(STATUS "Configured test: gate_rss_cap")
endif()
