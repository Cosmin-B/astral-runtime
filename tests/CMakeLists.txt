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
#
# When virtual memory is disabled, skip tests that require VM-backed runtime init/model load.
if(ASTRAL_ENABLE_VIRTUAL_MEMORY)
    set(TEST_FILES
        test_error_scope
        test_platform
        test_memory
        test_concurrency
        test_utf8
        test_logging
        test_core
        test_abi_invalid_args
        test_arena
        test_model_sources
        test_backend
        test_cuda_parity
        test_cuda_e2e
        test_inference
        test_continuous_batching
        test_embeddings
        test_media
        test_provider_harness
    )
else()
    set(TEST_FILES
        test_error_scope
        test_platform
        test_memory
        test_concurrency
        test_utf8
        test_logging
        test_arena
        test_model_sources
    )
endif()

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

        target_compile_definitions(${test_file} PRIVATE
            $<$<BOOL:${ASTRAL_ENABLE_CUDA}>:ASTRAL_ENABLE_CUDA=1>
            $<$<NOT:$<BOOL:${ASTRAL_ENABLE_CUDA}>>:ASTRAL_ENABLE_CUDA=0>
            $<$<BOOL:${ASTRAL_ENABLE_MTMD}>:ASTRAL_ENABLE_MTMD=1>
            $<$<NOT:$<BOOL:${ASTRAL_ENABLE_MTMD}>>:ASTRAL_ENABLE_MTMD=0>
            ASTRAL_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
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

        # Inference tests exercise timing-sensitive async scheduling + streaming.
        # Avoid parallel CTest flakiness on oversubscribed CI hosts.
        if(${test_file} STREQUAL "test_inference")
            set_tests_properties(${test_file} PROPERTIES RUN_SERIAL TRUE)
        endif()
        if(${test_file} STREQUAL "test_continuous_batching")
            set_tests_properties(${test_file} PROPERTIES RUN_SERIAL TRUE)
        endif()
        # CUDA E2E loads real models multiple times; keep it isolated to avoid resource contention.
        if(${test_file} STREQUAL "test_cuda_e2e")
            set_tests_properties(${test_file} PROPERTIES RUN_SERIAL TRUE)
        endif()
        # Integration is heavyweight and can be timing-sensitive under parallel CTest.
        if(${test_file} STREQUAL "test_integration")
            set_tests_properties(${test_file} PROPERTIES RUN_SERIAL TRUE)
        endif()

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

            target_compile_definitions(${test_file}_tsan PRIVATE
                $<$<BOOL:${ASTRAL_ENABLE_CUDA}>:ASTRAL_ENABLE_CUDA=1>
                $<$<NOT:$<BOOL:${ASTRAL_ENABLE_CUDA}>>:ASTRAL_ENABLE_CUDA=0>
                $<$<BOOL:${ASTRAL_ENABLE_MTMD}>:ASTRAL_ENABLE_MTMD=1>
                $<$<NOT:$<BOOL:${ASTRAL_ENABLE_MTMD}>>:ASTRAL_ENABLE_MTMD=0>
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
if(ASTRAL_ENABLE_VIRTUAL_MEMORY AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_integration.cpp)
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

if(TARGET astral_rt_shared AND UNIX AND NOT APPLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_shared_exports.cmake)
    add_test(
        NAME gate_shared_exports
        COMMAND ${CMAKE_COMMAND}
            -DASTRAL_SHARED_LIBRARY=$<TARGET_FILE:astral_rt_shared>
            -DASTRAL_PUBLIC_HEADER=${CMAKE_SOURCE_DIR}/include/astral_rt.h
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_shared_exports.cmake
    )
    set_tests_properties(gate_shared_exports PROPERTIES
        TIMEOUT 30
        LABELS "gate;abi"
    )
    message(STATUS "Configured test: gate_shared_exports")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_unreal_header_mirror.cmake)
    add_test(
        NAME gate_unreal_header_mirror
        COMMAND ${CMAKE_COMMAND}
            -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_unreal_header_mirror.cmake
    )
    set_tests_properties(gate_unreal_header_mirror PROPERTIES
        TIMEOUT 30
        LABELS "gate;abi;unreal"
    )
    message(STATUS "Configured test: gate_unreal_header_mirror")
endif()

if(UNIX AND NOT APPLE AND EXISTS ${CMAKE_SOURCE_DIR}/scripts/generate_abi_layout_report.sh)
    find_program(ASTRAL_BASH_EXECUTABLE bash)
    if(ASTRAL_BASH_EXECUTABLE)
        find_program(ASTRAL_PYTHON_EXECUTABLE python3)

        add_test(
            NAME gate_abi_layout_report
            COMMAND ${CMAKE_COMMAND} -E env
                CXX=${CMAKE_CXX_COMPILER}
                ${ASTRAL_BASH_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/generate_abi_layout_report.sh --check
        )
        set_tests_properties(gate_abi_layout_report PROPERTIES
            TIMEOUT 30
            LABELS "gate;abi"
        )
        message(STATUS "Configured test: gate_abi_layout_report")

        if(ASTRAL_PYTHON_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_metadata.cmake)
            add_test(
                NAME gate_release_metadata
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BUILD_DIR=${CMAKE_BINARY_DIR}
                    -DASTRAL_BASH_EXECUTABLE=${ASTRAL_BASH_EXECUTABLE}
                    -DASTRAL_PYTHON_EXECUTABLE=${ASTRAL_PYTHON_EXECUTABLE}
                    -DASTRAL_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_metadata.cmake
            )
            set_tests_properties(gate_release_metadata PROPERTIES
                TIMEOUT 30
                LABELS "gate;release;abi"
            )
            message(STATUS "Configured test: gate_release_metadata")
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_artifact_signature.cmake)
            add_test(
                NAME gate_release_artifact_signature
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BUILD_DIR=${CMAKE_BINARY_DIR}
                    -DASTRAL_BASH_EXECUTABLE=${ASTRAL_BASH_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_artifact_signature.cmake
            )
            set_tests_properties(gate_release_artifact_signature PROPERTIES
                TIMEOUT 30
                LABELS "gate;release"
            )
            message(STATUS "Configured test: gate_release_artifact_signature")
        endif()

        if(EXISTS ${CMAKE_SOURCE_DIR}/scripts/validate_release_notes.sh)
            add_test(
                NAME gate_release_notes
                COMMAND ${ASTRAL_BASH_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validate_release_notes.sh
            )
            set_tests_properties(gate_release_notes PROPERTIES
                TIMEOUT 30
                LABELS "gate;release"
            )
            message(STATUS "Configured test: gate_release_notes")
        endif()

        if(EXISTS ${CMAKE_SOURCE_DIR}/scripts/validate_dependency_pins.sh)
            add_test(
                NAME gate_dependency_pins
                COMMAND ${ASTRAL_BASH_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validate_dependency_pins.sh
            )
            set_tests_properties(gate_dependency_pins PROPERTIES
                TIMEOUT 30
                LABELS "gate;release"
            )
            message(STATUS "Configured test: gate_dependency_pins")
        endif()

        if(ASTRAL_PYTHON_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_comment_inventory.cmake)
            add_test(
                NAME gate_comment_inventory
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_PYTHON_EXECUTABLE=${ASTRAL_PYTHON_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_comment_inventory.cmake
            )
            set_tests_properties(gate_comment_inventory PROPERTIES
                TIMEOUT 30
                LABELS "gate"
            )
            message(STATUS "Configured test: gate_comment_inventory")
        endif()

        if(ASTRAL_PYTHON_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_doc_links.cmake)
            add_test(
                NAME gate_doc_links
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_PYTHON_EXECUTABLE=${ASTRAL_PYTHON_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_doc_links.cmake
            )
            set_tests_properties(gate_doc_links PROPERTIES
                TIMEOUT 30
                LABELS "gate"
            )
            message(STATUS "Configured test: gate_doc_links")
        endif()

        if(ASTRAL_PYTHON_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_error_taxonomy.cmake)
            add_test(
                NAME gate_error_taxonomy
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_PYTHON_EXECUTABLE=${ASTRAL_PYTHON_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_error_taxonomy.cmake
            )
            set_tests_properties(gate_error_taxonomy PROPERTIES
                TIMEOUT 30
                LABELS "gate;abi"
            )
            message(STATUS "Configured test: gate_error_taxonomy")
        endif()

        if(ASTRAL_PYTHON_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_evidence.cmake)
            add_test(
                NAME gate_release_evidence
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BUILD_DIR=${CMAKE_BINARY_DIR}
                    -DASTRAL_PYTHON_EXECUTABLE=${ASTRAL_PYTHON_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_evidence.cmake
            )
            set_tests_properties(gate_release_evidence PROPERTIES
                TIMEOUT 30
                LABELS "gate;release"
            )
            message(STATUS "Configured test: gate_release_evidence")
        endif()

        if(ASTRAL_PYTHON_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_unreal_automation_results.cmake)
            add_test(
                NAME gate_unreal_automation_results
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BUILD_DIR=${CMAKE_BINARY_DIR}
                    -DASTRAL_PYTHON_EXECUTABLE=${ASTRAL_PYTHON_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_unreal_automation_results.cmake
            )
            set_tests_properties(gate_unreal_automation_results PROPERTIES
                TIMEOUT 30
                LABELS "gate;unreal"
            )
            message(STATUS "Configured test: gate_unreal_automation_results")
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_unreal_container_runner.cmake)
            add_test(
                NAME gate_unreal_container_runner
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BUILD_DIR=${CMAKE_BINARY_DIR}
                    -DASTRAL_BASH_EXECUTABLE=${ASTRAL_BASH_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_unreal_container_runner.cmake
            )
            set_tests_properties(gate_unreal_container_runner PROPERTIES
                TIMEOUT 30
                LABELS "gate;unreal"
            )
            message(STATUS "Configured test: gate_unreal_container_runner")
        endif()

        if(ASTRAL_PYTHON_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_unity_editmode_results.cmake)
            add_test(
                NAME gate_unity_editmode_results
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BUILD_DIR=${CMAKE_BINARY_DIR}
                    -DASTRAL_PYTHON_EXECUTABLE=${ASTRAL_PYTHON_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_unity_editmode_results.cmake
            )
            set_tests_properties(gate_unity_editmode_results PROPERTIES
                TIMEOUT 30
                LABELS "gate;unity"
            )
            message(STATUS "Configured test: gate_unity_editmode_results")
        endif()

        if(ASTRAL_PYTHON_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_hf_matrix_log.cmake)
            add_test(
                NAME gate_hf_matrix_log
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BUILD_DIR=${CMAKE_BINARY_DIR}
                    -DASTRAL_PYTHON_EXECUTABLE=${ASTRAL_PYTHON_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_hf_matrix_log.cmake
            )
            set_tests_properties(gate_hf_matrix_log PROPERTIES
                TIMEOUT 30
                LABELS "gate;release"
            )
            message(STATUS "Configured test: gate_hf_matrix_log")
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_cuda_parity_runner.cmake)
            add_test(
                NAME gate_cuda_parity_runner
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BASH_EXECUTABLE=${ASTRAL_BASH_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_cuda_parity_runner.cmake
            )
            set_tests_properties(gate_cuda_parity_runner PROPERTIES
                TIMEOUT 30
                LABELS "gate;cuda"
            )
            message(STATUS "Configured test: gate_cuda_parity_runner")
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_required_plan.cmake)
            add_test(
                NAME gate_release_required_plan
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -DASTRAL_BASH_EXECUTABLE=${ASTRAL_BASH_EXECUTABLE}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_required_plan.cmake
            )
            set_tests_properties(gate_release_required_plan PROPERTIES
                TIMEOUT 30
                LABELS "gate;release"
            )
            message(STATUS "Configured test: gate_release_required_plan")
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_sign_workflow.cmake)
            add_test(
                NAME gate_release_sign_workflow
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_release_sign_workflow.cmake
            )
            set_tests_properties(gate_release_sign_workflow PROPERTIES
                TIMEOUT 30
                LABELS "gate;release"
            )
            message(STATUS "Configured test: gate_release_sign_workflow")
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_windows_large_page_runner.cmake)
            add_test(
                NAME gate_windows_large_page_runner
                COMMAND ${CMAKE_COMMAND}
                    -DASTRAL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gate_windows_large_page_runner.cmake
            )
            set_tests_properties(gate_windows_large_page_runner PROPERTIES
                TIMEOUT 30
                LABELS "gate;platform"
            )
            message(STATUS "Configured test: gate_windows_large_page_runner")
        endif()
    endif()
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

# Gate: no tracked heap allocations in steady-state decode/stream.
if(ASTRAL_ENABLE_VIRTUAL_MEMORY AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gate_allocations.cpp)
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

# Gate: no tracked steady-state I/O syscalls in decode/stream.
if(ASTRAL_ENABLE_VIRTUAL_MEMORY AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gate_io_syscalls.cpp)
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
if(ASTRAL_ENABLE_VIRTUAL_MEMORY AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gate_rss_cap.cpp)
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
