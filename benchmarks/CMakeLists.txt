# Astral Runtime Benchmarks
# Microbenchmarks and end-to-end performance tests

# Collect all benchmark source files
file(GLOB_RECURSE BENCHMARK_SOURCES
    CONFIGURE_DEPENDS
    "*.cpp"
)

# Only create targets if benchmark files exist
if(BENCHMARK_SOURCES)
    # Benchmark executable (requires main.cpp)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
        add_executable(astral_benchmarks ${BENCHMARK_SOURCES})

        target_link_libraries(astral_benchmarks PRIVATE astral_rt)

        # Benchmarks often need access to internal headers (concurrency primitives, etc).
        target_include_directories(astral_benchmarks PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}
        )

        set_target_properties(astral_benchmarks PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )

        if(UNIX AND NOT APPLE)
            target_link_libraries(astral_benchmarks PRIVATE pthread)
        endif()
    else()
        message(STATUS "No main.cpp found in benchmarks/ directory - skipping benchmark executable")
    endif()
else()
    message(STATUS "No benchmark files found in benchmarks/ directory - skipping benchmark targets")
endif()
