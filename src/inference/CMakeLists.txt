# Inference layer - Model management, session handling, sampling

add_library(astral_inference STATIC
    model.cpp
    model.hpp
    session.cpp
    session.hpp
    sampler.cpp
    sampler.hpp
)

target_link_libraries(astral_inference
    PRIVATE
        astral_backend
        astral_memory
        astral_concurrency
        astral_platform
)

target_include_directories(astral_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# C++20 required
target_compile_features(astral_inference PUBLIC cxx_std_20)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(astral_inference PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
    )
elseif(MSVC)
    target_compile_options(astral_inference PRIVATE
        /W4
        /wd4100  # Unused parameter
    )
endif()
