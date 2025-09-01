add_executable(astral_embedded_cli
    astral_embedded_cli.cpp
)

target_link_libraries(astral_embedded_cli PRIVATE astral_rt)

target_compile_options(astral_embedded_cli PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

add_executable(astral_concurrency_contract
    astral_concurrency_contract.cpp
)

target_link_libraries(astral_concurrency_contract PRIVATE astral_rt)

target_include_directories(astral_concurrency_contract PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(astral_concurrency_contract PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)
