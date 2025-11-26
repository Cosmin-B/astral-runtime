add_subdirectory(embedded)

add_executable(astral_c_quickstart
    astral_c_quickstart.c
)

target_link_libraries(astral_c_quickstart PRIVATE astral_rt)

set_target_properties(astral_c_quickstart PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX
)

target_include_directories(astral_c_quickstart PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(astral_c_quickstart PRIVATE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall -Wextra -Werror -Wno-unused-parameter>
    $<$<COMPILE_LANG_AND_ID:C,MSVC>:/W4 /WX>
)
