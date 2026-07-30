if(NOT DEFINED ASTRAL_SOURCE_DIR OR NOT DEFINED ASTRAL_CXX_COMPILER OR
   NOT DEFINED ASTRAL_CXX_COMPILER_ID)
  message(FATAL_ERROR "Queue capacity gate requires source directory and compiler details")
endif()

function(require_capacity_rejection source_name)
  set(source_file "${ASTRAL_SOURCE_DIR}/tests/compile_fail/${source_name}.cpp")
  set(object_file "${CMAKE_CURRENT_BINARY_DIR}/${source_name}.o")

  if(ASTRAL_CXX_COMPILER_ID STREQUAL "MSVC")
    execute_process(
      COMMAND "${ASTRAL_CXX_COMPILER}" /nologo /std:c++17 /c
              "/I${ASTRAL_SOURCE_DIR}" "${source_file}" "/Fo${object_file}"
      RESULT_VARIABLE compile_result
      OUTPUT_VARIABLE compile_stdout
      ERROR_VARIABLE compile_stderr
    )
  else()
    execute_process(
      COMMAND "${ASTRAL_CXX_COMPILER}" -std=c++17 "-I${ASTRAL_SOURCE_DIR}"
              -c "${source_file}" -o "${object_file}"
      RESULT_VARIABLE compile_result
      OUTPUT_VARIABLE compile_stdout
      ERROR_VARIABLE compile_stderr
    )
  endif()

  set(compile_output "${compile_stdout}\n${compile_stderr}")
  if(compile_result EQUAL 0)
    message(FATAL_ERROR "${source_name} accepted Capacity 1")
  endif()

  if(NOT compile_output MATCHES "Capacity must be at least 2")
    message(FATAL_ERROR
      "${source_name} Capacity 1 failed for the wrong reason:\n${compile_output}")
  endif()
endfunction()

require_capacity_rejection(mpmc_capacity_one)
require_capacity_rejection(mpsc_capacity_one)
require_capacity_rejection(mpsc_ticket_capacity_one)
