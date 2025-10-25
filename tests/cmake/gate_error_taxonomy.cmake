if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(script "${ASTRAL_SOURCE_DIR}/scripts/validate_error_taxonomy.py")
if(NOT EXISTS "${script}")
  message(FATAL_ERROR "Missing error taxonomy validator: ${script}")
endif()

set(header "${ASTRAL_SOURCE_DIR}/include/astral_rt.h")
set(doc "${ASTRAL_SOURCE_DIR}/docs/api/ERROR_HANDLING.md")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}" --header "${header}" --doc "${doc}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE taxonomy_result
  OUTPUT_VARIABLE taxonomy_output
  ERROR_VARIABLE taxonomy_error
)
if(NOT taxonomy_result EQUAL 0)
  message(FATAL_ERROR "Error taxonomy validation failed: ${taxonomy_output}${taxonomy_error}")
endif()

set(fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/gate_error_taxonomy_fixture")
file(REMOVE_RECURSE "${fixture_dir}")
file(MAKE_DIRECTORY "${fixture_dir}")
file(COPY "${header}" DESTINATION "${fixture_dir}")
file(WRITE "${fixture_dir}/ERROR_HANDLING.md" "# Error Handling\n\n`ASTRAL_OK`\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}"
    --header "${fixture_dir}/astral_rt.h"
    --doc "${fixture_dir}/ERROR_HANDLING.md"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_result
  OUTPUT_VARIABLE bad_output
  ERROR_VARIABLE bad_error
)
if(bad_result EQUAL 0)
  message(FATAL_ERROR "Error taxonomy validator accepted incomplete fixture: ${bad_output}${bad_error}")
endif()
if(NOT bad_error MATCHES "missing documented codes")
  message(FATAL_ERROR "Error taxonomy validator failed for the wrong fixture reason: ${bad_output}${bad_error}")
endif()

message(STATUS "gate_error_taxonomy: OK")
