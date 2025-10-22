if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(script "${ASTRAL_SOURCE_DIR}/scripts/validate_doc_links.py")
if(NOT EXISTS "${script}")
  message(FATAL_ERROR "validate_doc_links.py not found at ${script}")
endif()

set(fixture_root "${CMAKE_CURRENT_BINARY_DIR}/gate_doc_links_fixture")
file(REMOVE_RECURSE "${fixture_root}")
file(MAKE_DIRECTORY "${fixture_root}/docs")
file(WRITE "${fixture_root}/README.md" "[Valid](docs/page.md)\n[External](https://example.com)\n[Anchor](#local)\n")
file(WRITE "${fixture_root}/docs/page.md" "# Page\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}" --root "${fixture_root}"
  RESULT_VARIABLE good_result
  OUTPUT_VARIABLE good_output
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "validate_doc_links.py rejected valid fixture: ${good_output}${good_error}")
endif()

file(WRITE "${fixture_root}/README.md" "[Missing](docs/missing.md)\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}" --root "${fixture_root}"
  RESULT_VARIABLE bad_result
  OUTPUT_VARIABLE bad_output
  ERROR_VARIABLE bad_error
)
if(bad_result EQUAL 0)
  message(FATAL_ERROR "validate_doc_links.py accepted broken fixture: ${bad_output}${bad_error}")
endif()
if(NOT bad_error MATCHES "broken local Markdown link")
  message(FATAL_ERROR "validate_doc_links.py broken fixture did not explain failure: ${bad_output}${bad_error}")
endif()

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}" --root "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE repo_result
  OUTPUT_VARIABLE repo_output
  ERROR_VARIABLE repo_error
)
if(NOT repo_result EQUAL 0)
  message(FATAL_ERROR "Repository Markdown link validation failed: ${repo_output}${repo_error}")
endif()

message(STATUS "gate_doc_links: OK")
