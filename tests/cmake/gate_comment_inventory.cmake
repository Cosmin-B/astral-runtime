if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(inventory_script "${ASTRAL_SOURCE_DIR}/scripts/inventory_comments.py")
if(NOT EXISTS "${inventory_script}")
  message(FATAL_ERROR "Missing comment inventory script: ${inventory_script}")
endif()

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${inventory_script}"
    --root "${ASTRAL_SOURCE_DIR}"
    --format summary
    --fail-orphan-markers
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE inventory_result
  OUTPUT_VARIABLE inventory_output
  ERROR_VARIABLE inventory_error
)
if(NOT inventory_result EQUAL 0)
  message(FATAL_ERROR "Comment inventory failed: ${inventory_error}")
endif()
if(NOT inventory_output MATCHES "comment_inventory .*orphan_markers=0")
  message(FATAL_ERROR "Comment inventory summary did not prove zero orphan markers: ${inventory_output}")
endif()

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${inventory_script}"
    --root "${ASTRAL_SOURCE_DIR}"
    --format tsv
    --limit 1
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE tsv_result
  OUTPUT_VARIABLE tsv_output
  ERROR_VARIABLE tsv_error
)
if(NOT tsv_result EQUAL 0)
  message(FATAL_ERROR "Comment inventory TSV smoke failed: ${tsv_error}")
endif()
if(NOT tsv_output MATCHES "^path\tline\tkind\tmarker\tbead\ttext")
  message(FATAL_ERROR "Comment inventory TSV header changed: ${tsv_output}")
endif()

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${inventory_script}"
    --root "${ASTRAL_SOURCE_DIR}"
    --format review-tsv
    --limit 1
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE review_tsv_result
  OUTPUT_VARIABLE review_tsv_output
  ERROR_VARIABLE review_tsv_error
)
if(NOT review_tsv_result EQUAL 0)
  message(FATAL_ERROR "Comment inventory review TSV smoke failed: ${review_tsv_error}")
endif()
if(NOT review_tsv_output MATCHES "^decision\tissue\tnotes\tpath\tline\tkind\tmarker\tbead\ttext")
  message(FATAL_ERROR "Comment inventory review TSV header changed: ${review_tsv_output}")
endif()

message(STATUS "gate_comment_inventory: OK")
