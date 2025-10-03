if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/hf-matrix-log-gate")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")

set(good_log "${out_dir}/hf-good.txt")
file(WRITE "${good_log}"
"# Astral HF bench matrix

## preset=release-with-tests backend=cpu
model=tests/models/hf/model.gguf
embed_model=tests/models/all-MiniLM-L6-v2-Q2_K.gguf
features.embed enqueue+collect  1.000 Mops/s
features.kv state_save  2.000 Mops/s
features.kv state_load  3.000 Mops/s
features.grammar set_gbnf  4.000 Mops/s
features.logprobs meta_drain  5.000 Mops/s
features.kv bytes  ok  42 bytes
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/parse_hf_matrix_log.py"
    --in "${good_log}"
    --out "${out_dir}/good.csv"
    --require-pass
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_result
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "parse_hf_matrix_log.py rejected valid smoke log: ${good_error}")
endif()

set(failed_log "${out_dir}/hf-failed.txt")
file(WRITE "${failed_log}"
"# Astral HF bench matrix

## preset=release-with-tests backend=cpu
model=tests/models/hf/model.gguf
embed_model=tests/models/all-MiniLM-L6-v2-Q2_K.gguf
[bench] FAILED model=tests/models/hf/model.gguf
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/parse_hf_matrix_log.py"
    --in "${failed_log}"
    --out "${out_dir}/failed.csv"
    --require-pass
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE failed_result
  ERROR_VARIABLE failed_error
)
if(failed_result EQUAL 0)
  message(FATAL_ERROR "parse_hf_matrix_log.py accepted failed HF matrix log")
endif()
if(NOT failed_error MATCHES "failed HF matrix")
  message(FATAL_ERROR "parse_hf_matrix_log.py failed for the wrong failed-log reason: ${failed_error}")
endif()

set(empty_log "${out_dir}/hf-empty.txt")
file(WRITE "${empty_log}" "# empty\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/parse_hf_matrix_log.py"
    --in "${empty_log}"
    --out "${out_dir}/empty.csv"
    --require-pass
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE empty_result
  ERROR_VARIABLE empty_error
)
if(empty_result EQUAL 0)
  message(FATAL_ERROR "parse_hf_matrix_log.py accepted empty HF matrix log")
endif()
if(NOT empty_error MATCHES "no blocks found")
  message(FATAL_ERROR "parse_hf_matrix_log.py failed for the wrong empty-log reason: ${empty_error}")
endif()

message(STATUS "gate_hf_matrix_log: OK")
