if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
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

set(missing_feature_log "${out_dir}/hf-missing-feature.txt")
file(WRITE "${missing_feature_log}"
"# Astral HF bench matrix

## preset=release-with-tests backend=cpu
model=tests/models/hf/model.gguf
embed_model=tests/models/all-MiniLM-L6-v2-Q2_K.gguf
features.embed enqueue+collect  1.000 Mops/s
features.kv state_save  2.000 Mops/s
features.kv state_load  3.000 Mops/s
features.grammar set_gbnf  4.000 Mops/s
features.kv bytes  ok  42 bytes
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/parse_hf_matrix_log.py"
    --in "${missing_feature_log}"
    --out "${out_dir}/missing-feature.csv"
    --require-pass
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_feature_result
  ERROR_VARIABLE missing_feature_error
)
if(missing_feature_result EQUAL 0)
  message(FATAL_ERROR "parse_hf_matrix_log.py accepted an HF matrix log missing feature rows")
endif()
if(NOT missing_feature_error MATCHES "incomplete HF matrix")
  message(FATAL_ERROR "parse_hf_matrix_log.py failed for the wrong missing-feature reason: ${missing_feature_error}")
endif()

set(skip_only_log "${out_dir}/hf-skip-only.txt")
file(WRITE "${skip_only_log}"
"# Astral HF bench matrix

## preset=release-with-tests backend=cpu
model=tests/models/hf/mmproj-model.gguf
embed_model=(skipped: aux gguf)
[bench] SKIPPED aux_gguf=1
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/parse_hf_matrix_log.py"
    --in "${skip_only_log}"
    --out "${out_dir}/skip-only.csv"
    --require-pass
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE skip_only_result
  ERROR_VARIABLE skip_only_error
)
if(skip_only_result EQUAL 0)
  message(FATAL_ERROR "parse_hf_matrix_log.py accepted skip-only HF matrix evidence")
endif()
if(NOT skip_only_error MATCHES "no non-skipped")
  message(FATAL_ERROR "parse_hf_matrix_log.py failed for the wrong skip-only reason: ${skip_only_error}")
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

set(watchdog_dir "${out_dir}/watchdog")
set(watchdog_hf "${watchdog_dir}/hf")
set(watchdog_lfm25 "${watchdog_dir}/hf-lfm25")
set(watchdog_logs "${watchdog_dir}/logs")
file(MAKE_DIRECTORY "${watchdog_hf}" "${watchdog_lfm25}" "${watchdog_logs}")
file(WRITE "${watchdog_hf}/model.gguf.part" "partial\n")
file(WRITE "${watchdog_lfm25}/model.gguf.part" "partial\n")

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/hetzner_watchdog.sh"
    --dry-run
    --log-dir "${watchdog_logs}"
    --hf-models-dir "${watchdog_hf}"
    --lfm25-models-dir "${watchdog_lfm25}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE watchdog_result
  OUTPUT_VARIABLE watchdog_output
  ERROR_VARIABLE watchdog_error
)
if(NOT watchdog_result EQUAL 0)
  message(FATAL_ERROR "hetzner_watchdog.sh --dry-run failed: ${watchdog_error}")
endif()
foreach(required_text
  "./scripts/hf_gguf_download_manifest.sh --out ${watchdog_hf}"
  "./scripts/hf_gguf_download_lfm25_all.sh --out ${watchdog_lfm25}"
  "./scripts/run_hf_wait_and_bench.sh --models-dir ${watchdog_hf}"
  "./scripts/run_hf_wait_and_bench.sh --models-dir ${watchdog_lfm25}"
)
  string(FIND "${watchdog_output}" "${required_text}" required_pos)
  if(required_pos LESS 0)
    message(FATAL_ERROR "hetzner_watchdog.sh --dry-run missed command text '${required_text}': ${watchdog_output}")
  endif()
endforeach()

foreach(script_name run_hf_bench_matrix.sh run_hf_full_suite.sh run_hf_wait_and_bench.sh)
  execute_process(
    COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/${script_name}" --only invalid-mode
    WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
    RESULT_VARIABLE invalid_only_result
    ERROR_VARIABLE invalid_only_error
  )
  if(invalid_only_result EQUAL 0)
    message(FATAL_ERROR "${script_name} accepted an invalid --only mode")
  endif()
  if(NOT invalid_only_error MATCHES "unsupported --only mode")
    message(FATAL_ERROR "${script_name} failed for the wrong invalid --only reason: ${invalid_only_error}")
  endif()
endforeach()

message(STATUS "gate_hf_matrix_log: OK")
