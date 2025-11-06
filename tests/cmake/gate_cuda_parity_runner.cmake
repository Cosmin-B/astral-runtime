if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_cuda_parity.sh" --preset dev-cuda --check-env
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_result
  ERROR_VARIABLE missing_error
)
if(missing_result EQUAL 0)
  message(FATAL_ERROR "run_cuda_parity.sh accepted missing real-CUDA env flags")
endif()
if(NOT missing_error MATCHES "ASTRAL_TEST_CUDA_PARITY_INFER")
  message(FATAL_ERROR "run_cuda_parity.sh failed for the wrong missing-env reason: ${missing_error}")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_cuda_parity_matrix.sh" --preset-set release --check-env
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE matrix_missing_result
  ERROR_VARIABLE matrix_missing_error
)
if(matrix_missing_result EQUAL 0)
  message(FATAL_ERROR "run_cuda_parity_matrix.sh accepted missing real-CUDA env flags")
endif()
if(NOT matrix_missing_error MATCHES "ASTRAL_TEST_CUDA_PARITY_INFER")
  message(FATAL_ERROR "run_cuda_parity_matrix.sh failed for the wrong missing-env reason: ${matrix_missing_error}")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_cuda_parity_matrix.sh" --preset-set release --check-env --allow-probes
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE probe_result
  OUTPUT_VARIABLE probe_output
  ERROR_VARIABLE probe_error
)
if(NOT probe_result EQUAL 0)
  message(FATAL_ERROR "run_cuda_parity_matrix.sh --allow-probes --check-env failed: ${probe_error}")
endif()
if(NOT probe_output MATCHES "probe-only env policy OK")
  message(FATAL_ERROR "run_cuda_parity_matrix.sh --allow-probes --check-env did not report probe policy: ${probe_output}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_CUDA_PARITY_INFER=1"
    "ASTRAL_TEST_CUDA_E2E=1"
    "ASTRAL_CUDA_NVIDIA_SMI=${ASTRAL_SOURCE_DIR}/build/missing-nvidia-smi"
    "ASTRAL_CUDA_NVCC=${ASTRAL_SOURCE_DIR}/build/missing-nvcc"
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_cuda_parity.sh" --preset dev-cuda --check-runner
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_runner_result
  ERROR_VARIABLE missing_runner_error
)
if(missing_runner_result EQUAL 0)
  message(FATAL_ERROR "run_cuda_parity.sh accepted a real-CUDA runner without nvidia-smi")
endif()
if(NOT missing_runner_error MATCHES "missing nvidia-smi")
  message(FATAL_ERROR "run_cuda_parity.sh failed for the wrong missing-runner reason: ${missing_runner_error}")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_cuda_parity_matrix.sh" --preset-set release --check-runner --allow-probes
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE probe_runner_result
  OUTPUT_VARIABLE probe_runner_output
  ERROR_VARIABLE probe_runner_error
)
if(NOT probe_runner_result EQUAL 0)
  message(FATAL_ERROR "run_cuda_parity_matrix.sh --allow-probes --check-runner failed: ${probe_runner_error}")
endif()
if(NOT probe_runner_output MATCHES "probe-only runner policy OK")
  message(FATAL_ERROR "run_cuda_parity_matrix.sh --allow-probes --check-runner did not report probe runner policy: ${probe_runner_output}")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_cuda_parity.sh" --help
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE help_result
  OUTPUT_VARIABLE help_output
)
if(NOT help_result EQUAL 0)
  message(FATAL_ERROR "run_cuda_parity.sh --help failed")
endif()
if(NOT help_output MATCHES "--allow-probes")
  message(FATAL_ERROR "run_cuda_parity.sh --help does not document --allow-probes")
endif()
if(NOT help_output MATCHES "--check-runner")
  message(FATAL_ERROR "run_cuda_parity.sh --help does not document --check-runner")
endif()

message(STATUS "gate_cuda_parity_runner: OK")
