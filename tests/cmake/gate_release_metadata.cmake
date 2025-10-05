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
if(NOT DEFINED ASTRAL_CXX_COMPILER)
  message(FATAL_ERROR "ASTRAL_CXX_COMPILER not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/release-metadata-gate")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}/logs")
file(WRITE "${out_dir}/astral-0.1.0-linux-x86_64.zip" "smoke\n")
set(required_lanes
  native_dev_ctest
  native_release_ctest
  release_required_gates
  unreal_57_full_container
  unreal_57_slim_container
  unreal_compatibility_matrix
  unity_editmode_abi
  cuda_parity_matrix
  multimodal_validation
  hf_model_matrix
  windows_large_pages
  release_artifacts
  release_signing
  dependency_pins
  release_notes
)
foreach(lane IN LISTS required_lanes)
  file(WRITE "${out_dir}/logs/${lane}.log" "${lane} passed\n")
endforeach()
file(WRITE "${out_dir}/checksums.sha256.asc" "signature\n")
file(WRITE "${out_dir}/release-notes.md" "release notes\n")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CXX=${ASTRAL_CXX_COMPILER}"
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/generate_abi_layout_report.sh"
      --out "${out_dir}/abi-layout.json"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE abi_result
  ERROR_VARIABLE abi_error
)
if(NOT abi_result EQUAL 0)
  message(FATAL_ERROR "generate_abi_layout_report.sh failed: ${abi_error}")
endif()

file(WRITE "${out_dir}/release-evidence.json"
"{
  \"schema\": \"astral.release.evidence.v1\",
  \"release\": {
    \"version\": \"0.1.0\",
    \"git_commit\": \"gate-smoke\"
  },
  \"evidence\": {
    \"native_dev_ctest\": {
      \"status\": \"pass\",
      \"command\": \"cmake --preset dev && cmake --build --preset dev -j && ctest --preset dev -j --output-on-failure\",
      \"artifacts\": [\"logs/native_dev_ctest.log\"]
    },
    \"native_release_ctest\": {
      \"status\": \"pass\",
      \"command\": \"cmake --preset release-with-tests && cmake --build --preset release-with-tests -j && ctest --preset release-with-tests -j --output-on-failure\",
      \"artifacts\": [\"logs/native_release_ctest.log\"]
    },
    \"release_required_gates\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/run_release_required_gates.sh --cuda-strict --mtmd-bench\",
      \"artifacts\": [\"logs/release_required_gates.log\"]
    },
    \"unreal_57_full_container\": {
      \"status\": \"pass\",
      \"command\": \"docker run ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\",
      \"artifacts\": [\"logs/unreal_57_full_container.log\"]
    },
    \"unreal_57_slim_container\": {
      \"status\": \"pass\",
      \"command\": \"docker run ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\",
      \"artifacts\": [\"logs/unreal_57_slim_container.log\"]
    },
    \"unreal_compatibility_matrix\": {
      \"status\": \"pass\",
      \"command\": \"UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... ./scripts/run_unreal_compatibility_matrix.sh\",
      \"artifacts\": [\"logs/unreal_compatibility_matrix.log\"]
    },
    \"unity_editmode_abi\": {
      \"status\": \"pass\",
      \"command\": \"UNITY_EDITOR=... ./scripts/run_unity_ci_tests.sh\",
      \"artifacts\": [\"logs/unity_editmode_abi.log\"]
    },
    \"cuda_parity_matrix\": {
      \"status\": \"pass\",
      \"command\": \"ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 ./scripts/run_cuda_parity_matrix.sh --preset-set release --strict\",
      \"artifacts\": [\"logs/cuda_parity_matrix.log\"]
    },
    \"multimodal_validation\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/run_multimodal_validation.sh --bench\",
      \"artifacts\": [\"logs/multimodal_validation.log\"]
    },
    \"hf_model_matrix\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/run_hf_full_suite.sh\",
      \"artifacts\": [\"logs/hf_model_matrix.log\"]
    },
    \"windows_large_pages\": {
      \"status\": \"pass\",
      \"command\": \"pwsh -File ./scripts/run_windows_large_page_validation.ps1 -ExpectFallback; pwsh -File ./scripts/run_windows_large_page_validation.ps1 -ExpectLargePages\",
      \"artifacts\": [\"logs/windows_large_pages.log\"]
    },
    \"release_artifacts\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/validate_release_artifacts.sh --dist dist --expect-unity --expect-unreal --require-signature\",
      \"artifacts\": [\"checksums.sha256\", \"abi-layout.json\", \"dependency-manifest.json\"]
    },
    \"release_signing\": {
      \"status\": \"pass\",
      \"command\": \"gh workflow run release-sign.yml ...\",
      \"artifacts\": [\"checksums.sha256.asc\"]
    },
    \"dependency_pins\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/validate_dependency_pins.sh\",
      \"artifacts\": [\"logs/dependency_pins.log\"]
    },
    \"release_notes\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/validate_release_notes.sh release-notes.md\",
      \"artifacts\": [\"release-notes.md\"]
    }
  }
}
")

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/generate_release_metadata.sh" "${out_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE metadata_result
  ERROR_VARIABLE metadata_error
)
if(NOT metadata_result EQUAL 0)
  message(FATAL_ERROR "generate_release_metadata.sh failed: ${metadata_error}")
endif()

set(required_files
  "${out_dir}/abi-layout.json"
  "${out_dir}/dependency-manifest.json"
  "${out_dir}/checksums.sha256"
)
foreach(path IN LISTS required_files)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Release metadata gate expected file is missing: ${path}")
  endif()
endforeach()

file(READ "${out_dir}/checksums.sha256" checksums)
if(NOT checksums MATCHES "[ \t]abi-layout\\.json")
  message(FATAL_ERROR "checksums.sha256 does not cover abi-layout.json")
endif()
if(NOT checksums MATCHES "[ \t]dependency-manifest\\.json")
  message(FATAL_ERROR "checksums.sha256 does not cover dependency-manifest.json")
endif()
if(NOT checksums MATCHES "[ \t]astral-0\\.1\\.0-linux-x86_64\\.zip")
  message(FATAL_ERROR "checksums.sha256 does not cover packaged artifacts")
endif()
if(NOT checksums MATCHES "[ \t]release-evidence\\.json")
  message(FATAL_ERROR "checksums.sha256 does not cover release-evidence.json")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_artifacts.sh" --dist "${out_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE validate_result
  ERROR_VARIABLE validate_error
)
if(NOT validate_result EQUAL 0)
  message(FATAL_ERROR "validate_release_artifacts.sh failed: ${validate_error}")
endif()

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${out_dir}/release-evidence.json" --base-dir "${out_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE evidence_result
  ERROR_VARIABLE evidence_error
)
if(NOT evidence_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py failed: ${evidence_error}")
endif()

message(STATUS "gate_release_metadata: OK")
