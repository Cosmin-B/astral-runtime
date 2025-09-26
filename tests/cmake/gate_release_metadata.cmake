if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()
if(NOT DEFINED ASTRAL_CXX_COMPILER)
  message(FATAL_ERROR "ASTRAL_CXX_COMPILER not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/release-metadata-gate")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")
file(WRITE "${out_dir}/astral-smoke.zip" "smoke\n")

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
if(NOT checksums MATCHES "[ \t]astral-smoke\\.zip")
  message(FATAL_ERROR "checksums.sha256 does not cover packaged artifacts")
endif()

message(STATUS "gate_release_metadata: OK")
