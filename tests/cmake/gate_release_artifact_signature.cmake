if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/release-signature-gate")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")
file(WRITE "${out_dir}/astral-0.1.0-linux-x86_64.zip" "smoke\n")
file(WRITE "${out_dir}/abi-layout.json" "{}\n")
file(WRITE "${out_dir}/dependency-manifest.json" "{}\n")

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/generate_release_metadata.sh" "${out_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE metadata_result
  ERROR_VARIABLE metadata_error
)
if(NOT metadata_result EQUAL 0)
  message(FATAL_ERROR "generate_release_metadata.sh failed: ${metadata_error}")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_artifacts.sh" --dist "${out_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE unsigned_result
  ERROR_VARIABLE unsigned_error
)
if(NOT unsigned_result EQUAL 0)
  message(FATAL_ERROR "validate_release_artifacts.sh rejected unsigned artifacts without --require-signature: ${unsigned_error}")
endif()

file(WRITE "${out_dir}/checksums.sha256.asc" "not a valid detached signature\n")
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_artifacts.sh" --dist "${out_dir}" --require-signature
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_sig_result
  ERROR_VARIABLE bad_sig_error
)
if(bad_sig_result EQUAL 0)
  message(FATAL_ERROR "validate_release_artifacts.sh accepted an invalid checksum signature")
endif()
if(NOT bad_sig_error MATCHES "signature verification failed|gpg is required")
  message(FATAL_ERROR "validate_release_artifacts.sh failed for the wrong signature reason: ${bad_sig_error}")
endif()

file(WRITE "${out_dir}/astral-0.1.0-linux-x86_64.zip" "tampered\n")
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/sign_release_artifacts.sh" --checksums "${out_dir}/checksums.sha256" --tool gpg --dry-run
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_checksum_result
  ERROR_VARIABLE bad_checksum_error
)
if(bad_checksum_result EQUAL 0)
  message(FATAL_ERROR "sign_release_artifacts.sh accepted a mismatched checksum manifest before signing")
endif()
if(NOT bad_checksum_error MATCHES "checksum verification failed")
  message(FATAL_ERROR "sign_release_artifacts.sh failed for the wrong checksum-mismatch reason: ${bad_checksum_error}")
endif()

message(STATUS "gate_release_artifact_signature: OK")
