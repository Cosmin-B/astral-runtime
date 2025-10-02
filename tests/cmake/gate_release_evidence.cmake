if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/release-evidence-gate")
set(evidence_dir "${out_dir}/evidence")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${evidence_dir}/logs")
file(MAKE_DIRECTORY "${evidence_dir}/dist")
file(MAKE_DIRECTORY "${evidence_dir}/docs/release")

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
  file(WRITE "${evidence_dir}/logs/${lane}.log" "${lane} passed\n")
endforeach()
file(WRITE "${evidence_dir}/dist/checksums.sha256" "checksums\n")
file(WRITE "${evidence_dir}/dist/abi-layout.json" "{}\n")
file(WRITE "${evidence_dir}/dist/dependency-manifest.json" "{}\n")
file(WRITE "${evidence_dir}/dist/checksums.sha256.asc" "signature\n")
file(COPY_FILE
  "${ASTRAL_SOURCE_DIR}/docs/release/RELEASE_NOTES_TEMPLATE.md"
  "${evidence_dir}/docs/release/RELEASE_NOTES_TEMPLATE.md"
)

set(evidence_entries "")
foreach(lane IN LISTS required_lanes)
  set(path "logs/${lane}.log")
  if(lane STREQUAL "release_artifacts")
    set(path "dist/checksums.sha256")
  elseif(lane STREQUAL "release_signing")
    set(path "dist/checksums.sha256.asc")
  elseif(lane STREQUAL "release_notes")
    set(path "docs/release/RELEASE_NOTES_TEMPLATE.md")
  endif()
  string(APPEND evidence_entries
"    \"${lane}\": {
      \"status\": \"pass\",
      \"command\": \"smoke ${lane}\",
      \"artifacts\": [\"${path}\"]
    }")
  if(NOT lane STREQUAL "release_notes")
    string(APPEND evidence_entries ",\n")
  else()
    string(APPEND evidence_entries "\n")
  endif()
endforeach()

set(good_manifest "${out_dir}/release-evidence.json")
file(WRITE "${good_manifest}"
"{
  \"schema\": \"astral.release.evidence.v1\",
  \"release\": {
    \"version\": \"0.1.0\",
    \"git_commit\": \"gate-smoke\"
  },
  \"evidence\": {
${evidence_entries}  }
}
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${good_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_result
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py rejected valid smoke manifest: ${good_error}")
endif()

set(bad_manifest "${out_dir}/missing-evidence.json")
file(WRITE "${bad_manifest}"
"{
  \"schema\": \"astral.release.evidence.v1\",
  \"release\": {
    \"version\": \"0.1.0\",
    \"git_commit\": \"gate-smoke\"
  },
  \"evidence\": {
    \"native_dev_ctest\": {
      \"status\": \"pass\",
      \"command\": \"smoke native_dev_ctest\",
      \"artifacts\": [\"logs/native_dev_ctest.log\"]
    }
  }
}
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_result
  ERROR_VARIABLE bad_error
)
if(bad_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted a manifest missing required lanes")
endif()
if(NOT bad_error MATCHES "missing required lane")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong reason: ${bad_error}")
endif()

message(STATUS "gate_release_evidence: OK")
