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
  sanitizer_validation
  comment_review
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
file(WRITE "${evidence_dir}/logs/asan.log" "asan passed\n")
file(WRITE "${evidence_dir}/logs/tsan.log" "tsan passed\n")
file(WRITE "${evidence_dir}/logs/comment-review.tsv" "decision\tissue\tnotes\tpath\tline\tkind\tmarker\tbead\ttext\n")
file(WRITE "${evidence_dir}/logs/comment-inventory-summary.log" "comment_inventory files=1 comments=1 doc_lines=0 markers=0 orphan_markers=0\n")
file(WRITE "${evidence_dir}/dist/checksums.sha256" "checksums\n")
file(WRITE "${evidence_dir}/dist/abi-layout.json" "{}\n")
file(WRITE "${evidence_dir}/dist/dependency-manifest.json" "{}\n")
file(WRITE "${evidence_dir}/dist/release-sbom.spdx.json" "{}\n")
file(WRITE "${evidence_dir}/dist/checksums.sha256.asc" "signature\n")
file(COPY_FILE
  "${ASTRAL_SOURCE_DIR}/docs/release/RELEASE_NOTES_TEMPLATE.md"
  "${evidence_dir}/docs/release/RELEASE_NOTES_TEMPLATE.md"
)

set(evidence_entries "")
foreach(lane IN LISTS required_lanes)
  set(path "logs/${lane}.log")
  set(artifacts "[\"${path}\"]")
  set(command "smoke ${lane}")
  if(lane STREQUAL "release_artifacts")
    set(path "dist/checksums.sha256")
    set(artifacts "[\"dist/checksums.sha256\", \"dist/abi-layout.json\", \"dist/dependency-manifest.json\", \"dist/release-sbom.spdx.json\"]")
    set(command "./scripts/validate_release_artifacts.sh --dist dist --expect-unity --expect-unreal --require-signature")
  elseif(lane STREQUAL "release_signing")
    set(path "dist/checksums.sha256.asc")
    set(artifacts "[\"${path}\"]")
    set(command "gh workflow run release-sign.yml ...")
  elseif(lane STREQUAL "release_notes")
    set(path "docs/release/RELEASE_NOTES_TEMPLATE.md")
    set(artifacts "[\"${path}\"]")
    set(command "./scripts/validate_release_notes.sh release-notes.md")
  elseif(lane STREQUAL "release_required_gates")
    set(command "./scripts/run_release_required_gates.sh --cuda-strict --mtmd-bench")
  elseif(lane STREQUAL "sanitizer_validation")
    set(artifacts "[\"logs/asan.log\", \"logs/tsan.log\"]")
    set(command "./scripts/run_asan.sh && ./scripts/run_tsan.sh")
  elseif(lane STREQUAL "comment_review")
    set(artifacts "[\"logs/comment-review.tsv\", \"logs/comment-inventory-summary.log\"]")
    set(command "python3 ./scripts/inventory_comments.py --format review-tsv > logs/comment-review.tsv && python3 ./scripts/inventory_comments.py --format summary --fail-orphan-markers > logs/comment-inventory-summary.log")
  elseif(lane STREQUAL "unreal_57_full_container")
    set(command "docker run ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce")
  elseif(lane STREQUAL "unreal_57_slim_container")
    set(command "docker run ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6")
  elseif(lane STREQUAL "unreal_compatibility_matrix")
    set(command "UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... ./scripts/run_unreal_compatibility_matrix.sh")
  elseif(lane STREQUAL "unity_editmode_abi")
    set(command "UNITY_EDITOR=... ./scripts/run_unity_ci_tests.sh")
  elseif(lane STREQUAL "cuda_parity_matrix")
    set(command "ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 ./scripts/run_cuda_parity_matrix.sh --preset-set release --strict")
  elseif(lane STREQUAL "multimodal_validation")
    set(command "./scripts/run_multimodal_validation.sh --bench")
  elseif(lane STREQUAL "hf_model_matrix")
    set(command "./scripts/run_hf_full_suite.sh")
  elseif(lane STREQUAL "windows_large_pages")
    set(command "pwsh -File ./scripts/run_windows_large_page_validation.ps1 -ExpectFallback; pwsh -File ./scripts/run_windows_large_page_validation.ps1 -ExpectLargePages")
  elseif(lane STREQUAL "dependency_pins")
    set(command "./scripts/validate_dependency_pins.sh")
  endif()
  string(APPEND evidence_entries
"    \"${lane}\": {
      \"status\": \"pass\",
      \"command\": \"${command}\",
      \"artifacts\": ${artifacts}
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

set(bad_command_manifest "${out_dir}/bad-command-evidence.json")
file(WRITE "${bad_command_manifest}"
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
file(READ "${bad_command_manifest}" bad_command_text)
string(REPLACE
  "ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 ./scripts/run_cuda_parity_matrix.sh --preset-set release --strict"
  "./scripts/run_cuda_parity_matrix.sh --preset-set release --strict"
  bad_command_text
  "${bad_command_text}"
)
file(WRITE "${bad_command_manifest}" "${bad_command_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_command_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_command_result
  ERROR_VARIABLE bad_command_error
)
if(bad_command_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak CUDA evidence command")
endif()
if(NOT bad_command_error MATCHES "cuda_parity_matrix.command")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong bad-command reason: ${bad_command_error}")
endif()

set(bad_sanitizer_manifest "${out_dir}/bad-sanitizer-evidence.json")
file(READ "${good_manifest}" bad_sanitizer_text)
string(REPLACE
  "./scripts/run_asan.sh && ./scripts/run_tsan.sh"
  "./scripts/run_asan.sh"
  bad_sanitizer_text
  "${bad_sanitizer_text}"
)
file(WRITE "${bad_sanitizer_manifest}" "${bad_sanitizer_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_sanitizer_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_sanitizer_result
  ERROR_VARIABLE bad_sanitizer_error
)
if(bad_sanitizer_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak sanitizer evidence command")
endif()
if(NOT bad_sanitizer_error MATCHES "sanitizer_validation.command")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong sanitizer-command reason: ${bad_sanitizer_error}")
endif()

set(bad_comment_manifest "${out_dir}/bad-comment-review-evidence.json")
file(READ "${good_manifest}" bad_comment_text)
string(REPLACE
  "--format review-tsv"
  "--format tsv"
  bad_comment_text
  "${bad_comment_text}"
)
file(WRITE "${bad_comment_manifest}" "${bad_comment_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_comment_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_comment_result
  ERROR_VARIABLE bad_comment_error
)
if(bad_comment_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak comment-review evidence command")
endif()
if(NOT bad_comment_error MATCHES "comment_review.command")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong comment-review command reason: ${bad_comment_error}")
endif()

file(WRITE "${evidence_dir}/logs/comment-review.tsv" "path\tline\tkind\tmarker\tbead\ttext\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${good_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_comment_header_result
  ERROR_VARIABLE bad_comment_header_error
)
if(bad_comment_header_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted bad comment-review TSV header")
endif()
if(NOT bad_comment_header_error MATCHES "comment-review.tsv header")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong comment-review header reason: ${bad_comment_header_error}")
endif()
file(WRITE "${evidence_dir}/logs/comment-review.tsv" "decision\tissue\tnotes\tpath\tline\tkind\tmarker\tbead\ttext\n")

file(WRITE "${evidence_dir}/logs/comment-inventory-summary.log" "comment_inventory files=1 comments=1 doc_lines=0 markers=1 orphan_markers=1\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${good_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_comment_summary_result
  ERROR_VARIABLE bad_comment_summary_error
)
if(bad_comment_summary_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted comment-review summary with orphan markers")
endif()
if(NOT bad_comment_summary_error MATCHES "orphan_markers=0")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong comment-review summary reason: ${bad_comment_summary_error}")
endif()
file(WRITE "${evidence_dir}/logs/comment-inventory-summary.log" "comment_inventory files=1 comments=1 doc_lines=0 markers=0 orphan_markers=0\n")

set(pre_sign_manifest "${out_dir}/pre-sign-evidence.json")
file(WRITE "${pre_sign_manifest}" "${bad_command_text}")
file(READ "${good_manifest}" pre_sign_text)
string(REGEX REPLACE
  ",\n    \"release_signing\": \\{[^}]*\"artifacts\": \\[[^]]*\\]\n    \\}"
  ""
  pre_sign_text
  "${pre_sign_text}"
)
file(WRITE "${pre_sign_manifest}" "${pre_sign_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${pre_sign_manifest}" --base-dir "${evidence_dir}" --phase pre-sign
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE pre_sign_result
  ERROR_VARIABLE pre_sign_error
)
if(NOT pre_sign_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py rejected valid pre-sign evidence: ${pre_sign_error}")
endif()

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${pre_sign_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE complete_missing_sign_result
  ERROR_VARIABLE complete_missing_sign_error
)
if(complete_missing_sign_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted complete evidence without release_signing")
endif()
if(NOT complete_missing_sign_error MATCHES "release_signing")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong missing-signing reason: ${complete_missing_sign_error}")
endif()

file(READ "${ASTRAL_SOURCE_DIR}/docs/release/RELEASE_EVIDENCE_TEMPLATE.json" template_text)
foreach(required
  "sanitizer_validation"
  "run_asan.sh"
  "run_tsan.sh"
  "comment_review"
  "inventory_comments.py"
  "--format review-tsv"
  "--fail-orphan-markers"
  "cuda_parity_matrix"
  "ASTRAL_TEST_CUDA_PARITY_INFER=1"
  "ASTRAL_TEST_CUDA_E2E=1"
  "run_cuda_parity_matrix.sh --preset-set release --strict"
)
  if(NOT template_text MATCHES "${required}")
    message(FATAL_ERROR "RELEASE_EVIDENCE_TEMPLATE.json is missing release evidence requirement: ${required}")
  endif()
endforeach()

file(READ "${ASTRAL_SOURCE_DIR}/docs/release/RELEASE_ACCEPTANCE_MATRIX.md" acceptance_text)
foreach(required
  "Sanitizer validation"
  "sanitizer_validation"
  "run_asan.sh"
  "run_tsan.sh"
  "ASAN/UBSAN"
  "TSan"
  "Comment review"
  "comment_review"
  "inventory_comments.py"
  "review-tsv"
)
  if(NOT acceptance_text MATCHES "${required}")
    message(FATAL_ERROR "RELEASE_ACCEPTANCE_MATRIX.md is missing release evidence requirement: ${required}")
  endif()
endforeach()

message(STATUS "gate_release_evidence: OK")
