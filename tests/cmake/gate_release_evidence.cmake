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
  unreal_sample_package
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
file(WRITE "${evidence_dir}/logs/hf-model-matrix.log" "hf matrix passed\n")
file(WRITE "${evidence_dir}/logs/hf-model-matrix-summary.csv" "backend,model,status\ncpu,model.gguf,pass\n")
file(WRITE "${evidence_dir}/logs/multimodal-validation.log" "mtmd validation passed\n")
file(WRITE "${evidence_dir}/logs/mtmd-features.txt" "features.media feed_image  1.000 Mops/s\nfeatures.media feed_audio  2.000 Mops/s\n")
file(WRITE "${evidence_dir}/logs/comment-review.tsv" "decision\tissue\tnotes\tpath\tline\tkind\tmarker\tbead\ttext\n")
file(WRITE "${evidence_dir}/logs/comment-inventory-summary.log" "comment_inventory files=1 comments=1 doc_lines=0 markers=0 orphan_markers=0\n")
file(WRITE "${evidence_dir}/logs/unreal-57-full-container.log" "[unreal_container] Check image access: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Pull image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Local image digests:\nghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Test filter: AstralRT\n[unreal_container] Linux SDK: /UnrealEngine/Engine/Config/Linux/Linux_SDK.json\nv26 clang 20.1.8\n[unreal_container] clang:\nclang version 20.1.8\n-- Unreal ThirdParty provenance OK: libastral_rt.a\n[unreal_ci] Filter: AstralRT\nLogAutomationCommandLine: Display: Found 13 automation tests based on 'AstralRT'\nLogAutomationCommandLine: Display: \tAstralRT.Module.ShutdownRestart\n[unreal-results] OK: build/unreal-ci-results/unreal-automation.log\n")
file(WRITE "${evidence_dir}/logs/unreal-57-slim-container.log" "[unreal_container] Check image access: ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n[unreal_container] Pull image: ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n[unreal_container] Local image digests:\nghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n[unreal_container] Image: ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n[unreal_container] Test filter: AstralRT\n[unreal_container] Linux SDK: /UnrealEngine/Engine/Config/Linux/Linux_SDK.json\nv26 clang 20.1.8\n[unreal_container] clang:\nclang version 20.1.8\n-- Unreal ThirdParty provenance OK: libastral_rt.a\n[unreal_ci] Filter: AstralRT\nLogAutomationCommandLine: Display: Found 13 automation tests based on 'AstralRT'\nLogAutomationCommandLine: Display: \tAstralRT.Module.ShutdownRestart\n[unreal-results] OK: build/unreal-ci-results/unreal-automation.log\n")
file(WRITE "${evidence_dir}/logs/unreal-compatibility-matrix.log" "[unreal_matrix] UE 5.4: /opt/UE_5.4/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.4/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.4/unreal-automation.log\n[unreal_matrix] UE 5.5: /opt/UE_5.5/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.5/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.5/unreal-automation.log\n[unreal_matrix] UE 5.6: /opt/UE_5.6/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.6/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.6/unreal-automation.log\n[unreal_matrix] UE 5.7: /opt/UE_5.7/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.7/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.7/unreal-automation.log\n")
file(WRITE "${evidence_dir}/logs/unreal-sample-package.log" "[unreal_sample] Project: build/unreal-sample-package/AstralSample/AstralSample.uproject\n[unreal_sample] Archive: build/unreal-sample-package/archive\n[unreal_sample] RunUAT: /opt/UE_5.7/Engine/Build/BatchFiles/RunUAT.sh\n[unreal_sample] Platform: Linux\n[unreal_sample] Plugin mode: copy\n[unreal_sample] BuildCookRun\nRunUAT BuildCookRun -project=build/unreal-sample-package/AstralSample/AstralSample.uproject -platform=Linux -archive\n[unreal_sample] OK: build/unreal-sample-package/archive\n")
file(WRITE "${evidence_dir}/logs/unreal-sample-runtime.log" "Command Line: -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout\nLogPakFile: Display: Mounted IoStore container \"../../../AstralSample/Content/Paks/AstralSample-Linux.utoc\"\nLogPakFile: Display: Mounted Pak file '../../../AstralSample/Content/Paks/AstralSample-Linux.pak'\nLogAstralSample: Astral sample: packaged content bytes read from ../../../AstralSample/Content/AstralSample/Models/mock-model.bytes\nLogAstralSample: Astral sample: packaged content memory model loaded from 4 bytes\nLogAstralSample: Astral sample: saved cache bytes read from ../../../AstralSample/Saved/AstralSample/mock-model-cache.bytes\nLogAstralSample: Astral sample: saved cache memory model loaded from 4 bytes\n")
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
  elseif(lane STREQUAL "native_dev_ctest")
    set(command "cmake --preset dev && cmake --build --preset dev -j && ctest --preset dev -j --output-on-failure")
  elseif(lane STREQUAL "native_release_ctest")
    set(command "cmake --preset release-with-tests && cmake --build --preset release-with-tests -j && ctest --preset release-with-tests -j --output-on-failure")
  elseif(lane STREQUAL "release_required_gates")
    set(command "./scripts/run_release_required_gates.sh --cuda-arch native --cuda-strict --mtmd-bench")
  elseif(lane STREQUAL "sanitizer_validation")
    set(artifacts "[\"logs/asan.log\", \"logs/tsan.log\"]")
    set(command "./scripts/run_asan.sh && ./scripts/run_tsan.sh")
  elseif(lane STREQUAL "comment_review")
    set(artifacts "[\"logs/comment-review.tsv\", \"logs/comment-inventory-summary.log\"]")
    set(command "python3 ./scripts/inventory_comments.py --format review-tsv > logs/comment-review.tsv && python3 ./scripts/inventory_comments.py --format summary --fail-orphan-markers > logs/comment-inventory-summary.log")
  elseif(lane STREQUAL "unreal_57_full_container")
    set(artifacts "[\"logs/unreal-57-full-container.log\"]")
    set(command "cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j && ./scripts/run_unreal_container_ci.sh --variant full --image ghcr.io/epicgames/unreal-engine:dev-5.7.4 --digest sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce --filter AstralRT --skip-native-build")
  elseif(lane STREQUAL "unreal_57_slim_container")
    set(artifacts "[\"logs/unreal-57-slim-container.log\"]")
    set(command "cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j && ./scripts/run_unreal_container_ci.sh --variant slim --image ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4 --digest sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6 --filter AstralRT --skip-native-build")
  elseif(lane STREQUAL "unreal_compatibility_matrix")
    set(artifacts "[\"logs/unreal-compatibility-matrix.log\"]")
    set(command "UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... ./scripts/run_unreal_compatibility_matrix.sh --versions '5.4 5.5 5.6 5.7' --filter AstralRT")
  elseif(lane STREQUAL "unreal_sample_package")
    set(artifacts "[\"logs/unreal-sample-package.log\", \"logs/unreal-sample-runtime.log\"]")
    set(command "UNREAL_RUNUAT=/opt/UE_5.7/Engine/Build/BatchFiles/RunUAT.sh ./scripts/run_unreal_sample_package.sh --platform Linux && build/unreal-sample-package/archive/Linux/AstralSample.sh -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout")
  elseif(lane STREQUAL "unity_editmode_abi")
    set(command "UNITY_EDITOR=... ./scripts/run_unity_ci_tests.sh")
  elseif(lane STREQUAL "cuda_parity_matrix")
    set(command "ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 ./scripts/run_cuda_parity_matrix.sh --preset-set release --arch native --strict")
  elseif(lane STREQUAL "multimodal_validation")
    set(artifacts "[\"logs/multimodal-validation.log\", \"logs/mtmd-features.txt\"]")
    set(command "./scripts/run_multimodal_validation.sh --bench")
  elseif(lane STREQUAL "hf_model_matrix")
    set(artifacts "[\"logs/hf-model-matrix.log\", \"logs/hf-model-matrix-summary.csv\"]")
    set(command "./scripts/run_hf_full_suite.sh --arch native --only all")
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

set(bad_native_manifest "${out_dir}/bad-native-evidence.json")
file(READ "${good_manifest}" bad_native_text)
string(REPLACE
  "cmake --preset release-with-tests && cmake --build --preset release-with-tests -j && ctest --preset release-with-tests -j --output-on-failure"
  "smoke native_release_ctest"
  bad_native_text
  "${bad_native_text}"
)
file(WRITE "${bad_native_manifest}" "${bad_native_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_native_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_native_result
  ERROR_VARIABLE bad_native_error
)
if(bad_native_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak native release evidence command")
endif()
if(NOT bad_native_error MATCHES "native_release_ctest.command")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong native-command reason: ${bad_native_error}")
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
  "ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 ./scripts/run_cuda_parity_matrix.sh --preset-set release --arch native --strict"
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

set(bad_unreal_container_manifest "${out_dir}/bad-unreal-container-evidence.json")
file(READ "${good_manifest}" bad_unreal_container_text)
string(REPLACE
  "cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j && ./scripts/run_unreal_container_ci.sh --variant full --image ghcr.io/epicgames/unreal-engine:dev-5.7.4 --digest sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce --filter AstralRT --skip-native-build"
  "docker run ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce"
  bad_unreal_container_text
  "${bad_unreal_container_text}"
)
file(WRITE "${bad_unreal_container_manifest}" "${bad_unreal_container_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_unreal_container_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_unreal_container_result
  ERROR_VARIABLE bad_unreal_container_error
)
if(bad_unreal_container_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak Unreal container command evidence")
endif()
if(NOT bad_unreal_container_error MATCHES "unreal_57_full_container.command")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong Unreal container command reason: ${bad_unreal_container_error}")
endif()

set(bad_unreal_container_artifact_manifest "${out_dir}/bad-unreal-container-artifact-evidence.json")
file(READ "${good_manifest}" bad_unreal_container_artifact_text)
string(REPLACE
  "[unreal_container] Local image digests:"
  "[unreal_container] skipped digest output:"
  bad_unreal_container_artifact_text
  "${bad_unreal_container_artifact_text}"
)
file(WRITE "${bad_unreal_container_artifact_manifest}" "${bad_unreal_container_artifact_text}")
file(WRITE "${evidence_dir}/logs/unreal-57-full-container.log" "weak container log without local digest evidence\n")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_unreal_container_artifact_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_unreal_container_artifact_result
  ERROR_VARIABLE bad_unreal_container_artifact_error
)
if(bad_unreal_container_artifact_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak Unreal container log evidence")
endif()
if(NOT bad_unreal_container_artifact_error MATCHES "unreal_57_full_container log")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong Unreal container artifact reason: ${bad_unreal_container_artifact_error}")
endif()
file(WRITE "${evidence_dir}/logs/unreal-57-full-container.log" "[unreal_container] Check image access: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Pull image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Local image digests:\nghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Test filter: AstralRT\n[unreal_container] Linux SDK: /UnrealEngine/Engine/Config/Linux/Linux_SDK.json\nv26 clang 20.1.8\n[unreal_container] clang:\nclang version 20.1.8\n-- Unreal ThirdParty provenance OK: libastral_rt.a\n[unreal_ci] Filter: AstralRT\nLogAutomationCommandLine: Display: Found 13 automation tests based on 'AstralRT'\nLogAutomationCommandLine: Display: \tAstralRT.Module.ShutdownRestart\n[unreal-results] OK: build/unreal-ci-results/unreal-automation.log\n")

set(bad_unreal_lifecycle_artifact_manifest "${out_dir}/bad-unreal-lifecycle-artifact-evidence.json")
file(READ "${good_manifest}" bad_unreal_lifecycle_artifact_text)
file(WRITE "${bad_unreal_lifecycle_artifact_manifest}" "${bad_unreal_lifecycle_artifact_text}")
file(WRITE "${evidence_dir}/logs/unreal-57-full-container.log" "[unreal_container] Check image access: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Pull image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Local image digests:\nghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Test filter: AstralRT\n[unreal_container] Linux SDK: /UnrealEngine/Engine/Config/Linux/Linux_SDK.json\nv26 clang 20.1.8\n[unreal_container] clang:\nclang version 20.1.8\n-- Unreal ThirdParty provenance OK: libastral_rt.a\n[unreal_ci] Filter: AstralRT\nLogAutomationCommandLine: Display: Found 13 automation tests based on 'AstralRT'\n[unreal-results] OK: build/unreal-ci-results/unreal-automation.log\n")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_unreal_lifecycle_artifact_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_unreal_lifecycle_artifact_result
  ERROR_VARIABLE bad_unreal_lifecycle_artifact_error
)
if(bad_unreal_lifecycle_artifact_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted Unreal container evidence without lifecycle Automation marker")
endif()
if(NOT bad_unreal_lifecycle_artifact_error MATCHES "AstralRT.Module.ShutdownRestart")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong Unreal lifecycle artifact reason: ${bad_unreal_lifecycle_artifact_error}")
endif()
file(WRITE "${evidence_dir}/logs/unreal-57-full-container.log" "[unreal_container] Check image access: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Pull image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Local image digests:\nghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Test filter: AstralRT\n[unreal_container] Linux SDK: /UnrealEngine/Engine/Config/Linux/Linux_SDK.json\nv26 clang 20.1.8\n[unreal_container] clang:\nclang version 20.1.8\n-- Unreal ThirdParty provenance OK: libastral_rt.a\n[unreal_ci] Filter: AstralRT\nLogAutomationCommandLine: Display: Found 13 automation tests based on 'AstralRT'\nLogAutomationCommandLine: Display: \tAstralRT.Module.ShutdownRestart\n[unreal-results] OK: build/unreal-ci-results/unreal-automation.log\n")

set(bad_unreal_matrix_manifest "${out_dir}/bad-unreal-matrix-evidence.json")
file(READ "${good_manifest}" bad_unreal_matrix_text)
string(REPLACE
  "UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... ./scripts/run_unreal_compatibility_matrix.sh --versions '5.4 5.5 5.6 5.7' --filter AstralRT"
  "UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... ./scripts/run_unreal_compatibility_matrix.sh --versions '5.7' --allow-missing"
  bad_unreal_matrix_text
  "${bad_unreal_matrix_text}"
)
file(WRITE "${bad_unreal_matrix_manifest}" "${bad_unreal_matrix_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_unreal_matrix_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_unreal_matrix_result
  ERROR_VARIABLE bad_unreal_matrix_error
)
if(bad_unreal_matrix_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak Unreal compatibility matrix evidence")
endif()
if(NOT bad_unreal_matrix_error MATCHES "unreal_compatibility_matrix.command")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong Unreal compatibility command reason: ${bad_unreal_matrix_error}")
endif()

set(bad_unreal_matrix_artifact_manifest "${out_dir}/bad-unreal-matrix-artifact-evidence.json")
file(READ "${good_manifest}" bad_unreal_matrix_artifact_text)
file(WRITE "${bad_unreal_matrix_artifact_manifest}" "${bad_unreal_matrix_artifact_text}")
file(WRITE "${evidence_dir}/logs/unreal-compatibility-matrix.log" "[unreal_matrix] UE 5.4: /opt/UE_5.4/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.4/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.4/unreal-automation.log\n[unreal_matrix] Skipping UE 5.5: UNREAL_55_EDITOR is unset\n")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_unreal_matrix_artifact_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_unreal_matrix_artifact_result
  ERROR_VARIABLE bad_unreal_matrix_artifact_error
)
if(bad_unreal_matrix_artifact_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak Unreal compatibility log evidence")
endif()
if(NOT bad_unreal_matrix_artifact_error MATCHES "unreal_compatibility_matrix log")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong Unreal compatibility artifact reason: ${bad_unreal_matrix_artifact_error}")
endif()
file(WRITE "${evidence_dir}/logs/unreal-compatibility-matrix.log" "[unreal_matrix] UE 5.4: /opt/UE_5.4/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.4/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.4/unreal-automation.log\n[unreal_matrix] UE 5.5: /opt/UE_5.5/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.5/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.5/unreal-automation.log\n[unreal_matrix] UE 5.6: /opt/UE_5.6/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.6/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.6/unreal-automation.log\n[unreal_matrix] UE 5.7: /opt/UE_5.7/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.7/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.7/unreal-automation.log\n")

set(bad_unreal_sample_manifest "${out_dir}/bad-unreal-sample-evidence.json")
file(READ "${good_manifest}" bad_unreal_sample_text)
string(REPLACE
  "UNREAL_RUNUAT=/opt/UE_5.7/Engine/Build/BatchFiles/RunUAT.sh ./scripts/run_unreal_sample_package.sh --platform Linux && build/unreal-sample-package/archive/Linux/AstralSample.sh -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout"
  "./scripts/create_unreal_sample_project.sh --out /tmp/AstralSample"
  bad_unreal_sample_text
  "${bad_unreal_sample_text}"
)
file(WRITE "${bad_unreal_sample_manifest}" "${bad_unreal_sample_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_unreal_sample_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_unreal_sample_result
  ERROR_VARIABLE bad_unreal_sample_error
)
if(bad_unreal_sample_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak Unreal sample package command evidence")
endif()
if(NOT bad_unreal_sample_error MATCHES "unreal_sample_package.command")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong Unreal sample command reason: ${bad_unreal_sample_error}")
endif()

set(bad_unreal_sample_artifact_manifest "${out_dir}/bad-unreal-sample-artifact-evidence.json")
file(READ "${good_manifest}" bad_unreal_sample_artifact_text)
file(WRITE "${bad_unreal_sample_artifact_manifest}" "${bad_unreal_sample_artifact_text}")
file(WRITE "${evidence_dir}/logs/unreal-sample-package.log" "[unreal_sample] Project: build/unreal-sample-package/AstralSample/AstralSample.uproject\nMissing Unreal RunUAT path\n")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_unreal_sample_artifact_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_unreal_sample_artifact_result
  ERROR_VARIABLE bad_unreal_sample_artifact_error
)
if(bad_unreal_sample_artifact_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak Unreal sample package log evidence")
endif()
if(NOT bad_unreal_sample_artifact_error MATCHES "unreal_sample_package log")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong Unreal sample artifact reason: ${bad_unreal_sample_artifact_error}")
endif()
file(WRITE "${evidence_dir}/logs/unreal-sample-package.log" "[unreal_sample] Project: build/unreal-sample-package/AstralSample/AstralSample.uproject\n[unreal_sample] Archive: build/unreal-sample-package/archive\n[unreal_sample] RunUAT: /opt/UE_5.7/Engine/Build/BatchFiles/RunUAT.sh\n[unreal_sample] Platform: Linux\n[unreal_sample] Plugin mode: copy\n[unreal_sample] BuildCookRun\nRunUAT BuildCookRun -project=build/unreal-sample-package/AstralSample/AstralSample.uproject -platform=Linux -archive\n[unreal_sample] OK: build/unreal-sample-package/archive\n")

set(bad_unreal_sample_runtime_manifest "${out_dir}/bad-unreal-sample-runtime-evidence.json")
file(READ "${good_manifest}" bad_unreal_sample_runtime_text)
file(WRITE "${bad_unreal_sample_runtime_manifest}" "${bad_unreal_sample_runtime_text}")
file(WRITE "${evidence_dir}/logs/unreal-sample-runtime.log" "Command Line: -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout\nLogPakFile: Display: Mounted IoStore container \"../../../AstralSample/Content/Paks/AstralSample-Linux.utoc\"\nLogPakFile: Display: Mounted Pak file '../../../AstralSample/Content/Paks/AstralSample-Linux.pak'\nLogAstralSample: Astral sample: packaged content model read failed: ../../../AstralSample/Content/AstralSample/Models/mock-model.bytes\n")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_unreal_sample_runtime_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_unreal_sample_runtime_result
  ERROR_VARIABLE bad_unreal_sample_runtime_error
)
if(bad_unreal_sample_runtime_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak Unreal sample runtime log evidence")
endif()
if(NOT bad_unreal_sample_runtime_error MATCHES "unreal_sample_package runtime log")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong Unreal sample runtime reason: ${bad_unreal_sample_runtime_error}")
endif()
file(WRITE "${evidence_dir}/logs/unreal-sample-runtime.log" "Command Line: -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout\nLogPakFile: Display: Mounted IoStore container \"../../../AstralSample/Content/Paks/AstralSample-Linux.utoc\"\nLogPakFile: Display: Mounted Pak file '../../../AstralSample/Content/Paks/AstralSample-Linux.pak'\nLogAstralSample: Astral sample: packaged content bytes read from ../../../AstralSample/Content/AstralSample/Models/mock-model.bytes\nLogAstralSample: Astral sample: packaged content memory model loaded from 4 bytes\nLogAstralSample: Astral sample: saved cache bytes read from ../../../AstralSample/Saved/AstralSample/mock-model-cache.bytes\nLogAstralSample: Astral sample: saved cache memory model loaded from 4 bytes\n")

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

set(bad_sanitizer_artifacts_manifest "${out_dir}/bad-sanitizer-artifacts-evidence.json")
file(READ "${good_manifest}" bad_sanitizer_artifacts_text)
string(REPLACE
  "[\"logs/asan.log\", \"logs/tsan.log\"]"
  "[\"logs/asan.log\"]"
  bad_sanitizer_artifacts_text
  "${bad_sanitizer_artifacts_text}"
)
file(WRITE "${bad_sanitizer_artifacts_manifest}" "${bad_sanitizer_artifacts_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_sanitizer_artifacts_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_sanitizer_artifacts_result
  ERROR_VARIABLE bad_sanitizer_artifacts_error
)
if(bad_sanitizer_artifacts_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted sanitizer evidence without TSan artifact")
endif()
if(NOT bad_sanitizer_artifacts_error MATCHES "sanitizer_validation.artifacts")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong sanitizer-artifacts reason: ${bad_sanitizer_artifacts_error}")
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

set(bad_release_artifacts_manifest "${out_dir}/bad-release-artifacts-evidence.json")
file(READ "${good_manifest}" bad_release_artifacts_text)
string(REPLACE
  "[\"dist/checksums.sha256\", \"dist/abi-layout.json\", \"dist/dependency-manifest.json\", \"dist/release-sbom.spdx.json\"]"
  "[\"dist/checksums.sha256\", \"dist/abi-layout.json\", \"dist/dependency-manifest.json\"]"
  bad_release_artifacts_text
  "${bad_release_artifacts_text}"
)
file(WRITE "${bad_release_artifacts_manifest}" "${bad_release_artifacts_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_release_artifacts_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_release_artifacts_result
  ERROR_VARIABLE bad_release_artifacts_error
)
if(bad_release_artifacts_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted release evidence without SBOM artifact")
endif()
if(NOT bad_release_artifacts_error MATCHES "release_artifacts.artifacts")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong release-artifacts reason: ${bad_release_artifacts_error}")
endif()

set(bad_signing_artifacts_manifest "${out_dir}/bad-signing-artifacts-evidence.json")
file(READ "${good_manifest}" bad_signing_artifacts_text)
string(REPLACE
  "[\"dist/checksums.sha256.asc\"]"
  "[\"logs/release_signing.log\"]"
  bad_signing_artifacts_text
  "${bad_signing_artifacts_text}"
)
file(WRITE "${bad_signing_artifacts_manifest}" "${bad_signing_artifacts_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_signing_artifacts_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_signing_artifacts_result
  ERROR_VARIABLE bad_signing_artifacts_error
)
if(bad_signing_artifacts_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted signing evidence without checksum signature")
endif()
if(NOT bad_signing_artifacts_error MATCHES "release_signing.artifacts")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong signing-artifacts reason: ${bad_signing_artifacts_error}")
endif()

set(bad_hf_command_manifest "${out_dir}/bad-hf-command-evidence.json")
file(READ "${good_manifest}" bad_hf_command_text)
string(REPLACE
  "./scripts/run_hf_full_suite.sh --arch native --only all"
  "./scripts/run_hf_full_suite.sh --only cpu"
  bad_hf_command_text
  "${bad_hf_command_text}"
)
file(WRITE "${bad_hf_command_manifest}" "${bad_hf_command_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_hf_command_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_hf_command_result
  ERROR_VARIABLE bad_hf_command_error
)
if(bad_hf_command_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted weak HF matrix command evidence")
endif()
if(NOT bad_hf_command_error MATCHES "hf_model_matrix.command")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong HF command reason: ${bad_hf_command_error}")
endif()

set(bad_hf_artifacts_manifest "${out_dir}/bad-hf-artifacts-evidence.json")
file(READ "${good_manifest}" bad_hf_artifacts_text)
string(REPLACE
  "[\"logs/hf-model-matrix.log\", \"logs/hf-model-matrix-summary.csv\"]"
  "[\"logs/hf-model-matrix.log\"]"
  bad_hf_artifacts_text
  "${bad_hf_artifacts_text}"
)
file(WRITE "${bad_hf_artifacts_manifest}" "${bad_hf_artifacts_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_hf_artifacts_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_hf_artifacts_result
  ERROR_VARIABLE bad_hf_artifacts_error
)
if(bad_hf_artifacts_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted HF matrix evidence without summary artifact")
endif()
if(NOT bad_hf_artifacts_error MATCHES "hf_model_matrix.artifacts")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong HF artifact reason: ${bad_hf_artifacts_error}")
endif()

set(bad_mtmd_artifacts_manifest "${out_dir}/bad-mtmd-artifacts-evidence.json")
file(READ "${good_manifest}" bad_mtmd_artifacts_text)
string(REPLACE
  "[\"logs/multimodal-validation.log\", \"logs/mtmd-features.txt\"]"
  "[\"logs/multimodal-validation.log\"]"
  bad_mtmd_artifacts_text
  "${bad_mtmd_artifacts_text}"
)
file(WRITE "${bad_mtmd_artifacts_manifest}" "${bad_mtmd_artifacts_text}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${bad_mtmd_artifacts_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_mtmd_artifacts_result
  ERROR_VARIABLE bad_mtmd_artifacts_error
)
if(bad_mtmd_artifacts_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted MTMD evidence without feature bench artifact")
endif()
if(NOT bad_mtmd_artifacts_error MATCHES "multimodal_validation.artifacts")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong MTMD artifact reason: ${bad_mtmd_artifacts_error}")
endif()

file(WRITE "${evidence_dir}/logs/mtmd-features.txt" "features.media feed_image  1.000 Mops/s\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_evidence.py" "${good_manifest}" --base-dir "${evidence_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_mtmd_rows_result
  ERROR_VARIABLE bad_mtmd_rows_error
)
if(bad_mtmd_rows_result EQUAL 0)
  message(FATAL_ERROR "validate_release_evidence.py accepted MTMD bench output missing audio feed row")
endif()
if(NOT bad_mtmd_rows_error MATCHES "features.media feed_audio")
  message(FATAL_ERROR "validate_release_evidence.py failed for the wrong MTMD row reason: ${bad_mtmd_rows_error}")
endif()
file(WRITE "${evidence_dir}/logs/mtmd-features.txt" "features.media feed_image  1.000 Mops/s\nfeatures.media feed_audio  2.000 Mops/s\n")

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
  "native_dev_ctest"
  "cmake --preset dev"
  "ctest --preset dev"
  "native_release_ctest"
  "cmake --preset release-with-tests"
  "ctest --preset release-with-tests"
  "sanitizer_validation"
  "run_asan.sh"
  "run_tsan.sh"
  "comment_review"
  "inventory_comments.py"
  "--format review-tsv"
  "--fail-orphan-markers"
  "unreal_57_full_container"
  "cmake --build --preset unreal-plugin"
  "run_unreal_container_ci.sh --variant full"
  "--skip-native-build"
  "unreal_57_slim_container"
  "run_unreal_container_ci.sh --variant slim"
  "unreal_compatibility_matrix"
  "run_unreal_compatibility_matrix.sh"
  "5.4 5.5 5.6 5.7"
  "--filter AstralRT"
  "unreal_sample_package"
  "run_unreal_sample_package.sh"
  "cuda_parity_matrix"
  "ASTRAL_TEST_CUDA_PARITY_INFER=1"
  "ASTRAL_TEST_CUDA_E2E=1"
  "run_cuda_parity_matrix.sh --preset-set release"
  "--arch"
  "--strict"
  "multimodal_validation"
  "mtmd-features.txt"
  "hf_model_matrix"
  "run_hf_full_suite.sh"
  "--only all"
  "hf-model-matrix-summary.csv"
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
