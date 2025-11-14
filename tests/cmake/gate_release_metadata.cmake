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
set(zip_root "${out_dir}/zip-root")
file(REMOVE_RECURSE "${zip_root}")
file(MAKE_DIRECTORY "${zip_root}/plugins/unity")
file(MAKE_DIRECTORY "${zip_root}/plugins/unreal/AstralRT")
file(MAKE_DIRECTORY "${zip_root}/docs/release")
file(WRITE "${zip_root}/plugins/unity/package.json" "{\"name\":\"com.astral.runtime\"}\n")
file(WRITE "${zip_root}/plugins/unreal/AstralRT/AstralRT.uplugin" "{\"FileVersion\":3}\n")
file(WRITE "${zip_root}/LICENSE" "license\n")
file(WRITE "${zip_root}/NOTICE" "notice\n")
file(WRITE "${zip_root}/docs/release/THIRD_PARTY_NOTICES.md" "third-party notices\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar cf "${out_dir}/astral-0.1.0-unity-plugin-linux-x86_64.zip" --format=zip --
    "plugins/unity" "LICENSE" "NOTICE" "docs/release"
  WORKING_DIRECTORY "${zip_root}"
  RESULT_VARIABLE unity_zip_result
  ERROR_VARIABLE unity_zip_error
)
if(NOT unity_zip_result EQUAL 0)
  message(FATAL_ERROR "Unity zip smoke failed: ${unity_zip_error}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar cf "${out_dir}/astral-0.1.0-unreal-plugin-linux-x86_64.zip" --format=zip --
    "plugins/unreal/AstralRT" "LICENSE" "NOTICE" "docs/release"
  WORKING_DIRECTORY "${zip_root}"
  RESULT_VARIABLE unreal_zip_result
  ERROR_VARIABLE unreal_zip_error
)
if(NOT unreal_zip_result EQUAL 0)
  message(FATAL_ERROR "Unreal zip smoke failed: ${unreal_zip_error}")
endif()
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
  file(WRITE "${out_dir}/logs/${lane}.log" "${lane} passed\n")
endforeach()
file(WRITE "${out_dir}/logs/asan.log" "asan passed\n")
file(WRITE "${out_dir}/logs/tsan.log" "tsan passed\n")
file(WRITE "${out_dir}/logs/hf-model-matrix.log" "hf matrix passed\n")
file(WRITE "${out_dir}/logs/hf-model-matrix-summary.csv" "backend,model,status\ncpu,model.gguf,pass\n")
file(WRITE "${out_dir}/logs/multimodal-validation.log" "mtmd validation passed\n")
file(WRITE "${out_dir}/logs/mtmd-features.txt" "features.media feed_image  1.000 Mops/s\nfeatures.media feed_audio  2.000 Mops/s\n")
file(WRITE "${out_dir}/logs/comment-review.tsv" "decision\tissue\tnotes\tpath\tline\tkind\tmarker\tbead\ttext\n")
file(WRITE "${out_dir}/logs/comment-inventory-summary.log" "comment_inventory files=1 comments=1 doc_lines=0 markers=0 orphan_markers=0\n")
file(WRITE "${out_dir}/logs/unreal-57-full-container.log" "[unreal_container] Check image access: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Pull image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Local image digests:\nghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Image: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce\n[unreal_container] Test filter: AstralRT\n[unreal_container] Linux SDK: /UnrealEngine/Engine/Config/Linux/Linux_SDK.json\nv26 clang 20.1.8\n[unreal_container] clang:\nclang version 20.1.8\n-- Unreal ThirdParty provenance OK: libastral_rt.a\n[unreal_ci] Filter: AstralRT\n[unreal-results] OK: build/unreal-ci-results/unreal-automation.log\n")
file(WRITE "${out_dir}/logs/unreal-57-slim-container.log" "[unreal_container] Check image access: ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n[unreal_container] Pull image: ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n[unreal_container] Local image digests:\nghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n[unreal_container] Image: ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6\n[unreal_container] Test filter: AstralRT\n[unreal_container] Linux SDK: /UnrealEngine/Engine/Config/Linux/Linux_SDK.json\nv26 clang 20.1.8\n[unreal_container] clang:\nclang version 20.1.8\n-- Unreal ThirdParty provenance OK: libastral_rt.a\n[unreal_ci] Filter: AstralRT\n[unreal-results] OK: build/unreal-ci-results/unreal-automation.log\n")
file(WRITE "${out_dir}/logs/unreal-compatibility-matrix.log" "[unreal_matrix] UE 5.4: /opt/UE_5.4/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.4/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.4/unreal-automation.log\n[unreal_matrix] UE 5.5: /opt/UE_5.5/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.5/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.5/unreal-automation.log\n[unreal_matrix] UE 5.6: /opt/UE_5.6/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.6/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.6/unreal-automation.log\n[unreal_matrix] UE 5.7: /opt/UE_5.7/Engine/Binaries/Linux/UnrealEditor-Cmd\n[unreal_ci] Filter: AstralRT\n[unreal_ci] Report: build/unreal-ci-results/ue-5.7/automation-report\n[unreal-results] OK: build/unreal-ci-results/ue-5.7/unreal-automation.log\n")
file(WRITE "${out_dir}/logs/unreal-sample-package.log" "[unreal_sample] Project: build/unreal-sample-package/AstralSample/AstralSample.uproject\n[unreal_sample] Archive: build/unreal-sample-package/archive\n[unreal_sample] RunUAT: /opt/UE_5.7/Engine/Build/BatchFiles/RunUAT.sh\n[unreal_sample] Platform: Linux\n[unreal_sample] Plugin mode: copy\n[unreal_sample] BuildCookRun\nRunUAT BuildCookRun -project=build/unreal-sample-package/AstralSample/AstralSample.uproject -platform=Linux -archive\n[unreal_sample] OK: build/unreal-sample-package/archive\n")
file(WRITE "${out_dir}/logs/unreal-sample-runtime.log" "Command Line: -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout\nLogPakFile: Display: Mounted IoStore container \"../../../AstralSample/Content/Paks/AstralSample-Linux.utoc\"\nLogPakFile: Display: Mounted Pak file '../../../AstralSample/Content/Paks/AstralSample-Linux.pak'\nLogAstralSample: Astral sample: packaged content bytes read from ../../../AstralSample/Content/AstralSample/Models/mock-model.bytes\nLogAstralSample: Astral sample: packaged content memory model loaded from 4 bytes\nLogAstralSample: Astral sample: saved cache bytes read from ../../../AstralSample/Saved/AstralSample/mock-model-cache.bytes\nLogAstralSample: Astral sample: saved cache memory model loaded from 4 bytes\n")
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
      \"command\": \"./scripts/run_release_required_gates.sh --cuda-arch native --cuda-strict --mtmd-bench\",
      \"artifacts\": [\"logs/release_required_gates.log\"]
    },
    \"sanitizer_validation\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/run_asan.sh && ./scripts/run_tsan.sh\",
      \"artifacts\": [\"logs/asan.log\", \"logs/tsan.log\"]
    },
    \"comment_review\": {
      \"status\": \"pass\",
      \"command\": \"python3 ./scripts/inventory_comments.py --format review-tsv > logs/comment-review.tsv && python3 ./scripts/inventory_comments.py --format summary --fail-orphan-markers > logs/comment-inventory-summary.log\",
      \"artifacts\": [\"logs/comment-review.tsv\", \"logs/comment-inventory-summary.log\"]
    },
    \"unreal_57_full_container\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/run_unreal_container_ci.sh --variant full --image ghcr.io/epicgames/unreal-engine:dev-5.7.4 --digest sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce --filter AstralRT\",
      \"artifacts\": [\"logs/unreal-57-full-container.log\"]
    },
    \"unreal_57_slim_container\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/run_unreal_container_ci.sh --variant slim --image ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4 --digest sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6 --filter AstralRT\",
      \"artifacts\": [\"logs/unreal-57-slim-container.log\"]
    },
    \"unreal_compatibility_matrix\": {
      \"status\": \"pass\",
      \"command\": \"UNREAL_54_EDITOR=... UNREAL_55_EDITOR=... UNREAL_56_EDITOR=... UNREAL_57_EDITOR=... ./scripts/run_unreal_compatibility_matrix.sh --versions '5.4 5.5 5.6 5.7' --filter AstralRT\",
      \"artifacts\": [\"logs/unreal-compatibility-matrix.log\"]
    },
    \"unreal_sample_package\": {
      \"status\": \"pass\",
      \"command\": \"UNREAL_RUNUAT=/opt/UE_5.7/Engine/Build/BatchFiles/RunUAT.sh ./scripts/run_unreal_sample_package.sh --platform Linux && build/unreal-sample-package/archive/Linux/AstralSample.sh -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout\",
      \"artifacts\": [\"logs/unreal-sample-package.log\", \"logs/unreal-sample-runtime.log\"]
    },
    \"unity_editmode_abi\": {
      \"status\": \"pass\",
      \"command\": \"UNITY_EDITOR=... ./scripts/run_unity_ci_tests.sh\",
      \"artifacts\": [\"logs/unity_editmode_abi.log\"]
    },
    \"cuda_parity_matrix\": {
      \"status\": \"pass\",
      \"command\": \"ASTRAL_TEST_CUDA_PARITY_INFER=1 ASTRAL_TEST_CUDA_E2E=1 ./scripts/run_cuda_parity_matrix.sh --preset-set release --arch native --strict\",
      \"artifacts\": [\"logs/cuda_parity_matrix.log\"]
    },
    \"multimodal_validation\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/run_multimodal_validation.sh --bench\",
      \"artifacts\": [\"logs/multimodal-validation.log\", \"logs/mtmd-features.txt\"]
    },
    \"hf_model_matrix\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/run_hf_full_suite.sh --arch native --only all\",
      \"artifacts\": [\"logs/hf-model-matrix.log\", \"logs/hf-model-matrix-summary.csv\"]
    },
    \"windows_large_pages\": {
      \"status\": \"pass\",
      \"command\": \"pwsh -File ./scripts/run_windows_large_page_validation.ps1 -ExpectFallback; pwsh -File ./scripts/run_windows_large_page_validation.ps1 -ExpectLargePages\",
      \"artifacts\": [\"logs/windows_large_pages.log\"]
    },
    \"release_artifacts\": {
      \"status\": \"pass\",
      \"command\": \"./scripts/validate_release_artifacts.sh --dist dist --expect-unity --expect-unreal --require-signature\",
      \"artifacts\": [\"checksums.sha256\", \"abi-layout.json\", \"dependency-manifest.json\", \"release-sbom.spdx.json\"]
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
  "${out_dir}/release-sbom.spdx.json"
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
if(NOT checksums MATCHES "[ \t]release-sbom\\.spdx\\.json")
  message(FATAL_ERROR "checksums.sha256 does not cover release-sbom.spdx.json")
endif()
if(NOT checksums MATCHES "[ \t]astral-0\\.1\\.0-linux-x86_64\\.zip")
  message(FATAL_ERROR "checksums.sha256 does not cover packaged artifacts")
endif()
if(NOT checksums MATCHES "[ \t]astral-0\\.1\\.0-unity-plugin-linux-x86_64\\.zip")
  message(FATAL_ERROR "checksums.sha256 does not cover Unity plugin artifact")
endif()
if(NOT checksums MATCHES "[ \t]astral-0\\.1\\.0-unreal-plugin-linux-x86_64\\.zip")
  message(FATAL_ERROR "checksums.sha256 does not cover Unreal plugin artifact")
endif()
if(NOT checksums MATCHES "[ \t]release-evidence\\.json")
  message(FATAL_ERROR "checksums.sha256 does not cover release-evidence.json")
endif()

file(READ "${out_dir}/release-sbom.spdx.json" sbom)
if(NOT sbom MATCHES "\"spdxVersion\"[ \t\r\n]*:[ \t\r\n]*\"SPDX-2\\.3\"")
  message(FATAL_ERROR "release-sbom.spdx.json is missing SPDX-2.3 marker")
endif()
if(NOT sbom MATCHES "\"name\"[ \t\r\n]*:[ \t\r\n]*\"AstralRT\"")
  message(FATAL_ERROR "release-sbom.spdx.json does not list AstralRT")
endif()
if(NOT sbom MATCHES "\"name\"[ \t\r\n]*:[ \t\r\n]*\"llama\\.cpp\"")
  message(FATAL_ERROR "release-sbom.spdx.json does not list llama.cpp")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_artifacts.sh" --dist "${out_dir}" --expect-unity --expect-unreal
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE validate_result
  ERROR_VARIABLE validate_error
)
if(NOT validate_result EQUAL 0)
  message(FATAL_ERROR "validate_release_artifacts.sh failed: ${validate_error}")
endif()

set(bad_out_dir "${ASTRAL_BUILD_DIR}/release-metadata-bad-engine-zip-gate")
set(bad_zip_root "${bad_out_dir}/zip-root")
file(REMOVE_RECURSE "${bad_out_dir}")
file(MAKE_DIRECTORY "${bad_out_dir}")
file(MAKE_DIRECTORY "${bad_zip_root}/plugins/unity")
file(MAKE_DIRECTORY "${bad_zip_root}/plugins/unreal/AstralRT")
file(WRITE "${bad_out_dir}/astral-0.1.0-linux-x86_64.zip" "smoke\n")
file(COPY "${out_dir}/abi-layout.json" DESTINATION "${bad_out_dir}")
file(WRITE "${bad_zip_root}/plugins/unity/package.json" "{\"name\":\"com.astral.runtime\"}\n")
file(WRITE "${bad_zip_root}/plugins/unreal/AstralRT/AstralRT.uplugin" "{\"FileVersion\":3}\n")
file(WRITE "${bad_zip_root}/LICENSE" "license\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar cf "${bad_out_dir}/astral-0.1.0-unity-plugin-linux-x86_64.zip" --format=zip --
    "plugins/unity" "LICENSE"
  WORKING_DIRECTORY "${bad_zip_root}"
  RESULT_VARIABLE bad_unity_zip_result
  ERROR_VARIABLE bad_unity_zip_error
)
if(NOT bad_unity_zip_result EQUAL 0)
  message(FATAL_ERROR "Bad Unity zip smoke failed: ${bad_unity_zip_error}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar cf "${bad_out_dir}/astral-0.1.0-unreal-plugin-linux-x86_64.zip" --format=zip --
    "plugins/unreal/AstralRT" "LICENSE"
  WORKING_DIRECTORY "${bad_zip_root}"
  RESULT_VARIABLE bad_unreal_zip_result
  ERROR_VARIABLE bad_unreal_zip_error
)
if(NOT bad_unreal_zip_result EQUAL 0)
  message(FATAL_ERROR "Bad Unreal zip smoke failed: ${bad_unreal_zip_error}")
endif()
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/generate_release_metadata.sh" "${bad_out_dir}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_metadata_result
  ERROR_VARIABLE bad_metadata_error
)
if(NOT bad_metadata_result EQUAL 0)
  message(FATAL_ERROR "Bad zip metadata generation failed: ${bad_metadata_error}")
endif()
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_artifacts.sh" --dist "${bad_out_dir}" --expect-unity --expect-unreal
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_validate_result
  ERROR_VARIABLE bad_validate_error
)
if(bad_validate_result EQUAL 0)
  message(FATAL_ERROR "validate_release_artifacts.sh accepted engine plugin zips missing license/notice payloads")
endif()
if(NOT bad_validate_error MATCHES "missing required payload")
  message(FATAL_ERROR "validate_release_artifacts.sh failed for the wrong engine-zip reason: ${bad_validate_error}")
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
