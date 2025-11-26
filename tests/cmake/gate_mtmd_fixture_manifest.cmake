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

set(script "${ASTRAL_SOURCE_DIR}/scripts/validate_mtmd_fixture_manifest.py")
set(resolver "${ASTRAL_SOURCE_DIR}/scripts/resolve_mtmd_fixtures.py")
set(runner "${ASTRAL_SOURCE_DIR}/scripts/run_multimodal_validation.sh")
set(manifest "${ASTRAL_SOURCE_DIR}/scripts/mtmd_fixture_manifest_lfm25.json")
set(out_dir "${ASTRAL_BUILD_DIR}/mtmd-fixture-manifest-gate")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}" "${manifest}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_result
  OUTPUT_VARIABLE good_output
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture manifest validator rejected the committed manifest: ${good_output}${good_error}")
endif()

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${resolver}"
    --manifest "${manifest}"
    --fixture-dir "${out_dir}/fixtures"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE resolver_result
  OUTPUT_VARIABLE resolver_output
  ERROR_VARIABLE resolver_error
)
if(NOT resolver_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture resolver rejected the committed manifest: ${resolver_output}${resolver_error}")
endif()
foreach(required_resolved
    "vision_model"
    "vision_media"
    "audio_model"
    "audio_media")
  if(NOT resolver_output MATCHES "${required_resolved}")
    message(FATAL_ERROR "MTMD fixture resolver did not print ${required_resolved}: ${resolver_output}")
  endif()
endforeach()

set(unpinned_manifest "${out_dir}/unpinned.json")
file(WRITE "${unpinned_manifest}"
"{
  \"schema\": \"astral.mtmd-fixtures.v1\",
  \"license_id\": \"lfm1.0\",
  \"license_name\": \"LFM Open License v1.0\",
  \"license_url\": \"https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base-GGUF/blob/ff14effbc3d8eb8823f53a5c9b0ebf7419b57c68/LICENSE\",
  \"repos\": [
    {
      \"role\": \"vision\",
      \"repo\": \"LiquidAI/LFM2.5-VL-1.6B-GGUF\",
      \"revision\": \"main\",
      \"mode\": \"all\",
      \"max_gb_per_file\": 3,
      \"include\": [\"^LFM2\\\\.5-VL-1\\\\.6B-Q4_0\\\\.gguf$\", \"^mmproj-LFM2\\\\.5-VL-1\\\\.6b-Q8_0\\\\.gguf$\"],
      \"required_files\": {
        \"model\": \"LFM2.5-VL-1.6B-Q4_0.gguf\",
        \"projector\": \"mmproj-LFM2.5-VL-1.6b-Q8_0.gguf\"
      }
    },
    {
      \"role\": \"audio\",
      \"repo\": \"LiquidAI/LFM2.5-Audio-1.5B-GGUF\",
      \"revision\": \"7d525f883a077e20afb782f2ff618edcae0e39e4\",
      \"mode\": \"all\",
      \"max_gb_per_file\": 3,
      \"include\": [\"^LFM2\\\\.5-Audio-1\\\\.5B-Q4_0\\\\.gguf$\", \"^mmproj-LFM2\\\\.5-Audio-1\\\\.5B-Q4_0\\\\.gguf$\", \"^vocoder-LFM2\\\\.5-Audio-1\\\\.5B-Q4_0\\\\.gguf$\", \"^tokenizer-LFM2\\\\.5-Audio-1\\\\.5B-Q4_0\\\\.gguf$\"],
      \"required_files\": {
        \"model\": \"LFM2.5-Audio-1.5B-Q4_0.gguf\",
        \"projector\": \"mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf\",
        \"vocoder\": \"vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf\",
        \"tokenizer\": \"tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf\"
      }
    }
  ]
}
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}" "${unpinned_manifest}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE unpinned_result
  ERROR_VARIABLE unpinned_error
)
if(unpinned_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture manifest validator accepted an unpinned revision")
endif()
if(NOT unpinned_error MATCHES "40-character commit SHA")
  message(FATAL_ERROR "MTMD fixture manifest validator failed for the wrong unpinned reason: ${unpinned_error}")
endif()

set(missing_audio_manifest "${out_dir}/missing-audio.json")
file(WRITE "${missing_audio_manifest}"
"{
  \"schema\": \"astral.mtmd-fixtures.v1\",
  \"license_id\": \"lfm1.0\",
  \"license_name\": \"LFM Open License v1.0\",
  \"license_url\": \"https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base-GGUF/blob/ff14effbc3d8eb8823f53a5c9b0ebf7419b57c68/LICENSE\",
  \"repos\": [
    {
      \"role\": \"vision\",
      \"repo\": \"LiquidAI/LFM2.5-VL-1.6B-GGUF\",
      \"revision\": \"48c6a306939241d1ddc99b090df552cb47a066c6\",
      \"mode\": \"all\",
      \"max_gb_per_file\": 3,
      \"include\": [\"^LFM2\\\\.5-VL-1\\\\.6B-Q4_0\\\\.gguf$\", \"^mmproj-LFM2\\\\.5-VL-1\\\\.6b-Q8_0\\\\.gguf$\"],
      \"required_files\": {
        \"model\": \"LFM2.5-VL-1.6B-Q4_0.gguf\",
        \"projector\": \"mmproj-LFM2.5-VL-1.6b-Q8_0.gguf\"
      }
    }
  ]
}
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}" "${missing_audio_manifest}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_audio_result
  ERROR_VARIABLE missing_audio_error
)
if(missing_audio_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture manifest validator accepted a manifest without audio fixtures")
endif()
if(NOT missing_audio_error MATCHES "missing required MTMD roles")
  message(FATAL_ERROR "MTMD fixture manifest validator failed for the wrong missing-role reason: ${missing_audio_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_VISION_MODEL="
    "ASTRAL_TEST_VISION_MEDIA="
    "ASTRAL_TEST_AUDIO_MODEL="
    "ASTRAL_TEST_AUDIO_MEDIA="
    "${ASTRAL_BASH_EXECUTABLE}" "${runner}" --check-fixtures
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_fixture_result
  ERROR_VARIABLE missing_fixture_error
)
if(missing_fixture_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture preflight accepted missing fixture environment")
endif()
if(NOT missing_fixture_error MATCHES "missing vision model")
  message(FATAL_ERROR "MTMD fixture preflight failed for the wrong missing-fixture reason: ${missing_fixture_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_VISION_MODEL="
    "ASTRAL_TEST_VISION_MEDIA="
    "ASTRAL_TEST_AUDIO_MODEL="
    "ASTRAL_TEST_AUDIO_MEDIA="
    "${ASTRAL_BASH_EXECUTABLE}" "${runner}"
      --fixture-manifest "${manifest}"
      --check-fixtures
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_fixture_dir_result
  ERROR_VARIABLE missing_fixture_dir_error
)
if(missing_fixture_dir_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture preflight accepted manifest mode without a fixture directory")
endif()
if(NOT missing_fixture_dir_error MATCHES "missing fixture directory")
  message(FATAL_ERROR "MTMD fixture preflight failed for the wrong missing-dir reason: ${missing_fixture_dir_error}")
endif()

set(fixture_dir "${out_dir}/fixtures")
file(MAKE_DIRECTORY "${fixture_dir}")
set(vision_model "${fixture_dir}/vision-model.gguf")
set(vision_media "${fixture_dir}/vision-media.gguf")
set(audio_model "${fixture_dir}/audio-model.gguf")
set(audio_media "${fixture_dir}/audio-media.gguf")
file(WRITE "${vision_model}" "tiny")
file(WRITE "${vision_media}" "tiny")
file(WRITE "${audio_model}" "tiny")
file(WRITE "${audio_media}" "tiny")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_VISION_MODEL=${vision_model}"
    "ASTRAL_TEST_VISION_MEDIA=${vision_media}"
    "ASTRAL_TEST_AUDIO_MODEL=${audio_model}"
    "ASTRAL_TEST_AUDIO_MEDIA=${audio_media}"
    "ASTRAL_MTMD_MIN_MODEL_BYTES=16"
    "ASTRAL_MTMD_MIN_MEDIA_BYTES=16"
    "${ASTRAL_BASH_EXECUTABLE}" "${runner}" --check-fixtures
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE small_fixture_result
  ERROR_VARIABLE small_fixture_error
)
if(small_fixture_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture preflight accepted undersized fixtures")
endif()
if(NOT small_fixture_error MATCHES "too small")
  message(FATAL_ERROR "MTMD fixture preflight failed for the wrong undersized-fixture reason: ${small_fixture_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_VISION_MODEL=${vision_model}"
    "ASTRAL_TEST_VISION_MEDIA=${vision_media}"
    "ASTRAL_TEST_AUDIO_MODEL=${audio_model}"
    "ASTRAL_TEST_AUDIO_MEDIA=${audio_media}"
    "ASTRAL_MTMD_MIN_MODEL_BYTES=1"
    "ASTRAL_MTMD_MIN_MEDIA_BYTES=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${runner}" --check-fixtures
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_fixture_result
  OUTPUT_VARIABLE good_fixture_output
  ERROR_VARIABLE good_fixture_error
)
if(NOT good_fixture_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture preflight rejected valid fixtures: ${good_fixture_output}${good_fixture_error}")
endif()
if(NOT good_fixture_output MATCHES "fixture preflight OK")
  message(FATAL_ERROR "MTMD fixture preflight did not print success evidence: ${good_fixture_output}")
endif()

set(lfm_vision_model "${fixture_dir}/LFM2.5-VL-1.6B-Q4_0.gguf")
set(lfm_vision_media "${fixture_dir}/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf")
set(lfm_audio_model "${fixture_dir}/LFM2.5-Audio-1.5B-Q4_0.gguf")
set(lfm_audio_media "${fixture_dir}/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf")
file(WRITE "${lfm_vision_model}" "tiny")
file(WRITE "${lfm_vision_media}" "tiny")
file(WRITE "${lfm_audio_model}" "tiny")
file(WRITE "${lfm_audio_media}" "tiny")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_VISION_MODEL="
    "ASTRAL_TEST_VISION_MEDIA="
    "ASTRAL_TEST_AUDIO_MODEL="
    "ASTRAL_TEST_AUDIO_MEDIA="
    "ASTRAL_MTMD_MIN_MODEL_BYTES=1"
    "ASTRAL_MTMD_MIN_MEDIA_BYTES=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${runner}"
      --fixture-manifest "${manifest}"
      --fixture-dir "${fixture_dir}"
      --check-fixtures
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE manifest_fixture_result
  OUTPUT_VARIABLE manifest_fixture_output
  ERROR_VARIABLE manifest_fixture_error
)
if(NOT manifest_fixture_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture preflight rejected manifest-resolved fixtures: ${manifest_fixture_output}${manifest_fixture_error}")
endif()
if(NOT manifest_fixture_output MATCHES "fixture manifest: ${manifest}")
  message(FATAL_ERROR "MTMD fixture preflight did not report the manifest source: ${manifest_fixture_output}")
endif()
if(NOT manifest_fixture_output MATCHES "LFM2.5-VL-1.6B-Q4_0.gguf")
  message(FATAL_ERROR "MTMD fixture preflight did not report the manifest-resolved vision model: ${manifest_fixture_output}")
endif()

set(override_vision_model "${fixture_dir}/override-vision-model.gguf")
file(WRITE "${override_vision_model}" "tiny")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_VISION_MODEL="
    "ASTRAL_TEST_VISION_MEDIA="
    "ASTRAL_TEST_AUDIO_MODEL="
    "ASTRAL_TEST_AUDIO_MEDIA="
    "ASTRAL_MTMD_MIN_MODEL_BYTES=1"
    "ASTRAL_MTMD_MIN_MEDIA_BYTES=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${runner}"
      --fixture-manifest "${manifest}"
      --fixture-dir "${fixture_dir}"
      --vision-model "${override_vision_model}"
      --check-fixtures
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE override_fixture_result
  OUTPUT_VARIABLE override_fixture_output
  ERROR_VARIABLE override_fixture_error
)
if(NOT override_fixture_result EQUAL 0)
  message(FATAL_ERROR "MTMD fixture preflight rejected explicit override with manifest defaults: ${override_fixture_output}${override_fixture_error}")
endif()
if(NOT override_fixture_output MATCHES "override-vision-model.gguf")
  message(FATAL_ERROR "MTMD fixture preflight did not prefer explicit vision-model override: ${override_fixture_output}")
endif()

message(STATUS "gate_mtmd_fixture_manifest: OK")
