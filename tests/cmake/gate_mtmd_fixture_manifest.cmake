if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(script "${ASTRAL_SOURCE_DIR}/scripts/validate_mtmd_fixture_manifest.py")
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

message(STATUS "gate_mtmd_fixture_manifest: OK")
