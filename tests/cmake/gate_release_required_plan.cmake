if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_release_required_gates.sh" --print-plan
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_result
  ERROR_VARIABLE missing_error
)
if(missing_result EQUAL 0)
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan accepted missing release environment")
endif()
if(NOT missing_error MATCHES "ASTRAL_TEST_VISION_MODEL")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not report missing MTMD environment: ${missing_error}")
endif()

set(plan_env
  "ASTRAL_TEST_VISION_MODEL=/models/vision.gguf"
  "ASTRAL_TEST_VISION_MEDIA=/models/mmproj-vision.gguf"
  "ASTRAL_TEST_AUDIO_MODEL=/models/audio.gguf"
  "ASTRAL_TEST_AUDIO_MEDIA=/models/mmproj-audio.gguf"
  "UNREAL_54_EDITOR=/opt/Unreal-5.4/Engine/Binaries/Linux/UnrealEditor-Cmd"
  "UNREAL_55_EDITOR=/opt/Unreal-5.5/Engine/Binaries/Linux/UnrealEditor-Cmd"
  "UNREAL_56_EDITOR=/opt/Unreal-5.6/Engine/Binaries/Linux/UnrealEditor-Cmd"
  "UNREAL_57_EDITOR=/opt/Unreal-5.7/Engine/Binaries/Linux/UnrealEditor-Cmd"
  "UNREAL_RUNUAT=/opt/Unreal-5.7/Engine/Build/BatchFiles/RunUAT.sh"
  "UNITY_EDITOR=/opt/Unity/Editor/Unity"
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    ${plan_env}
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_release_required_gates.sh" --print-plan --cuda-arch native --cuda-strict --mtmd-bench
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_result
  OUTPUT_VARIABLE good_output
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan rejected complete environment: ${good_error}")
endif()
if(NOT good_output MATCHES "plan environment OK")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not print success: ${good_output}")
endif()
if(NOT good_output MATCHES "sanitizers: scripts/run_asan\\.sh && scripts/run_tsan\\.sh")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not require sanitizer lanes: ${good_output}")
endif()
if(NOT good_output MATCHES "--arch native")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not include explicit CUDA architecture: ${good_output}")
endif()
if(NOT good_output MATCHES "Unreal access: scripts/check_unreal_validation_access\\.sh --check-registry")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not require the Unreal access readiness lane: ${good_output}")
endif()
if(NOT good_output MATCHES "Unreal UE 5[.]7 full container: scripts/run_unreal_container_ci\\.sh --variant full --filter AstralRT --install-cmake")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not require the Unreal full container lane: ${good_output}")
endif()
if(NOT good_output MATCHES "Unreal UE 5[.]7 slim container: scripts/run_unreal_container_ci\\.sh --variant slim --filter AstralRT --skip-native-build")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not require the Unreal slim container lane: ${good_output}")
endif()
if(NOT good_output MATCHES "Unreal sample package: scripts/run_unreal_sample_package\\.sh --platform Linux --run-sample .* --sample-memory-backend mock --sample-media-backend mock")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not require the Unreal sample package lane: ${good_output}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_VISION_MODEL="
    "ASTRAL_TEST_VISION_MEDIA="
    "ASTRAL_TEST_AUDIO_MODEL="
    "ASTRAL_TEST_AUDIO_MEDIA="
    "ASTRAL_MTMD_FIXTURE_MANIFEST=${ASTRAL_SOURCE_DIR}/scripts/mtmd_fixture_manifest_lfm25.json"
    "ASTRAL_MTMD_FIXTURE_DIR=/models/hf-lfm25"
    "UNREAL_54_EDITOR=/opt/Unreal-5.4/Engine/Binaries/Linux/UnrealEditor-Cmd"
    "UNREAL_55_EDITOR=/opt/Unreal-5.5/Engine/Binaries/Linux/UnrealEditor-Cmd"
    "UNREAL_56_EDITOR=/opt/Unreal-5.6/Engine/Binaries/Linux/UnrealEditor-Cmd"
    "UNREAL_57_EDITOR=/opt/Unreal-5.7/Engine/Binaries/Linux/UnrealEditor-Cmd"
    "UNREAL_RUNUAT=/opt/Unreal-5.7/Engine/Build/BatchFiles/RunUAT.sh"
    "UNITY_EDITOR=/opt/Unity/Editor/Unity"
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_release_required_gates.sh" --print-plan --cuda-arch native
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE manifest_env_result
  OUTPUT_VARIABLE manifest_env_output
  ERROR_VARIABLE manifest_env_error
)
if(NOT manifest_env_result EQUAL 0)
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan rejected manifest-driven MTMD environment: ${manifest_env_error}")
endif()
if(NOT manifest_env_output MATCHES "plan environment OK")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan did not accept manifest-driven MTMD environment: ${manifest_env_output}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    ${plan_env}
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_release_required_gates.sh" --print-plan --cuda-strict --mtmd-bench
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_arch_result
  ERROR_VARIABLE missing_arch_error
)
if(missing_arch_result EQUAL 0)
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan accepted a release CUDA plan without --cuda-arch")
endif()
if(NOT missing_arch_error MATCHES "--cuda-arch")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan failed for the wrong missing CUDA arch reason: ${missing_arch_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ASTRAL_TEST_VISION_MODEL=/models/vision.gguf"
    "ASTRAL_TEST_VISION_MEDIA=/models/mmproj-vision.gguf"
    "ASTRAL_TEST_AUDIO_MODEL=/models/audio.gguf"
    "ASTRAL_TEST_AUDIO_MEDIA=/models/mmproj-audio.gguf"
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_release_required_gates.sh" --print-plan --cuda-arch native --skip-engine
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE skip_result
  OUTPUT_VARIABLE skip_output
  ERROR_VARIABLE skip_error
)
if(NOT skip_result EQUAL 0)
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan --skip-engine rejected MTMD-only environment: ${skip_error}")
endif()
if(NOT skip_output MATCHES "Unreal container, matrix, and sample package: skipped for local diagnosis")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan --skip-engine did not label Unreal container skip: ${skip_output}")
endif()
if(NOT skip_output MATCHES "skipped for local diagnosis")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan --skip-engine did not label skipped engine lanes: ${skip_output}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    ${plan_env}
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_release_required_gates.sh" --print-plan --cuda-arch native --skip-sanitizers
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE skip_sanitizers_result
  OUTPUT_VARIABLE skip_sanitizers_output
  ERROR_VARIABLE skip_sanitizers_error
)
if(NOT skip_sanitizers_result EQUAL 0)
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan --skip-sanitizers rejected complete environment: ${skip_sanitizers_error}")
endif()
if(NOT skip_sanitizers_output MATCHES "sanitizers: skipped for local diagnosis")
  message(FATAL_ERROR "run_release_required_gates.sh --print-plan --skip-sanitizers did not label sanitizer skip: ${skip_sanitizers_output}")
endif()

message(STATUS "gate_release_required_plan: OK")
