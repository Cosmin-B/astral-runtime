if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/unreal-container-runner-gate")
set(empty_home "${out_dir}/home")
set(empty_docker_config "${out_dir}/docker")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${empty_home}" "${empty_docker_config}")

set(fake_engine "${out_dir}/fake-container-engine")
file(WRITE "${fake_engine}" "#!/usr/bin/env bash\necho fake-container-engine should not be called >&2\nexit 99\n")
file(CHMOD "${fake_engine}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CONTAINER_ENGINE=${fake_engine}"
    "DOCKER_CONFIG=${empty_docker_config}"
    "HOME=${empty_home}"
    "ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_container_ci.sh" --variant slim
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_auth_result
  OUTPUT_VARIABLE missing_auth_output
  ERROR_VARIABLE missing_auth_error
)
if(missing_auth_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_container_ci.sh accepted missing Epic GHCR auth")
endif()
if(missing_auth_result EQUAL 99)
  message(FATAL_ERROR "run_unreal_container_ci.sh called the container engine before rejecting missing Epic GHCR auth")
endif()
set(missing_auth_text "${missing_auth_output}\n${missing_auth_error}")
if(NOT missing_auth_text MATCHES "Epic Unreal container access is not configured")
  message(FATAL_ERROR "run_unreal_container_ci.sh failed without the Epic GHCR auth message: ${missing_auth_text}")
endif()
if(NOT missing_auth_text MATCHES "dev-slim-5[.]7[.]4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6")
  message(FATAL_ERROR "run_unreal_container_ci.sh missing the pinned slim image ref in the auth failure: ${missing_auth_text}")
endif()

message(STATUS "gate_unreal_container_runner: OK")
