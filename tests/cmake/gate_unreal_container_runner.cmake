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

function(check_missing_auth variant expected_ref)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "CONTAINER_ENGINE=${fake_engine}"
      "DOCKER_CONFIG=${empty_docker_config}"
      "HOME=${empty_home}"
      "ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS=1"
      "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_container_ci.sh" --variant "${variant}"
    WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
    RESULT_VARIABLE missing_auth_result
    OUTPUT_VARIABLE missing_auth_output
    ERROR_VARIABLE missing_auth_error
  )
  if(missing_auth_result EQUAL 0)
    message(FATAL_ERROR "run_unreal_container_ci.sh accepted missing Epic GHCR auth for ${variant}")
  endif()
  if(missing_auth_result EQUAL 99)
    message(FATAL_ERROR "run_unreal_container_ci.sh called the container engine before rejecting missing Epic GHCR auth for ${variant}")
  endif()
  set(missing_auth_text "${missing_auth_output}\n${missing_auth_error}")
  if(NOT missing_auth_text MATCHES "Epic Unreal container access is not configured")
    message(FATAL_ERROR "run_unreal_container_ci.sh failed without the Epic GHCR auth message for ${variant}: ${missing_auth_text}")
  endif()
  if(NOT missing_auth_text MATCHES "${expected_ref}")
    message(FATAL_ERROR "run_unreal_container_ci.sh missing the pinned ${variant} image ref in the auth failure: ${missing_auth_text}")
  endif()
endfunction()

check_missing_auth("slim" "dev-slim-5[.]7[.]4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6")
check_missing_auth("full" "dev-5[.]7[.]4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CONTAINER_ENGINE=${fake_engine}"
    "DOCKER_CONFIG=${empty_docker_config}"
    "HOME=${empty_home}"
    "ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_container_ci.sh" --ue-version "5.4" --variant "slim"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_auth_54_result
  OUTPUT_VARIABLE missing_auth_54_output
  ERROR_VARIABLE missing_auth_54_error
)
if(missing_auth_54_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_container_ci.sh accepted missing Epic GHCR auth for UE 5.4 slim")
endif()
set(missing_auth_54_text "${missing_auth_54_output}\n${missing_auth_54_error}")
if(NOT missing_auth_54_text MATCHES "dev-slim-5[.]4[.]4")
  message(FATAL_ERROR "run_unreal_container_ci.sh --ue-version 5.4 did not select the UE 5.4 slim image: ${missing_auth_54_text}")
endif()
if(missing_auth_54_text MATCHES "@sha256:")
  message(FATAL_ERROR "run_unreal_container_ci.sh should not invent a digest for UE 5.4 slim: ${missing_auth_54_text}")
endif()

set(auth_docker_config "${out_dir}/auth-docker")
file(MAKE_DIRECTORY "${auth_docker_config}")
file(WRITE "${auth_docker_config}/config.json" "{\"auths\":{\"ghcr.io\":{\"auth\":\"test\"}}}\n")

set(manifest_fake_engine "${out_dir}/fake-manifest-engine")
file(WRITE "${manifest_fake_engine}" [=[
#!/usr/bin/env bash
if [[ "$1" == "manifest" && "$2" == "inspect" ]]; then
  echo "unauthorized" >&2
  exit 23
fi
echo fake-manifest-engine should only inspect manifests >&2
exit 99
]=])
file(CHMOD "${manifest_fake_engine}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CONTAINER_ENGINE=${manifest_fake_engine}"
    "DOCKER_CONFIG=${auth_docker_config}"
    "HOME=${empty_home}"
    "ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_container_ci.sh" --variant "slim"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE manifest_auth_result
  OUTPUT_VARIABLE manifest_auth_output
  ERROR_VARIABLE manifest_auth_error
)
if(manifest_auth_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_container_ci.sh accepted unauthorized Epic GHCR manifest access")
endif()
if(manifest_auth_result EQUAL 99)
  message(FATAL_ERROR "run_unreal_container_ci.sh continued past manifest access preflight")
endif()
set(manifest_auth_text "${manifest_auth_output}\n${manifest_auth_error}")
if(NOT manifest_auth_text MATCHES "Check image access")
  message(FATAL_ERROR "run_unreal_container_ci.sh did not announce manifest access preflight: ${manifest_auth_text}")
endif()
if(NOT manifest_auth_text MATCHES "Unable to inspect Unreal container manifest")
  message(FATAL_ERROR "run_unreal_container_ci.sh did not report unauthorized manifest access: ${manifest_auth_text}")
endif()
if(NOT manifest_auth_text MATCHES "dev-slim-5[.]7[.]4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6")
  message(FATAL_ERROR "run_unreal_container_ci.sh did not include the pinned slim image ref in the manifest failure: ${manifest_auth_text}")
endif()

set(skip_pull_fake_engine "${out_dir}/fake-skip-pull-engine")
file(WRITE "${skip_pull_fake_engine}" [=[
#!/usr/bin/env bash
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
  echo "ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6"
  exit 0
fi
if [[ "$1" == "run" ]]; then
  echo "fake skip-pull reached container run"
  exit 77
fi
echo fake-skip-pull-engine should not pull or inspect manifests >&2
exit 99
]=])
file(CHMOD "${skip_pull_fake_engine}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CONTAINER_ENGINE=${skip_pull_fake_engine}"
    "DOCKER_CONFIG=${empty_docker_config}"
    "HOME=${empty_home}"
    "ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_container_ci.sh" --variant "slim" --skip-pull --skip-native-build
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE skip_pull_result
  OUTPUT_VARIABLE skip_pull_output
  ERROR_VARIABLE skip_pull_error
)
if(NOT skip_pull_result EQUAL 77)
  message(FATAL_ERROR "run_unreal_container_ci.sh --skip-pull did not reach the container run path: result=${skip_pull_result} output=${skip_pull_output} error=${skip_pull_error}")
endif()
set(skip_pull_text "${skip_pull_output}\n${skip_pull_error}")
if(skip_pull_text MATCHES "Check image access|Pull image|Epic Unreal container access is not configured")
  message(FATAL_ERROR "run_unreal_container_ci.sh --skip-pull tried to use registry access: ${skip_pull_text}")
endif()
if(NOT skip_pull_text MATCHES "fake skip-pull reached container run")
  message(FATAL_ERROR "run_unreal_container_ci.sh --skip-pull did not reach the fake container run: ${skip_pull_text}")
endif()

message(STATUS "gate_unreal_container_runner: OK")
