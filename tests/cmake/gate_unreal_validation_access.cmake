if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

set(access_script "${ASTRAL_SOURCE_DIR}/scripts/check_unreal_validation_access.sh")
if(NOT EXISTS "${access_script}")
  message(FATAL_ERROR "check_unreal_validation_access.sh is missing")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/unreal-validation-access-gate")
set(empty_home "${out_dir}/home")
set(empty_docker_config "${out_dir}/docker")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${empty_home}" "${empty_docker_config}")

set(missing_engine "${out_dir}/fake-missing-engine")
file(WRITE "${missing_engine}" [=[
#!/usr/bin/env bash
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
  exit 1
fi
echo fake-missing-engine should not be used for other commands >&2
exit 99
]=])
file(CHMOD "${missing_engine}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CONTAINER_ENGINE=${missing_engine}"
    "DOCKER_CONFIG=${empty_docker_config}"
    "HOME=${empty_home}"
    "ASTRAL_UNREAL_ACCESS_TIMEOUT_SECONDS=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${access_script}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_result
  OUTPUT_VARIABLE missing_output
  ERROR_VARIABLE missing_error
)
if(missing_result EQUAL 0)
  message(FATAL_ERROR "check_unreal_validation_access.sh accepted missing Unreal access")
endif()
if(missing_result EQUAL 99)
  message(FATAL_ERROR "check_unreal_validation_access.sh called an unexpected container command")
endif()
set(missing_text "${missing_output}\n${missing_error}")
foreach(required_missing_text
    "BLOCKED: real Unreal validation needs Epic GHCR access"
    "dev-5[.]7[.]4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce"
    "dev-slim-5[.]7[.]4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6"
    "UNREAL_54_EDITOR is unset"
    "RunUAT path is not configured")
  if(NOT missing_text MATCHES "${required_missing_text}")
    message(FATAL_ERROR "Unreal access missing-output drifted; missing '${required_missing_text}': ${missing_text}")
  endif()
endforeach()

set(cached_engine "${out_dir}/fake-cached-engine")
file(WRITE "${cached_engine}" [=[
#!/usr/bin/env bash
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
  echo "ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce"
  echo "ghcr.io/epicgames/unreal-engine:dev-slim-5.7.4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6"
  exit 0
fi
echo fake-cached-engine should only inspect cached images >&2
exit 99
]=])
file(CHMOD "${cached_engine}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CONTAINER_ENGINE=${cached_engine}"
    "DOCKER_CONFIG=${empty_docker_config}"
    "HOME=${empty_home}"
    "ASTRAL_UNREAL_ACCESS_TIMEOUT_SECONDS=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${access_script}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE cached_result
  OUTPUT_VARIABLE cached_output
  ERROR_VARIABLE cached_error
)
if(NOT cached_result EQUAL 0)
  message(FATAL_ERROR "check_unreal_validation_access.sh rejected cached UE container images: ${cached_output}\n${cached_error}")
endif()
set(cached_text "${cached_output}\n${cached_error}")
if(NOT cached_text MATCHES "READY: UE 5[.]7 full/slim container access is available")
  message(FATAL_ERROR "Unreal access cached-output did not report container readiness: ${cached_text}")
endif()

set(auth_docker_config "${out_dir}/auth-docker")
file(MAKE_DIRECTORY "${auth_docker_config}")
file(WRITE "${auth_docker_config}/config.json" "{\"auths\":{\"ghcr.io\":{\"auth\":\"test\"}}}\n")

set(manifest_engine "${out_dir}/fake-manifest-engine")
file(WRITE "${manifest_engine}" [=[
#!/usr/bin/env bash
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
  exit 1
fi
if [[ "$1" == "manifest" && "$2" == "inspect" ]]; then
  echo "{}"
  exit 0
fi
echo fake-manifest-engine should only inspect images or manifests >&2
exit 99
]=])
file(CHMOD "${manifest_engine}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CONTAINER_ENGINE=${manifest_engine}"
    "DOCKER_CONFIG=${auth_docker_config}"
    "HOME=${empty_home}"
    "ASTRAL_UNREAL_ACCESS_TIMEOUT_SECONDS=1"
    "${ASTRAL_BASH_EXECUTABLE}" "${access_script}" --check-registry
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE manifest_result
  OUTPUT_VARIABLE manifest_output
  ERROR_VARIABLE manifest_error
)
if(NOT manifest_result EQUAL 0)
  message(FATAL_ERROR "check_unreal_validation_access.sh rejected readable UE registry manifests: ${manifest_output}\n${manifest_error}")
endif()
set(manifest_text "${manifest_output}\n${manifest_error}")
if(NOT manifest_text MATCHES "OK: full registry manifest is readable")
  message(FATAL_ERROR "Unreal access manifest-output did not report full manifest readiness: ${manifest_text}")
endif()
if(NOT manifest_text MATCHES "OK: slim registry manifest is readable")
  message(FATAL_ERROR "Unreal access manifest-output did not report slim manifest readiness: ${manifest_text}")
endif()

set(ue_root "${out_dir}/ue")
foreach(version IN ITEMS 5.4 5.5 5.6 5.7)
  set(version_root "${ue_root}/${version}")
  file(MAKE_DIRECTORY "${version_root}/Engine/Binaries/Linux")
  file(WRITE "${version_root}/Engine/Binaries/Linux/UnrealEditor-Cmd" "#!/usr/bin/env bash\nexit 0\n")
  file(CHMOD "${version_root}/Engine/Binaries/Linux/UnrealEditor-Cmd"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
endforeach()
file(MAKE_DIRECTORY "${ue_root}/5.7/Engine/Build/BatchFiles")
file(WRITE "${ue_root}/5.7/Engine/Build/BatchFiles/RunUAT.sh" "#!/usr/bin/env bash\nexit 0\n")
file(CHMOD "${ue_root}/5.7/Engine/Build/BatchFiles/RunUAT.sh"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CONTAINER_ENGINE=${out_dir}/no-container-engine"
    "DOCKER_CONFIG=${empty_docker_config}"
    "HOME=${empty_home}"
    "UNREAL_54_EDITOR=${ue_root}/5.4/Engine/Binaries/Linux/UnrealEditor-Cmd"
    "UNREAL_55_EDITOR=${ue_root}/5.5/Engine/Binaries/Linux/UnrealEditor-Cmd"
    "UNREAL_56_EDITOR=${ue_root}/5.6/Engine/Binaries/Linux/UnrealEditor-Cmd"
    "UNREAL_57_EDITOR=${ue_root}/5.7/Engine/Binaries/Linux/UnrealEditor-Cmd"
    "${ASTRAL_BASH_EXECUTABLE}" "${access_script}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE editor_result
  OUTPUT_VARIABLE editor_output
  ERROR_VARIABLE editor_error
)
if(NOT editor_result EQUAL 0)
  message(FATAL_ERROR "check_unreal_validation_access.sh rejected executable UE editor matrix: ${editor_output}\n${editor_error}")
endif()
set(editor_text "${editor_output}\n${editor_error}")
if(NOT editor_text MATCHES "READY: UE 5[.]4-5[.]7 editor matrix is configured")
  message(FATAL_ERROR "Unreal access editor-output did not report matrix readiness: ${editor_text}")
endif()
if(NOT editor_text MATCHES "OK: RunUAT available")
  message(FATAL_ERROR "Unreal access editor-output did not resolve RunUAT from UE 5.7 editor: ${editor_text}")
endif()

message(STATUS "gate_unreal_validation_access: OK")
