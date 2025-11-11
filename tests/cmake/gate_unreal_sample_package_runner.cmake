if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/unreal-sample-package-gate")
set(project_dir "${out_dir}/AstralSample")
set(archive_dir "${out_dir}/archive")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_sample_package.sh"
    --out "${project_dir}"
    --archive-dir "${archive_dir}"
    --skip-native-build
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_result
  OUTPUT_VARIABLE missing_output
  ERROR_VARIABLE missing_error
)
if(missing_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_sample_package.sh accepted a missing RunUAT path")
endif()
set(missing_text "${missing_output}\n${missing_error}")
if(NOT missing_text MATCHES "Missing Unreal RunUAT path")
  message(FATAL_ERROR "run_unreal_sample_package.sh failed without the RunUAT setup message: ${missing_text}")
endif()

set(fake_runuat "${out_dir}/RunUAT.sh")
file(WRITE "${fake_runuat}" "#!/usr/bin/env bash\necho \"fake RunUAT: $*\"\ncase \" $* \" in\n  *\" BuildCookRun \"*\" -platform=Linux \"*\" -archive \"*) exit 0 ;;\nesac\necho \"missing BuildCookRun package args\" >&2\nexit 42\n")
file(CHMOD "${fake_runuat}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_sample_package.sh"
    --out "${project_dir}"
    --archive-dir "${archive_dir}"
    --runuat "${fake_runuat}"
    --skip-native-build
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE package_result
  OUTPUT_VARIABLE package_output
  ERROR_VARIABLE package_error
)
if(NOT package_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_sample_package.sh failed fake RunUAT package smoke: ${package_error}")
endif()
set(package_text "${package_output}\n${package_error}")
foreach(required_text
    "[unreal_sample] Project:"
    "[unreal_sample] Archive:"
    "[unreal_sample] RunUAT:"
    "[unreal_sample] Platform: Linux"
    "[unreal_sample] Plugin mode: copy"
    "[unreal_sample] BuildCookRun"
    "fake RunUAT: BuildCookRun"
    "-archive"
    "[unreal_sample] OK:")
  string(FIND "${package_text}" "${required_text}" required_text_pos)
  if(required_text_pos EQUAL -1)
    message(FATAL_ERROR "run_unreal_sample_package.sh output missing '${required_text}': ${package_text}")
  endif()
endforeach()

foreach(required_file
    "${project_dir}/AstralSample.uproject"
    "${project_dir}/Plugins/AstralRT/AstralRT.uplugin"
    "${project_dir}/Source/AstralSample/AstralSampleActor.cpp")
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR "sample package runner did not create ${required_file}")
  endif()
endforeach()

message(STATUS "gate_unreal_sample_package_runner: OK")
