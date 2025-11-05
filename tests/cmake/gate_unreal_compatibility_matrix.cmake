if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

set(matrix_script "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_compatibility_matrix.sh")
if(NOT EXISTS "${matrix_script}")
  message(FATAL_ERROR "run_unreal_compatibility_matrix.sh is missing")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${matrix_script}" --versions "5.4" --skip-native-build
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_editor_result
  OUTPUT_VARIABLE missing_editor_output
  ERROR_VARIABLE missing_editor_error
)
if(missing_editor_result EQUAL 0)
  message(FATAL_ERROR "Unreal compatibility matrix accepted a missing required editor")
endif()
set(missing_editor_text "${missing_editor_output}\n${missing_editor_error}")
if(NOT missing_editor_text MATCHES "Missing UNREAL_54_EDITOR for UE 5[.]4")
  message(FATAL_ERROR "Unreal compatibility matrix missing-editor error drifted: ${missing_editor_text}")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${matrix_script}" --versions "5.8" --allow-missing --skip-native-build
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE unsupported_result
  OUTPUT_VARIABLE unsupported_output
  ERROR_VARIABLE unsupported_error
)
if(unsupported_result EQUAL 0)
  message(FATAL_ERROR "Unreal compatibility matrix accepted an unsupported version")
endif()
set(unsupported_text "${unsupported_output}\n${unsupported_error}")
if(NOT unsupported_text MATCHES "Unsupported Unreal version '5[.]8'")
  message(FATAL_ERROR "Unreal compatibility matrix unsupported-version error drifted: ${unsupported_text}")
endif()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${matrix_script}" --versions "5.4 5.5 5.6 5.7" --allow-missing --skip-native-build
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE all_skipped_result
  OUTPUT_VARIABLE all_skipped_output
  ERROR_VARIABLE all_skipped_error
)
if(all_skipped_result EQUAL 0)
  message(FATAL_ERROR "Unreal compatibility matrix accepted an all-skipped --allow-missing run")
endif()
set(all_skipped_text "${all_skipped_output}\n${all_skipped_error}")
foreach(required_skip_text
    "Skipping UE 5.4: UNREAL_54_EDITOR is unset"
    "Skipping UE 5.5: UNREAL_55_EDITOR is unset"
    "Skipping UE 5.6: UNREAL_56_EDITOR is unset"
    "Skipping UE 5.7: UNREAL_57_EDITOR is unset"
    "No Unreal versions ran")
  if(NOT all_skipped_text MATCHES "${required_skip_text}")
    message(FATAL_ERROR "Unreal compatibility matrix all-skipped output is missing '${required_skip_text}': ${all_skipped_text}")
  endif()
endforeach()

message(STATUS "gate_unreal_compatibility_matrix: OK")
