if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/unreal-results-gate")
set(report_dir "${out_dir}/report")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${report_dir}")

set(log_file "${out_dir}/unreal-automation.log")
file(WRITE "${log_file}" "Automation RunTests AstralRT\nAstralRT.Mock passed\n")
file(WRITE "${report_dir}/index.json" "{\"tests\":[{\"name\":\"AstralRT.Mock\",\"state\":\"Success\"}]}\n")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_unreal_automation_results.py"
    --log "${log_file}"
    --report-dir "${report_dir}"
    --filter "AstralRT"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_result
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "validate_unreal_automation_results.py rejected valid smoke output: ${good_error}")
endif()

file(REMOVE_RECURSE "${report_dir}")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_unreal_automation_results.py"
    --log "${log_file}"
    --report-dir "${report_dir}"
    --filter "AstralRT"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_result
  ERROR_VARIABLE missing_error
)
if(missing_result EQUAL 0)
  message(FATAL_ERROR "validate_unreal_automation_results.py accepted a missing report directory")
endif()
if(NOT missing_error MATCHES "missing Automation report directory")
  message(FATAL_ERROR "validate_unreal_automation_results.py failed for the wrong missing-report reason: ${missing_error}")
endif()

file(MAKE_DIRECTORY "${report_dir}")
file(WRITE "${log_file}" "Automation RunTests AstralRT\nLogAutomationController: Error: Automation Test Failed: AstralRT.Mock\n")
file(WRITE "${report_dir}/index.json" "{\"tests\":[{\"name\":\"AstralRT.Mock\",\"state\":\"Success\"}]}\n")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_unreal_automation_results.py"
    --log "${log_file}"
    --report-dir "${report_dir}"
    --filter "AstralRT"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE failure_result
  ERROR_VARIABLE failure_error
)
if(failure_result EQUAL 0)
  message(FATAL_ERROR "validate_unreal_automation_results.py accepted a failing Automation log")
endif()
if(NOT failure_error MATCHES "failure marker")
  message(FATAL_ERROR "validate_unreal_automation_results.py failed for the wrong failure-log reason: ${failure_error}")
endif()

file(WRITE "${log_file}" "Automation RunTests AstralRT\nAstralRT.Mock completed\n")
file(WRITE "${report_dir}/index.json" "{\"tests\":[{\"name\":\"AstralRT.Mock\",\"state\":\"Fail\"}]}\n")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_unreal_automation_results.py"
    --log "${log_file}"
    --report-dir "${report_dir}"
    --filter "AstralRT"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE report_failure_result
  ERROR_VARIABLE report_failure_error
)
if(report_failure_result EQUAL 0)
  message(FATAL_ERROR "validate_unreal_automation_results.py accepted a failing Automation JSON report")
endif()
if(NOT report_failure_error MATCHES "failed test")
  message(FATAL_ERROR "validate_unreal_automation_results.py failed for the wrong report-state reason: ${report_failure_error}")
endif()

file(WRITE "${report_dir}/index.json" "{\"tests\":[]}\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_unreal_automation_results.py"
    --log "${log_file}"
    --report-dir "${report_dir}"
    --filter "AstralRT"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE empty_report_result
  ERROR_VARIABLE empty_report_error
)
if(empty_report_result EQUAL 0)
  message(FATAL_ERROR "validate_unreal_automation_results.py accepted an Automation JSON report without matching tests")
endif()
if(NOT empty_report_error MATCHES "no test entries")
  message(FATAL_ERROR "validate_unreal_automation_results.py failed for the wrong empty-report reason: ${empty_report_error}")
endif()

message(STATUS "gate_unreal_automation_results: OK")
