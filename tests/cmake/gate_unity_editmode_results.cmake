if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/unity-results-gate")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")

set(unity_runner "${ASTRAL_SOURCE_DIR}/scripts/run_unity_ci_tests.sh")
if(NOT EXISTS "${unity_runner}")
  message(FATAL_ERROR "Unity CI runner is missing: ${unity_runner}")
endif()
set(gameci_runner "${ASTRAL_SOURCE_DIR}/scripts/run_unity_gameci_tests.sh")
if(NOT EXISTS "${gameci_runner}")
  message(FATAL_ERROR "Unity GameCI runner is missing: ${gameci_runner}")
endif()
file(READ "${unity_runner}" unity_runner_text)
foreach(required
  "command -v \\\"[$]\\{unity_editor\\}\\\""
  "unity_editor=\\\"[$]\\(command -v \\\"[$]\\{unity_editor\\}\\\"\\)\\\""
  "\\[unity-ci\\] Editor:"
  "\\[unity-ci\\] Project:"
  "\\[unity-ci\\] Results:"
)
  if(NOT unity_runner_text MATCHES "${required}")
    message(FATAL_ERROR "Unity CI runner is missing preflight text: ${required}")
  endif()
endforeach()
file(READ "${gameci_runner}" gameci_runner_text)
foreach(required
  "unityci/editor:ubuntu-[$]\\{unity_version\\}-[$]\\{image_component\\}-[$]\\{image_version\\}"
  "ASTRAL_UNITY_REQUIRE_NATIVE=1"
  "UNITY_LICENSE"
  "https://game.ci/docs/docker/docker-images/"
  "/opt/unity/Editor/Unity"
  "--dry-run"
)
  if(NOT gameci_runner_text MATCHES "${required}")
    message(FATAL_ERROR "Unity GameCI runner is missing text: ${required}")
  endif()
endforeach()

execute_process(
  COMMAND "${gameci_runner}" --skip-build --skip-pull --dry-run --results-dir "${out_dir}/gameci-dry-run"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE gameci_dry_result
  OUTPUT_VARIABLE gameci_dry_output
  ERROR_VARIABLE gameci_dry_error
)
if(NOT gameci_dry_result EQUAL 0)
  message(FATAL_ERROR "Unity GameCI runner dry-run failed: ${gameci_dry_error}")
endif()
foreach(required
  "unityci/editor:ubuntu-6000.0.57f1-base-3.2.2"
  "\\[unity-gameci\\] Docs: https://game.ci/docs/docker/docker-images/"
  "scripts/run_unity_ci_tests.sh"
  "--editor"
  "/opt/unity/Editor/Unity"
  "build/unity-gameci-results"
)
  if(NOT gameci_dry_output MATCHES "${required}")
    message(FATAL_ERROR "Unity GameCI runner dry-run is missing '${required}'")
  endif()
endforeach()

set(good_xml "${out_dir}/editmode-good.xml")
file(WRITE "${good_xml}"
"<test-run result=\"Passed\" total=\"1\" passed=\"1\" failed=\"0\">
  <test-suite type=\"Assembly\" result=\"Passed\">
    <test-case fullname=\"AstralRT.Tests.NativeAbi\" result=\"Passed\" />
  </test-suite>
</test-run>
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_unity_editmode_results.py" "${good_xml}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_result
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "validate_unity_editmode_results.py rejected valid smoke XML: ${good_error}")
endif()

set(failed_xml "${out_dir}/editmode-failed.xml")
file(WRITE "${failed_xml}"
"<test-run result=\"Failed\" total=\"1\" passed=\"0\" failed=\"1\">
  <test-suite type=\"Assembly\" result=\"Failed\">
    <test-case fullname=\"AstralRT.Tests.NativeAbi\" result=\"Failed\" />
  </test-suite>
</test-run>
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_unity_editmode_results.py" "${failed_xml}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE failed_result
  ERROR_VARIABLE failed_error
)
if(failed_result EQUAL 0)
  message(FATAL_ERROR "validate_unity_editmode_results.py accepted failed Unity XML")
endif()
if(NOT failed_error MATCHES "failed=1")
  message(FATAL_ERROR "validate_unity_editmode_results.py failed for the wrong failed-XML reason: ${failed_error}")
endif()

set(bad_xml "${out_dir}/editmode-malformed.xml")
file(WRITE "${bad_xml}" "<test-run><test-case></test-run>\n")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_unity_editmode_results.py" "${bad_xml}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE bad_result
  ERROR_VARIABLE bad_error
)
if(bad_result EQUAL 0)
  message(FATAL_ERROR "validate_unity_editmode_results.py accepted malformed Unity XML")
endif()
if(NOT bad_error MATCHES "malformed Unity result XML")
  message(FATAL_ERROR "validate_unity_editmode_results.py failed for the wrong malformed-XML reason: ${bad_error}")
endif()

message(STATUS "gate_unity_editmode_results: OK")
