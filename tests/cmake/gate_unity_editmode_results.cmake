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
