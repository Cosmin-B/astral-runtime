if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()

set(script "${ASTRAL_SOURCE_DIR}/scripts/run_windows_large_page_validation.ps1")
if(NOT EXISTS "${script}")
  message(FATAL_ERROR "Windows large-page validation runner is missing: ${script}")
endif()

file(READ "${script}" script_text)
foreach(required
  "ASTRAL_TEST_EXPECT_LARGE_PAGES"
  "ASTRAL_TEST_EXPECT_LARGE_PAGE_FALLBACK"
  "whoami /all"
  "test_platform|test_core"
)
  if(NOT script_text MATCHES "${required}")
    message(FATAL_ERROR "Windows large-page runner is missing required text: ${required}")
  endif()
endforeach()

file(READ "${ASTRAL_SOURCE_DIR}/tests/test_platform.cpp" test_text)
foreach(required
  "ASTRAL_TEST_EXPECT_LARGE_PAGES"
  "ASTRAL_TEST_EXPECT_LARGE_PAGE_FALLBACK"
  "ASSERT_FALSE\\(expect_large_pages && expect_large_page_fallback\\)"
)
  if(NOT test_text MATCHES "${required}")
    message(FATAL_ERROR "test_platform.cpp is missing Windows large-page expectation hook: ${required}")
  endif()
endforeach()

message(STATUS "gate_windows_large_page_runner: OK")
