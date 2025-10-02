if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()

set(workflow "${ASTRAL_SOURCE_DIR}/.github/workflows/release-sign.yml")
if(NOT EXISTS "${workflow}")
  message(FATAL_ERROR "release-sign workflow is missing: ${workflow}")
endif()
set(ci_workflow "${ASTRAL_SOURCE_DIR}/.github/workflows/ci.yml")
if(NOT EXISTS "${ci_workflow}")
  message(FATAL_ERROR "CI workflow is missing: ${ci_workflow}")
endif()

file(READ "${workflow}" text)
file(READ "${ci_workflow}" ci_text)

foreach(required_regex
  "Validate release evidence"
  "validate_release_evidence.py"
  "Import release signing key"
  "Sign checksum manifests"
)
  if(NOT text MATCHES "${required_regex}")
    message(FATAL_ERROR "release-sign workflow is missing required text: ${required_regex}")
  endif()
endforeach()

foreach(required_text
  "release-artifacts/*/release-evidence.json"
  "release-artifacts/**/release-evidence.json"
)
  string(FIND "${text}" "${required_text}" required_pos)
  if(required_pos LESS 0)
    message(FATAL_ERROR "release-sign workflow is missing required text: ${required_text}")
  endif()
endforeach()

string(FIND "${text}" "Validate release evidence" evidence_pos)
string(FIND "${text}" "Import release signing key" import_pos)
string(FIND "${text}" "Sign checksum manifests" sign_pos)

if(evidence_pos LESS 0 OR import_pos LESS 0 OR sign_pos LESS 0)
  message(FATAL_ERROR "release-sign workflow ordering markers are missing")
endif()
if(NOT evidence_pos LESS import_pos)
  message(FATAL_ERROR "release evidence must be validated before importing the signing key")
endif()
if(NOT evidence_pos LESS sign_pos)
  message(FATAL_ERROR "release evidence must be validated before signing checksum manifests")
endif()
if(NOT ci_text MATCHES "dist/release-evidence\\.json")
  message(FATAL_ERROR "desktop release artifact upload must include dist/release-evidence.json when present")
endif()

message(STATUS "gate_release_sign_workflow: OK")
