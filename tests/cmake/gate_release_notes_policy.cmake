if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

set(notes_dir "${ASTRAL_BUILD_DIR}/release-notes-policy")
file(MAKE_DIRECTORY "${notes_dir}")

set(common_notes [=[
# Astral 0.1.0 Release Notes

## Compatibility

- Runtime ABI: 0.1.0
- Supported desktop targets: Linux x86_64
- Unity support: Unity 6000.0.57f1, Collections 1.4.0
- Unreal support: UE 5.4 through UE 5.7
- Compiler/toolchain: clang 20.1.8

## Artifacts

- Core runtime zip: astral-0.1.0-linux-x86_64.zip
- Unity package zip: AstralUnity-0.1.0.zip
- Unreal plugin zip: AstralRT-0.1.0.zip
- Dependency manifest: `dependency-manifest.json`
- ABI layout report: `abi-layout.json`
- Checksums: `checksums.sha256`
- Checksum signature: checksums.sha256.asc
- Signing key or fingerprint: public verification key https://example.invalid/astral.asc
- Signature verification command: `./scripts/validate_release_artifacts.sh --dist dist --expect-unity --expect-unreal --require-signature`

## Validation Evidence

- Native debug gate: `cmake --preset dev && cmake --build --preset dev -j && ctest --preset dev -j --output-on-failure` passed
- Native release gate: `cmake --preset release-with-tests && cmake --build --preset release-with-tests -j && ctest --preset release-with-tests -j --output-on-failure` passed
- CUDA gate: self-hosted runner, CUDA 12.8, RTX 4090, `scripts/run_cuda_parity_matrix.sh --preset-set release --arch native --strict` passed
- MTMD gate: vision/audio fixtures, `scripts/run_multimodal_validation.sh --bench` passed
- HF matrix: full manifest, logs, and summary passed

## Engine Evidence

- Unity Editor ABI tests: Unity 6000.0.57f1, `scripts/run_unity_ci_tests.sh` passed
- Unreal Automation tests: UE 5.4/5.5/5.6/5.7, `scripts/run_unreal_ci_tests.sh` passed
- Unreal container images: ghcr.io/epicgames/unreal-engine:dev-5.7.4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce

## Rollback

- Previous known-good artifact: astral-0.0.9-linux-x86_64.zip sha256:1111111111111111111111111111111111111111111111111111111111111111
- Previous dependency pins: llama.cpp eff06702b, Tracy v0.11.1, Unreal 5.7.4, Unity 6000.0.57f1
- Rollback command or release-manager procedure: promote astral-0.0.9 artifacts and restore dependency-pins.tsv

## Known Gaps

- CUDA multi-GPU release breadth, owner release-manager, waiver expiration 2026-06-01, follow-up issue reference ASTRAL-9d5.8.3
]=])

set(good_notes "${notes_dir}/release-notes-good.md")
file(WRITE "${good_notes}" "${common_notes}")
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_notes.sh" "${good_notes}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_result
  OUTPUT_VARIABLE good_output
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "validate_release_notes.sh rejected strict release notes: ${good_error}")
endif()

set(placeholder_notes "${notes_dir}/release-notes-placeholder.md")
string(REPLACE "Runtime ABI: 0.1.0" "Runtime ABI: <version>" placeholder_text "${common_notes}")
file(WRITE "${placeholder_notes}" "${placeholder_text}")
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_notes.sh" "${placeholder_notes}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE placeholder_result
  ERROR_VARIABLE placeholder_error
)
if(placeholder_result EQUAL 0)
  message(FATAL_ERROR "validate_release_notes.sh accepted template placeholders in strict mode")
endif()
if(NOT placeholder_error MATCHES "template placeholders")
  message(FATAL_ERROR "validate_release_notes.sh failed for the wrong placeholder reason: ${placeholder_error}")
endif()

set(pin_notes "${notes_dir}/release-notes-missing-pins.md")
string(REPLACE "Previous dependency pins: llama.cpp eff06702b, Tracy v0.11.1, Unreal 5.7.4, Unity 6000.0.57f1" "Previous dependency pins: Tracy v0.11.1" pin_text "${common_notes}")
file(WRITE "${pin_notes}" "${pin_text}")
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_notes.sh" "${pin_notes}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE pin_result
  ERROR_VARIABLE pin_error
)
if(pin_result EQUAL 0)
  message(FATAL_ERROR "validate_release_notes.sh accepted rollback notes without llama.cpp and engine pins")
endif()
if(NOT pin_error MATCHES "llama.cpp")
  message(FATAL_ERROR "validate_release_notes.sh failed for the wrong rollback-pin reason: ${pin_error}")
endif()

set(expiration_notes "${notes_dir}/release-notes-missing-expiration.md")
string(REPLACE "waiver expiration 2026-06-01" "waiver expiration pending" expiration_text "${common_notes}")
file(WRITE "${expiration_notes}" "${expiration_text}")
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/validate_release_notes.sh" "${expiration_notes}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE expiration_result
  ERROR_VARIABLE expiration_error
)
if(expiration_result EQUAL 0)
  message(FATAL_ERROR "validate_release_notes.sh accepted a waiver without an ISO expiration date")
endif()
if(NOT expiration_error MATCHES "ISO expiration date")
  message(FATAL_ERROR "validate_release_notes.sh failed for the wrong waiver-expiration reason: ${expiration_error}")
endif()

message(STATUS "gate_release_notes_policy: OK")
