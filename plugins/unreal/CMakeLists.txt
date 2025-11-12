# Unreal plugin packaging (no Unreal Engine required at CMake time).
#
# This target copies the built `astral_rt` static library and public headers into the
# plugin's ThirdParty layout so UnrealBuildTool can link against it.

set(ASTRAL_UE_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/AstralRT")
set(ASTRAL_UE_THIRDPARTY_DIR "${ASTRAL_UE_PLUGIN_DIR}/Source/ThirdParty/AstralCore")
set(ASTRAL_UE_INCLUDE_DIR "${ASTRAL_UE_THIRDPARTY_DIR}/include")
set(ASTRAL_UE_LIB_DIR "${ASTRAL_UE_THIRDPARTY_DIR}/lib")

if(WIN32)
  set(ASTRAL_UE_PLATFORM_DIR "Win64")
elseif(APPLE)
  set(ASTRAL_UE_PLATFORM_DIR "Mac")
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(ASTRAL_UE_PLATFORM_DIR "IOS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(ASTRAL_UE_PLATFORM_DIR "Android/arm64-v8a")
elseif(UNIX)
  set(ASTRAL_UE_PLATFORM_DIR "Linux")
else()
  set(ASTRAL_UE_PLATFORM_DIR "Unknown")
endif()

set(ASTRAL_UE_OUT_LIB_DIR "${ASTRAL_UE_LIB_DIR}/${ASTRAL_UE_PLATFORM_DIR}")

add_custom_target(astral_unreal_plugin_package ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${ASTRAL_UE_INCLUDE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${ASTRAL_UE_OUT_LIB_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/include/astral_rt.h"
          "${ASTRAL_UE_INCLUDE_DIR}/astral_rt.h"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:astral_rt>"
          "${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:astral_rt>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:llama>"
          "${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:llama>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:llama-common>"
          "${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:llama-common>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:llama-common-base>"
          "${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:llama-common-base>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:ggml>"
          "${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:ggml>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:ggml-cpu>"
          "${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:ggml-cpu>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:ggml-base>"
          "${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:ggml-base>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:cpp-httplib>"
          "${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:cpp-httplib>"
  COMMAND ${CMAKE_COMMAND}
          -DASTRAL_SOURCE_HEADER=${CMAKE_SOURCE_DIR}/include/astral_rt.h
          -DASTRAL_PACKAGED_HEADER=${ASTRAL_UE_INCLUDE_DIR}/astral_rt.h
          -DASTRAL_SOURCE_LIBRARY=$<TARGET_FILE:astral_rt>
          -DASTRAL_PACKAGED_LIBRARY=${ASTRAL_UE_OUT_LIB_DIR}/$<TARGET_FILE_NAME:astral_rt>
          -P "${CMAKE_SOURCE_DIR}/cmake/validate_unreal_thirdparty_provenance.cmake"
  DEPENDS astral_rt llama llama-common llama-common-base ggml ggml-cpu ggml-base cpp-httplib
  COMMENT "Packaging AstralRT Unreal plugin ThirdParty artifacts"
  VERBATIM
)
