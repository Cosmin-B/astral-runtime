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
  set(ASTRAL_UE_LIB_NAME "astral_rt.lib")
elseif(APPLE)
  set(ASTRAL_UE_PLATFORM_DIR "Mac")
  set(ASTRAL_UE_LIB_NAME "libastral_rt.a")
elseif(UNIX)
  set(ASTRAL_UE_PLATFORM_DIR "Linux")
  set(ASTRAL_UE_LIB_NAME "libastral_rt.a")
else()
  set(ASTRAL_UE_PLATFORM_DIR "Unknown")
  set(ASTRAL_UE_LIB_NAME "")
endif()

set(ASTRAL_UE_OUT_LIB_DIR "${ASTRAL_UE_LIB_DIR}/${ASTRAL_UE_PLATFORM_DIR}")

if(ASTRAL_UE_LIB_NAME STREQUAL "")
  add_custom_target(astral_unreal_plugin_package ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ASTRAL_UE_INCLUDE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/include/astral_rt.h"
            "${ASTRAL_UE_INCLUDE_DIR}/astral_rt.h"
    DEPENDS astral_rt
    COMMENT "Packaging AstralRT Unreal plugin headers (unsupported platform)"
    VERBATIM
  )
else()
  set(ASTRAL_UE_OUT_LIB "${ASTRAL_UE_OUT_LIB_DIR}/${ASTRAL_UE_LIB_NAME}")

  add_custom_target(astral_unreal_plugin_package ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ASTRAL_UE_INCLUDE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ASTRAL_UE_OUT_LIB_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/include/astral_rt.h"
            "${ASTRAL_UE_INCLUDE_DIR}/astral_rt.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:astral_rt>"
            "${ASTRAL_UE_OUT_LIB}"
    DEPENDS astral_rt
    COMMENT "Packaging AstralRT Unreal plugin ThirdParty artifacts"
    VERBATIM
  )
endif()

