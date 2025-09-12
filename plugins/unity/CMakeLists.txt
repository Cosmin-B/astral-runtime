# Unity plugin packaging.
#
# This CMake subdir exists so `cmake --preset unity-plugin` can configure cleanly.
# It packages the built native library into the Unity package layout under:
#   plugins/unity/Runtime/Plugins/<platform>/
#
# Note: non-iOS targets package the shared library artifact.

set(ASTRAL_UNITY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(ASTRAL_UNITY_PLUGIN_DIR "${ASTRAL_UNITY_DIR}/Runtime/Plugins")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  # Unity expects Android native libs under Plugins/Android/<abi>/.
  set(ASTRAL_UNITY_PLATFORM_DIR "Android/arm64-v8a")
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(ASTRAL_UNITY_PLATFORM_DIR "iOS")
elseif(WIN32)
  set(ASTRAL_UNITY_PLATFORM_DIR "x86_64")
elseif(APPLE)
  if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(ASTRAL_UNITY_PLATFORM_DIR "arm64")
  else()
    set(ASTRAL_UNITY_PLATFORM_DIR "x86_64")
  endif()
elseif(UNIX)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ASTRAL_UNITY_PLATFORM_DIR "arm64")
  else()
    set(ASTRAL_UNITY_PLATFORM_DIR "x86_64")
  endif()
else()
  set(ASTRAL_UNITY_PLATFORM_DIR "x86_64")
endif()

set(ASTRAL_UNITY_OUT_DIR "${ASTRAL_UNITY_PLUGIN_DIR}/${ASTRAL_UNITY_PLATFORM_DIR}")

if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(ASTRAL_UNITY_LIB_TARGET astral_rt)
else()
  set(ASTRAL_UNITY_LIB_TARGET astral_rt_shared)
endif()

add_custom_target(astral_unity_plugin_package ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${ASTRAL_UNITY_OUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:${ASTRAL_UNITY_LIB_TARGET}>"
          "${ASTRAL_UNITY_OUT_DIR}/$<TARGET_FILE_NAME:${ASTRAL_UNITY_LIB_TARGET}>"
  DEPENDS ${ASTRAL_UNITY_LIB_TARGET}
  COMMENT "Packaging Astral Unity native plugin artifacts"
  VERBATIM
)
