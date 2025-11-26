if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()

set(root_header "${ASTRAL_SOURCE_DIR}/include/astral_rt.h")
set(unreal_header "${ASTRAL_SOURCE_DIR}/plugins/unreal/AstralRT/Source/ThirdParty/AstralCore/include/astral_rt.h")

if(NOT EXISTS "${root_header}")
  message(FATAL_ERROR "Root ABI header not found: ${root_header}")
endif()

if(NOT EXISTS "${unreal_header}")
  message(FATAL_ERROR "Unreal ThirdParty ABI header not found: ${unreal_header}")
endif()

file(SHA256 "${root_header}" root_hash)
file(SHA256 "${unreal_header}" unreal_hash)

if(NOT root_hash STREQUAL unreal_hash)
  message(FATAL_ERROR
    "Unreal ThirdParty astral_rt.h is out of sync with include/astral_rt.h. "
    "Run: cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j")
endif()

message(STATUS "gate_unreal_header_mirror: OK (${root_hash})")
