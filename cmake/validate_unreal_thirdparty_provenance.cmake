foreach(name
    ASTRAL_SOURCE_HEADER
    ASTRAL_PACKAGED_HEADER
    ASTRAL_SOURCE_LIBRARY
    ASTRAL_PACKAGED_LIBRARY)
  if(NOT DEFINED ${name} OR "${${name}}" STREQUAL "")
    message(FATAL_ERROR "${name} not set")
  endif()
endforeach()

foreach(path
    "${ASTRAL_SOURCE_HEADER}"
    "${ASTRAL_PACKAGED_HEADER}"
    "${ASTRAL_SOURCE_LIBRARY}"
    "${ASTRAL_PACKAGED_LIBRARY}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Unreal ThirdParty provenance input is missing: ${path}")
  endif()
endforeach()

file(SHA256 "${ASTRAL_SOURCE_HEADER}" source_header_sha)
file(SHA256 "${ASTRAL_PACKAGED_HEADER}" packaged_header_sha)
if(NOT source_header_sha STREQUAL packaged_header_sha)
  message(FATAL_ERROR
    "Unreal ThirdParty astral_rt.h is stale: ${ASTRAL_PACKAGED_HEADER} "
    "does not match ${ASTRAL_SOURCE_HEADER}")
endif()

file(SIZE "${ASTRAL_SOURCE_LIBRARY}" source_library_size)
file(SIZE "${ASTRAL_PACKAGED_LIBRARY}" packaged_library_size)
if(source_library_size EQUAL 0)
  message(FATAL_ERROR "Built Astral runtime library is empty: ${ASTRAL_SOURCE_LIBRARY}")
endif()
if(packaged_library_size EQUAL 0)
  message(FATAL_ERROR "Unreal ThirdParty runtime library is empty: ${ASTRAL_PACKAGED_LIBRARY}")
endif()

file(SHA256 "${ASTRAL_SOURCE_LIBRARY}" source_library_sha)
file(SHA256 "${ASTRAL_PACKAGED_LIBRARY}" packaged_library_sha)
if(NOT source_library_sha STREQUAL packaged_library_sha)
  message(FATAL_ERROR
    "Unreal ThirdParty runtime library is stale: ${ASTRAL_PACKAGED_LIBRARY} "
    "does not match ${ASTRAL_SOURCE_LIBRARY}")
endif()

message(STATUS "Unreal ThirdParty provenance OK: ${ASTRAL_PACKAGED_LIBRARY}")
