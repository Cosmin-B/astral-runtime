if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()

set(root "${ASTRAL_SOURCE_DIR}")
set(presets_file "${root}/CMakePresets.json")
set(unity_cmake "${root}/plugins/unity/CMakeLists.txt")
set(package_json "${root}/plugins/unity/package.json")
set(plugin_readme "${root}/plugins/unity/Runtime/Plugins/README.md")
set(native_cs "${root}/plugins/unity/Runtime/AstralNative.cs")

foreach(path "${presets_file}" "${unity_cmake}" "${package_json}" "${plugin_readme}" "${native_cs}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Unity mobile package layout input is missing: ${path}")
  endif()
endforeach()

set(unity_sample_workflows
  StreamingChat
  StatefulNpc
  LocalKnowledge
  CharacterVariants
  MultimodalInput
  MultipleConversations
)
foreach(workflow IN LISTS unity_sample_workflows)
  set(sample_source "${root}/plugins/unity/Samples~/${workflow}/${workflow}Example.cs")
  if(NOT EXISTS "${sample_source}")
    message(FATAL_ERROR "Unity mobile package layout input is missing: ${sample_source}")
  endif()
endforeach()

file(READ "${presets_file}" presets_text)
foreach(required
  "\"name\": \"unity-plugin-android-arm64\""
  "\"CMAKE_TOOLCHAIN_FILE\": \"[$]env\\{ANDROID_NDK_HOME\\}/build/cmake/android.toolchain.cmake\""
  "\"ANDROID_ABI\": \"arm64-v8a\""
  "\"ANDROID_PLATFORM\": \"android-21\""
  "\"name\": \"unity-plugin-ios-arm64\""
  "\"CMAKE_SYSTEM_NAME\": \"iOS\""
  "\"CMAKE_OSX_ARCHITECTURES\": \"arm64\""
  "\"ASTRAL_BUILD_STATIC_LIB\": \"ON\""
  "\"ASTRAL_BUILD_SHARED_LIB\": \"OFF\""
)
  if(NOT presets_text MATCHES "${required}")
    message(FATAL_ERROR "CMakePresets.json is missing Unity mobile setting: ${required}")
  endif()
endforeach()

file(READ "${unity_cmake}" unity_cmake_text)
foreach(required
  "CMAKE_SYSTEM_NAME STREQUAL \"Android\""
  "Android/arm64-v8a"
  "CMAKE_SYSTEM_NAME STREQUAL \"iOS\""
  "set\\(ASTRAL_UNITY_PLATFORM_DIR \"iOS\"\\)"
  "set\\(ASTRAL_UNITY_LIB_TARGET astral_rt\\)"
  "set\\(ASTRAL_UNITY_LIB_TARGET astral_rt_shared\\)"
)
  if(NOT unity_cmake_text MATCHES "${required}")
    message(FATAL_ERROR "Unity CMake package layout is missing: ${required}")
  endif()
endforeach()

file(READ "${package_json}" package_json_text)
foreach(required "\"samples\"")
  if(NOT package_json_text MATCHES "${required}")
    message(FATAL_ERROR "Unity package metadata is missing: ${required}")
  endif()
endforeach()
foreach(workflow IN LISTS unity_sample_workflows)
  set(required_path "\"path\": \"Samples~/${workflow}\"")
  if(NOT package_json_text MATCHES "${required_path}")
    message(FATAL_ERROR "Unity package metadata is missing: ${required_path}")
  endif()
endforeach()

file(READ "${plugin_readme}" readme_text)
foreach(required
  "Android/arm64-v8a/"
  "libastral_rt.so  # Android ARM64"
  "cmake --preset unity-plugin-android-arm64"
  "cmake --build --preset unity-plugin-android-arm64 -j"
  "cmake --preset unity-plugin-ios-arm64"
  "cmake --build --preset unity-plugin-ios-arm64 -j"
  "Android/arm64-v8a/libastral_rt.so"
  "iOS/libastral_rt.a"
)
  if(NOT readme_text MATCHES "${required}")
    message(FATAL_ERROR "Unity plugin README is missing mobile package text: ${required}")
  endif()
endforeach()

file(READ "${native_cs}" native_cs_text)
foreach(required
  "UNITY_IOS && !UNITY_EDITOR"
  "DllName = \"__Internal\""
)
  if(NOT native_cs_text MATCHES "${required}")
    message(FATAL_ERROR "Unity native binding is missing iOS static-link setting: ${required}")
  endif()
endforeach()

message(STATUS "gate_unity_mobile_package_layout: OK")
