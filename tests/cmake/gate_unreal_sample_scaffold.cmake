if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/unreal-sample-scaffold/AstralSample")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${ASTRAL_BUILD_DIR}/unreal-sample-scaffold")

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/create_unreal_sample_project.sh"
    --out "${out_dir}"
    --plugin-mode none
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE scaffold_result
  OUTPUT_VARIABLE scaffold_output
  ERROR_VARIABLE scaffold_error
)
if(NOT scaffold_result EQUAL 0)
  message(FATAL_ERROR "create_unreal_sample_project.sh failed: ${scaffold_error}")
endif()

set(required_files
  "${out_dir}/AstralSample.uproject"
  "${out_dir}/Config/DefaultEngine.ini"
  "${out_dir}/Config/DefaultGame.ini"
  "${out_dir}/Content/AstralSample/Models/mock-model.bytes"
  "${out_dir}/Source/AstralSample.Target.cs"
  "${out_dir}/Source/AstralSampleEditor.Target.cs"
  "${out_dir}/Source/AstralSample/AstralSample.Build.cs"
  "${out_dir}/Source/AstralSample/AstralSampleGameMode.h"
  "${out_dir}/Source/AstralSample/AstralSampleGameMode.cpp"
  "${out_dir}/Source/AstralSample/AstralSampleActor.h"
  "${out_dir}/Source/AstralSample/AstralSampleActor.cpp"
  "${out_dir}/README.md")
foreach(required_file IN LISTS required_files)
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR "Unreal sample scaffold missing ${required_file}")
  endif()
endforeach()

file(READ "${out_dir}/AstralSample.uproject" uproject_text)
file(READ "${out_dir}/Config/DefaultGame.ini" default_game_text)
file(READ "${out_dir}/Source/AstralSample/AstralSample.Build.cs" build_text)
file(READ "${out_dir}/Source/AstralSample/AstralSampleGameMode.cpp" game_mode_text)
file(READ "${out_dir}/Source/AstralSample/AstralSampleActor.cpp" actor_text)
file(READ "${out_dir}/README.md" readme_text)
file(READ "${out_dir}/Content/AstralSample/Models/mock-model.bytes" mock_model_text)

foreach(required_project_text
    "\"EngineAssociation\": \"5.7\""
    "\"Name\": \"AstralRT\""
    "\"Enabled\": true")
  if(NOT uproject_text MATCHES "${required_project_text}")
    message(FATAL_ERROR "Unreal sample uproject missing ${required_project_text}")
  endif()
endforeach()

if(NOT build_text MATCHES "\"AstralRT\"")
  message(FATAL_ERROR "Unreal sample module must depend on AstralRT")
endif()

if(NOT default_game_text MATCHES "DirectoriesToAlwaysStageAsUFS")
  message(FATAL_ERROR "Unreal sample must stage packaged model bytes as UFS content")
endif()
if(NOT game_mode_text MATCHES "SpawnActor<AAstralSampleActor>")
  message(FATAL_ERROR "Unreal sample GameMode must spawn the runtime validation actor")
endif()
if(NOT mock_model_text STREQUAL "mock")
  message(FATAL_ERROR "Unreal sample mock model payload changed unexpectedly")
endif()

foreach(required_actor_text
    "RunGenerationDemo"
    "CancelStreamingDemo"
    "RunEmbeddingDemo"
    "RunPackagedMemorySourceDemo"
    "RunSavedCacheDemo"
    "RunErrorDemo"
    "UAstralModel"
    "UAstralSession"
    "UAstralEmbedder"
    "ProjectContentDir"
    "ProjectSavedDir"
    "LoadFileToArray"
    "SaveArrayToFile"
    "EAstralModelSourceKind::Memory"
    "ModelBytes"
    "AstralSampleAutoQuit"
    "OnStreamBytesNative"
    "EmbedUtf8Bytes"
    "RunMediaFeedDemo"
    "UAstralMediaLibrary::MakeRGBA8ImageFromBytes"
    "MediaSession->FeedAudio"
    "astral_last_error"
    "LogAstralSample")
  if(NOT actor_text MATCHES "${required_actor_text}")
    message(FATAL_ERROR "Unreal sample actor is missing ${required_actor_text}")
  endif()
endforeach()

foreach(required_readme_text
    "Generated sidecar"
    "UE 5.7"
    "model load"
    "streaming generation"
    "cancellation"
    "embeddings"
    "image/audio media feed"
    "packaged content bytes"
    "Saved cache bytes"
    "expected error logging")
  if(NOT readme_text MATCHES "${required_readme_text}")
    message(FATAL_ERROR "Unreal sample README is missing ${required_readme_text}")
  endif()
endforeach()

message(STATUS "gate_unreal_sample_scaffold: OK")
