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
set(template_dir "${ASTRAL_SOURCE_DIR}/examples/unreal/AstralSample")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${ASTRAL_BUILD_DIR}/unreal-sample-scaffold")

set(template_files
  "AstralSample.uproject"
  "Config/DefaultEngine.ini"
  "Config/DefaultGame.ini"
  "Content/AstralSample/Models/mock-model.bytes"
  "Source/AstralSample.Target.cs"
  "Source/AstralSampleEditor.Target.cs"
  "Source/AstralSample/AstralSample.Build.cs"
  "Source/AstralSample/AstralSample.cpp"
  "Source/AstralSample/AstralSampleGameMode.h"
  "Source/AstralSample/AstralSampleGameMode.cpp"
  "Source/AstralSample/AstralSampleActor.h"
  "Source/AstralSample/AstralSampleActor.cpp"
  "Source/AstralSample/Examples/AstralStreamingChatComponent.h"
  "Source/AstralSample/Examples/AstralStreamingChatComponent.cpp"
  "Source/AstralSample/Examples/AstralMultipleConversationsComponent.h"
  "Source/AstralSample/Examples/AstralMultipleConversationsComponent.cpp"
  "Source/AstralSample/Examples/AstralStatefulNpcComponent.h"
  "Source/AstralSample/Examples/AstralStatefulNpcComponent.cpp"
  "Source/AstralSample/Examples/AstralLocalKnowledgeComponent.h"
  "Source/AstralSample/Examples/AstralLocalKnowledgeComponent.cpp"
  "README.md"
)
foreach(relative_file IN LISTS template_files)
  if(NOT EXISTS "${template_dir}/${relative_file}")
    message(FATAL_ERROR "Unreal sample template missing ${template_dir}/${relative_file}")
  endif()
endforeach()

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
  "${out_dir}/Source/AstralSample/Examples/AstralStreamingChatComponent.h"
  "${out_dir}/Source/AstralSample/Examples/AstralStreamingChatComponent.cpp"
  "${out_dir}/Source/AstralSample/Examples/AstralMultipleConversationsComponent.h"
  "${out_dir}/Source/AstralSample/Examples/AstralMultipleConversationsComponent.cpp"
  "${out_dir}/Source/AstralSample/Examples/AstralStatefulNpcComponent.h"
  "${out_dir}/Source/AstralSample/Examples/AstralStatefulNpcComponent.cpp"
  "${out_dir}/Source/AstralSample/Examples/AstralLocalKnowledgeComponent.h"
  "${out_dir}/Source/AstralSample/Examples/AstralLocalKnowledgeComponent.cpp"
  "${out_dir}/README.md")
foreach(required_file IN LISTS required_files)
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR "Unreal sample scaffold missing ${required_file}")
  endif()
endforeach()

foreach(relative_file IN LISTS template_files)
  file(SHA256 "${template_dir}/${relative_file}" template_hash)
  file(SHA256 "${out_dir}/${relative_file}" generated_hash)
  if(NOT template_hash STREQUAL generated_hash)
    message(FATAL_ERROR "Generated Unreal sample differs from template: ${relative_file}")
  endif()
endforeach()

file(READ "${out_dir}/AstralSample.uproject" uproject_text)
file(READ "${out_dir}/Config/DefaultGame.ini" default_game_text)
file(READ "${out_dir}/Source/AstralSample.Target.cs" target_text)
file(READ "${out_dir}/Source/AstralSampleEditor.Target.cs" editor_target_text)
file(READ "${out_dir}/Source/AstralSample/AstralSample.Build.cs" build_text)
file(READ "${out_dir}/Source/AstralSample/AstralSampleGameMode.cpp" game_mode_text)
file(READ "${out_dir}/Source/AstralSample/AstralSampleActor.h" actor_header_text)
file(READ "${out_dir}/Source/AstralSample/AstralSampleActor.cpp" actor_text)
file(READ "${out_dir}/Source/AstralSample/Examples/AstralStreamingChatComponent.h" streaming_chat_header_text)
file(READ "${out_dir}/Source/AstralSample/Examples/AstralStreamingChatComponent.cpp" streaming_chat_text)
file(READ "${out_dir}/Source/AstralSample/Examples/AstralMultipleConversationsComponent.h" multiple_conversations_header_text)
file(READ "${out_dir}/Source/AstralSample/Examples/AstralMultipleConversationsComponent.cpp" multiple_conversations_text)
file(READ "${out_dir}/Source/AstralSample/Examples/AstralStatefulNpcComponent.h" stateful_npc_header_text)
file(READ "${out_dir}/Source/AstralSample/Examples/AstralStatefulNpcComponent.cpp" stateful_npc_text)
file(READ "${out_dir}/Source/AstralSample/Examples/AstralLocalKnowledgeComponent.h" local_knowledge_header_text)
file(READ "${out_dir}/Source/AstralSample/Examples/AstralLocalKnowledgeComponent.cpp" local_knowledge_text)
file(READ "${out_dir}/README.md" readme_text)
file(READ "${out_dir}/Content/AstralSample/Models/mock-model.bytes" mock_model_text)
set(actor_full_text "${actor_header_text}\n${actor_text}")
set(streaming_chat_full_text "${streaming_chat_header_text}\n${streaming_chat_text}")
set(multiple_conversations_full_text "${multiple_conversations_header_text}\n${multiple_conversations_text}")
set(stateful_npc_full_text "${stateful_npc_header_text}\n${stateful_npc_text}")
set(local_knowledge_full_text "${local_knowledge_header_text}\n${local_knowledge_text}")

foreach(required_project_text
    "\"EngineAssociation\": \"5.7\""
    "\"Name\": \"AstralRT\""
    "\"Enabled\": true")
  if(NOT uproject_text MATCHES "${required_project_text}")
    message(FATAL_ERROR "Unreal sample uproject missing ${required_project_text}")
  endif()
endforeach()

foreach(required_stateful_npc_text
    "UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))"
    "class ASTRALSAMPLE_API UAstralStatefulNpcComponent"
    "FAstralOperationResult LastOperation"
    "CreateToolsetResult"
    "CreateAgentResult"
    "SetAgentSystemPromptResult"
    "SetAgentSummaryResult"
    "SetAgentMemoryContextResult"
    "EnqueueAgentChatResult"
    "CreateAgentChatRequestResult"
    "ReadAgentChatResult"
    "GetAgentChatToolCallResultStatus"
    "SaveAgentHistoryResult"
    "LoadAgentHistoryResult"
    "ReleaseAgentSlotResult"
    "CancelAgentChatResult"
    "DestroyAgent"
    "DestroyToolset"
    "ProjectSavedDir"
    "EndPlay")
  string(FIND "${stateful_npc_full_text}" "${required_stateful_npc_text}" required_stateful_npc_text_pos)
  if(required_stateful_npc_text_pos EQUAL -1)
    message(FATAL_ERROR "Unreal stateful NPC component is missing ${required_stateful_npc_text}")
  endif()
endforeach()

foreach(required_local_knowledge_text
    "UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))"
    "class ASTRALSAMPLE_API UAstralLocalKnowledgeComponent"
    "TObjectPtr<UAstralEmbedder> Embedder"
    "FAstralOperationResult LastOperation"
    "ChunkText"
    "CopyChunkTextResult"
    "MakeMemoryRecordFromChunkResult"
    "AddMemoryBatchResult"
    "SearchMemoryIndexResult"
    "SaveMemoryIndexResult"
    "LoadMemoryIndexResult"
    "DestroyMemoryIndex"
    "EmbedText"
    "ProjectSavedDir"
    "embeddings-capable model"
    "EndPlay")
  string(FIND "${local_knowledge_full_text}" "${required_local_knowledge_text}" required_local_knowledge_text_pos)
  if(required_local_knowledge_text_pos EQUAL -1)
    message(FATAL_ERROR "Unreal local knowledge component is missing ${required_local_knowledge_text}")
  endif()
endforeach()

foreach(required_streaming_chat_text
    "UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))"
    "class ASTRALSAMPLE_API UAstralStreamingChatComponent"
    "TObjectPtr<UAstralModel> Model"
    "TObjectPtr<UAstralSession> Session"
    "FAstralRequestRef ActiveRequest"
    "OnStreamBytesNative().AddUObject"
    "CreateSessionRequestResult"
    "CancelRequestResult"
    "SetSystemPrompt"
    "FeedPrompt"
    "GetStats"
    "EndPlay")
  string(FIND "${streaming_chat_full_text}" "${required_streaming_chat_text}" required_streaming_chat_text_pos)
  if(required_streaming_chat_text_pos EQUAL -1)
    message(FATAL_ERROR "Unreal streaming chat component is missing ${required_streaming_chat_text}")
  endif()
endforeach()

foreach(required_multiple_conversations_text
    "UCLASS(ClassGroup = (Astral), BlueprintType, meta = (BlueprintSpawnableComponent))"
    "class ASTRALSAMPLE_API UAstralMultipleConversationsComponent"
    "TObjectPtr<UAstralModel> Model"
    "TArray<TObjectPtr<UAstralConversation>> Conversations"
    "TArray<FAstralRequestRef> Requests"
    "TArray<int32> LastErrorCodes"
    "ConfigureExecutor"
    "bAutoPumpStream = false"
    "CreateConversationRequestResult"
    "StreamRead(StreamBuffers[Index], 0)"
    "ResetConversation"
    "CancelConversation"
    "GetStats"
    "AbortPartialRun"
    "EndPlay")
  string(FIND "${multiple_conversations_full_text}" "${required_multiple_conversations_text}" required_multiple_conversations_text_pos)
  if(required_multiple_conversations_text_pos EQUAL -1)
    message(FATAL_ERROR "Unreal multiple-conversation component is missing ${required_multiple_conversations_text}")
  endif()
endforeach()

if(NOT build_text MATCHES "\"AstralRT\"")
  message(FATAL_ERROR "Unreal sample module must depend on AstralRT")
endif()
foreach(target_source IN ITEMS "${target_text}" "${editor_target_text}")
  if(NOT target_source MATCHES "DefaultBuildSettings = BuildSettingsVersion[.]Latest")
    message(FATAL_ERROR "Unreal sample targets must use latest build settings")
  endif()
  if(NOT target_source MATCHES "IncludeOrderVersion = EngineIncludeOrderVersion[.]Latest")
    message(FATAL_ERROR "Unreal sample targets must use version-appropriate include order")
  endif()
endforeach()

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
    "UAstralStatefulNpcComponent"
    "UAstralLocalKnowledgeComponent"
    "CreateDefaultSubobject<UAstralStatefulNpcComponent>"
    "CreateDefaultSubobject<UAstralLocalKnowledgeComponent>"
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
    "FString MediaPath"
    "EAstralUnrealPathRoot MediaPathRoot = EAstralUnrealPathRoot::Raw"
    "AstralSampleParsePathRoot"
    "FParse::Value(CommandLine, TEXT(\"AstralMediaPath=\"), OverrideValue)"
    "FParse::Value(CommandLine, TEXT(\"AstralMediaPathRoot=\"), OverrideValue)"
    "ModelDesc.ModelPath = ModelPath;"
    "MediaDesc.MediaPath = MediaPath;"
    "MediaDesc.MediaPathRoot = MediaPathRoot;"
    "MediaModel->InitMedia(MediaDesc)"
    "Astral sample: media projector initialized"
    "UAstralMediaLibrary::MakeRGBA8ImageFromBytes"
    "UAstralMediaLibrary::MakeRGBA8ImageFromTexture"
    "UTexture2D::CreateTransient"
    "PF_B8G8R8A8"
    "MediaSession->FeedImage(TextureImage, false)"
    "MediaSession->FeedAudio"
    "astral_last_error"
    "LogAstralSample")
  string(FIND "${actor_full_text}" "${required_actor_text}" required_actor_text_pos)
  if(required_actor_text_pos EQUAL -1)
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
    "Stateful Npc"
    "Local Knowledge"
    "embeddings-capable model"
    "expected error logging")
  if(NOT readme_text MATCHES "${required_readme_text}")
    message(FATAL_ERROR "Unreal sample README is missing ${required_readme_text}")
  endif()
endforeach()

message(STATUS "gate_unreal_sample_scaffold: OK")
