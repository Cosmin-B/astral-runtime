if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()

set(plugin_dir "${ASTRAL_SOURCE_DIR}/plugins/unreal/AstralRT/Source/AstralRT")
set(types_header "${plugin_dir}/Public/AstralTypes.h")
set(model_header "${plugin_dir}/Public/AstralModel.h")
set(model_source "${plugin_dir}/Private/AstralModel.cpp")
set(blueprint_header "${plugin_dir}/Public/AstralBlueprintLibrary.h")
set(blueprint_source "${plugin_dir}/Private/AstralBlueprintLibrary.cpp")
set(conv_header "${plugin_dir}/Public/AstralConversation.h")
set(conv_source "${plugin_dir}/Private/AstralConversation.cpp")
set(session_header "${plugin_dir}/Public/AstralSession.h")
set(session_source "${plugin_dir}/Private/AstralSession.cpp")
set(test_source "${plugin_dir}/Private/Tests/AstralRTTests.cpp")

foreach(required_file IN ITEMS
    "${types_header}"
    "${model_header}"
    "${model_source}"
    "${blueprint_header}"
    "${blueprint_source}"
    "${conv_header}"
    "${conv_source}"
    "${session_header}"
    "${session_source}"
    "${test_source}")
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR "Unreal conversation wrapper file missing: ${required_file}")
  endif()
endforeach()

file(READ "${session_header}" session_header_text)
file(READ "${session_source}" session_source_text)
foreach(required_session_grammar_text IN ITEMS
    "SetGrammarGbnf"
    "SetGrammarJsonSchema"
    "ClearGrammar")
  string(FIND "${session_header_text}" "${required_session_grammar_text}" grammar_index)
  if(grammar_index EQUAL -1)
    message(FATAL_ERROR "Unreal session wrapper is missing ${required_session_grammar_text}")
  endif()
endforeach()
foreach(required_session_grammar_native_text IN ITEMS
    "astral_session_set_grammar_gbnf"
    "astral_session_set_grammar_json_schema"
    "astral_session_clear_grammar")
  string(FIND "${session_source_text}" "${required_session_grammar_native_text}" grammar_native_index)
  if(grammar_native_index EQUAL -1)
    message(FATAL_ERROR "Unreal session wrapper is missing ${required_session_grammar_native_text}")
  endif()
endforeach()

file(READ "${blueprint_header}" blueprint_header_text)
file(READ "${blueprint_source}" blueprint_source_text)
foreach(required_text IN ITEMS "${blueprint_header_text}" "${blueprint_source_text}")
  string(FIND "${required_text}" "CreateAstralConversation" factory_index)
  if(factory_index EQUAL -1)
    message(FATAL_ERROR "The Unreal Blueprint library is missing CreateAstralConversation")
  endif()
endforeach()

file(READ "${types_header}" types_text)
foreach(required_token IN ITEMS
    "FAstralExecutorDesc"
    "FAstralConversationDesc"
    "bAutoPumpStream"
    "FAstralConversationStats")
  string(FIND "${types_text}" "${required_token}" token_index)
  if(token_index EQUAL -1)
    message(FATAL_ERROR "AstralTypes.h is missing ${required_token}")
  endif()
endforeach()

file(READ "${model_header}" model_header_text)
file(READ "${model_source}" model_source_text)
foreach(required_token IN ITEMS
    "ConfigureExecutor"
    "FAstralExecutorDesc")
  string(FIND "${model_header_text}" "${required_token}" token_index)
  if(token_index EQUAL -1)
    message(FATAL_ERROR "AstralModel.h is missing ${required_token}")
  endif()
endforeach()
string(FIND "${model_source_text}" "astral_model_executor_configure" configure_index)
if(configure_index EQUAL -1)
  message(FATAL_ERROR "AstralModel.cpp does not configure the native executor")
endif()

file(READ "${conv_header}" conv_header_text)
foreach(required_token IN ITEMS
    "class ASTRALRT_API UAstralConversation"
    "bool Create"
    "bool FeedPrompt"
    "bool SetSystemPrompt"
    "bool Decode"
    "bool Cancel"
    "int32 Wait"
    "bool Reset"
    "int32 StreamRead"
    "FAstralConversationStats GetStats"
    "OnStreamBytesNative"
    "BeginDestroy")
  string(FIND "${conv_header_text}" "${required_token}" token_index)
  if(token_index EQUAL -1)
    message(FATAL_ERROR "AstralConversation.h is missing ${required_token}")
  endif()
endforeach()

file(READ "${conv_source}" conv_source_text)
foreach(required_token IN ITEMS
    "astral_conv_create"
    "astral_conv_destroy"
    "astral_conv_feed"
    "astral_conv_decode"
    "astral_conv_cancel"
    "astral_conv_wait"
    "astral_conv_reset"
    "astral_conv_stream_read"
    "astral_conv_stats"
    "GetRuntimeGeneration"
    "FAstralSessionStreamPump::Tick")
  string(FIND "${conv_source_text}" "${required_token}" token_index)
  if(token_index EQUAL -1)
    message(FATAL_ERROR "AstralConversation.cpp is missing ${required_token}")
  endif()
endforeach()

file(READ "${test_source}" test_text)
foreach(required_token IN ITEMS
    "AstralRT.Mock.ConversationWrapper"
    "UAstralConversation"
    "CreateConversationRequestResult")
  string(FIND "${test_text}" "${required_token}" token_index)
  if(token_index EQUAL -1)
    message(FATAL_ERROR "Unreal Automation coverage is missing ${required_token}")
  endif()
endforeach()

message(STATUS "gate_unreal_conversation_wrapper: OK")
