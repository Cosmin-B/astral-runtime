if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()

set(package_dir "${ASTRAL_SOURCE_DIR}/plugins/unity")
set(package_json "${package_dir}/package.json")
set(package_readme "${package_dir}/README.md")
set(integration_doc "${ASTRAL_SOURCE_DIR}/docs/integration/UNITY_INTEGRATION.md")
if(NOT EXISTS "${package_json}")
  message(FATAL_ERROR "Unity package manifest not found: ${package_json}")
endif()

file(READ "${package_json}" package_text)
string(JSON sample_count ERROR_VARIABLE json_error LENGTH "${package_text}" samples)
if(json_error)
  message(FATAL_ERROR "Unity package samples are not valid JSON: ${json_error}")
endif()

set(declared_paths)
if(sample_count GREATER 0)
  math(EXPR last_sample "${sample_count} - 1")
  foreach(sample_index RANGE 0 ${last_sample})
    string(JSON sample_path GET "${package_text}" samples ${sample_index} path)
    list(APPEND declared_paths "${sample_path}")
  endforeach()
endif()

set(workflows
  StreamingChat
  StatefulNpc
  LocalKnowledge
  CharacterVariants
  MultimodalInput
  MultipleConversations
)

list(LENGTH workflows expected_sample_count)
if(NOT sample_count EQUAL expected_sample_count)
  message(FATAL_ERROR
    "Unity package must declare ${expected_sample_count} workflow samples; found ${sample_count}")
endif()

foreach(workflow IN LISTS workflows)
  set(relative_path "Samples~/${workflow}")
  set(sample_dir "${package_dir}/${relative_path}")
  set(sample_source "${sample_dir}/${workflow}Example.cs")
  set(sample_readme "${sample_dir}/README.md")

  list(FIND declared_paths "${relative_path}" declared_path_index)
  if(declared_path_index EQUAL -1)
    message(FATAL_ERROR "Unity package does not declare ${relative_path}")
  endif()
  if(NOT EXISTS "${sample_source}")
    message(FATAL_ERROR "Unity workflow source not found: ${sample_source}")
  endif()
  if(NOT EXISTS "${sample_readme}")
    message(FATAL_ERROR "Unity workflow README not found: ${sample_readme}")
  endif()

  file(READ "${sample_source}" source_text)
  if(NOT source_text MATCHES "namespace Astral[.]Examples")
    message(FATAL_ERROR "${sample_source} must use namespace Astral.Examples")
  endif()
  if(NOT source_text MATCHES "class ${workflow}Example[ \t\r\n]*:[ \t\r\n]*MonoBehaviour")
    message(FATAL_ERROR "${sample_source} must define ${workflow}Example as a MonoBehaviour")
  endif()

  file(GLOB_RECURSE native_artifacts
    "${sample_dir}/*.dll"
    "${sample_dir}/*.so"
    "${sample_dir}/*.dylib"
    "${sample_dir}/*.a"
    "${sample_dir}/*.lib"
  )
  if(native_artifacts)
    message(FATAL_ERROR "Unity samples must not contain native binaries: ${native_artifacts}")
  endif()
endforeach()

file(READ "${package_readme}" package_readme_text)
file(READ "${integration_doc}" integration_doc_text)
foreach(workflow IN LISTS workflows)
  foreach(document_text IN ITEMS package_readme_text integration_doc_text)
    string(FIND "${${document_text}}" "Samples~/${workflow}/README.md" workflow_link_index)
    if(workflow_link_index EQUAL -1)
      message(FATAL_ERROR "${document_text} does not link the ${workflow} sample")
    endif()
  endforeach()
endforeach()

if(EXISTS "${package_dir}/Runtime/AstralExample.cs")
  message(FATAL_ERROR "Unity example behavior must live under Samples~, not Runtime")
endif()

function(require_workflow_tokens workflow)
  set(sample_source "${package_dir}/Samples~/${workflow}/${workflow}Example.cs")
  file(READ "${sample_source}" source_text)
  foreach(required_token IN LISTS ARGN)
    string(FIND "${source_text}" "${required_token}" token_index)
    if(token_index EQUAL -1)
      message(FATAL_ERROR "${workflow} sample is missing ${required_token}")
    endif()
  endforeach()
endfunction()

require_workflow_tokens(StreamingChat
  "NativeArray<byte>"
  "ReadStream"
  "AstralRequest.FromSession"
  "AstralRequest.TryCancel"
  "ASTRAL_E_TIMEOUT"
  "GetStats"
  "DisposeRequest"
)
require_workflow_tokens(MultipleConversations
  "ConfigureExecutor"
  "AstralConversation.Create"
  "AstralRequest.FromConversation"
  "NonBlockingTimeoutMs"
  "GetStats"
  "DisposeActive"
)
require_workflow_tokens(StatefulNpc
  "AstralAgent.Create"
  "AstralToolset.Create"
  "AstralRequest.FromAgentChat"
  "SaveHistory"
  "LoadHistory"
  "SetSummary"
  "SetMemoryContext"
  "ReleaseSlot"
  "GetChatToolCallResult"
)
require_workflow_tokens(LocalKnowledge
  "AstralChunker.Ranges"
  "AstralEmbedder.Create"
  "AddBatch"
  "Search"
  "Save"
  "AstralMemoryIndex.Load"
  "CopyTextToString"
)
require_workflow_tokens(CharacterVariants
  "AstralPromptCache.KeyFromString"
  "PutTokens"
  "GetTokens"
  "AstralPromptCache.Load"
  "LoadAdapter"
  "AddAdapter"
  "SetAdapterScale"
  "StopAdd"
  "SetGrammarJsonSchema"
  "CountTokens"
)
require_workflow_tokens(MultimodalInput
  "InitMediaFromPath"
  "GetCaps"
  "ASTRAL_CAP_IMAGE"
  "ASTRAL_CAP_AUDIO"
  "ASTRAL_CAP_MM_EMBEDDINGS"
  "GetRawTextureData<byte>"
  "FeedImage"
  "FeedAudio"
  "EnqueueMultimodal"
  "AstralRequest.FromEmbedding"
)

set(unity_runner "${ASTRAL_SOURCE_DIR}/scripts/run_unity_ci_tests.sh")
file(READ "${unity_runner}" unity_runner_text)
foreach(required_runner_token IN ITEMS
    "AstralWorkflowSamples"
    "plugins/unity/Samples~"
    "trap cleanup_samples EXIT")
  string(FIND "${unity_runner_text}" "${required_runner_token}" runner_token_index)
  if(runner_token_index EQUAL -1)
    message(FATAL_ERROR "Unity runner does not stage workflow samples: ${required_runner_token}")
  endif()
endforeach()

message(STATUS "gate_unity_samples: OK")
