if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BASH_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_BASH_EXECUTABLE not set")
endif()

set(out_dir "${ASTRAL_BUILD_DIR}/unreal-sample-package-gate")
set(project_dir "${out_dir}/AstralSample")
set(archive_dir "${out_dir}/archive")
set(matrix_runner "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_small_model_matrix.sh")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_sample_package.sh"
    --out "${project_dir}"
    --archive-dir "${archive_dir}"
    --skip-native-build
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_result
  OUTPUT_VARIABLE missing_output
  ERROR_VARIABLE missing_error
)
if(missing_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_sample_package.sh accepted a missing RunUAT path")
endif()
set(missing_text "${missing_output}\n${missing_error}")
if(NOT missing_text MATCHES "Missing Unreal RunUAT path")
  message(FATAL_ERROR "run_unreal_sample_package.sh failed without the RunUAT setup message: ${missing_text}")
endif()

set(fake_runuat "${out_dir}/RunUAT.sh")
file(WRITE "${fake_runuat}" [=[
#!/usr/bin/env bash
echo "fake RunUAT: $*"
archive_dir=""
for arg in "$@"; do
  case "$arg" in
    -archivedirectory=*) archive_dir="${arg#-archivedirectory=}" ;;
  esac
done
case " $* " in
  *" BuildCookRun "*" -platform=Linux "*" -archive "*)
    if [[ -n "${archive_dir}" ]]; then
      mkdir -p "${archive_dir}/Linux"
      cat >"${archive_dir}/Linux/AstralSample.sh" <<'SH'
#!/usr/bin/env bash
echo "fake sample: $*"
for token in \
  "-AstralSampleAutoQuit" \
  "-AstralBackend=cpu" \
  "-AstralMemoryBackend=mock" \
  "-AstralMediaBackend=mock" \
  "-AstralModel=/tmp/text.gguf" \
  "-AstralEmbeddingModel=/tmp/embed.gguf" \
  "-AstralMediaPath=/tmp/mmproj.gguf" \
  "-AstralMediaPathRoot=Raw"; do
  case " $* " in
    *" ${token} "*) ;;
    *) echo "missing runtime arg: ${token}" >&2; exit 43 ;;
  esac
done
SH
      chmod +x "${archive_dir}/Linux/AstralSample.sh"
    fi
    exit 0
    ;;
esac
echo "missing BuildCookRun package args" >&2
exit 42
]=])
file(CHMOD "${fake_runuat}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_sample_package.sh"
    --out "${project_dir}"
    --archive-dir "${archive_dir}"
    --runuat "${fake_runuat}"
    --skip-native-build
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE package_result
  OUTPUT_VARIABLE package_output
  ERROR_VARIABLE package_error
)
if(NOT package_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_sample_package.sh failed fake RunUAT package smoke: ${package_error}")
endif()
set(package_text "${package_output}\n${package_error}")
foreach(required_text
    "[unreal_sample] Project:"
    "[unreal_sample] Archive:"
    "[unreal_sample] RunUAT:"
    "[unreal_sample] Platform: Linux"
    "[unreal_sample] Plugin mode: copy"
    "[unreal_sample] BuildCookRun"
    "fake RunUAT: BuildCookRun"
    "-archive"
    "[unreal_sample] OK:")
  string(FIND "${package_text}" "${required_text}" required_text_pos)
  if(required_text_pos EQUAL -1)
    message(FATAL_ERROR "run_unreal_sample_package.sh output missing '${required_text}': ${package_text}")
  endif()
endforeach()

set(runtime_log "${out_dir}/runtime.log")
execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${ASTRAL_SOURCE_DIR}/scripts/run_unreal_sample_package.sh"
    --out "${project_dir}"
    --archive-dir "${archive_dir}"
    --runuat "${fake_runuat}"
    --skip-native-build
    --run-sample
    --runtime-log "${runtime_log}"
    --sample-model "/tmp/text.gguf"
    --sample-embedding-model "/tmp/embed.gguf"
    --sample-media-path "/tmp/mmproj.gguf"
    --sample-media-path-root "Raw"
    --sample-prompt "hello from gate"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE runtime_result
  OUTPUT_VARIABLE runtime_output
  ERROR_VARIABLE runtime_error
)
if(NOT runtime_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_sample_package.sh failed fake runtime smoke: ${runtime_error}")
endif()
set(runtime_text "${runtime_output}\n${runtime_error}")
foreach(required_runtime_text
    "[unreal_sample] Runtime:"
    "[unreal_sample] Runtime backend: cpu"
    "[unreal_sample] Runtime memory backend: mock"
    "[unreal_sample] Runtime media backend: mock"
    "[unreal_sample] Runtime media path: /tmp/mmproj.gguf"
    "[unreal_sample] Runtime media path root: Raw"
    "[unreal_sample] Runtime log:"
    "fake sample:"
    "-AstralBackend=cpu"
    "-AstralMemoryBackend=mock"
    "-AstralMediaBackend=mock"
    "-AstralModel=/tmp/text.gguf"
    "-AstralEmbeddingModel=/tmp/embed.gguf"
    "-AstralMediaPath=/tmp/mmproj.gguf"
    "-AstralMediaPathRoot=Raw"
    "[unreal_sample] Runtime OK")
  string(FIND "${runtime_text}" "${required_runtime_text}" required_runtime_text_pos)
  if(required_runtime_text_pos EQUAL -1)
    message(FATAL_ERROR "run_unreal_sample_package.sh runtime output missing '${required_runtime_text}': ${runtime_text}")
  endif()
endforeach()
if(NOT EXISTS "${runtime_log}")
  message(FATAL_ERROR "sample package runner did not write ${runtime_log}")
endif()

set(matrix_models_dir "${out_dir}/models")
file(MAKE_DIRECTORY "${matrix_models_dir}")
set(matrix_text_model "${matrix_models_dir}/Qwen3-0.6B-Q8_0.gguf")
set(matrix_alt_model "${matrix_models_dir}/gemma-3-270m-q4_k_m.gguf")
set(matrix_embed_model "${matrix_models_dir}/Qwen3-Embedding-0.6B-Q8_0.gguf")
file(WRITE "${matrix_text_model}" "tiny")
file(WRITE "${matrix_alt_model}" "tiny")
file(WRITE "${matrix_embed_model}" "tiny")

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${matrix_runner}"
    --models-dir "${matrix_models_dir}"
    --list
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE matrix_list_result
  OUTPUT_VARIABLE matrix_list_output
  ERROR_VARIABLE matrix_list_error
)
if(NOT matrix_list_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_small_model_matrix.sh rejected list mode with fake fixtures: ${matrix_list_error}")
endif()
foreach(required_matrix_list_text
    "Qwen3-0.6B-Q8_0.gguf"
    "gemma-3-270m-q4_k_m.gguf"
    "embedding_model: ${matrix_embed_model}")
  string(FIND "${matrix_list_output}" "${required_matrix_list_text}" required_matrix_list_pos)
  if(required_matrix_list_pos EQUAL -1)
    message(FATAL_ERROR "small-model matrix list output missing '${required_matrix_list_text}': ${matrix_list_output}")
  endif()
endforeach()

execute_process(
  COMMAND "${ASTRAL_BASH_EXECUTABLE}" "${matrix_runner}"
    --models-dir "${matrix_models_dir}"
    --preset qwen3-0.6b-q8
    --out "${out_dir}/matrix"
    --runuat "${fake_runuat}"
    --skip-native-build
    --dry-run
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE matrix_dry_run_result
  OUTPUT_VARIABLE matrix_dry_run_output
  ERROR_VARIABLE matrix_dry_run_error
)
if(NOT matrix_dry_run_result EQUAL 0)
  message(FATAL_ERROR "run_unreal_small_model_matrix.sh rejected dry-run mode: ${matrix_dry_run_error}")
endif()
foreach(required_matrix_dry_run_text
    "[unreal_small_matrix] run:"
    "run_unreal_sample_package.sh"
    "--run-sample"
    "--sample-model"
    "Qwen3-0.6B-Q8_0.gguf"
    "--sample-embedding-model"
    "Qwen3-Embedding-0.6B-Q8_0.gguf"
    "[unreal_small_matrix] validate:"
    "validate_unreal_sample_runtime_log.py"
    "--expect-model"
    "--expect-embedding-model"
    "--skip-native-build"
    "[unreal_small_matrix] OK")
  string(FIND "${matrix_dry_run_output}" "${required_matrix_dry_run_text}" required_matrix_dry_run_pos)
  if(required_matrix_dry_run_pos EQUAL -1)
    message(FATAL_ERROR "small-model matrix dry-run output missing '${required_matrix_dry_run_text}': ${matrix_dry_run_output}")
  endif()
endforeach()

foreach(required_file
    "${project_dir}/AstralSample.uproject"
    "${project_dir}/Plugins/AstralRT/AstralRT.uplugin"
    "${project_dir}/Source/AstralSample/AstralSampleActor.h"
    "${project_dir}/Source/AstralSample/AstralSampleActor.cpp")
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR "sample package runner did not create ${required_file}")
  endif()
endforeach()

set(sample_actor_header "${project_dir}/Source/AstralSample/AstralSampleActor.h")
set(sample_actor_cpp "${project_dir}/Source/AstralSample/AstralSampleActor.cpp")
file(READ "${sample_actor_header}" sample_actor_header_text)
file(READ "${sample_actor_cpp}" sample_actor_cpp_text)

if(sample_actor_header_text MATCHES "UFUNCTION\\([^\\n]+\\)\\n    UFUNCTION")
  message(FATAL_ERROR "generated AstralSampleActor.h has duplicate adjacent UFUNCTION markers")
endif()
if(sample_actor_cpp_text MATCHES "void AAstralSampleActor::RunErrorDemo\\(\\)\\n\\{\\n\\{")
  message(FATAL_ERROR "generated RunErrorDemo has an extra opening brace")
endif()
foreach(required_sample_text
    "FString MemoryBackendName = TEXT(\"mock\");"
    "FString EmbeddingModelPath;"
    "FString MediaPath"
    "EAstralUnrealPathRoot MediaPathRoot = EAstralUnrealPathRoot::Raw"
    "void ApplyCommandLineOverrides();"
    "AstralSampleParsePathRoot"
    "FParse::Value(CommandLine, TEXT(\"AstralBackend=\"), OverrideValue)"
    "FParse::Value(CommandLine, TEXT(\"AstralMemoryBackend=\"), OverrideValue)"
    "FParse::Value(CommandLine, TEXT(\"AstralMediaBackend=\"), OverrideValue)"
    "FParse::Value(CommandLine, TEXT(\"AstralModel=\"), OverrideValue)"
    "FParse::Value(CommandLine, TEXT(\"AstralEmbeddingModel=\"), OverrideValue)"
    "FParse::Value(CommandLine, TEXT(\"AstralMediaPath=\"), OverrideValue)"
    "FParse::Value(CommandLine, TEXT(\"AstralMediaPathRoot=\"), OverrideValue)"
    "FParse::Value(CommandLine, TEXT(\"AstralPrompt=\"), OverrideValue)"
    "memory_backend=%s"
    "media_backend=%s"
    "media_path=%s"
    "media_path_root=%s"
    "UE_LOG(LogAstralSample, Display"
    "Desc.BackendName = MemoryBackendName;"
    "void RunMediaFeedDemo();"
    "ModelDesc.ModelPath = ModelPath;"
    "MediaDesc.MediaPath = MediaPath;"
    "MediaDesc.MediaPathRoot = MediaPathRoot;"
    "MediaModel->InitMedia(MediaDesc)"
    "Astral sample: media projector initialized"
    "UAstralMediaLibrary::MakeRGBA8ImageFromBytes"
    "UAstralMediaLibrary::MakeRGBA8ImageFromTexture"
    "UTexture2D::CreateTransient"
    "PF_B8G8R8A8"
    "UAstralMediaLibrary::MakePCM16AudioFromBytes"
    "MediaSession->FeedImage(Image, false)"
    "MediaSession->FeedImage(TextureImage, false)"
    "MediaSession->FeedAudio(Audio, true)"
    "Astral sample: media feed demo loaded"
    "ApplyCommandLineOverrides();"
    "Desc.ModelPath = ModelPath;"
    "Desc.ModelPath = EmbeddingModelPath.IsEmpty() ? ModelPath : EmbeddingModelPath;"
    "void RunPackagedMemorySourceDemo();"
    "void AAstralSampleActor::RunErrorDemo()"
    "Astral sample: expected load failure:")
  string(FIND "${sample_actor_header_text}\n${sample_actor_cpp_text}" "${required_sample_text}" sample_text_pos)
  if(sample_text_pos EQUAL -1)
    message(FATAL_ERROR "generated Astral sample source is missing '${required_sample_text}'")
  endif()
endforeach()

message(STATUS "gate_unreal_sample_package_runner: OK")
