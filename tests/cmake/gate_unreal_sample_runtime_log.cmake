if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()
if(NOT DEFINED ASTRAL_BUILD_DIR)
  message(FATAL_ERROR "ASTRAL_BUILD_DIR not set")
endif()
if(NOT DEFINED ASTRAL_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ASTRAL_PYTHON_EXECUTABLE not set")
endif()

set(script "${ASTRAL_SOURCE_DIR}/scripts/validate_unreal_sample_runtime_log.py")
set(out_dir "${ASTRAL_BUILD_DIR}/unreal-sample-runtime-log-gate")
file(REMOVE_RECURSE "${out_dir}")
file(MAKE_DIRECTORY "${out_dir}")

set(good_log "${out_dir}/runtime-good.log")
file(WRITE "${good_log}"
"LogCsvProfiler: Display: Metadata set : engineversion=\"5.7.4-51494982+UE5\"
LogCsvProfiler: Display: Metadata set : commandline=\"\" AstralSample -NullRHI -Unattended -NoSplash -NoSound -AstralSampleAutoQuit -log -stdout -AstralBackend=cpu -AstralMemoryBackend=mock -AstralMediaBackend=mock -AstralModel=/workspace/astral/tests/models/Qwen3-0.6B-Q8_0.gguf -AstralEmbeddingModel=/workspace/astral/tests/models/Qwen3-Embedding-0.6B-Q8_0.gguf \"\"
LogPakFile: Display: Mounted IoStore container \"../../../AstralSample/Content/Paks/AstralSample-Linux.utoc\"
LogPakFile: Display: Mounted Pak file '../../../AstralSample/Content/Paks/AstralSample-Linux.pak', mount point: '../../../'
LogAstralSample: Display: Astral sample: backend=cpu memory_backend=mock media_backend=mock model=/workspace/astral/tests/models/Qwen3-0.6B-Q8_0.gguf embedding_model=/workspace/astral/tests/models/Qwen3-Embedding-0.6B-Q8_0.gguf media_path=<none> media_path_root=Raw
LogAstralSample: Display: Astral sample: canceled stream wait result -4
LogAstralSample: Display: Astral sample: embedding dimension 1024
LogAstralSample: Display: Astral sample: media feed demo loaded mock backend with RGBA byte image, texture image, and PCM16 audio
LogAstralSample: Display: Astral sample: packaged content bytes read from ../../../AstralSample/Content/AstralSample/Models/mock-model.bytes
LogAstralSample: Display: Astral sample: packaged content memory model loaded from 4 bytes
LogAstralSample: Display: Astral sample: saved cache bytes read from ../../../AstralSample/Saved/AstralSample/mock-model-cache.bytes
LogAstralSample: Display: Astral sample: saved cache memory model loaded from 4 bytes
")

execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}"
    --log "${good_log}"
    --expect-engine-version "5.7.4"
    --expect-model "Qwen3-0.6B-Q8_0.gguf"
    --expect-embedding-model "Qwen3-Embedding-0.6B-Q8_0.gguf"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE good_result
  OUTPUT_VARIABLE good_output
  ERROR_VARIABLE good_error
)
if(NOT good_result EQUAL 0)
  message(FATAL_ERROR "validate_unreal_sample_runtime_log.py rejected valid runtime log: ${good_output}${good_error}")
endif()

set(missing_model_log "${out_dir}/runtime-missing-model.log")
file(WRITE "${missing_model_log}"
"Command Line: -AstralSampleAutoQuit
LogPakFile: Display: Mounted IoStore container \"../../../AstralSample/Content/Paks/AstralSample-Linux.utoc\"
LogPakFile: Display: Mounted Pak file '../../../AstralSample/Content/Paks/AstralSample-Linux.pak'
LogAstralSample: Display: Astral sample: media feed demo loaded mock backend with RGBA byte image, texture image, and PCM16 audio
LogAstralSample: Display: Astral sample: packaged content bytes read from ../../../AstralSample/Content/AstralSample/Models/mock-model.bytes
LogAstralSample: Display: Astral sample: packaged content memory model loaded from 4 bytes
LogAstralSample: Display: Astral sample: saved cache bytes read from ../../../AstralSample/Saved/AstralSample/mock-model-cache.bytes
LogAstralSample: Display: Astral sample: saved cache memory model loaded from 4 bytes
")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}"
    --log "${missing_model_log}"
    --require-real-model
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE missing_model_result
  ERROR_VARIABLE missing_model_error
)
if(missing_model_result EQUAL 0)
  message(FATAL_ERROR "validate_unreal_sample_runtime_log.py accepted a runtime log without real model evidence")
endif()
if(NOT missing_model_error MATCHES "backend=cpu")
  message(FATAL_ERROR "validate_unreal_sample_runtime_log.py failed for the wrong missing-model reason: ${missing_model_error}")
endif()

set(failure_log "${out_dir}/runtime-failure.log")
file(READ "${good_log}" good_log_text)
file(WRITE "${failure_log}" "${good_log_text}
LogAstralSample: Error: Astral sample: media feed demo failed
")
execute_process(
  COMMAND "${ASTRAL_PYTHON_EXECUTABLE}" "${script}" --log "${failure_log}"
  WORKING_DIRECTORY "${ASTRAL_SOURCE_DIR}"
  RESULT_VARIABLE failure_result
  ERROR_VARIABLE failure_error
)
if(failure_result EQUAL 0)
  message(FATAL_ERROR "validate_unreal_sample_runtime_log.py accepted a failure marker")
endif()
if(NOT failure_error MATCHES "failure marker")
  message(FATAL_ERROR "validate_unreal_sample_runtime_log.py failed for the wrong failure-marker reason: ${failure_error}")
endif()

message(STATUS "gate_unreal_sample_runtime_log: OK")
