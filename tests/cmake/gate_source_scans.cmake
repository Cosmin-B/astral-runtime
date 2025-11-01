if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()

set(ROOT "${ASTRAL_SOURCE_DIR}")

set(SCAN_DIRS
  include
  src
  tests
  benchmarks
  cmake
  docs
  plugins
  scripts
)

set(SCAN_GLOBS
  "*.h"
  "*.hpp"
  "*.c"
  "*.cc"
  "*.cpp"
  "*.cs"
  "*.md"
  "*.txt"
  "*.sh"
  "*.py"
  "*.cmake"
  "CMakeLists.txt"
  "*.json"
  "*.asmdef"
)

set(FILES)
foreach(dir IN LISTS SCAN_DIRS)
  if(EXISTS "${ROOT}/${dir}")
    foreach(glob IN LISTS SCAN_GLOBS)
      file(GLOB_RECURSE dir_files "${ROOT}/${dir}/${glob}")
      list(APPEND FILES ${dir_files})
    endforeach()
  endif()
endforeach()

foreach(glob IN LISTS SCAN_GLOBS)
  file(GLOB root_files "${ROOT}/${glob}")
  list(APPEND FILES ${root_files})
endforeach()

list(REMOVE_DUPLICATES FILES)

list(LENGTH FILES FILES_LEN)
if(FILES_LEN EQUAL 0)
  message(FATAL_ERROR "No files found to scan under ${ROOT}")
endif()

set(token_forbidden_1 "compare")
string(APPEND token_forbidden_1 "_")
string(APPEND token_forbidden_1 "exchange")

set(token_forbidden_2 "cmp")
string(APPEND token_forbidden_2 "xchg")

set(token_forbidden_3 "Compare")
string(APPEND token_forbidden_3 "Exchange")

set(FORBIDDEN_STRINGS
  "${token_forbidden_1}"
  "${token_forbidden_2}"
  "${token_forbidden_3}"
)

set(todo_token "TO")
string(APPEND todo_token "DO")
set(fixme_token "FIX")
string(APPEND fixme_token "ME")
set(hack_token "HA")
string(APPEND hack_token "CK")

set(TRACKED_COMMENT_TOKENS
  "${todo_token}"
  "${fixme_token}"
  "${hack_token}"
)

set(ai_generation_phrase "AI")
string(APPEND ai_generation_phrase " generation")
set(automated_tools_phrase "automated")
string(APPEND automated_tools_phrase " tools")
set(proper_cleanup_phrase "proper")
string(APPEND proper_cleanup_phrase " cleanup")
set(proper_lifetime_phrase "proper")
string(APPEND proper_lifetime_phrase " lifetime management")
set(raii_pattern_phrase "RAII")
string(APPEND raii_pattern_phrase " pattern")
set(always_dispose_phrase "Always call Dispose")
string(APPEND always_dispose_phrase "() when done")
set(disposed_properly_phrase "disposed")
string(APPEND disposed_properly_phrase " properly")
set(create_temporary_arrays_phrase "Create")
string(APPEND create_temporary_arrays_phrase " temporary arrays for intermediate results")
set(dispose_temporary_arrays_phrase "Dispose")
string(APPEND dispose_temporary_arrays_phrase " temporary arrays when jobs complete")
set(unreviewed_name_phrase "review")
string(APPEND unreviewed_name_phrase " tooling")
set(handle_error_phrase "handle")
string(APPEND handle_error_phrase " the error")
set(process_data_phrase "process")
string(APPEND process_data_phrase " the data")
set(ensure_proper_phrase "ensure")
string(APPEND ensure_proper_phrase " proper")
set(this_function_phrase "this")
string(APPEND this_function_phrase " function")
set(for_safety_phrase "for")
string(APPEND for_safety_phrase " safety")
set(just_in_case_phrase "just")
string(APPEND just_in_case_phrase " in case")
set(no_assertion_phrase "No")
string(APPEND no_assertion_phrase " assertion")
set(just_verify_phrase "Just")
string(APPEND just_verify_phrase " verify")
set(just_verify_lower_phrase "just")
string(APPEND just_verify_lower_phrase " verify")
set(should_not_crash_phrase "should")
string(APPEND should_not_crash_phrase " not crash")
set(best_effort_basis_phrase "best")
string(APPEND best_effort_basis_phrase "-effort basis")
set(best_effort_may_fail_phrase "best")
string(APPEND best_effort_may_fail_phrase "-effort, may fail")
set(coarse_plots_best_effort_phrase "Coarse plots (best")
string(APPEND coarse_plots_best_effort_phrase "-effort)")
set(init_penalty_best_effort_phrase "Init penalty state from prompt once (best")
string(APPEND init_penalty_best_effort_phrase "-effort)")
set(grammar_constraints_best_effort_phrase "Apply grammar constraints at candidate level (best")
string(APPEND grammar_constraints_best_effort_phrase "-effort)")
set(reference_counted_best_effort_phrase "Initialize llama.cpp")
string(APPEND reference_counted_best_effort_phrase " backend")
set(skip_gracefully_phrase "skip")
string(APPEND skip_gracefully_phrase " gracefully")
set(skips_if_native_missing_phrase "skips if native")
string(APPEND skips_if_native_missing_phrase " lib missing")
set(cuda_machine_best_effort_phrase "CUDA machine tests")
string(APPEND cuda_machine_best_effort_phrase " (manual/CI best-effort)")
set(cuda_bench_best_effort_phrase "best")
string(APPEND cuda_bench_best_effort_phrase "-effort CUDA bench runs")
set(cuda_scaffold_best_effort_phrase "CUDA is a v0.1")
string(APPEND cuda_scaffold_best_effort_phrase " ")
string(APPEND cuda_scaffold_best_effort_phrase "\"best-effort scaffold\"")
set(stale_release_matrix_phrase "Define the release")
string(APPEND stale_release_matrix_phrase " acceptance matrix")
set(unity_high_performance_phrase "High-performance LLM")
string(APPEND unity_high_performance_phrase " inference for Unity")
set(unity_zero_gc_phrase "Zero GC")
string(APPEND unity_zero_gc_phrase " Allocations")
set(unity_profiler_validated_phrase "validated with Unity")
string(APPEND unity_profiler_validated_phrase " Profiler")
set(unity_il2cpp_safe_phrase "IL2CPP")
string(APPEND unity_il2cpp_safe_phrase " safe")
set(unity_raii_pattern_phrase "RAII")
string(APPEND unity_raii_pattern_phrase " Pattern")
set(unity_pinvoke_all_tested_phrase "All P/Invoke")
string(APPEND unity_pinvoke_all_tested_phrase " declarations tested")
set(unity_zero_alloc_streaming_phrase "Zero-Allocation")
string(APPEND unity_zero_alloc_streaming_phrase " Streaming")
set(unity_zero_gc_allocation_phrase "zero GC")
string(APPEND unity_zero_gc_allocation_phrase " allocation")
set(unity_maximum_performance_phrase "maximum")
string(APPEND unity_maximum_performance_phrase " performance")
set(multimodal_optin_best_effort_phrase "opt-in.*best")
string(APPEND multimodal_optin_best_effort_phrase "-effort")
set(model_source_desktop_best_effort_phrase "desktop .best")
string(APPEND model_source_desktop_best_effort_phrase "-effort.")
set(media_feed_best_effort_phrase "media init.*best")
string(APPEND media_feed_best_effort_phrase "-effort")
set(gpu_device_best_effort_phrase "CUDA device index .best")
string(APPEND gpu_device_best_effort_phrase "-effort.")
set(gpu_mask_best_effort_phrase "allowed devices .best")
string(APPEND gpu_mask_best_effort_phrase "-effort.")
set(gpu_fields_best_effort_phrase "gpu_stream.*are best")
string(APPEND gpu_fields_best_effort_phrase "-effort")
set(gpu_settings_best_effort_phrase "settings are best")
string(APPEND gpu_settings_best_effort_phrase "-effort")
set(handle_valid_best_effort_phrase "use-after-free detection is best")
string(APPEND handle_valid_best_effort_phrase "-effort")
set(model_limits_best_effort_phrase "Query model limits .best")
string(APPEND model_limits_best_effort_phrase "-effort")
set(executor_tuning_best_effort_phrase "This is best")
string(APPEND executor_tuning_best_effort_phrase "-effort")
set(unity_cuda_config_best_effort_phrase "CUDA multi-GPU config .best")
string(APPEND unity_cuda_config_best_effort_phrase "-effort.")
set(vm_aligned_best_effort_phrase "requested alignment .best")
string(APPEND vm_aligned_best_effort_phrase "-effort")
set(vm_huge_best_effort_phrase "This is a best")
string(APPEND vm_huge_best_effort_phrase "-effort operation")
set(vm_silently_fail_phrase "may silently")
string(APPEND vm_silently_fail_phrase " fail")
set(vm_silent_fail_comment_phrase "Silently")
string(APPEND vm_silent_fail_comment_phrase " fail")
set(vm_commit_best_effort_phrase "vm_commit.*best")
string(APPEND vm_commit_best_effort_phrase "-effort")
set(alloc_best_effort_phrase "best")
string(APPEND alloc_best_effort_phrase "-effort heap allocation")
set(allocation_portable_best_effort_phrase "allocations.*best")
string(APPEND allocation_portable_best_effort_phrase "-effort portable")
set(io_portable_best_effort_phrase "I/O syscalls.*best")
string(APPEND io_portable_best_effort_phrase "-effort portable")
set(worker_id_best_effort_phrase "worker_id.*best")
string(APPEND worker_id_best_effort_phrase "-effort")
set(slot_reset_best_effort_phrase "session_slot_reset.*best")
string(APPEND slot_reset_best_effort_phrase "-effort")
set(provider_slot_best_effort_phrase "Best")
string(APPEND provider_slot_best_effort_phrase "-effort: clear slot state")
set(memory_source_mmap_best_effort_phrase "ASTRAL_CPU_MEMORY_SOURCE_MMAP.*best")
string(APPEND memory_source_mmap_best_effort_phrase "-effort")
set(no_gguf_skip_phrase "No GGUF")
string(APPEND no_gguf_skip_phrase " present.*skip")
set(skip_rss_cap_now_phrase "skip RSS")
string(APPEND skip_rss_cap_now_phrase " cap for now")
set(optional_inference_parity_phrase "optional inference")
string(APPEND optional_inference_parity_phrase " parity section")
set(cuda_best_effort_ci_job_phrase "best")
string(APPEND cuda_best_effort_ci_job_phrase "-effort CI job")
set(real_cuda_best_effort_phrase "Real CUDA remains best")
string(APPEND real_cuda_best_effort_phrase "-effort")
set(promote_cuda_best_effort_phrase "Promote CUDA validation from best")
string(APPEND promote_cuda_best_effort_phrase "-effort")
set(unreal_high_performance_phrase "High-performance LLM")
string(APPEND unreal_high_performance_phrase " inference for Unreal")
set(unreal_automation_optional_phrase "Unreal Automation [(]")
string(APPEND unreal_automation_optional_phrase "Optional[)]")
set(root_high_performance_headline_phrase "High-Performance LLM")
string(APPEND root_high_performance_headline_phrase " Inference")
set(public_abi_high_performance_phrase "Stable C interface for high")
string(APPEND public_abi_high_performance_phrase "-performance LLM inference")
set(memory_high_performance_phrase "High")
string(APPEND memory_high_performance_phrase "-performance memory allocators")
set(memory_zero_allocation_hot_path_phrase "Zero Hot Path")
string(APPEND memory_zero_allocation_hot_path_phrase " Allocations")
set(utils_zero_allocations_hot_paths_phrase "Zero Allocations")
string(APPEND utils_zero_allocations_hot_paths_phrase " in Hot Paths")
set(unity_high_performance_config_phrase "High")
string(APPEND unity_high_performance_config_phrase "-performance desktop configuration")
set(unity_placeholder_binary_phrase "[(]placeholder[)]")
set(high_performance_infrastructure_phrase "high")
string(APPEND high_performance_infrastructure_phrase "-performance infrastructure")
set(high_performance_parallel_phrase "high")
string(APPEND high_performance_parallel_phrase "-performance parallel execution")
set(best_effort_timing_phrase "Best")
string(APPEND best_effort_timing_phrase "-effort timing")
set(best_effort_queue_depth_phrase "Best")
string(APPEND best_effort_queue_depth_phrase "-effort queue depth tracking")
set(best_effort_no_exceptions_phrase "Best")
string(APPEND best_effort_no_exceptions_phrase "-effort \"no exceptions cross the C ABI\"")
set(best_effort_cleanup_phrase "Best")
string(APPEND best_effort_cleanup_phrase "-effort cleanup")
set(best_effort_global_new_phrase "Best")
string(APPEND best_effort_global_new_phrase "-effort global new tracking")
set(best_effort_newline_phrase "Best")
string(APPEND best_effort_newline_phrase "-effort: resolve a single-token")
set(best_effort_hint_phrase "Best")
string(APPEND best_effort_hint_phrase "-effort hint")
set(root_placeholder_repo_phrase "github.com/your")
string(APPEND root_placeholder_repo_phrase "-org/astral")
set(root_license_tbd_phrase "[[]License")
string(APPEND root_license_tbd_phrase " TBD")
set(public_header_license_tbd_phrase "License: [[]T")
string(APPEND public_header_license_tbd_phrase "BD[]]")
set(root_link_tbd_phrase "Link")
string(APPEND root_link_tbd_phrase " TBD")
set(root_email_tbd_phrase "Email")
string(APPEND root_email_tbd_phrase " TBD")
set(root_zero_leaks_phrase "Production-ready [(]1M[+] tokens, zero")
string(APPEND root_zero_leaks_phrase " leaks[)]")
set(root_full_test_coverage_phrase "Full test")
string(APPEND root_full_test_coverage_phrase " coverage [(]unit, integration, stress[)]")
set(root_contributions_welcome_phrase "Contributions")
string(APPEND root_contributions_welcome_phrase " welcome!")
set(root_more_workstreams_phrase "More workstreams")
string(APPEND root_more_workstreams_phrase " coming")
set(feature_parity_all_features_phrase "All")
string(APPEND feature_parity_all_features_phrase " Features")
set(feature_parity_better_performance_phrase "Better")
string(APPEND feature_parity_better_performance_phrase " Performance")
set(feature_parity_zero_allocations_phrase "Zero")
string(APPEND feature_parity_zero_allocations_phrase " Allocations.*Proven zero GC")
set(feature_parity_zero_allocation_hot_paths_phrase "Zero-Allocation")
string(APPEND feature_parity_zero_allocation_hot_paths_phrase " Hot Paths")
set(feature_parity_best_in_class_phrase "Best-in")
string(APPEND feature_parity_best_in_class_phrase "-class")
set(ignored_feature_parity_doc_phrase "docs/FEATURE_")
string(APPEND ignored_feature_parity_doc_phrase "PARITY.md")
set(stale_master_spec_phrase "MASTER")
string(APPEND stale_master_spec_phrase "_SPEC")
set(stale_master_spec_doc_phrase "docs/MASTER")
string(APPEND stale_master_spec_doc_phrase "_SPEC.md")
set(stale_workstreams_phrase "docs/work")
string(APPEND stale_workstreams_phrase "streams")
set(stale_core_agent_phrase "CORE_RUNTIME")
string(APPEND stale_core_agent_phrase "_AGENT.md")
set(stale_memory_agent_phrase "MEMORY_SPECIALIST")
string(APPEND stale_memory_agent_phrase "_AGENT.md")
set(stale_concurrency_agent_phrase "CONCURRENCY_SPECIALIST")
string(APPEND stale_concurrency_agent_phrase "_AGENT.md")
set(stale_roadmap_product_ready_phrase "world")
string(APPEND stale_roadmap_product_ready_phrase "-class, product-ready framework")
set(stale_roadmap_cuda_scaffold_phrase "runtime optional scaffold in v0.1")
string(APPEND stale_roadmap_cuda_scaffold_phrase " [(]best-effort[)]")
set(stale_roadmap_memory_estimates_phrase "best")
string(APPEND stale_roadmap_memory_estimates_phrase "-effort memory estimates")
set(stale_roadmap_ue51_phrase "UE 5.1")
string(APPEND stale_roadmap_ue51_phrase ".5.7")
set(stale_unity_shippable_phrase "current,")
string(APPEND stale_unity_shippable_phrase " shippable Unity API")
set(stale_unity_platform_coverage_phrase "Platform")
string(APPEND stale_unity_platform_coverage_phrase " Coverage")
set(stale_unity_portable_phrase "Portable")
string(APPEND stale_unity_portable_phrase ".*armv7")
set(stale_unity_realtime_streaming_phrase "Streaming Support")
string(APPEND stale_unity_realtime_streaming_phrase ".*Real-time token streaming")
set(stale_sample_zero_copy_phrase "Zero")
string(APPEND stale_sample_zero_copy_phrase "-copy NativeArray conversions")
set(stale_sample_proper_cleanup_phrase "Proper")
string(APPEND stale_sample_proper_cleanup_phrase " resource cleanup")
set(stale_sample_ai_response_phrase "AI")
string(APPEND stale_sample_ai_response_phrase " response")

set(UNREVIEWED_PROSE_STRINGS
  "${ai_generation_phrase}"
  "${automated_tools_phrase}"
  "${proper_cleanup_phrase}"
  "${proper_lifetime_phrase}"
  "${raii_pattern_phrase}"
  "${always_dispose_phrase}"
  "${disposed_properly_phrase}"
  "${create_temporary_arrays_phrase}"
  "${dispose_temporary_arrays_phrase}"
  "${unreviewed_name_phrase}"
  "${handle_error_phrase}"
  "${process_data_phrase}"
  "${ensure_proper_phrase}"
  "${this_function_phrase}"
  "${for_safety_phrase}"
  "${just_in_case_phrase}"
  "${no_assertion_phrase}"
  "${just_verify_phrase}"
  "${just_verify_lower_phrase}"
  "${should_not_crash_phrase}"
  "${best_effort_basis_phrase}"
  "${best_effort_may_fail_phrase}"
  "${coarse_plots_best_effort_phrase}"
  "${init_penalty_best_effort_phrase}"
  "${grammar_constraints_best_effort_phrase}"
  "${reference_counted_best_effort_phrase}"
  "${skip_gracefully_phrase}"
  "${skips_if_native_missing_phrase}"
  "${cuda_machine_best_effort_phrase}"
  "${cuda_bench_best_effort_phrase}"
  "${cuda_scaffold_best_effort_phrase}"
  "${stale_release_matrix_phrase}"
  "${unity_high_performance_phrase}"
  "${unity_zero_gc_phrase}"
  "${unity_profiler_validated_phrase}"
  "${unity_il2cpp_safe_phrase}"
  "${unity_raii_pattern_phrase}"
  "${unity_pinvoke_all_tested_phrase}"
  "${unity_zero_alloc_streaming_phrase}"
  "${unity_zero_gc_allocation_phrase}"
  "${unity_maximum_performance_phrase}"
  "${multimodal_optin_best_effort_phrase}"
  "${model_source_desktop_best_effort_phrase}"
  "${media_feed_best_effort_phrase}"
  "${gpu_device_best_effort_phrase}"
  "${gpu_mask_best_effort_phrase}"
  "${gpu_fields_best_effort_phrase}"
  "${gpu_settings_best_effort_phrase}"
  "${handle_valid_best_effort_phrase}"
  "${model_limits_best_effort_phrase}"
  "${executor_tuning_best_effort_phrase}"
  "${unity_cuda_config_best_effort_phrase}"
  "${vm_aligned_best_effort_phrase}"
  "${vm_huge_best_effort_phrase}"
  "${vm_silently_fail_phrase}"
  "${vm_silent_fail_comment_phrase}"
  "${vm_commit_best_effort_phrase}"
  "${alloc_best_effort_phrase}"
  "${allocation_portable_best_effort_phrase}"
  "${io_portable_best_effort_phrase}"
  "${worker_id_best_effort_phrase}"
  "${slot_reset_best_effort_phrase}"
  "${provider_slot_best_effort_phrase}"
  "${memory_source_mmap_best_effort_phrase}"
  "${no_gguf_skip_phrase}"
  "${skip_rss_cap_now_phrase}"
  "${optional_inference_parity_phrase}"
  "${cuda_best_effort_ci_job_phrase}"
  "${real_cuda_best_effort_phrase}"
  "${promote_cuda_best_effort_phrase}"
  "${unreal_high_performance_phrase}"
  "${unreal_automation_optional_phrase}"
  "${root_high_performance_headline_phrase}"
  "${public_abi_high_performance_phrase}"
  "${memory_high_performance_phrase}"
  "${memory_zero_allocation_hot_path_phrase}"
  "${utils_zero_allocations_hot_paths_phrase}"
  "${unity_high_performance_config_phrase}"
  "${unity_placeholder_binary_phrase}"
  "${high_performance_infrastructure_phrase}"
  "${high_performance_parallel_phrase}"
  "${best_effort_timing_phrase}"
  "${best_effort_queue_depth_phrase}"
  "${best_effort_no_exceptions_phrase}"
  "${best_effort_cleanup_phrase}"
  "${best_effort_global_new_phrase}"
  "${best_effort_newline_phrase}"
  "${best_effort_hint_phrase}"
  "${root_placeholder_repo_phrase}"
  "${root_license_tbd_phrase}"
  "${public_header_license_tbd_phrase}"
  "${root_link_tbd_phrase}"
  "${root_email_tbd_phrase}"
  "${root_zero_leaks_phrase}"
  "${root_full_test_coverage_phrase}"
  "${root_contributions_welcome_phrase}"
  "${root_more_workstreams_phrase}"
  "${feature_parity_all_features_phrase}"
  "${feature_parity_better_performance_phrase}"
  "${feature_parity_zero_allocations_phrase}"
  "${feature_parity_zero_allocation_hot_paths_phrase}"
  "${feature_parity_best_in_class_phrase}"
  "${ignored_feature_parity_doc_phrase}"
  "${stale_master_spec_phrase}"
  "${stale_master_spec_doc_phrase}"
  "${stale_workstreams_phrase}"
  "${stale_core_agent_phrase}"
  "${stale_memory_agent_phrase}"
  "${stale_concurrency_agent_phrase}"
  "${stale_roadmap_product_ready_phrase}"
  "${stale_roadmap_cuda_scaffold_phrase}"
  "${stale_roadmap_memory_estimates_phrase}"
  "${stale_roadmap_ue51_phrase}"
  "${stale_unity_shippable_phrase}"
  "${stale_unity_platform_coverage_phrase}"
  "${stale_unity_portable_phrase}"
  "${stale_unity_realtime_streaming_phrase}"
  "${stale_sample_zero_copy_phrase}"
  "${stale_sample_proper_cleanup_phrase}"
  "${stale_sample_ai_response_phrase}"
)

foreach(path IN LISTS FILES)
  file(READ "${path}" content)
  get_filename_component(ext "${path}" EXT)
  set(is_lambda_gated_source OFF)
  if(path MATCHES "/include/" OR path MATCHES "/src/" OR path MATCHES "/backend_plugins/" OR path MATCHES "/plugins/unreal/AstralRT/Source/AstralRT/Private/")
    if(NOT path MATCHES "/tests/" AND NOT path MATCHES "/benchmarks/" AND NOT path MATCHES "/examples/")
      set(is_lambda_gated_source ON)
    endif()
  endif()

  foreach(token IN LISTS FORBIDDEN_STRINGS)
    if(content MATCHES "${token}")
      message(FATAL_ERROR "Forbidden token '${token}' found in ${path}")
    endif()
  endforeach()

  # Gate: no full-vocab logits memcpy (static heuristic).
  # We forbid any memcpy call that references logits or vocab_size in the argument list.
  if(content MATCHES "memcpy\\([^\\)]*logits" OR content MATCHES "memcpy\\([^\\)]*vocab_size")
    message(FATAL_ERROR "Suspicious logits/vocab memcpy found in ${path}")
  endif()

  file(STRINGS "${path}" lines)
  set(line_no 0)
  foreach(line IN LISTS lines)
    math(EXPR line_no "${line_no} + 1")
    foreach(token IN LISTS TRACKED_COMMENT_TOKENS)
      if(line MATCHES "${token}" AND NOT line MATCHES "workspace-[A-Za-z0-9]+")
        message(FATAL_ERROR "Untracked ${token} marker in ${path}:${line_no}; add a issue tracker issue id or remove it")
      endif()
    endforeach()
    foreach(phrase IN LISTS UNREVIEWED_PROSE_STRINGS)
      if(line MATCHES "${phrase}")
        message(FATAL_ERROR "Unreviewed generic prose '${phrase}' found in ${path}:${line_no}; rewrite it with project-specific ownership, lifecycle, or failure-mode language")
      endif()
    endforeach()
    if(is_lambda_gated_source AND ext MATCHES "^\\.(h|hpp|c|cc|cpp)$" AND line MATCHES "\\[[^]]*\\][ \t]*\\(" AND NOT line MATCHES "operator[^[]*\\[[ \t]*\\]")
      message(FATAL_ERROR "Lambda expression found in ${path}:${line_no}; use a named helper so ownership, profiling, and control flow stay reviewable")
    endif()
    if(path MATCHES "/tests/.*\\.(cpp|hpp)$")
      if(line MATCHES "ASSERT_TRUE\\(true\\)")
        message(FATAL_ERROR "No-op ASSERT_TRUE(true) found in ${path}:${line_no}; use a real assertion or SKIP_TEST with an explicit fixture/probe reason")
      endif()
      if(line MATCHES "\\[SKIP\\]")
        message(FATAL_ERROR "Ad hoc [SKIP] output found in ${path}:${line_no}; use SKIP_TEST so skipped fixture/probe coverage is counted")
      endif()
    endif()
  endforeach()
endforeach()

set(test_framework_file "${ROOT}/tests/test_framework.hpp")
if(NOT EXISTS "${test_framework_file}")
  message(FATAL_ERROR "Test framework header missing: ${test_framework_file}")
endif()
file(READ "${test_framework_file}" test_framework_content)
foreach(required_test_skip_text
    "struct TestSkipped"
    "[  SKIPPED ]"
    "[ SKIPPED  ]"
    "test_skip_msg"
    "#define SKIP_TEST")
  string(FIND "${test_framework_content}" "${required_test_skip_text}" test_skip_pos)
  if(test_skip_pos EQUAL -1)
    message(FATAL_ERROR "Test framework must keep explicit skip reporting: missing '${required_test_skip_text}'")
  endif()
endforeach()

set(concurrency_model_doc "${ROOT}/docs/architecture/CONCURRENCY_MODEL.md")
if(NOT EXISTS "${concurrency_model_doc}")
  message(FATAL_ERROR "Concurrency memory-order contract doc missing: ${concurrency_model_doc}")
endif()
file(READ "${concurrency_model_doc}" concurrency_model_content)
foreach(required_concurrency_doc_text
    "MPMC queue (work dispatch)"
    "SPSC ring (token streaming)"
    "Memory-order contract"
    "slot.seq.load(acquire)"
    "slot.seq.store(release"
    "head.store(release)"
    "head` with acquire"
    "reset()` is a lifecycle operation"
    "ARM hardware runs for the weak-memory contract")
  string(FIND "${concurrency_model_content}" "${required_concurrency_doc_text}" concurrency_doc_contract_pos)
  if(concurrency_doc_contract_pos EQUAL -1)
    message(FATAL_ERROR "Concurrency model doc must keep reviewed memory-order contract text: missing '${required_concurrency_doc_text}'")
  endif()
endforeach()

set(spsc_ring_file "${ROOT}/src/concurrency/spsc_ring.hpp")
if(NOT EXISTS "${spsc_ring_file}")
  message(FATAL_ERROR "SPSC ring header missing: ${spsc_ring_file}")
endif()
file(READ "${spsc_ring_file}" spsc_ring_content)
foreach(required_spsc_text
    "memory_order_release on head write"
    "memory_order_acquire on head read"
    "tail_.load(std::memory_order_acquire)"
    "head_.load(std::memory_order_acquire)"
    "head_.store(next_head, std::memory_order_release)"
    "tail_.store(tail + 1, std::memory_order_release)"
    "Must not be called concurrently with push/pop"
    "Thread-safety: Single producer, single consumer only")
  string(FIND "${spsc_ring_content}" "${required_spsc_text}" spsc_contract_pos)
  if(spsc_contract_pos EQUAL -1)
    message(FATAL_ERROR "SPSC ring must keep reviewed memory-order contract evidence: missing '${required_spsc_text}'")
  endif()
endforeach()

set(mpmc_queue_file "${ROOT}/src/concurrency/mpmc_queue.hpp")
if(NOT EXISTS "${mpmc_queue_file}")
  message(FATAL_ERROR "MPMC queue header missing: ${mpmc_queue_file}")
endif()
file(READ "${mpmc_queue_file}" mpmc_queue_content)
foreach(required_mpmc_text
    "per-slot sequence"
    "fetch_add + per-slot sequence + WFE/SEV"
    "slot.seq.load(std::memory_order_acquire) == pos"
    "slot.seq.load(std::memory_order_acquire) == pos + 1"
    "slot.seq.store(pos + 1, std::memory_order_release)"
    "slot.seq.store(pos + Capacity, std::memory_order_release)"
    "Correct on weak memory models")
  string(FIND "${mpmc_queue_content}" "${required_mpmc_text}" mpmc_contract_pos)
  if(mpmc_contract_pos EQUAL -1)
    message(FATAL_ERROR "MPMC queue must keep reviewed memory-order contract evidence: missing '${required_mpmc_text}'")
  endif()
endforeach()

set(ci_workflow_file "${ROOT}/.github/workflows/ci.yml")
if(NOT EXISTS "${ci_workflow_file}")
  message(FATAL_ERROR "CI workflow missing: ${ci_workflow_file}")
endif()
file(READ "${ci_workflow_file}" ci_workflow_content)
set(github_runner_os_expr "$")
string(APPEND github_runner_os_expr "{{ runner.os }}")
set(shell_runner_os_expr "$")
string(APPEND shell_runner_os_expr "{RUNNER_OS}")
foreach(required_ci_ctest_text
    "ctest --preset dev -j --output-on-failure"
    "ctest --preset dev-prof -j --output-on-failure"
    "ctest-dev-${github_runner_os_expr}"
    "ctest-dev-prof-${github_runner_os_expr}"
    "build/test-logs/ctest-${shell_runner_os_expr}-dev.log"
    "build/test-logs/ctest-dev-prof-${shell_runner_os_expr}.log")
  string(FIND "${ci_workflow_content}" "${required_ci_ctest_text}" ci_ctest_pos)
  if(ci_ctest_pos EQUAL -1)
    message(FATAL_ERROR "CI native CTest log workflow must keep '${required_ci_ctest_text}'")
  endif()
endforeach()

set(fast_presubmit_file "${ROOT}/scripts/run_fast_presubmit.sh")
if(NOT EXISTS "${fast_presubmit_file}")
  message(FATAL_ERROR "Fast presubmit runner missing: ${fast_presubmit_file}")
endif()
file(READ "${fast_presubmit_file}" fast_presubmit_content)
foreach(required_fast_presubmit_text
    "cmake --preset"
    "cmake --build --preset"
    "ctest --preset"
    "--output-on-failure"
    "dev|dev-prof|dev-prof-micro"
    "fast-presubmit")
  string(FIND "${fast_presubmit_content}" "${required_fast_presubmit_text}" fast_presubmit_pos)
  if(fast_presubmit_pos EQUAL -1)
    message(FATAL_ERROR "Fast presubmit runner must keep '${required_fast_presubmit_text}'")
  endif()
endforeach()
foreach(forbidden_fast_presubmit_text
    "model_downloader"
    "run_release_required_gates"
    "run_unreal"
    "run_unity"
    "run_cuda"
    "run_multimodal"
    "package_release"
    "run_hf"
    "run_windows")
  string(FIND "${fast_presubmit_content}" "${forbidden_fast_presubmit_text}" fast_presubmit_forbidden_pos)
  if(NOT fast_presubmit_forbidden_pos EQUAL -1)
    message(FATAL_ERROR "Fast presubmit runner must not invoke slow or external lane '${forbidden_fast_presubmit_text}'")
  endif()
endforeach()

set(unreal_stream_pump_file "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralSessionStreamPump.cpp")
if(EXISTS "${unreal_stream_pump_file}")
  file(READ "${unreal_stream_pump_file}" unreal_stream_pump_content)
  if(NOT unreal_stream_pump_content MATCHES "TRACE_CPUPROFILER_EVENT_SCOPE\\(AstralRT_StreamPump_Tick\\)")
    message(FATAL_ERROR "Unreal stream pump must keep a CPU profiler scope for game-thread handoff diagnosis")
  endif()
endif()

set(unreal_session_file "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralSession.cpp")
if(EXISTS "${unreal_session_file}")
  file(READ "${unreal_session_file}" unreal_session_content)
  if(NOT unreal_session_content MATCHES "TRACE_CPUPROFILER_EVENT_SCOPE\\(AstralRT_Session_TickStream\\)")
    message(FATAL_ERROR "Unreal session ticker must keep a CPU profiler scope for game-thread handoff diagnosis")
  endif()
endif()

set(unreal_embedder_file "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralEmbedder.cpp")
if(EXISTS "${unreal_embedder_file}")
  file(READ "${unreal_embedder_file}" unreal_embedder_content)
  foreach(scope_name
      AstralRT_Embedder_EnqueueUtf8Bytes
      AstralRT_Embedder_EnqueueImage
      AstralRT_Embedder_EnqueueAudio
      AstralRT_Embedder_EnqueueMultimodal
      AstralRT_Embedder_Collect)
    if(NOT unreal_embedder_content MATCHES "TRACE_CPUPROFILER_EVENT_SCOPE\\(${scope_name}\\)")
      message(FATAL_ERROR "Unreal embedder must keep CPU profiler scope ${scope_name}")
    endif()
  endforeach()
endif()

set(unreal_build_rules_file "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/AstralRT.Build.cs")
if(EXISTS "${unreal_build_rules_file}")
  file(READ "${unreal_build_rules_file}" unreal_build_rules_content)
  foreach(required_build_rule_text
      "RequireThirdPartyFile"
      "BuildException"
      "cmake --preset unreal-plugin && cmake --build --preset unreal-plugin -j"
      "AstralRT does not ship a native library")
    if(NOT unreal_build_rules_content MATCHES "${required_build_rule_text}")
      message(FATAL_ERROR "Unreal build rules must fail clearly on missing ThirdParty artifacts: missing '${required_build_rule_text}'")
    endif()
  endforeach()
endif()

file(GLOB_RECURSE unreal_plugin_source_files
  "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Public/*.h"
  "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/*.h"
  "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/*.cpp"
)
foreach(unreal_plugin_source IN LISTS unreal_plugin_source_files)
  if(NOT unreal_plugin_source MATCHES "/Private/Tests/")
    file(STRINGS "${unreal_plugin_source}" unreal_plugin_lines)
    set(unreal_plugin_line_no 0)
    foreach(unreal_plugin_line IN LISTS unreal_plugin_lines)
      math(EXPR unreal_plugin_line_no "${unreal_plugin_line_no} + 1")
      if(unreal_plugin_line MATCHES "UE_LOG\\(LogTemp")
        message(FATAL_ERROR "Use LogAstralRT instead of LogTemp in ${unreal_plugin_source}:${unreal_plugin_line_no}")
      endif()
    endforeach()
  endif()
endforeach()

foreach(unreal_doc
    "${ROOT}/docs/integration/UNREAL_INTEGRATION.md"
    "${ROOT}/plugins/unreal/AstralRT/README.md")
  if(EXISTS "${unreal_doc}")
    file(STRINGS "${unreal_doc}" unreal_doc_lines)
    set(unreal_doc_line_no 0)
    foreach(unreal_doc_line IN LISTS unreal_doc_lines)
      math(EXPR unreal_doc_line_no "${unreal_doc_line_no} + 1")
      if(unreal_doc_line MATCHES "UE_LOG\\(LogTemp")
        message(FATAL_ERROR "Use LogAstralRT instead of LogTemp in Unreal docs: ${unreal_doc}:${unreal_doc_line_no}")
      endif()
    endforeach()
  endif()
endforeach()

message(STATUS "gate_source_scans: OK (${FILES_LEN} files)")
