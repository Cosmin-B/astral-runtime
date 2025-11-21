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
  ci
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

set(test_embeddings_file "${ROOT}/tests/test_embeddings.cpp")
file(READ "${test_embeddings_file}" test_embeddings_content)
foreach(required_embedding_probe_text
    "ASTRAL_TEST_EMBED_MODEL"
    "embeddings_cpu_e2e_fixture_probe"
    "[embedding_probe] backend=cpu"
    "sum_abs="
    "ASSERT_GT(sum_abs, 0.0)")
  string(FIND "${test_embeddings_content}" "${required_embedding_probe_text}" embedding_probe_pos)
  if(embedding_probe_pos EQUAL -1)
    message(FATAL_ERROR "Embedding fixture probe is missing '${required_embedding_probe_text}'")
  endif()
endforeach()

set(unreal_embedding_probe_file "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/Tests/AstralRTTests.cpp")
file(READ "${unreal_embedding_probe_file}" unreal_embedding_probe_content)
foreach(required_unreal_embedding_probe_text
    "AstralRT.Real.EmbeddingProbe"
    "ASTRAL_UNREAL_TEST_EMBED_MODEL"
    "ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING"
    "[unreal_embedding_probe] backend=cpu"
    "SumAbs > 0.0")
  string(FIND "${unreal_embedding_probe_content}" "${required_unreal_embedding_probe_text}" unreal_embedding_probe_pos)
  if(unreal_embedding_probe_pos EQUAL -1)
    message(FATAL_ERROR "Unreal real embedding probe is missing '${required_unreal_embedding_probe_text}'")
  endif()
endforeach()

foreach(required_unreal_generation_smoke_text
    "AstralRT.Real.GenerationSmoke"
    "ASTRAL_UNREAL_TEST_MODEL"
    "ASTRAL_UNREAL_REQUIRE_REAL_GENERATION"
    "[unreal_generation_smoke] backend=cpu"
    "ByteCount > 0")
  string(FIND "${unreal_embedding_probe_content}" "${required_unreal_generation_smoke_text}" unreal_generation_smoke_pos)
  if(unreal_generation_smoke_pos EQUAL -1)
    message(FATAL_ERROR "Unreal real generation smoke is missing '${required_unreal_generation_smoke_text}'")
  endif()
endforeach()

foreach(required_unreal_lifecycle_smoke_text
    "AstralRT.Real.SessionLifecycle"
    "ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE"
    "[unreal_session_lifecycle] backend=cpu"
    "CancelWait == ASTRAL_E_CANCELED"
    "ReuseByteCount > 0")
  string(FIND "${unreal_embedding_probe_content}" "${required_unreal_lifecycle_smoke_text}" unreal_lifecycle_smoke_pos)
  if(unreal_lifecycle_smoke_pos EQUAL -1)
    message(FATAL_ERROR "Unreal real session lifecycle smoke is missing '${required_unreal_lifecycle_smoke_text}'")
  endif()
endforeach()

set(model_downloader_file "${ROOT}/tests/model_downloader.sh")
file(READ "${model_downloader_file}" model_downloader_content)
foreach(required_model_downloader_text
    "qwen3-0.6b-q8"
    "Qwen/Qwen3-0.6B-GGUF"
    "Qwen3-0.6B-Q8_0.gguf"
    "gemma3-270m-q4km"
    "gguf-org/gemma-3-270m-gguf"
    "gemma-3-270m-q4_k_m.gguf"
    "gemma3-1b-it-q4km"
    "ggml-org/gemma-3-1b-it-GGUF"
    "gemma-3-1b-it-Q4_K_M.gguf"
    "smollm3-3b-q4km"
    "ggml-org/SmolLM3-3B-GGUF"
    "SmolLM3-Q4_K_M.gguf"
    "qwen3-embed-0.6b-q8"
    "Qwen/Qwen3-Embedding-0.6B-GGUF"
    "Qwen3-Embedding-0.6B-Q8_0.gguf"
    "qwen3-1.7b-q8"
    "Qwen/Qwen3-1.7B-GGUF"
    "Qwen3-1.7B-Q8_0.gguf")
  string(FIND "${model_downloader_content}" "${required_model_downloader_text}" model_downloader_pos)
  if(model_downloader_pos EQUAL -1)
    message(FATAL_ERROR "Model downloader is missing '${required_model_downloader_text}'")
  endif()
endforeach()

set(hf_manifest_file "${ROOT}/scripts/hf_gguf_manifest_full.json")
file(READ "${hf_manifest_file}" hf_manifest_content)
foreach(required_hf_manifest_text
    "Qwen/Qwen3-0.6B-GGUF"
    "^Qwen3-0\\\\.6B-Q8_0\\\\.gguf$"
    "gguf-org/gemma-3-270m-gguf"
    "^gemma-3-270m-q4_k_m\\\\.gguf$"
    "ggml-org/gemma-3-1b-it-GGUF"
    "^gemma-3-1b-it-Q4_K_M\\\\.gguf$"
    "ggml-org/SmolLM3-3B-GGUF"
    "^SmolLM3-Q4_K_M\\\\.gguf$"
    "Qwen/Qwen3-1.7B-GGUF"
    "^Qwen3-1\\\\.7B-Q8_0\\\\.gguf$"
    "Qwen/Qwen3-Embedding-0.6B-GGUF"
    "^Qwen3-Embedding-0\\\\.6B-Q8_0\\\\.gguf$")
  string(FIND "${hf_manifest_content}" "${required_hf_manifest_text}" hf_manifest_pos)
  if(hf_manifest_pos EQUAL -1)
    message(FATAL_ERROR "HF GGUF manifest is missing '${required_hf_manifest_text}'")
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
    "run_windows"
    "soak")
  string(FIND "${fast_presubmit_content}" "${forbidden_fast_presubmit_text}" fast_presubmit_forbidden_pos)
  if(NOT fast_presubmit_forbidden_pos EQUAL -1)
    message(FATAL_ERROR "Fast presubmit runner must not invoke slow or external lane '${forbidden_fast_presubmit_text}'")
  endif()
endforeach()

set(unreal_sample_script "${ROOT}/scripts/create_unreal_sample_project.sh")
set(cmake_presets_file "${ROOT}/CMakePresets.json")
set(root_cmake_file "${ROOT}/CMakeLists.txt")
set(unreal_sample_gate "${ROOT}/tests/cmake/gate_unreal_sample_scaffold.cmake")
set(unreal_sample_package_script "${ROOT}/scripts/run_unreal_sample_package.sh")
set(unreal_sample_package_gate "${ROOT}/tests/cmake/gate_unreal_sample_package_runner.cmake")
if(NOT EXISTS "${unreal_sample_script}")
  message(FATAL_ERROR "Unreal sample project scaffold script is missing")
endif()
if(NOT EXISTS "${cmake_presets_file}")
  message(FATAL_ERROR "CMake presets file is missing")
endif()
if(NOT EXISTS "${root_cmake_file}")
  message(FATAL_ERROR "Root CMake file is missing")
endif()
if(NOT EXISTS "${unreal_sample_gate}")
  message(FATAL_ERROR "Unreal sample project scaffold gate is missing")
endif()
if(NOT EXISTS "${unreal_sample_package_script}")
  message(FATAL_ERROR "Unreal sample package runner is missing")
endif()
if(NOT EXISTS "${unreal_sample_package_gate}")
  message(FATAL_ERROR "Unreal sample package runner gate is missing")
endif()
file(READ "${unreal_sample_script}" unreal_sample_script_text)
file(READ "${cmake_presets_file}" cmake_presets_text)
file(READ "${root_cmake_file}" root_cmake_text)
file(READ "${unreal_sample_gate}" unreal_sample_gate_text)
file(READ "${unreal_sample_package_script}" unreal_sample_package_script_text)
file(READ "${unreal_sample_package_gate}" unreal_sample_package_gate_text)
foreach(required_unreal_sample_script_text
    "AstralSample.uproject"
    "EngineAssociation"
    "5.7"
    "RunGenerationDemo"
    "CancelStreamingDemo"
    "RunEmbeddingDemo"
    "RunMediaFeedDemo"
    "RunErrorDemo"
    "AstralMediaBackend="
    "UE_LOG(LogAstralSample, Display"
    "UAstralMediaLibrary::MakeRGBA8ImageFromBytes"
    "MediaSession->FeedAudio"
    "OnStreamBytesNative"
    "EmbedUtf8Bytes"
    "astral_last_error"
    "plugin-mode")
  string(FIND "${unreal_sample_script_text}" "${required_unreal_sample_script_text}" unreal_sample_script_pos)
  if(unreal_sample_script_pos EQUAL -1)
    message(FATAL_ERROR "Unreal sample scaffold script is missing '${required_unreal_sample_script_text}'")
  endif()
endforeach()
foreach(required_unreal_cmake_preset_text
    "\"name\": \"unreal-plugin\""
    "\"CMAKE_CXX_STANDARD\": \"20\""
    "\"CMAKE_CXX_STANDARD_REQUIRED\": \"ON\"")
  string(FIND "${cmake_presets_text}" "${required_unreal_cmake_preset_text}" unreal_cmake_preset_pos)
  if(unreal_cmake_preset_pos EQUAL -1)
    message(FATAL_ERROR "Unreal CMake preset is missing '${required_unreal_cmake_preset_text}'")
  endif()
endforeach()
foreach(required_unreal_root_cmake_text
    "if(ASTRAL_BUILD_UNREAL_PLUGIN)"
    "add_compile_definitions(JSON_HAS_CPP_17)"
    "ASTRAL_BUILD_UNREAL_PLUGIN OR ASTRAL_BUILD_TESTS"
    "CMAKE_POSITION_INDEPENDENT_CODE ON")
  string(FIND "${root_cmake_text}" "${required_unreal_root_cmake_text}" unreal_root_cmake_pos)
  if(unreal_root_cmake_pos EQUAL -1)
    message(FATAL_ERROR "Unreal CMake package is missing '${required_unreal_root_cmake_text}'")
  endif()
endforeach()
foreach(required_unreal_sample_gate_text
    "gate_unreal_sample_scaffold"
    "create_unreal_sample_project.sh"
    "RunGenerationDemo"
    "CancelStreamingDemo"
    "RunEmbeddingDemo"
    "RunErrorDemo")
  string(FIND "${unreal_sample_gate_text}" "${required_unreal_sample_gate_text}" unreal_sample_gate_pos)
  if(unreal_sample_gate_pos EQUAL -1)
    message(FATAL_ERROR "Unreal sample scaffold gate is missing '${required_unreal_sample_gate_text}'")
  endif()
endforeach()
foreach(required_unreal_sample_package_script_text
    "create_unreal_sample_project.sh"
    "plugin-mode copy"
    "RunUAT"
    "BuildCookRun"
    "-platform="
    "-archive"
    "--run-sample"
    "ASTRAL_UNREAL_SAMPLE_MODEL"
    "-AstralMemoryBackend="
    "[unreal_sample] Runtime OK"
    "[unreal_sample] OK:")
  string(FIND "${unreal_sample_package_script_text}" "${required_unreal_sample_package_script_text}" unreal_sample_package_script_pos)
  if(unreal_sample_package_script_pos EQUAL -1)
    message(FATAL_ERROR "Unreal sample package runner is missing '${required_unreal_sample_package_script_text}'")
  endif()
endforeach()
foreach(required_unreal_sample_package_gate_text
    "gate_unreal_sample_package_runner"
    "run_unreal_sample_package.sh"
    "Missing Unreal RunUAT path"
    "fake RunUAT: BuildCookRun"
    "AstralSample.uproject")
  string(FIND "${unreal_sample_package_gate_text}" "${required_unreal_sample_package_gate_text}" unreal_sample_package_gate_pos)
  if(unreal_sample_package_gate_pos EQUAL -1)
    message(FATAL_ERROR "Unreal sample package gate is missing '${required_unreal_sample_package_gate_text}'")
  endif()
endforeach()

set(unreal_container_script "${ROOT}/scripts/run_unreal_container_ci.sh")
set(unreal_container_gate "${ROOT}/tests/cmake/gate_unreal_container_runner.cmake")
set(unreal_access_script "${ROOT}/scripts/check_unreal_validation_access.sh")
set(unreal_access_gate "${ROOT}/tests/cmake/gate_unreal_validation_access.cmake")
set(unreal_ci_script "${ROOT}/scripts/run_unreal_ci_tests.sh")
if(NOT EXISTS "${unreal_container_script}")
  message(FATAL_ERROR "Unreal container runner script is missing")
endif()
if(NOT EXISTS "${unreal_ci_script}")
  message(FATAL_ERROR "Unreal CI runner script is missing")
endif()
if(NOT EXISTS "${unreal_container_gate}")
  message(FATAL_ERROR "Unreal container runner gate is missing")
endif()
if(NOT EXISTS "${unreal_access_script}")
  message(FATAL_ERROR "Unreal validation access checker is missing")
endif()
if(NOT EXISTS "${unreal_access_gate}")
  message(FATAL_ERROR "Unreal validation access checker gate is missing")
endif()
file(READ "${unreal_container_script}" unreal_container_script_text)
file(READ "${unreal_container_gate}" unreal_container_gate_text)
file(READ "${unreal_access_script}" unreal_access_script_text)
file(READ "${unreal_access_gate}" unreal_access_gate_text)
file(READ "${unreal_ci_script}" unreal_ci_script_text)
foreach(required_unreal_container_script_text
    "dev-5.7.4"
    "sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce"
    "dev-slim-5.7.4"
    "sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6"
    "ASTRAL_UNREAL_TEST_MODEL"
    "ASTRAL_UNREAL_REQUIRE_REAL_GENERATION"
    "ASTRAL_UNREAL_REQUIRE_REAL_LIFECYCLE"
    "ASTRAL_UNREAL_TEST_EMBED_MODEL"
    "ASTRAL_UNREAL_REQUIRE_REAL_EMBEDDING"
    "Epic Unreal container access is not configured"
    "ASTRAL_UNREAL_PULL_TIMEOUT_SECONDS"
    "ASTRAL_UNREAL_REQUIRED_CLANG_VERSION"
    "ASTRAL_UNREAL_REQUIRED_LINUX_SDK_TOOLCHAIN"
    "ASTRAL_UNREAL_REQUIRED_LINUX_SDK_CLANG"
    "20.1.8"
    "v26"
    "x86_64-unknown-linux-gnu/bin/clang"
    "clang version mismatch"
    "clang not found in PATH or UE Linux SDK"
    "Native package C++ runtime: UE libc++"
    "-nostdinc++"
    "include/c++/v1"
    "-DCMAKE_CXX_FLAGS"
    "Linux SDK metadata mismatch")
  string(FIND "${unreal_container_script_text}" "${required_unreal_container_script_text}" unreal_container_script_pos)
  if(unreal_container_script_pos EQUAL -1)
    message(FATAL_ERROR "Unreal container runner script is missing '${required_unreal_container_script_text}'")
  endif()
endforeach()
foreach(required_unreal_container_gate_text
    "gate_unreal_container_runner"
    "check_missing_auth(\"full\""
    "check_missing_auth(\"slim\""
    "dev-5[.]7[.]4@sha256:582895c09ada64db1f3e46053afe29e4fdd0d55da53d60b7b29741f6ecfb34ce"
    "dev-slim-5[.]7[.]4@sha256:5d8fa43dbbc07ea53e6474c0f3ac33af092cc264070b0985a2d3e8c4697940f6")
  string(FIND "${unreal_container_gate_text}" "${required_unreal_container_gate_text}" unreal_container_gate_pos)
  if(unreal_container_gate_pos EQUAL -1)
    message(FATAL_ERROR "Unreal container runner gate is missing '${required_unreal_container_gate_text}'")
  endif()
endforeach()
foreach(required_unreal_access_script_text
    "check_unreal_validation_access.sh"
    "dev-5.7.4"
    "dev-slim-5.7.4"
    "UNREAL_54_EDITOR"
    "UNREAL_57_EDITOR"
    "RunUAT"
    "READY: UE 5.7 full/slim container access is available"
    "READY: UE 5.4-5.7 editor matrix is configured"
    "BLOCKED: real Unreal validation needs Epic GHCR access")
  string(FIND "${unreal_access_script_text}" "${required_unreal_access_script_text}" unreal_access_script_pos)
  if(unreal_access_script_pos EQUAL -1)
    message(FATAL_ERROR "Unreal validation access checker is missing '${required_unreal_access_script_text}'")
  endif()
endforeach()
foreach(required_unreal_access_gate_text
    "gate_unreal_validation_access"
    "fake-cached-engine"
    "fake-manifest-engine"
    "UNREAL_54_EDITOR"
    "READY: UE 5[.]4-5[.]7 editor matrix is configured"
    "OK: RunUAT available")
  string(FIND "${unreal_access_gate_text}" "${required_unreal_access_gate_text}" unreal_access_gate_pos)
  if(unreal_access_gate_pos EQUAL -1)
    message(FATAL_ERROR "Unreal validation access checker gate is missing '${required_unreal_access_gate_text}'")
  endif()
endforeach()
foreach(required_unreal_ci_script_text
    "copy_directory()"
    "cmake -E copy_directory"
    "cp -a"
    "engine_root_from_editor()"
    "RunUBT.sh"
    "AstralCiUnrealProjectEditor"
    "-NoHotReload"
    "UNREAL_TEST_FILTER:-AstralRT"
    "-ReportExportPath"
    "validate_unreal_automation_results.py")
  string(FIND "${unreal_ci_script_text}" "${required_unreal_ci_script_text}" unreal_ci_script_pos)
  if(unreal_ci_script_pos EQUAL -1)
    message(FATAL_ERROR "Unreal CI runner script is missing '${required_unreal_ci_script_text}'")
  endif()
endforeach()

set(unreal_ci_project_file "${ROOT}/ci/unreal/AstralCiUnrealProject/AstralCiUnrealProject.uproject")
set(unreal_ci_project_config "${ROOT}/ci/unreal/AstralCiUnrealProject/Config/DefaultEngine.ini")
set(unreal_ci_project_target "${ROOT}/ci/unreal/AstralCiUnrealProject/Source/AstralCiUnrealProjectEditor.Target.cs")
set(unreal_ci_project_module "${ROOT}/ci/unreal/AstralCiUnrealProject/Source/AstralCiUnrealProject/AstralCiUnrealProject.Build.cs")
if(NOT EXISTS "${unreal_ci_project_file}")
  message(FATAL_ERROR "Unreal CI project file is missing")
endif()
if(NOT EXISTS "${unreal_ci_project_config}")
  message(FATAL_ERROR "Unreal CI project engine config is missing")
endif()
if(NOT EXISTS "${unreal_ci_project_target}")
  message(FATAL_ERROR "Unreal CI editor target is missing")
endif()
if(NOT EXISTS "${unreal_ci_project_module}")
  message(FATAL_ERROR "Unreal CI module rules are missing")
endif()
file(READ "${unreal_ci_project_file}" unreal_ci_project_text)
file(READ "${unreal_ci_project_config}" unreal_ci_project_config_text)
file(READ "${unreal_ci_project_target}" unreal_ci_project_target_text)
file(READ "${unreal_ci_project_module}" unreal_ci_project_module_text)
foreach(required_unreal_ci_project_text
    "\"Name\": \"AstralCiUnrealProject\""
    "\"Type\": \"Runtime\""
    "\"Name\": \"AstralRT\"")
  string(FIND "${unreal_ci_project_text}" "${required_unreal_ci_project_text}" unreal_ci_project_pos)
  if(unreal_ci_project_pos EQUAL -1)
    message(FATAL_ERROR "Unreal CI project is missing '${required_unreal_ci_project_text}'")
  endif()
endforeach()
foreach(required_unreal_ci_project_config_text
    "EditorStartupMap=/Engine/Maps/Entry"
    "GameDefaultMap=/Engine/Maps/Entry")
  string(FIND "${unreal_ci_project_config_text}" "${required_unreal_ci_project_config_text}" unreal_ci_project_config_pos)
  if(unreal_ci_project_config_pos EQUAL -1)
    message(FATAL_ERROR "Unreal CI project engine config is missing '${required_unreal_ci_project_config_text}'")
  endif()
endforeach()
foreach(required_unreal_ci_target_text
    "TargetType.Editor"
    "BuildSettingsVersion.Latest"
    "ExtraModuleNames.Add(\"AstralCiUnrealProject\")")
  string(FIND "${unreal_ci_project_target_text}" "${required_unreal_ci_target_text}" unreal_ci_target_pos)
  if(unreal_ci_target_pos EQUAL -1)
    message(FATAL_ERROR "Unreal CI editor target is missing '${required_unreal_ci_target_text}'")
  endif()
endforeach()
foreach(required_unreal_ci_module_text
    "PCHUsageMode.UseExplicitOrSharedPCHs"
    "\"AstralRT\"")
  string(FIND "${unreal_ci_project_module_text}" "${required_unreal_ci_module_text}" unreal_ci_module_pos)
  if(unreal_ci_module_pos EQUAL -1)
    message(FATAL_ERROR "Unreal CI module rules are missing '${required_unreal_ci_module_text}'")
  endif()
endforeach()

set(unreal_plugin_build_rules "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/AstralRT.Build.cs")
set(compiler_flags_file "${ROOT}/cmake/CompilerFlags.cmake")
if(NOT EXISTS "${unreal_plugin_build_rules}")
  message(FATAL_ERROR "AstralRT Unreal build rules are missing")
endif()
if(NOT EXISTS "${compiler_flags_file}")
  message(FATAL_ERROR "Compiler flags CMake file is missing")
endif()
file(READ "${unreal_plugin_build_rules}" unreal_plugin_build_rules_text)
file(READ "${compiler_flags_file}" compiler_flags_text)
foreach(required_unreal_plugin_build_rule_text
    "../ThirdParty/AstralCore"
    "RequireThirdPartyFile"
    "libastral_rt.a"
    "libllama.a"
    "libllama-common.a"
    "libllama-common-base.a"
    "libggml.a"
    "libggml-cpu.a"
    "libggml-base.a"
    "libcpp-httplib.a"
    "PublicAdditionalLibraries.Add(LinuxArchive)"
    "PublicSystemLibraryPaths.Add(LinuxLibPath)"
    "PublicSystemLibraries.AddRange"
    "\"astral_rt\""
    "\"llama\""
    "\"llama-common\""
    "\"llama-common-base\""
    "\"ggml\""
    "\"ggml-cpu\""
    "\"ggml-base\""
    "\"cpp-httplib\"")
  string(FIND "${unreal_plugin_build_rules_text}" "${required_unreal_plugin_build_rule_text}" unreal_build_rule_pos)
  if(unreal_build_rule_pos EQUAL -1)
    message(FATAL_ERROR "AstralRT Unreal build rules are missing '${required_unreal_plugin_build_rule_text}'")
  endif()
endforeach()

set(unreal_plugin_cmake "${ROOT}/plugins/unreal/CMakeLists.txt")
if(NOT EXISTS "${unreal_plugin_cmake}")
  message(FATAL_ERROR "AstralRT Unreal plugin CMake package file is missing")
endif()
file(READ "${unreal_plugin_cmake}" unreal_plugin_cmake_text)
foreach(required_unreal_plugin_cmake_text
    "$<TARGET_FILE:astral_rt>"
    "$<TARGET_FILE:llama>"
    "$<TARGET_FILE:llama-common>"
    "$<TARGET_FILE:llama-common-base>"
    "$<TARGET_FILE:ggml>"
    "$<TARGET_FILE:ggml-cpu>"
    "$<TARGET_FILE:ggml-base>"
    "$<TARGET_FILE:cpp-httplib>"
    "DEPENDS astral_rt llama llama-common llama-common-base ggml ggml-cpu ggml-base cpp-httplib")
  string(FIND "${unreal_plugin_cmake_text}" "${required_unreal_plugin_cmake_text}" unreal_plugin_cmake_pos)
  if(unreal_plugin_cmake_pos EQUAL -1)
    message(FATAL_ERROR "AstralRT Unreal CMake package is missing '${required_unreal_plugin_cmake_text}'")
  endif()
endforeach()
foreach(required_unreal_compiler_flag_text
    "if(NOT ASTRAL_BUILD_UNREAL_PLUGIN)"
    "INTERPROCEDURAL_OPTIMIZATION TRUE")
  string(FIND "${compiler_flags_text}" "${required_unreal_compiler_flag_text}" unreal_compiler_flag_pos)
  if(unreal_compiler_flag_pos EQUAL -1)
    message(FATAL_ERROR "Unreal plugin package must keep non-LTO static archives: missing '${required_unreal_compiler_flag_text}'")
  endif()
endforeach()

set(unreal_plugin_types "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Public/AstralTypes.h")
if(NOT EXISTS "${unreal_plugin_types}")
  message(FATAL_ERROR "AstralRT Unreal public type header is missing")
endif()
file(READ "${unreal_plugin_types}" unreal_plugin_types_text)
foreach(required_unreal_plugin_type_text
    "enum class EAstralImageFormat : uint8"
    "enum class EAstralAudioFormat : uint8"
    "enum class EAstralMediaFlags : uint8"
    "enum class EAstralGpuRouteFlags : uint8"
    "enum class EAstralModelSourceKind : uint8"
    "enum class EAstralError : int32"
    "int32 GpuLayers = 0"
    "int64 GpuDeviceMask = 0"
    "int64 GpuStream = 0"
    "int64 FrameCount = 0"
    "int32 MaxTokens = 512"
    "int64 BytesCommitted = 0")
  string(FIND "${unreal_plugin_types_text}" "${required_unreal_plugin_type_text}" unreal_plugin_type_pos)
  if(unreal_plugin_type_pos EQUAL -1)
    message(FATAL_ERROR "AstralRT Unreal public types are missing '${required_unreal_plugin_type_text}'")
  endif()
endforeach()
foreach(forbidden_unreal_plugin_type_text
    "enum class EAstralImageFormat : uint32"
    "enum class EAstralAudioFormat : uint32"
    "enum class EAstralMediaFlags : uint32"
    "enum class EAstralGpuRouteFlags : uint32"
    "enum class EAstralModelSourceKind : uint32"
    "uint32 GpuLayers"
    "uint32 MaxTokens"
    "uint32 TopK"
    "uint64 GpuDeviceMask"
    "uint64 GpuStream"
    "uint64 FrameCount"
    "uint64 BytesCommitted")
  string(FIND "${unreal_plugin_types_text}" "${forbidden_unreal_plugin_type_text}" unreal_plugin_type_forbidden_pos)
  if(NOT unreal_plugin_type_forbidden_pos EQUAL -1)
    message(FATAL_ERROR "AstralRT Blueprint enum '${forbidden_unreal_plugin_type_text}' must use uint8 for UnrealHeaderTool")
  endif()
endforeach()

set(unreal_plugin_session_header "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Public/AstralSession.h")
set(unreal_plugin_session_source "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralSession.cpp")
if(NOT EXISTS "${unreal_plugin_session_header}")
  message(FATAL_ERROR "AstralRT Unreal session header is missing")
endif()
if(NOT EXISTS "${unreal_plugin_session_source}")
  message(FATAL_ERROR "AstralRT Unreal session source is missing")
endif()
file(READ "${unreal_plugin_session_header}" unreal_plugin_session_header_text)
file(READ "${unreal_plugin_session_source}" unreal_plugin_session_source_text)
foreach(required_unreal_plugin_session_text
    "int32 Wait(int32 TimeoutMs = 0)"
    "FString StreamReadString(int32 TimeoutMs = 0)")
  string(FIND "${unreal_plugin_session_header_text}" "${required_unreal_plugin_session_text}" unreal_session_header_pos)
  if(unreal_session_header_pos EQUAL -1)
    message(FATAL_ERROR "AstralRT Unreal session header is missing '${required_unreal_plugin_session_text}'")
  endif()
endforeach()
set(unreal_plugin_model_header "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Public/AstralModel.h")
if(NOT EXISTS "${unreal_plugin_model_header}")
  message(FATAL_ERROR "AstralRT Unreal model header is missing")
endif()
file(READ "${unreal_plugin_model_header}" unreal_plugin_model_header_text)
string(FIND "${unreal_plugin_model_header_text}" "bool GetCaps(int64& OutCaps) const" unreal_model_header_caps_pos)
if(unreal_model_header_caps_pos EQUAL -1)
  message(FATAL_ERROR "AstralRT Unreal model header must expose GetCaps through Blueprint-compatible int64")
endif()
foreach(required_unreal_plugin_session_source_text
    "FMath::Max(TimeoutMs, 0)"
    "const uint32 NativeTimeoutMs")
  string(FIND "${unreal_plugin_session_source_text}" "${required_unreal_plugin_session_source_text}" unreal_session_source_pos)
  if(unreal_session_source_pos EQUAL -1)
    message(FATAL_ERROR "AstralRT Unreal session source is missing '${required_unreal_plugin_session_source_text}'")
  endif()
endforeach()

set(unreal_plugin_embedder_source "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralEmbedder.cpp")
set(unreal_plugin_stream_pump_source "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralSessionStreamPump.cpp")
if(NOT EXISTS "${unreal_plugin_embedder_source}")
  message(FATAL_ERROR "AstralRT Unreal embedder source is missing")
endif()
if(NOT EXISTS "${unreal_plugin_stream_pump_source}")
  message(FATAL_ERROR "AstralRT Unreal stream pump source is missing")
endif()
file(READ "${unreal_plugin_embedder_source}" unreal_plugin_embedder_source_text)
file(READ "${unreal_plugin_stream_pump_source}" unreal_plugin_stream_pump_source_text)
foreach(required_unreal_plugin_embedder_text
    "uint64_t Ticket = 0"
    "static_cast<uint64_t>(Ticket)")
  string(FIND "${unreal_plugin_embedder_source_text}" "${required_unreal_plugin_embedder_text}" unreal_embedder_source_pos)
  if(unreal_embedder_source_pos EQUAL -1)
    message(FATAL_ERROR "AstralRT Unreal embedder source is missing '${required_unreal_plugin_embedder_text}'")
  endif()
endforeach()
foreach(forbidden_unreal_plugin_stream_pump_text
    "Containers/StringBuilder.h"
    "TStringBuilder")
  string(FIND "${unreal_plugin_stream_pump_source_text}" "${forbidden_unreal_plugin_stream_pump_text}" unreal_stream_forbidden_pos)
  if(NOT unreal_stream_forbidden_pos EQUAL -1)
    message(FATAL_ERROR "AstralRT Unreal stream pump must avoid unavailable UE 5.7 StringBuilder include/token '${forbidden_unreal_plugin_stream_pump_text}'")
  endif()
endforeach()

set(unreal_compat_matrix_script "${ROOT}/scripts/run_unreal_compatibility_matrix.sh")
set(unreal_compat_matrix_gate "${ROOT}/tests/cmake/gate_unreal_compatibility_matrix.cmake")
if(NOT EXISTS "${unreal_compat_matrix_gate}")
  message(FATAL_ERROR "Unreal compatibility matrix gate is missing")
endif()
file(READ "${unreal_compat_matrix_script}" unreal_compat_matrix_script_text)
file(READ "${unreal_compat_matrix_gate}" unreal_compat_matrix_gate_text)
foreach(required_unreal_matrix_script_text
    "UNREAL_54_EDITOR"
    "UNREAL_55_EDITOR"
    "UNREAL_56_EDITOR"
    "UNREAL_57_EDITOR"
    "No Unreal versions ran"
    "Unsupported Unreal version")
  string(FIND "${unreal_compat_matrix_script_text}" "${required_unreal_matrix_script_text}" unreal_matrix_script_pos)
  if(unreal_matrix_script_pos EQUAL -1)
    message(FATAL_ERROR "Unreal compatibility matrix script is missing '${required_unreal_matrix_script_text}'")
  endif()
endforeach()
foreach(required_unreal_matrix_gate_text
    "gate_unreal_compatibility_matrix"
    "Missing UNREAL_54_EDITOR"
    "Unsupported Unreal version '5[.]8'"
    "No Unreal versions ran"
    "Skipping UE 5.7")
  string(FIND "${unreal_compat_matrix_gate_text}" "${required_unreal_matrix_gate_text}" unreal_matrix_gate_pos)
  if(unreal_matrix_gate_pos EQUAL -1)
    message(FATAL_ERROR "Unreal compatibility matrix gate is missing '${required_unreal_matrix_gate_text}'")
  endif()
endforeach()

set(test_cmake_file "${ROOT}/tests/CMakeLists.txt")
set(model_churn_soak_file "${ROOT}/tests/gate_model_churn_soak.cpp")
file(READ "${test_cmake_file}" test_cmake_content)
file(READ "${model_churn_soak_file}" model_churn_soak_content)
foreach(required_model_churn_cmake
    "gate_model_churn_soak"
    "RUN_SERIAL TRUE"
    "LABELS \"gate;memory;soak\"")
  string(FIND "${test_cmake_content}" "${required_model_churn_cmake}" model_churn_cmake_pos)
  if(model_churn_cmake_pos EQUAL -1)
    message(FATAL_ERROR "Model churn soak gate is not wired in CTest: missing '${required_model_churn_cmake}'")
  endif()
endforeach()
foreach(required_model_churn_text
    "ASTRAL_SOAK_MODEL"
    "ASTRAL_SOAK_MOCK_CYCLES"
    "ASTRAL_SOAK_REAL_CYCLES"
    "ASTRAL_SOAK_RSS_DRIFT_MB"
    "gate_model_churn_soak_real_model_probe"
    "[model_churn] backend="
    "rss_peak_kb="
    "model churn RSS drift exceeded")
  string(FIND "${model_churn_soak_content}" "${required_model_churn_text}" model_churn_text_pos)
  if(model_churn_text_pos EQUAL -1)
    message(FATAL_ERROR "Model churn soak gate is missing '${required_model_churn_text}'")
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

set(unreal_module_header "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Public/IAstralRT.h")
set(unreal_module_source "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralRuntimeModule.cpp")
file(READ "${unreal_module_header}" unreal_module_header_text)
file(READ "${unreal_module_source}" unreal_module_source_text)
foreach(required_unreal_allocator_header
    "FAstralRTAllocatorStats"
    "ResetAllocatorStats"
    "GetAllocatorStats")
  if(NOT unreal_module_header_text MATCHES "${required_unreal_allocator_header}")
    message(FATAL_ERROR "Unreal runtime interface is missing allocator stats surface: ${required_unreal_allocator_header}")
  endif()
endforeach()
foreach(required_unreal_allocator_impl
    "FMemory::Malloc"
    "FMemory::Free"
    "GAllocatorCounters"
    "AllocCalls.fetch_add"
    "FreeCalls.fetch_add")
  if(NOT unreal_module_source_text MATCHES "${required_unreal_allocator_impl}")
    message(FATAL_ERROR "Unreal runtime module no longer pins FMemory allocator accounting: ${required_unreal_allocator_impl}")
  endif()
endforeach()

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

set(unreal_types_header "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Public/AstralTypes.h")
set(unreal_model_source "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralModel.cpp")
file(READ "${unreal_types_header}" unreal_types_text)
file(READ "${unreal_model_source}" unreal_model_text)

foreach(required_unreal_type
    "EAstralUnrealPathRoot"
    "EAstralModelSourceKind SourceKind"
    "EAstralUnrealPathRoot PathRoot"
    "TArray<uint8> ModelBytes")
  if(NOT unreal_types_text MATCHES "${required_unreal_type}")
    message(FATAL_ERROR "Unreal model source wrapper is missing ${required_unreal_type}")
  endif()
endforeach()

foreach(required_unreal_model_text
    "FPaths::ProjectContentDir"
    "FPaths::ProjectSavedDir"
    "FPaths::ProjectPersistentDownloadDir"
    "Native.source_kind = static_cast<AstralModelSourceKind>"
    "Native.model_bytes.data = Desc.ModelBytes.GetData")
  if(NOT unreal_model_text MATCHES "${required_unreal_model_text}")
    message(FATAL_ERROR "Unreal model loader no longer preserves packaged-source handling: ${required_unreal_model_text}")
  endif()
endforeach()

set(unreal_automation_source "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/Tests/AstralRTTests.cpp")
file(READ "${unreal_automation_source}" unreal_automation_text)
string(FIND "${unreal_automation_text}" "->BeginDestroy()" unreal_automation_begin_destroy_pos)
if(NOT unreal_automation_begin_destroy_pos EQUAL -1)
  message(FATAL_ERROR "AstralRT Unreal Automation tests must use ConditionalBeginDestroy, not direct BeginDestroy")
endif()
string(FIND "${unreal_automation_text}" "ConditionalBeginDestroy()" unreal_automation_conditional_destroy_pos)
if(unreal_automation_conditional_destroy_pos EQUAL -1)
  message(FATAL_ERROR "AstralRT Unreal Automation tests must release UObjects through ConditionalBeginDestroy")
endif()
foreach(required_unreal_lifecycle_text
    "AstralRT.Mock.FailedLoadRecovery"
    "AstralRT.Mock.SessionCancelReset"
    "AstralRT.Mock.DestroyInvalidation"
    "AstralRT.Module.ShutdownRestart"
    "AstralRT.Module.EnginePreExit"
    "AstralRT.Module.EndPIE"
    "empty memory model load fails"
    "invalid model create rejected"
    "runtime reinitializes after startup"
    "runtime reinitializes after engine pre-exit"
    "runtime remains initialized after editor EndPIE"
    "Session->Cancel"
    "Session->Reset"
    "ASTRAL_E_CANCELED"
    "ASTRAL_E_INVALID")
  if(NOT unreal_automation_text MATCHES "${required_unreal_lifecycle_text}")
    message(FATAL_ERROR "Unreal lifecycle Automation coverage is missing ${required_unreal_lifecycle_text}")
  endif()
endforeach()
if(NOT unreal_automation_text MATCHES "#if WITH_EDITOR[^\n]*\nIMPLEMENT_SIMPLE_AUTOMATION_TEST\\(\n    FAstralRTModuleEndPIETest")
  message(FATAL_ERROR "AstralRT.Module.EndPIE Automation must stay behind WITH_EDITOR for packaged game targets")
endif()

set(unreal_runtime_module_source "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralRuntimeModule.cpp")
file(READ "${unreal_runtime_module_source}" unreal_runtime_module_text)
foreach(required_unreal_preexit_text
    "FCoreDelegates::OnPreExit"
    "HandleEnginePreExit"
    "SimulateEnginePreExitForAutomation"
    "ShutdownRuntime"
    "EnginePreExit"
    "ModuleUnload")
  if(NOT unreal_runtime_module_text MATCHES "${required_unreal_preexit_text}")
    message(FATAL_ERROR "Unreal runtime module is missing editor pre-exit lifecycle hook: ${required_unreal_preexit_text}")
  endif()
endforeach()

foreach(required_unreal_endpie_text
    "FEditorDelegates::EndPIE"
    "HandleEditorEndPIE"
    "SimulateEditorEndPIEForAutomation"
    "ShutdownRuntime[(]TEXT[(]\"EndPIE\"[)][)]"
    "InitializeRuntime[(][)]")
  if(NOT unreal_runtime_module_text MATCHES "${required_unreal_endpie_text}")
    message(FATAL_ERROR "Unreal runtime module is missing editor EndPIE lifecycle hook: ${required_unreal_endpie_text}")
  endif()
endforeach()

set(unreal_build_rules "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/AstralRT.Build.cs")
file(READ "${unreal_build_rules}" unreal_build_rules_text)
foreach(required_unreal_editor_dependency
    "Target.bBuildEditor"
    "PrivateDependencyModuleNames.Add[(]\"UnrealEd\"[)]")
  if(NOT unreal_build_rules_text MATCHES "${required_unreal_editor_dependency}")
    message(FATAL_ERROR "Unreal build rules are missing editor-only UnrealEd dependency: ${required_unreal_editor_dependency}")
  endif()
endforeach()

foreach(required_unreal_allocator_test
    "AstralRT.Memory.FMemoryAllocator"
    "model load uses FMemory allocator"
    "session create uses FMemory allocator"
    "native alloc callback called"
    "native free callback called")
  if(NOT unreal_automation_text MATCHES "${required_unreal_allocator_test}")
    message(FATAL_ERROR "Unreal FMemory allocator Automation coverage is missing ${required_unreal_allocator_test}")
  endif()
endforeach()

foreach(required_unreal_embedder_text
    "AstralRT.Mock.EmbedderQueuePressure"
    "overflow returns busy through wrapper"
    "collect out of order"
    "stale ticket rejected"
    "enqueue after drain")
  if(NOT unreal_automation_text MATCHES "${required_unreal_embedder_text}")
    message(FATAL_ERROR "Unreal embedder queue-pressure Automation coverage is missing ${required_unreal_embedder_text}")
  endif()
endforeach()

set(unreal_media_header "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Public/AstralMediaLibrary.h")
set(unreal_media_source "${ROOT}/plugins/unreal/AstralRT/Source/AstralRT/Private/AstralMediaLibrary.cpp")
file(READ "${unreal_media_header}" unreal_media_header_text)
file(READ "${unreal_media_source}" unreal_media_source_text)
foreach(required_unreal_media_text
    "UAstralMediaLibrary"
    "MakeRGBA8ImageFromBytes"
    "MakeRGBA8ImageFromTexture"
    "MakePCM16AudioFromBytes")
  if(NOT unreal_media_header_text MATCHES "${required_unreal_media_text}")
    message(FATAL_ERROR "Unreal media bridge header is missing ${required_unreal_media_text}")
  endif()
endforeach()
foreach(required_unreal_media_impl
    "PF_B8G8R8A8"
    "Mip.BulkData.LockReadOnly"
    "EAstralImageFormat::RGBA8"
    "EAstralAudioFormat::I16")
  if(NOT unreal_media_source_text MATCHES "${required_unreal_media_impl}")
    message(FATAL_ERROR "Unreal media bridge implementation is missing ${required_unreal_media_impl}")
  endif()
endforeach()
foreach(required_unreal_media_test
    "AstralRT.Media.DescriptorHelpers"
    "MakeRGBA8ImageFromBytes"
    "MakePCM16AudioFromBytes"
    "feed helper image"
    "feed helper audio")
  if(NOT unreal_automation_text MATCHES "${required_unreal_media_test}")
    message(FATAL_ERROR "Unreal media bridge Automation coverage is missing ${required_unreal_media_test}")
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
