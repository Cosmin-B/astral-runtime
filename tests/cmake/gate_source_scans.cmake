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

set(UNREVIEWED_PROSE_STRINGS
  "${ai_generation_phrase}"
  "${automated_tools_phrase}"
  "${proper_cleanup_phrase}"
  "${proper_lifetime_phrase}"
  "${raii_pattern_phrase}"
  "${always_dispose_phrase}"
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
  endforeach()
endforeach()

message(STATUS "gate_source_scans: OK (${FILES_LEN} files)")
