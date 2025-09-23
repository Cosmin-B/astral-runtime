if(NOT DEFINED ASTRAL_SOURCE_DIR)
  message(FATAL_ERROR "ASTRAL_SOURCE_DIR not set")
endif()

set(ROOT "${ASTRAL_SOURCE_DIR}")

set(SCAN_DIRS
  include
  src
  tests
  benchmarks
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

foreach(path IN LISTS FILES)
  file(READ "${path}" content)

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
  endforeach()
endforeach()

message(STATUS "gate_source_scans: OK (${FILES_LEN} files)")
