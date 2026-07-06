#include "tooling.hpp"

#include "../core/handles.hpp"
#include "../core/runtime_alloc.hpp"
#include "../platform/compiler.hpp"

#include <cstring>

namespace astral::inference {

namespace {

constexpr uint32_t kMaxTools = 64u;
constexpr uint32_t kU32Max = 0xFFFFFFFFu;
constexpr uint32_t kJsonQuotedKeyMinOverhead = 4u;
constexpr uint32_t kJsonQuotedKeyScanOverhead = 3u;
constexpr char kJsonNameKey[] = "name";
constexpr char kJsonToolKey[] = "tool";
constexpr char kJsonArgumentsKey[] = "arguments";
constexpr uint32_t kJsonNameKeyLen = sizeof(kJsonNameKey) - 1u;
constexpr uint32_t kJsonToolKeyLen = sizeof(kJsonToolKey) - 1u;
constexpr uint32_t kJsonArgumentsKeyLen = sizeof(kJsonArgumentsKey) - 1u;
constexpr uint32_t kToolNameTagBytes = 4u;

inline bool choice_mode_valid(AstralToolChoiceMode mode) {
  return mode == ASTRAL_TOOL_CHOICE_AUTO || mode == ASTRAL_TOOL_CHOICE_REQUIRED ||
         mode == ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL;
}

inline bool span_valid(AstralSpanU8 span) {
  return span.data != nullptr && span.len != 0;
}

inline bool add_size(uint32_t a, uint32_t b, uint32_t* out) {
  if (out == nullptr || kU32Max - a < b) {
    return false;
  }
  *out = a + b;
  return true;
}

inline AstralSpanU8 tool_span(const Toolset* toolset, uint32_t off, uint32_t len) {
  AstralSpanU8 span{};
  if (toolset != nullptr && len != 0) {
    span.data = toolset->bytes + off;
    span.len = len;
  }
  return span;
}

inline bool span_equals(AstralSpanU8 a, const uint8_t* b, uint32_t b_len) {
  return a.len == b_len && (a.len == 0 || std::memcmp(a.data, b, a.len) == 0);
}

inline uint32_t name_tag(const uint8_t* data, uint32_t len) {
  if (ASTRAL_PREDICT_TRUE(len >= kToolNameTagBytes)) {
    uint32_t tag = 0;
    std::memcpy(&tag, data, sizeof(tag));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return __builtin_bswap32(tag);
#else
    return tag;
#endif
  }

  uint32_t tag = 0;
  for (uint32_t i = 0; i < len; ++i) {
    tag |= static_cast<uint32_t>(data[i]) << (i * 8u);
  }
  return tag;
}

inline bool is_ws(uint8_t c) {
  return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

bool find_json_tool_name_key(AstralSpanU8 text, uint32_t start, AstralSpanU8* out_value,
                             uint32_t* out_end, uint32_t* out_depth) {
  if (text.data == nullptr || out_value == nullptr || out_end == nullptr || out_depth == nullptr) {
    return false;
  }

  const uint8_t* data = text.data;
  uint32_t depth = 0;
  uint32_t p = start;
  while (p < text.len) {
    const uint8_t c = data[p];
    if (c == '"') {
      uint32_t key_len = 0;
      if (p + kJsonNameKeyLen + 1u < text.len &&
          std::memcmp(data + p + 1u, kJsonNameKey, kJsonNameKeyLen) == 0 &&
          data[p + kJsonNameKeyLen + 1u] == '"') {
        key_len = kJsonNameKeyLen;
      } else if (p + kJsonToolKeyLen + 1u < text.len &&
                 std::memcmp(data + p + 1u, kJsonToolKey, kJsonToolKeyLen) == 0 &&
                 data[p + kJsonToolKeyLen + 1u] == '"') {
        key_len = kJsonToolKeyLen;
      }

      if (key_len != 0) {
        uint32_t value_pos = p + key_len + 2u;
        while (value_pos < text.len && is_ws(data[value_pos])) {
          ++value_pos;
        }
        if (value_pos >= text.len || data[value_pos] != ':') {
          return false;
        }
        ++value_pos;
        while (value_pos < text.len && is_ws(data[value_pos])) {
          ++value_pos;
        }
        if (value_pos >= text.len || data[value_pos] != '"') {
          return false;
        }
        const uint32_t begin = ++value_pos;
        while (value_pos < text.len) {
          if (data[value_pos] == '\\') {
            value_pos += (value_pos + 1u < text.len) ? 2u : 1u;
            continue;
          }
          if (data[value_pos] == '"') {
            out_value->data = data + begin;
            out_value->len = value_pos - begin;
            *out_end = value_pos + 1u;
            *out_depth = depth;
            return true;
          }
          ++value_pos;
        }
        return false;
      }

      ++p;
      while (p < text.len) {
        if (data[p] == '\\') {
          p += (p + 1u < text.len) ? 2u : 1u;
          continue;
        }
        if (data[p] == '"') {
          ++p;
          break;
        }
        ++p;
      }
      continue;
    }
    if (c == '{') {
      ++depth;
    } else if (c == '}' && depth != 0) {
      --depth;
    }
    ++p;
  }
  return false;
}

bool find_json_object_key(AstralSpanU8 text, const char* key, uint32_t key_len,
                          AstralSpanU8* out_value) {
  if (text.data == nullptr || out_value == nullptr) {
    return false;
  }
  if (key_len == 0 || text.len < key_len + kJsonQuotedKeyMinOverhead) {
    return false;
  }

  const uint8_t* data = text.data;
  for (uint32_t i = 0; i + key_len + kJsonQuotedKeyScanOverhead < text.len; ++i) {
    if (data[i] != '"' || std::memcmp(data + i + 1u, key, key_len) != 0 ||
        data[i + key_len + 1u] != '"') {
      continue;
    }
    uint32_t p = i + key_len + 2u;
    while (p < text.len && is_ws(data[p])) {
      ++p;
    }
    if (p >= text.len || data[p] != ':') {
      continue;
    }
    ++p;
    while (p < text.len && is_ws(data[p])) {
      ++p;
    }
    if (p >= text.len || data[p] != '{') {
      return false;
    }

    const uint32_t begin = p;
    uint32_t depth = 0;
    bool in_string = false;
    while (p < text.len) {
      const uint8_t c = data[p];
      if (in_string) {
        if (c == '\\') {
          p += (p + 1u < text.len) ? 2u : 1u;
          continue;
        }
        if (c == '"') {
          in_string = false;
        }
        ++p;
        continue;
      }
      if (c == '"') {
        in_string = true;
      } else if (c == '{') {
        ++depth;
      } else if (c == '}') {
        if (depth == 0) {
          return false;
        }
        --depth;
        if (depth == 0) {
          out_value->data = data + begin;
          out_value->len = p + 1u - begin;
          return true;
        }
      }
      ++p;
    }
    return false;
  }
  return false;
}

ToolRecord* find_tool_by_name(Toolset* toolset, AstralSpanU8 name) {
  if (toolset == nullptr || name.data == nullptr) {
    return nullptr;
  }
  const uint32_t tag = name_tag(name.data, name.len);
  for (uint32_t i = 0; i < toolset->tool_count; ++i) {
    ToolRecord& tool = toolset->tools[i];
    if (tool.name_tag == tag && span_equals(name, toolset->bytes + tool.name_off, tool.name_len)) {
      return &tool;
    }
  }
  return nullptr;
}

} // namespace

AstralErr toolset_create(const AstralToolsetDesc* desc, Toolset** out_toolset) {
  if (desc == nullptr || out_toolset == nullptr || desc->size != sizeof(AstralToolsetDesc) ||
      desc->tools == nullptr || desc->tool_count == 0 || desc->tool_count > kMaxTools ||
      !choice_mode_valid(desc->choice_mode)) {
    return ASTRAL_E_INVALID;
  }

  uint32_t bytes_len = 0;
  for (uint32_t i = 0; i < desc->tool_count; ++i) {
    const AstralToolDesc& tool = desc->tools[i];
    if (tool.size != sizeof(AstralToolDesc) || !span_valid(tool.name) ||
        !span_valid(tool.json_schema)) {
      return ASTRAL_E_INVALID;
    }
    if (!add_size(bytes_len, tool.name.len, &bytes_len) ||
        !add_size(bytes_len, tool.description.len, &bytes_len) ||
        !add_size(bytes_len, tool.json_schema.len, &bytes_len)) {
      return ASTRAL_E_NOMEM;
    }
  }

  Toolset* toolset = core::runtime_new<Toolset>();
  if (toolset == nullptr) {
    return ASTRAL_E_NOMEM;
  }
  toolset->handle = 0;
  toolset->refcount.store(1, std::memory_order_relaxed);
  toolset->tool_count = desc->tool_count;
  toolset->choice_mode = desc->choice_mode;
  toolset->bytes_len = bytes_len;
  toolset->tools = core::runtime_alloc_array<ToolRecord>(desc->tool_count);
  toolset->bytes =
      bytes_len != 0 ? static_cast<uint8_t*>(core::runtime_alloc(bytes_len, 1)) : nullptr;
  if (toolset->tools == nullptr || (bytes_len != 0 && toolset->bytes == nullptr)) {
    if (toolset->tools != nullptr) {
      core::runtime_free_array(toolset->tools, desc->tool_count);
    }
    if (toolset->bytes != nullptr) {
      core::runtime_free(toolset->bytes, bytes_len, 1);
    }
    core::runtime_delete(toolset);
    return ASTRAL_E_NOMEM;
  }

  uint32_t cursor = 0;
  for (uint32_t i = 0; i < desc->tool_count; ++i) {
    const AstralToolDesc& src = desc->tools[i];
    ToolRecord& dst = toolset->tools[i];
    dst.tool_id = src.tool_id;
    dst.name_tag = name_tag(src.name.data, src.name.len);
    dst.name_off = cursor;
    dst.name_len = src.name.len;
    std::memcpy(toolset->bytes + cursor, src.name.data, src.name.len);
    cursor += src.name.len;
    dst.desc_off = cursor;
    dst.desc_len = src.description.len;
    if (src.description.len != 0) {
      std::memcpy(toolset->bytes + cursor, src.description.data, src.description.len);
      cursor += src.description.len;
    }
    dst.schema_off = cursor;
    dst.schema_len = src.json_schema.len;
    std::memcpy(toolset->bytes + cursor, src.json_schema.data, src.json_schema.len);
    cursor += src.json_schema.len;
  }

  const AstralHandle handle = core::register_handle(core::HandleKind::Toolset, toolset);
  if (handle == 0) {
    core::runtime_free_array(toolset->tools, toolset->tool_count);
    core::runtime_free(toolset->bytes, toolset->bytes_len, 1);
    core::runtime_delete(toolset);
    return ASTRAL_E_BUSY;
  }
  toolset->handle = handle;
  *out_toolset = toolset;
  return ASTRAL_OK;
}

void toolset_retain(Toolset* toolset) {
  if (toolset != nullptr) {
    toolset->refcount.fetch_add(1, std::memory_order_relaxed);
  }
}

void toolset_release(Toolset* toolset) {
  if (toolset == nullptr) {
    return;
  }
  const uint32_t prev = toolset->refcount.fetch_sub(1, std::memory_order_acq_rel);
  if (prev != 1) {
    return;
  }
  core::unregister_handle(toolset->handle, core::HandleKind::Toolset);
  toolset->handle = 0;
  core::runtime_free_array(toolset->tools, toolset->tool_count);
  core::runtime_free(toolset->bytes, toolset->bytes_len, 1);
  core::runtime_delete(toolset);
}

AstralErr toolset_count(Toolset* toolset, uint32_t* out_count) {
  if (toolset == nullptr || out_count == nullptr) {
    return ASTRAL_E_INVALID;
  }
  *out_count = toolset->tool_count;
  return ASTRAL_OK;
}

AstralErr toolset_get(Toolset* toolset, uint32_t index, AstralToolInfo* out_info) {
  if (toolset == nullptr || out_info == nullptr || out_info->size != sizeof(AstralToolInfo)) {
    return ASTRAL_E_INVALID;
  }
  if (index >= toolset->tool_count) {
    return ASTRAL_E_NOT_FOUND;
  }
  const ToolRecord& tool = toolset->tools[index];
  out_info->tool_id = tool.tool_id;
  out_info->name = tool_span(toolset, tool.name_off, tool.name_len);
  out_info->description = tool_span(toolset, tool.desc_off, tool.desc_len);
  out_info->json_schema = tool_span(toolset, tool.schema_off, tool.schema_len);
  return ASTRAL_OK;
}

AstralErr toolset_parse_call(Toolset* toolset, AstralSpanU8 generated_text,
                             AstralToolCallResult* out_result) {
  if (toolset == nullptr || generated_text.data == nullptr || generated_text.len == 0 ||
      out_result == nullptr || out_result->size != sizeof(AstralToolCallResult)) {
    return ASTRAL_E_INVALID;
  }

  out_result->tool_id = 0;
  out_result->parse_status = ASTRAL_E_NOT_FOUND;
  out_result->name = {};
  out_result->arguments_json = {};

  ToolRecord* tool = nullptr;
  ToolRecord* first_known_tool = nullptr;
  AstralSpanU8 best_arguments{};
  ToolRecord* best_tool = nullptr;
  uint32_t best_depth = kU32Max;
  uint32_t scan = 0;
  while (scan < generated_text.len) {
    AstralSpanU8 candidate_name{};
    uint32_t name_end = 0;
    uint32_t name_depth = 0;
    if (!find_json_tool_name_key(generated_text, scan, &candidate_name, &name_end, &name_depth)) {
      break;
    }
    ToolRecord* candidate_tool = find_tool_by_name(toolset, candidate_name);
    if (candidate_tool != nullptr) {
      AstralSpanU8 argument_scan = generated_text;
      argument_scan.data = generated_text.data + name_end;
      argument_scan.len = generated_text.len - name_end;
      AstralSpanU8 candidate_arguments{};
      const bool has_arguments =
          find_json_object_key(argument_scan, kJsonArgumentsKey, kJsonArgumentsKeyLen,
                               &candidate_arguments) ||
          (name_depth == 1u && find_json_object_key(generated_text, kJsonArgumentsKey,
                                                    kJsonArgumentsKeyLen, &candidate_arguments));
      if (has_arguments) {
        if (best_tool == nullptr || name_depth < best_depth) {
          best_tool = candidate_tool;
          best_arguments = candidate_arguments;
          best_depth = name_depth;
          if (name_depth <= 1u) {
            break;
          }
        }
      } else if (first_known_tool == nullptr) {
        first_known_tool = candidate_tool;
      }
    }
    scan = name_end;
  }

  if (tool == nullptr && best_tool != nullptr) {
    tool = best_tool;
    out_result->arguments_json = best_arguments;
  }
  if (tool == nullptr && first_known_tool != nullptr) {
    tool = first_known_tool;
    out_result->parse_status = ASTRAL_E_INVALID;
  }
  if (tool == nullptr) {
    return ASTRAL_E_NOT_FOUND;
  }

  out_result->tool_id = tool->tool_id;
  out_result->name = tool_span(toolset, tool->name_off, tool->name_len);
  if (out_result->arguments_json.data != nullptr) {
    out_result->parse_status = ASTRAL_OK;
  }
  return ASTRAL_OK;
}

} // namespace astral::inference
