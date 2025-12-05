#pragma once

#include "../../include/astral_rt.h"

#include <atomic>
#include <cstdint>

namespace astral::inference {

struct ToolRecord {
    uint32_t tool_id;
    uint32_t name_off;
    uint32_t name_len;
    uint32_t desc_off;
    uint32_t desc_len;
    uint32_t schema_off;
    uint32_t schema_len;
};

struct Toolset {
    AstralHandle handle;
    std::atomic<uint32_t> refcount;
    uint32_t tool_count;
    AstralToolChoiceMode choice_mode;
    ToolRecord* tools;
    uint8_t* bytes;
    uint32_t bytes_len;
};

AstralErr toolset_create(const AstralToolsetDesc* desc, Toolset** out_toolset);
void toolset_retain(Toolset* toolset);
void toolset_release(Toolset* toolset);
AstralErr toolset_count(Toolset* toolset, uint32_t* out_count);
AstralErr toolset_get(Toolset* toolset, uint32_t index, AstralToolInfo* out_info);
AstralErr toolset_parse_call(Toolset* toolset, AstralSpanU8 generated_text, AstralToolCallResult* out_result);

} // namespace astral::inference
