#pragma once

#include "../../include/astral_rt.h"

namespace astral::inference {

AstralErr chunk_count(const AstralChunkerDesc* desc, AstralSpanU8 text, uint32_t* out_count);
AstralErr chunk_ranges(const AstralChunkerDesc* desc, AstralSpanU8 text,
                       AstralChunkRange* out_ranges, uint32_t max_ranges, uint32_t* out_count);
AstralErr chunk_text_copy(AstralSpanU8 text, const AstralChunkRange* range,
                          AstralMutSpanU8 out_text, uint32_t* out_len);
AstralErr token_chunk_count(const AstralChunkerDesc* desc, uint32_t token_count,
                            uint32_t* out_count);
AstralErr token_chunk_ranges(const AstralChunkerDesc* desc, uint32_t token_count,
                             AstralChunkRange* out_ranges, uint32_t max_ranges,
                             uint32_t* out_count);

} // namespace astral::inference
