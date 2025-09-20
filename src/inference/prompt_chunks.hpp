#pragma once

#include "../../include/astral_rt.h"
#include "../core/runtime_alloc.hpp"

#include <cstdint>

namespace astral::inference {

enum class PromptChunkKind : uint8_t {
    Text = 0,
    Image = 1,
    Audio = 2,
};

struct PromptChunk {
    PromptChunkKind kind;
    uint8_t finalize;
    uint16_t _padding0;
    uint32_t token_start;
    uint32_t token_count;
    AstralImageDesc image;
    AstralAudioDesc audio;
    uint8_t* owned_buffer;
    uint64_t owned_bytes;
    uint32_t owned_align;
};

constexpr uint32_t kMaxPromptChunks = 128u;

inline void prompt_chunk_release(PromptChunk& chunk) {
    if (chunk.owned_buffer != nullptr && chunk.owned_bytes > 0) {
        core::runtime_free(chunk.owned_buffer, static_cast<size_t>(chunk.owned_bytes), chunk.owned_align);
    }
    chunk.owned_buffer = nullptr;
    chunk.owned_bytes = 0;
    chunk.owned_align = 1;
}

inline void prompt_chunk_reset(PromptChunk& chunk) {
    prompt_chunk_release(chunk);
    chunk = PromptChunk{};
}

} // namespace astral::inference
