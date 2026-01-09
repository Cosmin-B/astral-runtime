#include "chunking.hpp"

#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace astral::inference {

namespace {

constexpr uint32_t kNoTokenRange = 0xFFFFFFFFu;
constexpr uint32_t kUtf8AsciiMask = 0x80u;
constexpr uint32_t kUtf8TwoByteMask = 0xE0u;
constexpr uint32_t kUtf8TwoByteLead = 0xC0u;
constexpr uint32_t kUtf8ThreeByteMask = 0xF0u;
constexpr uint32_t kUtf8ThreeByteLead = 0xE0u;
constexpr uint32_t kUtf8FourByteMask = 0xF8u;
constexpr uint32_t kUtf8FourByteLead = 0xF0u;
constexpr uint8_t kAsciiWhitespaceMax = ' ';
constexpr uint64_t kByteHighBits = 0x8080808080808080ull;
constexpr uint64_t kAsciiWhitespaceLimitBytes = 0x2121212121212121ull;
constexpr uint32_t kBitsPerByte = 8u;
constexpr uint8_t kDefaultSentenceDelimiters[] = {'.', '!', '?', '\n'};

struct UnitRange {
  uint32_t begin;
  uint32_t end;
};

inline bool desc_valid(const AstralChunkerDesc* desc) {
  if (desc == nullptr || desc->size != sizeof(AstralChunkerDesc) || desc->max_units == 0) {
    return false;
  }
  if (desc->overlap_units >= desc->max_units) {
    return false;
  }
  return desc->mode == ASTRAL_CHUNK_MODE_NONE || desc->mode == ASTRAL_CHUNK_MODE_CHAR ||
         desc->mode == ASTRAL_CHUNK_MODE_WORD || desc->mode == ASTRAL_CHUNK_MODE_SENTENCE ||
         desc->mode == ASTRAL_CHUNK_MODE_TOKEN;
}

inline bool text_valid(AstralSpanU8 text) {
  return text.data != nullptr || text.len == 0;
}

inline bool is_space(uint8_t c) {
  return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v';
}

inline bool can_be_space(uint8_t c) {
  return c <= kAsciiWhitespaceMax;
}

inline uint64_t ascii_space_candidate_mask(uint64_t word) {
  return (word - kAsciiWhitespaceLimitBytes) & ~word & kByteHighBits;
}

inline uint32_t trailing_zero_bits(uint64_t mask) {
#if defined(_MSC_VER)
  unsigned long index = 0;
  _BitScanForward64(&index, mask);
  return static_cast<uint32_t>(index);
#else
  return static_cast<uint32_t>(__builtin_ctzll(mask));
#endif
}

uint32_t find_ascii_space_candidate(AstralSpanU8 text, uint32_t pos) {
  while (pos + sizeof(uint64_t) <= text.len) {
    uint64_t word = 0;
    std::memcpy(&word, text.data + pos, sizeof(word));
    const uint64_t mask = ascii_space_candidate_mask(word);
    if (mask != 0) {
      return pos + trailing_zero_bits(mask) / kBitsPerByte;
    }
    pos += static_cast<uint32_t>(sizeof(uint64_t));
  }
  while (pos < text.len && !can_be_space(text.data[pos])) {
    ++pos;
  }
  return pos;
}

inline uint32_t utf8_step(AstralSpanU8 text, uint32_t pos) {
  if (pos >= text.len) {
    return pos;
  }

  const uint8_t c = text.data[pos];
  uint32_t width = 1;
  if ((c & kUtf8AsciiMask) == 0) {
    return pos + 1u;
  }
  if ((c & kUtf8TwoByteMask) == kUtf8TwoByteLead) {
    width = 2;
  } else if ((c & kUtf8ThreeByteMask) == kUtf8ThreeByteLead) {
    width = 3;
  } else if ((c & kUtf8FourByteMask) == kUtf8FourByteLead) {
    width = 4;
  }

  return (text.len - pos >= width) ? pos + width : pos + 1u;
}

inline bool is_sentence_delim(const AstralChunkerDesc* desc, uint8_t c) {
  if (desc->delimiters.data != nullptr && desc->delimiters.len != 0) {
    for (uint32_t i = 0; i < desc->delimiters.len; ++i) {
      if (desc->delimiters.data[i] == c) {
        return true;
      }
    }
    return false;
  }

  for (uint32_t i = 0; i < sizeof(kDefaultSentenceDelimiters); ++i) {
    if (kDefaultSentenceDelimiters[i] == c) {
      return true;
    }
  }
  return false;
}

bool next_char_unit(AstralSpanU8 text, uint32_t* pos, UnitRange* out) {
  if (*pos >= text.len) {
    return false;
  }
  out->begin = *pos;
  out->end = utf8_step(text, *pos);
  *pos = out->end;
  return true;
}

bool next_word_unit(AstralSpanU8 text, uint32_t* pos, UnitRange* out) {
  uint32_t p = *pos;
  while (p < text.len) {
    const uint8_t c = text.data[p];
    if (!can_be_space(c) || !is_space(c)) {
      break;
    }
    ++p;
  }
  if (p >= text.len) {
    *pos = p;
    return false;
  }

  out->begin = p;
  while (p < text.len) {
    p = find_ascii_space_candidate(text, p);
    if (p >= text.len) {
      break;
    }
    if (is_space(text.data[p])) {
      break;
    }
    ++p;
  }
  out->end = p;
  *pos = p;
  return true;
}

bool next_sentence_unit(const AstralChunkerDesc* desc, AstralSpanU8 text, uint32_t* pos,
                        UnitRange* out) {
  uint32_t p = *pos;
  while (p < text.len && is_space(text.data[p])) {
    ++p;
  }
  if (p >= text.len) {
    *pos = p;
    return false;
  }

  out->begin = p;
  while (p < text.len) {
    const uint8_t c = text.data[p];
    p = utf8_step(text, p);
    if (is_sentence_delim(desc, c)) {
      break;
    }
  }
  out->end = p;
  *pos = p;
  return true;
}

bool next_unit(const AstralChunkerDesc* desc, AstralSpanU8 text, uint32_t* pos, UnitRange* out) {
  switch (desc->mode) {
  case ASTRAL_CHUNK_MODE_CHAR:
    return next_char_unit(text, pos, out);
  case ASTRAL_CHUNK_MODE_WORD:
    return next_word_unit(text, pos, out);
  case ASTRAL_CHUNK_MODE_SENTENCE:
    return next_sentence_unit(desc, text, pos, out);
  default:
    return false;
  }
}

inline void fill_text_range(const AstralChunkerDesc* desc, uint32_t chunk_id, uint32_t byte_begin,
                            uint32_t byte_end, AstralChunkRange* out) {
  out->size = sizeof(AstralChunkRange);
  out->document_id = desc->document_id;
  out->chunk_id = chunk_id;
  out->group_id = desc->group_id;
  out->byte_begin = byte_begin;
  out->byte_end = byte_end;
  out->token_begin = kNoTokenRange;
  out->token_end = kNoTokenRange;
}

inline void fill_token_range(const AstralChunkerDesc* desc, uint32_t chunk_id, uint32_t token_begin,
                             uint32_t token_end, AstralChunkRange* out) {
  out->size = sizeof(AstralChunkRange);
  out->document_id = desc->document_id;
  out->chunk_id = chunk_id;
  out->group_id = desc->group_id;
  out->byte_begin = 0;
  out->byte_end = 0;
  out->token_begin = token_begin;
  out->token_end = token_end;
}

AstralErr emit_word_chunks(const AstralChunkerDesc* desc, AstralSpanU8 text,
                           AstralChunkRange* out_ranges, uint32_t max_ranges,
                           uint32_t* out_count) {
  const uint32_t max_units = desc->max_units;
  const uint32_t overlap_units = desc->overlap_units;
  uint32_t required = 0;
  uint32_t pos = 0;
  while (pos < text.len) {
    UnitRange first{};
    uint32_t scan = pos;
    if (!next_word_unit(text, &scan, &first)) {
      break;
    }

    const uint32_t chunk_begin = first.begin;
    uint32_t chunk_end = first.end;
    uint32_t units = 1;
    uint32_t overlap_begin = first.begin;

    while (units < max_units) {
      UnitRange unit{};
      const uint32_t before = scan;
      if (!next_word_unit(text, &scan, &unit)) {
        scan = before;
        break;
      }
      if (overlap_units != 0 && units + overlap_units == max_units) {
        overlap_begin = unit.begin;
      }
      chunk_end = unit.end;
      ++units;
    }

    if (out_ranges != nullptr && required < max_ranges) {
      fill_text_range(desc, required, chunk_begin, chunk_end, &out_ranges[required]);
    }
    ++required;

    if (scan >= text.len) {
      break;
    }
    pos = overlap_units != 0 ? overlap_begin : scan;
  }

  *out_count = required;
  return (out_ranges != nullptr && max_ranges < required) ? ASTRAL_E_NOMEM : ASTRAL_OK;
}

AstralErr emit_text_chunks(const AstralChunkerDesc* desc, AstralSpanU8 text,
                           AstralChunkRange* out_ranges, uint32_t max_ranges, uint32_t* out_count) {
  uint32_t required = 0;
  if (desc->mode == ASTRAL_CHUNK_MODE_NONE) {
    const bool keep_empty = (desc->flags & ASTRAL_CHUNK_FLAG_KEEP_EMPTY) != 0;
    if (text.len != 0 || keep_empty) {
      required = 1;
      if (out_ranges != nullptr && max_ranges != 0) {
        fill_text_range(desc, 0, 0, text.len, &out_ranges[0]);
      }
    }
    *out_count = required;
    return (out_ranges != nullptr && max_ranges < required) ? ASTRAL_E_NOMEM : ASTRAL_OK;
  }

  if (desc->mode == ASTRAL_CHUNK_MODE_WORD) {
    return emit_word_chunks(desc, text, out_ranges, max_ranges, out_count);
  }

  uint32_t pos = 0;
  while (pos < text.len) {
    UnitRange first{};
    uint32_t scan = pos;
    if (!next_unit(desc, text, &scan, &first)) {
      break;
    }

    const uint32_t chunk_begin = first.begin;
    uint32_t chunk_end = first.end;
    uint32_t units = 1;
    uint32_t overlap_begin = first.begin;

    while (units < desc->max_units) {
      UnitRange unit{};
      const uint32_t before = scan;
      if (!next_unit(desc, text, &scan, &unit)) {
        scan = before;
        break;
      }
      if (desc->overlap_units != 0 && units + desc->overlap_units == desc->max_units) {
        overlap_begin = unit.begin;
      }
      chunk_end = unit.end;
      ++units;
    }

    if (out_ranges != nullptr && required < max_ranges) {
      fill_text_range(desc, required, chunk_begin, chunk_end, &out_ranges[required]);
    }
    ++required;

    if (scan >= text.len) {
      break;
    }
    pos = desc->overlap_units != 0 ? overlap_begin : scan;
  }

  *out_count = required;
  return (out_ranges != nullptr && max_ranges < required) ? ASTRAL_E_NOMEM : ASTRAL_OK;
}

AstralErr emit_token_chunks(const AstralChunkerDesc* desc, uint32_t token_count,
                            AstralChunkRange* out_ranges, uint32_t max_ranges,
                            uint32_t* out_count) {
  uint32_t required = 0;
  uint32_t begin = 0;
  while (begin < token_count) {
    uint32_t end = begin + desc->max_units;
    if (end > token_count) {
      end = token_count;
    }
    if (out_ranges != nullptr && required < max_ranges) {
      fill_token_range(desc, required, begin, end, &out_ranges[required]);
    }
    ++required;
    if (end >= token_count) {
      break;
    }
    begin = desc->overlap_units != 0 ? end - desc->overlap_units : end;
  }

  *out_count = required;
  return (out_ranges != nullptr && max_ranges < required) ? ASTRAL_E_NOMEM : ASTRAL_OK;
}

} // namespace

AstralErr chunk_count(const AstralChunkerDesc* desc, AstralSpanU8 text, uint32_t* out_count) {
  if (!desc_valid(desc) || desc->mode == ASTRAL_CHUNK_MODE_TOKEN || !text_valid(text) ||
      out_count == nullptr) {
    return ASTRAL_E_INVALID;
  }
  return emit_text_chunks(desc, text, nullptr, 0, out_count);
}

AstralErr chunk_ranges(const AstralChunkerDesc* desc, AstralSpanU8 text,
                       AstralChunkRange* out_ranges, uint32_t max_ranges, uint32_t* out_count) {
  if (!desc_valid(desc) || desc->mode == ASTRAL_CHUNK_MODE_TOKEN || !text_valid(text) ||
      out_count == nullptr) {
    return ASTRAL_E_INVALID;
  }
  if (max_ranges != 0 && out_ranges == nullptr) {
    return ASTRAL_E_INVALID;
  }
  return emit_text_chunks(desc, text, out_ranges, max_ranges, out_count);
}

AstralErr chunk_text_copy(AstralSpanU8 text, const AstralChunkRange* range,
                          AstralMutSpanU8 out_text, uint32_t* out_len) {
  if (!text_valid(text) || range == nullptr || range->size != sizeof(AstralChunkRange) ||
      out_len == nullptr) {
    return ASTRAL_E_INVALID;
  }
  if (range->byte_begin > range->byte_end || range->byte_end > text.len) {
    return ASTRAL_E_INVALID;
  }

  const uint32_t len = range->byte_end - range->byte_begin;
  *out_len = len;
  if (out_text.data == nullptr || out_text.len < len) {
    return ASTRAL_E_NOMEM;
  }
  if (len != 0) {
    std::memcpy(out_text.data, text.data + range->byte_begin, len);
  }
  return ASTRAL_OK;
}

AstralErr token_chunk_count(const AstralChunkerDesc* desc, uint32_t token_count,
                            uint32_t* out_count) {
  if (!desc_valid(desc) || desc->mode != ASTRAL_CHUNK_MODE_TOKEN || out_count == nullptr) {
    return ASTRAL_E_INVALID;
  }
  return emit_token_chunks(desc, token_count, nullptr, 0, out_count);
}

AstralErr token_chunk_ranges(const AstralChunkerDesc* desc, uint32_t token_count,
                             AstralChunkRange* out_ranges, uint32_t max_ranges,
                             uint32_t* out_count) {
  if (!desc_valid(desc) || desc->mode != ASTRAL_CHUNK_MODE_TOKEN || out_count == nullptr) {
    return ASTRAL_E_INVALID;
  }
  if (max_ranges != 0 && out_ranges == nullptr) {
    return ASTRAL_E_INVALID;
  }
  return emit_token_chunks(desc, token_count, out_ranges, max_ranges, out_count);
}

} // namespace astral::inference
