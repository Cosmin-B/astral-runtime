/**
 * mock_backend.cpp - Mock backend provider (no llama dependency)
 *
 * Purpose:
 * - Validate Astral's provider-agnostic backend contract.
 * - Exercise the core decode + sampler loop without relying on llama.cpp.
 *
 * This backend is intentionally tiny and deterministic:
 * - Tokenization: 1 byte -> 1 token (0..255), optional BOS.
 * - Detokenization: token 0..255 -> byte, BOS/EOS -> empty.
 * - Logits: argmax points to the next byte in a fixed message, then EOS.
 */

#include "../backend.hpp"
#include "../../core/runtime_alloc.hpp"
#include "../../core/model_sources.hpp"

#include <cstring>
#include <limits>

namespace astral::backend {

namespace {

struct MockModel {
    uint32_t vocab_size;
    uint32_t ctx_size;
    uint32_t emb_dim;
    int32_t token_bos;
    int32_t token_eos;
    bool infinite;
    bool sampler_mode;
    bool tool_call_mode;
    bool long_piece_mode;
};

struct MockSession {
    MockModel* model;
    uint32_t step;
    uint32_t slot_id;
    uint32_t n_slots;
    uint32_t slot_step[32];
    uint32_t slot_output_step[32];
    bool slot_has_logits[32];
    uint32_t adapter_count;
    uint32_t grammar_enabled;
    uint32_t grammar_allow_token; // 0..255, or 0xFFFFFFFF for none
    uint32_t slot_grammar_enabled[32];
    uint32_t slot_grammar_allow_token[32];
    bool has_logits;
    float* logits;
    uint32_t last_batch_outputs;
    float* batch_logits;
};

struct MockEmbedder {
    MockModel* model;
};

constexpr uint32_t kMockVocabSize = 260;
constexpr int32_t kMockTokenBos = 256;
constexpr int32_t kMockTokenEos = 257;
constexpr int32_t kMockTokenLongPiece = 258;
constexpr uint32_t kMockEmbDim = 8;
constexpr uint32_t kMockMaxSlots = 32;
constexpr uint32_t kMockMaxBatchOutputs = 64;

constexpr const char* kMockMessage = "mock-backend";
constexpr const char* kMockToolCallMessage = "{\"name\":\"search\",\"arguments\":{\"query\":\"agent\"}}";
constexpr char kMockLongPiecePattern[] =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
constexpr uint32_t kMockLongPieceBytes = 9000;

inline uint8_t mock_long_piece_byte(uint32_t offset) {
  constexpr uint32_t pattern_len = sizeof(kMockLongPiecePattern) - 1u;
  return static_cast<uint8_t>(kMockLongPiecePattern[offset % pattern_len]);
}

inline bool span_equals_ascii(AstralSpanU8 span, const char* lit) {
    if (lit == nullptr) {
        return false;
    }
    const size_t lit_len = std::strlen(lit);
    if (span.data == nullptr || span.len != lit_len) {
        return false;
    }
    return std::memcmp(span.data, lit, lit_len) == 0;
}

inline uint32_t sum_bytes(const uint8_t* data, uint64_t len) {
    if (data == nullptr || len == 0) {
        return 0u;
    }
    uint32_t sum = 0;
    for (uint64_t i = 0; i < len; ++i) {
        sum += data[i];
    }
    return sum;
}

static bool parse_astral_source_id(AstralSpanU8 path, uint64_t* out_id) {
    if (out_id == nullptr) {
        return false;
    }
    *out_id = 0;

    if (path.data == nullptr || path.len == 0) {
        return false;
    }

    static constexpr char kPrefix[] = "astral-src:";
    static constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
    if (path.len <= kPrefixLen) {
        return false;
    }

    if (std::memcmp(path.data, kPrefix, kPrefixLen) != 0) {
        return false;
    }

    uint64_t id = 0;
    const uint8_t* p = path.data + kPrefixLen;
    const uint8_t* end = path.data + path.len;
    for (; p < end; ++p) {
        const uint8_t c = *p;
        if (c < static_cast<uint8_t>('0') || c > static_cast<uint8_t>('9')) {
            return false;
        }
        const uint64_t next = id * 10u + static_cast<uint64_t>(c - static_cast<uint8_t>('0'));
        if (next < id) {
            return false;
        }
        id = next;
    }

    if (id == 0) {
        return false;
    }

    *out_id = id;
    return true;
}

void* mock_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }

    if (desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    AstralSpanU8 tag = desc->model_path;
    uint64_t src_id = 0;
    if (parse_astral_source_id(desc->model_path, &src_id)) {
        astral::core::ModelSource src{};
        if (!astral::core::model_source_take(src_id, &src)) {
            *out_err = ASTRAL_E_INVALID;
            return nullptr;
        }

        if (src.kind == ASTRAL_MODEL_SOURCE_MEMORY) {
            tag = src.bytes;
        } else if (src.kind == ASTRAL_MODEL_SOURCE_IO) {
            uint8_t tmp[64];
            const uint64_t size = src.io.size ? src.io.size(src.io.user) : 0;
            if (size == 0 || src.io.read_at == nullptr) {
                *out_err = ASTRAL_E_INVALID;
                return nullptr;
            }
            const uint32_t n = src.io.read_at(src.io.user, 0, tmp, static_cast<uint32_t>(sizeof(tmp)));
            if (n == 0) {
                *out_err = ASTRAL_E_INVALID;
                return nullptr;
            }
            tag.data = tmp;
            tag.len = n;
        } else {
            *out_err = ASTRAL_E_UNSUPPORTED;
            return nullptr;
        }
    }

    MockModel* model = astral::core::runtime_new<MockModel>();
    if (model == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    model->vocab_size = kMockVocabSize;
    model->ctx_size = desc->n_ctx != 0 ? desc->n_ctx : 2048;
    model->emb_dim = kMockEmbDim;
    model->token_bos = kMockTokenBos;
    model->infinite = span_equals_ascii(tag, "infinite");
    model->sampler_mode = span_equals_ascii(tag, "sampler");
    model->tool_call_mode = span_equals_ascii(tag, "toolcall");
    model->long_piece_mode = span_equals_ascii(tag, "long-piece");
    model->token_eos = (model->infinite || model->sampler_mode) ? -1 : kMockTokenEos;

    *out_err = ASTRAL_OK;
    return model;
}

void mock_model_unload(void* model_ctx) {
    astral::core::runtime_delete(static_cast<MockModel*>(model_ctx));
}

AstralErr mock_tokenize(void* model_ctx, AstralSpanU8 text,
                        int32_t* out_tokens, uint32_t max_tokens,
                        uint8_t add_special, uint8_t parse_special,
                        uint32_t* out_count) {
    (void)parse_special;

    if (model_ctx == nullptr || out_count == nullptr) {
        return ASTRAL_E_INVALID;
    }

    uint32_t needed = text.len + (add_special != 0 ? 1u : 0u);
    *out_count = needed;
    if (out_tokens == nullptr && max_tokens == 0) {
        return ASTRAL_OK;
    }
    if (needed > max_tokens) {
        *out_count = needed;
        return ASTRAL_E_NOMEM;
    }

    uint32_t n = 0;
    if (add_special != 0) {
        out_tokens[n++] = kMockTokenBos;
    }

    for (uint32_t i = 0; i < text.len; ++i) {
        out_tokens[n++] = static_cast<int32_t>(text.data ? text.data[i] : 0);
    }

    *out_count = n;
    return ASTRAL_OK;
}

AstralErr mock_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count,
                          AstralMutSpanU8 out_text, uint32_t* out_len) {
    if (model_ctx == nullptr || (tokens == nullptr && count != 0) || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t t = tokens[i];
        if (t == kMockTokenBos || t == kMockTokenEos) {
            continue;
        }

        if (t == kMockTokenLongPiece) {
          if (out_text.data != nullptr &&
              (offset > out_text.len || kMockLongPieceBytes > out_text.len - offset)) {
            *out_len = offset + kMockLongPieceBytes;
            return ASTRAL_E_NOMEM;
          }
          if (out_text.data != nullptr) {
            for (uint32_t j = 0; j < kMockLongPieceBytes; ++j) {
              out_text.data[offset + j] = mock_long_piece_byte(j);
            }
          }
          offset += kMockLongPieceBytes;
          continue;
        }

        if (out_text.data != nullptr && offset >= out_text.len) {
            return ASTRAL_E_NOMEM;
        }

        const uint8_t b = (t >= 0 && t <= 255) ? static_cast<uint8_t>(t) : static_cast<uint8_t>('?');
        if (out_text.data != nullptr) {
            out_text.data[offset] = b;
        }
        offset++;
    }

    *out_len = offset;
    return ASTRAL_OK;
}

AstralErr mock_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    if (model_ctx == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockModel* model = static_cast<MockModel*>(model_ctx);
    *out_vocab_size = model->vocab_size;
    *out_ctx_size = model->ctx_size;
    return ASTRAL_OK;
}

AstralErr mock_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    if (model_ctx == nullptr || out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockModel* model = static_cast<MockModel*>(model_ctx);
    *out_bos = model->token_bos;
    *out_eos = model->token_eos;
    return ASTRAL_OK;
}

AstralErr mock_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
    if (model_ctx == nullptr || out_dim == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockModel* model = static_cast<MockModel*>(model_ctx);
    *out_dim = model->emb_dim;
    return ASTRAL_OK;
}

AstralErr mock_model_media_init(void* model_ctx, const AstralModelMediaDesc* desc) {
    if (model_ctx == nullptr || desc == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralModelMediaDesc)) {
        return ASTRAL_E_INVALID;
    }
    return ASTRAL_OK;
}

AstralErr mock_model_media_info(void* model_ctx, AstralMediaInfo* out_info) {
    if (model_ctx == nullptr || out_info == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (out_info->size != sizeof(AstralMediaInfo)) {
        return ASTRAL_E_INVALID;
    }
    out_info->supports_image = 1;
    out_info->supports_audio = 1;
    out_info->audio_sample_rate = 16000;
    out_info->image_min_tokens = 1;
    out_info->image_max_tokens = 16;
    return ASTRAL_OK;
}

void* mock_session_create_ex(void* model_ctx, const AstralSessionDesc* desc, uint32_t max_slots, AstralErr* out_err);

void* mock_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
    return mock_session_create_ex(model_ctx, desc, /*max_slots=*/kMockMaxSlots, out_err);
}

void* mock_session_create_ex(void* model_ctx, const AstralSessionDesc* desc, uint32_t max_slots, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }

    if (model_ctx == nullptr || desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    MockModel* model = static_cast<MockModel*>(model_ctx);

    MockSession* session = astral::core::runtime_new<MockSession>();
    if (session == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    session->logits = astral::core::runtime_alloc_array<float>(model->vocab_size);
    session->batch_logits = astral::core::runtime_alloc_array<float>(model->vocab_size * kMockMaxBatchOutputs);
    if (session->logits == nullptr || session->batch_logits == nullptr) {
        astral::core::runtime_free_array(session->logits, model->vocab_size);
        astral::core::runtime_free_array(session->batch_logits, model->vocab_size * kMockMaxBatchOutputs);
        astral::core::runtime_delete(session);
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    session->model = model;
    session->step = 0;
    session->slot_id = 0;
    session->n_slots = max_slots == 0 ? 1u : (max_slots > kMockMaxSlots ? kMockMaxSlots : max_slots);
    session->adapter_count = 0;
    session->grammar_enabled = 0;
    session->grammar_allow_token = 0xFFFFFFFFu;
    session->has_logits = false;
    session->last_batch_outputs = 0;

    for (uint32_t i = 0; i < kMockMaxSlots; ++i) {
        session->slot_step[i] = 0;
        session->slot_output_step[i] = 0;
        session->slot_has_logits[i] = false;
        session->slot_grammar_enabled[i] = 0;
        session->slot_grammar_allow_token[i] = 0xFFFFFFFFu;
    }

    *out_err = ASTRAL_OK;
    return session;
}

void mock_session_destroy(void* session_ctx) {
    if (session_ctx == nullptr) {
        return;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    astral::core::runtime_free_array(session->logits, session->model->vocab_size);
    astral::core::runtime_free_array(session->batch_logits, session->model->vocab_size * kMockMaxBatchOutputs);
    astral::core::runtime_delete(session);
}

AstralErr mock_session_reset(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    session->step = 0;
    session->adapter_count = 0;
    session->has_logits = false;
    session->last_batch_outputs = 0;
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        session->slot_step[i] = 0;
        session->slot_has_logits[i] = false;
    }
    return ASTRAL_OK;
}

AstralErr mock_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    (void)tokens;
    (void)count;

    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    session->step = 0;
    session->has_logits = true;
    if (session->slot_id < session->n_slots) {
        session->slot_step[session->slot_id] = session->step;
        session->slot_has_logits[session->slot_id] = true;
    }
    return ASTRAL_OK;
}

AstralErr mock_session_feed_image(void* session_ctx, const AstralImageDesc* image, uint8_t finalize) {
    (void)finalize;
    if (session_ctx == nullptr || image == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (image->size != sizeof(AstralImageDesc) || image->pixels.data == nullptr || image->pixels.len == 0) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (session->slot_id >= session->n_slots) {
        return ASTRAL_E_STATE;
    }

    session->slot_step[session->slot_id] += 1u;
    session->step = session->slot_step[session->slot_id];
    session->has_logits = true;
    session->slot_has_logits[session->slot_id] = true;
    return ASTRAL_OK;
}

AstralErr mock_session_feed_audio(void* session_ctx, const AstralAudioDesc* audio, uint8_t finalize) {
    (void)finalize;
    if (session_ctx == nullptr || audio == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (audio->size != sizeof(AstralAudioDesc) || audio->samples.data == nullptr || audio->samples.len == 0) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (session->slot_id >= session->n_slots) {
        return ASTRAL_E_STATE;
    }

    session->slot_step[session->slot_id] += 1u;
    session->step = session->slot_step[session->slot_id];
    session->has_logits = true;
    session->slot_has_logits[session->slot_id] = true;
    return ASTRAL_OK;
}

AstralErr mock_session_logits(void* session_ctx, BackendLogitsView* out_view) {
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (session->slot_id >= session->n_slots) {
        return ASTRAL_E_STATE;
    }
    if (!session->slot_has_logits[session->slot_id] || !session->has_logits || session->model == nullptr || session->logits == nullptr) {
        return ASTRAL_E_STATE;
    }

    // Default all logits low (small vocab; ok for tests).
    for (uint32_t i = 0; i < session->model->vocab_size; ++i) {
        session->logits[i] = -1000.0f;
    }

    if (session->model != nullptr && session->model->sampler_mode) {
        // Deterministic 2-token distribution for sampler tests.
        const uint32_t a = static_cast<uint32_t>('a');
        const uint32_t b = static_cast<uint32_t>('b');
        if (a < session->model->vocab_size) session->logits[a] = 10.0f;
        if (b < session->model->vocab_size) session->logits[b] = 10.0f;
    } else if (session->model->long_piece_mode) {
      const uint32_t out_step = session->slot_output_step[session->slot_id];
      const int32_t next = out_step == 0 ? kMockTokenLongPiece : kMockTokenEos;
      session->logits[static_cast<uint32_t>(next)] = 10.0f;
    } else {
      const char* message = session->model->tool_call_mode ? kMockToolCallMessage : kMockMessage;
      const size_t msg_len = std::strlen(message);
      int32_t next = kMockTokenEos;
      if (msg_len > 0 && session->model != nullptr && session->model->tool_call_mode) {
        const uint32_t out_step = session->slot_output_step[session->slot_id];
        if (out_step < msg_len) {
          next = static_cast<int32_t>(static_cast<uint8_t>(message[out_step]));
        }
      } else if (msg_len > 0 && session->model != nullptr && session->model->infinite) {
        next = static_cast<int32_t>(static_cast<uint8_t>(message[session->step % msg_len]));
      } else if (session->step < msg_len) {
        next = static_cast<int32_t>(static_cast<uint8_t>(message[session->step]));
      }

      if (next >= 0 && static_cast<uint32_t>(next) < session->model->vocab_size) {
        session->logits[static_cast<uint32_t>(next)] = 10.0f;
      }
    }

    out_view->logits = session->logits;
    out_view->vocab_size = session->model->vocab_size;
    return ASTRAL_OK;
}

AstralErr mock_session_accept(void* session_ctx, int32_t token) {
    (void)token;

    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (session->slot_id >= session->n_slots) {
        return ASTRAL_E_STATE;
    }
    if (!session->slot_has_logits[session->slot_id] || !session->has_logits) {
        return ASTRAL_E_STATE;
    }

    session->step++;
    if (session->slot_id < session->n_slots) {
      if (session->model != nullptr &&
          (session->model->tool_call_mode || session->model->long_piece_mode)) {
        session->slot_output_step[session->slot_id] += 1u;
      }
        session->slot_step[session->slot_id] = session->step;
        session->slot_has_logits[session->slot_id] = true;
    }
    return ASTRAL_OK;
}

// -----------------------------------------------------------------------------
// Optional generation controls
// -----------------------------------------------------------------------------

struct MockAdapter {
    uint32_t id;
};

static inline bool parse_u32_dec(AstralSpanU8 s, uint32_t* out) {
    if (out == nullptr || s.data == nullptr || s.len == 0) {
        return false;
    }
    uint32_t v = 0;
    for (uint32_t i = 0; i < s.len; ++i) {
        const uint8_t c = s.data[i];
        if (c < '0' || c > '9') {
            return false;
        }
        const uint32_t d = static_cast<uint32_t>(c - '0');
        const uint32_t nv = v * 10u + d;
        if (nv < v) {
            return false;
        }
        v = nv;
    }
    *out = v;
    return true;
}

AstralErr mock_session_grammar_set_gbnf(void* session_ctx, AstralSpanU8 gbnf, AstralSpanU8 root) {
    (void)root;
    if (session_ctx == nullptr || gbnf.data == nullptr || gbnf.len == 0) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);

    // Mock grammar format:
    // - "allow=<single ascii char>"
    // - "allow_byte=<0..255>"
    if (gbnf.len >= 6 && std::memcmp(gbnf.data, "allow=", 6) == 0) {
        if (gbnf.len != 7) {
            return ASTRAL_E_INVALID;
        }
        const uint8_t ch = gbnf.data[6];
        s->grammar_allow_token = static_cast<uint32_t>(ch);
        s->grammar_enabled = 1;
        return ASTRAL_OK;
    }

    if (gbnf.len >= 11 && std::memcmp(gbnf.data, "allow_byte=", 11) == 0) {
        AstralSpanU8 num{};
        num.data = gbnf.data + 11;
        num.len = gbnf.len - 11;
        uint32_t v = 0;
        if (!parse_u32_dec(num, &v) || v > 255u) {
            return ASTRAL_E_INVALID;
        }
        s->grammar_allow_token = v;
        s->grammar_enabled = 1;
        return ASTRAL_OK;
    }

    return ASTRAL_E_INVALID;
}

[[maybe_unused]] AstralErr mock_session_grammar_set_json_schema(void* session_ctx, AstralSpanU8 json_schema) {
    (void)session_ctx;
    (void)json_schema;
    return ASTRAL_E_UNSUPPORTED;
}

AstralErr mock_session_grammar_clear(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    s->grammar_enabled = 0;
    s->grammar_allow_token = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < s->n_slots; ++i) {
        s->slot_grammar_enabled[i] = 0;
        s->slot_grammar_allow_token[i] = 0xFFFFFFFFu;
    }
    return ASTRAL_OK;
}

AstralErr mock_session_apply_grammar(void* session_ctx, uint32_t* tokens, float* logits, uint32_t count) {
    if (session_ctx == nullptr || tokens == nullptr || logits == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    if (s->grammar_enabled == 0 || s->grammar_allow_token > 255u) {
        return ASTRAL_OK;
    }
    const uint32_t allowed = s->grammar_allow_token;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < count; ++i) {
        if (tokens[i] != allowed) {
            logits[i] = neg_inf;
        }
    }
    return ASTRAL_OK;
}

AstralErr mock_session_grammar_set_gbnf_for_slot(void* session_ctx, uint32_t slot_id, AstralSpanU8 gbnf, AstralSpanU8 root) {
    (void)root;
    if (session_ctx == nullptr || gbnf.data == nullptr || gbnf.len == 0) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    if (slot_id >= s->n_slots) {
        return ASTRAL_E_INVALID;
    }

    // Same mini-format as mock_session_grammar_set_gbnf:
    // - "allow=<single ascii char>"
    // - "allow_byte=<0..255>"
    if (gbnf.len >= 6 && std::memcmp(gbnf.data, "allow=", 6) == 0) {
        if (gbnf.len != 7) {
            return ASTRAL_E_INVALID;
        }
        const uint8_t ch = gbnf.data[6];
        s->slot_grammar_allow_token[slot_id] = static_cast<uint32_t>(ch);
        s->slot_grammar_enabled[slot_id] = 1;
        return ASTRAL_OK;
    }

    if (gbnf.len >= 11 && std::memcmp(gbnf.data, "allow_byte=", 11) == 0) {
        AstralSpanU8 num{};
        num.data = gbnf.data + 11;
        num.len = gbnf.len - 11;
        uint32_t v = 0;
        if (!parse_u32_dec(num, &v) || v > 255u) {
            return ASTRAL_E_INVALID;
        }
        s->slot_grammar_allow_token[slot_id] = v;
        s->slot_grammar_enabled[slot_id] = 1;
        return ASTRAL_OK;
    }

    return ASTRAL_E_INVALID;
}

AstralErr mock_session_grammar_set_json_schema_for_slot(void* session_ctx, uint32_t slot_id, AstralSpanU8 json_schema) {
    (void)session_ctx;
    (void)slot_id;
    (void)json_schema;
    return ASTRAL_E_UNSUPPORTED;
}

AstralErr mock_session_grammar_clear_for_slot(void* session_ctx, uint32_t slot_id) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    if (slot_id >= s->n_slots) {
        return ASTRAL_E_INVALID;
    }
    s->slot_grammar_enabled[slot_id] = 0;
    s->slot_grammar_allow_token[slot_id] = 0xFFFFFFFFu;
    return ASTRAL_OK;
}

AstralErr mock_session_apply_grammar_for_slot(void* session_ctx, uint32_t slot_id, uint32_t* tokens, float* logits, uint32_t count) {
    if (session_ctx == nullptr || tokens == nullptr || logits == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    if (slot_id >= s->n_slots) {
        return ASTRAL_E_INVALID;
    }
    if (s->slot_grammar_enabled[slot_id] == 0 || s->slot_grammar_allow_token[slot_id] > 255u) {
        return ASTRAL_OK;
    }
    const uint32_t allowed = s->slot_grammar_allow_token[slot_id];
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < count; ++i) {
        if (tokens[i] != allowed) {
            logits[i] = neg_inf;
        }
    }
    return ASTRAL_OK;
}

AstralErr mock_session_state_size(void* session_ctx, uint64_t* out_bytes) {
    if (session_ctx == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_bytes = 16;
    return ASTRAL_OK;
}

AstralErr mock_session_state_save(void* session_ctx, uint8_t* out_bytes, uint64_t capacity, uint64_t* out_written) {
    if (session_ctx == nullptr || out_bytes == nullptr || out_written == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (capacity < 16) {
        return ASTRAL_E_NOMEM;
    }
    const MockSession* s = static_cast<const MockSession*>(session_ctx);
    std::memcpy(out_bytes + 0, &s->step, 4);
    std::memcpy(out_bytes + 4, &s->slot_id, 4);
    std::memcpy(out_bytes + 8, &s->grammar_allow_token, 4);
    std::memcpy(out_bytes + 12, &s->adapter_count, 4);
    *out_written = 16;
    return ASTRAL_OK;
}

AstralErr mock_session_state_load(void* session_ctx, const uint8_t* bytes, uint64_t size) {
    if (session_ctx == nullptr || bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (size < 16) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    std::memcpy(&s->step, bytes + 0, 4);
    std::memcpy(&s->slot_id, bytes + 4, 4);
    std::memcpy(&s->grammar_allow_token, bytes + 8, 4);
    std::memcpy(&s->adapter_count, bytes + 12, 4);
    if (s->slot_id >= s->n_slots) {
        s->slot_id = 0;
    }
    s->slot_step[s->slot_id] = s->step;
    s->slot_has_logits[s->slot_id] = true;
    s->grammar_enabled = (s->grammar_allow_token <= 255u) ? 1u : 0u;
    s->has_logits = true;
    return ASTRAL_OK;
}

void* mock_model_adapter_load(void* model_ctx, AstralSpanU8 path, AstralErr* out_err) {
    (void)model_ctx;
    if (out_err == nullptr) {
        return nullptr;
    }
    if (path.data == nullptr || path.len == 0) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < path.len; ++i) {
        h ^= static_cast<uint32_t>(path.data[i]);
        h *= 16777619u;
    }

    MockAdapter* a = astral::core::runtime_new<MockAdapter>();
    if (a == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    a->id = h;
    *out_err = ASTRAL_OK;
    return a;
}

void mock_model_adapter_unload(void* model_ctx, void* adapter_ctx) {
    (void)model_ctx;
    astral::core::runtime_delete(static_cast<MockAdapter*>(adapter_ctx));
}

AstralErr mock_session_adapter_clear(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    s->adapter_count = 0;
    return ASTRAL_OK;
}

AstralErr mock_session_adapter_add(void* session_ctx, void* adapter_ctx, float scale) {
    (void)scale;
    if (session_ctx == nullptr || adapter_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    s->adapter_count++;
    return ASTRAL_OK;
}

AstralErr mock_session_set_slot(void* session_ctx, uint32_t slot_id) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    if (slot_id >= s->n_slots) {
        return ASTRAL_E_INVALID;
    }
    s->slot_id = slot_id;
    s->step = s->slot_step[slot_id];
    s->has_logits = s->slot_has_logits[slot_id];
    return ASTRAL_OK;
}

AstralErr mock_session_slot_pos(void* session_ctx, uint32_t slot_id, uint32_t* out_pos) {
    if (session_ctx == nullptr || out_pos == nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockSession* s = static_cast<MockSession*>(session_ctx);
    if (slot_id >= s->n_slots) {
        return ASTRAL_E_INVALID;
    }
    *out_pos = s->slot_step[slot_id];
    return ASTRAL_OK;
}

AstralErr mock_session_batch_eval(void* session_ctx,
                                  const AstralBackendBatchToken* tokens,
                                  uint32_t token_count,
                                  uint32_t* out_output_count) {
    if (session_ctx == nullptr || out_output_count == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (token_count > 0 && tokens == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (session->model == nullptr || session->batch_logits == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    uint32_t outputs = 0;
    for (uint32_t i = 0; i < token_count; ++i) {
        const AstralBackendBatchToken& t = tokens[i];
        if (t.slot_id >= session->n_slots) {
            return ASTRAL_E_INVALID;
        }

        const uint32_t next_step = t.pos + 1u;
        if (next_step > session->slot_step[t.slot_id]) {
            session->slot_step[t.slot_id] = next_step;
        }
        session->slot_has_logits[t.slot_id] = false;

        if (t.want_logits != 0) {
            if (outputs >= kMockMaxBatchOutputs) {
                return ASTRAL_E_NOMEM;
            }

            float* out = session->batch_logits +
                static_cast<size_t>(outputs) * static_cast<size_t>(session->model->vocab_size);
            std::memset(out, 0, static_cast<size_t>(session->model->vocab_size) * sizeof(float));

            int32_t next = kMockTokenEos;
            if (session->model->long_piece_mode) {
              const uint32_t out_step = session->slot_output_step[t.slot_id];
              next = out_step == 0 ? kMockTokenLongPiece : kMockTokenEos;
              session->slot_output_step[t.slot_id] = out_step + 1u;
            } else if (session->model->tool_call_mode) {
              const char* message = kMockToolCallMessage;
              const size_t msg_len = std::strlen(message);
              const uint32_t out_step = session->slot_output_step[t.slot_id];
              if (out_step < msg_len) {
                next = static_cast<int32_t>(static_cast<uint8_t>(message[out_step]));
                session->slot_output_step[t.slot_id] = out_step + 1u;
              }
            } else if (next_step < 16) {
              next = static_cast<int32_t>((t.slot_id * 17u + next_step * 3u) & 0xFFu);
            }

            if (next >= 0 && static_cast<uint32_t>(next) < session->model->vocab_size) {
                out[static_cast<uint32_t>(next)] = 1.0f;
            }

            outputs++;
        }
    }

    session->last_batch_outputs = outputs;
    *out_output_count = outputs;
    return ASTRAL_OK;
}

AstralErr mock_session_batch_logits(void* session_ctx, uint32_t output_index, BackendLogitsView* out_view) {
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (session->model == nullptr || session->batch_logits == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    if (output_index >= session->last_batch_outputs) {
        return ASTRAL_E_INVALID;
    }

    out_view->logits = session->batch_logits +
        static_cast<size_t>(output_index) * static_cast<size_t>(session->model->vocab_size);
    out_view->vocab_size = session->model->vocab_size;
    return ASTRAL_OK;
}

AstralErr mock_session_slot_reset(void* session_ctx, uint32_t slot_id) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockSession* session = static_cast<MockSession*>(session_ctx);
    if (slot_id >= session->n_slots) {
        return ASTRAL_E_INVALID;
    }

    session->slot_step[slot_id] = 0;
    session->slot_output_step[slot_id] = 0;
    session->slot_has_logits[slot_id] = false;
    if (slot_id == session->slot_id) {
        session->step = 0;
        session->has_logits = false;
    }
    return ASTRAL_OK;
}

void* mock_embedder_create(void* model_ctx, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    MockEmbedder* e = astral::core::runtime_new<MockEmbedder>();
    if (e == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    e->model = static_cast<MockModel*>(model_ctx);
    *out_err = ASTRAL_OK;
    return e;
}

void mock_embedder_destroy(void* embedder_ctx) {
    astral::core::runtime_delete(static_cast<MockEmbedder*>(embedder_ctx));
}

AstralErr mock_embedder_reset(void* embedder_ctx) {
    if (embedder_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    return ASTRAL_OK;
}

AstralErr mock_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec, uint32_t vec_dim) {
    if (embedder_ctx == nullptr || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }

    MockEmbedder* e = static_cast<MockEmbedder*>(embedder_ctx);
    if (e->model == nullptr || e->model->emb_dim == 0) {
        return ASTRAL_E_BACKEND;
    }

    if (tokens == nullptr && count > 0) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t dim = e->model->emb_dim;
    if (vec_dim < dim) {
        return ASTRAL_E_NOMEM;
    }

    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t t = tokens[i];
        sum += static_cast<uint32_t>(t >= 0 ? t : -t);
    }

    for (uint32_t i = 0; i < dim; ++i) {
        out_vec[i] = static_cast<float>(sum + i);
    }

    return ASTRAL_OK;
}

static AstralErr mock_embedder_fill(MockEmbedder* e, uint32_t sum, float* out_vec, uint32_t vec_dim) {
    if (e == nullptr || e->model == nullptr || e->model->emb_dim == 0 || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }
    const uint32_t dim = e->model->emb_dim;
    if (vec_dim < dim) {
        return ASTRAL_E_NOMEM;
    }
    for (uint32_t i = 0; i < dim; ++i) {
        out_vec[i] = static_cast<float>(sum + i);
    }
    return ASTRAL_OK;
}

AstralErr mock_embedder_embed_image(void* embedder_ctx, const AstralImageDesc* image, float* out_vec, uint32_t vec_dim) {
    if (embedder_ctx == nullptr || image == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (image->size != sizeof(AstralImageDesc) || image->pixels.data == nullptr || image->pixels.len == 0) {
        return ASTRAL_E_INVALID;
    }
    MockEmbedder* e = static_cast<MockEmbedder*>(embedder_ctx);
    const uint32_t sum = sum_bytes(image->pixels.data, image->pixels.len) + 11u;
    return mock_embedder_fill(e, sum, out_vec, vec_dim);
}

AstralErr mock_embedder_embed_audio(void* embedder_ctx, const AstralAudioDesc* audio, float* out_vec, uint32_t vec_dim) {
    if (embedder_ctx == nullptr || audio == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (audio->size != sizeof(AstralAudioDesc) || audio->samples.data == nullptr || audio->samples.len == 0) {
        return ASTRAL_E_INVALID;
    }
    MockEmbedder* e = static_cast<MockEmbedder*>(embedder_ctx);
    const uint32_t sum = sum_bytes(audio->samples.data, audio->samples.len) + 29u;
    return mock_embedder_fill(e, sum, out_vec, vec_dim);
}

AstralErr mock_embedder_embed_multimodal(void* embedder_ctx,
                                         AstralSpanU8 text,
                                         const AstralImageDesc* image,
                                         const AstralAudioDesc* audio,
                                         float* out_vec,
                                         uint32_t vec_dim) {
    if (embedder_ctx == nullptr || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (image != nullptr && audio != nullptr) {
        return ASTRAL_E_INVALID;
    }
    MockEmbedder* e = static_cast<MockEmbedder*>(embedder_ctx);

    uint32_t sum = sum_bytes(text.data, text.len);
    if (image != nullptr) {
        if (image->size != sizeof(AstralImageDesc) || image->pixels.data == nullptr || image->pixels.len == 0) {
            return ASTRAL_E_INVALID;
        }
        sum += sum_bytes(image->pixels.data, image->pixels.len);
    }
    if (audio != nullptr) {
        if (audio->size != sizeof(AstralAudioDesc) || audio->samples.data == nullptr || audio->samples.len == 0) {
            return ASTRAL_E_INVALID;
        }
        sum += sum_bytes(audio->samples.data, audio->samples.len);
    }
    sum += 3u;
    return mock_embedder_fill(e, sum, out_vec, vec_dim);
}

static const BackendOps kMockBackendOps = [] {
  BackendOps ops{};
  ops.model_load = mock_model_load;
  ops.model_unload = mock_model_unload;
  ops.tokenize = mock_tokenize;
  ops.detokenize = mock_detokenize;
  ops.model_info = mock_model_info;
  ops.model_special_tokens = mock_model_special_tokens;
  ops.model_embedding_dim = mock_model_embedding_dim;
  ops.model_media_init = mock_model_media_init;
  ops.model_media_info = mock_model_media_info;
  ops.session_create = mock_session_create;
  ops.session_create_ex = mock_session_create_ex;
  ops.session_destroy = mock_session_destroy;
  ops.session_reset = mock_session_reset;
  ops.session_feed = mock_session_feed;
  ops.session_feed_image = mock_session_feed_image;
  ops.session_feed_audio = mock_session_feed_audio;
  ops.session_logits = mock_session_logits;
  ops.session_accept = mock_session_accept;
  ops.session_batch_eval = mock_session_batch_eval;
  ops.session_batch_logits = mock_session_batch_logits;
  ops.session_slot_reset = mock_session_slot_reset;
  ops.embedder_create = mock_embedder_create;
  ops.embedder_destroy = mock_embedder_destroy;
  ops.embedder_reset = mock_embedder_reset;
  ops.embedder_embed = mock_embedder_embed;
  ops.embedder_embed_image = mock_embedder_embed_image;
  ops.embedder_embed_audio = mock_embedder_embed_audio;
  ops.embedder_embed_multimodal = mock_embedder_embed_multimodal;
  ops.session_grammar_set_gbnf = mock_session_grammar_set_gbnf;
  ops.session_grammar_clear = mock_session_grammar_clear;
  ops.session_apply_grammar = mock_session_apply_grammar;
  ops.session_grammar_set_gbnf_for_slot = mock_session_grammar_set_gbnf_for_slot;
  ops.session_grammar_set_json_schema_for_slot = mock_session_grammar_set_json_schema_for_slot;
  ops.session_grammar_clear_for_slot = mock_session_grammar_clear_for_slot;
  ops.session_apply_grammar_for_slot = mock_session_apply_grammar_for_slot;
  ops.session_state_size = mock_session_state_size;
  ops.session_state_save = mock_session_state_save;
  ops.session_state_load = mock_session_state_load;
  ops.model_adapter_load = mock_model_adapter_load;
  ops.model_adapter_unload = mock_model_adapter_unload;
  ops.session_adapter_clear = mock_session_adapter_clear;
  ops.session_adapter_add = mock_session_adapter_add;
  ops.session_set_slot = mock_session_set_slot;
  ops.session_slot_pos = mock_session_slot_pos;
  return ops;
}();

static const BackendProvider kMockBackendProvider = {
    /*name=*/"mock",
    /*ops=*/&kMockBackendOps,
    /*supports_gpu=*/false,
    /*min_gpu_layers=*/0,
};

} // namespace

const BackendProvider* builtin_mock_backend_provider() {
    return &kMockBackendProvider;
}

} // namespace astral::backend
