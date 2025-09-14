#pragma once

#include "../../include/astral_rt.h"
#include "conversation.hpp"

namespace astral::inference {

AstralErr conv_create(const AstralConvDesc* desc, Conversation** out_conv);
void conv_destroy(Conversation* conv);

AstralErr conv_feed(Conversation* conv, AstralSpanU8 prompt_chunk, uint8_t finalize);
AstralErr conv_decode(Conversation* conv);
AstralErr conv_cancel(Conversation* conv);
AstralErr conv_state(Conversation* conv, AstralSessionState* out_state);
AstralErr conv_wait(Conversation* conv, uint32_t timeout_ms);
AstralErr conv_reset(Conversation* conv, const AstralConvDesc* desc);

AstralErr conv_set_sampler(Conversation* conv, const AstralSamplerDesc* desc);
AstralErr conv_penalty_prompt_set_tokens(Conversation* conv, const int32_t* tokens, uint32_t count);
AstralErr conv_stop_clear(Conversation* conv);
AstralErr conv_stop_add_utf8(Conversation* conv, AstralSpanU8 utf8);
AstralErr conv_stop_set_utf8(Conversation* conv, const AstralSpanU8* seqs, uint32_t count);
AstralErr conv_set_logprobs(Conversation* conv, uint32_t n_probs);

AstralErr conv_grammar_set_gbnf(Conversation* conv, AstralSpanU8 gbnf, AstralSpanU8 root);
AstralErr conv_grammar_set_json_schema(Conversation* conv, AstralSpanU8 json_schema);
AstralErr conv_grammar_clear(Conversation* conv);

AstralErr conv_stats(Conversation* conv, AstralConvStats* out_stats);

int32_t conv_stream_read(Conversation* conv, AstralMutSpanU8 out_buf, uint32_t timeout_ms);
int32_t conv_stream_read_meta(Conversation* conv, AstralTokenMeta* out_events, uint32_t capacity, uint32_t timeout_ms);

} // namespace astral::inference
