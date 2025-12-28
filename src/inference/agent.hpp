#pragma once

#include "../../include/astral_rt.h"

namespace astral::inference {

struct Agent;

AstralHandle agent_handle(Agent* agent);
AstralErr agent_create(const AstralAgentDesc* desc, Agent** out_agent);
void agent_destroy(Agent* agent);
AstralErr agent_set_system_prompt(Agent* agent, AstralSpanU8 system_prompt);
AstralErr agent_get_system_prompt_size(Agent* agent, uint32_t* out_bytes);
AstralErr agent_get_system_prompt(Agent* agent, AstralMutSpanU8 out_text, uint32_t* out_len);
AstralErr agent_set_summary(Agent* agent, AstralSpanU8 summary);
AstralErr agent_get_summary_size(Agent* agent, uint32_t* out_bytes);
AstralErr agent_get_summary(Agent* agent, AstralMutSpanU8 out_text, uint32_t* out_len);
AstralErr agent_set_memory_context(Agent* agent, AstralSpanU8 memory_context);
AstralErr agent_get_memory_context_size(Agent* agent, uint32_t* out_bytes);
AstralErr agent_get_memory_context(Agent* agent, AstralMutSpanU8 out_text, uint32_t* out_len);
AstralErr agent_parse_tool_call(Agent* agent, AstralSpanU8 generated_text, AstralToolCallResult* out_result);
AstralErr agent_message_add(Agent* agent, const AstralAgentMessage* message);
AstralErr agent_history_clear(Agent* agent);
AstralErr agent_history_count(Agent* agent, uint32_t* out_count);
AstralErr agent_history_save_size(Agent* agent, uint32_t* out_bytes);
AstralErr agent_history_save(Agent* agent, AstralMutSpanU8 out_bytes, uint32_t* out_len);
AstralErr agent_history_load(Agent* agent, AstralSpanU8 bytes);
AstralErr agent_chat_enqueue(Agent* agent, const AstralAgentChatDesc* desc);
AstralErr agent_chat_cancel(Agent* agent);
int32_t agent_chat_stream_read(Agent* agent, AstralMutSpanU8 out_buf, uint32_t timeout_ms);
AstralErr agent_chat_result(Agent* agent, AstralAgentChatResult* out_result);

} // namespace astral::inference
