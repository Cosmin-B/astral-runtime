# Structured Output And Tools

Astral exposes tool definitions as native runtime objects. Toolsets are copied
once at setup, can be attached to sessions or conversations, and can parse a
completed tool-call payload without requiring engine wrappers to scrape streams.

## Native API

- `astral_toolset_create(desc, out_toolset)` copies `AstralToolDesc` entries
  into runtime memory.
- `astral_toolset_destroy(toolset)` releases the caller's reference.
- `astral_toolset_count(toolset, out_count)` reports the number of tools.
- `astral_toolset_get(toolset, index, out_info)` returns lifetime-bound spans
  for one tool.
- `astral_toolset_parse_call(toolset, generated_text, out_result)` detects a
  tool name and returns the raw JSON object span for `arguments`.
- `astral_session_set_toolset(session, toolset, choice_mode)` binds a toolset
  to a session.
- `astral_session_clear_toolset(session)` clears the session binding.
- `astral_conv_set_toolset(conv, toolset, choice_mode)` binds a toolset to a
  conversation.
- `astral_conv_clear_toolset(conv)` clears the conversation binding.
- `astral_agent_parse_tool_call(agent, generated_text, out_result)` parses a
  completed payload against the toolset bound at agent creation.
- `astral_agent_chat_tool_call_result(agent, out_result)` parses the latest
  drained agent chat stream captured by the native agent.

`AstralToolChoiceMode` is one of:

- `ASTRAL_TOOL_CHOICE_AUTO`
- `ASTRAL_TOOL_CHOICE_REQUIRED`
- `ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL`

## Lifetime

Tool names, descriptions, and JSON schemas are copied by
`astral_toolset_create()`. Spans returned by `astral_toolset_get()` remain valid
until the toolset is destroyed. Session, conversation, and agent bindings retain
the toolset, so callers may release their own handle after binding.

`astral_toolset_parse_call()` and `astral_agent_parse_tool_call()` return
`arguments_json` as a span into the caller-provided generated text. Keep that
text alive until the result has been consumed.

`astral_agent_chat_tool_call_result()` returns spans owned by the agent. They
remain valid until the next chat enqueue or until the agent is destroyed. The
agent captures bytes as `astral_agent_chat_stream_read()` drains them, so call
it after the stream has reached end-of-stream.

## Error Behavior

Malformed descriptors return `ASTRAL_E_INVALID`. Missing tool calls or unknown
tool names return `ASTRAL_E_NOT_FOUND`. If a known tool is found but
`arguments` is missing or is not a JSON object, the function returns
`ASTRAL_OK` and sets `AstralToolCallResult.parse_status` to
`ASTRAL_E_INVALID`.

## Performance

Tool definition copying and result parsing are setup/finalization work. The
decode loop does not inspect tool descriptors or parse JSON. The result parser
is a small single-pass scanner for the `name`/`tool` field and the raw
`arguments` object span; it does not build a DOM or allocate.

Agent chat capture is enabled only when the agent has a toolset. It reuses a
bounded native buffer sized from the request token limit and appends drained
stream bytes outside the decode loop.

## Unreal And Unity

Unreal exposes Blueprint-safe helpers on `UAstralBlueprintLibrary`:

- `CreateToolset()`
- `DestroyToolset()`
- `ParseToolCall()`
- `ParseAgentToolCall()`
- `GetAgentChatToolCallResult()`

`UAstralSession::SetToolset()` and `UAstralSession::ClearToolset()` bind and
clear native toolset handles. The wrapper uses Unreal containers for transient
UTF-8 conversion and passes all core ownership to the native runtime.

Unity exposes `AstralToolset` as an owned handle over the same native C ABI.
`AstralToolset.Create()` copies managed tool definitions into native runtime
memory, `ParseCall()` parses a completed tool-call payload, and
`AstralAgent.ParseToolCall()` parses caller-provided text and
`AstralAgent.GetChatToolCallResult()` reads the native result captured from the
latest drained chat stream. `AstralSession.SetToolset()` / `ClearToolset()`
bind or clear the handle between requests. `AstralSession.SetGrammarGbnf()`,
`SetGrammarJsonSchema()`, and `ClearGrammar()` expose direct grammar binding for
callers that do not need a toolset. `AstralToolCall.Parsed`, `Missing`, and
`Malformed` map native parse status to Unity-friendly predicates without
parsing generated text twice.

## Example

```c
AstralToolDesc tool = {0};
const char name[] = "search";
const char description[] = "Search indexed text";
const char schema[] = "{\"type\":\"object\"}";

tool.size = sizeof(AstralToolDesc);
tool.tool_id = 1;
tool.name.data = (const uint8_t*)name;
tool.name.len = (uint32_t)(sizeof(name) - 1);
tool.description.data = (const uint8_t*)description;
tool.description.len = (uint32_t)(sizeof(description) - 1);
tool.json_schema.data = (const uint8_t*)schema;
tool.json_schema.len = (uint32_t)(sizeof(schema) - 1);

AstralToolsetDesc desc = {0};
desc.size = sizeof(AstralToolsetDesc);
desc.tool_count = 1;
desc.choice_mode = ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL;
desc.tools = &tool;

AstralHandle toolset = 0;
AstralErr err = astral_toolset_create(&desc, &toolset);
```

## Validation

```bash
cmake --build --preset dev -j8 --target test_inference test_abi_invalid_args
ctest --preset dev -R '^(test_inference|test_abi_invalid_args|gate_abi_layout_report|gate_doc_links)$' --output-on-failure
ASTRAL_BENCH_PROMPT_CACHE_ONLY=1 ASTRAL_BENCH_FEATURE_ITERS=200000 ./build/dev/benchmarks/astral_benchmarks --only features
```

Expected evidence includes `inference_toolset_parse_and_bind_mock` passing,
invalid argument coverage for the C ABI boundary, and ABI layout coverage for
`AstralToolCallResult`. Benchmark output should include
`features.toolset parse`.
