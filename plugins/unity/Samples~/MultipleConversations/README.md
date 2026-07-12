# Multiple Conversations

Attach `MultipleConversationsExample` and `AstralRuntimeInitializer` to a
GameObject. The default mock backend starts two independent NPC conversations
through one model executor.

Each conversation owns its native slot and stream buffer. The coroutine polls
every active slot once per frame, so one quiet NPC does not block the others.
`Cancel` uses generic request references, and teardown releases conversations
before the shared model.

Set `modelPath` and `backendName` for a real generation model. Keep
`maxBatchTokens` within the model's configured batch capacity.
