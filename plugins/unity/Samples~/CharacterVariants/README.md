# Character Variants

Attach `CharacterVariantsExample` and `AstralRuntimeInitializer` to a GameObject.
The component exposes two focused paths:

- `PreparePromptCache` tokenizes and stores the system prompt, verifies the
  token entry, and saves a native cache snapshot. `RunCachedAgent` binds that
  cache to an agent and reports reused prompt tokens.
- `RunAdaptedSession` applies JSON-schema output, a stop sequence, and an
  optional model-scoped LoRA adapter with a configurable scale.

The mock backend exercises cache, grammar, stop, and lifecycle behavior. Set a
real adapter path only when the selected provider reports LoRA support. Adapter
ownership remains with the model and outlives the session that references it.
