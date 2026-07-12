# Stateful NPC

Attach `StatefulNpcExample` and `AstralRuntimeInitializer` to a GameObject. Call
`Ask` from a UI button or gameplay interaction. The default mock backend covers
agent lifecycle, history ownership, cancellation, and tool-call parsing without
an external model.

The component keeps system prompt, summary, and retrieved memory context in the
native agent. It saves history under `Application.persistentDataPath`, restores
that history on the next run, and releases the assigned executor slot after a
chat finishes. The `move_to` tool demonstrates how generated structured actions
cross into gameplay code.

Use `ClearHistory` when starting a new save game. For real dialogue, set the
model path and backend and replace the log calls with your subtitle and action
dispatch systems.
