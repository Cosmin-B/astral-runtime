# Streaming chat

Attach `StreamingChatExample` and `AstralRuntimeInitializer` to a GameObject.
Set `modelPath` to a GGUF file visible to the player and set `backendName` to
`cpu` or another packaged provider.

`Run` creates one session and polls its UTF-8 stream once per frame. `Cancel`
uses the generic request reference, and a second run disposes the previous
session and caller-owned `NativeArray` before starting again.

The example converts bytes to a managed string only at the `Debug.Log` boundary.
Replace that call with your UI or gameplay text sink.
