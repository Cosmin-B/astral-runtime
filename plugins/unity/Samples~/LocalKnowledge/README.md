# Local Knowledge

Attach `LocalKnowledgeExample` and `AstralRuntimeInitializer` to a GameObject.
`BuildIndex` splits the serialized document into overlapping word ranges,
embeds each chunk, and inserts the records and vectors in one memory-index
batch. `Search` embeds a question and prints the highest-scoring source chunks.

`Save` and `Load` persist the native index under
`Application.persistentDataPath`. Keep the same document and embedding model
when restoring a snapshot because search results refer to the original chunk
indices and vector dimension.

The mock backend exercises the complete ownership and retrieval flow. Replace
it with an embeddings-capable GGUF model for meaningful semantic ranking.
