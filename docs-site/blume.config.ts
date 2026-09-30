import { defineConfig } from "blume";

const item = (label: string, href: string) => ({ label, href });
const page = (label: string, root: string) => ({ label, root });

export default defineConfig({
  title: "Astral Documentation",
  description: "Native inference runtime reference for applications and game engines",
  deployment: {
    site: "https://astralruntime.dev",
    base: "/docs",
  },
  redirects: [
    { from: "/docs/docs/integration/UNITY_INTEGRATION", to: "/plugins/unity", status: 308 },
    { from: "/docs/docs/integration/UNREAL_INTEGRATION", to: "/plugins/unreal/AstralRT", status: 308 },
    { from: "/docs/integration/UNITY_INTEGRATION", to: "/plugins/unity", status: 308 },
    { from: "/docs/integration/UNREAL_INTEGRATION", to: "/plugins/unreal/AstralRT", status: 308 },
  ],
  content: {
    root: "docs",
  },
  feedback: false,
  navigation: {
    repo: "https://github.com/Cosmin-B/astral-runtime",
    sidebar: {
      display: "group",
      items: [
        { label: "Overview", items: [page("Astral runtime", "/")] },
        { label: "Get Started", items: [
          item("Build and configure", "/BUILD"),
          item("Feature matrix", "/docs/FEATURE_MATRIX"),
          item("Model paths", "/docs/api/MODEL_PATHS"),
          item("Model presets", "/docs/api/MODEL_PRESETS"),
          item("Error handling", "/docs/api/ERROR_HANDLING"),
        ] },
        { label: "Runtime APIs", items: [
          item("Continuous batching", "/docs/api/CONTINUOUS_BATCHING"),
          item("Generation controls", "/docs/api/GENERATION_CONTROLS"),
          item("Tokenization", "/docs/api/TOKENIZATION"),
          item("Prompt cache", "/docs/api/PROMPT_CACHE"),
          item("Structured output", "/docs/api/STRUCTURED_OUTPUT"),
          item("LoRA adapters", "/docs/api/LORA_ADAPTERS"),
          item("Chunking", "/docs/api/CHUNKING"),
          item("Memory index", "/docs/api/MEMORY_INDEX"),
          item("Agent runtime", "/docs/api/AGENT_RUNTIME"),
          item("Async delivery", "/docs/api/ASYNC_DELIVERY"),
          item("Remote runtime", "/docs/api/REMOTE_RUNTIME"),
          item("Hot-path profiling", "/docs/api/HOT_PATH_PROFILING"),
        ] },
        { label: "Architecture", items: [
          item("Runtime architecture", "/docs/architecture/RUNTIME_ARCHITECTURE"),
          item("Retrieval architecture", "/docs/architecture/RETRIEVAL_ARCHITECTURE"),
          item("Backend architecture", "/docs/architecture/BACKEND_ARCHITECTURE"),
          item("Concurrency model", "/docs/architecture/CONCURRENCY_MODEL"),
          item("Memory architecture", "/docs/architecture/MEMORY_ARCHITECTURE"),
          item("Low-level primitives", "/docs/architecture/LOW_LEVEL_PRIMITIVES"),
          item("Embedded profile", "/docs/EMBEDDED_PROFILE"),
          item("Embedded examples", "/docs/embedded"),
          item("Vision and audio", "/docs/VISION_AUDIO"),
        ] },
        { label: "Integration", items: [
          item("Unity", "/plugins/unity"),
          { label: "Unity samples", items: [
            item("Streaming chat", "/plugins/unity/Samples~/StreamingChat"),
            item("Multiple conversations", "/plugins/unity/Samples~/MultipleConversations"),
            item("Stateful NPC", "/plugins/unity/Samples~/StatefulNpc"),
            item("Local knowledge", "/plugins/unity/Samples~/LocalKnowledge"),
            item("Character variants", "/plugins/unity/Samples~/CharacterVariants"),
            item("Multimodal input", "/plugins/unity/Samples~/MultimodalInput"),
          ] },
          item("Unreal Engine", "/plugins/unreal/AstralRT"),
          item("Unreal sample", "/examples/unreal/AstralSample"),
          item("Unreal Engine 5.7 quickstart", "/docs/integration/UNREAL_57_QUICKSTART"),
        ] },
        { label: "Profiling and CUDA", items: [
          item("Tracy profiling", "/docs/PROFILING_TRACY"),
          item("CUDA parity", "/docs/CUDA_PARITY"),
          item("CUDA kernel strategy", "/docs/CUDA_KERNEL_STRATEGY"),
        ] },
        { label: "Release", items: [
          item("ABI versioning", "/docs/ABI_VERSIONING"),
          item("Acceptance matrix", "/docs/release/RELEASE_ACCEPTANCE_MATRIX"),
          item("ABI manifest", "/docs/release/ABI_MANIFEST"),
          item("Dependency manifest", "/docs/release/DEPENDENCY_MANIFEST"),
          item("Third-party notices", "/docs/release/THIRD_PARTY_NOTICES"),
          item("Release notes template", "/docs/release/RELEASE_NOTES_TEMPLATE"),
        ] },
        { label: "Project", items: [
          page("Documentation map", "/docs"),
          item("Contributing", "/CONTRIBUTING"),
          item("Security", "/SECURITY"),
          item("Changelog", "/CHANGELOG"),
          item("Roadmap", "/ROADMAP"),
          item("Coding standards", "/docs/rules/CODING_STANDARDS"),
        ] },
      ],
    },
  },
  theme: {
    mode: "system",
    accent: { light: "#08766f", dark: "#61c5bc" },
    background: { light: "#ffffff", dark: "#11181f" },
    radius: "sm",
  },
});
