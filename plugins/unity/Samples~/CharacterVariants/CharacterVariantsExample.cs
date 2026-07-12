using System.Collections;
using System.IO;
using Astral.Runtime;
using Unity.Collections;
using UnityEngine;

namespace Astral.Examples
{
    public sealed class CharacterVariantsExample : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private string modelPath = "mock";
        [SerializeField] private string backendName = "mock";
        [SerializeField] private string adapterPath = "";
        [SerializeField] [Range(0.0f, 2.0f)] private float adapterScale = 1.0f;

        [Header("Character")]
        [SerializeField] [TextArea(2, 6)] private string systemPrompt =
            "You are a cautious scout. Return one JSON object with a line and confidence.";
        [SerializeField] [TextArea(2, 6)] private string userPrompt = "Report what is beyond the ridge.";
        [SerializeField] private string stopSequence = "</turn>";
        [SerializeField] [TextArea(3, 8)] private string jsonSchema =
            "{\"type\":\"object\",\"properties\":{\"line\":{\"type\":\"string\"},\"confidence\":{\"type\":\"number\"}},\"required\":[\"line\",\"confidence\"]}";
        [SerializeField] private string cacheFileName = "astral-character-prompts.cache";

        private AstralModel model;
        private AstralPromptCache promptCache;
        private AstralAdapter adapter;
        private AstralAgent cachedAgent;
        private AstralSession adaptedSession;
        private NativeArray<byte> streamBuffer;
        private Coroutine activeRun;

        private string CachePath => Path.Combine(Application.persistentDataPath, cacheFileName);

        public void PreparePromptCache()
        {
            if (!EnsureRuntime())
            {
                return;
            }

            try
            {
                EnsureModel();
                promptCache?.Dispose();
                promptCache = AstralPromptCache.Create(new AstralPromptCacheConfig
                {
                    flags = AstralNative.AstralPromptCacheFlags.TrackStats
                });

                uint tokenCount = model.CountTokens(systemPrompt, addSpecial: true);
                using (var tokens = model.Tokenize(systemPrompt, Allocator.Temp, addSpecial: true))
                using (var restored = new NativeArray<int>(tokens.Length, Allocator.Temp))
                {
                    var key = AstralPromptCache.KeyFromString(
                        model,
                        AstralNative.AstralPromptSectionKind.System,
                        generation: 1,
                        systemPrompt);
                    promptCache.PutTokens(ref key, tokens);
                    uint restoredCount = promptCache.GetTokens(ref key, restored);
                    Debug.Log($"[CharacterVariants] Cached {restoredCount}/{tokenCount} system-prompt tokens.");
                }

                File.WriteAllBytes(CachePath, promptCache.Save());
                Debug.Log($"[CharacterVariants] Saved prompt cache to {CachePath}.");
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[CharacterVariants] {ex.Message}");
            }
        }

        public void LoadPromptCache()
        {
            if (!EnsureRuntime() || !File.Exists(CachePath))
            {
                Debug.LogError($"[CharacterVariants] Prompt cache snapshot not found: {CachePath}");
                return;
            }

            try
            {
                EnsureModel();
                promptCache?.Dispose();
                promptCache = AstralPromptCache.Load(
                    new AstralPromptCacheConfig { flags = AstralNative.AstralPromptCacheFlags.TrackStats },
                    File.ReadAllBytes(CachePath));
                Debug.Log($"[CharacterVariants] Loaded {promptCache.GetStats().entries} cache entries.");
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[CharacterVariants] {ex.Message}");
            }
        }

        public void RunCachedAgent()
        {
            if (!EnsureRuntime())
            {
                return;
            }

            StopActive();
            try
            {
                EnsureModel();
                if (promptCache == null || !promptCache.IsValid)
                {
                    PreparePromptCache();
                }
                if (promptCache == null || !promptCache.IsValid)
                {
                    throw new AstralException("Prompt cache preparation failed.");
                }
                var config = AstralAgentConfig.Default;
                config.systemPrompt = systemPrompt;
                config.promptCache = promptCache.Handle;
                config.maxTokens = 96;
                cachedAgent = AstralAgent.Create(model, config);
                streamBuffer = new NativeArray<byte>(4096, Allocator.Persistent);
                activeRun = StartCoroutine(StreamAgent());
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[CharacterVariants] {ex.Message}");
                DisposeActive();
            }
        }

        public void RunAdaptedSession()
        {
            if (!EnsureRuntime())
            {
                return;
            }

            StopActive();
            try
            {
                EnsureModel();
                adaptedSession = AstralSession.Create(model);
                adaptedSession.SetSystemPrompt(systemPrompt);
                adaptedSession.StopAdd(stopSequence);
                adaptedSession.SetGrammarJsonSchema(jsonSchema);

                if (!string.IsNullOrWhiteSpace(adapterPath))
                {
                    if (!File.Exists(adapterPath) && backendName != "mock")
                    {
                        throw new AstralException($"Adapter not found: {adapterPath}");
                    }
                    adapter?.Dispose();
                    adapter = model.LoadAdapter(adapterPath);
                    adaptedSession.AddAdapter(adapter, adapterScale);
                    adaptedSession.SetAdapterScale(0, adapterScale);
                }

                Debug.Log($"[CharacterVariants] User prompt has {model.CountTokens(userPrompt)} tokens.");
                adaptedSession.Feed(userPrompt, finalize: true);
                adaptedSession.Decode();
                streamBuffer = new NativeArray<byte>(4096, Allocator.Persistent);
                activeRun = StartCoroutine(StreamSession());
            }
            catch (AstralException ex)
            {
                Debug.LogError($"[CharacterVariants] {ex.Message}");
                DisposeActive();
            }
        }

        public void Cancel()
        {
            if (cachedAgent != null && cachedAgent.IsValid)
            {
                AstralRequest.TryCancel(AstralRequest.FromAgentChat(cachedAgent), out _);
            }
            if (adaptedSession != null && adaptedSession.IsValid)
            {
                AstralRequest.TryCancel(AstralRequest.FromSession(adaptedSession), out _);
            }
        }

        private IEnumerator StreamAgent()
        {
            cachedAgent.EnqueueChat(userPrompt);
            while (true)
            {
                int bytesRead = cachedAgent.ReadChat(streamBuffer, timeoutMs: 0);
                if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
                {
                    yield return null;
                    continue;
                }
                if (bytesRead == 0)
                {
                    break;
                }
                if (bytesRead < 0)
                {
                    Debug.Log($"[CharacterVariants] Agent chat ended: {AstralRuntime.GetErrorString(bytesRead)}");
                    break;
                }
                Debug.Log($"[CharacterVariants] {new NativeSlice<byte>(streamBuffer, 0, bytesRead).ToUtf8String()}");
                yield return null;
            }

            var result = cachedAgent.GetChatResult();
            Debug.Log($"[CharacterVariants] Cache hits {result.prompt_cache_hits}; reused {result.prompt_cache_reused_tokens} tokens.");
            cachedAgent.ReleaseSlot();
            DisposeActive();
            activeRun = null;
        }

        private IEnumerator StreamSession()
        {
            while (true)
            {
                int bytesRead = adaptedSession.ReadStream(streamBuffer, timeoutMs: 0);
                if (bytesRead == AstralNative.ASTRAL_E_TIMEOUT)
                {
                    yield return null;
                    continue;
                }
                if (bytesRead <= 0)
                {
                    break;
                }
                Debug.Log($"[CharacterVariants] {new NativeSlice<byte>(streamBuffer, 0, bytesRead).ToUtf8String()}");
                yield return null;
            }
            DisposeActive();
            activeRun = null;
        }

        private bool EnsureRuntime()
        {
            if (AstralRuntime.IsInitialized)
            {
                return true;
            }
            Debug.LogError("[CharacterVariants] Add AstralRuntimeInitializer before running the sample.");
            return false;
        }

        private void EnsureModel()
        {
            if (model != null && model.IsValid)
            {
                return;
            }
            model = AstralModel.Load(modelPath, new AstralModelConfig { backendName = backendName });
        }

        private void StopActive()
        {
            Cancel();
            if (activeRun != null)
            {
                StopCoroutine(activeRun);
                activeRun = null;
            }
            DisposeActive();
        }

        private void DisposeActive()
        {
            cachedAgent?.Dispose();
            cachedAgent = null;
            adaptedSession?.Dispose();
            adaptedSession = null;
            if (streamBuffer.IsCreated)
            {
                streamBuffer.Dispose();
                streamBuffer = default;
            }
        }

        private void OnDestroy()
        {
            StopActive();
            adapter?.Dispose();
            promptCache?.Dispose();
            model?.Dispose();
        }
    }
}
