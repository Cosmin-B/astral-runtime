using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NUnit.Framework;
using Unity.Collections;
using UnityEngine;

namespace Astral.Runtime.Tests
{
    public sealed class AstralAbiTests
    {
        private static void RequireNative()
        {
            try
            {
                AstralNative.astral_version(out _, out _, out _);
            }
            catch (DllNotFoundException e)
            {
                if (RequireNativeLibrary)
                {
                    Assert.Fail($"Astral native library not found: {e.Message}");
                }
                Assert.Ignore($"Astral native library not found: {e.Message}");
            }
            catch (EntryPointNotFoundException e)
            {
                if (RequireNativeLibrary)
                {
                    Assert.Fail($"Astral native entry point not found: {e.Message}");
                }
                Assert.Ignore($"Astral native entry point not found: {e.Message}");
            }
            catch (BadImageFormatException e)
            {
                if (RequireNativeLibrary)
                {
                    Assert.Fail($"Astral native library is not loadable for this platform: {e.Message}");
                }
                Assert.Ignore($"Astral native library is not loadable for this platform: {e.Message}");
            }
        }

        private static bool RequireNativeLibrary =>
            Environment.GetEnvironmentVariable("ASTRAL_UNITY_REQUIRE_NATIVE") == "1";

        [Test]
        public void Abi_StructSizes_AreExpected()
        {
            int expectedSpan = IntPtr.Size == 8 ? 16 : 8;
            Assert.AreEqual(expectedSpan, Marshal.SizeOf<AstralNative.AstralSpanU8>());
            Assert.AreEqual(expectedSpan, Marshal.SizeOf<AstralNative.AstralMutSpanU8>());

            Assert.AreEqual(8, Marshal.SizeOf<AstralNative.AstralHandle>());
            Assert.AreEqual(8, Marshal.SizeOf<ulong>()); // AstralCaps
            Assert.AreEqual(IntPtr.Size == 8 ? 24 : 12, Marshal.SizeOf<AstralNative.AstralTokenizeRequest>());

            int expectedInit = IntPtr.Size == 8 ? 64 : 48;
            Assert.AreEqual(expectedInit, Marshal.SizeOf<AstralNative.AstralInit>());

            int expectedModelDesc = IntPtr.Size == 8 ? 168 : 116;
            Assert.AreEqual(expectedModelDesc, Marshal.SizeOf<AstralNative.AstralModelDesc>());
            int expectedModelPathResolveDesc = IntPtr.Size == 8 ? 96 : 56;
            Assert.AreEqual(expectedModelPathResolveDesc, Marshal.SizeOf<AstralNative.AstralModelPathResolveDesc>());

            int expectedImageDesc = IntPtr.Size == 8 ? 64 : 52;
            int expectedAudioDesc = IntPtr.Size == 8 ? 72 : 60;
            int expectedModelMediaDesc = IntPtr.Size == 8 ? 104 : 72;
            int expectedMediaInfo = 28;
            Assert.AreEqual(expectedImageDesc, Marshal.SizeOf<AstralNative.AstralImageDesc>());
            Assert.AreEqual(expectedAudioDesc, Marshal.SizeOf<AstralNative.AstralAudioDesc>());
            Assert.AreEqual(expectedModelMediaDesc, Marshal.SizeOf<AstralNative.AstralModelMediaDesc>());
            Assert.AreEqual(expectedMediaInfo, Marshal.SizeOf<AstralNative.AstralMediaInfo>());

            Assert.AreEqual(32, Marshal.SizeOf<AstralNative.AstralSessionDesc>());
            Assert.AreEqual(16, Marshal.SizeOf<AstralNative.AstralModelLimits>());
            Assert.AreEqual(56, Marshal.SizeOf<AstralNative.AstralSamplerDesc>());
            Assert.AreEqual(40, Marshal.SizeOf<AstralNative.AstralStats>());
            Assert.AreEqual(24, Marshal.SizeOf<AstralNative.AstralRequestRef>());
            Assert.AreEqual(40, Marshal.SizeOf<AstralNative.AstralRequestStatus>());
            Assert.AreEqual(24, Marshal.SizeOf<AstralNative.AstralPromptCacheDesc>());
            Assert.AreEqual(32, Marshal.SizeOf<AstralNative.AstralPromptCacheKey>());
            Assert.AreEqual(56, Marshal.SizeOf<AstralNative.AstralPromptCacheStats>());
            Assert.AreEqual(24, Marshal.SizeOf<AstralNative.AstralAdapterDesc>());
            Assert.AreEqual(24, Marshal.SizeOf<AstralNative.AstralAdapterInfo>());
            Assert.AreEqual(56, Marshal.SizeOf<AstralNative.AstralToolDesc>());
            Assert.AreEqual(24, Marshal.SizeOf<AstralNative.AstralToolsetDesc>());
            Assert.AreEqual(56, Marshal.SizeOf<AstralNative.AstralToolInfo>());
            Assert.AreEqual(48, Marshal.SizeOf<AstralNative.AstralToolCallResult>());
            Assert.AreEqual(32, Marshal.SizeOf<AstralNative.AstralMemoryIndexDesc>());
            Assert.AreEqual(32, Marshal.SizeOf<AstralNative.AstralMemoryRecord>());
            Assert.AreEqual(20, Marshal.SizeOf<AstralNative.AstralMemorySearchDesc>());
            Assert.AreEqual(32, Marshal.SizeOf<AstralNative.AstralMemorySearchResult>());
            Assert.AreEqual(120, Marshal.SizeOf<AstralNative.AstralMemoryStats>());
            Assert.AreEqual(104, Marshal.SizeOf<AstralNative.AstralMemorySnapshotInfo>());
            int expectedAgentDesc = IntPtr.Size == 8 ? 136 : 108;
            Assert.AreEqual(expectedAgentDesc, Marshal.SizeOf<AstralNative.AstralAgentDesc>());
            Assert.AreEqual(64, Marshal.SizeOf<AstralNative.AstralAgentMemoryContextDesc>());
            Assert.AreEqual(72, Marshal.SizeOf<AstralNative.AstralAgentChatResult>());
        }

        [Test]
        public void ModelPath_Raw_ResolvesThroughNativeAbi()
        {
            RequireNative();

            const string modelPath = "Models/model.gguf";
            Assert.AreEqual(modelPath, AstralModelPath.Raw(modelPath).Resolve());
        }

        [Test]
        public void ModelPath_WrapperRoots_ResolveThroughNativeAbi()
        {
            RequireNative();

            const string relativePath = "Models/model.gguf";
            Assert.AreEqual(
                JoinPath(Application.streamingAssetsPath, relativePath),
                AstralModelPath.StreamingAssets(relativePath).Resolve());
            Assert.AreEqual(
                JoinPath(Application.persistentDataPath, relativePath),
                AstralModelPath.PersistentData(relativePath).Resolve());
            Assert.AreEqual(
                JoinPath(Application.temporaryCachePath, relativePath),
                AstralModelPath.TemporaryCache(relativePath).Resolve());
            Assert.AreEqual(
                JoinPath(Application.persistentDataPath, relativePath),
                AstralModelPath.Download(relativePath).Resolve());
        }

        [Test]
        public void ModelPath_NativeResolver_JoinsContentRoot()
        {
            RequireNative();

            const string relativePath = "Models/model.gguf";
            const string contentRoot = "/project/content";
            const string expectedPath = "/project/content/Models/model.gguf";
            using var pathBytes = new NativeArray<byte>(Encoding.UTF8.GetBytes(relativePath), Allocator.Temp);
            using var contentRootBytes = new NativeArray<byte>(Encoding.UTF8.GetBytes(contentRoot), Allocator.Temp);

            AstralNative.AstralModelPathResolveDesc desc = new AstralNative.AstralModelPathResolveDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralModelPathResolveDesc>(),
                root = AstralNative.AstralModelPathRoot.Content,
                path = AstralNative.AstralSpanU8.FromNativeArray(pathBytes),
                content_root = AstralNative.AstralSpanU8.FromNativeArray(contentRootBytes),
                flags = AstralNative.AstralModelPathResolveFlags.None
            };

            uint requiredBytes = 0;
            int err = AstralNative.astral_model_path_resolve(
                ref desc,
                new AstralNative.AstralMutSpanU8 { data = IntPtr.Zero, len = 0 },
                out requiredBytes);
            Assert.AreEqual(AstralNative.ASTRAL_E_NOMEM, err);
            Assert.AreEqual(Encoding.UTF8.GetByteCount(expectedPath), requiredBytes);

            byte[] output = new byte[(int)requiredBytes];
            GCHandle handle = GCHandle.Alloc(output, GCHandleType.Pinned);
            try
            {
                AstralNative.AstralMutSpanU8 outSpan = new AstralNative.AstralMutSpanU8
                {
                    data = handle.AddrOfPinnedObject(),
                    len = requiredBytes
                };
                err = AstralNative.astral_model_path_resolve(ref desc, outSpan, out requiredBytes);
            }
            finally
            {
                handle.Free();
            }

            Assert.AreEqual(AstralNative.ASTRAL_OK, err);
            Assert.AreEqual(expectedPath, Encoding.UTF8.GetString(output));
        }

        private static string JoinPath(string root, string relativePath)
        {
            if (root.EndsWith("/", StringComparison.Ordinal) || root.EndsWith("\\", StringComparison.Ordinal))
            {
                return root + relativePath;
            }

            return root + "/" + relativePath;
        }

        private static AstralNative.AstralRequestStatus NewRequestStatus(
            AstralNative.AstralRequestState state)
        {
            return new AstralNative.AstralRequestStatus
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralRequestStatus>(),
                kind = AstralNative.AstralRequestKind.Session,
                state = state,
                flags = AstralNative.AstralRequestFlags.None,
                result = AstralNative.ASTRAL_OK
            };
        }

        private static void AssertRequestState(
            AstralNative.AstralRequestState state,
            bool queued,
            bool running,
            bool completed,
            bool canceled,
            bool failed,
            bool active,
            bool terminal)
        {
            var status = NewRequestStatus(state);
            Assert.AreEqual(queued, AstralRequest.IsQueued(status));
            Assert.AreEqual(running, AstralRequest.IsRunning(status));
            Assert.AreEqual(completed, AstralRequest.IsCompleted(status));
            Assert.AreEqual(canceled, AstralRequest.IsCanceled(status));
            Assert.AreEqual(failed, AstralRequest.IsFailed(status));
            Assert.AreEqual(active, AstralRequest.IsActive(status));
            Assert.AreEqual(terminal, AstralRequest.IsTerminal(status));
            Assert.False(AstralRequest.HasTicket(status));
            Assert.False(AstralRequest.IsStream(status));
        }

        [Test]
        public void Abi_Handle_ZeroIsInvalid()
        {
            RequireNative();

            var h = new AstralNative.AstralHandle { value = 0 };
            Assert.False(h.IsValid);
            Assert.AreEqual(0, AstralNative.astral_handle_valid(h));
        }

        [Test]
        public void RequestLifecycle_InvalidHandles_ReturnNativeErrors()
        {
            RequireNative();

            var invalidHandle = AstralNative.AstralHandle.Invalid;
            var request = new AstralNative.AstralRequestRef();
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_request_from_session(invalidHandle, out request));
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_request_from_conversation(invalidHandle, out request));
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_request_from_agent_chat(invalidHandle, out request));
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_request_from_embedding(invalidHandle, 1, out request));
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_request_from_memory_search(invalidHandle, out request));
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_agent_release_slot(invalidHandle));

            request = new AstralNative.AstralRequestRef
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralRequestRef>(),
                kind = AstralNative.AstralRequestKind.Session,
                owner = invalidHandle,
                ticket = 0
            };
            var status = new AstralNative.AstralRequestStatus
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralRequestStatus>()
            };
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_request_state(ref request, ref status));
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_request_wait(ref request, 0, ref status));
            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, AstralNative.astral_request_cancel(ref request));
        }

        [Test]
        public void RequestLifecycle_StatusHelpers_MapNativeStateAndFlags()
        {
            var status = new AstralNative.AstralRequestStatus
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralRequestStatus>(),
                kind = AstralNative.AstralRequestKind.Embedding,
                state = AstralNative.AstralRequestState.Queued,
                flags = AstralNative.AstralRequestFlags.Ticket,
                ticket = 42,
                result = AstralNative.ASTRAL_OK
            };

            Assert.True(AstralRequest.IsQueued(status));
            Assert.True(AstralRequest.IsActive(status));
            Assert.True(AstralRequest.HasTicket(status));
            Assert.AreEqual("Embedding", AstralRequest.KindName(status.kind));
            Assert.AreEqual("Queued", AstralRequest.StateName(status.state));
            Assert.False(AstralRequest.IsTerminal(status));
            Assert.False(AstralRequest.IsStream(status));

            status.state = AstralNative.AstralRequestState.Running;
            status.flags |= AstralNative.AstralRequestFlags.Stream;
            Assert.True(AstralRequest.IsRunning(status));
            Assert.True(AstralRequest.IsActive(status));
            Assert.AreEqual("Running", AstralRequest.StateName(status.state));
            Assert.True(AstralRequest.IsStream(status));

            status.state = AstralNative.AstralRequestState.Completed;
            Assert.True(AstralRequest.IsCompleted(status));
            Assert.True(AstralRequest.IsTerminal(status));
            Assert.AreEqual("Completed", AstralRequest.StateName(status.state));
            Assert.True(AstralRequest.IsSuccessful(status));
            Assert.False(AstralRequest.IsActive(status));

            status.state = AstralNative.AstralRequestState.Failed;
            status.result = AstralNative.ASTRAL_E_BACKEND;
            Assert.True(AstralRequest.IsFailed(status));
            Assert.True(AstralRequest.IsTerminal(status));
            Assert.AreEqual("Failed", AstralRequest.StateName(status.state));
            Assert.False(AstralRequest.IsSuccessful(status));

            status.state = AstralNative.AstralRequestState.Canceled;
            status.result = AstralNative.ASTRAL_E_CANCELED;
            Assert.True(AstralRequest.IsCanceled(status));
            Assert.AreEqual("Canceled", AstralRequest.StateName(status.state));
            Assert.True(AstralRequest.IsTerminal(status));
        }

        [Test]
        public void RequestLifecycle_LabelHelpers_CoverNativeEnums()
        {
            Assert.AreEqual("None", AstralRequest.KindName(AstralNative.AstralRequestKind.None));
            Assert.AreEqual("Session", AstralRequest.KindName(AstralNative.AstralRequestKind.Session));
            Assert.AreEqual("Conversation", AstralRequest.KindName(AstralNative.AstralRequestKind.Conversation));
            Assert.AreEqual("AgentChat", AstralRequest.KindName(AstralNative.AstralRequestKind.AgentChat));
            Assert.AreEqual("Embedding", AstralRequest.KindName(AstralNative.AstralRequestKind.Embedding));
            Assert.AreEqual("MemorySearch", AstralRequest.KindName(AstralNative.AstralRequestKind.MemorySearch));
            Assert.AreEqual("None", AstralRequest.KindName((AstralNative.AstralRequestKind)999u));

            Assert.AreEqual("Invalid", AstralRequest.StateName(AstralNative.AstralRequestState.Invalid));
            Assert.AreEqual("Queued", AstralRequest.StateName(AstralNative.AstralRequestState.Queued));
            Assert.AreEqual("Running", AstralRequest.StateName(AstralNative.AstralRequestState.Running));
            Assert.AreEqual("Completed", AstralRequest.StateName(AstralNative.AstralRequestState.Completed));
            Assert.AreEqual("Canceled", AstralRequest.StateName(AstralNative.AstralRequestState.Canceled));
            Assert.AreEqual("Failed", AstralRequest.StateName(AstralNative.AstralRequestState.Failed));
            Assert.AreEqual("Invalid", AstralRequest.StateName((AstralNative.AstralRequestState)999u));
        }

        [Test]
        public void RequestLifecycle_StatusHelpers_CoverAllNativeStates()
        {
            AssertRequestState(
                AstralNative.AstralRequestState.Invalid,
                queued: false,
                running: false,
                completed: false,
                canceled: false,
                failed: false,
                active: false,
                terminal: false);
            AssertRequestState(
                AstralNative.AstralRequestState.Queued,
                queued: true,
                running: false,
                completed: false,
                canceled: false,
                failed: false,
                active: true,
                terminal: false);
            AssertRequestState(
                AstralNative.AstralRequestState.Running,
                queued: false,
                running: true,
                completed: false,
                canceled: false,
                failed: false,
                active: true,
                terminal: false);
            AssertRequestState(
                AstralNative.AstralRequestState.Completed,
                queued: false,
                running: false,
                completed: true,
                canceled: false,
                failed: false,
                active: false,
                terminal: true);
            AssertRequestState(
                AstralNative.AstralRequestState.Canceled,
                queued: false,
                running: false,
                completed: false,
                canceled: true,
                failed: false,
                active: false,
                terminal: true);
            AssertRequestState(
                AstralNative.AstralRequestState.Failed,
                queued: false,
                running: false,
                completed: false,
                canceled: false,
                failed: true,
                active: false,
                terminal: true);

            var status = NewRequestStatus(AstralNative.AstralRequestState.Completed);
            status.flags = AstralNative.AstralRequestFlags.Ticket | AstralNative.AstralRequestFlags.Stream;
            status.result = AstralNative.ASTRAL_E_TIMEOUT;
            Assert.True(AstralRequest.HasTicket(status));
            Assert.True(AstralRequest.IsStream(status));
            Assert.False(AstralRequest.IsSuccessful(status));
        }

        [Test]
        public void PromptCache_InvalidModelKey_ReturnsNativeError()
        {
            RequireNative();

            var key = new AstralNative.AstralPromptCacheKey
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralPromptCacheKey>()
            };
            var empty = new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 };

            int err = AstralNative.astral_prompt_cache_key_from_bytes(
                AstralNative.AstralHandle.Invalid,
                AstralNative.AstralPromptSectionKind.System,
                1,
                empty,
                ref key);

            Assert.AreEqual(AstralNative.ASTRAL_E_INVALID, err);
        }

        [Test]
        public void Adapter_InvalidHandleDiagnostics_ReturnNativeErrors()
        {
            RequireNative();

            var info = new AstralNative.AstralAdapterInfo
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralAdapterInfo>()
            };
            Assert.AreEqual(
                AstralNative.ASTRAL_E_INVALID,
                AstralNative.astral_model_adapter_info(AstralNative.AstralHandle.Invalid, ref info));

            Assert.AreEqual(
                AstralNative.ASTRAL_E_INVALID,
                AstralNative.astral_model_adapter_path_copy(
                    AstralNative.AstralHandle.Invalid,
                    new AstralNative.AstralMutSpanU8 { data = IntPtr.Zero, len = 0 },
                    out _));
        }

        [Test]
        public void MemoryIndex_SearchBatch_ReturnsPerQueryResults()
        {
            RequireNative();

            AstralRuntime.Initialize(new AstralConfig
            {
                reserveBytes = 64UL << 20,
                threadCount = 1,
                useUnityAllocator = false,
                enableLogging = false
            });

            try
            {
                using var index = AstralMemoryIndex.Create(new AstralMemoryIndexConfig
                {
                    dim = 2,
                    capacity = 2,
                    metric = AstralNative.AstralMemoryMetric.Cosine,
                    indexKind = AstralNative.AstralMemoryIndexKind.Flat
                });

                using var records = new NativeArray<AstralNative.AstralMemoryRecord>(2, Allocator.Temp);
                records[0] = new AstralNative.AstralMemoryRecord
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralMemoryRecord>(),
                    group_id = 7,
                    key = 1001,
                    document_id = 10,
                    chunk_id = 100
                };
                records[1] = new AstralNative.AstralMemoryRecord
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralMemoryRecord>(),
                    group_id = 7,
                    key = 1002,
                    document_id = 20,
                    chunk_id = 200
                };
                using var vectors = new NativeArray<float>(new[] { 1.0f, 0.0f, 0.0f, 1.0f }, Allocator.Temp);
                index.AddBatch(records, vectors);

                var desc = AstralMemoryIndex.SearchDesc(topK: 1, groupId: AstralMemoryIndex.DefaultSearchGroupId);
                using var queries = new NativeArray<float>(new[] { 1.0f, 0.0f, 0.0f, 1.0f }, Allocator.Temp);
                using var results = new NativeArray<AstralNative.AstralMemorySearchResult>(2, Allocator.Temp);
                using var counts = new NativeArray<uint>(2, Allocator.Temp);

                index.SearchBatch(ref desc, queries, queryCount: 2, results, counts);

                Assert.AreEqual(1u, counts[0]);
                Assert.AreEqual(1u, counts[1]);
                Assert.AreEqual(1001UL, results[0].key);
                Assert.AreEqual(1002UL, results[1].key);
            }
            finally
            {
                AstralRuntime.Shutdown();
            }
        }

        [Test]
        public void ToolCall_StatusHelpers_MapParseStatus()
        {
            var parsed = new AstralToolCall
            {
                found = true,
                parseStatus = AstralNative.ASTRAL_OK
            };
            Assert.True(parsed.Parsed);
            Assert.False(parsed.Missing);
            Assert.False(parsed.Malformed);

            var missing = new AstralToolCall
            {
                found = false,
                parseStatus = AstralNative.ASTRAL_E_NOT_FOUND
            };
            Assert.False(missing.Parsed);
            Assert.True(missing.Missing);
            Assert.False(missing.Malformed);

            var malformed = new AstralToolCall
            {
                found = true,
                parseStatus = AstralNative.ASTRAL_E_INVALID
            };
            Assert.False(malformed.Parsed);
            Assert.False(malformed.Missing);
            Assert.True(malformed.Malformed);
        }

        [Test]
        public void AgentWrapper_Toolset_ParsesToolCalls()
        {
            RequireNative();

            AstralRuntime.Initialize(new AstralConfig
            {
                reserveBytes = 256UL << 20,
                threadCount = 1,
                useUnityAllocator = false,
                enableLogging = false
            });

            try
            {
                using var model = AstralModel.Load("mock-model", new AstralModelConfig
                {
                    backendName = "mock",
                    contextSize = 128,
                    batchSize = 64
                });
                using var toolset = AstralToolset.Create(
                    new[]
                    {
                        new AstralToolDefinition
                        {
                            toolId = 41,
                            name = "agent_lookup",
                            description = "search agent memory",
                            jsonSchema = "{}"
                        }
                    },
                    AstralNative.AstralToolChoiceMode.Required);

                var config = AstralAgentConfig.Default.WithToolset(toolset, AstralNative.AstralToolChoiceMode.Required);
                config.maxTokens = 8;
                config.maxMessages = 4;
                config.maxPromptBytes = 1024;
                config.systemPrompt = "stay terse.";
                config.summary = "prior context";
                config.memoryContext = "retrieved context";

                using var agent = AstralAgent.Create(model, config);
                Assert.AreEqual(config.systemPrompt, agent.GetSystemPrompt());
                Assert.AreEqual(config.summary, agent.GetSummary());
                Assert.AreEqual(config.memoryContext, agent.GetMemoryContext());
                AstralToolCall call = agent.ParseToolCall("{\"name\":\"agent_lookup\",\"arguments\":{\"q\":\"agent\"}}");

                Assert.True(call.Parsed);
                Assert.AreEqual(41u, call.toolId);
                Assert.AreEqual("agent_lookup", call.name);
                Assert.AreEqual("{\"q\":\"agent\"}", call.argumentsJson);
            }
            finally
            {
                AstralRuntime.Shutdown();
            }
        }

        [Test]
        public void MockBackend_E2E_StreamAndReset_Works()
        {
            RequireNative();

            var cfg = new AstralNative.AstralInit
            {
                reserve_bytes = 256UL << 20,
                thread_count = 1,
                numa_node = 0xFFFFFFFFu,
                enable_hugepages = 0
            };

            int err = AstralNative.astral_init(ref cfg);
            Assert.AreEqual(AstralNative.ASTRAL_OK, err);

            AstralNative.AstralHandle model = AstralNative.AstralHandle.Invalid;
            AstralNative.AstralHandle session = AstralNative.AstralHandle.Invalid;

            try
            {
                using var backendNameBytes = new NativeArray<byte>(
                    Encoding.UTF8.GetBytes("mock"),
                    Allocator.Temp
                );

                var modelDesc = new AstralNative.AstralModelDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralModelDesc>(),
                    source_kind = AstralNative.AstralModelSourceKind.Path,
                    model_path = new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 },
                    backend_name = AstralNative.AstralSpanU8.FromNativeArray(backendNameBytes),
                    gpu_layers = 0,
                    n_ctx = 0,
                    n_batch = 0,
                    n_threads = 0,
                    embeddings_only = 0
                };

                err = AstralNative.astral_model_load(ref modelDesc, out model);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(model.IsValid);

                var sessionDesc = new AstralNative.AstralSessionDesc
                {
                    model = model,
                    max_tokens = 64,
                    temperature = 0.0f,
                    top_k = 0,
                    top_p = 1.0f,
                    stream_enabled = 1,
                    seed = 1234
                };

                err = AstralNative.astral_session_create(ref sessionDesc, out session);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(session.IsValid);

                string first = RunMockOnce(session);
                Assert.AreEqual("mock-backend", first);

                err = AstralNative.astral_session_reset(session, ref sessionDesc);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);

                string second = RunMockOnce(session);
                Assert.AreEqual("mock-backend", second);
            }
            finally
            {
                if (session.IsValid)
                {
                    AstralNative.astral_session_destroy(session);
                }
                if (model.IsValid)
                {
                    AstralNative.astral_model_release(model);
                }
                AstralNative.astral_shutdown();
            }
        }

        [Test]
        public void MockBackend_StopSequences_SuppressOutput_Works()
        {
            RequireNative();

            var cfg = new AstralNative.AstralInit
            {
                reserve_bytes = 256UL << 20,
                thread_count = 1,
                numa_node = 0xFFFFFFFFu,
                enable_hugepages = 0
            };

            int err = AstralNative.astral_init(ref cfg);
            Assert.AreEqual(AstralNative.ASTRAL_OK, err);

            AstralNative.AstralHandle model = AstralNative.AstralHandle.Invalid;
            AstralNative.AstralHandle session = AstralNative.AstralHandle.Invalid;

            try
            {
                using var backendNameBytes = new NativeArray<byte>(
                    Encoding.UTF8.GetBytes("mock"),
                    Allocator.Temp
                );

                var modelDesc = new AstralNative.AstralModelDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralModelDesc>(),
                    source_kind = AstralNative.AstralModelSourceKind.Path,
                    model_path = new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 },
                    backend_name = AstralNative.AstralSpanU8.FromNativeArray(backendNameBytes),
                    gpu_layers = 0,
                    n_ctx = 0,
                    n_batch = 0,
                    n_threads = 0,
                    embeddings_only = 0
                };

                err = AstralNative.astral_model_load(ref modelDesc, out model);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(model.IsValid);

                var sessionDesc = new AstralNative.AstralSessionDesc
                {
                    model = model,
                    max_tokens = 64,
                    temperature = 0.0f,
                    top_k = 0,
                    top_p = 1.0f,
                    stream_enabled = 1,
                    seed = 1
                };

                err = AstralNative.astral_session_create(ref sessionDesc, out session);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(session.IsValid);

                err = AstralNative.astral_session_stop_clear(session);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);

                using var stopBytes = new NativeArray<byte>(Encoding.UTF8.GetBytes("backend"), Allocator.Temp);
                var stopSpan = AstralNative.AstralSpanU8.FromNativeArray(stopBytes);
                err = AstralNative.astral_session_stop_add_utf8(session, stopSpan);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);

                string outText = RunMockOnce(session);
                Assert.AreEqual("mock-", outText);
            }
            finally
            {
                if (session.IsValid)
                {
                    AstralNative.astral_session_destroy(session);
                }
                if (model.IsValid)
                {
                    AstralNative.astral_model_release(model);
                }
                AstralNative.astral_shutdown();
            }
        }

        [Test]
        public void MockBackend_Embeddings_Works()
        {
            RequireNative();

            var cfg = new AstralNative.AstralInit
            {
                reserve_bytes = 256UL << 20,
                thread_count = 1,
                numa_node = 0xFFFFFFFFu,
                enable_hugepages = 0
            };

            int err = AstralNative.astral_init(ref cfg);
            Assert.AreEqual(AstralNative.ASTRAL_OK, err);

            AstralNative.AstralHandle model = AstralNative.AstralHandle.Invalid;
            AstralNative.AstralHandle emb = AstralNative.AstralHandle.Invalid;

            try
            {
                using var backendNameBytes = new NativeArray<byte>(
                    Encoding.UTF8.GetBytes("mock"),
                    Allocator.Temp
                );

                var modelDesc = new AstralNative.AstralModelDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralModelDesc>(),
                    source_kind = AstralNative.AstralModelSourceKind.Path,
                    model_path = new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 },
                    backend_name = AstralNative.AstralSpanU8.FromNativeArray(backendNameBytes),
                    gpu_layers = 0,
                    n_ctx = 0,
                    n_batch = 0,
                    n_threads = 0,
                    embeddings_only = 1
                };

                err = AstralNative.astral_model_load(ref modelDesc, out model);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(model.IsValid);

                err = AstralNative.astral_model_embedding_dim(model, out uint dim);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.AreEqual(8u, dim);

                err = AstralNative.astral_embed_create(model, out emb);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(emb.IsValid);

                using var textBytes = new NativeArray<byte>(Encoding.UTF8.GetBytes("abc"), Allocator.Temp);
                var textSpan = AstralNative.AstralSpanU8.FromNativeArray(textBytes);
                err = AstralNative.astral_embed_enqueue(emb, textSpan, out ulong ticket);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);

                using var outVec = new NativeArray<float>((int)dim, Allocator.Temp);
                unsafe
                {
                    var outSpan = new AstralNative.AstralMutSpanU8
                    {
                        data = (IntPtr)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafePtr(outVec),
                        len = (uint)(outVec.Length * sizeof(float))
                    };
                    err = AstralNative.astral_embed_collect(emb, ticket, outSpan);
                    Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                }

                // mock embedder: sum(tokens incl BOS=256) + i, so for "abc": 256 + 97 + 98 + 99 = 550.
                Assert.AreEqual(550.0f, outVec[0]);
                Assert.AreEqual(551.0f, outVec[1]);
                Assert.AreEqual(557.0f, outVec[7]);
            }
            finally
            {
                if (emb.IsValid)
                {
                    AstralNative.astral_embed_destroy(emb);
                }
                if (model.IsValid)
                {
                    AstralNative.astral_model_release(model);
                }
                AstralNative.astral_shutdown();
            }
        }

        [Test]
        public void MockBackend_MediaFeed_Works()
        {
            RequireNative();

            var cfg = new AstralNative.AstralInit
            {
                reserve_bytes = 256UL << 20,
                thread_count = 1,
                numa_node = 0xFFFFFFFFu,
                enable_hugepages = 0
            };

            int err = AstralNative.astral_init(ref cfg);
            Assert.AreEqual(AstralNative.ASTRAL_OK, err);

            AstralNative.AstralHandle model = AstralNative.AstralHandle.Invalid;
            AstralNative.AstralHandle session = AstralNative.AstralHandle.Invalid;

            try
            {
                using var backendNameBytes = new NativeArray<byte>(
                    Encoding.UTF8.GetBytes("mock"),
                    Allocator.Temp
                );

                var modelDesc = new AstralNative.AstralModelDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralModelDesc>(),
                    source_kind = AstralNative.AstralModelSourceKind.Path,
                    model_path = new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 },
                    backend_name = AstralNative.AstralSpanU8.FromNativeArray(backendNameBytes),
                    gpu_layers = 0,
                    n_ctx = 0,
                    n_batch = 0,
                    n_threads = 0,
                    embeddings_only = 0
                };

                err = AstralNative.astral_model_load(ref modelDesc, out model);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(model.IsValid);

                var sessionDesc = new AstralNative.AstralSessionDesc
                {
                    model = model,
                    max_tokens = 16,
                    temperature = 0.0f,
                    top_k = 0,
                    top_p = 1.0f,
                    stream_enabled = 0,
                    seed = 42
                };

                err = AstralNative.astral_session_create(ref sessionDesc, out session);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(session.IsValid);

                using var imageBytes = new NativeArray<byte>(3, Allocator.Temp);
                var imageDesc = new AstralNative.AstralImageDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralImageDesc>(),
                    format = AstralNative.AstralImageFormat.RGB8,
                    width = 1,
                    height = 1,
                    row_stride = 0,
                    flags = 0,
                    pixels = AstralNative.AstralSpanU8.FromNativeArray(imageBytes)
                };

                err = AstralNative.astral_session_feed_image(session, ref imageDesc, finalize: 1);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);

                using var audioBytes = new NativeArray<byte>(8, Allocator.Temp);
                var audioDesc = new AstralNative.AstralAudioDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>(),
                    format = AstralNative.AstralAudioFormat.I16,
                    channels = 1,
                    sample_rate = 16000,
                    frame_count = 4,
                    samples = AstralNative.AstralSpanU8.FromNativeArray(audioBytes),
                    flags = 0
                };

                err = AstralNative.astral_session_feed_audio(session, ref audioDesc, finalize: 1);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
            }
            finally
            {
                if (session.IsValid)
                {
                    AstralNative.astral_session_destroy(session);
                }
                if (model.IsValid)
                {
                    AstralNative.astral_model_release(model);
                }
                AstralNative.astral_shutdown();
            }
        }

        [Test]
        public unsafe void MockBackend_MultimodalEmbeddings_Works()
        {
            RequireNative();

            var cfg = new AstralNative.AstralInit
            {
                reserve_bytes = 256UL << 20,
                thread_count = 1,
                numa_node = 0xFFFFFFFFu,
                enable_hugepages = 0
            };

            int err = AstralNative.astral_init(ref cfg);
            Assert.AreEqual(AstralNative.ASTRAL_OK, err);

            AstralNative.AstralHandle model = AstralNative.AstralHandle.Invalid;
            AstralNative.AstralHandle emb = AstralNative.AstralHandle.Invalid;

            try
            {
                using var backendNameBytes = new NativeArray<byte>(
                    Encoding.UTF8.GetBytes("mock"),
                    Allocator.Temp
                );

                var modelDesc = new AstralNative.AstralModelDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralModelDesc>(),
                    source_kind = AstralNative.AstralModelSourceKind.Path,
                    model_path = new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 },
                    backend_name = AstralNative.AstralSpanU8.FromNativeArray(backendNameBytes),
                    gpu_layers = 0,
                    n_ctx = 0,
                    n_batch = 0,
                    n_threads = 0,
                    embeddings_only = 1
                };

                err = AstralNative.astral_model_load(ref modelDesc, out model);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(model.IsValid);

                err = AstralNative.astral_model_embedding_dim(model, out uint dim);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.AreEqual(8u, dim);

                err = AstralNative.astral_embed_create(model, out emb);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
                Assert.True(emb.IsValid);

                using var textBytes = new NativeArray<byte>(Encoding.UTF8.GetBytes("abc"), Allocator.Temp);
                var textSpan = AstralNative.AstralSpanU8.FromNativeArray(textBytes);

                using var imageBytes = new NativeArray<byte>(3, Allocator.Temp);
                var imageDesc = new AstralNative.AstralImageDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralImageDesc>(),
                    format = AstralNative.AstralImageFormat.RGB8,
                    width = 1,
                    height = 1,
                    row_stride = 0,
                    flags = 0,
                    pixels = AstralNative.AstralSpanU8.FromNativeArray(imageBytes)
                };

                using var audioBytes = new NativeArray<byte>(8, Allocator.Temp);
                var audioDesc = new AstralNative.AstralAudioDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>(),
                    format = AstralNative.AstralAudioFormat.I16,
                    channels = 1,
                    sample_rate = 16000,
                    frame_count = 4,
                    samples = AstralNative.AstralSpanU8.FromNativeArray(audioBytes),
                    flags = 0
                };

                ulong ticket;
                err = AstralNative.astral_embed_enqueue_multimodal(emb, textSpan, (IntPtr)(&imageDesc), (IntPtr)(&audioDesc), out ticket);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);

                using var outVec = new NativeArray<float>((int)dim, Allocator.Temp);
                var outSpan = new AstralNative.AstralMutSpanU8
                {
                    data = (IntPtr)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafePtr(outVec),
                    len = (uint)(outVec.Length * sizeof(float))
                };
                err = AstralNative.astral_embed_collect(emb, ticket, outSpan);
                Assert.AreEqual(AstralNative.ASTRAL_OK, err);
            }
            finally
            {
                if (emb.IsValid)
                {
                    AstralNative.astral_embed_destroy(emb);
                }
                if (model.IsValid)
                {
                    AstralNative.astral_model_release(model);
                }
                AstralNative.astral_shutdown();
            }
        }

        private static string RunMockOnce(AstralNative.AstralHandle session)
        {
            int err = AstralNative.astral_session_feed(
                session,
                new AstralNative.AstralSpanU8 { data = IntPtr.Zero, len = 0 },
                finalize: 1
            );
            Assert.AreEqual(AstralNative.ASTRAL_OK, err);

            err = AstralNative.astral_session_decode(session);
            Assert.AreEqual(AstralNative.ASTRAL_OK, err);

            var bytes = new List<byte>(64);
            using var buf = new NativeArray<byte>(128, Allocator.Temp);
            var outSpan = AstralNative.AstralMutSpanU8.FromNativeArray(buf);

            while (true)
            {
                int n = AstralNative.astral_stream_read(session, outSpan, timeout_ms: 100);
                if (n == 0)
                {
                    break;
                }
                if (n == AstralNative.ASTRAL_E_TIMEOUT)
                {
                    continue;
                }
                if (n < 0)
                {
                    Assert.Fail($"astral_stream_read failed: {n}");
                }

                for (int i = 0; i < n; ++i)
                {
                    bytes.Add(buf[i]);
                }
            }

            err = AstralNative.astral_session_wait(session, timeout_ms: 5000);
            Assert.AreEqual(AstralNative.ASTRAL_OK, err);

            return Encoding.UTF8.GetString(bytes.ToArray());
        }
    }
}
