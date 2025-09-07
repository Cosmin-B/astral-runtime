using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NUnit.Framework;
using Unity.Collections;

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
                Assert.Ignore($"Astral native library not found: {e.Message}");
            }
            catch (EntryPointNotFoundException e)
            {
                Assert.Ignore($"Astral native entry point not found: {e.Message}");
            }
            catch (BadImageFormatException e)
            {
                Assert.Ignore($"Astral native library is not loadable for this platform: {e.Message}");
            }
        }

        [Test]
        public void Abi_StructSizes_AreExpected()
        {
            int expectedSpan = IntPtr.Size == 8 ? 16 : 8;
            Assert.AreEqual(expectedSpan, Marshal.SizeOf<AstralNative.AstralSpanU8>());
            Assert.AreEqual(expectedSpan, Marshal.SizeOf<AstralNative.AstralMutSpanU8>());

            Assert.AreEqual(8, Marshal.SizeOf<AstralNative.AstralHandle>());
            Assert.AreEqual(8, Marshal.SizeOf<ulong>()); // AstralCaps

            int expectedInit = IntPtr.Size == 8 ? 64 : 48;
            Assert.AreEqual(expectedInit, Marshal.SizeOf<AstralNative.AstralInit>());

            int expectedModelDesc = IntPtr.Size == 8 ? 56 : 36;
            Assert.AreEqual(expectedModelDesc, Marshal.SizeOf<AstralNative.AstralModelDesc>());

            Assert.AreEqual(32, Marshal.SizeOf<AstralNative.AstralSessionDesc>());
            Assert.AreEqual(16, Marshal.SizeOf<AstralNative.AstralModelLimits>());
            Assert.AreEqual(56, Marshal.SizeOf<AstralNative.AstralSamplerDesc>());
            Assert.AreEqual(40, Marshal.SizeOf<AstralNative.AstralStats>());
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
