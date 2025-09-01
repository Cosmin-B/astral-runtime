// AstralEmbedder.cs - Embeddings wrapper with RAII pattern
//
// Always call Dispose() when done (use 'using' statement).

using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    /// <summary>
    /// Embeddings handle wrapper.
    /// </summary>
    public sealed class AstralEmbedder : IDisposable
    {
        private AstralNative.AstralHandle m_handle;
        private AstralModel m_model;
        private uint m_dim;
        private bool m_disposed;

        public bool IsValid => !m_disposed && m_handle.IsValid;
        public uint Dimension => m_dim;

        public static AstralEmbedder Create(AstralModel model)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }
            if (!model.IsValid)
            {
                throw new AstralException("Model is not valid (disposed or not loaded).");
            }

            int err = AstralNative.astral_embed_create(model.Handle, out var handle);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_embed_create failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            uint dim = model.GetEmbeddingDim();

            return new AstralEmbedder
            {
                m_handle = handle,
                m_model = model,
                m_dim = dim,
                m_disposed = false,
            };
        }

        public ulong Enqueue(NativeArray<byte> utf8Text)
        {
            if (!IsValid)
            {
                throw new AstralException("Embedder is not valid (disposed or not created).");
            }

            var span = AstralNative.AstralSpanU8.FromNativeArray(utf8Text);
            int err = AstralNative.astral_embed_enqueue(m_handle, span, out ulong ticket);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_embed_enqueue failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return ticket;
        }

        public void Collect(ulong ticket, NativeArray<float> outVector)
        {
            if (!IsValid)
            {
                throw new AstralException("Embedder is not valid (disposed or not created).");
            }

            if (!outVector.IsCreated)
            {
                throw new ArgumentException("outVector must be created", nameof(outVector));
            }

            if ((ulong)outVector.Length < (ulong)m_dim)
            {
                throw new ArgumentException($"outVector too small (need at least {m_dim} floats)", nameof(outVector));
            }

            unsafe
            {
                var outSpan = new AstralNative.AstralMutSpanU8
                {
                    data = (IntPtr)outVector.GetUnsafePtr(),
                    len = (uint)((ulong)outVector.Length * (ulong)sizeof(float)),
                };

                int err = AstralNative.astral_embed_collect(m_handle, ticket, outSpan);
                if (err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"astral_embed_collect failed: {AstralRuntime.GetErrorString(err)}", err);
                }
            }
        }

        public void Embed(NativeArray<byte> utf8Text, NativeArray<float> outVector)
        {
            ulong ticket = Enqueue(utf8Text);
            Collect(ticket, outVector);
        }

        public void Embed(string text, NativeArray<float> outVector)
        {
            NativeArray<byte> tmp;
            var span = AstralNative.AstralSpanU8.FromString(text, out tmp);
            try
            {
                int err = AstralNative.astral_embed_enqueue(m_handle, span, out ulong ticket);
                if (err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"astral_embed_enqueue failed: {AstralRuntime.GetErrorString(err)}", err);
                }
                Collect(ticket, outVector);
            }
            finally
            {
                if (tmp.IsCreated)
                {
                    tmp.Dispose();
                }
            }
        }

        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_embed_destroy(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_model = null;
            m_dim = 0;
            m_disposed = true;
        }

        ~AstralEmbedder()
        {
            Dispose();
        }
    }
}

