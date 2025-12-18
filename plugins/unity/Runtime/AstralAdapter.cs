// AstralAdapter.cs - Unity owner for native model-scoped LoRA adapter handles.

using System;
using System.Runtime.InteropServices;
using Unity.Collections;

namespace Astral.Runtime
{
    /// <summary>
    /// Model-scoped LoRA adapter handle.
    /// </summary>
    public sealed class AstralAdapter : IDisposable
    {
        private AstralNative.AstralHandle m_handle;
        private bool m_disposed;

        public bool IsValid => !m_disposed && m_handle.IsValid;
        internal AstralNative.AstralHandle Handle => m_handle;

        public static AstralAdapter Load(AstralModel model, string adapterPath)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }
            if (!model.IsValid)
            {
                throw new ArgumentException("model must be valid", nameof(model));
            }
            if (string.IsNullOrEmpty(adapterPath))
            {
                throw new ArgumentNullException(nameof(adapterPath));
            }

            NativeArray<byte> pathArray;
            var pathSpan = AstralNative.AstralSpanU8.FromString(adapterPath, out pathArray);
            try
            {
                var desc = new AstralNative.AstralAdapterDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralAdapterDesc>(),
                    path = pathSpan
                };

                int err = AstralNative.astral_model_adapter_load(model.Handle, ref desc, out var handle);
                if (err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"astral_model_adapter_load failed: {AstralRuntime.GetErrorString(err)}", err);
                }

                return new AstralAdapter
                {
                    m_handle = handle,
                    m_disposed = false
                };
            }
            finally
            {
                if (pathArray.IsCreated)
                {
                    pathArray.Dispose();
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
                AstralNative.astral_model_adapter_release(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_disposed = true;
        }
    }

    public struct AstralSessionAdapterInfo
    {
        public AstralNative.AstralHandle adapter;
        public float scale;
    }
}
