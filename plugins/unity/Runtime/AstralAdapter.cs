// AstralAdapter.cs - Unity owner for native model-scoped LoRA adapter handles.

using System;
using System.Runtime.InteropServices;
using System.Text;
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

        public AstralNative.AstralAdapterInfo GetInfo()
        {
            ThrowIfInvalid();
            var info = new AstralNative.AstralAdapterInfo
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralAdapterInfo>()
            };
            int err = AstralNative.astral_model_adapter_info(m_handle, ref info);
            ThrowIfError(err, "astral_model_adapter_info");
            return info;
        }

        public string GetPath()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_model_adapter_path_copy(
                m_handle,
                new AstralNative.AstralMutSpanU8 { data = IntPtr.Zero, len = 0 },
                out uint byteCount);
            ThrowIfError(err, "astral_model_adapter_path_copy");
            if (byteCount == 0)
            {
                return string.Empty;
            }

            using (var bytes = new NativeArray<byte>(
                (int)byteCount,
                Allocator.Temp,
                NativeArrayOptions.UninitializedMemory))
            {
                err = AstralNative.astral_model_adapter_path_copy(
                    m_handle,
                    AstralNative.AstralMutSpanU8.FromNativeArray(bytes),
                    out uint written);
                ThrowIfError(err, "astral_model_adapter_path_copy");

                var managed = new byte[written];
                NativeArray<byte>.Copy(bytes, managed, (int)written);
                return Encoding.UTF8.GetString(managed);
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

        private void ThrowIfInvalid()
        {
            if (!IsValid)
            {
                throw new AstralException("Adapter is not valid (disposed or not loaded).");
            }
        }

        private static void ThrowIfError(int err, string call)
        {
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"{call} failed: {AstralRuntime.GetErrorString(err)}", err);
            }
        }
    }

    public struct AstralSessionAdapterInfo
    {
        public AstralNative.AstralHandle adapter;
        public float scale;
    }
}
