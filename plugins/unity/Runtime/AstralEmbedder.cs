// AstralEmbedder.cs - Embeddings handle with explicit native ownership.
//
// Dispose releases the native embedder handle; prefer a using scope.

using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    /// <summary>
    /// Embeddings handle wrapper with explicit native ownership.
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

        public ulong EnqueueImage(ref AstralNative.AstralImageDesc image)
        {
            if (!IsValid)
            {
                throw new AstralException("Embedder is not valid (disposed or not created).");
            }

            image.size = (uint)Marshal.SizeOf<AstralNative.AstralImageDesc>();
            int err = AstralNative.astral_embed_enqueue_image(m_handle, ref image, out ulong ticket);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_embed_enqueue_image failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return ticket;
        }

        public ulong EnqueueImage(
            NativeArray<byte> pixels,
            uint width,
            uint height,
            AstralNative.AstralImageFormat format,
            uint rowStride = 0,
            uint flags = 0)
        {
            if (!pixels.IsCreated)
            {
                throw new ArgumentException("pixels must be created", nameof(pixels));
            }
            if (pixels.Length == 0)
            {
                throw new ArgumentException("pixels must not be empty", nameof(pixels));
            }

            var desc = new AstralNative.AstralImageDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralImageDesc>(),
                format = format,
                width = width,
                height = height,
                row_stride = rowStride,
                flags = flags,
                pixels = AstralNative.AstralSpanU8.FromNativeArray(pixels)
            };

            return EnqueueImage(ref desc);
        }

        public ulong EnqueueAudio(ref AstralNative.AstralAudioDesc audio)
        {
            if (!IsValid)
            {
                throw new AstralException("Embedder is not valid (disposed or not created).");
            }

            audio.size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>();
            int err = AstralNative.astral_embed_enqueue_audio(m_handle, ref audio, out ulong ticket);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_embed_enqueue_audio failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return ticket;
        }

        public ulong EnqueueAudio(
            NativeArray<byte> samples,
            uint channels,
            uint sampleRate,
            AstralNative.AstralAudioFormat format,
            ulong frameCount = 0,
            uint flags = 0)
        {
            if (!samples.IsCreated)
            {
                throw new ArgumentException("samples must be created", nameof(samples));
            }
            if (samples.Length == 0)
            {
                throw new ArgumentException("samples must not be empty", nameof(samples));
            }
            if (channels == 0)
            {
                throw new ArgumentException("channels must be > 0", nameof(channels));
            }

            if (frameCount == 0)
            {
                uint bytesPerSample = format == AstralNative.AstralAudioFormat.F32 ? 4u : 2u;
                ulong totalSamples = (ulong)samples.Length / bytesPerSample;
                if (totalSamples % channels != 0)
                {
                    throw new ArgumentException("samples length is not aligned to channels", nameof(samples));
                }
                frameCount = totalSamples / channels;
            }

            var desc = new AstralNative.AstralAudioDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>(),
                format = format,
                channels = channels,
                sample_rate = sampleRate,
                frame_count = frameCount,
                samples = AstralNative.AstralSpanU8.FromNativeArray(samples),
                flags = flags
            };

            return EnqueueAudio(ref desc);
        }

        public ulong EnqueueAudio(
            NativeArray<float> samples,
            uint channels,
            uint sampleRate,
            ulong frameCount = 0,
            uint flags = 0)
        {
            if (!samples.IsCreated)
            {
                throw new ArgumentException("samples must be created", nameof(samples));
            }
            if (samples.Length == 0)
            {
                throw new ArgumentException("samples must not be empty", nameof(samples));
            }
            if (channels == 0)
            {
                throw new ArgumentException("channels must be > 0", nameof(channels));
            }

            if (samples.Length % (int)channels != 0)
            {
                throw new ArgumentException("samples length is not aligned to channels", nameof(samples));
            }

            ulong computedFrames = (ulong)samples.Length / channels;
            if (frameCount == 0)
            {
                frameCount = computedFrames;
            }

            unsafe
            {
                var span = new AstralNative.AstralSpanU8
                {
                    data = (IntPtr)samples.GetUnsafeReadOnlyPtr(),
                    len = (uint)((ulong)samples.Length * (ulong)sizeof(float))
                };

                var desc = new AstralNative.AstralAudioDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>(),
                    format = AstralNative.AstralAudioFormat.F32,
                    channels = channels,
                    sample_rate = sampleRate,
                    frame_count = frameCount,
                    samples = span,
                    flags = flags
                };

                return EnqueueAudio(ref desc);
            }
        }

        public ulong EnqueueAudio(
            NativeArray<short> samples,
            uint channels,
            uint sampleRate,
            ulong frameCount = 0,
            uint flags = 0)
        {
            if (!samples.IsCreated)
            {
                throw new ArgumentException("samples must be created", nameof(samples));
            }
            if (samples.Length == 0)
            {
                throw new ArgumentException("samples must not be empty", nameof(samples));
            }
            if (channels == 0)
            {
                throw new ArgumentException("channels must be > 0", nameof(channels));
            }

            if (samples.Length % (int)channels != 0)
            {
                throw new ArgumentException("samples length is not aligned to channels", nameof(samples));
            }

            ulong computedFrames = (ulong)samples.Length / channels;
            if (frameCount == 0)
            {
                frameCount = computedFrames;
            }

            unsafe
            {
                var span = new AstralNative.AstralSpanU8
                {
                    data = (IntPtr)samples.GetUnsafeReadOnlyPtr(),
                    len = (uint)((ulong)samples.Length * (ulong)sizeof(short))
                };

                var desc = new AstralNative.AstralAudioDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>(),
                    format = AstralNative.AstralAudioFormat.I16,
                    channels = channels,
                    sample_rate = sampleRate,
                    frame_count = frameCount,
                    samples = span,
                    flags = flags
                };

                return EnqueueAudio(ref desc);
            }
        }

        public unsafe ulong EnqueueMultimodal(string text, ref AstralNative.AstralImageDesc image)
        {
            image.size = (uint)Marshal.SizeOf<AstralNative.AstralImageDesc>();
            fixed (AstralNative.AstralImageDesc* pImage = &image)
            {
                return EnqueueMultimodalInternal(text, (IntPtr)pImage, IntPtr.Zero);
            }
        }

        public unsafe ulong EnqueueMultimodal(string text, ref AstralNative.AstralAudioDesc audio)
        {
            audio.size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>();
            fixed (AstralNative.AstralAudioDesc* pAudio = &audio)
            {
                return EnqueueMultimodalInternal(text, IntPtr.Zero, (IntPtr)pAudio);
            }
        }

        public unsafe ulong EnqueueMultimodal(string text, ref AstralNative.AstralImageDesc image, ref AstralNative.AstralAudioDesc audio)
        {
            image.size = (uint)Marshal.SizeOf<AstralNative.AstralImageDesc>();
            audio.size = (uint)Marshal.SizeOf<AstralNative.AstralAudioDesc>();
            fixed (AstralNative.AstralImageDesc* pImage = &image)
            fixed (AstralNative.AstralAudioDesc* pAudio = &audio)
            {
                return EnqueueMultimodalInternal(text, (IntPtr)pImage, (IntPtr)pAudio);
            }
        }

        private ulong EnqueueMultimodalInternal(string text, IntPtr imagePtr, IntPtr audioPtr)
        {
            if (!IsValid)
            {
                throw new AstralException("Embedder is not valid (disposed or not created).");
            }

            NativeArray<byte> tmp;
            var span = AstralNative.AstralSpanU8.FromString(text ?? string.Empty, out tmp);
            try
            {
                int err = AstralNative.astral_embed_enqueue_multimodal(m_handle, span, imagePtr, audioPtr, out ulong ticket);
                if (err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"astral_embed_enqueue_multimodal failed: {AstralRuntime.GetErrorString(err)}", err);
                }

                return ticket;
            }
            finally
            {
                if (tmp.IsCreated)
                {
                    tmp.Dispose();
                }
            }
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

        public void Cancel(ulong ticket)
        {
            if (!IsValid)
            {
                throw new AstralException("Embedder is not valid (disposed or not created).");
            }

            int err = AstralNative.astral_embed_cancel(m_handle, ticket);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_embed_cancel failed: {AstralRuntime.GetErrorString(err)}", err);
            }
        }

        public void Embed(NativeArray<byte> utf8Text, NativeArray<float> outVector)
        {
            ulong ticket = Enqueue(utf8Text);
            Collect(ticket, outVector);
        }

        public void EmbedImage(ref AstralNative.AstralImageDesc image, NativeArray<float> outVector)
        {
            ulong ticket = EnqueueImage(ref image);
            Collect(ticket, outVector);
        }

        public void EmbedAudio(ref AstralNative.AstralAudioDesc audio, NativeArray<float> outVector)
        {
            ulong ticket = EnqueueAudio(ref audio);
            Collect(ticket, outVector);
        }

        public void EmbedMultimodal(string text, ref AstralNative.AstralImageDesc image, NativeArray<float> outVector)
        {
            ulong ticket = EnqueueMultimodal(text, ref image);
            Collect(ticket, outVector);
        }

        public void EmbedMultimodal(string text, ref AstralNative.AstralAudioDesc audio, NativeArray<float> outVector)
        {
            ulong ticket = EnqueueMultimodal(text, ref audio);
            Collect(ticket, outVector);
        }

        public void EmbedMultimodal(string text, ref AstralNative.AstralImageDesc image, ref AstralNative.AstralAudioDesc audio, NativeArray<float> outVector)
        {
            ulong ticket = EnqueueMultimodal(text, ref image, ref audio);
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
