// AstralPromptCache.cs - Native prompt-cache handle with explicit ownership.
//
// The native cache owns token storage and lookup state. This wrapper keeps
// Unity ownership at API boundaries: NativeArray tokens for direct calls and
// managed byte arrays for snapshots.

using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    /// <summary>
    /// Native prompt-cache wrapper. Dispose releases the native cache handle.
    /// </summary>
    public sealed class AstralPromptCache : IDisposable
    {
        internal const uint DefaultMaxEntries = 128;
        internal const uint DefaultMaxTokens = 32768;
        internal const uint DefaultMaxBytes = 0;

        private AstralNative.AstralHandle m_handle;
        private bool m_disposed;

        public bool IsValid => !m_disposed && m_handle.IsValid;
        public AstralNative.AstralHandle Handle => m_handle;

        public static AstralPromptCache Create(AstralPromptCacheConfig config = null)
        {
            config = config ?? AstralPromptCacheConfig.Default;
            var desc = config.ToNativeDesc();
            int err = AstralNative.astral_prompt_cache_create(ref desc, out var handle);
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"astral_prompt_cache_create failed: {AstralRuntime.GetErrorString(err)}", err);
            }

            return new AstralPromptCache
            {
                m_handle = handle,
                m_disposed = false
            };
        }

        public static AstralPromptCache Load(AstralPromptCacheConfig config, byte[] snapshot)
        {
            if (snapshot == null || snapshot.Length == 0)
            {
                throw new ArgumentException("snapshot must not be empty", nameof(snapshot));
            }

            config = config ?? AstralPromptCacheConfig.Default;
            using (var bytes = new NativeArray<byte>(snapshot, Allocator.Temp))
            {
                var desc = config.ToNativeDesc();
                int err = AstralNative.astral_prompt_cache_load(ref desc, AstralNative.AstralSpanU8.FromNativeArray(bytes), out var handle);
                if (err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"astral_prompt_cache_load failed: {AstralRuntime.GetErrorString(err)}", err);
                }

                return new AstralPromptCache
                {
                    m_handle = handle,
                    m_disposed = false
                };
            }
        }

        public void Clear()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_prompt_cache_clear(m_handle);
            ThrowIfError(err, "astral_prompt_cache_clear");
        }

        public AstralNative.AstralPromptCacheStats GetStats()
        {
            ThrowIfInvalid();
            var stats = new AstralNative.AstralPromptCacheStats
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralPromptCacheStats>()
            };
            int err = AstralNative.astral_prompt_cache_stats(m_handle, ref stats);
            ThrowIfError(err, "astral_prompt_cache_stats");
            return stats;
        }

        public byte[] Save()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_prompt_cache_save_size(m_handle, out uint byteCount);
            ThrowIfError(err, "astral_prompt_cache_save_size");
            if (byteCount == 0)
            {
                return Array.Empty<byte>();
            }

            using (var bytes = new NativeArray<byte>((int)byteCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory))
            {
                err = AstralNative.astral_prompt_cache_save(m_handle, AstralNative.AstralMutSpanU8.FromNativeArray(bytes), out uint written);
                ThrowIfError(err, "astral_prompt_cache_save");

                var managed = new byte[written];
                NativeArray<byte>.Copy(bytes, managed, (int)written);
                return managed;
            }
        }

        public unsafe void PutTokens(ref AstralNative.AstralPromptCacheKey key, NativeArray<int> tokens)
        {
            ThrowIfInvalid();
            if (!tokens.IsCreated)
            {
                throw new ArgumentException("tokens must be created", nameof(tokens));
            }

            int err = AstralNative.astral_prompt_cache_put_tokens(
                m_handle,
                ref key,
                (int*)tokens.GetUnsafeReadOnlyPtr(),
                (uint)tokens.Length);
            ThrowIfError(err, "astral_prompt_cache_put_tokens");
        }

        public unsafe uint GetTokens(ref AstralNative.AstralPromptCacheKey key, NativeArray<int> outTokens)
        {
            ThrowIfInvalid();
            if (!outTokens.IsCreated)
            {
                throw new ArgumentException("outTokens must be created", nameof(outTokens));
            }

            int err = AstralNative.astral_prompt_cache_get_tokens(
                m_handle,
                ref key,
                (int*)outTokens.GetUnsafePtr(),
                (uint)outTokens.Length,
                out uint tokenCount);
            ThrowIfError(err, "astral_prompt_cache_get_tokens");
            return tokenCount;
        }

        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_prompt_cache_destroy(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_disposed = true;
        }

        private void ThrowIfInvalid()
        {
            if (!IsValid)
            {
                throw new AstralException("Prompt cache is not valid (disposed or not created).");
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

    /// <summary>
    /// Native prompt-cache creation settings.
    /// </summary>
    [Serializable]
    public class AstralPromptCacheConfig
    {
        public uint maxEntries = AstralPromptCache.DefaultMaxEntries;
        public uint maxTokens = AstralPromptCache.DefaultMaxTokens;
        public uint maxBytes = AstralPromptCache.DefaultMaxBytes;
        public AstralNative.AstralPromptCacheEvictionPolicy evictionPolicy = AstralNative.AstralPromptCacheEvictionPolicy.Fifo;
        public AstralNative.AstralPromptCacheFlags flags = AstralNative.AstralPromptCacheFlags.None;

        public static AstralPromptCacheConfig Default => new AstralPromptCacheConfig();

        internal AstralNative.AstralPromptCacheDesc ToNativeDesc()
        {
            return new AstralNative.AstralPromptCacheDesc
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralPromptCacheDesc>(),
                max_entries = maxEntries,
                max_tokens = maxTokens,
                max_bytes = maxBytes,
                eviction_policy = evictionPolicy,
                flags = flags
            };
        }
    }
}
