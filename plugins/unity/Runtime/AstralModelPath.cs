using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;

namespace Astral.Runtime
{
    public enum AstralModelPathRoot
    {
        Raw,
        StreamingAssets,
        PersistentData,
        TemporaryCache,
        Download
    }

    [Serializable]
    public readonly struct AstralModelPath
    {
        public readonly AstralModelPathRoot root;
        public readonly string path;

        public AstralModelPath(AstralModelPathRoot root, string path)
        {
            this.root = root;
            this.path = path;
        }

        public static AstralModelPath Raw(string path)
        {
            return new AstralModelPath(AstralModelPathRoot.Raw, path);
        }

        public static AstralModelPath StreamingAssets(string path)
        {
            return new AstralModelPath(AstralModelPathRoot.StreamingAssets, path);
        }

        public static AstralModelPath PersistentData(string path)
        {
            return new AstralModelPath(AstralModelPathRoot.PersistentData, path);
        }

        public static AstralModelPath TemporaryCache(string path)
        {
            return new AstralModelPath(AstralModelPathRoot.TemporaryCache, path);
        }

        public static AstralModelPath Download(string path)
        {
            return new AstralModelPath(AstralModelPathRoot.Download, path);
        }

        public string Resolve()
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentNullException(nameof(path));
            }

            NativeArray<byte> pathBytes = default;
            NativeArray<byte> contentRootBytes = default;
            NativeArray<byte> savedRootBytes = default;
            NativeArray<byte> cacheRootBytes = default;
            NativeArray<byte> downloadRootBytes = default;

            try
            {
                AstralNative.AstralModelPathResolveDesc desc = new AstralNative.AstralModelPathResolveDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralModelPathResolveDesc>(),
                    root = ToNativeRoot(root),
                    path = AstralNative.AstralSpanU8.FromString(path, out pathBytes),
                    content_root = AstralNative.AstralSpanU8.FromString(Application.streamingAssetsPath, out contentRootBytes),
                    saved_root = AstralNative.AstralSpanU8.FromString(Application.persistentDataPath, out savedRootBytes),
                    cache_root = AstralNative.AstralSpanU8.FromString(Application.temporaryCachePath, out cacheRootBytes),
                    download_root = AstralNative.AstralSpanU8.FromString(Application.persistentDataPath, out downloadRootBytes),
                    flags = AstralNative.AstralModelPathResolveFlags.None
                };

                uint requiredBytes = 0;
                int err = AstralNative.astral_model_path_resolve(
                    ref desc,
                    new AstralNative.AstralMutSpanU8 { data = IntPtr.Zero, len = 0 },
                    out requiredBytes);
                if (err != AstralNative.ASTRAL_E_NOMEM && err != AstralNative.ASTRAL_OK)
                {
                    throw new AstralException($"Failed to resolve model path '{path}': {AstralRuntime.GetErrorString(err)}", err);
                }
                if (requiredBytes > MaxManagedPathBytes)
                {
                    throw new AstralException("Resolved model path is too large for a managed string.", AstralNative.ASTRAL_E_NOMEM);
                }

                int outputBytes = (int)requiredBytes;
                byte[] output = new byte[outputBytes];
                GCHandle handle = GCHandle.Alloc(output, GCHandleType.Pinned);
                try
                {
                    AstralNative.AstralMutSpanU8 outSpan = new AstralNative.AstralMutSpanU8
                    {
                        data = handle.AddrOfPinnedObject(),
                        len = requiredBytes
                    };
                    err = AstralNative.astral_model_path_resolve(ref desc, outSpan, out requiredBytes);
                    if (err != AstralNative.ASTRAL_OK)
                    {
                        throw new AstralException($"Failed to resolve model path '{path}': {AstralRuntime.GetErrorString(err)}", err);
                    }
                }
                finally
                {
                    handle.Free();
                }

                return System.Text.Encoding.UTF8.GetString(output);
            }
            finally
            {
                DisposeIfCreated(pathBytes);
                DisposeIfCreated(contentRootBytes);
                DisposeIfCreated(savedRootBytes);
                DisposeIfCreated(cacheRootBytes);
                DisposeIfCreated(downloadRootBytes);
            }
        }

        private const uint MaxManagedPathBytes = int.MaxValue;

        private static AstralNative.AstralModelPathRoot ToNativeRoot(AstralModelPathRoot root)
        {
            switch (root)
            {
                case AstralModelPathRoot.StreamingAssets:
                    return AstralNative.AstralModelPathRoot.Content;
                case AstralModelPathRoot.PersistentData:
                    return AstralNative.AstralModelPathRoot.Saved;
                case AstralModelPathRoot.TemporaryCache:
                    return AstralNative.AstralModelPathRoot.Cache;
                case AstralModelPathRoot.Download:
                    return AstralNative.AstralModelPathRoot.Download;
                case AstralModelPathRoot.Raw:
                default:
                    return AstralNative.AstralModelPathRoot.Raw;
            }
        }

        private static void DisposeIfCreated(NativeArray<byte> bytes)
        {
            if (bytes.IsCreated)
            {
                bytes.Dispose();
            }
        }
    }
}
