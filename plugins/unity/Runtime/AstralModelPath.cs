using System;
using System.IO;
using UnityEngine;

namespace Astral.Runtime
{
    public enum AstralModelPathRoot
    {
        Raw,
        StreamingAssets,
        PersistentData,
        TemporaryCache
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

        public string Resolve()
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentNullException(nameof(path));
            }
            if (Path.IsPathRooted(path) || root == AstralModelPathRoot.Raw)
            {
                return path;
            }

            return Path.GetFullPath(Path.Combine(RootPath(root), path));
        }

        private static string RootPath(AstralModelPathRoot root)
        {
            switch (root)
            {
                case AstralModelPathRoot.StreamingAssets:
                    return Application.streamingAssetsPath;
                case AstralModelPathRoot.PersistentData:
                    return Application.persistentDataPath;
                case AstralModelPathRoot.TemporaryCache:
                    return Application.temporaryCachePath;
                case AstralModelPathRoot.Raw:
                default:
                    return string.Empty;
            }
        }
    }
}
