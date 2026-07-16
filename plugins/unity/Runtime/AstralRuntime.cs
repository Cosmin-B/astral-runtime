// AstralRuntime.cs - High-level wrapper for Astral runtime initialization
//
//  Call AstralRuntime.Initialize() before any Astral operations
//  Call AstralRuntime.Shutdown() on application quit
//  No GC allocations in hot paths

using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;

namespace Astral.Runtime
{
    /// <summary>
    /// High-level wrapper for Astral runtime.
    /// Manages initialization and shutdown.
    /// Thread-safety: Initialize/Shutdown must be called from main thread only.
    /// </summary>
    public static class AstralRuntime
    {
        private static bool s_initialized = false;
        private static AstralConfig s_config;

        /// <summary>
        /// Initialize Astral runtime.
        /// Must be called before any other Astral operations.
        /// Thread-safety: Not thread-safe; call from main thread only.
        /// </summary>
        /// <param name="config">Runtime configuration (null = default config)</param>
        /// <exception cref="AstralException">Thrown if initialization fails</exception>
        public static void Initialize(AstralConfig config = null)
        {
            if (!TryInitialize(config, out int err))
            {
                throw new AstralException($"Failed to initialize Astral runtime: {GetErrorString(err)}", err);
            }
        }

        /// <summary>
        /// Initialize Astral runtime without throwing.
        /// Thread-safety: Not thread-safe; call from main thread only.
        /// </summary>
        /// <param name="config">Runtime configuration (null = default config)</param>
        /// <param name="err">Output: error code (ASTRAL_OK on success)</param>
        /// <returns>true on success, false on failure</returns>
        public static bool TryInitialize(AstralConfig config, out int err)
        {
            if (s_initialized)
            {
                Debug.LogWarning("[Astral] Runtime already initialized");
                err = AstralNative.ASTRAL_OK;
                return true;
            }

            s_config = config ?? AstralConfig.Default;

            if (s_config.enableLogging)
            {
                AstralLogging.SetMaxLogLevel(s_config.maxLogLevel);
            }

            var init = new AstralNative.AstralInit
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralInit>(),
                sys_alloc = s_config.useUnityAllocator ? AstralAllocatorBridge.CreateUnityAllocator() : default,
                log_cb = s_config.enableLogging ? AstralLogging.GetLogCallback() : IntPtr.Zero,
                log_user = IntPtr.Zero,
                reserve_bytes = s_config.reserveBytes,
                thread_count = s_config.threadCount,
                numa_node = 0xFFFFFFFF, // Any NUMA node
                enable_hugepages = (byte)(s_config.enableHugePages ? 1 : 0),
                memory_mode = AstralNative.AstralMemoryMode.VirtualMemory
            };

            err = AstralNative.astral_init(ref init);
            if (err != AstralNative.ASTRAL_OK)
            {
                return false;
            }

            s_initialized = true;
            Debug.Log($"[Astral] Runtime initialized (reserve={s_config.reserveBytes / (1024 * 1024)}MB, threads={s_config.threadCount})");
            return true;
        }

        /// <summary>
        /// Shutdown Astral runtime.
        /// Must be called on application quit.
        /// All models and sessions must be released before calling this.
        /// Thread-safety: Not thread-safe; call from main thread only.
        /// </summary>
        public static void Shutdown()
        {
            if (!s_initialized)
            {
                return;
            }

            AstralNative.astral_shutdown();
            s_initialized = false;

            if (s_config != null && s_config.useUnityAllocator)
            {
                AstralAllocatorBridge.ValidateNoLeaks();
            }
            Debug.Log("[Astral] Runtime shutdown");
        }

        /// <summary>
        /// Get human-readable error string from error code.
        /// Thread-safety: Safe to call from any thread.
        /// </summary>
        public static string GetErrorString(int err)
        {
            IntPtr ptr = AstralNative.astral_error_string(err);
            return Marshal.PtrToStringAnsi(ptr) ?? $"Unknown error ({err})";
        }

        /// <summary>
        /// Check if runtime is initialized.
        /// Thread-safety: Safe to call from any thread.
        /// </summary>
        public static bool IsInitialized => s_initialized;

        /// <summary>
        /// Get current runtime configuration.
        /// </summary>
        public static AstralConfig Config => s_config;
    }

    /// <summary>
    /// Configuration for Astral runtime.
    /// Immutable after initialization.
    /// </summary>
    [Serializable]
    public class AstralConfig
    {
        /// <summary>
        /// Virtual memory to reserve (default: 2GB).
        ///  Must be large enough for all models + KV caches.
        /// Mobile: 512MB-1GB; Desktop: 2GB-4GB
        /// </summary>
        public ulong reserveBytes = 2UL << 30; // 2GB

        /// <summary>
        /// Worker thread count (0 = auto-detect).
        ///  Too many threads causes contention; too few underutilizes CPU.
        /// Recommended: Physical cores - 1 (leave one for main thread)
        /// </summary>
        public uint threadCount = 0; // Auto-detect

        /// <summary>
        /// Try to use huge pages (2MB/1GB) for better TLB performance.
        /// Requires OS support and elevated privileges on some platforms.
        /// Linux: echo 'vm.nr_hugepages=1024' >> /etc/sysctl.conf
        /// Windows: SeLockMemoryPrivilege required
        /// </summary>
        public bool enableHugePages = false;

        /// <summary>
        /// Use Unity's native allocator (Allocator.Persistent) for Astral heap allocations.
        /// This improves profiler visibility and routes allocations through Unity's allocator hooks.
        /// </summary>
        public bool useUnityAllocator = true;

        /// <summary>
        /// Forward Astral native logs to Unity's logger.
        /// </summary>
        public bool enableLogging = true;

        /// <summary>
        /// Maximum log verbosity forwarded to Unity (0=error .. 4=trace).
        /// Default: info.
        /// </summary>
        public int maxLogLevel = AstralNative.ASTRAL_LOG_INFO;

        /// <summary>
        /// Default configuration for desktop platforms.
        /// </summary>
        public static AstralConfig Default => new AstralConfig();

        /// <summary>
        /// Mobile-optimized configuration (smaller memory footprint).
        /// </summary>
        public static AstralConfig Mobile => new AstralConfig
        {
            reserveBytes = 512UL << 20, // 512MB
            threadCount = 2,             // Conservative for mobile
            enableHugePages = false,     // Not available on mobile
            useUnityAllocator = true,
            enableLogging = true,
            maxLogLevel = AstralNative.ASTRAL_LOG_WARN
        };

        /// <summary>
        /// Desktop preset with a larger native reservation and huge-page attempts enabled.
        /// </summary>
        public static AstralConfig HighPerformance => new AstralConfig
        {
            reserveBytes = 4UL << 30,    // 4GB
            threadCount = 0,             // Auto-detect (usually physical cores - 1)
            enableHugePages = true,      // Request large pages when the OS grants them
            useUnityAllocator = true,
            enableLogging = true,
            maxLogLevel = AstralNative.ASTRAL_LOG_INFO
        };
    }

    /// <summary>
    /// Astral exception.
    /// Thrown when native Astral operations fail.
    /// </summary>
    public class AstralException : Exception
    {
        public int ErrorCode { get; }

        public AstralException(string message) : base(message)
        {
            ErrorCode = AstralNative.ASTRAL_E_INVALID;
        }

        public AstralException(string message, int errorCode) : base(message)
        {
            ErrorCode = errorCode;
        }
    }

    /// <summary>
    /// Runtime initializer - automatically initializes/shuts down Astral runtime.
    /// Add this component to a persistent GameObject in your scene.
    /// </summary>
    [DefaultExecutionOrder(-1000)] // Initialize early
    public class AstralRuntimeInitializer : MonoBehaviour
    {
        [SerializeField]
        private bool autoInitialize = true;

        [SerializeField]
        private bool useMobileConfig = false;

        [SerializeField]
        private bool useHighPerformanceConfig = false;

        private void Awake()
        {
            if (autoInitialize && !AstralRuntime.IsInitialized)
            {
                AstralConfig config;
                if (useMobileConfig)
                {
                    config = AstralConfig.Mobile;
                }
                else if (useHighPerformanceConfig)
                {
                    config = AstralConfig.HighPerformance;
                }
                else
                {
                    config = AstralConfig.Default;
                }

                AstralRuntime.Initialize(config);
            }
        }

        private void OnApplicationQuit()
        {
            if (AstralRuntime.IsInitialized)
            {
                AstralRuntime.Shutdown();
            }
        }

        private void OnDestroy()
        {
            if (AstralRuntime.IsInitialized && Application.isPlaying)
            {
                AstralRuntime.Shutdown();
            }
        }
    }
}
