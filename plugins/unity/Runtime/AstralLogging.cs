// Astral Unity Logging Bridge
// Forwards Astral runtime logs to Unity Debug.Log
//
// Thread-safety: All callbacks are thread-safe (Unity Debug.Log is thread-safe)
// IL2CPP: Uses MonoPInvokeCallback for IL2CPP compatibility
// Performance: Non-blocking (drops logs if Unity logger is slow)

using System;
using System.Runtime.InteropServices;
using AOT; // Required for IL2CPP MonoPInvokeCallback
using UnityEngine;

namespace Astral.Runtime
{
    /// <summary>
    /// Bridge between Astral runtime logging and Unity Debug.Log.
    ///
    /// 
    /// - Callback must be non-blocking (Astral drops logs if callback is slow >10ms)
    /// - Must never throw exceptions (catch all, log to Unity on best-effort)
    /// - Must be thread-safe (Unity Debug.Log is thread-safe)
    /// </summary>
    internal static class AstralLogging
    {
        // Callback delegate for Astral C ABI
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void LogFn(IntPtr user, int level, AstralNative.AstralSpanU8 msg);

        // Keep delegate alive to prevent GC collection
        private static LogFn s_logDelegate;

        // Minimum log level filter (set by AstralConfig)
        private static int s_minLogLevel = 2; // Info

        /// <summary>
        /// Get log callback for passing to astral_init().
        /// </summary>
        internal static IntPtr GetLogCallback()
        {
            // Initialize delegate (GC root)
            s_logDelegate = UnityLogCallback;

            return Marshal.GetFunctionPointerForDelegate(s_logDelegate);
        }

        /// <summary>
        /// Set minimum log level filter.
        /// </summary>
        internal static void SetMinLogLevel(int level)
        {
            s_minLogLevel = level;
        }

        /// <summary>
        /// Logging callback - forwards to Unity Debug.Log.
        ///
        /// CRITICAL REQUIREMENTS:
        /// - MUST be non-blocking (Astral drops logs if >10ms)
        /// - MUST NOT throw exceptions (catch all)
        /// - MUST be thread-safe (Unity Debug.Log is thread-safe)
        ///
        /// MonoPInvokeCallback required for IL2CPP compatibility.
        /// </summary>
        [MonoPInvokeCallback(typeof(LogFn))]
        private static void UnityLogCallback(IntPtr user, int level, AstralNative.AstralSpanU8 msg)
        {
            try
            {
                // Filter by minimum log level
                if (level > s_minLogLevel)
                {
                    return;
                }

                // Convert UTF-8 span to C# string
                string message = msg.ToUtf8String();

                // Prefix with [Astral]
                string prefixed = $"[Astral] {message}";

                // Forward to Unity logger based on level
                switch (level)
                {
                    case 0: // ASTRAL_LOG_ERROR
                        Debug.LogError(prefixed);
                        break;

                    case 1: // ASTRAL_LOG_WARN
                        Debug.LogWarning(prefixed);
                        break;

                    case 2: // ASTRAL_LOG_INFO
                    case 3: // ASTRAL_LOG_DEBUG
                    case 4: // ASTRAL_LOG_TRACE
                        Debug.Log(prefixed);
                        break;

                    default:
                        Debug.Log($"[Astral] [UNKNOWN LEVEL {level}] {message}");
                        break;
                }
            }
            catch (Exception ex)
            {
                //  Never throw across C ABI boundary
                // Log to Unity on best-effort basis
                try
                {
                    Debug.LogError($"[Astral] Logging callback exception: {ex.Message}");
                }
                catch
                {
                    // Even Unity logging failed, give up
                }
            }
        }
    }
}
