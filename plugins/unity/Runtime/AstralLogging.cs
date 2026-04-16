using System;
using System.Text;
using System.Runtime.InteropServices;
using AOT; // Required for IL2CPP MonoPInvokeCallback
using UnityEngine;

namespace Astral.Runtime
{
    /// <summary>
    /// Creates the native log callback used by AstralRuntime.Initialize().
    /// </summary>
    internal static class AstralLogging
    {
        private const string Prefix = "[Astral] ";

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void LogFn(IntPtr user, int level, AstralNative.AstralSpanU8 msg);

        private static LogFn s_logDelegate;

        // Maximum log level to forward (0=error .. 4=trace).
        private static int s_maxLogLevel = AstralNative.ASTRAL_LOG_INFO;

        [ThreadStatic]
        private static StringBuilder s_sb;

        private static StringBuilder GetStringBuilder()
        {
            StringBuilder sb = s_sb;
            if (sb == null)
            {
                sb = new StringBuilder(256);
                s_sb = sb;
            }

            sb.Clear();
            return sb;
        }

        private static unsafe void AppendUtf8ToStringBuilder(StringBuilder sb, byte* bytes, int len)
        {
            int i = 0;
            while (i < len)
            {
                uint b0 = bytes[i++];
                if (b0 < 0x80u)
                {
                    sb.Append((char)b0);
                    continue;
                }

                // 2-byte sequence: 110xxxxx 10xxxxxx
                if ((b0 & 0xE0u) == 0xC0u)
                {
                    if (i >= len)
                    {
                        sb.Append('\uFFFD');
                        break;
                    }

                    uint b1 = bytes[i++];
                    if ((b1 & 0xC0u) != 0x80u)
                    {
                        sb.Append('\uFFFD');
                        continue;
                    }

                    uint code = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
                    if (code < 0x80u)
                    {
                        sb.Append('\uFFFD');
                        continue;
                    }

                    sb.Append((char)code);
                    continue;
                }

                // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
                if ((b0 & 0xF0u) == 0xE0u)
                {
                    if (i + 1 >= len)
                    {
                        sb.Append('\uFFFD');
                        break;
                    }

                    uint b1 = bytes[i++];
                    uint b2 = bytes[i++];
                    if (((b1 & 0xC0u) != 0x80u) || ((b2 & 0xC0u) != 0x80u))
                    {
                        sb.Append('\uFFFD');
                        continue;
                    }

                    uint code = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
                    if (code < 0x800u || (code >= 0xD800u && code <= 0xDFFFu))
                    {
                        sb.Append('\uFFFD');
                        continue;
                    }

                    sb.Append((char)code);
                    continue;
                }

                // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                if ((b0 & 0xF8u) == 0xF0u)
                {
                    if (i + 2 >= len)
                    {
                        sb.Append('\uFFFD');
                        break;
                    }

                    uint b1 = bytes[i++];
                    uint b2 = bytes[i++];
                    uint b3 = bytes[i++];
                    if (((b1 & 0xC0u) != 0x80u) || ((b2 & 0xC0u) != 0x80u) || ((b3 & 0xC0u) != 0x80u))
                    {
                        sb.Append('\uFFFD');
                        continue;
                    }

                    uint code = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
                    if (code < 0x10000u || code > 0x10FFFFu)
                    {
                        sb.Append('\uFFFD');
                        continue;
                    }

                    code -= 0x10000u;
                    sb.Append((char)(0xD800u + (code >> 10)));
                    sb.Append((char)(0xDC00u + (code & 0x3FFu)));
                    continue;
                }

                sb.Append('\uFFFD');
            }
        }

        /// <summary>
        /// Get log callback for passing to astral_init().
        /// </summary>
        internal static IntPtr GetLogCallback()
        {
            s_logDelegate = UnityLogCallback;

            return Marshal.GetFunctionPointerForDelegate(s_logDelegate);
        }

        /// <summary>
        /// Set maximum log level to forward (0=error .. 4=trace).
        /// </summary>
        internal static void SetMaxLogLevel(int level)
        {
            s_maxLogLevel = level;
        }

        /// <summary>
        /// Native log callback. Managed exceptions are logged and swallowed at the ABI boundary.
        /// </summary>
        [MonoPInvokeCallback(typeof(LogFn))]
        private static void UnityLogCallback(IntPtr user, int level, AstralNative.AstralSpanU8 msg)
        {
            try
            {
                // Filter by verbosity (0=error .. 4=trace)
                if (level > s_maxLogLevel)
                {
                    return;
                }

                if (msg.data == IntPtr.Zero || msg.len == 0)
                {
                    return;
                }

                unsafe
                {
                    byte* bytes = (byte*)msg.data;

                    // Avoid unbounded stack usage: truncate extremely large log messages.
                    // Note: this is a logging-only path (not a hot path), and truncation is acceptable.
                    const int MaxUtf8Bytes = 2048;
                    uint cappedLen = msg.len > (uint)MaxUtf8Bytes ? (uint)MaxUtf8Bytes : msg.len;
                    int byteCount = (int)cappedLen;

                    StringBuilder sb = GetStringBuilder();
                    sb.Append(Prefix);
                    sb.EnsureCapacity(Prefix.Length + byteCount + 16);

                    // Avoid intermediate managed allocations (no `char[]`, no temporary decoded string).
                    AppendUtf8ToStringBuilder(sb, bytes, byteCount);

                    if (msg.len > cappedLen)
                    {
                        sb.Append('\u2026'); // '…'
                    }

                    string full = sb.ToString();

                    // Forward to Unity logger based on level
                    switch (level)
                    {
                        case AstralNative.ASTRAL_LOG_ERROR:
                            Debug.LogError(full);
                            break;

                        case AstralNative.ASTRAL_LOG_WARN:
                            Debug.LogWarning(full);
                            break;

                        case AstralNative.ASTRAL_LOG_INFO:
                        case AstralNative.ASTRAL_LOG_DEBUG:
                        case AstralNative.ASTRAL_LOG_TRACE:
                            Debug.Log(full);
                            break;

                        default:
                            Debug.Log(full);
                            break;
                    }
                }
            }
            catch (Exception ex)
            {
                // Managed exceptions cannot cross the C ABI callback.
                try
                {
                    Debug.LogError($"[Astral] Logging callback exception: {ex.Message}");
                }
                catch
                {
                    // Unity logging threw while reporting a callback failure.
                }
            }
        }
    }
}
