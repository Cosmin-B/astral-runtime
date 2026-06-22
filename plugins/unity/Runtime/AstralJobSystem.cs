using System;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace Astral.Runtime
{
    /// <summary>
    /// Polls an Astral stream into caller-owned NativeArray storage.
    /// </summary>
    public struct AstralStreamReadJob : IJob
    {
        /// <summary>Session handle (opaque pointer from astral_session_create)</summary>
        public AstralNative.AstralHandle session;

        /// <summary>Output buffer for token bytes (UTF-8)</summary>
        public NativeArray<byte> outputBuffer;

        /// <summary>Timeout in milliseconds (0 = non-blocking)</summary>
        public uint timeout_ms;

        /// <summary>Output: number of bytes read (or error code if negative)</summary>
        public NativeArray<int> bytesRead;

        public void Execute()
        {
            var span = outputBuffer.AsMutSpan();

            int result = AstralNative.astral_stream_read(session, span, timeout_ms);

            bytesRead[0] = result;
        }
    }

    /// <summary>
    /// Feeds a UTF-8 prompt chunk from caller-owned NativeArray storage.
    /// </summary>
    public struct AstralSessionFeedJob : IJob
    {
        /// <summary>Session handle</summary>
        public AstralNative.AstralHandle session;

        /// <summary>Prompt chunk (UTF-8 bytes)</summary>
        [ReadOnly] public NativeArray<byte> promptChunk;

        /// <summary>Finalize prompt (1 = last chunk, 0 = more chunks coming)</summary>
        public byte finalize;

        /// <summary>Output: error code (ASTRAL_OK = 0 on success)</summary>
        public NativeArray<int> errorCode;

        public void Execute()
        {
            var span = AstralNative.AstralSpanU8.FromNativeArray(promptChunk);

            int result = (int)AstralNative.astral_session_feed(session, span, finalize);

            errorCode[0] = result;
        }
    }

    /// <summary>
    /// Job for triggering decode on Astral session.
    ///
    /// Use case: Kick off inference on background thread, main thread polls results.
    /// </summary>
    public struct AstralSessionDecodeJob : IJob
    {
        /// <summary>Session handle</summary>
        public AstralNative.AstralHandle session;

        /// <summary>Output: error code (ASTRAL_OK = 0 on success)</summary>
        public NativeArray<int> errorCode;

        public void Execute()
        {
            // Trigger decode (non-blocking, queues work)
            int result = (int)AstralNative.astral_session_decode(session);

            // Store error code
            errorCode[0] = result;
        }
    }

    /// <summary>
    /// Helper for scheduling Astral operations on Unity Jobs System.
    ///
    /// DESIGN PATTERNS:
    /// - All methods return JobHandle for dependency chaining
    /// - All NativeArrays must be disposed by caller after job completes
    /// - Job data stays blittable for Unity job scheduling.
    ///
    /// USAGE PATTERN:
    ///   // Schedule decode job
    ///   var errorCode = new NativeArray{int}(1, Allocator.TempJob);
    ///   var decodeHandle = AstralJobs.ScheduleDecode(session, errorCode);
    ///
    ///   // Schedule stream read job (depends on decode)
    ///   var buffer = new NativeArray{byte}(1024, Allocator.TempJob);
    ///   var bytesRead = new NativeArray{int}(1, Allocator.TempJob);
    ///   var readHandle = AstralJobs.ScheduleStreamRead(session, buffer, bytesRead, 1000, decodeHandle);
    ///
    ///   // Wait for completion
    ///   readHandle.Complete();
    ///
    ///   // Check results
    ///   if (bytesRead[0] > 0)
    ///   {
    ///       var slice = new NativeSlice{byte}(buffer, 0, bytesRead[0]);
    ///       string response = slice.ToUtf8String();
    ///       Debug.Log(response);
    ///   }
    ///
    ///   // Cleanup
    ///   buffer.Dispose();
    ///   bytesRead.Dispose();
    ///   errorCode.Dispose();
    /// </summary>
    public static class AstralJobs
    {
        /// <summary>Schedules a stream read job.</summary>
        public static JobHandle ScheduleStreamRead(
            AstralNative.AstralHandle session,
            NativeArray<byte> outputBuffer,
            NativeArray<int> bytesRead,
            uint timeout_ms = 1000,
            JobHandle dependency = default)
        {
            // Validate inputs
            if (!session.IsValid)
            {
                Debug.LogError("[Astral] ScheduleStreamRead: invalid session handle");
                bytesRead[0] = AstralNative.ASTRAL_E_INVALID;
                return dependency;
            }

            if (!outputBuffer.IsCreated || !bytesRead.IsCreated)
            {
                Debug.LogError("[Astral] ScheduleStreamRead: outputBuffer or bytesRead not created");
                return dependency;
            }

            if (bytesRead.Length < 1)
            {
                Debug.LogError("[Astral] ScheduleStreamRead: bytesRead must have length >= 1");
                return dependency;
            }

            var job = new AstralStreamReadJob
            {
                session = session,
                outputBuffer = outputBuffer,
                timeout_ms = timeout_ms,
                bytesRead = bytesRead
            };

            return job.Schedule(dependency);
        }

        /// <summary>Schedules a prompt feed job.</summary>
        public static JobHandle ScheduleFeed(
            AstralNative.AstralHandle session,
            NativeArray<byte> promptChunk,
            byte finalize,
            NativeArray<int> errorCode,
            JobHandle dependency = default)
        {
            // Validate inputs
            if (!session.IsValid)
            {
                Debug.LogError("[Astral] ScheduleFeed: invalid session handle");
                errorCode[0] = AstralNative.ASTRAL_E_INVALID;
                return dependency;
            }

            if (!promptChunk.IsCreated || !errorCode.IsCreated)
            {
                Debug.LogError("[Astral] ScheduleFeed: promptChunk or errorCode not created");
                return dependency;
            }

            if (errorCode.Length < 1)
            {
                Debug.LogError("[Astral] ScheduleFeed: errorCode must have length >= 1");
                return dependency;
            }

            var job = new AstralSessionFeedJob
            {
                session = session,
                promptChunk = promptChunk,
                finalize = finalize,
                errorCode = errorCode
            };

            return job.Schedule(dependency);
        }

        /// <summary>Schedules a decode request job.</summary>
        public static JobHandle ScheduleDecode(
            AstralNative.AstralHandle session,
            NativeArray<int> errorCode,
            JobHandle dependency = default)
        {
            // Validate inputs
            if (!session.IsValid)
            {
                Debug.LogError("[Astral] ScheduleDecode: invalid session handle");
                errorCode[0] = AstralNative.ASTRAL_E_INVALID;
                return dependency;
            }

            if (!errorCode.IsCreated)
            {
                Debug.LogError("[Astral] ScheduleDecode: errorCode not created");
                return dependency;
            }

            if (errorCode.Length < 1)
            {
                Debug.LogError("[Astral] ScheduleDecode: errorCode must have length >= 1");
                return dependency;
            }

            var job = new AstralSessionDecodeJob
            {
                session = session,
                errorCode = errorCode
            };

            return job.Schedule(dependency);
        }

        /// <summary>
        /// Schedules feed, decode, and one stream read as a dependent job chain.
        /// The caller owns all NativeArray inputs and outputs.
        /// </summary>
        public static JobHandle ScheduleInference(
            AstralNative.AstralHandle session,
            NativeArray<byte> prompt,
            NativeArray<byte> outputBuffer,
            NativeArray<int> bytesRead,
            uint timeout_ms = 5000)
        {
            // Job-owned scratch slots carry native error codes across the chain.
            var feedErrorCode = new NativeArray<int>(1, Allocator.TempJob);
            var decodeErrorCode = new NativeArray<int>(1, Allocator.TempJob);

            // Feed, decode, and stream read share one dependency chain.
            var feedHandle = ScheduleFeed(session, prompt, finalize: 1, feedErrorCode);
            var decodeHandle = ScheduleDecode(session, decodeErrorCode, feedHandle);
            var readHandle = ScheduleStreamRead(session, outputBuffer, bytesRead, timeout_ms, decodeHandle);

            // This synchronous helper owns the TempJob scratch lifetime.
            readHandle.Complete();
            feedErrorCode.Dispose();
            decodeErrorCode.Dispose();

            return readHandle;
        }
    }
}
