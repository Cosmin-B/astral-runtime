// Astral Unity Jobs System integration.
// Jobs poll token streams on worker threads with caller-owned NativeArray buffers.
//
// Design: Use Unity Jobs System for async token streaming.
// Allocation model: callers own NativeArray buffers before scheduling.
// Thread-safety: NativeArray owns the cross-thread byte buffers.

using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace Astral.Runtime
{
    /// <summary>
    /// Job for polling Astral token stream on background thread.
    ///
    /// BURST COMPATIBILITY:
    /// - All fields are blittable types
    /// - No managed references (IntPtr for session handle)
    /// - No exceptions (returns error codes)
    ///
    /// THREAD-SAFETY:
    /// - NativeArray owns the cross-thread byte buffers
    /// - Session handle is thread-safe (Astral runtime handles locking)
    /// - No shared state between jobs
    ///
    /// PERFORMANCE:
    /// - Burst compiles the job wrapper.
    /// - Caller-owned buffers avoid per-token managed string conversion.
    /// - Native stream reads come from the runtime SPSC queue.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
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
            // Convert NativeArray to mutable span (zero-copy)
            var span = outputBuffer.AsMutSpan();

            // Read from Astral stream
            // Returns: >0 = bytes read, 0 = timeout/no data, <0 = error code
            int result = AstralNative.astral_stream_read(session, span, timeout_ms);

            // Store result
            bytesRead[0] = result;
        }
    }

    /// <summary>
    /// Job for feeding prompt chunks to Astral session.
    ///
    /// Use case: Large prompts can be fed incrementally on background thread
    /// to avoid blocking main thread.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
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
            // Convert NativeArray to read-only span (zero-copy)
            var span = promptChunk.AsSpan();

            // Feed to Astral session
            int result = (int)AstralNative.astral_session_feed(session, span, finalize);

            // Store error code
            errorCode[0] = result;
        }
    }

    /// <summary>
    /// Job for triggering decode on Astral session.
    ///
    /// Use case: Kick off inference on background thread, main thread polls results.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
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
    /// - Job data stays blittable for Burst-compatible scheduling.
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
        /// <summary>
        /// Schedule stream read on background thread.
        ///
        /// PARAMETERS:
        /// - session: Astral session handle
        /// - outputBuffer: Pre-allocated buffer for token bytes (caller must dispose)
        /// - bytesRead: Output array (length 1) for bytes read count (caller must dispose)
        /// - timeout_ms: Timeout in milliseconds (0 = non-blocking)
        /// - dependency: Optional job dependency
        ///
        /// RETURNS:
        /// - JobHandle: Use .Complete() to wait or chain dependencies
        ///
        /// THREAD-SAFETY:
        /// - Safe to call from any thread
        /// - outputBuffer/bytesRead must not be accessed until job completes
        /// </summary>
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

        /// <summary>
        /// Schedule prompt feed on background thread.
        ///
        /// PARAMETERS:
        /// - session: Astral session handle
        /// - promptChunk: UTF-8 encoded prompt bytes (caller must dispose)
        /// - finalize: 1 = last chunk, 0 = more chunks coming
        /// - errorCode: Output array (length 1) for error code (caller must dispose)
        /// - dependency: Optional job dependency
        ///
        /// RETURNS:
        /// - JobHandle: Use .Complete() to wait or chain dependencies
        /// </summary>
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

        /// <summary>
        /// Schedule decode on background thread.
        ///
        /// PARAMETERS:
        /// - session: Astral session handle
        /// - errorCode: Output array (length 1) for error code (caller must dispose)
        /// - dependency: Optional job dependency
        ///
        /// RETURNS:
        /// - JobHandle: Use .Complete() to wait or chain dependencies
        ///
        /// NOTE: Decode is typically fast (just queues work), but scheduling on
        /// background thread avoids any potential blocking on main thread.
        /// </summary>
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
        /// Schedule full inference pipeline: Feed -> Decode -> StreamRead.
        ///
        /// CONVENIENCE METHOD: Chains all three operations with dependencies.
        ///
        /// PARAMETERS:
        /// - session: Astral session handle
        /// - prompt: UTF-8 encoded prompt bytes
        /// - outputBuffer: Pre-allocated buffer for response
        /// - bytesRead: Output array for bytes read count
        /// - timeout_ms: Timeout for stream read
        ///
        /// RETURNS:
        /// - JobHandle: Completes when all operations finish
        ///
        /// CLEANUP: Caller must dispose prompt, outputBuffer, bytesRead after handle completes
        /// </summary>
        public static JobHandle ScheduleInference(
            AstralNative.AstralHandle session,
            NativeArray<byte> prompt,
            NativeArray<byte> outputBuffer,
            NativeArray<int> bytesRead,
            uint timeout_ms = 5000)
        {
            // Create temporary arrays for intermediate results
            var feedErrorCode = new NativeArray<int>(1, Allocator.TempJob);
            var decodeErrorCode = new NativeArray<int>(1, Allocator.TempJob);

            // Chain jobs: Feed -> Decode -> StreamRead
            var feedHandle = ScheduleFeed(session, prompt, finalize: 1, feedErrorCode);
            var decodeHandle = ScheduleDecode(session, decodeErrorCode, feedHandle);
            var readHandle = ScheduleStreamRead(session, outputBuffer, bytesRead, timeout_ms, decodeHandle);

            // Dispose temporary arrays when jobs complete
            // NOTE: This is safe because Unity Jobs System guarantees completion before disposal
            readHandle.Complete(); // Wait for all jobs
            feedErrorCode.Dispose();
            decodeErrorCode.Dispose();

            return readHandle;
        }
    }
}
