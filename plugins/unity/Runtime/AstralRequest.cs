using System;
using System.Runtime.InteropServices;

namespace Astral.Runtime
{
    public static class AstralRequest
    {
        public static AstralNative.AstralRequestRef FromSession(AstralSession session)
        {
            if (session == null)
            {
                throw new ArgumentNullException(nameof(session));
            }
            if (!session.IsValid)
            {
                throw new AstralException("Session is not valid.");
            }

            int err = AstralNative.astral_request_from_session(session.Handle, out var request);
            ThrowIfError(err, "astral_request_from_session");
            return request;
        }

        public static AstralNative.AstralRequestRef FromConversation(AstralConversation conversation)
        {
            if (conversation == null)
            {
                throw new ArgumentNullException(nameof(conversation));
            }
            if (!conversation.IsValid)
            {
                throw new AstralException("Conversation is not valid.");
            }

            int err = AstralNative.astral_request_from_conversation(conversation.Handle, out var request);
            ThrowIfError(err, "astral_request_from_conversation");
            return request;
        }

        public static AstralNative.AstralRequestRef FromAgentChat(AstralAgent agent)
        {
            if (agent == null)
            {
                throw new ArgumentNullException(nameof(agent));
            }
            if (!agent.IsValid)
            {
                throw new AstralException("Agent is not valid.");
            }

            int err = AstralNative.astral_request_from_agent_chat(agent.Handle, out var request);
            ThrowIfError(err, "astral_request_from_agent_chat");
            return request;
        }

        public static AstralNative.AstralRequestRef FromEmbedding(AstralEmbedder embedder, ulong ticket)
        {
            if (embedder == null)
            {
                throw new ArgumentNullException(nameof(embedder));
            }
            if (!embedder.IsValid)
            {
                throw new AstralException("Embedder is not valid.");
            }
            if (ticket == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(ticket));
            }

            int err = AstralNative.astral_request_from_embedding(embedder.Handle, ticket, out var request);
            ThrowIfError(err, "astral_request_from_embedding");
            return request;
        }

        public static AstralNative.AstralRequestRef FromMemorySearch(AstralNative.AstralHandle cursor)
        {
            if (!cursor.IsValid)
            {
                throw new ArgumentException("Cursor handle is not valid.", nameof(cursor));
            }

            int err = AstralNative.astral_request_from_memory_search(cursor, out var request);
            ThrowIfError(err, "astral_request_from_memory_search");
            return request;
        }

        public static AstralNative.AstralRequestStatus GetStatus(AstralNative.AstralRequestRef request)
        {
            var status = NewStatus();
            int err = AstralNative.astral_request_state(ref request, ref status);
            ThrowIfError(err, "astral_request_state");
            return status;
        }

        public static bool TryGetStatus(
            AstralNative.AstralRequestRef request,
            out AstralNative.AstralRequestStatus status,
            out int errorCode)
        {
            status = NewStatus();
            errorCode = AstralNative.astral_request_state(ref request, ref status);
            return errorCode == AstralNative.ASTRAL_OK;
        }

        public static AstralNative.AstralRequestStatus Wait(AstralNative.AstralRequestRef request, uint timeoutMs)
        {
            var status = NewStatus();
            int err = AstralNative.astral_request_wait(ref request, timeoutMs, ref status);
            ThrowIfError(err, "astral_request_wait");
            return status;
        }

        public static bool TryWait(
            AstralNative.AstralRequestRef request,
            uint timeoutMs,
            out AstralNative.AstralRequestStatus status,
            out int errorCode)
        {
            status = NewStatus();
            errorCode = AstralNative.astral_request_wait(ref request, timeoutMs, ref status);
            return errorCode == AstralNative.ASTRAL_OK;
        }

        public static void Cancel(AstralNative.AstralRequestRef request)
        {
            int err = AstralNative.astral_request_cancel(ref request);
            ThrowIfError(err, "astral_request_cancel");
        }

        public static bool TryCancel(AstralNative.AstralRequestRef request, out int errorCode)
        {
            errorCode = AstralNative.astral_request_cancel(ref request);
            return errorCode == AstralNative.ASTRAL_OK;
        }

        private static AstralNative.AstralRequestStatus NewStatus()
        {
            return new AstralNative.AstralRequestStatus
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralRequestStatus>()
            };
        }

        private static void ThrowIfError(int err, string call)
        {
            if (err != AstralNative.ASTRAL_OK)
            {
                throw new AstralException($"{call} failed: {AstralRuntime.GetErrorString(err)}", err);
            }
        }
    }
}
