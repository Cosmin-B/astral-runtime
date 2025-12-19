// AstralToolset.cs - Unity owner for native structured-output toolsets.

using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Astral.Runtime
{
    [Serializable]
    public struct AstralToolDefinition
    {
        public uint toolId;
        public string name;
        public string description;
        public string jsonSchema;
    }

    public struct AstralToolInfo
    {
        public uint toolId;
        public string name;
        public string description;
        public string jsonSchema;
    }

    public struct AstralToolCall
    {
        public bool found;
        public uint toolId;
        public int parseStatus;
        public string name;
        public string argumentsJson;
    }

    /// <summary>
    /// Native structured-output toolset handle.
    /// </summary>
    public sealed class AstralToolset : IDisposable
    {
        private const int FirstByteIndex = 0;
        private const int EmptyToolCount = 0;

        private AstralNative.AstralHandle m_handle;
        private bool m_disposed;

        public bool IsValid => !m_disposed && m_handle.IsValid;
        internal AstralNative.AstralHandle Handle => m_handle;

        public static unsafe AstralToolset Create(
            AstralToolDefinition[] tools,
            AstralNative.AstralToolChoiceMode choiceMode = AstralNative.AstralToolChoiceMode.Auto)
        {
            if (tools == null || tools.Length == EmptyToolCount)
            {
                throw new ArgumentException("tools must not be empty", nameof(tools));
            }

            int toolCount = tools.Length;
            var nativeTools = new NativeArray<AstralNative.AstralToolDesc>(
                toolCount,
                Allocator.Temp,
                NativeArrayOptions.UninitializedMemory);
            var names = new NativeArray<byte>[toolCount];
            var descriptions = new NativeArray<byte>[toolCount];
            var schemas = new NativeArray<byte>[toolCount];

            try
            {
                for (int index = FirstByteIndex; index < toolCount; ++index)
                {
                    AstralToolDefinition tool = tools[index];
                    ValidateToolDefinition(ref tool, index);

                    AstralNative.AstralSpanU8 name = AstralNative.AstralSpanU8.FromString(tool.name, out names[index]);
                    AstralNative.AstralSpanU8 description =
                        AstralNative.AstralSpanU8.FromString(tool.description, out descriptions[index]);
                    AstralNative.AstralSpanU8 schema =
                        AstralNative.AstralSpanU8.FromString(tool.jsonSchema, out schemas[index]);

                    nativeTools[index] = new AstralNative.AstralToolDesc
                    {
                        size = (uint)Marshal.SizeOf<AstralNative.AstralToolDesc>(),
                        tool_id = tool.toolId,
                        name = name,
                        description = description,
                        json_schema = schema
                    };
                }

                var desc = new AstralNative.AstralToolsetDesc
                {
                    size = (uint)Marshal.SizeOf<AstralNative.AstralToolsetDesc>(),
                    tool_count = (uint)toolCount,
                    choice_mode = (uint)choiceMode,
                    tools = (IntPtr)nativeTools.GetUnsafeReadOnlyPtr()
                };

                int err = AstralNative.astral_toolset_create(ref desc, out var handle);
                ThrowIfError(err, "astral_toolset_create");

                return new AstralToolset
                {
                    m_handle = handle,
                    m_disposed = false
                };
            }
            finally
            {
                DisposeTempArrays(names);
                DisposeTempArrays(descriptions);
                DisposeTempArrays(schemas);
                nativeTools.Dispose();
            }
        }

        public uint Count()
        {
            ThrowIfInvalid();
            int err = AstralNative.astral_toolset_count(m_handle, out uint count);
            ThrowIfError(err, "astral_toolset_count");
            return count;
        }

        public AstralToolInfo GetTool(uint index)
        {
            ThrowIfInvalid();
            var info = new AstralNative.AstralToolInfo
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralToolInfo>()
            };
            int err = AstralNative.astral_toolset_get(m_handle, index, ref info);
            ThrowIfError(err, "astral_toolset_get");

            return new AstralToolInfo
            {
                toolId = info.tool_id,
                name = info.name.ToUtf8String(),
                description = info.description.ToUtf8String(),
                jsonSchema = info.json_schema.ToUtf8String()
            };
        }

        public AstralToolCall ParseCall(string generatedText)
        {
            NativeArray<byte> textArray;
            var textSpan = AstralNative.AstralSpanU8.FromString(generatedText, out textArray);
            try
            {
                return ParseCall(textSpan);
            }
            finally
            {
                if (textArray.IsCreated)
                {
                    textArray.Dispose();
                }
            }
        }

        public AstralToolCall ParseCall(NativeArray<byte> generatedText)
        {
            if (!generatedText.IsCreated)
            {
                throw new ArgumentException("generatedText must be created", nameof(generatedText));
            }

            return ParseCall(AstralNative.AstralSpanU8.FromNativeArray(generatedText));
        }

        public void Dispose()
        {
            if (m_disposed)
            {
                return;
            }

            if (m_handle.IsValid)
            {
                AstralNative.astral_toolset_destroy(m_handle);
                m_handle = AstralNative.AstralHandle.Invalid;
            }

            m_disposed = true;
        }

        private AstralToolCall ParseCall(AstralNative.AstralSpanU8 generatedText)
        {
            ThrowIfInvalid();
            var result = new AstralNative.AstralToolCallResult
            {
                size = (uint)Marshal.SizeOf<AstralNative.AstralToolCallResult>()
            };

            int err = AstralNative.astral_toolset_parse_call(m_handle, generatedText, ref result);
            if (err == AstralNative.ASTRAL_E_NOT_FOUND)
            {
                return new AstralToolCall
                {
                    found = false,
                    parseStatus = err
                };
            }
            ThrowIfError(err, "astral_toolset_parse_call");

            return new AstralToolCall
            {
                found = true,
                toolId = result.tool_id,
                parseStatus = result.parse_status,
                name = result.name.ToUtf8String(),
                argumentsJson = result.arguments_json.ToUtf8String()
            };
        }

        private void ThrowIfInvalid()
        {
            if (!IsValid)
            {
                throw new AstralException("Toolset is not valid (disposed or not created).");
            }
        }

        private static void ValidateToolDefinition(ref AstralToolDefinition tool, int index)
        {
            if (string.IsNullOrEmpty(tool.name))
            {
                throw new ArgumentException($"tools[{index}].name must not be empty", nameof(tool));
            }
            if (string.IsNullOrEmpty(tool.jsonSchema))
            {
                throw new ArgumentException($"tools[{index}].jsonSchema must not be empty", nameof(tool));
            }
        }

        private static void DisposeTempArrays(NativeArray<byte>[] arrays)
        {
            for (int index = FirstByteIndex; index < arrays.Length; ++index)
            {
                if (arrays[index].IsCreated)
                {
                    arrays[index].Dispose();
                }
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
}
