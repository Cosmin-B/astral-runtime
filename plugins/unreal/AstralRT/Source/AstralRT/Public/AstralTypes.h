#pragma once

#include "CoreMinimal.h"

#include "AstralTypes.generated.h"

enum class EAstralError : int32
{
    OK = 0,
    Invalid = -1,
    NoMem = -2,
    Busy = -3,
    Timeout = -4,
    State = -5,
    Backend = -6,
    Canceled = -7,
    Unsupported = -8,
    NotFound = -9
};

UENUM(BlueprintType)
enum class EAstralRequestKind : uint8
{
    None = 0,
    Session = 1,
    Conversation = 2,
    AgentChat = 3,
    Embedding = 4,
    MemorySearch = 5
};

UENUM(BlueprintType)
enum class EAstralRequestState : uint8
{
    Invalid = 0,
    Queued = 1,
    Running = 2,
    Completed = 3,
    Canceled = 4,
    Failed = 5
};

/** Blueprint-safe status for native ticketed operations. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralAsyncResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bSuccess = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ErrorCode = static_cast<int32>(EAstralError::OK);

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 Ticket = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bBackpressure = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bTimeout = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bCanceled = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bUnsupported = false;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralRequestRef
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    EAstralRequestKind Kind = EAstralRequestKind::None;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 OwnerHandle = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 Ticket = 0;
};

USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralRequestStatus
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    EAstralRequestKind Kind = EAstralRequestKind::None;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    EAstralRequestState State = EAstralRequestState::Invalid;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ErrorCode = static_cast<int32>(EAstralError::OK);

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 OwnerHandle = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 Ticket = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 QueueDepth = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bHasTicket = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bStream = false;
};

/** Blueprint-safe status for native handle and polling operations. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralOperationResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bSuccess = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ErrorCode = static_cast<int32>(EAstralError::OK);

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 Handle = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 Count = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bBackpressure = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bTimeout = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bCanceled = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bUnsupported = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bNotFound = false;

    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    bool bEndOfStream = false;
};

/** Pixel layouts accepted by Astral image media calls. */
UENUM(BlueprintType)
enum class EAstralImageFormat : uint8
{
    RGB8 = 0,
    RGBA8 = 1,
    RGB_F32 = 2
};

/** Packed sample layouts accepted by Astral audio media calls. */
UENUM(BlueprintType)
enum class EAstralAudioFormat : uint8
{
    F32 = 0,
    I16 = 1
};

/** Media initialization flags mirrored from the native C ABI. */
UENUM(BlueprintType)
enum class EAstralMediaFlags : uint8
{
    None = 0,
    UseGPU = 1u << 0,
    Warmup = 1u << 1
};

/** GPU routing fields that should be honored by media-aware backends. */
UENUM(BlueprintType)
enum class EAstralGpuRouteFlags : uint8
{
    None = 0,
    Device = 1u << 0,
    DeviceMask = 1u << 1,
    Stream = 1u << 2
};

/** Backing source for a model or media projector. Unreal currently exposes path and bytes. */
UENUM(BlueprintType)
enum class EAstralModelSourceKind : uint8
{
    Path = 0,
    Memory = 1,
    IO = 2
};

/** Structured-output tool selection mode. */
UENUM(BlueprintType)
enum class EAstralToolChoiceMode : uint8
{
    Auto = 0,
    Required = 1,
    TextOrTool = 2
};

/** Native chat history role. */
UENUM(BlueprintType)
enum class EAstralAgentRole : uint8
{
    None = 0,
    System = 1,
    User = 2,
    Assistant = 3,
    Tool = 4
};

/** Native agent history overflow behavior. */
UENUM(BlueprintType)
enum class EAstralAgentOverflowPolicy : uint8
{
    Reject = 0,
    TruncateOldest = 1
};

/** Prompt section encoded into native prompt cache keys. */
UENUM(BlueprintType)
enum class EAstralPromptSectionKind : uint8
{
    None = 0,
    System = 1,
    Tools = 2,
    Memory = 3,
    History = 4,
    User = 5,
    Raw = 6
};

/** Prompt cache eviction policy. */
UENUM(BlueprintType)
enum class EAstralPromptCacheEvictionPolicy : uint8
{
    Fifo = 0
};

/** Text or token splitting mode for native chunking. */
UENUM(BlueprintType)
enum class EAstralChunkMode : uint8
{
    None = 0,
    Char = 1,
    Word = 2,
    Sentence = 3,
    Token = 4
};

/** Similarity metric used by native memory search. */
UENUM(BlueprintType)
enum class EAstralMemoryMetric : uint8
{
    Dot = 0,
    Cosine = 1,
    L2 = 2
};

/** Native memory index search structure. */
UENUM(BlueprintType)
enum class EAstralMemoryIndexKind : uint8
{
    Flat = 0,
    Graph = 1
};

/** Native memory index vector storage format. */
UENUM(BlueprintType)
enum class EAstralMemoryStorageKind : uint8
{
    F32 = 0,
    Q8 = 1
};

/** Root used to resolve relative filesystem paths before they cross the C ABI. */
UENUM(BlueprintType)
enum class EAstralUnrealPathRoot : uint8
{
    Raw = 0,
    ProjectContent = 1,
    ProjectSaved = 2,
    ProjectPersistentDownload = 3
};

inline constexpr int32 AstralChunkDefaultMaxUnits = 128;
inline constexpr int32 AstralPromptCacheDefaultMaxEntries = 64;
inline constexpr int32 AstralPromptCacheDefaultMaxTokens = 8192;
inline constexpr int32 AstralPromptCacheTrackStatsFlag = 1;

/** Load settings for a native Astral model. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelDesc
{
    GENERATED_BODY()

    /** Backing source used for the model payload. Path and Memory are supported by this wrapper. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralModelSourceKind SourceKind = EAstralModelSourceKind::Path;

    /** Root for relative ModelPath values. Absolute paths are passed through unchanged. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralUnrealPathRoot PathRoot = EAstralUnrealPathRoot::Raw;

    /** Filesystem path to a GGUF model when SourceKind is Path. Empty is valid only for providers that do not need a file. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString ModelPath;

    /** Model bytes when SourceKind is Memory, usually from cooked bulk data or a staged cache file. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> ModelBytes;

    /** Provider name such as "cpu" or "mock". Empty lets the runtime pick. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString BackendName;

    /** Bearer key used by the remote backend. Leave empty for local providers or unauthenticated endpoints. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString RemoteApiKey;

    /** Number of layers requested on GPU-capable backends; 0 keeps execution on CPU. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuLayers = 0;

    /** Token context window requested for the model. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 ContextSize = 2048;

    /** Native batch size hint for prompt/decode work. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 BatchSize = 512;

    /** Worker count requested by the caller; 0 uses the runtime default. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 NumThreads = 0;

    /** Load the model for embeddings without preparing generation state where supported. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bEmbeddingsOnly = false;
};

/** Image payload passed to session feed and embedding calls. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralImageDesc
{
    GENERATED_BODY()

    /** Pixel format of Pixels. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralImageFormat Format = EAstralImageFormat::RGB8;

    /** Image width in pixels. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Width = 0;

    /** Image height in pixels. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Height = 0;

    /** Bytes per row. Use 0 when rows are tightly packed for the selected format. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 RowStride = 0;

    /** Native image flags for backend-specific media behavior. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Flags = 0;

    /** Pixel bytes. The wrapper passes this storage to native code for the call duration. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> Pixels;

    /** CUDA device index requested when Device routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    /** Bitmask of EAstralGpuRouteFlags values. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuRouteFlags = 0;

    /** Allowed CUDA device mask requested when DeviceMask routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuDeviceMask = 0;

    /** Backend-specific stream pointer encoded as an integer for Blueprint transport. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuStream = 0;
};

/** Audio payload passed to session feed and embedding calls. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralAudioDesc
{
    GENERATED_BODY()

    /** Sample format of Samples. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralAudioFormat Format = EAstralAudioFormat::F32;

    /** Number of interleaved channels in Samples. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Channels = 1;

    /** Sample rate in hertz. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 SampleRate = 16000;

    /** Frame count. Use 0 to infer from Samples, Format, and Channels. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 FrameCount = 0;

    /** Interleaved sample bytes. The wrapper passes this storage to native code for the call duration. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> Samples;

    /** Native audio flags for backend-specific media behavior. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Flags = 0;

    /** CUDA device index requested when Device routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    /** Bitmask of EAstralGpuRouteFlags values. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuRouteFlags = 0;

    /** Allowed CUDA device mask requested when DeviceMask routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuDeviceMask = 0;

    /** Backend-specific stream pointer encoded as an integer for Blueprint transport. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuStream = 0;
};

/** Projector or encoder settings for model media initialization. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelMediaDesc
{
    GENERATED_BODY()

    /** Source kind for the media projector. Path and Memory are exposed by the wrapper. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralModelSourceKind SourceKind = EAstralModelSourceKind::Path;

    /** Filesystem path used when SourceKind is Path. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString MediaPath;

    /** Root for relative MediaPath values. Absolute paths are passed through unchanged. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralUnrealPathRoot MediaPathRoot = EAstralUnrealPathRoot::Raw;

    /** Projector bytes used when SourceKind is Memory. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    TArray<uint8> MediaBytes;

    /** Native media init flags such as GPU use and warmup. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Flags = 0;

    /** Minimum image token budget requested from the media projector. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 ImageMinTokens = 0;

    /** Maximum image token budget requested from the media projector. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 ImageMaxTokens = 0;

    /** CUDA device index requested when Device routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuDevice = 0;

    /** Bitmask of EAstralGpuRouteFlags values. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 GpuRouteFlags = 0;

    /** Allowed CUDA device mask requested when DeviceMask routing is enabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuDeviceMask = 0;

    /** Backend-specific stream pointer encoded as an integer for Blueprint transport. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int64 GpuStream = 0;
};

/** LoRA or adapter payload loaded against a native model. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralAdapterDesc
{
    GENERATED_BODY()

    /** Filesystem path to an adapter file. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    FString AdapterPath;

    /** Root for relative AdapterPath values. Absolute paths are passed through unchanged. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    EAstralUnrealPathRoot PathRoot = EAstralUnrealPathRoot::Raw;
};

/** Native diagnostics for one loaded LoRA or adapter payload. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralAdapterInfo
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Adapters")
    int64 ModelHandle = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Adapters")
    int32 RefCount = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Adapters")
    int32 PathBytes = 0;
};

/** One tool/function definition for structured output. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralToolDesc
{
    GENERATED_BODY()

    /** Stable native id returned with parsed tool calls. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Tools")
    int32 ToolId = 0;

    /** Tool name used in generated tool-call JSON. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Tools")
    FString Name;

    /** Human-readable tool description. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Tools")
    FString Description;

    /** JSON schema for the tool arguments object. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Tools")
    FString JsonSchema;
};

/** Parsed structured-output tool call. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralToolCallResult
{
    GENERATED_BODY()

    /** True when a known tool name was found. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral|Tools")
    bool bFound = false;

    /** Native parse status for the arguments payload. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral|Tools")
    int32 ParseStatus = 0;

    /** Stable tool id from the native toolset. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral|Tools")
    int32 ToolId = 0;

    /** Matched tool name. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral|Tools")
    FString Name;

    /** Raw JSON object for the tool arguments. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral|Tools")
    FString ArgumentsJson;
};

/** Native chunking settings for text or token ranges. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralChunkerDesc
{
    GENERATED_BODY()

    /** Split mode. Token mode expects an already-tokenized input count. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Chunking")
    EAstralChunkMode Mode = EAstralChunkMode::Word;

    /** Maximum words, sentences, characters, or tokens in one range. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Chunking")
    int32 MaxUnits = AstralChunkDefaultMaxUnits;

    /** Units repeated at the start of the next range. Must be smaller than MaxUnits. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Chunking")
    int32 OverlapUnits = 0;

    /** Caller-defined document id copied into emitted ranges. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Chunking")
    int32 DocumentId = 0;

    /** Caller-defined group id copied into emitted ranges. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Chunking")
    int32 GroupId = 0;

    /** Optional delimiter bytes for sentence mode. Empty uses the native default delimiters. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Chunking")
    FString Delimiters;
};

/** One text or token range emitted by native chunking. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralChunkRange
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Chunking")
    int32 DocumentId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Chunking")
    int32 ChunkId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Chunking")
    int32 GroupId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Chunking")
    int32 ByteBegin = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Chunking")
    int32 ByteEnd = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Chunking")
    int32 TokenBegin = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Chunking")
    int32 TokenEnd = 0;
};

/** Fixed-size native vector memory index settings. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralMemoryIndexDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 Dimension = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 Capacity = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    EAstralMemoryMetric Metric = EAstralMemoryMetric::Cosine;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    EAstralMemoryIndexKind IndexKind = EAstralMemoryIndexKind::Flat;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 GraphNeighbors = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 GraphSearch = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 GraphQuerySearch = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    EAstralMemoryStorageKind StorageKind = EAstralMemoryStorageKind::F32;
};

/** Metadata associated with one vector in a native memory index. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralMemoryRecord
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int64 Key = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 GroupId = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 DocumentId = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 ChunkId = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Memory")
    int32 Flags = 0;
};

/** Result from a native memory search. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralMemorySearchResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 Key = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 GroupId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 DocumentId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 ChunkId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    float Score = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 Flags = 0;
};

/** Runtime footprint and snapshot size for a native memory index. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralMemoryStats
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 Dimension = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 Capacity = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 Count = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    EAstralMemoryMetric Metric = EAstralMemoryMetric::Cosine;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    EAstralMemoryIndexKind IndexKind = EAstralMemoryIndexKind::Flat;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 GraphNeighbors = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 GraphSearch = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 GraphQuerySearch = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int32 GraphLevels = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    EAstralMemoryStorageKind StorageKind = EAstralMemoryStorageKind::F32;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 VectorBytes = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 MetadataBytes = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 GraphBytes = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 GraphEdges = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 GraphBaseEdges = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 GraphUpperEdges = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 GraphBuildScoreEvals = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 GraphBuildCandidateVisits = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 TotalBytes = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Memory")
    int64 SaveBytes = 0;
};

/** Native prompt cache capacity and telemetry settings. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralPromptCacheDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    int32 MaxEntries = AstralPromptCacheDefaultMaxEntries;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    int32 MaxTokens = AstralPromptCacheDefaultMaxTokens;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    int32 MaxBytes = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    EAstralPromptCacheEvictionPolicy EvictionPolicy = EAstralPromptCacheEvictionPolicy::Fifo;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    bool bTrackStats = true;
};

/** Native prompt cache lookup key. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralPromptCacheKey
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    EAstralPromptSectionKind Section = EAstralPromptSectionKind::System;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    int64 ModelHandle = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    int64 Key = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Prompt Cache")
    int32 Generation = 0;
};

/** Native prompt cache counters and capacity. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralPromptCacheStats
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int32 Entries = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int32 MaxEntries = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int32 Tokens = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int32 MaxTokens = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int32 Bytes = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int32 MaxBytes = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int64 Hits = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int64 Misses = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Prompt Cache")
    int64 Evictions = 0;
};

/** Native agent configuration. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralAgentDesc
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int64 ModelHandle = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int64 PromptCacheHandle = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int64 MemoryIndexHandle = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int64 ToolsetHandle = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int32 MaxTokens = 128;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    float Temperature = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int32 TopK = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    float TopP = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    bool bStream = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int32 Seed = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    EAstralToolChoiceMode ToolChoiceMode = EAstralToolChoiceMode::Auto;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int32 MaxMessages = 64;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int32 MaxPromptBytes = 65536;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    EAstralAgentOverflowPolicy OverflowPolicy = EAstralAgentOverflowPolicy::Reject;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    int32 SlotAffinity = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    FString SystemPrompt;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    FString Summary;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral|Agent")
    FString MemoryContext;
};

/** Result from a native agent chat request. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralAgentChatResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 State = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 PromptBytes = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 HistoryMessages = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 PromptTokens = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 PromptCacheReusedTokens = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 PromptCacheNewTokens = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 PromptCacheHits = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 PromptCacheMisses = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int32 LastError = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    double PromptBuildMs = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    int64 GeneratedTokens = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    double TimeToFirstTokenMs = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "Astral|Agent")
    double TokensPerSecond = 0.0;
};

/** Media capabilities reported after InitMedia succeeds. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralMediaInfo
{
    GENERATED_BODY()

    /** Nonzero when the initialized projector accepts image inputs. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 SupportsImage = 0;

    /** Nonzero when the initialized projector accepts audio inputs. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 SupportsAudio = 0;

    /** Preferred audio sample rate reported by the projector. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 AudioSampleRate = 0;

    /** Minimum image-token budget reported by the projector. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ImageMinTokens = 0;

    /** Maximum image-token budget reported by the projector. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ImageMaxTokens = 0;
};

/** Session creation and reset settings for generation. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralSessionDesc
{
    GENERATED_BODY()

    /** Maximum tokens to generate before the session stops. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 MaxTokens = 512;

    /** Sampling temperature. Use 0 for greedy-style deterministic sampling. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float Temperature = 0.7f;

    /** Top-k candidate limit. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 TopK = 40;

    /** Nucleus sampling probability cutoff. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TopP = 0.95f;

    /** Enable the ticker that drains native stream bytes onto Unreal delegates. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bStreamEnabled = true;

    /** Random seed forwarded to native sampling. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 Seed = 0;
};

/** Runtime sampler settings that can be changed after session creation. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralSamplerDesc
{
    GENERATED_BODY()

    /** Sampling temperature. Use 0 for greedy-style deterministic sampling. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float Temperature = 0.7f;

    /** Top-k candidate limit. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 TopK = 40;

    /** Nucleus sampling probability cutoff. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TopP = 0.95f;

    /** Minimum probability cutoff relative to the most likely token. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float MinP = 0.0f;

    /** Typical sampling cutoff. Values >= 1 leave typical sampling disabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float TypicalP = 1.0f;

    /** Repetition penalty multiplier. 1 leaves logits unchanged. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float RepeatPenalty = 1.0f;

    /** Repetition window: 0 disables it, -1 uses the model context size. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    int32 RepeatLastN = 0;

    /** Include newline tokens in repetition-penalty accounting. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    bool bPenalizeNewline = false;

    /** Presence penalty added for tokens that appeared in the recent window. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float PresencePenalty = 0.0f;

    /** Frequency penalty scaled by token count in the recent window. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Astral")
    float FrequencyPenalty = 0.0f;
};

/** Static limits reported by a loaded model. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralModelLimits
{
    GENERATED_BODY()

    /** Vocabulary size reported by the backend. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 VocabSize = 0;

    /** Context size used by the loaded model. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 ContextSize = 0;

    /** Maximum native batch size reported by the backend. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 MaxBatch = 0;

    /** Maximum concurrent slots reported by continuous-batching backends. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int32 MaxSlots = 0;
};

/** Runtime counters exposed for profiling and production diagnostics. */
USTRUCT(BlueprintType)
struct ASTRALRT_API FAstralStats
{
    GENERATED_BODY()

    /** Model/session initialization time in milliseconds when reported. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double InitTimeMs = 0.0;

    /** Milliseconds from decode start to first token when reported. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double FirstTokenTimeMs = 0.0;

    /** Throughput reported by the native backend. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    double TokensPerSecond = 0.0;

    /** Bytes committed by the runtime arena. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 BytesCommitted = 0;

    /** Bytes reserved by the runtime arena. */
    UPROPERTY(BlueprintReadOnly, Category = "Astral")
    int64 BytesReserved = 0;
};
