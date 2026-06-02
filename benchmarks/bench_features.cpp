#include "bench_clock.hpp"
#include "bench_common.hpp"

#include "../include/astral_rt.h"
#include "../src/platform/atomics.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace astral::bench {

namespace {

static constexpr uint64_t kPromptCacheOnlyReserveBytes = 64ULL << 20;
static constexpr uint64_t kFeatureDefaultIters = 2000;
static constexpr uint32_t kBenchBytesPerKiB = 1024;
static constexpr char kDisabledEnvValue[] = "0";
static constexpr uint32_t kBenchToolSearchId = 1;
static constexpr uint32_t kBenchToolOpenId = 2;
static constexpr uint32_t kBenchToolCount = 2;
static constexpr uint32_t kBenchToolManyCount = 16;
static constexpr uint32_t kBenchToolManyTargetIndex = kBenchToolManyCount - 1u;
static constexpr uint32_t kBenchToolManyIdBase = 100;
static constexpr uint32_t kBenchChunkMaxWords = 32;
static constexpr uint32_t kBenchChunkOverlapWords = 4;
static constexpr uint32_t kBenchChunkRangeCapacity = 64;
static constexpr uint32_t kBenchChunkTokenCount = 4096;
static constexpr uint32_t kBenchChunkMaxTokens = 256;
static constexpr uint32_t kBenchChunkOverlapTokens = 32;
static constexpr uint32_t kBenchChunkTokenRangeCapacity = 19;
static constexpr uint32_t kBenchMemoryDim = 32;
static constexpr uint32_t kBenchMemoryCapacity = 1024;
static constexpr uint32_t kBenchMemoryTopOne = 1;
static constexpr uint32_t kBenchMemoryTopK = 8;
static constexpr uint32_t kBenchMemoryFetchK = 4;
static constexpr uint32_t kBenchMemoryBatchQueries = 8;
static constexpr uint32_t kBenchMemoryMinDim = 1;
static constexpr uint32_t kBenchMemoryMaxDim = 8192;
static constexpr uint32_t kBenchMemoryMinCapacity = 1;
static constexpr uint32_t kBenchMemorySweepTiny = 100;
static constexpr uint32_t kBenchMemorySweepSmall = 1000;
static constexpr uint32_t kBenchMemorySweepMedium = 10000;
static constexpr uint32_t kBenchMemorySweepLarge = 100000;
static constexpr uint32_t kBenchMemoryHashMul0 = 0x7FEB352Du;
static constexpr uint32_t kBenchMemoryHashMul1 = 0x846CA68Bu;
static constexpr uint32_t kBenchMemoryHashShift0 = 16u;
static constexpr uint32_t kBenchMemoryHashShift1 = 15u;
static constexpr uint32_t kBenchMemoryHashShift2 = 16u;
static constexpr uint32_t kBenchMemoryHashRowMul = 0x9E3779B9u;
static constexpr uint32_t kBenchMemoryHashColMul = 0x85EBCA6Bu;
static constexpr uint64_t kBenchMemoryKeyHashMul0 = 0xBF58476D1CE4E5B9ull;
static constexpr uint64_t kBenchMemoryKeyHashMul1 = 0x94D049BB133111EBull;
static constexpr uint32_t kBenchMemoryKeyHashShift0 = 30u;
static constexpr uint32_t kBenchMemoryKeyHashShift1 = 27u;
static constexpr uint32_t kBenchMemoryKeyHashShift2 = 31u;
static constexpr float kBenchMemoryI32Scale = 1.0f / 2147483648.0f;
static constexpr uint32_t kBenchMemoryRecallQueries = 32u;
static constexpr uint32_t kBenchMemoryMinRecallQueries = 1u;
static constexpr uint32_t kBenchMemoryMaxRecallQueries = 4096u;
static constexpr uint32_t kBenchMemoryRecallNameBytes = 80u;
static constexpr uint32_t kBenchMemorySnapshotPathBytes = 128u;
static constexpr uint32_t kBenchMemoryGroupMask = 1u;
static constexpr uint32_t kBenchMemoryDocShift = 4u;
static constexpr uint64_t kBenchMemoryKeyBase = 1u;
static constexpr uint32_t kBenchMemoryGraphNeighbors = 32;
static constexpr uint32_t kBenchMemoryGraphMaxNeighbors = 64;
static constexpr uint32_t kBenchMemoryGraphSearch = 68;
static constexpr uint32_t kBenchMemoryGraphMinSearch = 4;
static constexpr uint32_t kBenchMemoryGraphMaxLevels = 16;
static constexpr double kBenchMemoryPercentScale = 100.0;
static constexpr uint32_t kBenchAgentContextTokens = 256;
static constexpr uint32_t kBenchAgentSlots = 1;
static constexpr uint32_t kBenchAgentBatchTokens = 16;
static constexpr uint32_t kBenchAgentMaxTokens = 0;
static constexpr uint32_t kBenchAgentMaxMessages = 8;
static constexpr uint32_t kBenchAgentMaxPromptBytes = 4096;
static constexpr uint32_t kBenchAgentPromptCacheEntries = 4;
static constexpr uint32_t kBenchAgentPromptCacheTokens = 256;
static constexpr uint32_t kBenchAgentSeed = 11;
static constexpr uint32_t kBenchAgentPollLimit = 65536;
static constexpr double kMsToNs = 1000000.0;
static constexpr float kBenchAdapterScale = 1.0f;
static constexpr float kBenchAdapterUpdatedScale = 0.5f;
static constexpr char kBenchAdapterPath[] = "bench-adapter";
static constexpr uint32_t kBenchSystemPromptCacheEntries = 4;
static constexpr uint32_t kBenchSystemPromptCacheTokens = 256;
static constexpr uint32_t kBenchSystemPromptTokenCapacity = 256;
static constexpr uint32_t kBenchSystemPromptSessionTokens = 256;
static constexpr uint64_t kBenchSystemPromptKey = 0xA57A5000ull;
static constexpr uint32_t kBenchSystemPromptGeneration = 1;
static constexpr char kBenchSystemPromptText[] =
    "You are a low-latency native assistant. Answer with precise steps, preserve caller-owned buffers, "
    "and keep repeated prompt sections reusable.";
static constexpr char kBenchMemoryCapacityEnv[] = "ASTRAL_BENCH_MEMORY_CAPACITY";
static constexpr char kBenchMemoryDimEnv[] = "ASTRAL_BENCH_MEMORY_DIM";
static constexpr char kBenchMemoryMetricEnv[] = "ASTRAL_BENCH_MEMORY_METRIC";
static constexpr char kBenchMemoryOnlyEnv[] = "ASTRAL_BENCH_MEMORY_ONLY";
static constexpr char kBenchMemoryCaseEnv[] = "ASTRAL_BENCH_MEMORY_CASE";
static constexpr char kBenchMemorySweepEnv[] = "ASTRAL_BENCH_MEMORY_SWEEP";
static constexpr char kBenchMemoryStorageEnv[] = "ASTRAL_BENCH_MEMORY_STORAGE";
static constexpr char kBenchMemoryGraphNeighborsEnv[] = "ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS";
static constexpr char kBenchMemoryGraphSearchEnv[] = "ASTRAL_BENCH_MEMORY_GRAPH_SEARCH";
static constexpr char kBenchMemoryGraphQuerySearchEnv[] = "ASTRAL_BENCH_MEMORY_GRAPH_QUERY_SEARCH";
static constexpr char kBenchMemoryRecallQueriesEnv[] = "ASTRAL_BENCH_MEMORY_RECALL_QUERIES";
static constexpr char kBenchTokenizeOnlyEnv[] = "ASTRAL_BENCH_TOKENIZE_ONLY";
static constexpr char kBenchMemoryMetricDot[] = "dot";
static constexpr char kBenchMemoryMetricL2[] = "l2";
static constexpr char kBenchMemoryMetricCosine[] = "cosine";
static constexpr char kBenchMemoryStorageF32[] = "f32";
static constexpr char kBenchMemoryStorageQ8[] = "q8";
static constexpr char kBenchMemoryStorageF6E2M3[] = "f6e2m3";
static constexpr char kBenchMemoryStorageF8E5M2[] = "f8e5m2";
static constexpr char kBenchMemoryCaseAddBatch[] = "add_batch";
static constexpr char kBenchMemoryCaseGraphAddBatch[] = "graph_add_batch";
static constexpr char kBenchMemoryCaseGraphAddLatency[] = "graph_add_latency";
static constexpr char kBenchMemoryCaseGraphLoad[] = "graph_load";
static constexpr char kBenchMemoryCaseSnapshotSearch[] = "snapshot_search";
static constexpr char kBenchMemoryCaseSnapshotViewSearch[] = "snapshot_view_search";
static constexpr char kBenchMemoryCaseGraphSnapshotViewSearch[] = "graph_snapshot_view_search";
static constexpr char kBenchMemoryCaseFlatSearchTop1[] = "flat_search_top1";
static constexpr char kBenchMemoryCaseFlatSearch[] = "flat_search";
static constexpr char kBenchMemoryCaseFlatSearchLatency[] = "flat_search_latency";
static constexpr char kBenchMemoryCaseFlatSearchBatch[] = "flat_search_batch";
static constexpr char kBenchMemoryCaseFlatQ8RecallSearch[] = "flat_q8_recall_search";
static constexpr char kBenchMemoryCaseFlatCompactRecallSearch[] = "flat_compact_recall_search";
static constexpr char kBenchMemoryCaseGraphTop1[] = "graph_top1";
static constexpr char kBenchMemoryCaseGraphSearch[] = "graph_search";
static constexpr char kBenchMemoryCaseGraphSearchBatch[] = "graph_search_batch";
static constexpr char kBenchMemoryCaseGraphSearchLatency[] = "graph_search_latency";
static constexpr char kBenchMemoryCaseGraphRecall[] = "graph_recall";
static constexpr char kBenchMemoryCaseGraphRecallTop1[] = "graph_recall_top1";
static constexpr char kBenchMemoryCaseGraphRecallSearch[] = "graph_recall_search";
static constexpr char kBenchMemoryCaseGraphSnapshotViewRecallSearch[] =
    "graph_snapshot_view_recall_search";
static constexpr char kBenchMemoryCaseGraphRecallSearchSweep[] = "graph_recall_search_sweep";
static constexpr char kBenchMemoryCaseGraphRecallDetail[] = "graph_recall_detail";
static constexpr char kBenchMemoryCaseGraphLevelStats[] = "graph_level_stats";
static constexpr char kBenchMemoryCaseGraphEdgeStats[] = "graph_edge_stats";
static constexpr char kBenchMemoryCaseCursorBeginFetch[] = "cursor_begin_fetch";
static constexpr char kBenchMemoryCaseMemoryStatus[] = "memory_status";
static constexpr const char* kBenchMemorySweepAddNames[] = {
    "features.memory add_batch_100",
    "features.memory add_batch_1k",
    "features.memory add_batch_10k",
    "features.memory add_batch_100k",
};
static constexpr const char* kBenchMemorySweepTopOneNames[] = {
    "features.memory top1_100",
    "features.memory top1_1k",
    "features.memory top1_10k",
    "features.memory top1_100k",
};
static constexpr const char* kBenchMemorySweepTopKNames[] = {
    "features.memory topk_100",
    "features.memory topk_1k",
    "features.memory topk_10k",
    "features.memory topk_100k",
};
static constexpr const char* kBenchMemorySweepCursorNames[] = {
    "features.memory cursor_100",
    "features.memory cursor_1k",
    "features.memory cursor_10k",
    "features.memory cursor_100k",
};
static constexpr uint32_t kBenchMemorySweepCapacities[] = {
    kBenchMemorySweepTiny,
    kBenchMemorySweepSmall,
    kBenchMemorySweepMedium,
    kBenchMemorySweepLarge,
};
static constexpr const char* kBenchToolManyNames[kBenchToolManyCount] = {
    "a000", "b001", "c002", "d003",
    "e004", "f005", "g006", "h007",
    "i008", "j009", "k010", "l011",
    "m012", "n013", "o014", "p015",
};

static uint32_t parse_u32_env(const char* key, uint32_t fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    unsigned long x = std::strtoul(v, &end, 10);
    if (end == v) {
        return fallback;
    }
    return static_cast<uint32_t>(x);
}

static uint64_t parse_u64_env(const char* key, uint64_t fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v) {
        return fallback;
    }
    return static_cast<uint64_t>(x);
}

static bool file_is_large_enough(const char* path, uint64_t min_bytes) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
    struct stat st {};
    if (stat(path, &st) != 0) {
        return false;
    }
    return static_cast<uint64_t>(st.st_size) >= min_bytes;
}

static bool env_enabled(const char* key) {
    const char* value = std::getenv(key);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, kDisabledEnvValue) != 0;
}

static bool memory_case_enabled(const char* name) {
    const char* value = std::getenv(kBenchMemoryCaseEnv);
    return value == nullptr || value[0] == '\0' || std::strcmp(value, name) == 0;
}

static uint32_t bounded_env_u32(const char* key, uint32_t fallback, uint32_t min_value, uint32_t max_value) {
    const uint32_t parsed = parse_u32_env(key, fallback);
    if (parsed < min_value) {
        return min_value;
    }
    if (parsed > max_value) {
        return max_value;
    }
    return parsed;
}

static void sort_ticks(std::vector<uint64_t>& values) {
  const size_t count = values.size();
  for (size_t i = 1; i < count; ++i) {
    const uint64_t value = values[i];
    size_t j = i;
    while (j > 0 && values[j - 1] > value) {
      values[j] = values[j - 1];
      --j;
    }
    values[j] = value;
  }
}

static uint64_t percentile_ticks(std::vector<uint64_t>& values, uint32_t percentile) {
  if (values.empty()) {
    return 0;
  }
  sort_ticks(values);
  const uint64_t scaled = static_cast<uint64_t>(values.size() - 1u) * percentile;
  const uint64_t scale = static_cast<uint64_t>(kBenchMemoryPercentScale);
  return values[static_cast<size_t>(scaled / scale)];
}

static AstralMemoryMetric parse_memory_metric_env() {
    const char* value = std::getenv(kBenchMemoryMetricEnv);
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, kBenchMemoryMetricCosine) == 0) {
        return ASTRAL_MEMORY_METRIC_COSINE;
    }
    if (std::strcmp(value, kBenchMemoryMetricDot) == 0) {
        return ASTRAL_MEMORY_METRIC_DOT;
    }
    if (std::strcmp(value, kBenchMemoryMetricL2) == 0) {
        return ASTRAL_MEMORY_METRIC_L2;
    }
    return ASTRAL_MEMORY_METRIC_COSINE;
}

static AstralMemoryStorageKind parse_memory_storage_env() {
    const char* value = std::getenv(kBenchMemoryStorageEnv);
    if (value != nullptr && std::strcmp(value, kBenchMemoryStorageQ8) == 0) {
        return ASTRAL_MEMORY_STORAGE_Q8;
    }
    if (value != nullptr && std::strcmp(value, kBenchMemoryStorageF6E2M3) == 0) {
      return ASTRAL_MEMORY_STORAGE_F6_E2M3;
    }
    if (value != nullptr && std::strcmp(value, kBenchMemoryStorageF8E5M2) == 0) {
      return ASTRAL_MEMORY_STORAGE_F8_E5M2;
    }
    return ASTRAL_MEMORY_STORAGE_F32;
}

static uint32_t memory_graph_neighbors() {
    return bounded_env_u32(
        kBenchMemoryGraphNeighborsEnv,
        kBenchMemoryGraphNeighbors,
        1u,
        kBenchMemoryGraphMaxNeighbors);
}

static uint32_t memory_graph_search() {
    return bounded_env_u32(kBenchMemoryGraphSearchEnv, kBenchMemoryGraphSearch, 4u, UINT32_MAX);
}

static uint32_t memory_graph_query_search() {
  const char* value = std::getenv(kBenchMemoryGraphQuerySearchEnv);
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  return bounded_env_u32(kBenchMemoryGraphQuerySearchEnv, kBenchMemoryGraphMinSearch,
                         kBenchMemoryGraphMinSearch, UINT32_MAX);
}

static uint32_t memory_recall_queries() {
  return bounded_env_u32(kBenchMemoryRecallQueriesEnv, kBenchMemoryRecallQueries,
                         kBenchMemoryMinRecallQueries, kBenchMemoryMaxRecallQueries);
}

static uint64_t memory_key_hash_mix(uint64_t x) {
  x ^= x >> kBenchMemoryKeyHashShift0;
  x *= kBenchMemoryKeyHashMul0;
  x ^= x >> kBenchMemoryKeyHashShift1;
  x *= kBenchMemoryKeyHashMul1;
  x ^= x >> kBenchMemoryKeyHashShift2;
  return x;
}

static uint32_t memory_graph_level_capacity(uint32_t capacity, uint32_t neighbors) {
  uint32_t levels = 1;
  uint32_t remaining = capacity;
  while (remaining >= neighbors && levels < kBenchMemoryGraphMaxLevels) {
    remaining /= neighbors;
    ++levels;
  }
  return levels;
}

static uint32_t memory_graph_level_for_key(uint64_t key, uint32_t level_capacity,
                                           uint32_t neighbors) {
  uint32_t level = 0;
  uint64_t hash = memory_key_hash_mix(key);
  const uint32_t threshold = UINT32_MAX / neighbors;
  while (level + 1u < level_capacity && static_cast<uint32_t>(hash) <= threshold) {
    ++level;
    hash = memory_key_hash_mix(hash + key);
  }
  return level;
}

static uint32_t memory_bench_dim() {
    return bounded_env_u32(kBenchMemoryDimEnv, kBenchMemoryDim, kBenchMemoryMinDim, kBenchMemoryMaxDim);
}

static uint32_t memory_bench_capacity(uint32_t fallback, uint32_t dim) {
    return bounded_env_u32(kBenchMemoryCapacityEnv, fallback, kBenchMemoryMinCapacity, UINT32_MAX / dim);
}

static uint32_t memory_bench_mix(uint32_t x) {
    x ^= x >> kBenchMemoryHashShift0;
    x *= kBenchMemoryHashMul0;
    x ^= x >> kBenchMemoryHashShift1;
    x *= kBenchMemoryHashMul1;
    x ^= x >> kBenchMemoryHashShift2;
    return x;
}

static float memory_bench_value(uint32_t row, uint32_t col) {
    const uint32_t mixed = memory_bench_mix(row * kBenchMemoryHashRowMul + col * kBenchMemoryHashColMul);
    return static_cast<float>(static_cast<int32_t>(mixed)) * kBenchMemoryI32Scale;
}

static const char* find_model_path() {
    const char* env = std::getenv("ASTRAL_BENCH_MODEL");
    if (env && env[0] != '\0') {
        if (file_is_large_enough(env, 10ull * 1024ull * 1024ull)) {
            return env;
        }
        std::fprintf(stderr, "[bench] ASTRAL_BENCH_MODEL set but file missing/too small: %s\n", env);
        return nullptr;
    }

    env = std::getenv("ASTRAL_TEST_MODEL");
    if (env && env[0] != '\0') {
        if (file_is_large_enough(env, 10ull * 1024ull * 1024ull)) {
            return env;
        }
        std::fprintf(stderr, "[bench] ASTRAL_TEST_MODEL set but file missing/too small: %s\n", env);
        return nullptr;
    }

    static const char* paths[] = {
        "tests/models/gpt2.Q2_K.gguf",
        "../tests/models/gpt2.Q2_K.gguf",
        "../../tests/models/gpt2.Q2_K.gguf",
        "../../../tests/models/gpt2.Q2_K.gguf",
        "../../../../tests/models/gpt2.Q2_K.gguf",
    };
    for (const char* p : paths) {
        if (file_is_large_enough(p, 10ull * 1024ull * 1024ull)) {
            return p;
        }
    }
    return nullptr;
}

static const char* find_embed_model_path_or_fallback(const char* fallback_model_path) {
    const char* env = std::getenv("ASTRAL_BENCH_EMBED_MODEL");
    if (env && env[0] != '\0') {
        if (file_is_large_enough(env, 5ull * 1024ull * 1024ull)) {
            return env;
        }
        std::fprintf(stderr, "[bench] ASTRAL_BENCH_EMBED_MODEL set but file missing/too small: %s\n", env);
    }

    // Prefer a small embedding GGUF if present locally.
    static const char* paths[] = {
        "tests/models/all-MiniLM-L6-v2-Q2_K.gguf",
        "../tests/models/all-MiniLM-L6-v2-Q2_K.gguf",
        "../../tests/models/all-MiniLM-L6-v2-Q2_K.gguf",
        "../../../tests/models/all-MiniLM-L6-v2-Q2_K.gguf",
        "../../../../tests/models/all-MiniLM-L6-v2-Q2_K.gguf",
    };
    for (const char* p : paths) {
        if (file_is_large_enough(p, 5ull * 1024ull * 1024ull)) {
            return p;
        }
    }

    return fallback_model_path;
}

static AstralSpanU8 span_from_cstr(const char* s) {
    AstralSpanU8 out{};
    out.data = reinterpret_cast<const uint8_t*>(s);
    out.len = s ? static_cast<uint32_t>(std::strlen(s)) : 0u;
    return out;
}

static AstralImageDesc make_bench_image(std::vector<uint8_t>& storage) {
    const uint32_t width = parse_u32_env("ASTRAL_BENCH_MEDIA_IMAGE_W", 224);
    const uint32_t height = parse_u32_env("ASTRAL_BENCH_MEDIA_IMAGE_H", 224);
    const uint32_t channels = 3;
    const uint64_t bytes = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * channels;
    storage.resize(static_cast<size_t>(bytes));
    for (uint64_t i = 0; i < bytes; ++i) {
        storage[static_cast<size_t>(i)] = static_cast<uint8_t>(i & 0xFFu);
    }

    AstralImageDesc img{};
    img.size = sizeof(AstralImageDesc);
    img.format = ASTRAL_IMAGE_FORMAT_RGB8;
    img.width = width;
    img.height = height;
    img.row_stride = 0;
    img.flags = 0;
    img.pixels.data = storage.data();
    img.pixels.len = static_cast<uint32_t>(storage.size());
    return img;
}

static AstralAudioDesc make_bench_audio(std::vector<uint8_t>& storage) {
    const uint32_t sample_rate = parse_u32_env("ASTRAL_BENCH_MEDIA_AUDIO_RATE", 16000);
    const uint32_t channels = parse_u32_env("ASTRAL_BENCH_MEDIA_AUDIO_CHANNELS", 1);
    const uint32_t frames = parse_u32_env("ASTRAL_BENCH_MEDIA_AUDIO_FRAMES", sample_rate);
    const uint64_t bytes = static_cast<uint64_t>(frames) * channels * sizeof(float);
    storage.resize(static_cast<size_t>(bytes));
    for (uint64_t i = 0; i < bytes; ++i) {
        storage[static_cast<size_t>(i)] = static_cast<uint8_t>((i * 3u) & 0xFFu);
    }

    AstralAudioDesc audio{};
    audio.size = sizeof(AstralAudioDesc);
    audio.format = ASTRAL_AUDIO_FORMAT_F32;
    audio.channels = channels;
    audio.sample_rate = sample_rate;
    audio.frame_count = frames;
    audio.samples.data = storage.data();
    audio.samples.len = static_cast<uint32_t>(storage.size());
    audio.flags = 0;
    return audio;
}

static AstralHandle load_model(const char* backend, const char* path, uint32_t gpu_layers, uint8_t embeddings_only) {
    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    desc.backend_name = span_from_cstr(backend);
    desc.model_path = span_from_cstr(path);
    desc.gpu_layers = gpu_layers;
    desc.n_ctx = 512;
    desc.n_batch = 128;
    desc.n_threads = parse_u32_env("ASTRAL_BENCH_MODEL_THREADS", 0);
    desc.embeddings_only = embeddings_only;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&desc, &model);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_model_load failed: %s (%s)\n",
                     astral_error_string(err), astral_last_error());
        return 0;
    }
    return model;
}

static BenchResult bench_embed_roundtrip(AstralHandle model, uint64_t iters) {
    BenchResult r{};
    r.name = "features.embed enqueue+collect";
    r.ops = iters;

    AstralHandle emb = 0;
    AstralErr err = astral_embed_create(model, &emb);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    if (err != ASTRAL_OK || dim == 0) {
        astral_embed_destroy(emb);
        r.ops = 0;
        return r;
    }

    const uint64_t bytes = static_cast<uint64_t>(dim) * sizeof(float);
    std::vector<uint8_t> buf(bytes);
    AstralMutSpanU8 out{};
    out.data = buf.data();
    out.len = static_cast<uint32_t>(buf.size());

    const char* text = "Once upon a time";
    const AstralSpanU8 text_span = span_from_cstr(text);

    // Warmup once.
    {
        uint64_t ticket = 0;
        if (astral_embed_enqueue(emb, text_span, &ticket) == ASTRAL_OK) {
            (void)astral_embed_collect(emb, ticket, out);
        }
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t ticket = 0;
        err = astral_embed_enqueue(emb, text_span, &ticket);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        err = astral_embed_collect(emb, ticket, out);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;

    astral_embed_destroy(emb);
    return r;
}

static BenchResult bench_tokenize_count(AstralHandle model, uint64_t iters) {
    BenchResult r{};
    r.name = "features.tokenize count";
    r.ops = iters;

    const AstralSpanU8 text = span_from_cstr("The quick brown fox jumps over the lazy dog.");
    uint32_t count = 0;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        const AstralErr err = astral_tokenize_count(model, text, 1, 0, &count);
        if (err != ASTRAL_OK || count == 0) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

static BenchResult bench_tokenize_batch(AstralHandle model, uint64_t iters) {
    BenchResult r{};
    r.name = "features.tokenize batch";
    r.ops = iters;

    AstralTokenizeRequest reqs[4]{};
    reqs[0].text = span_from_cstr("alpha");
    reqs[1].text = span_from_cstr("beta");
    reqs[2].text = span_from_cstr("gamma");
    reqs[3].text = span_from_cstr("delta");

    uint32_t offsets[5]{};
    uint32_t total = 0;
    AstralErr err = astral_tokenize_batch(model, reqs, 4, offsets, nullptr, 0, &total);
    if (err != ASTRAL_OK || total == 0) {
        r.ops = 0;
        return r;
    }

    std::vector<int32_t> tokens(total);
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_tokenize_batch(model, reqs, 4, offsets, tokens.data(), total, &total);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

static BenchResult bench_prompt_cache_get(uint64_t iters) {
    static constexpr uint32_t kEntryCount = 1024;
    static constexpr uint32_t kTokenBudgetKiB = 16;
    static constexpr uint32_t kTokenBudget = kTokenBudgetKiB * kBenchBytesPerKiB;
    static constexpr uint32_t kTokenCount = 8;
    static constexpr AstralHandle kModelHandle = 0x0100000100000001ull;
    static constexpr uint64_t kKeyBase = 0xA57A1000ull;

    BenchResult r{};
    r.name = "features.prompt_cache get";
    r.ops = iters;

    AstralPromptCacheDesc desc{};
    desc.size = sizeof(AstralPromptCacheDesc);
    desc.max_entries = kEntryCount;
    desc.max_tokens = kTokenBudget;
    desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;

    AstralHandle cache = 0;
    if (astral_prompt_cache_create(&desc, &cache) != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    int32_t tokens[kTokenCount]{};
    for (uint32_t i = 0; i < kTokenCount; ++i) {
        tokens[i] = static_cast<int32_t>(i + 1u);
    }

    AstralPromptCacheKey key{};
    key.size = sizeof(AstralPromptCacheKey);
    key.section_kind = ASTRAL_PROMPT_SECTION_SYSTEM;
    key.model = kModelHandle;
    key.generation = 1;

    for (uint32_t i = 0; i < kEntryCount; ++i) {
        key.key = kKeyBase + i;
        if (astral_prompt_cache_put_tokens(cache, &key, tokens, kTokenCount) != ASTRAL_OK) {
            astral_prompt_cache_destroy(cache);
            r.ops = 0;
            return r;
        }
    }

    int32_t out[kTokenCount]{};
    uint32_t count = 0;
    const uint32_t mask = kEntryCount - 1u;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        key.key = kKeyBase + (static_cast<uint32_t>(i) & mask);
        const AstralErr err = astral_prompt_cache_get_tokens(cache, &key, out, kTokenCount, &count);
        if (err != ASTRAL_OK || count != kTokenCount) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_prompt_cache_destroy(cache);
    return r;
}

static BenchResult bench_prompt_cache_view(uint64_t iters) {
    static constexpr uint32_t kEntryCount = 1024;
    static constexpr uint32_t kTokenBudgetKiB = 16;
    static constexpr uint32_t kTokenBudget = kTokenBudgetKiB * kBenchBytesPerKiB;
    static constexpr uint32_t kTokenCount = 8;
    static constexpr AstralHandle kModelHandle = 0x0100000100000001ull;
    static constexpr uint64_t kKeyBase = 0xA57A2000ull;

    BenchResult r{};
    r.name = "features.prompt_cache view";
    r.ops = iters;

    AstralPromptCacheDesc desc{};
    desc.size = sizeof(AstralPromptCacheDesc);
    desc.max_entries = kEntryCount;
    desc.max_tokens = kTokenBudget;
    desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;

    AstralHandle cache = 0;
    if (astral_prompt_cache_create(&desc, &cache) != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    int32_t tokens[kTokenCount]{};
    for (uint32_t i = 0; i < kTokenCount; ++i) {
        tokens[i] = static_cast<int32_t>(i + 1u);
    }

    AstralPromptCacheKey key{};
    key.size = sizeof(AstralPromptCacheKey);
    key.section_kind = ASTRAL_PROMPT_SECTION_SYSTEM;
    key.model = kModelHandle;
    key.generation = 1;

    for (uint32_t i = 0; i < kEntryCount; ++i) {
        key.key = kKeyBase + i;
        if (astral_prompt_cache_put_tokens(cache, &key, tokens, kTokenCount) != ASTRAL_OK) {
            astral_prompt_cache_destroy(cache);
            r.ops = 0;
            return r;
        }
    }

    const int32_t* out = nullptr;
    uint32_t count = 0;
    const uint32_t mask = kEntryCount - 1u;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        key.key = kKeyBase + (static_cast<uint32_t>(i) & mask);
        const AstralErr err = astral_prompt_cache_get_token_view(cache, &key, &out, &count);
        if (err != ASTRAL_OK || count != kTokenCount || out == nullptr) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_prompt_cache_destroy(cache);
    return r;
}

static BenchResult bench_prompt_cache_hot_view(uint64_t iters) {
    static constexpr uint32_t kEntryCount = 1024;
    static constexpr uint32_t kTokenBudgetKiB = 16;
    static constexpr uint32_t kTokenBudget = kTokenBudgetKiB * kBenchBytesPerKiB;
    static constexpr uint32_t kTokenCount = 8;
    static constexpr AstralHandle kModelHandle = 0x0100000100000001ull;
    static constexpr uint64_t kKeyBase = 0xA57A2500ull;
    static constexpr uint64_t kHotKey = kKeyBase + 17u;

    BenchResult r{};
    r.name = "features.prompt_cache hot_view";
    r.ops = iters;

    AstralPromptCacheDesc desc{};
    desc.size = sizeof(AstralPromptCacheDesc);
    desc.max_entries = kEntryCount;
    desc.max_tokens = kTokenBudget;
    desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;

    AstralHandle cache = 0;
    if (astral_prompt_cache_create(&desc, &cache) != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    int32_t tokens[kTokenCount]{};
    for (uint32_t i = 0; i < kTokenCount; ++i) {
        tokens[i] = static_cast<int32_t>(i + 1u);
    }

    AstralPromptCacheKey key{};
    key.size = sizeof(AstralPromptCacheKey);
    key.section_kind = ASTRAL_PROMPT_SECTION_SYSTEM;
    key.model = kModelHandle;
    key.generation = 1;

    for (uint32_t i = 0; i < kEntryCount; ++i) {
        key.key = kKeyBase + i;
        if (astral_prompt_cache_put_tokens(cache, &key, tokens, kTokenCount) != ASTRAL_OK) {
            astral_prompt_cache_destroy(cache);
            r.ops = 0;
            return r;
        }
    }

    key.key = kHotKey;
    const int32_t* out = nullptr;
    uint32_t count = 0;
    if (astral_prompt_cache_get_token_view(cache, &key, &out, &count) != ASTRAL_OK ||
        count != kTokenCount || out == nullptr) {
        astral_prompt_cache_destroy(cache);
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        const AstralErr err = astral_prompt_cache_get_token_view(cache, &key, &out, &count);
        if (err != ASTRAL_OK || count != kTokenCount || out == nullptr) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_prompt_cache_destroy(cache);
    return r;
}

static BenchResult bench_prompt_cache_miss(uint64_t iters) {
    static constexpr uint32_t kEntryCount = 1024;
    static constexpr uint32_t kTokenBudgetKiB = 16;
    static constexpr uint32_t kTokenBudget = kTokenBudgetKiB * kBenchBytesPerKiB;
    static constexpr uint32_t kTokenCount = 8;
    static constexpr AstralHandle kModelHandle = 0x0100000100000001ull;
    static constexpr uint64_t kKeyBase = 0xA57A4000ull;
    static constexpr uint64_t kMissingKeyBase = 0xA57A8000ull;
    static constexpr uint32_t kGeneration = 1;
    static constexpr uint32_t kTokenSeed = 1;

    BenchResult r{};
    r.name = "features.prompt_cache miss";
    r.ops = iters;

    AstralPromptCacheDesc desc{};
    desc.size = sizeof(AstralPromptCacheDesc);
    desc.max_entries = kEntryCount;
    desc.max_tokens = kTokenBudget;
    desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;

    AstralHandle cache = 0;
    if (astral_prompt_cache_create(&desc, &cache) != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    int32_t tokens[kTokenCount]{};
    for (uint32_t i = 0; i < kTokenCount; ++i) {
        tokens[i] = static_cast<int32_t>(i + kTokenSeed);
    }

    AstralPromptCacheKey key{};
    key.size = sizeof(AstralPromptCacheKey);
    key.section_kind = ASTRAL_PROMPT_SECTION_SYSTEM;
    key.model = kModelHandle;
    key.generation = kGeneration;

    for (uint32_t i = 0; i < kEntryCount; ++i) {
        key.key = kKeyBase + i;
        if (astral_prompt_cache_put_tokens(cache, &key, tokens, kTokenCount) != ASTRAL_OK) {
            astral_prompt_cache_destroy(cache);
            r.ops = 0;
            return r;
        }
    }

    int32_t out[kTokenCount]{};
    uint32_t count = 0;
    const uint32_t mask = kEntryCount - 1u;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        key.key = kMissingKeyBase + (static_cast<uint32_t>(i) & mask);
        const AstralErr err = astral_prompt_cache_get_tokens(cache, &key, out, kTokenCount, &count);
        if (err != ASTRAL_E_NOT_FOUND || count != 0) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_prompt_cache_destroy(cache);
    return r;
}

static BenchResult bench_prompt_cache_fifo_evict(uint64_t iters) {
    static constexpr uint32_t kEntryCount = 1024;
    static constexpr uint32_t kTokenBudgetKiB = 16;
    static constexpr uint32_t kTokenBudget = kTokenBudgetKiB * kBenchBytesPerKiB;
    static constexpr uint32_t kTokenCount = 8;
    static constexpr AstralHandle kModelHandle = 0x0100000100000001ull;
    static constexpr uint64_t kKeyBase = 0xA57A3000ull;
    static constexpr uint32_t kGeneration = 1;
    static constexpr uint32_t kTokenSeed = 1;

    BenchResult r{};
    r.name = "features.prompt_cache evict";
    r.ops = iters;

    AstralPromptCacheDesc desc{};
    desc.size = sizeof(AstralPromptCacheDesc);
    desc.max_entries = kEntryCount;
    desc.max_tokens = kTokenBudget;
    desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;

    AstralHandle cache = 0;
    if (astral_prompt_cache_create(&desc, &cache) != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    int32_t tokens[kTokenCount]{};
    for (uint32_t i = 0; i < kTokenCount; ++i) {
        tokens[i] = static_cast<int32_t>(i + kTokenSeed);
    }

    AstralPromptCacheKey key{};
    key.size = sizeof(AstralPromptCacheKey);
    key.section_kind = ASTRAL_PROMPT_SECTION_SYSTEM;
    key.model = kModelHandle;
    key.generation = kGeneration;

    for (uint32_t i = 0; i < kEntryCount; ++i) {
        key.key = kKeyBase + i;
        if (astral_prompt_cache_put_tokens(cache, &key, tokens, kTokenCount) != ASTRAL_OK) {
            astral_prompt_cache_destroy(cache);
            r.ops = 0;
            return r;
        }
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        key.key = kKeyBase + kEntryCount + i;
        const AstralErr err = astral_prompt_cache_put_tokens(cache, &key, tokens, kTokenCount);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_prompt_cache_destroy(cache);
    return r;
}

static BenchResult bench_prompt_cache_compacted_evict(uint64_t iters) {
    static constexpr uint32_t kEntryCount = 1024;
    static constexpr uint32_t kTokenBudgetKiB = 8;
    static constexpr uint32_t kTokenBudget = kTokenBudgetKiB * kBenchBytesPerKiB;
    static constexpr uint32_t kTokenCount = 8;
    static constexpr uint32_t kUpdatedTokenCount = 1;
    static constexpr AstralHandle kModelHandle = 0x0100000100000001ull;
    static constexpr uint64_t kKeyBase = 0xA57A9000ull;
    static constexpr uint32_t kGeneration = 1;
    static constexpr uint32_t kTokenSeed = 3;

    BenchResult r{};
    r.name = "features.prompt_cache compacted_evict";
    r.ops = iters;

    AstralPromptCacheDesc desc{};
    desc.size = sizeof(AstralPromptCacheDesc);
    desc.max_entries = kEntryCount;
    desc.max_tokens = kTokenBudget;
    desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;

    AstralHandle cache = 0;
    if (astral_prompt_cache_create(&desc, &cache) != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    int32_t tokens[kTokenCount]{};
    for (uint32_t i = 0; i < kTokenCount; ++i) {
        tokens[i] = static_cast<int32_t>(i + kTokenSeed);
    }

    AstralPromptCacheKey key{};
    key.size = sizeof(AstralPromptCacheKey);
    key.section_kind = ASTRAL_PROMPT_SECTION_SYSTEM;
    key.model = kModelHandle;
    key.generation = kGeneration;

    for (uint32_t i = 0; i < kEntryCount; ++i) {
        key.key = kKeyBase + i;
        if (astral_prompt_cache_put_tokens(cache, &key, tokens, kTokenCount) != ASTRAL_OK) {
            astral_prompt_cache_destroy(cache);
            r.ops = 0;
            return r;
        }
    }

    key.key = kKeyBase + (kEntryCount >> 1u);
    if (astral_prompt_cache_put_tokens(cache, &key, tokens, kUpdatedTokenCount) != ASTRAL_OK) {
        astral_prompt_cache_destroy(cache);
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        key.key = kKeyBase + kEntryCount + i;
        const AstralErr err = astral_prompt_cache_put_tokens(cache, &key, tokens, kTokenCount);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_prompt_cache_destroy(cache);
    return r;
}

static BenchResult bench_toolset_parse(uint64_t iters) {
    BenchResult r{};
    r.name = "features.toolset parse";
    r.ops = iters;

    AstralToolDesc tools[kBenchToolCount]{};
    tools[0].size = sizeof(AstralToolDesc);
    tools[0].tool_id = kBenchToolSearchId;
    tools[0].name = span_from_cstr("search");
    tools[0].description = span_from_cstr("Search indexed text");
    tools[0].json_schema = span_from_cstr("{\"type\":\"object\"}");
    tools[1].size = sizeof(AstralToolDesc);
    tools[1].tool_id = kBenchToolOpenId;
    tools[1].name = span_from_cstr("open");
    tools[1].description = span_from_cstr("Open one result");
    tools[1].json_schema = span_from_cstr("{\"type\":\"object\"}");

    AstralToolsetDesc desc{};
    desc.size = sizeof(AstralToolsetDesc);
    desc.tool_count = kBenchToolCount;
    desc.choice_mode = ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL;
    desc.tools = tools;

    AstralHandle toolset = 0;
    if (astral_toolset_create(&desc, &toolset) != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    const AstralSpanU8 text = span_from_cstr("{\"name\":\"search\",\"arguments\":{\"query\":\"latency\",\"k\":4}}");
    AstralToolCallResult call{};
    call.size = sizeof(AstralToolCallResult);

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        const AstralErr err = astral_toolset_parse_call(toolset, text, &call);
        if (err != ASTRAL_OK || call.tool_id != kBenchToolSearchId || call.parse_status != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_toolset_destroy(toolset);
    return r;
}

static BenchResult bench_toolset_parse_many(uint64_t iters) {
    BenchResult r{};
    r.name = "features.toolset parse_many";
    r.ops = iters;

    AstralToolDesc tools[kBenchToolManyCount]{};
    for (uint32_t i = 0; i < kBenchToolManyCount; ++i) {
        tools[i].size = sizeof(AstralToolDesc);
        tools[i].tool_id = kBenchToolManyIdBase + i;
        tools[i].name = span_from_cstr(kBenchToolManyNames[i]);
        tools[i].description = span_from_cstr("Synthetic lookup fixture");
        tools[i].json_schema = span_from_cstr("{\"type\":\"object\"}");
    }

    AstralToolsetDesc desc{};
    desc.size = sizeof(AstralToolsetDesc);
    desc.tool_count = kBenchToolManyCount;
    desc.choice_mode = ASTRAL_TOOL_CHOICE_TEXT_OR_TOOL;
    desc.tools = tools;

    AstralHandle toolset = 0;
    if (astral_toolset_create(&desc, &toolset) != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    const AstralSpanU8 text = span_from_cstr("{\"name\":\"p015\",\"arguments\":{\"query\":\"latency\",\"k\":4}}");
    AstralToolCallResult call{};
    call.size = sizeof(AstralToolCallResult);
    const uint32_t expected_id = kBenchToolManyIdBase + kBenchToolManyTargetIndex;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        const AstralErr err = astral_toolset_parse_call(toolset, text, &call);
        if (err != ASTRAL_OK || call.tool_id != expected_id || call.parse_status != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_toolset_destroy(toolset);
    return r;
}

static BenchResult bench_chunk_word_ranges(uint64_t iters) {
    BenchResult r{};
    r.name = "features.chunk word_ranges";
    r.ops = iters;

    const AstralSpanU8 text = span_from_cstr(
        "Astral keeps chunking native so RAG ingest can build deterministic byte ranges without copying strings. "
        "The wrappers can materialize text only for selected ranges while the core works on spans and caller buffers. "
        "This benchmark exercises the range generation path used by document ingest and memory search preparation."
    );

    AstralChunkerDesc desc{};
    desc.size = sizeof(AstralChunkerDesc);
    desc.mode = ASTRAL_CHUNK_MODE_WORD;
    desc.max_units = kBenchChunkMaxWords;
    desc.overlap_units = kBenchChunkOverlapWords;

    AstralChunkRange ranges[kBenchChunkRangeCapacity]{};
    uint32_t count = 0;
    AstralErr err = astral_chunk_ranges(&desc, text, ranges, kBenchChunkRangeCapacity, &count);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_chunk_ranges(&desc, text, ranges, kBenchChunkRangeCapacity, &count);
        if (err != ASTRAL_OK || count == 0) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

static BenchResult bench_chunk_token_count(uint64_t iters) {
    BenchResult r{};
    r.name = "features.chunk token_count";
    r.ops = iters;

    AstralChunkerDesc desc{};
    desc.size = sizeof(AstralChunkerDesc);
    desc.mode = ASTRAL_CHUNK_MODE_TOKEN;
    desc.max_units = kBenchChunkMaxTokens;
    desc.overlap_units = kBenchChunkOverlapTokens;

    uint32_t count = 0;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        const AstralErr err = astral_token_chunk_count(&desc, kBenchChunkTokenCount, &count);
        if (err != ASTRAL_OK || count == 0) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

static BenchResult bench_chunk_token_ranges(uint64_t iters) {
    BenchResult r{};
    r.name = "features.chunk token_ranges";
    r.ops = iters;

    AstralChunkerDesc desc{};
    desc.size = sizeof(AstralChunkerDesc);
    desc.mode = ASTRAL_CHUNK_MODE_TOKEN;
    desc.max_units = kBenchChunkMaxTokens;
    desc.overlap_units = kBenchChunkOverlapTokens;

    AstralChunkRange ranges[kBenchChunkTokenRangeCapacity]{};
    uint32_t count = 0;
    AstralErr err = astral_token_chunk_ranges(&desc, kBenchChunkTokenCount, ranges, kBenchChunkTokenRangeCapacity, &count);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_token_chunk_ranges(&desc, kBenchChunkTokenCount, ranges, kBenchChunkTokenRangeCapacity, &count);
        if (err != ASTRAL_OK || count == 0) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

static void fill_memory_fixture(std::vector<AstralMemoryRecord>& records,
                                std::vector<float>& vectors,
                                uint32_t capacity,
                                uint32_t dim) {
    for (uint32_t row = 0; row < capacity; ++row) {
        records[row].size = sizeof(AstralMemoryRecord);
        records[row].key = static_cast<uint64_t>(row) + kBenchMemoryKeyBase;
        records[row].group_id = row & kBenchMemoryGroupMask;
        records[row].document_id = row >> kBenchMemoryDocShift;
        records[row].chunk_id = row;
        for (uint32_t col = 0; col < dim; ++col) {
            vectors[static_cast<size_t>(row) * dim + col] = memory_bench_value(row, col);
        }
    }
}

static void fill_memory_query(std::vector<float>& query) {
    for (uint32_t i = 0; i < static_cast<uint32_t>(query.size()); ++i) {
        query[i] = memory_bench_value(0, i);
    }
}

static void fill_memory_query(float* query, uint32_t dim, uint32_t query_index) {
    for (uint32_t i = 0; i < dim; ++i) {
        query[i] = memory_bench_value(query_index, i);
    }
}

static void fill_memory_query(std::vector<float>& query, uint32_t query_index) {
    fill_memory_query(query.data(), static_cast<uint32_t>(query.size()), query_index);
}

static BenchResult bench_memory_add_batch_impl(uint32_t capacity, const char* name) {
    BenchResult r{};
    r.name = name;
    const uint32_t dim = memory_bench_dim();
    r.ops = capacity;
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_destroy(index);
    return r;
}

static BenchResult bench_memory_add_batch(uint64_t iters) {
    (void)iters;
    const uint32_t dim = memory_bench_dim();
    return bench_memory_add_batch_impl(memory_bench_capacity(kBenchMemoryCapacity, dim), "features.memory add_batch");
}

static BenchResult bench_memory_graph_add_batch(uint64_t iters) {
    (void)iters;
    BenchResult r{};
    r.name = "features.memory graph_add_batch";
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    r.ops = capacity;
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
    desc.graph_neighbors = memory_graph_neighbors();
    desc.graph_search = memory_graph_search();
    desc.graph_query_search = memory_graph_query_search();
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_destroy(index);
    return r;
}

static LatencyResult bench_memory_graph_add_latency(uint64_t iters) {
  (void)iters;
  LatencyResult r{};
  r.name = "features.memory graph_add_latency";
  r.tick_to_ns = clock_info().tick_to_ns;
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc desc{};
  desc.size = sizeof(AstralMemoryIndexDesc);
  desc.dim = dim;
  desc.capacity = capacity;
  desc.metric = metric;
  desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
  desc.graph_neighbors = memory_graph_neighbors();
  desc.graph_search = memory_graph_search();
  desc.graph_query_search = memory_graph_query_search();
  desc.storage_kind = parse_memory_storage_env();

  AstralHandle index = 0;
  AstralErr err = astral_memory_create(&desc, &index);
  if (err != ASTRAL_OK) {
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  std::vector<uint64_t> samples;
  samples.reserve(capacity);
  fill_memory_fixture(records, vectors, capacity, dim);

  for (uint32_t i = 0; i < capacity; ++i) {
    const uint64_t t0 = ticks_now();
    err = astral_memory_add_batch(index, &records[i], vectors.data() + static_cast<size_t>(i) * dim,
                                  1);
    const uint64_t t1 = ticks_now();
    if (err != ASTRAL_OK) {
      break;
    }
    samples.push_back(t1 - t0);
  }

  if (!samples.empty()) {
    uint64_t max_ticks = 0;
    for (uint64_t sample : samples) {
      if (sample > max_ticks) {
        max_ticks = sample;
      }
    }
    r.max_ticks = max_ticks;
    r.p50_ticks = percentile_ticks(samples, 50);
    r.p95_ticks = percentile_ticks(samples, 95);
    r.p99_ticks = percentile_ticks(samples, 99);
  }

  astral_memory_destroy(index);
  return r;
}

static BenchResult bench_memory_graph_load(uint64_t iters) {
  BenchResult r{};
  r.name = "features.memory graph_load";
  r.ops = iters;
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc desc{};
  desc.size = sizeof(AstralMemoryIndexDesc);
  desc.dim = dim;
  desc.capacity = capacity;
  desc.metric = metric;
  desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
  desc.graph_neighbors = memory_graph_neighbors();
  desc.graph_search = memory_graph_search();
  desc.graph_query_search = memory_graph_query_search();
  desc.storage_kind = parse_memory_storage_env();

  AstralHandle index = 0;
  AstralErr err = astral_memory_create(&desc, &index);
  if (err != ASTRAL_OK) {
    r.ops = 0;
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }

  uint64_t save_bytes = 0;
  err = astral_memory_save_size(index, &save_bytes);
  if (err != ASTRAL_OK || save_bytes > UINT32_MAX) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }
  std::vector<uint8_t> blob(static_cast<size_t>(save_bytes));
  AstralMutSpanU8 out{};
  out.data = blob.data();
  out.len = static_cast<uint32_t>(blob.size());
  uint64_t written = 0;
  err = astral_memory_save(index, out, &written);
  astral_memory_destroy(index);
  if (err != ASTRAL_OK || written != save_bytes) {
    r.ops = 0;
    return r;
  }

  AstralSpanU8 span{};
  span.data = blob.data();
  span.len = static_cast<uint32_t>(blob.size());
  const uint64_t t0 = ticks_now();
  const uint64_t n0 = ns_now();
  for (uint64_t i = 0; i < iters; ++i) {
    AstralHandle loaded = 0;
    err = astral_memory_load(&desc, span, &loaded);
    if (err != ASTRAL_OK) {
      r.ops = i;
      break;
    }
    astral_memory_destroy(loaded);
  }
  const uint64_t t1 = ticks_now();
  const uint64_t n1 = ns_now();
  r.ticks = t1 - t0;
  r.ns = n1 - n0;
  return r;
}

static BenchResult bench_memory_snapshot_search(uint64_t iters) {
  BenchResult r{};
  r.name = "features.memory snapshot_search";
  r.ops = iters;
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc desc{};
  desc.size = sizeof(AstralMemoryIndexDesc);
  desc.dim = dim;
  desc.capacity = capacity;
  desc.metric = metric;
  desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
  desc.storage_kind = parse_memory_storage_env();

  AstralHandle index = 0;
  AstralErr err = astral_memory_create(&desc, &index);
  if (err != ASTRAL_OK) {
    r.ops = 0;
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }

  uint64_t save_bytes = 0;
  err = astral_memory_save_size(index, &save_bytes);
  if (err != ASTRAL_OK || save_bytes > UINT32_MAX) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }
  std::vector<uint8_t> blob(static_cast<size_t>(save_bytes));
  AstralMutSpanU8 out{};
  out.data = blob.data();
  out.len = static_cast<uint32_t>(blob.size());
  uint64_t written = 0;
  err = astral_memory_save(index, out, &written);
  astral_memory_destroy(index);
  if (err != ASTRAL_OK || written != save_bytes) {
    r.ops = 0;
    return r;
  }

  std::vector<float> query(dim);
  fill_memory_query(query);
  AstralMemorySearchDesc search{};
  search.size = sizeof(AstralMemorySearchDesc);
  search.top_k = kBenchMemoryTopK;
  search.group_id = ASTRAL_MEMORY_GROUP_ANY;
  AstralSpanU8 span{};
  span.data = blob.data();
  span.len = static_cast<uint32_t>(blob.size());
  AstralMemorySearchResult results[kBenchMemoryTopK]{};
  uint32_t result_count = 0;

  const uint64_t t0 = ticks_now();
  const uint64_t n0 = ns_now();
  for (uint64_t i = 0; i < iters; ++i) {
    err = astral_memory_snapshot_search(span, &search, query.data(), results, kBenchMemoryTopK,
                                        &result_count);
    if (err != ASTRAL_OK || result_count == 0) {
      r.ops = i;
      break;
    }
  }
  const uint64_t t1 = ticks_now();
  const uint64_t n1 = ns_now();

  r.ticks = t1 - t0;
  r.ns = n1 - n0;
  return r;
}

static BenchResult bench_memory_snapshot_view_search(uint64_t iters) {
  BenchResult r{};
  r.name = "features.memory snapshot_view_search";
  r.ops = iters;
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc desc{};
  desc.size = sizeof(AstralMemoryIndexDesc);
  desc.dim = dim;
  desc.capacity = capacity;
  desc.metric = metric;
  desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
  desc.storage_kind = parse_memory_storage_env();

  AstralHandle index = 0;
  AstralErr err = astral_memory_create(&desc, &index);
  if (err != ASTRAL_OK) {
    r.ops = 0;
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }

  uint64_t save_bytes = 0;
  err = astral_memory_save_size(index, &save_bytes);
  if (err != ASTRAL_OK || save_bytes > UINT32_MAX) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }
  std::vector<uint8_t> blob(static_cast<size_t>(save_bytes));
  AstralMutSpanU8 out{};
  out.data = blob.data();
  out.len = static_cast<uint32_t>(blob.size());
  uint64_t written = 0;
  err = astral_memory_save(index, out, &written);
  astral_memory_destroy(index);
  if (err != ASTRAL_OK || written != save_bytes) {
    r.ops = 0;
    return r;
  }

  char snapshot_path[kBenchMemorySnapshotPathBytes]{};
  std::snprintf(snapshot_path, sizeof(snapshot_path), "/tmp/astral-bench-memory-view-%p.bin",
                static_cast<const void*>(blob.data()));
  FILE* snapshot_file = std::fopen(snapshot_path, "wb");
  if (snapshot_file == nullptr) {
    r.ops = 0;
    return r;
  }
  const size_t bytes_written = std::fwrite(blob.data(), 1, blob.size(), snapshot_file);
  const int close_result = std::fclose(snapshot_file);
  if (bytes_written != blob.size() || close_result != 0) {
    std::remove(snapshot_path);
    r.ops = 0;
    return r;
  }

  AstralMemorySnapshotInfo mapped_info{};
  mapped_info.size = sizeof(AstralMemorySnapshotInfo);
  AstralHandle mapped_view = 0;
  AstralSpanU8 path_span{};
  path_span.data = reinterpret_cast<const uint8_t*>(snapshot_path);
  path_span.len = static_cast<uint32_t>(std::strlen(snapshot_path));
  err = astral_memory_snapshot_map(path_span, &mapped_info, &mapped_view);
  if (err != ASTRAL_OK) {
    std::remove(snapshot_path);
    r.ops = 0;
    return r;
  }

  std::vector<float> query(dim);
  fill_memory_query(query);
  AstralMemorySearchDesc search{};
  search.size = sizeof(AstralMemorySearchDesc);
  search.top_k = kBenchMemoryTopK;
  search.group_id = ASTRAL_MEMORY_GROUP_ANY;
  AstralMemorySearchResult results[kBenchMemoryTopK]{};
  uint32_t result_count = 0;

  const uint64_t t0 = ticks_now();
  const uint64_t n0 = ns_now();
  for (uint64_t i = 0; i < iters; ++i) {
    err = astral_memory_snapshot_view_search(mapped_view, &search, query.data(), results,
                                             kBenchMemoryTopK, &result_count);
    if (err != ASTRAL_OK || result_count == 0) {
      r.ops = i;
      break;
    }
  }
  const uint64_t t1 = ticks_now();
  const uint64_t n1 = ns_now();

  astral_memory_snapshot_unmap(mapped_view);
  std::remove(snapshot_path);
  r.ticks = t1 - t0;
  r.ns = n1 - n0;
  return r;
}

static BenchResult bench_memory_graph_snapshot_view_search(uint64_t iters) {
  BenchResult r{};
  r.name = "features.memory graph_snapshot_view_search";
  r.ops = iters;
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc desc{};
  desc.size = sizeof(AstralMemoryIndexDesc);
  desc.dim = dim;
  desc.capacity = capacity;
  desc.metric = metric;
  desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
  desc.graph_neighbors = memory_graph_neighbors();
  desc.graph_search = memory_graph_search();
  desc.graph_query_search = memory_graph_query_search();
  desc.storage_kind = parse_memory_storage_env();

  AstralHandle index = 0;
  AstralErr err = astral_memory_create(&desc, &index);
  if (err != ASTRAL_OK) {
    r.ops = 0;
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }

  uint64_t save_bytes = 0;
  err = astral_memory_save_size(index, &save_bytes);
  if (err != ASTRAL_OK || save_bytes > UINT32_MAX) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }
  std::vector<uint8_t> blob(static_cast<size_t>(save_bytes));
  AstralMutSpanU8 out{};
  out.data = blob.data();
  out.len = static_cast<uint32_t>(blob.size());
  uint64_t written = 0;
  err = astral_memory_save(index, out, &written);
  astral_memory_destroy(index);
  if (err != ASTRAL_OK || written != save_bytes) {
    r.ops = 0;
    return r;
  }

  char snapshot_path[kBenchMemorySnapshotPathBytes]{};
  std::snprintf(snapshot_path, sizeof(snapshot_path), "/tmp/astral-bench-memory-graph-view-%p.bin",
                static_cast<const void*>(blob.data()));
  FILE* snapshot_file = std::fopen(snapshot_path, "wb");
  if (snapshot_file == nullptr) {
    r.ops = 0;
    return r;
  }
  const size_t bytes_written = std::fwrite(blob.data(), 1, blob.size(), snapshot_file);
  const int close_result = std::fclose(snapshot_file);
  if (bytes_written != blob.size() || close_result != 0) {
    std::remove(snapshot_path);
    r.ops = 0;
    return r;
  }

  AstralMemorySnapshotInfo mapped_info{};
  mapped_info.size = sizeof(AstralMemorySnapshotInfo);
  AstralHandle mapped_view = 0;
  AstralSpanU8 path_span{};
  path_span.data = reinterpret_cast<const uint8_t*>(snapshot_path);
  path_span.len = static_cast<uint32_t>(std::strlen(snapshot_path));
  err = astral_memory_snapshot_map(path_span, &mapped_info, &mapped_view);
  if (err != ASTRAL_OK || mapped_info.index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    if (astral_handle_valid(mapped_view)) {
      astral_memory_snapshot_unmap(mapped_view);
    }
    std::remove(snapshot_path);
    r.ops = 0;
    return r;
  }

  std::vector<float> query(dim);
  fill_memory_query(query);
  AstralMemorySearchDesc search{};
  search.size = sizeof(AstralMemorySearchDesc);
  search.top_k = kBenchMemoryTopK;
  search.group_id = ASTRAL_MEMORY_GROUP_ANY;
  search.graph_search = memory_graph_query_search();
  AstralMemorySearchResult results[kBenchMemoryTopK]{};
  uint32_t result_count = 0;

  const uint64_t t0 = ticks_now();
  const uint64_t n0 = ns_now();
  for (uint64_t i = 0; i < iters; ++i) {
    err = astral_memory_snapshot_view_search(mapped_view, &search, query.data(), results,
                                             kBenchMemoryTopK, &result_count);
    if (err != ASTRAL_OK || result_count == 0) {
      r.ops = i;
      break;
    }
  }
  const uint64_t t1 = ticks_now();
  const uint64_t n1 = ns_now();

  astral_memory_snapshot_unmap(mapped_view);
  std::remove(snapshot_path);
  r.ticks = t1 - t0;
  r.ns = n1 - n0;
  return r;
}

static BenchResult bench_memory_flat_search_impl(uint64_t iters, uint32_t capacity, const char* name) {
    BenchResult r{};
    r.name = name;
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    std::vector<float> query(dim);
    fill_memory_query(query);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    search.graph_search = memory_graph_query_search();
    AstralMemorySearchResult results[kBenchMemoryTopK]{};
    uint32_t result_count = 0;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_memory_search(index, &search, query.data(), results, kBenchMemoryTopK, &result_count);
        if (err != ASTRAL_OK || result_count == 0) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_destroy(index);
    return r;
}

static BenchResult bench_memory_flat_search(uint64_t iters) {
    const uint32_t dim = memory_bench_dim();
    return bench_memory_flat_search_impl(iters, memory_bench_capacity(kBenchMemoryCapacity, dim), "features.memory flat_search");
}

static BenchResult bench_memory_flat_search_batch(uint64_t iters) {
    BenchResult r{};
    r.name = "features.memory flat_search_batch";
    r.ops = iters * kBenchMemoryBatchQueries;
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    std::vector<float> queries(static_cast<size_t>(kBenchMemoryBatchQueries) * dim);
    for (uint32_t i = 0; i < kBenchMemoryBatchQueries; ++i) {
        fill_memory_query(queries.data() + static_cast<size_t>(i) * dim, dim, i);
    }
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    search.graph_search = memory_graph_query_search();
    AstralMemorySearchResult results[kBenchMemoryBatchQueries * kBenchMemoryTopK]{};
    uint32_t counts[kBenchMemoryBatchQueries]{};

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_memory_search_batch(
            index,
            &search,
            queries.data(),
            kBenchMemoryBatchQueries,
            results,
            kBenchMemoryBatchQueries * kBenchMemoryTopK,
            counts);
        if (err != ASTRAL_OK || counts[0] == 0) {
            r.ops = i * kBenchMemoryBatchQueries;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_destroy(index);
    return r;
}

static BenchResult bench_memory_graph_search(uint64_t iters) {
    BenchResult r{};
    r.name = "features.memory graph_search";
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
    desc.graph_neighbors = memory_graph_neighbors();
    desc.graph_search = memory_graph_search();
    desc.graph_query_search = memory_graph_query_search();
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    std::vector<float> query(dim);
    fill_memory_query(query);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    AstralMemorySearchResult results[kBenchMemoryTopK]{};
    uint32_t result_count = 0;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_memory_search(index, &search, query.data(), results, kBenchMemoryTopK, &result_count);
        if (err != ASTRAL_OK || result_count == 0) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_destroy(index);
    return r;
}

static BenchResult bench_memory_graph_search_batch(uint64_t iters) {
  BenchResult r{};
  r.name = "features.memory graph_search_batch";
  r.ops = iters * kBenchMemoryBatchQueries;
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc desc{};
  desc.size = sizeof(AstralMemoryIndexDesc);
  desc.dim = dim;
  desc.capacity = capacity;
  desc.metric = metric;
  desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
  desc.graph_neighbors = memory_graph_neighbors();
  desc.graph_search = memory_graph_search();
  desc.graph_query_search = memory_graph_query_search();
  desc.storage_kind = parse_memory_storage_env();

  AstralHandle index = 0;
  AstralErr err = astral_memory_create(&desc, &index);
  if (err != ASTRAL_OK) {
    r.ops = 0;
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(index);
    r.ops = 0;
    return r;
  }

  std::vector<float> queries(static_cast<size_t>(kBenchMemoryBatchQueries) * dim);
  for (uint32_t i = 0; i < kBenchMemoryBatchQueries; ++i) {
    fill_memory_query(queries.data() + static_cast<size_t>(i) * dim, dim, i);
  }
  AstralMemorySearchDesc search{};
  search.size = sizeof(AstralMemorySearchDesc);
  search.top_k = kBenchMemoryTopK;
  search.group_id = ASTRAL_MEMORY_GROUP_ANY;
  search.graph_search = memory_graph_query_search();
  AstralMemorySearchResult results[kBenchMemoryBatchQueries * kBenchMemoryTopK]{};
  uint32_t counts[kBenchMemoryBatchQueries]{};

  const uint64_t t0 = ticks_now();
  const uint64_t n0 = ns_now();
  for (uint64_t i = 0; i < iters; ++i) {
    err = astral_memory_search_batch(index, &search, queries.data(), kBenchMemoryBatchQueries,
                                     results, kBenchMemoryBatchQueries * kBenchMemoryTopK, counts);
    if (err != ASTRAL_OK || counts[0] == 0) {
      r.ops = i * kBenchMemoryBatchQueries;
      break;
    }
  }
  const uint64_t t1 = ticks_now();
  const uint64_t n1 = ns_now();

  r.ticks = t1 - t0;
  r.ns = n1 - n0;
  astral_memory_destroy(index);
  return r;
}

static LatencyResult bench_memory_search_latency(uint64_t iters, bool graph) {
  LatencyResult r{};
  r.name = graph ? "features.memory graph_search_latency" : "features.memory flat_search_latency";
  r.tick_to_ns = clock_info().tick_to_ns;

  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc desc{};
  desc.size = sizeof(AstralMemoryIndexDesc);
  desc.dim = dim;
  desc.capacity = capacity;
  desc.metric = metric;
  desc.index_kind = graph ? ASTRAL_MEMORY_INDEX_GRAPH : ASTRAL_MEMORY_INDEX_FLAT;
  desc.graph_neighbors = graph ? memory_graph_neighbors() : 0u;
  desc.graph_search = graph ? memory_graph_search() : 0u;
  desc.graph_query_search = graph ? memory_graph_query_search() : 0u;
  desc.storage_kind = parse_memory_storage_env();

  AstralHandle index = 0;
  AstralErr err = astral_memory_create(&desc, &index);
  if (err != ASTRAL_OK) {
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(index);
    return r;
  }

  std::vector<float> query(dim);
  fill_memory_query(query);
  AstralMemorySearchDesc search{};
  search.size = sizeof(AstralMemorySearchDesc);
  search.top_k = kBenchMemoryTopK;
  search.group_id = ASTRAL_MEMORY_GROUP_ANY;
  search.graph_search = memory_graph_query_search();
  AstralMemorySearchResult results[kBenchMemoryTopK]{};
  uint32_t result_count = 0;
  err =
      astral_memory_search(index, &search, query.data(), results, kBenchMemoryTopK, &result_count);
  if (err != ASTRAL_OK || result_count == 0) {
    astral_memory_destroy(index);
    return r;
  }

  std::vector<uint64_t> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (uint64_t i = 0; i < iters; ++i) {
    const uint64_t t0 = ticks_now();
    err = astral_memory_search(index, &search, query.data(), results, kBenchMemoryTopK,
                               &result_count);
    const uint64_t t1 = ticks_now();
    if (err != ASTRAL_OK || result_count == 0) {
      break;
    }
    samples.push_back(t1 - t0);
  }

  if (!samples.empty()) {
    uint64_t max_ticks = 0;
    for (uint64_t sample : samples) {
      if (sample > max_ticks) {
        max_ticks = sample;
      }
    }
    r.max_ticks = max_ticks;
    r.p50_ticks = percentile_ticks(samples, 50);
    r.p95_ticks = percentile_ticks(samples, 95);
    r.p99_ticks = percentile_ticks(samples, 99);
  }

  astral_memory_destroy(index);
  return r;
}

static BenchResult bench_memory_graph_top1(uint64_t iters) {
    BenchResult r{};
    r.name = "features.memory graph_top1";
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
    desc.graph_neighbors = memory_graph_neighbors();
    desc.graph_search = memory_graph_search();
    desc.graph_query_search = memory_graph_query_search();
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    std::vector<float> query(dim);
    fill_memory_query(query);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopOne;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    search.graph_search = memory_graph_query_search();
    AstralMemorySearchResult result{};
    uint32_t result_count = 0;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_memory_search(index, &search, query.data(), &result, kBenchMemoryTopOne, &result_count);
        if (err != ASTRAL_OK || result_count == 0) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_destroy(index);
    return r;
}

static BenchResult bench_memory_graph_recall(uint64_t iters) {
    BenchResult r{};
    r.name = "features.memory graph_recall";
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    const uint32_t recall_queries = memory_recall_queries();
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc flat_desc{};
    flat_desc.size = sizeof(AstralMemoryIndexDesc);
    flat_desc.dim = dim;
    flat_desc.capacity = capacity;
    flat_desc.metric = metric;
    flat_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

    AstralMemoryIndexDesc graph_desc = flat_desc;
    graph_desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
    graph_desc.graph_neighbors = memory_graph_neighbors();
    graph_desc.graph_search = memory_graph_search();
    graph_desc.graph_query_search = memory_graph_query_search();
    graph_desc.storage_kind = parse_memory_storage_env();

    AstralHandle flat_index = 0;
    AstralHandle graph_index = 0;
    AstralErr err = astral_memory_create(&flat_desc, &flat_index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }
    err = astral_memory_create(&graph_desc, &graph_index);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(flat_index);
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(flat_index, records.data(), vectors.data(), capacity);
    if (err == ASTRAL_OK) {
        err = astral_memory_add_batch(graph_index, records.data(), vectors.data(), capacity);
    }
    if (err != ASTRAL_OK) {
        astral_memory_destroy(graph_index);
        astral_memory_destroy(flat_index);
        r.ops = 0;
        return r;
    }

    std::vector<float> query(dim);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    search.graph_search = memory_graph_query_search();
    AstralMemorySearchResult flat_results[kBenchMemoryTopK]{};
    AstralMemorySearchResult graph_results[kBenchMemoryTopK]{};
    uint32_t flat_count = 0;
    uint32_t graph_count = 0;
    uint64_t matched = 0;
    uint64_t expected = 0;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        const uint32_t query_row =
            static_cast<uint32_t>((static_cast<uint64_t>(i % recall_queries) * capacity) /
                                  static_cast<uint64_t>(recall_queries));
        fill_memory_query(query, query_row);
        err = astral_memory_search(flat_index, &search, query.data(), flat_results, kBenchMemoryTopK, &flat_count);
        if (err == ASTRAL_OK) {
            err = astral_memory_search(graph_index, &search, query.data(), graph_results, kBenchMemoryTopK, &graph_count);
        }
        if (err != ASTRAL_OK || flat_count == 0 || graph_count == 0) {
            r.ops = i;
            break;
        }
        expected += flat_count;
        for (uint32_t fi = 0; fi < flat_count; ++fi) {
            for (uint32_t gi = 0; gi < graph_count; ++gi) {
                if (flat_results[fi].key == graph_results[gi].key) {
                    ++matched;
                    break;
                }
            }
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    r.extra_label = "recall_pct";
    r.extra_value = expected != 0 ? (static_cast<double>(matched) * kBenchMemoryPercentScale) /
                                        static_cast<double>(expected)
                                  : 0.0;
    astral_memory_destroy(graph_index);
    astral_memory_destroy(flat_index);
    return r;
}

static BenchResult bench_memory_flat_compact_recall_search(uint64_t iters,
                                                           AstralMemoryStorageKind storage_kind,
                                                           const char* name) {
  BenchResult r{};
  r.name = name;
  r.ops = iters;
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const uint32_t recall_queries = memory_recall_queries();
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc flat_desc{};
  flat_desc.size = sizeof(AstralMemoryIndexDesc);
  flat_desc.dim = dim;
  flat_desc.capacity = capacity;
  flat_desc.metric = metric;
  flat_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
  flat_desc.storage_kind = ASTRAL_MEMORY_STORAGE_F32;

  AstralMemoryIndexDesc compact_desc = flat_desc;
  compact_desc.storage_kind = storage_kind;

  AstralHandle flat_index = 0;
  AstralHandle compact_index = 0;
  AstralErr err = astral_memory_create(&flat_desc, &flat_index);
  if (err != ASTRAL_OK) {
    r.ops = 0;
    return r;
  }
  err = astral_memory_create(&compact_desc, &compact_index);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(flat_index, records.data(), vectors.data(), capacity);
  if (err == ASTRAL_OK) {
    err = astral_memory_add_batch(compact_index, records.data(), vectors.data(), capacity);
  }
  if (err != ASTRAL_OK) {
    astral_memory_destroy(compact_index);
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }

  std::vector<float> queries(static_cast<size_t>(recall_queries) * dim);
  std::vector<uint64_t> oracle_keys(static_cast<size_t>(recall_queries) * kBenchMemoryTopK);
  std::vector<uint32_t> oracle_counts(recall_queries);
  AstralMemorySearchDesc search{};
  search.size = sizeof(AstralMemorySearchDesc);
  search.top_k = kBenchMemoryTopK;
  search.group_id = ASTRAL_MEMORY_GROUP_ANY;
  AstralMemorySearchResult flat_results[kBenchMemoryTopK]{};
  AstralMemorySearchResult compact_results[kBenchMemoryTopK]{};
  uint32_t flat_count = 0;
  uint32_t compact_count = 0;

  for (uint32_t qi = 0; qi < recall_queries; ++qi) {
    const uint32_t query_row = static_cast<uint32_t>((static_cast<uint64_t>(qi) * capacity) /
                                                     static_cast<uint64_t>(recall_queries));
    float* query = queries.data() + static_cast<size_t>(qi) * dim;
    fill_memory_query(query, dim, query_row);
    err = astral_memory_search(flat_index, &search, query, flat_results, kBenchMemoryTopK,
                               &flat_count);
    if (err != ASTRAL_OK || flat_count == 0) {
      astral_memory_destroy(compact_index);
      astral_memory_destroy(flat_index);
      r.ops = 0;
      return r;
    }
    oracle_counts[qi] = flat_count;
    for (uint32_t fi = 0; fi < flat_count; ++fi) {
      oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi] = flat_results[fi].key;
    }
  }

  uint64_t matched = 0;
  uint64_t expected = 0;
  const uint64_t t0 = ticks_now();
  const uint64_t n0 = ns_now();
  for (uint64_t i = 0; i < iters; ++i) {
    const uint32_t qi = static_cast<uint32_t>(i % recall_queries);
    const float* query = queries.data() + static_cast<size_t>(qi) * dim;
    err = astral_memory_search(compact_index, &search, query, compact_results, kBenchMemoryTopK,
                               &compact_count);
    if (err != ASTRAL_OK || compact_count == 0) {
      r.ops = i;
      break;
    }
    const uint32_t expected_count = oracle_counts[qi];
    expected += expected_count;
    for (uint32_t fi = 0; fi < expected_count; ++fi) {
      const uint64_t key = oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi];
      for (uint32_t ci = 0; ci < compact_count; ++ci) {
        if (key == compact_results[ci].key) {
          ++matched;
          break;
        }
      }
    }
  }
  const uint64_t t1 = ticks_now();
  const uint64_t n1 = ns_now();

  r.ticks = t1 - t0;
  r.ns = n1 - n0;
  r.extra_label = "recall_pct";
  r.extra_value = expected != 0 ? (static_cast<double>(matched) * kBenchMemoryPercentScale) /
                                      static_cast<double>(expected)
                                : 0.0;
  astral_memory_destroy(compact_index);
  astral_memory_destroy(flat_index);
  return r;
}

static BenchResult bench_memory_flat_q8_recall_search(uint64_t iters) {
  return bench_memory_flat_compact_recall_search(iters, ASTRAL_MEMORY_STORAGE_Q8,
                                                 "features.memory flat_q8_recall_search");
}

static BenchResult bench_memory_flat_compact_recall_search(uint64_t iters) {
  return bench_memory_flat_compact_recall_search(iters, parse_memory_storage_env(),
                                                 "features.memory flat_compact_recall_search");
}

static BenchResult bench_memory_graph_recall_search(uint64_t iters) {
    BenchResult r{};
    r.name = "features.memory graph_recall_search";
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    const uint32_t recall_queries = memory_recall_queries();
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc flat_desc{};
    flat_desc.size = sizeof(AstralMemoryIndexDesc);
    flat_desc.dim = dim;
    flat_desc.capacity = capacity;
    flat_desc.metric = metric;
    flat_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

    AstralMemoryIndexDesc graph_desc = flat_desc;
    graph_desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
    graph_desc.graph_neighbors = memory_graph_neighbors();
    graph_desc.graph_search = memory_graph_search();
    graph_desc.graph_query_search = memory_graph_query_search();
    graph_desc.storage_kind = parse_memory_storage_env();

    AstralHandle flat_index = 0;
    AstralHandle graph_index = 0;
    AstralErr err = astral_memory_create(&flat_desc, &flat_index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }
    err = astral_memory_create(&graph_desc, &graph_index);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(flat_index);
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(flat_index, records.data(), vectors.data(), capacity);
    if (err == ASTRAL_OK) {
        err = astral_memory_add_batch(graph_index, records.data(), vectors.data(), capacity);
    }
    if (err != ASTRAL_OK) {
        astral_memory_destroy(graph_index);
        astral_memory_destroy(flat_index);
        r.ops = 0;
        return r;
    }

    std::vector<float> queries(static_cast<size_t>(recall_queries) * dim);
    std::vector<uint64_t> oracle_keys(static_cast<size_t>(recall_queries) * kBenchMemoryTopK);
    std::vector<uint32_t> oracle_counts(recall_queries);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    search.graph_search = memory_graph_query_search();
    AstralMemorySearchResult flat_results[kBenchMemoryTopK]{};
    AstralMemorySearchResult graph_results[kBenchMemoryTopK]{};
    uint32_t flat_count = 0;
    uint32_t graph_count = 0;

    for (uint32_t qi = 0; qi < recall_queries; ++qi) {
        const uint32_t query_row =
            static_cast<uint32_t>((static_cast<uint64_t>(qi) * capacity) /
                                  static_cast<uint64_t>(recall_queries));
        float* query = queries.data() + static_cast<size_t>(qi) * dim;
        fill_memory_query(query, dim, query_row);
        err = astral_memory_search(flat_index, &search, query, flat_results, kBenchMemoryTopK, &flat_count);
        if (err != ASTRAL_OK || flat_count == 0) {
            astral_memory_destroy(graph_index);
            astral_memory_destroy(flat_index);
            r.ops = 0;
            return r;
        }
        oracle_counts[qi] = flat_count;
        for (uint32_t fi = 0; fi < flat_count; ++fi) {
            oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi] = flat_results[fi].key;
        }
    }

    uint64_t matched = 0;
    uint64_t expected = 0;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        const uint32_t qi = static_cast<uint32_t>(i % recall_queries);
        const float* query = queries.data() + static_cast<size_t>(qi) * dim;
        err = astral_memory_search(graph_index, &search, query, graph_results, kBenchMemoryTopK, &graph_count);
        if (err != ASTRAL_OK || graph_count == 0) {
            r.ops = i;
            break;
        }
        const uint32_t expected_count = oracle_counts[qi];
        expected += expected_count;
        for (uint32_t fi = 0; fi < expected_count; ++fi) {
            const uint64_t key = oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi];
            for (uint32_t gi = 0; gi < graph_count; ++gi) {
                if (key == graph_results[gi].key) {
                    ++matched;
                    break;
                }
            }
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    r.extra_label = "recall_pct";
    r.extra_value = expected != 0 ? (static_cast<double>(matched) * kBenchMemoryPercentScale) /
                                        static_cast<double>(expected)
                                  : 0.0;
    astral_memory_destroy(graph_index);
    astral_memory_destroy(flat_index);
    return r;
}

static BenchResult bench_memory_graph_snapshot_view_recall_search(uint64_t iters) {
  BenchResult r{};
  r.name = "features.memory graph_snapshot_view_recall_search";
  r.ops = iters;
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const uint32_t recall_queries = memory_recall_queries();
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc flat_desc{};
  flat_desc.size = sizeof(AstralMemoryIndexDesc);
  flat_desc.dim = dim;
  flat_desc.capacity = capacity;
  flat_desc.metric = metric;
  flat_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

  AstralMemoryIndexDesc graph_desc = flat_desc;
  graph_desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
  graph_desc.graph_neighbors = memory_graph_neighbors();
  graph_desc.graph_search = memory_graph_search();
  graph_desc.graph_query_search = memory_graph_query_search();
  graph_desc.storage_kind = parse_memory_storage_env();

  AstralHandle flat_index = 0;
  AstralHandle graph_index = 0;
  AstralErr err = astral_memory_create(&flat_desc, &flat_index);
  if (err != ASTRAL_OK) {
    r.ops = 0;
    return r;
  }
  err = astral_memory_create(&graph_desc, &graph_index);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(flat_index, records.data(), vectors.data(), capacity);
  if (err == ASTRAL_OK) {
    err = astral_memory_add_batch(graph_index, records.data(), vectors.data(), capacity);
  }
  if (err != ASTRAL_OK) {
    astral_memory_destroy(graph_index);
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }

  uint64_t save_bytes = 0;
  err = astral_memory_save_size(graph_index, &save_bytes);
  if (err != ASTRAL_OK || save_bytes > UINT32_MAX) {
    astral_memory_destroy(graph_index);
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }
  std::vector<uint8_t> blob(static_cast<size_t>(save_bytes));
  AstralMutSpanU8 out{};
  out.data = blob.data();
  out.len = static_cast<uint32_t>(blob.size());
  uint64_t written = 0;
  err = astral_memory_save(graph_index, out, &written);
  astral_memory_destroy(graph_index);
  graph_index = 0;
  if (err != ASTRAL_OK || written != save_bytes) {
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }

  char snapshot_path[kBenchMemorySnapshotPathBytes]{};
  std::snprintf(snapshot_path, sizeof(snapshot_path),
                "/tmp/astral-bench-memory-graph-recall-view-%p.bin",
                static_cast<const void*>(blob.data()));
  FILE* snapshot_file = std::fopen(snapshot_path, "wb");
  if (snapshot_file == nullptr) {
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }
  const size_t bytes_written = std::fwrite(blob.data(), 1, blob.size(), snapshot_file);
  const int close_result = std::fclose(snapshot_file);
  if (bytes_written != blob.size() || close_result != 0) {
    std::remove(snapshot_path);
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }

  AstralMemorySnapshotInfo mapped_info{};
  mapped_info.size = sizeof(AstralMemorySnapshotInfo);
  AstralHandle mapped_view = 0;
  AstralSpanU8 path_span{};
  path_span.data = reinterpret_cast<const uint8_t*>(snapshot_path);
  path_span.len = static_cast<uint32_t>(std::strlen(snapshot_path));
  err = astral_memory_snapshot_map(path_span, &mapped_info, &mapped_view);
  if (err != ASTRAL_OK || mapped_info.index_kind != ASTRAL_MEMORY_INDEX_GRAPH) {
    if (astral_handle_valid(mapped_view)) {
      astral_memory_snapshot_unmap(mapped_view);
    }
    std::remove(snapshot_path);
    astral_memory_destroy(flat_index);
    r.ops = 0;
    return r;
  }

  std::vector<float> queries(static_cast<size_t>(recall_queries) * dim);
  std::vector<uint64_t> oracle_keys(static_cast<size_t>(recall_queries) * kBenchMemoryTopK);
  std::vector<uint32_t> oracle_counts(recall_queries);
  AstralMemorySearchDesc search{};
  search.size = sizeof(AstralMemorySearchDesc);
  search.top_k = kBenchMemoryTopK;
  search.group_id = ASTRAL_MEMORY_GROUP_ANY;
  search.graph_search = memory_graph_query_search();
  AstralMemorySearchResult flat_results[kBenchMemoryTopK]{};
  AstralMemorySearchResult graph_results[kBenchMemoryTopK]{};
  uint32_t flat_count = 0;
  uint32_t graph_count = 0;

  for (uint32_t qi = 0; qi < recall_queries; ++qi) {
    const uint32_t query_row = static_cast<uint32_t>((static_cast<uint64_t>(qi) * capacity) /
                                                     static_cast<uint64_t>(recall_queries));
    float* query = queries.data() + static_cast<size_t>(qi) * dim;
    fill_memory_query(query, dim, query_row);
    err = astral_memory_search(flat_index, &search, query, flat_results, kBenchMemoryTopK,
                               &flat_count);
    if (err != ASTRAL_OK || flat_count == 0) {
      astral_memory_snapshot_unmap(mapped_view);
      std::remove(snapshot_path);
      astral_memory_destroy(flat_index);
      r.ops = 0;
      return r;
    }
    oracle_counts[qi] = flat_count;
    for (uint32_t fi = 0; fi < flat_count; ++fi) {
      oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi] = flat_results[fi].key;
    }
  }

  uint64_t matched = 0;
  uint64_t expected = 0;
  const uint64_t t0 = ticks_now();
  const uint64_t n0 = ns_now();
  for (uint64_t i = 0; i < iters; ++i) {
    const uint32_t qi = static_cast<uint32_t>(i % recall_queries);
    const float* query = queries.data() + static_cast<size_t>(qi) * dim;
    err = astral_memory_snapshot_view_search(mapped_view, &search, query, graph_results,
                                             kBenchMemoryTopK, &graph_count);
    if (err != ASTRAL_OK || graph_count == 0) {
      r.ops = i;
      break;
    }
    const uint32_t expected_count = oracle_counts[qi];
    expected += expected_count;
    for (uint32_t fi = 0; fi < expected_count; ++fi) {
      const uint64_t key = oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi];
      for (uint32_t gi = 0; gi < graph_count; ++gi) {
        if (key == graph_results[gi].key) {
          ++matched;
          break;
        }
      }
    }
  }
  const uint64_t t1 = ticks_now();
  const uint64_t n1 = ns_now();

  r.ticks = t1 - t0;
  r.ns = n1 - n0;
  r.extra_label = "recall_pct";
  r.extra_value = expected != 0 ? (static_cast<double>(matched) * kBenchMemoryPercentScale) /
                                      static_cast<double>(expected)
                                : 0.0;
  astral_memory_snapshot_unmap(mapped_view);
  std::remove(snapshot_path);
  astral_memory_destroy(flat_index);
  return r;
}

static BenchResult bench_memory_graph_recall_top1(uint64_t iters) {
    BenchResult r{};
    r.name = "features.memory graph_recall_top1";
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    const uint32_t recall_queries = memory_recall_queries();
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc flat_desc{};
    flat_desc.size = sizeof(AstralMemoryIndexDesc);
    flat_desc.dim = dim;
    flat_desc.capacity = capacity;
    flat_desc.metric = metric;
    flat_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

    AstralMemoryIndexDesc graph_desc = flat_desc;
    graph_desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
    graph_desc.graph_neighbors = memory_graph_neighbors();
    graph_desc.graph_search = memory_graph_search();
    graph_desc.graph_query_search = memory_graph_query_search();
    graph_desc.storage_kind = parse_memory_storage_env();

    AstralHandle flat_index = 0;
    AstralHandle graph_index = 0;
    AstralErr err = astral_memory_create(&flat_desc, &flat_index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }
    err = astral_memory_create(&graph_desc, &graph_index);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(flat_index);
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(flat_index, records.data(), vectors.data(), capacity);
    if (err == ASTRAL_OK) {
        err = astral_memory_add_batch(graph_index, records.data(), vectors.data(), capacity);
    }
    if (err != ASTRAL_OK) {
        astral_memory_destroy(graph_index);
        astral_memory_destroy(flat_index);
        r.ops = 0;
        return r;
    }

    std::vector<float> queries(static_cast<size_t>(recall_queries) * dim);
    std::vector<uint64_t> oracle_keys(recall_queries);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopOne;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    search.graph_search = memory_graph_query_search();
    AstralMemorySearchResult flat_result{};
    AstralMemorySearchResult graph_result{};
    uint32_t flat_count = 0;
    uint32_t graph_count = 0;

    for (uint32_t qi = 0; qi < recall_queries; ++qi) {
        const uint32_t query_row =
            static_cast<uint32_t>((static_cast<uint64_t>(qi) * capacity) /
                                  static_cast<uint64_t>(recall_queries));
        float* query = queries.data() + static_cast<size_t>(qi) * dim;
        fill_memory_query(query, dim, query_row);
        err = astral_memory_search(flat_index, &search, query, &flat_result, kBenchMemoryTopOne, &flat_count);
        if (err != ASTRAL_OK || flat_count == 0) {
            astral_memory_destroy(graph_index);
            astral_memory_destroy(flat_index);
            r.ops = 0;
            return r;
        }
        oracle_keys[qi] = flat_result.key;
    }

    uint64_t matched = 0;
    uint64_t expected = 0;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        const uint32_t qi = static_cast<uint32_t>(i % recall_queries);
        const float* query = queries.data() + static_cast<size_t>(qi) * dim;
        err = astral_memory_search(graph_index, &search, query, &graph_result, kBenchMemoryTopOne, &graph_count);
        if (err != ASTRAL_OK || graph_count == 0) {
            r.ops = i;
            break;
        }
        ++expected;
        if (graph_result.key == oracle_keys[qi]) {
            ++matched;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    r.extra_label = "top1_recall_pct";
    r.extra_value = expected != 0 ? (static_cast<double>(matched) * kBenchMemoryPercentScale) /
                                        static_cast<double>(expected)
                                  : 0.0;
    astral_memory_destroy(graph_index);
    astral_memory_destroy(flat_index);
    return r;
}

static void print_memory_graph_recall_search_sweep(uint64_t iters) {
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    const uint32_t recall_queries = memory_recall_queries();
    const uint32_t max_graph_search = memory_graph_search();
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc flat_desc{};
    flat_desc.size = sizeof(AstralMemoryIndexDesc);
    flat_desc.dim = dim;
    flat_desc.capacity = capacity;
    flat_desc.metric = metric;
    flat_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

    AstralMemoryIndexDesc graph_desc = flat_desc;
    graph_desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
    graph_desc.graph_neighbors = memory_graph_neighbors();
    graph_desc.graph_search = max_graph_search;
    graph_desc.graph_query_search = max_graph_search;
    graph_desc.storage_kind = parse_memory_storage_env();

    AstralHandle flat_index = 0;
    AstralHandle graph_index = 0;
    AstralErr err = astral_memory_create(&flat_desc, &flat_index);
    if (err != ASTRAL_OK) {
        return;
    }
    err = astral_memory_create(&graph_desc, &graph_index);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(flat_index);
        return;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(flat_index, records.data(), vectors.data(), capacity);
    if (err == ASTRAL_OK) {
        err = astral_memory_add_batch(graph_index, records.data(), vectors.data(), capacity);
    }
    if (err != ASTRAL_OK) {
        astral_memory_destroy(graph_index);
        astral_memory_destroy(flat_index);
        return;
    }

    std::vector<float> queries(static_cast<size_t>(recall_queries) * dim);
    std::vector<uint64_t> oracle_keys(static_cast<size_t>(recall_queries) * kBenchMemoryTopK);
    std::vector<uint32_t> oracle_counts(recall_queries);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    AstralMemorySearchResult flat_results[kBenchMemoryTopK]{};
    AstralMemorySearchResult graph_results[kBenchMemoryTopK]{};
    uint32_t flat_count = 0;
    uint32_t graph_count = 0;

    for (uint32_t qi = 0; qi < recall_queries; ++qi) {
        const uint32_t query_row =
            static_cast<uint32_t>((static_cast<uint64_t>(qi) * capacity) /
                                  static_cast<uint64_t>(recall_queries));
        float* query = queries.data() + static_cast<size_t>(qi) * dim;
        fill_memory_query(query, dim, query_row);
        err = astral_memory_search(flat_index, &search, query, flat_results, kBenchMemoryTopK, &flat_count);
        if (err != ASTRAL_OK || flat_count == 0) {
            astral_memory_destroy(graph_index);
            astral_memory_destroy(flat_index);
            return;
        }
        oracle_counts[qi] = flat_count;
        for (uint32_t fi = 0; fi < flat_count; ++fi) {
            oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi] = flat_results[fi].key;
        }
    }

    for (uint32_t budget = kBenchMemoryGraphMinSearch; budget <= max_graph_search; budget <<= 1u) {
        search.graph_search = budget;
        BenchResult r{};
        char name[64];
        std::snprintf(name, sizeof(name), "features.memory graph_recall_s%u", budget);
        r.name = name;
        r.ops = iters;
        uint64_t matched = 0;
        uint64_t expected = 0;
        const uint64_t t0 = ticks_now();
        const uint64_t n0 = ns_now();
        for (uint64_t i = 0; i < iters; ++i) {
            const uint32_t qi = static_cast<uint32_t>(i % recall_queries);
            const float* query = queries.data() + static_cast<size_t>(qi) * dim;
            err = astral_memory_search(graph_index, &search, query, graph_results, kBenchMemoryTopK, &graph_count);
            if (err != ASTRAL_OK || graph_count == 0) {
                r.ops = i;
                break;
            }
            const uint32_t expected_count = oracle_counts[qi];
            expected += expected_count;
            for (uint32_t fi = 0; fi < expected_count; ++fi) {
                const uint64_t key = oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi];
                for (uint32_t gi = 0; gi < graph_count; ++gi) {
                    if (key == graph_results[gi].key) {
                        ++matched;
                        break;
                    }
                }
            }
        }
        const uint64_t t1 = ticks_now();
        const uint64_t n1 = ns_now();
        r.ticks = t1 - t0;
        r.ns = n1 - n0;
        r.extra_label = "recall_pct";
        r.extra_value = expected != 0 ? (static_cast<double>(matched) * kBenchMemoryPercentScale) /
                                            static_cast<double>(expected)
                                      : 0.0;
        print_result(r, clock_info().name);
        if (budget > max_graph_search / 2u) {
            break;
        }
    }

    astral_memory_destroy(graph_index);
    astral_memory_destroy(flat_index);
}

static void print_memory_graph_recall_detail(uint64_t iters) {
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const uint32_t recall_queries = memory_recall_queries();
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc flat_desc{};
  flat_desc.size = sizeof(AstralMemoryIndexDesc);
  flat_desc.dim = dim;
  flat_desc.capacity = capacity;
  flat_desc.metric = metric;
  flat_desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;

  AstralMemoryIndexDesc graph_desc = flat_desc;
  graph_desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
  graph_desc.graph_neighbors = memory_graph_neighbors();
  graph_desc.graph_search = memory_graph_search();
  graph_desc.graph_query_search = memory_graph_query_search();
  graph_desc.storage_kind = parse_memory_storage_env();

  AstralHandle flat_index = 0;
  AstralHandle graph_index = 0;
  AstralErr err = astral_memory_create(&flat_desc, &flat_index);
  if (err != ASTRAL_OK) {
    return;
  }
  err = astral_memory_create(&graph_desc, &graph_index);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(flat_index);
    return;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(flat_index, records.data(), vectors.data(), capacity);
  if (err == ASTRAL_OK) {
    err = astral_memory_add_batch(graph_index, records.data(), vectors.data(), capacity);
  }
  if (err != ASTRAL_OK) {
    astral_memory_destroy(graph_index);
    astral_memory_destroy(flat_index);
    return;
  }

  std::vector<float> queries(static_cast<size_t>(recall_queries) * dim);
  std::vector<uint64_t> oracle_keys(static_cast<size_t>(recall_queries) * kBenchMemoryTopK);
  std::vector<uint32_t> oracle_counts(recall_queries);
  AstralMemorySearchDesc search{};
  search.size = sizeof(AstralMemorySearchDesc);
  search.top_k = kBenchMemoryTopK;
  search.group_id = ASTRAL_MEMORY_GROUP_ANY;
  search.graph_search = memory_graph_query_search();
  AstralMemorySearchResult flat_results[kBenchMemoryTopK]{};
  AstralMemorySearchResult graph_results[kBenchMemoryTopK]{};
  uint32_t flat_count = 0;
  uint32_t graph_count = 0;

  for (uint32_t qi = 0; qi < recall_queries; ++qi) {
    const uint32_t query_row = static_cast<uint32_t>((static_cast<uint64_t>(qi) * capacity) /
                                                     static_cast<uint64_t>(recall_queries));
    float* query = queries.data() + static_cast<size_t>(qi) * dim;
    fill_memory_query(query, dim, query_row);
    err = astral_memory_search(flat_index, &search, query, flat_results, kBenchMemoryTopK,
                               &flat_count);
    if (err != ASTRAL_OK || flat_count == 0) {
      astral_memory_destroy(graph_index);
      astral_memory_destroy(flat_index);
      return;
    }
    oracle_counts[qi] = flat_count;
    for (uint32_t fi = 0; fi < flat_count; ++fi) {
      oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi] = flat_results[fi].key;
    }
  }

  for (uint32_t qi = 0; qi < recall_queries; ++qi) {
    BenchResult r{};
    char name[kBenchMemoryRecallNameBytes];
    const uint32_t query_row = static_cast<uint32_t>((static_cast<uint64_t>(qi) * capacity) /
                                                     static_cast<uint64_t>(recall_queries));
    std::snprintf(name, sizeof(name), "features.memory graph_recall_q%03u_row%06u", qi, query_row);
    r.name = name;
    r.ops = iters;
    const float* query = queries.data() + static_cast<size_t>(qi) * dim;
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
      err = astral_memory_search(graph_index, &search, query, graph_results, kBenchMemoryTopK,
                                 &graph_count);
      if (err != ASTRAL_OK || graph_count == 0) {
        r.ops = i;
        break;
      }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();
    uint32_t matched = 0;
    const uint32_t expected_count = oracle_counts[qi];
    for (uint32_t fi = 0; fi < expected_count; ++fi) {
      const uint64_t key = oracle_keys[static_cast<size_t>(qi) * kBenchMemoryTopK + fi];
      for (uint32_t gi = 0; gi < graph_count; ++gi) {
        if (key == graph_results[gi].key) {
          ++matched;
          break;
        }
      }
    }
    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    r.extra_label = "recall_pct";
    r.extra_value = expected_count != 0
                        ? (static_cast<double>(matched) * kBenchMemoryPercentScale) /
                              static_cast<double>(expected_count)
                        : 0.0;
    print_result(r, clock_info().name);
    BenchResult match_result{};
    char match_name[kBenchMemoryRecallNameBytes];
    std::snprintf(match_name, sizeof(match_name),
                  "features.memory graph_recall_match_q%03u_row%06u", qi, query_row);
    match_result.name = match_name;
    match_result.ops = expected_count;
    match_result.extra_label = "matched_count";
    match_result.extra_value = static_cast<double>(matched);
    print_result(match_result, clock_info().name);
  }

  astral_memory_destroy(graph_index);
  astral_memory_destroy(flat_index);
}

static void print_memory_graph_level_stats() {
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const uint32_t neighbors = memory_graph_neighbors();
  const uint32_t level_capacity = memory_graph_level_capacity(capacity, neighbors);
  std::vector<uint32_t> level_counts(level_capacity);
  for (uint32_t row = 0; row < capacity; ++row) {
    const uint64_t key = static_cast<uint64_t>(row) + kBenchMemoryKeyBase;
    const uint32_t level = memory_graph_level_for_key(key, level_capacity, neighbors);
    ++level_counts[level];
  }

  for (uint32_t level = 0; level < level_capacity; ++level) {
    uint32_t nodes_at_or_above = 0;
    for (uint32_t candidate_level = level; candidate_level < level_capacity; ++candidate_level) {
      nodes_at_or_above += level_counts[candidate_level];
    }
    BenchResult r{};
    char name[kBenchMemoryRecallNameBytes];
    std::snprintf(name, sizeof(name), "features.memory graph_level_%02u", level);
    r.name = name;
    r.ops = capacity;
    r.extra_label = "node_count";
    r.extra_value = static_cast<double>(nodes_at_or_above);
    print_result(r, clock_info().name);
  }
}

static void print_memory_graph_edge_stats() {
  const uint32_t dim = memory_bench_dim();
  const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
  const AstralMemoryMetric metric = parse_memory_metric_env();

  AstralMemoryIndexDesc desc{};
  desc.size = sizeof(AstralMemoryIndexDesc);
  desc.dim = dim;
  desc.capacity = capacity;
  desc.metric = metric;
  desc.index_kind = ASTRAL_MEMORY_INDEX_GRAPH;
  desc.graph_neighbors = memory_graph_neighbors();
  desc.graph_search = memory_graph_search();
  desc.graph_query_search = memory_graph_query_search();
  desc.storage_kind = parse_memory_storage_env();

  AstralHandle index = 0;
  AstralErr err = astral_memory_create(&desc, &index);
  if (err != ASTRAL_OK) {
    return;
  }

  std::vector<AstralMemoryRecord> records(capacity);
  std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
  fill_memory_fixture(records, vectors, capacity, dim);
  err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(index);
    return;
  }

  AstralMemoryStats stats{};
  stats.size = sizeof(AstralMemoryStats);
  err = astral_memory_stats(index, &stats);
  if (err != ASTRAL_OK) {
    astral_memory_destroy(index);
    return;
  }

  struct EdgeStat {
    const char* name;
    const char* label;
    uint64_t value;
  };
  const EdgeStat edge_stats[] = {
      {"features.memory graph_edges", "edge_count", stats.graph_edges},
      {"features.memory graph_base_edges", "edge_count", stats.graph_base_edges},
      {"features.memory graph_upper_edges", "edge_count", stats.graph_upper_edges},
      {"features.memory graph_build_score_evals", "score_eval_count",
       stats.graph_build_score_evals},
      {"features.memory graph_build_candidate_visits", "candidate_visit_count",
       stats.graph_build_candidate_visits},
  };
  for (const EdgeStat& stat : edge_stats) {
    BenchResult r{};
    r.name = stat.name;
    r.ops = capacity;
    r.extra_label = stat.label;
    r.extra_value = static_cast<double>(stat.value);
    print_result(r, clock_info().name);
  }
  astral_memory_destroy(index);
}

static BenchResult bench_memory_flat_search_top1_impl(uint64_t iters, uint32_t capacity, const char* name) {
    BenchResult r{};
    r.name = name;
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    std::vector<float> query(dim);
    fill_memory_query(query);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopOne;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    AstralMemorySearchResult results[kBenchMemoryTopOne]{};
    uint32_t result_count = 0;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_memory_search(index, &search, query.data(), results, kBenchMemoryTopOne, &result_count);
        if (err != ASTRAL_OK || result_count != kBenchMemoryTopOne) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_destroy(index);
    return r;
}

static BenchResult bench_memory_flat_search_top1(uint64_t iters) {
    const uint32_t dim = memory_bench_dim();
    return bench_memory_flat_search_top1_impl(
        iters,
        memory_bench_capacity(kBenchMemoryCapacity, dim),
        "features.memory flat_search_top1"
    );
}

static BenchResult bench_memory_cursor_fetch_impl(uint64_t iters, uint32_t capacity, const char* name) {
    BenchResult r{};
    r.name = name;
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    std::vector<float> query(dim);
    fill_memory_query(query);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;
    AstralMemorySearchResult results[kBenchMemoryFetchK]{};
    uint32_t result_count = 0;

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        AstralHandle cursor = 0;
        err = astral_memory_search_begin(index, &search, query.data(), &cursor);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        err = astral_memory_search_fetch(cursor, results, kBenchMemoryFetchK, &result_count);
        astral_memory_search_end(cursor);
        if (err != ASTRAL_OK || result_count != kBenchMemoryFetchK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_destroy(index);
    return r;
}

static BenchResult bench_memory_cursor_fetch(uint64_t iters) {
    const uint32_t dim = memory_bench_dim();
    return bench_memory_cursor_fetch_impl(
        iters,
        memory_bench_capacity(kBenchMemoryCapacity, dim),
        "features.memory cursor_begin_fetch"
    );
}

static BenchResult bench_memory_request_status(uint64_t iters) {
    BenchResult r{};
    r.name = "features.request memory_status";
    r.ops = iters;
    const uint32_t dim = memory_bench_dim();
    const uint32_t capacity = memory_bench_capacity(kBenchMemoryCapacity, dim);
    const AstralMemoryMetric metric = parse_memory_metric_env();

    AstralMemoryIndexDesc desc{};
    desc.size = sizeof(AstralMemoryIndexDesc);
    desc.dim = dim;
    desc.capacity = capacity;
    desc.metric = metric;
    desc.index_kind = ASTRAL_MEMORY_INDEX_FLAT;
    desc.storage_kind = parse_memory_storage_env();

    AstralHandle index = 0;
    AstralErr err = astral_memory_create(&desc, &index);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    std::vector<AstralMemoryRecord> records(capacity);
    std::vector<float> vectors(static_cast<size_t>(capacity) * dim);
    fill_memory_fixture(records, vectors, capacity, dim);
    err = astral_memory_add_batch(index, records.data(), vectors.data(), capacity);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    std::vector<float> query(dim);
    fill_memory_query(query);
    AstralMemorySearchDesc search{};
    search.size = sizeof(AstralMemorySearchDesc);
    search.top_k = kBenchMemoryTopK;
    search.group_id = ASTRAL_MEMORY_GROUP_ANY;

    AstralHandle cursor = 0;
    err = astral_memory_search_begin(index, &search, query.data(), &cursor);
    if (err != ASTRAL_OK) {
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    AstralRequestRef request{};
    err = astral_request_from_memory_search(cursor, &request);
    if (err != ASTRAL_OK) {
        astral_memory_search_end(cursor);
        astral_memory_destroy(index);
        r.ops = 0;
        return r;
    }

    AstralRequestStatus status{};
    status.size = sizeof(AstralRequestStatus);
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        status.size = sizeof(AstralRequestStatus);
        err = astral_request_state(&request, &status);
        if (err != ASTRAL_OK || status.state != ASTRAL_REQUEST_COMPLETED) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_memory_search_end(cursor);
    astral_memory_destroy(index);
    return r;
}

static void print_memory_benchmarks(uint64_t iters) {
    if (!env_enabled(kBenchMemorySweepEnv)) {
        if (memory_case_enabled(kBenchMemoryCaseAddBatch)) {
            print_result(bench_memory_add_batch(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphAddBatch)) {
            print_result(bench_memory_graph_add_batch(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphAddLatency)) {
          print_latency_result(bench_memory_graph_add_latency(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphLoad)) {
          print_result(bench_memory_graph_load(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseSnapshotSearch)) {
          print_result(bench_memory_snapshot_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseSnapshotViewSearch)) {
          print_result(bench_memory_snapshot_view_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphSnapshotViewSearch)) {
          print_result(bench_memory_graph_snapshot_view_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseFlatSearchTop1)) {
          print_result(bench_memory_flat_search_top1(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseFlatSearch)) {
          print_result(bench_memory_flat_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseFlatSearchLatency)) {
          print_latency_result(bench_memory_search_latency(iters, false), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseFlatSearchBatch)) {
          print_result(bench_memory_flat_search_batch(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseFlatQ8RecallSearch)) {
          print_result(bench_memory_flat_q8_recall_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseFlatCompactRecallSearch)) {
          print_result(bench_memory_flat_compact_recall_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphTop1)) {
            print_result(bench_memory_graph_top1(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphSearch)) {
            print_result(bench_memory_graph_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphSearchBatch)) {
          print_result(bench_memory_graph_search_batch(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphSearchLatency)) {
          print_latency_result(bench_memory_search_latency(iters, true), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphRecall)) {
            print_result(bench_memory_graph_recall(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphRecallTop1)) {
            print_result(bench_memory_graph_recall_top1(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphRecallSearch)) {
            print_result(bench_memory_graph_recall_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphSnapshotViewRecallSearch)) {
          print_result(bench_memory_graph_snapshot_view_recall_search(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphRecallSearchSweep)) {
            print_memory_graph_recall_search_sweep(iters);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphRecallDetail)) {
          print_memory_graph_recall_detail(iters);
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphLevelStats)) {
          print_memory_graph_level_stats();
        }
        if (memory_case_enabled(kBenchMemoryCaseGraphEdgeStats)) {
          print_memory_graph_edge_stats();
        }
        if (memory_case_enabled(kBenchMemoryCaseCursorBeginFetch)) {
            print_result(bench_memory_cursor_fetch(iters), clock_info().name);
        }
        if (memory_case_enabled(kBenchMemoryCaseMemoryStatus)) {
            print_result(bench_memory_request_status(iters), clock_info().name);
        }
        return;
    }

    constexpr size_t sweep_count = sizeof(kBenchMemorySweepCapacities) / sizeof(kBenchMemorySweepCapacities[0]);
    for (size_t i = 0; i < sweep_count; ++i) {
        const uint32_t capacity = kBenchMemorySweepCapacities[i];
        print_result(bench_memory_add_batch_impl(capacity, kBenchMemorySweepAddNames[i]), clock_info().name);
        print_result(bench_memory_flat_search_top1_impl(iters, capacity, kBenchMemorySweepTopOneNames[i]), clock_info().name);
        print_result(bench_memory_flat_search_impl(iters, capacity, kBenchMemorySweepTopKNames[i]), clock_info().name);
        print_result(bench_memory_cursor_fetch_impl(iters, capacity, kBenchMemorySweepCursorNames[i]), clock_info().name);
    }
}

static AstralHandle load_mock_model_for_agent_bench() {
    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    desc.backend_name = span_from_cstr("mock");
    desc.n_ctx = kBenchAgentContextTokens;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&desc, &model);
    return err == ASTRAL_OK ? model : 0;
}

static BenchResult bench_agent_prompt_warmup_impl(uint64_t iters, bool use_prompt_cache) {
    BenchResult r{};
    r.name = use_prompt_cache ? "features.agent prompt_cache_warmup" : "features.agent prompt_warmup";
    r.ops = iters;

    AstralHandle model = load_mock_model_for_agent_bench();
    if (!astral_handle_valid(model)) {
        r.ops = 0;
        return r;
    }

    AstralExecutorDesc ex{};
    ex.size = sizeof(AstralExecutorDesc);
    ex.max_slots = kBenchAgentSlots;
    ex.max_batch_tokens = kBenchAgentBatchTokens;
    if (astral_model_executor_configure(model, &ex) != ASTRAL_OK) {
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    AstralHandle prompt_cache = 0;
    if (use_prompt_cache) {
        AstralPromptCacheDesc cache_desc{};
        cache_desc.size = sizeof(AstralPromptCacheDesc);
        cache_desc.max_entries = kBenchAgentPromptCacheEntries;
        cache_desc.max_tokens = kBenchAgentPromptCacheTokens;
        cache_desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;
        if (astral_prompt_cache_create(&cache_desc, &prompt_cache) != ASTRAL_OK) {
            astral_model_release(model);
            r.ops = 0;
            return r;
        }
    }

    AstralAgentDesc desc{};
    desc.size = sizeof(AstralAgentDesc);
    desc.model = model;
    desc.prompt_cache = prompt_cache;
    desc.max_tokens = kBenchAgentMaxTokens;
    desc.temperature = 0.0f;
    desc.top_p = 1.0f;
    desc.stream_enabled = 0;
    desc.seed = kBenchAgentSeed;
    desc.max_messages = kBenchAgentMaxMessages;
    desc.max_prompt_bytes = kBenchAgentMaxPromptBytes;

    AstralHandle agent = 0;
    if (astral_agent_create(&desc, &agent) != ASTRAL_OK) {
        if (prompt_cache != 0) {
            astral_prompt_cache_destroy(prompt_cache);
        }
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    (void)astral_agent_set_system_prompt(agent, span_from_cstr("Answer briefly."));
    AstralAgentMessage history{};
    history.size = sizeof(AstralAgentMessage);
    history.role = ASTRAL_AGENT_ROLE_USER;
    history.content = span_from_cstr("hello");
    (void)astral_agent_message_add(agent, &history);

    AstralAgentChatDesc chat{};
    chat.size = sizeof(AstralAgentChatDesc);
    chat.flags = ASTRAL_AGENT_CHAT_FLAG_WARMUP;
    chat.user_message = span_from_cstr("summarize");

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    double prompt_build_ms_total = 0.0;
    for (uint64_t i = 0; i < iters; ++i) {
        const AstralErr err = astral_agent_chat_enqueue(agent, &chat);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        AstralAgentChatResult result{};
        result.size = sizeof(AstralAgentChatResult);
        for (uint32_t poll = 0; poll < kBenchAgentPollLimit; ++poll) {
            if (astral_agent_chat_result(agent, &result) != ASTRAL_OK) {
                break;
            }
            if (result.state == ASTRAL_SESSION_COMPLETED || result.state == ASTRAL_SESSION_FAILED ||
                result.state == ASTRAL_SESSION_CANCELED) {
                break;
            }
            platform::cpu_pause();
        }
        if (result.state != ASTRAL_SESSION_COMPLETED) {
            r.ops = i;
            break;
        }
        prompt_build_ms_total += result.prompt_build_ms;
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    if (r.ops != 0) {
        r.extra_label = "prompt_build_ns/op";
        r.extra_value = (prompt_build_ms_total * kMsToNs) / static_cast<double>(r.ops);
    }
    astral_agent_destroy(agent);
    if (prompt_cache != 0) {
        astral_prompt_cache_destroy(prompt_cache);
    }
    astral_model_release(model);
    return r;
}

static BenchResult bench_agent_prompt_warmup(uint64_t iters) {
    return bench_agent_prompt_warmup_impl(iters, false);
}

static BenchResult bench_agent_prompt_cache_warmup(uint64_t iters) {
    return bench_agent_prompt_warmup_impl(iters, true);
}

static AstralHandle create_session(AstralHandle model, uint32_t max_tokens, float temperature, uint32_t top_k, float top_p, uint32_t seed);

static AstralErr init_media_for_model(AstralHandle model, const char* media_path) {
    if (media_path == nullptr || media_path[0] == '\0') {
        return ASTRAL_E_INVALID;
    }

    AstralModelMediaDesc desc{};
    desc.size = sizeof(AstralModelMediaDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    desc.media_path = span_from_cstr(media_path);
    desc.flags = 0;

    return astral_model_media_init(model, &desc);
}

static BenchResult bench_media_feed_image(AstralHandle model, const AstralImageDesc& image, uint64_t iters) {
    BenchResult r{};
    r.name = "features.media feed_image";
    r.ops = iters;

    AstralHandle session = create_session(model, /*max_tokens=*/1, /*temperature=*/0.0f, /*top_k=*/0, /*top_p=*/1.0f, /*seed=*/1);
    if (!astral_handle_valid(session)) {
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        AstralErr err = astral_session_reset(session, nullptr);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        err = astral_session_feed_image(session, &image, 1);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;

    astral_session_destroy(session);
    return r;
}

static BenchResult bench_media_feed_audio(AstralHandle model, const AstralAudioDesc& audio, uint64_t iters) {
    BenchResult r{};
    r.name = "features.media feed_audio";
    r.ops = iters;

    AstralHandle session = create_session(model, /*max_tokens=*/1, /*temperature=*/0.0f, /*top_k=*/0, /*top_p=*/1.0f, /*seed=*/1);
    if (!astral_handle_valid(session)) {
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        AstralErr err = astral_session_reset(session, nullptr);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        err = astral_session_feed_audio(session, &audio, 1);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;

    astral_session_destroy(session);
    return r;
}

static AstralHandle create_session(AstralHandle model, uint32_t max_tokens, float temperature, uint32_t top_k, float top_p, uint32_t seed) {
    AstralSessionDesc sd{};
    sd.model = model;
    sd.max_tokens = max_tokens;
    sd.temperature = temperature;
    sd.top_k = top_k;
    sd.top_p = top_p;
    sd.stream_enabled = 0;
    sd.seed = seed;

    AstralHandle session = 0;
    const AstralErr err = astral_session_create(&sd, &session);
    if (err != ASTRAL_OK) {
        return 0;
    }
    return session;
}

static BenchResult bench_adapter_attach_clear(uint64_t iters) {
    BenchResult r{};
    r.name = "features.adapter attach_clear";
    r.ops = iters;

    AstralHandle model = load_mock_model_for_agent_bench();
    if (!astral_handle_valid(model)) {
        r.ops = 0;
        return r;
    }

    AstralHandle session = create_session(
        model,
        kBenchSystemPromptSessionTokens,
        /*temperature=*/0.0f,
        /*top_k=*/0,
        /*top_p=*/1.0f,
        kBenchAgentSeed
    );
    if (!astral_handle_valid(session)) {
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    AstralAdapterDesc desc{};
    desc.size = sizeof(AstralAdapterDesc);
    desc.path = span_from_cstr(kBenchAdapterPath);

    AstralHandle adapter = 0;
    if (astral_model_adapter_load(model, &desc, &adapter) != ASTRAL_OK) {
        astral_session_destroy(session);
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        AstralErr err = astral_session_adapters_add(session, adapter, kBenchAdapterScale);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        err = astral_session_adapters_set_scale(session, 0, kBenchAdapterUpdatedScale);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        err = astral_session_adapters_clear(session);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_model_adapter_release(adapter);
    astral_session_destroy(session);
    astral_model_release(model);
    return r;
}

static BenchResult bench_system_prompt_text(uint64_t iters) {
    BenchResult r{};
    r.name = "features.system_prompt text";
    r.ops = iters;

    AstralHandle model = load_mock_model_for_agent_bench();
    if (!astral_handle_valid(model)) {
        r.ops = 0;
        return r;
    }

    AstralHandle session = create_session(
        model,
        kBenchSystemPromptSessionTokens,
        /*temperature=*/0.0f,
        /*top_k=*/0,
        /*top_p=*/1.0f,
        kBenchAgentSeed
    );
    if (!astral_handle_valid(session)) {
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    const AstralSpanU8 system_prompt = span_from_cstr(kBenchSystemPromptText);
    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        AstralErr err = astral_session_reset(session, nullptr);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        err = astral_session_set_system_prompt(session, system_prompt);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_session_destroy(session);
    astral_model_release(model);
    return r;
}

static BenchResult bench_system_prompt_cached_tokens(uint64_t iters) {
    BenchResult r{};
    r.name = "features.system_prompt cached_tokens";
    r.ops = iters;

    AstralHandle model = load_mock_model_for_agent_bench();
    if (!astral_handle_valid(model)) {
        r.ops = 0;
        return r;
    }

    AstralHandle session = create_session(
        model,
        kBenchSystemPromptSessionTokens,
        /*temperature=*/0.0f,
        /*top_k=*/0,
        /*top_p=*/1.0f,
        kBenchAgentSeed
    );
    if (!astral_handle_valid(session)) {
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    AstralPromptCacheDesc desc{};
    desc.size = sizeof(AstralPromptCacheDesc);
    desc.max_entries = kBenchSystemPromptCacheEntries;
    desc.max_tokens = kBenchSystemPromptCacheTokens;
    desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;

    AstralHandle cache = 0;
    if (astral_prompt_cache_create(&desc, &cache) != ASTRAL_OK) {
        astral_session_destroy(session);
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    int32_t tokens[kBenchSystemPromptTokenCapacity]{};
    uint32_t cached_token_count = 0;
    const AstralSpanU8 system_prompt = span_from_cstr(kBenchSystemPromptText);
    if (astral_tokenize(
            model,
            system_prompt,
            tokens,
            kBenchSystemPromptTokenCapacity,
            /*add_special=*/1,
            /*parse_special=*/0,
            &cached_token_count
        ) != ASTRAL_OK ||
        cached_token_count == 0) {
        astral_prompt_cache_destroy(cache);
        astral_session_destroy(session);
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    AstralPromptCacheKey key{};
    key.size = sizeof(AstralPromptCacheKey);
    key.section_kind = ASTRAL_PROMPT_SECTION_SYSTEM;
    key.model = model;
    key.generation = kBenchSystemPromptGeneration;
    key.key = kBenchSystemPromptKey;

    if (astral_prompt_cache_put_tokens(cache, &key, tokens, cached_token_count) != ASTRAL_OK) {
        astral_prompt_cache_destroy(cache);
        astral_session_destroy(session);
        astral_model_release(model);
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();
    for (uint64_t i = 0; i < iters; ++i) {
        AstralErr err = astral_session_reset(session, nullptr);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        const int32_t* view = nullptr;
        uint32_t token_count = 0;
        err = astral_prompt_cache_get_token_view(cache, &key, &view, &token_count);
        if (err != ASTRAL_OK || token_count != cached_token_count || view == nullptr) {
            r.ops = i;
            break;
        }
        err = astral_session_feed_tokens(session, view, token_count, 0);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }
    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    astral_prompt_cache_destroy(cache);
    astral_session_destroy(session);
    astral_model_release(model);
    return r;
}

static bool session_decode_n(AstralHandle session, const char* prompt) {
    const AstralErr e1 = astral_session_feed(session, span_from_cstr(prompt), 1);
    if (e1 != ASTRAL_OK) {
        return false;
    }
    const AstralErr e2 = astral_session_decode(session);
    if (e2 != ASTRAL_OK) {
        return false;
    }
    const AstralErr e3 = astral_session_wait(session, 60000);
    return e3 == ASTRAL_OK;
}

static BenchResult bench_kv_state_save(AstralHandle session, uint64_t iters, uint64_t* out_bytes) {
    BenchResult r{};
    r.name = "features.kv state_save";
    r.ops = iters;

    uint64_t bytes = 0;
    AstralErr err = astral_session_state_size(session, &bytes);
    if (err != ASTRAL_OK || bytes == 0) {
        r.ops = 0;
        return r;
    }
    if (out_bytes) {
        *out_bytes = bytes;
    }

    std::vector<uint8_t> buf(static_cast<size_t>(bytes));
    AstralMutSpanU8 out{};
    out.data = buf.data();
    out.len = static_cast<uint32_t>(buf.size());

    // Warmup once.
    uint64_t written = 0;
    err = astral_session_state_save(session, out, &written);
    if (err != ASTRAL_OK || written == 0) {
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        written = 0;
        err = astral_session_state_save(session, out, &written);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

static BenchResult bench_kv_state_load(AstralHandle session, AstralSpanU8 state, uint64_t iters) {
    BenchResult r{};
    r.name = "features.kv state_load";
    r.ops = iters;

    // Warmup once.
    AstralErr err = astral_session_state_load(session, state);
    if (err != ASTRAL_OK) {
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_session_state_load(session, state);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    return r;
}

static BenchResult bench_grammar_set(AstralHandle model, uint64_t iters) {
    BenchResult r{};
    r.name = "features.grammar set_gbnf";
    r.ops = iters;

    AstralHandle session = create_session(model, /*max_tokens=*/8, /*temperature=*/0.0f, /*top_k=*/0, /*top_p=*/1.0f, /*seed=*/1);
    if (!astral_handle_valid(session)) {
        r.ops = 0;
        return r;
    }

    const char* gbnf =
        "root ::= piece root | piece\n"
        "piece ::= \" a\"\n";
    const AstralSpanU8 g = span_from_cstr(gbnf);

    // Warmup once.
    AstralErr err = astral_session_set_grammar_gbnf(session, g, AstralSpanU8{});
    if (err != ASTRAL_OK) {
        astral_session_destroy(session);
        r.ops = 0;
        return r;
    }

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    for (uint64_t i = 0; i < iters; ++i) {
        err = astral_session_clear_grammar(session);
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
        err = astral_session_set_grammar_gbnf(session, g, AstralSpanU8{});
        if (err != ASTRAL_OK) {
            r.ops = i;
            break;
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;

    astral_session_destroy(session);
    return r;
}

static BenchResult bench_logprobs_drain_meta(AstralHandle model, uint32_t tokens) {
    BenchResult r{};
    r.name = "features.logprobs meta_drain";
    r.ops = tokens;

    AstralHandle session = create_session(model, tokens, /*temperature=*/1.0f, /*top_k=*/32, /*top_p=*/1.0f, /*seed=*/123);
    if (!astral_handle_valid(session)) {
        r.ops = 0;
        return r;
    }

    AstralErr err = astral_session_set_logprobs(session, 8);
    if (err != ASTRAL_OK) {
        astral_session_destroy(session);
        r.ops = 0;
        return r;
    }

    if (!session_decode_n(session, "Hello")) {
        astral_session_destroy(session);
        r.ops = 0;
        return r;
    }

    std::vector<AstralTokenMeta> meta(tokens + 16);

    const uint64_t t0 = ticks_now();
    const uint64_t n0 = ns_now();

    uint64_t got = 0;
    while (got < meta.size()) {
        const int32_t n = astral_stream_read_meta(session, meta.data() + got, static_cast<uint32_t>(meta.size() - got), 0);
        if (n == 0) {
            break;
        }
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        if (n < 0) {
            break;
        }
        got += static_cast<uint64_t>(n);
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.ticks = t1 - t0;
    r.ns = n1 - n0;
    r.ops = got;

    astral_session_destroy(session);
    return r;
}

static void print_features_header(const char* backend, uint32_t gpu_layers, const char* model_path, const char* embed_model_path) {
    std::printf("\n== Feature surfaces (%s, gpu_layers=%u) ==\n", backend ? backend : "?", gpu_layers);
    std::printf("model: %s\n", model_path ? model_path : "(null)");
    if (embed_model_path && model_path && std::strcmp(embed_model_path, model_path) != 0) {
        std::printf("embed_model: %s\n", embed_model_path);
    }
    std::printf("env:\n");
    std::printf("  ASTRAL_BENCH_FEATURE_BACKEND=%s\n", backend ? backend : "");
    std::printf("  ASTRAL_BENCH_GPU_LAYERS=%u\n", gpu_layers);
    std::printf("  ASTRAL_BENCH_FEATURE_ITERS=%llu\n", (unsigned long long)parse_u64_env("ASTRAL_BENCH_FEATURE_ITERS", kFeatureDefaultIters));
    std::printf("  ASTRAL_BENCH_FEATURE_TOKENS=%u\n", parse_u32_env("ASTRAL_BENCH_FEATURE_TOKENS", 64));
    std::printf("  ASTRAL_BENCH_MEMORY_CAPACITY=%u\n",
                bounded_env_u32(kBenchMemoryCapacityEnv,
                                kBenchMemoryCapacity,
                                kBenchMemoryMinCapacity,
                                UINT32_MAX / kBenchMemoryMinDim));
    std::printf("  ASTRAL_BENCH_MEMORY_DIM=%u\n",
                bounded_env_u32(kBenchMemoryDimEnv, kBenchMemoryDim, kBenchMemoryMinDim, kBenchMemoryMaxDim));
    std::printf("  ASTRAL_BENCH_MEMORY_METRIC=%s\n",
                std::getenv(kBenchMemoryMetricEnv) ? std::getenv(kBenchMemoryMetricEnv) : kBenchMemoryMetricCosine);
    std::printf("  ASTRAL_BENCH_MEMORY_STORAGE=%s\n",
                std::getenv(kBenchMemoryStorageEnv) ? std::getenv(kBenchMemoryStorageEnv) : kBenchMemoryStorageF32);
    std::printf("  ASTRAL_BENCH_MEMORY_CASE=%s\n",
                std::getenv(kBenchMemoryCaseEnv) ? std::getenv(kBenchMemoryCaseEnv) : "");
    std::printf("  ASTRAL_BENCH_MEMORY_SWEEP=%s\n",
                std::getenv(kBenchMemorySweepEnv) ? std::getenv(kBenchMemorySweepEnv) : "");
    std::printf("  ASTRAL_BENCH_MEMORY_GRAPH_NEIGHBORS=%u\n", memory_graph_neighbors());
    std::printf("  ASTRAL_BENCH_MEMORY_GRAPH_SEARCH=%u\n", memory_graph_search());
    std::printf("  ASTRAL_BENCH_MEMORY_GRAPH_QUERY_SEARCH=%u\n", memory_graph_query_search());
    std::printf("  ASTRAL_BENCH_MEMORY_RECALL_QUERIES=%u\n", memory_recall_queries());
    std::printf("  ASTRAL_BENCH_EMBED_MODEL=%s\n", std::getenv("ASTRAL_BENCH_EMBED_MODEL") ? std::getenv("ASTRAL_BENCH_EMBED_MODEL") : "");
    std::printf("  ASTRAL_BENCH_VISION_MODEL=%s\n", std::getenv("ASTRAL_BENCH_VISION_MODEL") ? std::getenv("ASTRAL_BENCH_VISION_MODEL") : "");
    std::printf("  ASTRAL_BENCH_VISION_MEDIA=%s\n", std::getenv("ASTRAL_BENCH_VISION_MEDIA") ? std::getenv("ASTRAL_BENCH_VISION_MEDIA") : "");
    std::printf("  ASTRAL_BENCH_AUDIO_MODEL=%s\n", std::getenv("ASTRAL_BENCH_AUDIO_MODEL") ? std::getenv("ASTRAL_BENCH_AUDIO_MODEL") : "");
    std::printf("  ASTRAL_BENCH_AUDIO_MEDIA=%s\n", std::getenv("ASTRAL_BENCH_AUDIO_MEDIA") ? std::getenv("ASTRAL_BENCH_AUDIO_MEDIA") : "");
}

} // namespace

void bench_feature_surfaces_print(void) {
    if (env_enabled(kBenchMemoryOnlyEnv)) {
        const uint64_t iters = parse_u64_env("ASTRAL_BENCH_FEATURE_ITERS", kFeatureDefaultIters);

        AstralInit cfg{};
        cfg.reserve_bytes = kPromptCacheOnlyReserveBytes;
        cfg.thread_count = parse_u32_env("ASTRAL_BENCH_RUNTIME_THREADS", 1);
        cfg.numa_node = 0xFFFFFFFFu;
        cfg.enable_hugepages = 0;

        const AstralErr init_err = astral_init(&cfg);
        if (init_err != ASTRAL_OK) {
            std::fprintf(stderr, "[bench] astral_init failed: %s (%s)\n",
                         astral_error_string(init_err), astral_last_error());
            return;
        }

        std::printf("%-28s  %8s  %12s  %12s  %12s\n", "benchmark", "clock", "ops", "ns/op", "ticks/op");
        print_memory_benchmarks(iters);
        astral_shutdown();
        return;
    }

    if (env_enabled("ASTRAL_BENCH_PROMPT_CACHE_ONLY")) {
        const uint64_t iters = parse_u64_env("ASTRAL_BENCH_FEATURE_ITERS", kFeatureDefaultIters);

        AstralInit cfg{};
        cfg.reserve_bytes = kPromptCacheOnlyReserveBytes;
        cfg.thread_count = parse_u32_env("ASTRAL_BENCH_RUNTIME_THREADS", 1);
        cfg.numa_node = 0xFFFFFFFFu;
        cfg.enable_hugepages = 0;

        const AstralErr init_err = astral_init(&cfg);
        if (init_err != ASTRAL_OK) {
            std::fprintf(stderr, "[bench] astral_init failed: %s (%s)\n",
                         astral_error_string(init_err), astral_last_error());
            return;
        }

        std::printf("%-28s  %8s  %12s  %12s  %12s\n", "benchmark", "clock", "ops", "ns/op", "ticks/op");
        const AstralHandle model = load_mock_model_for_agent_bench();
        if (astral_handle_valid(model)) {
            print_result(bench_tokenize_count(model, iters), clock_info().name);
            print_result(bench_tokenize_batch(model, iters), clock_info().name);
            astral_model_release(model);
        }
        print_result(bench_prompt_cache_get(iters), clock_info().name);
        print_result(bench_prompt_cache_view(iters), clock_info().name);
        print_result(bench_prompt_cache_hot_view(iters), clock_info().name);
        print_result(bench_prompt_cache_miss(iters), clock_info().name);
        print_result(bench_prompt_cache_fifo_evict(iters), clock_info().name);
        print_result(bench_prompt_cache_compacted_evict(iters), clock_info().name);
        print_result(bench_system_prompt_text(iters), clock_info().name);
        print_result(bench_system_prompt_cached_tokens(iters), clock_info().name);
        print_result(bench_adapter_attach_clear(iters), clock_info().name);
        print_result(bench_toolset_parse(iters), clock_info().name);
        print_result(bench_toolset_parse_many(iters), clock_info().name);
        print_result(bench_chunk_word_ranges(iters), clock_info().name);
        print_result(bench_chunk_token_count(iters), clock_info().name);
        print_result(bench_chunk_token_ranges(iters), clock_info().name);
        print_memory_benchmarks(iters);
        print_result(bench_agent_prompt_warmup(iters), clock_info().name);
        print_result(bench_agent_prompt_cache_warmup(iters), clock_info().name);
        astral_shutdown();
        return;
    }

    const char* model_path = find_model_path();
    if (model_path == nullptr) {
        std::fprintf(stderr, "[bench] no GGUF model found for feature benches (set ASTRAL_BENCH_MODEL)\n");
        return;
    }
    const char* embed_model_path = find_embed_model_path_or_fallback(model_path);

    const char* backend = std::getenv("ASTRAL_BENCH_FEATURE_BACKEND");
    if (backend == nullptr || backend[0] == '\0') {
        backend = "cpu";
    }
    const uint32_t gpu_layers = parse_u32_env("ASTRAL_BENCH_GPU_LAYERS", std::strcmp(backend, "cuda") == 0 ? 8u : 0u);
    const uint64_t iters = parse_u64_env("ASTRAL_BENCH_FEATURE_ITERS", kFeatureDefaultIters);
    const uint32_t tokens = parse_u32_env("ASTRAL_BENCH_FEATURE_TOKENS", 64);

    AstralInit cfg{};
    cfg.reserve_bytes = 2ULL << 30;
    cfg.thread_count = parse_u32_env("ASTRAL_BENCH_RUNTIME_THREADS", 1);
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;

    const AstralErr init_err = astral_init(&cfg);
    if (init_err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_init failed: %s (%s)\n",
                         astral_error_string(init_err), astral_last_error());
        return;
    }

    if (env_enabled(kBenchTokenizeOnlyEnv)) {
        std::printf("\n== Tokenization (%s, gpu_layers=%u) ==\n", backend, gpu_layers);
        std::printf("model: %s\n", model_path);
        std::printf("env:\n");
        std::printf("  ASTRAL_BENCH_FEATURE_BACKEND=%s\n", backend);
        std::printf("  ASTRAL_BENCH_GPU_LAYERS=%u\n", gpu_layers);
        std::printf("  ASTRAL_BENCH_FEATURE_ITERS=%llu\n", (unsigned long long)iters);
        std::printf("  ASTRAL_BENCH_TOKENIZE_ONLY=1\n");
        std::printf("%-28s  %8s  %12s  %12s  %12s\n", "benchmark", "clock", "ops", "ns/op", "ticks/op");
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/0);
        if (astral_handle_valid(model)) {
            print_result(bench_tokenize_count(model, iters), clock_info().name);
            print_result(bench_tokenize_batch(model, iters), clock_info().name);
            astral_model_release(model);
        }
        astral_shutdown();
        return;
    }

    print_features_header(backend, gpu_layers, model_path, embed_model_path);
    print_result(bench_prompt_cache_get(iters), clock_info().name);
    print_result(bench_prompt_cache_view(iters), clock_info().name);
    print_result(bench_prompt_cache_hot_view(iters), clock_info().name);
    print_result(bench_prompt_cache_miss(iters), clock_info().name);
    print_result(bench_prompt_cache_fifo_evict(iters), clock_info().name);
    print_result(bench_prompt_cache_compacted_evict(iters), clock_info().name);
    print_result(bench_system_prompt_text(iters), clock_info().name);
    print_result(bench_system_prompt_cached_tokens(iters), clock_info().name);
    print_result(bench_adapter_attach_clear(iters), clock_info().name);
    print_result(bench_toolset_parse(iters), clock_info().name);
    print_result(bench_toolset_parse_many(iters), clock_info().name);
    print_result(bench_chunk_word_ranges(iters), clock_info().name);
    print_result(bench_chunk_token_count(iters), clock_info().name);
    print_result(bench_chunk_token_ranges(iters), clock_info().name);
    print_memory_benchmarks(iters);
    print_result(bench_agent_prompt_warmup(iters), clock_info().name);
    print_result(bench_agent_prompt_cache_warmup(iters), clock_info().name);

    // Embeddings.
    {
        const AstralHandle model = load_model(backend, embed_model_path, gpu_layers, /*embeddings_only=*/1);
        if (astral_handle_valid(model)) {
            print_result(bench_embed_roundtrip(model, iters), clock_info().name);
            astral_model_release(model);
        }
    }

    // Tokenization.
    {
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/0);
        if (astral_handle_valid(model)) {
            print_result(bench_tokenize_count(model, iters), clock_info().name);
            print_result(bench_tokenize_batch(model, iters), clock_info().name);
            astral_model_release(model);
        }
    }

    // Media feed (vision/audio) benches when models + media are provided.
    {
        const char* vision_model = std::getenv("ASTRAL_BENCH_VISION_MODEL");
        const char* vision_media = std::getenv("ASTRAL_BENCH_VISION_MEDIA");
        if (file_is_large_enough(vision_model, 100ull * 1024ull * 1024ull) &&
            file_is_large_enough(vision_media, 1ull * 1024ull * 1024ull)) {
            const AstralHandle model = load_model(backend, vision_model, gpu_layers, /*embeddings_only=*/0);
            if (astral_handle_valid(model)) {
                const AstralErr m_err = init_media_for_model(model, vision_media);
                if (m_err == ASTRAL_OK) {
                    std::vector<uint8_t> img_storage;
                    const AstralImageDesc img = make_bench_image(img_storage);
                    print_result(bench_media_feed_image(model, img, iters), clock_info().name);
                } else {
                    std::fprintf(stderr, "[bench] media init failed (vision): %s (%s)\n",
                                 astral_error_string(m_err), astral_last_error());
                }
                astral_model_release(model);
            }
        }

        const char* audio_model = std::getenv("ASTRAL_BENCH_AUDIO_MODEL");
        const char* audio_media = std::getenv("ASTRAL_BENCH_AUDIO_MEDIA");
        if (file_is_large_enough(audio_model, 100ull * 1024ull * 1024ull) &&
            file_is_large_enough(audio_media, 1ull * 1024ull * 1024ull)) {
            const AstralHandle model = load_model(backend, audio_model, gpu_layers, /*embeddings_only=*/0);
            if (astral_handle_valid(model)) {
                const AstralErr m_err = init_media_for_model(model, audio_media);
                if (m_err == ASTRAL_OK) {
                    std::vector<uint8_t> audio_storage;
                    const AstralAudioDesc audio = make_bench_audio(audio_storage);
                    print_result(bench_media_feed_audio(model, audio, iters), clock_info().name);
                } else {
                    std::fprintf(stderr, "[bench] media init failed (audio): %s (%s)\n",
                                 astral_error_string(m_err), astral_last_error());
                }
                astral_model_release(model);
            }
        }
    }

    // KV state save/load.
    {
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/0);
        if (astral_handle_valid(model)) {
            AstralHandle session = create_session(model, /*max_tokens=*/16, /*temperature=*/0.0f, /*top_k=*/0, /*top_p=*/1.0f, /*seed=*/1);
            if (astral_handle_valid(session) && session_decode_n(session, "Hello")) {
                uint64_t bytes = 0;
                const BenchResult save_r = bench_kv_state_save(session, iters, &bytes);
                print_result(save_r, clock_info().name);
                std::printf("%-28s  %8s  %10llu bytes\n", "features.kv bytes", "", (unsigned long long)bytes);

                // Capture one state blob for the load bench.
                std::vector<uint8_t> buf(static_cast<size_t>(bytes));
                AstralMutSpanU8 out{};
                out.data = buf.data();
                out.len = static_cast<uint32_t>(buf.size());
                uint64_t written = 0;
                if (astral_session_state_save(session, out, &written) == ASTRAL_OK && written > 0) {
                    AstralSpanU8 in{};
                    in.data = buf.data();
                    in.len = static_cast<uint32_t>(written);
                    print_result(bench_kv_state_load(session, in, iters), clock_info().name);
                }
            }
            if (astral_handle_valid(session)) {
                astral_session_destroy(session);
            }
            astral_model_release(model);
        }
    }

    // Grammar set/clear cost.
    {
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/0);
        if (astral_handle_valid(model)) {
            print_result(bench_grammar_set(model, iters), clock_info().name);
            astral_model_release(model);
        }
    }

    // Logprobs meta drain.
    {
        const AstralHandle model = load_model(backend, model_path, gpu_layers, /*embeddings_only=*/0);
        if (astral_handle_valid(model)) {
            print_result(bench_logprobs_drain_meta(model, tokens), clock_info().name);
            astral_model_release(model);
        }
    }

    astral_shutdown();
}

} // namespace astral::bench
