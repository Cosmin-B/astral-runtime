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
static constexpr uint32_t kBenchMemoryDim = 32;
static constexpr uint32_t kBenchMemoryCapacity = 1024;
static constexpr uint32_t kBenchMemoryTopOne = 1;
static constexpr uint32_t kBenchMemoryTopK = 8;
static constexpr uint32_t kBenchMemoryFetchK = 4;
static constexpr uint32_t kBenchMemoryMinDim = 1;
static constexpr uint32_t kBenchMemoryMaxDim = 8192;
static constexpr uint32_t kBenchMemoryMinCapacity = 1;
static constexpr uint32_t kBenchMemorySweepTiny = 100;
static constexpr uint32_t kBenchMemorySweepSmall = 1000;
static constexpr uint32_t kBenchMemorySweepMedium = 10000;
static constexpr uint32_t kBenchMemorySweepLarge = 100000;
static constexpr uint32_t kBenchMemoryValueMask = 0xFu;
static constexpr uint32_t kBenchMemoryColumnBias = 3;
static constexpr uint32_t kBenchMemoryQueryMask = 7u;
static constexpr uint32_t kBenchMemoryGroupMask = 1u;
static constexpr uint32_t kBenchMemoryDocShift = 4u;
static constexpr uint64_t kBenchMemoryKeyBase = 1u;
static constexpr uint32_t kBenchMemoryGraphNeighbors = 16;
static constexpr uint32_t kBenchMemoryGraphSearch = 64;
static constexpr float kBenchMemoryValueBias = 1.0f;
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
static constexpr char kBenchMemorySweepEnv[] = "ASTRAL_BENCH_MEMORY_SWEEP";
static constexpr char kBenchMemoryMetricDot[] = "dot";
static constexpr char kBenchMemoryMetricL2[] = "l2";
static constexpr char kBenchMemoryMetricCosine[] = "cosine";
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

static uint32_t memory_bench_dim() {
    return bounded_env_u32(kBenchMemoryDimEnv, kBenchMemoryDim, kBenchMemoryMinDim, kBenchMemoryMaxDim);
}

static uint32_t memory_bench_capacity(uint32_t fallback, uint32_t dim) {
    return bounded_env_u32(kBenchMemoryCapacityEnv, fallback, kBenchMemoryMinCapacity, UINT32_MAX / dim);
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
            vectors[static_cast<size_t>(row) * dim + col] =
                static_cast<float>(((row + kBenchMemoryKeyBase) * (col + kBenchMemoryColumnBias)) & kBenchMemoryValueMask) +
                kBenchMemoryValueBias;
        }
    }
}

static void fill_memory_query(std::vector<float>& query) {
    for (uint32_t i = 0; i < static_cast<uint32_t>(query.size()); ++i) {
        query[i] = static_cast<float>((i & kBenchMemoryQueryMask) + kBenchMemoryValueBias);
    }
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

static BenchResult bench_memory_flat_search(uint64_t iters) {
    const uint32_t dim = memory_bench_dim();
    return bench_memory_flat_search_impl(iters, memory_bench_capacity(kBenchMemoryCapacity, dim), "features.memory flat_search");
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
    desc.graph_neighbors = kBenchMemoryGraphNeighbors;
    desc.graph_search = kBenchMemoryGraphSearch;

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

static void print_memory_benchmarks(uint64_t iters) {
    if (!env_enabled(kBenchMemorySweepEnv)) {
        print_result(bench_memory_add_batch(iters), clock_info().name);
        print_result(bench_memory_flat_search_top1(iters), clock_info().name);
        print_result(bench_memory_flat_search(iters), clock_info().name);
        print_result(bench_memory_graph_search(iters), clock_info().name);
        print_result(bench_memory_cursor_fetch(iters), clock_info().name);
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
    std::printf("  ASTRAL_BENCH_MEMORY_SWEEP=%s\n",
                std::getenv(kBenchMemorySweepEnv) ? std::getenv(kBenchMemorySweepEnv) : "");
    std::printf("  ASTRAL_BENCH_EMBED_MODEL=%s\n", std::getenv("ASTRAL_BENCH_EMBED_MODEL") ? std::getenv("ASTRAL_BENCH_EMBED_MODEL") : "");
    std::printf("  ASTRAL_BENCH_VISION_MODEL=%s\n", std::getenv("ASTRAL_BENCH_VISION_MODEL") ? std::getenv("ASTRAL_BENCH_VISION_MODEL") : "");
    std::printf("  ASTRAL_BENCH_VISION_MEDIA=%s\n", std::getenv("ASTRAL_BENCH_VISION_MEDIA") ? std::getenv("ASTRAL_BENCH_VISION_MEDIA") : "");
    std::printf("  ASTRAL_BENCH_AUDIO_MODEL=%s\n", std::getenv("ASTRAL_BENCH_AUDIO_MODEL") ? std::getenv("ASTRAL_BENCH_AUDIO_MODEL") : "");
    std::printf("  ASTRAL_BENCH_AUDIO_MEDIA=%s\n", std::getenv("ASTRAL_BENCH_AUDIO_MEDIA") ? std::getenv("ASTRAL_BENCH_AUDIO_MEDIA") : "");
}

} // namespace

void bench_feature_surfaces_print(void) {
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
        print_result(bench_system_prompt_text(iters), clock_info().name);
        print_result(bench_system_prompt_cached_tokens(iters), clock_info().name);
        print_result(bench_adapter_attach_clear(iters), clock_info().name);
        print_result(bench_toolset_parse(iters), clock_info().name);
        print_result(bench_toolset_parse_many(iters), clock_info().name);
        print_result(bench_chunk_word_ranges(iters), clock_info().name);
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

    print_features_header(backend, gpu_layers, model_path, embed_model_path);
    print_result(bench_prompt_cache_get(iters), clock_info().name);
    print_result(bench_prompt_cache_view(iters), clock_info().name);
    print_result(bench_prompt_cache_hot_view(iters), clock_info().name);
    print_result(bench_prompt_cache_miss(iters), clock_info().name);
    print_result(bench_prompt_cache_fifo_evict(iters), clock_info().name);
    print_result(bench_system_prompt_text(iters), clock_info().name);
    print_result(bench_system_prompt_cached_tokens(iters), clock_info().name);
    print_result(bench_adapter_attach_clear(iters), clock_info().name);
    print_result(bench_toolset_parse(iters), clock_info().name);
    print_result(bench_toolset_parse_many(iters), clock_info().name);
    print_result(bench_chunk_word_ranges(iters), clock_info().name);
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
