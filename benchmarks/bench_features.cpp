#include "bench_clock.hpp"
#include "bench_common.hpp"

#include "../include/astral_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace astral::bench {

namespace {

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
    std::printf("  ASTRAL_BENCH_FEATURE_ITERS=%llu\n", (unsigned long long)parse_u64_env("ASTRAL_BENCH_FEATURE_ITERS", 2000));
    std::printf("  ASTRAL_BENCH_FEATURE_TOKENS=%u\n", parse_u32_env("ASTRAL_BENCH_FEATURE_TOKENS", 64));
    std::printf("  ASTRAL_BENCH_EMBED_MODEL=%s\n", std::getenv("ASTRAL_BENCH_EMBED_MODEL") ? std::getenv("ASTRAL_BENCH_EMBED_MODEL") : "");
    std::printf("  ASTRAL_BENCH_VISION_MODEL=%s\n", std::getenv("ASTRAL_BENCH_VISION_MODEL") ? std::getenv("ASTRAL_BENCH_VISION_MODEL") : "");
    std::printf("  ASTRAL_BENCH_VISION_MEDIA=%s\n", std::getenv("ASTRAL_BENCH_VISION_MEDIA") ? std::getenv("ASTRAL_BENCH_VISION_MEDIA") : "");
    std::printf("  ASTRAL_BENCH_AUDIO_MODEL=%s\n", std::getenv("ASTRAL_BENCH_AUDIO_MODEL") ? std::getenv("ASTRAL_BENCH_AUDIO_MODEL") : "");
    std::printf("  ASTRAL_BENCH_AUDIO_MEDIA=%s\n", std::getenv("ASTRAL_BENCH_AUDIO_MEDIA") ? std::getenv("ASTRAL_BENCH_AUDIO_MEDIA") : "");
}

} // namespace

void bench_feature_surfaces_print(void) {
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
    const uint64_t iters = parse_u64_env("ASTRAL_BENCH_FEATURE_ITERS", 2000);
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
