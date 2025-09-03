#include "bench_clock.hpp"

#include "../include/astral_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
  #include <unistd.h>
#endif

namespace astral::bench {

namespace {

struct EmbedBenchResult {
    uint64_t total_ticks;
    uint64_t total_ns;
    uint64_t embeddings;
    uint64_t tokens_per_embed;
    uint32_t dim;
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

static uint32_t bench_runtime_threads() {
    // The benchmark only needs a small worker pool to run the decode/embed job.
    // Default to 1 to avoid idle worker oversubscription affecting model scaling.
    return parse_u32_env("ASTRAL_BENCH_RUNTIME_THREADS", 1);
}

static uint32_t bench_model_threads() {
    // Prefer an explicit model-thread override, but accept the older ASTRAL_BENCH_THREADS
    // as a compatibility alias.
    const uint32_t explicit_model = parse_u32_env("ASTRAL_BENCH_MODEL_THREADS", 0);
    if (explicit_model != 0) {
        return explicit_model;
    }
    return parse_u32_env("ASTRAL_BENCH_THREADS", 0);
}

static void try_load_default_toy_plugin() {
#if defined(__linux__)
    char exe_path[1024];
    const ssize_t n = ::readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n <= 0) {
        return;
    }
    exe_path[n] = '\0';

    // Strip filename.
    char* last_slash = std::strrchr(exe_path, '/');
    if (last_slash == nullptr) {
        return;
    }
    *last_slash = '\0';

    // Default build layout: <build>/benchmarks/astral_benchmarks and <build>/tests/<plugin>.
    char plugin_path[1200];
    std::snprintf(plugin_path, sizeof(plugin_path), "%s/../tests/libastral_backend_toy_plugin.so", exe_path);

    AstralSpanU8 path{};
    path.data = reinterpret_cast<const uint8_t*>(plugin_path);
    path.len = static_cast<uint32_t>(std::strlen(plugin_path));
    (void)astral_backend_load_plugin(path);
#endif
}

static void maybe_load_plugin_from_env() {
    const char* plugin = std::getenv("ASTRAL_BENCH_EMB_PLUGIN");
    if (plugin == nullptr || plugin[0] == '\0') {
        try_load_default_toy_plugin();
        return;
    }

    AstralSpanU8 path{};
    path.data = reinterpret_cast<const uint8_t*>(plugin);
    path.len = static_cast<uint32_t>(std::strlen(plugin));
    (void)astral_backend_load_plugin(path);
}

static uint32_t count_tokens(AstralHandle model, const char* text) {
    int32_t tmp[4096];
    uint32_t out = 0;

    AstralSpanU8 span{};
    span.data = reinterpret_cast<const uint8_t*>(text);
    span.len = static_cast<uint32_t>(std::strlen(text));

    const AstralErr err = astral_tokenize(model, span, tmp, static_cast<uint32_t>(sizeof(tmp) / sizeof(tmp[0])), 1, 0, &out);
    if (err != ASTRAL_OK) {
        return 0;
    }
    return out;
}

static EmbedBenchResult run_embeddings(AstralHandle model, AstralHandle emb, const char* text, uint64_t iters) {
    EmbedBenchResult r{};
    r.embeddings = iters;

    uint32_t dim = 0;
    if (astral_model_embedding_dim(model, &dim) != ASTRAL_OK || dim == 0) {
        return r;
    }
    r.dim = dim;

    const uint32_t token_count = count_tokens(model, text);
    r.tokens_per_embed = token_count;

    // Allocate output buffer once.
    const uint64_t bytes = static_cast<uint64_t>(dim) * sizeof(float);
    float* vec = static_cast<float*>(std::malloc(bytes));
    if (vec == nullptr) {
        return r;
    }
    std::memset(vec, 0, bytes);

    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = static_cast<uint32_t>(bytes);

    AstralSpanU8 text_span{};
    text_span.data = reinterpret_cast<const uint8_t*>(text);
    text_span.len = static_cast<uint32_t>(std::strlen(text));

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
        const AstralErr e = astral_embed_enqueue(emb, text_span, &ticket);
        if (e != ASTRAL_OK) {
            break;
        }
        const AstralErr c = astral_embed_collect(emb, ticket, out);
        if (c != ASTRAL_OK) {
            break;
        }
    }

    const uint64_t t1 = ticks_now();
    const uint64_t n1 = ns_now();

    r.total_ticks = t1 - t0;
    r.total_ns = n1 - n0;

    std::free(vec);
    return r;
}

} // namespace

void bench_embeddings_print(uint32_t dim_override, uint64_t iters) {
    maybe_load_plugin_from_env();

    const char* backend = std::getenv("ASTRAL_BENCH_EMB_BACKEND");
    if (backend == nullptr || backend[0] == '\0') {
        backend = "toy";
    }

    char model_path_buf[64];
    const char* model_path = std::getenv("ASTRAL_BENCH_EMB_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
        if (dim_override > 0) {
            std::snprintf(model_path_buf, sizeof(model_path_buf), "dim=%u", dim_override);
            model_path = model_path_buf;
        } else {
            model_path = "dim=256";
        }
    }

    AstralInit cfg{};
    cfg.reserve_bytes = 512ull << 20;
    cfg.thread_count = bench_runtime_threads();
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_init failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        return;
    }

    AstralModelDesc model_desc{};
    model_desc.embeddings_only = 1;
    model_desc.n_ctx = 256;
    model_desc.n_batch = 128;
    model_desc.n_threads = bench_model_threads();
    model_desc.gpu_layers = 0;

    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));

    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_model_load failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        astral_shutdown();
        return;
    }

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_model_embedding_dim failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        astral_model_release(model);
        astral_shutdown();
        return;
    }

    AstralHandle emb = 0;
    err = astral_embed_create(model, &emb);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "[bench] astral_embed_create failed: %s (%s)\n",
                     astral_error_string(err),
                     astral_last_error());
        astral_model_release(model);
        astral_shutdown();
        return;
    }

    const char* text = "Once upon a time";
    const EmbedBenchResult r = run_embeddings(model, emb, text, iters);

    const double total_s = static_cast<double>(r.total_ns) * 1e-9;
    const double emb_per_s = (total_s > 0.0) ? static_cast<double>(r.embeddings) / total_s : 0.0;
    const double tok_per_s = emb_per_s * static_cast<double>(r.tokens_per_embed);
    const double out_mb_per_s =
        emb_per_s * (static_cast<double>(r.dim) * static_cast<double>(sizeof(float))) / (1024.0 * 1024.0);

    std::printf("\nEmbeddings benchmark (ticks: %s)\n", clock_info().name);
    std::printf("Backend: %s\n", backend);
    std::printf("Model:   %s\n", model_path);
    std::printf("Runtime threads (requested): %u\n", cfg.thread_count);
    std::printf("Model threads (requested):   %u\n", model_desc.n_threads);
    std::printf("Dim:     %u\n", dim);
    std::printf("Iters:   %llu\n", static_cast<unsigned long long>(iters));
    std::printf("Tokens/embed: %llu\n", static_cast<unsigned long long>(r.tokens_per_embed));
    std::printf("Total:   %.2f ms  (%llu ticks)\n",
                static_cast<double>(r.total_ns) / 1e6,
                static_cast<unsigned long long>(r.total_ticks));
    std::printf("Emb/s:   %.2f\n", emb_per_s);
    std::printf("Tok/s:   %.2f\n", tok_per_s);
    std::printf("Out:     %.2f MiB/s\n\n", out_mb_per_s);

    astral_embed_destroy(emb);
    astral_model_release(model);
    astral_shutdown();
}

} // namespace astral::bench
