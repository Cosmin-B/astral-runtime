/**
 * test_embeddings.cpp - Embeddings API tests
 *
 * Validates:
 * - model embedding dim query
 * - embedder create/enqueue/collect lifecycle
 * - deterministic mock embeddings output
 */

#include "test_framework.hpp"
#include "astral_rt.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

namespace {

static bool file_exists_min_size(const char* path, uint64_t min_bytes) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    return static_cast<uint64_t>(st.st_size) >= min_bytes;
}

static const char* find_env_model_path(const char* env_name) {
    const char* path = std::getenv(env_name);
    if (file_exists_min_size(path, 5ULL * 1024ULL * 1024ULL)) {
        return path;
    }
    if (path == nullptr || path[0] == '\0' || path[0] == '/') {
        return nullptr;
    }

    static char resolved[4096];
    static const char* prefixes[] = {
        "../",
        "../../",
        "../../../",
    };
    for (const char* prefix : prefixes) {
        const int written = std::snprintf(resolved, sizeof(resolved), "%s%s", prefix, path);
        if (written > 0 &&
            static_cast<size_t>(written) < sizeof(resolved) &&
            file_exists_min_size(resolved, 5ULL * 1024ULL * 1024ULL)) {
            return resolved;
        }
    }

    return nullptr;
}

static const char* find_test_model_path() {
    const char* env_embed = find_env_model_path("ASTRAL_TEST_EMBED_MODEL");
    if (env_embed != nullptr) {
        return env_embed;
    }

    const char* env = find_env_model_path("ASTRAL_TEST_MODEL");
    if (env != nullptr) {
        return env;
    }

    static const char* paths[] = {
        "tests/models/all-MiniLM-L6-v2-Q2_K.gguf",
        "../tests/models/all-MiniLM-L6-v2-Q2_K.gguf",
        "../../tests/models/all-MiniLM-L6-v2-Q2_K.gguf",
        "../../../tests/models/all-MiniLM-L6-v2-Q2_K.gguf",

        "tests/models/gpt2.Q2_K.gguf",
        "../tests/models/gpt2.Q2_K.gguf",
        "../../tests/models/gpt2.Q2_K.gguf",
        "../../../tests/models/gpt2.Q2_K.gguf",
    };

    for (const char* p : paths) {
        if (file_exists_min_size(p, 5ULL * 1024ULL * 1024ULL)) {
            return p;
        }
    }

    return nullptr;
}

static const char* find_required_embedding_model_path() {
    return find_env_model_path("ASTRAL_TEST_EMBED_MODEL");
}

static uint32_t embedding_throughput_iters() {
    const char* env = std::getenv("ASTRAL_TEST_EMBED_THROUGHPUT_ITERS");
    if (env == nullptr || env[0] == '\0') {
        return 16u;
    }

    const unsigned long parsed = std::strtoul(env, nullptr, 10);
    if (parsed < 4ul) {
        return 4u;
    }
    if (parsed > 256ul) {
        return 256u;
    }
    return static_cast<uint32_t>(parsed);
}

static AstralSpanU8 span_from_cstr(const char* text) {
    AstralSpanU8 span{};
    span.data = reinterpret_cast<const uint8_t*>(text);
    span.len = text != nullptr ? static_cast<uint32_t>(std::strlen(text)) : 0u;
    return span;
}

static AstralHandle load_mock_embedding_model() {
    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path = span_from_cstr("infinite");
    model_desc.backend_name = span_from_cstr("mock");
    model_desc.embeddings_only = 1;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));
    return model;
}

static float mock_embedding_first_value(const char* text) {
    uint32_t sum = 256u;
    if (text != nullptr) {
        for (const char* p = text; *p != '\0'; ++p) {
            sum += static_cast<uint8_t>(*p);
        }
    }
    return static_cast<float>(sum);
}

} // namespace

TEST(embeddings_mock_e2e) {
    AstralInit cfg{};
    cfg.reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* backend = "mock";
    const char* model_path = "infinite";

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    model_desc.embeddings_only = 1;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(dim, 0u);

    AstralHandle emb = 0;
    err = astral_embed_create(model, &emb);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(emb));

    const char* text = "abc";
    AstralSpanU8 text_span{};
    text_span.data = reinterpret_cast<const uint8_t*>(text);
    text_span.len = static_cast<uint32_t>(std::strlen(text));

    uint64_t ticket = 0;
    err = astral_embed_enqueue(emb, text_span, &ticket);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_NE(ticket, 0ULL);

    float vec[64] = {};
    ASSERT_LE(dim, 64u);

    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = static_cast<uint32_t>(sizeof(vec));

    err = astral_embed_collect(emb, ticket, out);
    ASSERT_EQ(err, ASTRAL_OK);

    // Mock embedding: sum(abs(tokens)) + i, where tokens include BOS (256) and bytes of "abc" (97, 98, 99).
    const float expected0 = static_cast<float>(256 + 97 + 98 + 99);
    ASSERT_EQ(vec[0], expected0);
    ASSERT_EQ(vec[1], expected0 + 1.0f);

    astral_embed_destroy(emb);
    astral_model_release(model);
    astral_shutdown();
}

TEST(embeddings_request_ticket_state_mock) {
    AstralInit cfg{};
    cfg.reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;

    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

    const AstralHandle model = load_mock_embedding_model();

    AstralHandle emb = 0;
    ASSERT_EQ(astral_embed_create(model, &emb), ASTRAL_OK);

    uint64_t ticket = 0;
    ASSERT_EQ(astral_embed_enqueue(emb, span_from_cstr("request"), &ticket), ASTRAL_OK);

    AstralRequestRef request{};
    ASSERT_EQ(astral_request_from_embedding(emb, ticket, &request), ASTRAL_OK);
    ASSERT_EQ(request.kind, ASTRAL_REQUEST_EMBEDDING);
    ASSERT_EQ(request.owner, emb);
    ASSERT_EQ(request.ticket, ticket);

    AstralRequestStatus status{};
    status.size = sizeof(AstralRequestStatus);
    ASSERT_EQ(astral_request_state(&request, &status), ASTRAL_OK);
    ASSERT_EQ(status.state, ASTRAL_REQUEST_QUEUED);
    ASSERT_EQ(status.flags & ASTRAL_REQUEST_FLAG_TICKET, ASTRAL_REQUEST_FLAG_TICKET);
    ASSERT_GT(status.queue_depth, 0u);

    ASSERT_EQ(astral_request_cancel(&request), ASTRAL_OK);

    status = AstralRequestStatus{};
    status.size = sizeof(AstralRequestStatus);
    ASSERT_EQ(astral_request_state(&request, &status), ASTRAL_OK);
    ASSERT_EQ(status.state, ASTRAL_REQUEST_CANCELED);
    ASSERT_EQ(status.result, ASTRAL_E_CANCELED);

    float vec[64] = {};
    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = static_cast<uint32_t>(sizeof(vec));
    ASSERT_EQ(astral_embed_collect(emb, ticket, out), ASTRAL_E_CANCELED);

    astral_embed_destroy(emb);
    astral_model_release(model);
    astral_shutdown();
}

TEST(embeddings_mock_queue_pressure) {
    constexpr uint32_t kInflight = 8;
    constexpr uint32_t kCanceledIndex = 3;

    AstralInit cfg{};
    cfg.reserve_bytes = 256ULL * 1024ULL * 1024ULL;
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralHandle model = load_mock_embedding_model();

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(dim, 0u);
    ASSERT_LE(dim, 64u);

    AstralHandle emb = 0;
    err = astral_embed_create(model, &emb);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* texts[kInflight] = {
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
    };
    uint64_t tickets[kInflight] = {};

    for (uint32_t i = 0; i < kInflight; ++i) {
        err = astral_embed_enqueue(emb, span_from_cstr(texts[i]), &tickets[i]);
        ASSERT_EQ(err, ASTRAL_OK);
        ASSERT_NE(tickets[i], 0ULL);
    }

    uint64_t overflow_ticket = 0;
    err = astral_embed_enqueue(emb, span_from_cstr("overflow"), &overflow_ticket);
    ASSERT_EQ(err, ASTRAL_E_BUSY);

    err = astral_embed_cancel(emb, tickets[kCanceledIndex]);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_embed_cancel(emb, tickets[kCanceledIndex]);
    ASSERT_EQ(err, ASTRAL_E_NOT_FOUND);

    uint64_t replacement_ticket = 0;
    err = astral_embed_enqueue(emb, span_from_cstr("replacement"), &replacement_ticket);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_NE(replacement_ticket, 0ULL);

    float vec[64] = {};
    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = static_cast<uint32_t>(sizeof(vec));

    uint32_t collected = 0;
    double sum_abs = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t offset = 0; offset < kInflight; ++offset) {
        const uint32_t i = kInflight - 1u - offset;
        if (i == kCanceledIndex) {
            continue;
        }
        std::memset(vec, 0, sizeof(vec));
        err = astral_embed_collect(emb, tickets[i], out);
        ASSERT_EQ(err, ASTRAL_OK);
        ASSERT_EQ(vec[0], mock_embedding_first_value(texts[i]));
        ASSERT_EQ(vec[1], mock_embedding_first_value(texts[i]) + 1.0f);
        sum_abs += vec[0] >= 0.0f ? vec[0] : -vec[0];
        ++collected;
    }

    std::memset(vec, 0, sizeof(vec));
    err = astral_embed_collect(emb, replacement_ticket, out);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(vec[0], mock_embedding_first_value("replacement"));
    sum_abs += vec[0] >= 0.0f ? vec[0] : -vec[0];
    ++collected;
    const auto end = std::chrono::steady_clock::now();

    err = astral_embed_collect(emb, tickets[0], out);
    ASSERT_EQ(err, ASTRAL_E_NOT_FOUND);
    err = astral_embed_collect(emb, tickets[kCanceledIndex], out);
    ASSERT_EQ(err, ASTRAL_E_CANCELED);

    uint64_t reuse_ticket = 0;
    err = astral_embed_enqueue(emb, span_from_cstr("reuse"), &reuse_ticket);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_NE(reuse_ticket, 0ULL);
    std::memset(vec, 0, sizeof(vec));
    err = astral_embed_collect(emb, reuse_ticket, out);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(vec[0], mock_embedding_first_value("reuse"));

    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();
    const double embeds_per_sec = seconds > 0.0 ? static_cast<double>(collected) / seconds : 0.0;
    std::printf("[embedding_mock_acceptance] batch=%u canceled=1 backpressure=busy seconds=%.6f embeds_per_sec=%.3f sum_abs=%.6f\n",
                collected,
                seconds,
                embeds_per_sec,
                sum_abs);
    ASSERT_EQ(collected, kInflight);
    ASSERT_GT(sum_abs, 0.0);
    ASSERT_GT(embeds_per_sec, 0.0);

    astral_embed_destroy(emb);
    astral_model_release(model);
    astral_shutdown();
}

TEST(embeddings_cpu_e2e_fixture_probe) {
    const char* model_path = find_test_model_path();
    if (model_path == nullptr) {
        SKIP_TEST("ASTRAL_TEST_EMBED_MODEL or ASTRAL_TEST_MODEL is required for CPU embedding fixture coverage");
    }

    AstralInit cfg{};
    cfg.reserve_bytes = 2ULL << 30;
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* backend = "cpu";

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    model_desc.n_ctx = 0;
    model_desc.n_batch = 0;
    model_desc.n_threads = 0;
    model_desc.embeddings_only = 1;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(dim, 0u);
    ASSERT_LT(dim, 8192u);

    AstralHandle emb = 0;
    err = astral_embed_create(model, &emb);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* text = "hi";
    AstralSpanU8 text_span{};
    text_span.data = reinterpret_cast<const uint8_t*>(text);
    text_span.len = static_cast<uint32_t>(std::strlen(text));

    uint64_t ticket = 0;
    err = astral_embed_enqueue(emb, text_span, &ticket);
    ASSERT_EQ(err, ASTRAL_OK);

    static float vec[8192];
    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = static_cast<uint32_t>(sizeof(vec));

    err = astral_embed_collect(emb, ticket, out);
    ASSERT_EQ(err, ASTRAL_OK);

    // Basic sanity: not all zeros.
    double sum_abs = 0.0;
    for (uint32_t i = 0; i < dim; ++i) {
        const float v = vec[i];
        sum_abs += (v >= 0.0f) ? v : -v;
    }
    std::printf("[embedding_probe] backend=cpu model=%s dim=%u sum_abs=%.6f first=%.6f\n",
                model_path,
                dim,
                sum_abs,
                vec[0]);
    ASSERT_GT(sum_abs, 0.0);

    astral_embed_destroy(emb);
    astral_model_release(model);
    astral_shutdown();
}

TEST(embeddings_cpu_throughput_fixture_probe) {
    const char* model_path = find_required_embedding_model_path();
    if (model_path == nullptr) {
        SKIP_TEST("ASTRAL_TEST_EMBED_MODEL is required for CPU embedding throughput coverage");
    }

    AstralInit cfg{};
    cfg.reserve_bytes = 2ULL << 30;
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path = span_from_cstr(model_path);
    model_desc.backend_name = span_from_cstr("cpu");
    model_desc.n_ctx = 0;
    model_desc.n_batch = 0;
    model_desc.n_threads = 4;
    model_desc.embeddings_only = 1;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(dim, 0u);
    ASSERT_LE(dim, 8192u);

    AstralHandle emb = 0;
    err = astral_embed_create(model, &emb);
    ASSERT_EQ(err, ASTRAL_OK);

    static float vec[8192];
    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec);
    out.len = static_cast<uint32_t>(sizeof(vec));

    const char* texts[] = {
        "search result title",
        "gameplay quest state",
        "player inventory note",
        "dialogue memory shard",
    };
    constexpr uint32_t kTextCount = static_cast<uint32_t>(sizeof(texts) / sizeof(texts[0]));
    const uint32_t iters = embedding_throughput_iters();

    double sum_abs = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iters; ++i) {
        uint64_t ticket = 0;
        err = astral_embed_enqueue(emb, span_from_cstr(texts[i % kTextCount]), &ticket);
        ASSERT_EQ(err, ASTRAL_OK);
        ASSERT_NE(ticket, 0ULL);

        err = astral_embed_collect(emb, ticket, out);
        ASSERT_EQ(err, ASTRAL_OK);

        for (uint32_t j = 0; j < dim; ++j) {
            const float v = vec[j];
            sum_abs += (v >= 0.0f) ? v : -v;
        }
    }
    const auto end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();
    const double embeds_per_sec = seconds > 0.0 ? static_cast<double>(iters) / seconds : 0.0;
    std::printf("[embedding_throughput] backend=cpu model=%s dim=%u iters=%u seconds=%.6f embeds_per_sec=%.3f sum_abs=%.6f\n",
                model_path,
                dim,
                iters,
                seconds,
                embeds_per_sec,
                sum_abs);
    ASSERT_GT(sum_abs, 0.0);
    ASSERT_GT(embeds_per_sec, 0.0);

    astral_embed_destroy(emb);
    astral_model_release(model);
    astral_shutdown();
}
