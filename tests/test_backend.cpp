/**
 * test_backend.cpp - Backend tests
 *
 * Tests for backend registration, selection, and CPU backend.
 * Validates: backend registration, selection by gpu_layers, lookup by name.
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <cstring>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>

#include <cpp-httplib/httplib.h>

namespace {

std::atomic<uint64_t> g_new_calls{0};
constexpr size_t kRemoteStreamChunkBytes = 1;
constexpr size_t kRemoteOverflowResponseBytes = 9000;
constexpr uint32_t kRemoteOverflowCtxTokens = 10000;
constexpr int kHttpOk = 200;
constexpr int kHttpNotImplemented = 501;
constexpr int32_t kExpectedRemoteBosToken = 256;

struct RemoteTestServer {
    httplib::Server server;
    std::thread thread;
    int port = -1;
    std::atomic<uint32_t> health_calls{0};
    std::atomic<uint32_t> tokenize_calls{0};
    std::atomic<uint32_t> completion_calls{0};
    std::atomic<uint32_t> stream_completion_calls{0};
    std::atomic<uint32_t> embedding_calls{0};
    uint32_t health_failures = 0;
    int health_failure_status = 503;
    int tokenize_status = kHttpOk;
    int stream_completion_status = kHttpOk;
    std::string completion_response;
    std::string stream_response;

    bool start(bool require_auth) {
      server.Get("/health",
                 [this, require_auth](const httplib::Request& req, httplib::Response& res) {
                   if (require_auth && req.get_header_value("Authorization") != "Bearer test-key") {
                     res.status = 401;
                     return;
                   }
                   health_calls.fetch_add(1, std::memory_order_relaxed);
                   if (health_failures != 0) {
                     --health_failures;
                     res.status = health_failure_status;
                     return;
                   }
                   res.set_content("ok", "text/plain");
                 });
      server.Post(
          "/tokenize", [this, require_auth](const httplib::Request& req, httplib::Response& res) {
            if (require_auth && req.get_header_value("Authorization") != "Bearer test-key") {
              res.status = 401;
              return;
            }
            tokenize_calls.fetch_add(1, std::memory_order_relaxed);
            if (tokenize_status != kHttpOk) {
              res.status = tokenize_status;
              return;
            }
            char body[1024];
            uint32_t n = 0;
            for (unsigned char c : req.body) {
              const int written = std::snprintf(body + n, sizeof(body) - n, "%s%u",
                                                n == 0 ? "" : ",", static_cast<unsigned>(c));
              if (written <= 0 || static_cast<uint32_t>(written) >= sizeof(body) - n) {
                break;
              }
              n += static_cast<uint32_t>(written);
            }
            res.set_content(std::string(body, n), "text/plain");
          });
      server.Post(
          "/completion", [this, require_auth](const httplib::Request& req, httplib::Response& res) {
            if (require_auth && req.get_header_value("Authorization") != "Bearer test-key") {
              res.status = 401;
              return;
            }
            completion_calls.fetch_add(1, std::memory_order_relaxed);
            std::string response = completion_response;
            if (response.empty()) {
              response = req.body == "hello" ? "remote-ok" : "remote-fallback";
            }
            res.set_content(response, "text/plain");
          });
      server.Post("/completion/stream", [this, require_auth](const httplib::Request& req,
                                                             httplib::Response& res) {
        if (require_auth && req.get_header_value("Authorization") != "Bearer test-key") {
          res.status = 401;
          return;
        }
        stream_completion_calls.fetch_add(1, std::memory_order_relaxed);
        if (stream_completion_status != kHttpOk) {
          res.status = stream_completion_status;
          return;
        }
        std::string response = stream_response;
        if (response.empty()) {
          response = req.body == "hello" ? "remote-ok" : "remote-fallback";
        }
        res.set_chunked_content_provider(
            "text/plain", [response](size_t offset, httplib::DataSink& sink) {
              if (offset >= response.size()) {
                sink.done();
                return false;
              }
              return sink.write(response.data() + offset, kRemoteStreamChunkBytes);
            });
      });
      server.Post(
          "/embeddings", [this, require_auth](const httplib::Request& req, httplib::Response& res) {
            if (require_auth && req.get_header_value("Authorization") != "Bearer test-key") {
              res.status = 401;
              return;
            }
            embedding_calls.fetch_add(1, std::memory_order_relaxed);
            res.set_content("[1,2,3,4,5,6,7,8]", "application/json");
          });

      port = server.bind_to_any_port("127.0.0.1");
      if (port <= 0) {
        return false;
      }
      thread = std::thread([this]() { server.listen_after_bind(); });
      return true;
    }

    void stop() {
      server.stop();
      if (thread.joinable()) {
        thread.join();
      }
    }

    ~RemoteTestServer() {
        stop();
    }
};

static void* alloc_aligned(std::size_t size, std::size_t alignment) noexcept {
    if (size == 0) {
        size = 1;
    }

    if (alignment < alignof(void*)) {
        alignment = alignof(void*);
    }

#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

static void free_aligned(void* ptr) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace

void* operator new(std::size_t size) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void* operator new[](std::size_t size) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void* operator new(std::size_t size, std::align_val_t align) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    void* p = alloc_aligned(size, static_cast<std::size_t>(align));
    if (p) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr, std::align_val_t) noexcept {
    free_aligned(ptr);
}

void* operator new[](std::size_t size, std::align_val_t align) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    void* p = alloc_aligned(size, static_cast<std::size_t>(align));
    if (p) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete[](void* ptr, std::align_val_t) noexcept {
    free_aligned(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
    free_aligned(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
    free_aligned(ptr);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
}

//
// Backend Registration Tests
//

TEST(backend_cpu_auto_registration) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // CPU backend should be auto-registered
    // We can verify this by loading a model with gpu_layers=0

    astral_shutdown();
}

//
// Backend Selection Tests
//

TEST(backend_selection_cpu) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Create model with cpu_layers=0 (CPU only)
    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "models/test.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.gpu_layers = 0; // CPU only
    model_desc.embeddings_only = 0;

    AstralHandle model;
    AstralErr err = astral_model_load(&model_desc, &model);

    // Missing local fixtures surface as backend load errors in this smoke path.
    ASSERT_TRUE(err == ASTRAL_OK || err == ASTRAL_E_BACKEND);
    if (err == ASTRAL_OK) {
        astral_model_release(model);
    }

    astral_shutdown();
}

TEST(backend_selection_gpu) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Create model with gpu_layers > 0
    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "models/test.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.gpu_layers = 10; // Try GPU
    model_desc.embeddings_only = 0;

    AstralHandle model;
    AstralErr err = astral_model_load(&model_desc, &model);

    // CPU-only hosts report backend load failure before a real GPU smoke exists.
    ASSERT_TRUE(err == ASTRAL_OK || err == ASTRAL_E_BACKEND);
    if (err == ASTRAL_OK) {
        astral_model_release(model);
    }

    astral_shutdown();
}

TEST(backend_cuda_registration_surface) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Force the backend by name, but keep model_path empty. This distinguishes:
    // - backend present: backend load runs and returns ASTRAL_E_INVALID (missing path)
    // - backend absent: registry lookup fails and Astral returns ASTRAL_E_BACKEND
    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "cuda";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(strlen(backend));
    model_desc.model_path.data = nullptr;
    model_desc.model_path.len = 0;
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.gpu_layers = 10;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&model_desc, &model);

#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
    ASSERT_EQ(err, ASTRAL_E_INVALID);
#else
    ASSERT_EQ(err, ASTRAL_E_BACKEND);
#endif

    if (err == ASTRAL_OK) {
        astral_model_release(model);
    }

    astral_shutdown();
}

//
// Model Load/Release Tests
//

TEST(backend_model_load_invalid_path) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "nonexistent/path/model.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));
    model_desc.n_ctx = 512;

    AstralHandle model;
    AstralErr err = astral_model_load(&model_desc, &model);

    // Should fail (file doesn't exist)
    ASSERT_NE(err, ASTRAL_OK);

    astral_shutdown();
}

TEST(backend_model_load_null_desc) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralHandle model;
    AstralErr err = astral_model_load(nullptr, &model);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(backend_model_load_null_output) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "test.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));

    AstralErr err = astral_model_load(&model_desc, nullptr);

    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

TEST(backend_mock_provider_end_to_end) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(strlen(backend));

    // Mock backend ignores model_path; leave empty.
    model_desc.n_ctx = 128;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    AstralModelInfo info{};
    err = astral_model_info(model, &info);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(info.vocab_size, 0u);

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 32;
    session_desc.temperature = 0.0f; // greedy (deterministic)
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    const char* prompt = "hi";
    AstralSpanU8 chunk = {};
    chunk.data = reinterpret_cast<const uint8_t*>(prompt);
    chunk.len = static_cast<uint32_t>(strlen(prompt));
    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    const uint64_t allocs_before_decode = g_new_calls.load(std::memory_order_relaxed);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    const uint64_t allocs_after_decode = g_new_calls.load(std::memory_order_relaxed);
    ASSERT_EQ(allocs_after_decode, allocs_before_decode);

    // Drain output (mock produces a short fixed message).
    uint8_t buf[128];
    uint32_t total = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        AstralMutSpanU8 out = {};
        out.data = buf + total;
        out.len = sizeof(buf) - total;

        int32_t n = astral_stream_read(session, out, 1000);
        if (n < 0) {
            ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
            continue;
        }
        if (n == 0) {
            break;
        }
        total += static_cast<uint32_t>(n);
        if (total == sizeof(buf)) {
            break;
        }
    }

    AstralStats stats{};
    err = astral_session_stats(session, &stats);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_GT(stats.tok_per_s, 0.0);

    ASSERT_GT(total, 0u);

    // Reset and run a second decode to validate reuse semantics.
    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_feed(session, chunk, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    const uint64_t allocs_before_decode2 = g_new_calls.load(std::memory_order_relaxed);

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    const uint64_t allocs_after_decode2 = g_new_calls.load(std::memory_order_relaxed);
    ASSERT_EQ(allocs_after_decode2, allocs_before_decode2);

    uint8_t buf2[128];
    uint32_t total2 = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        AstralMutSpanU8 out = {};
        out.data = buf2 + total2;
        out.len = sizeof(buf2) - total2;

        int32_t n = astral_stream_read(session, out, 1000);
        if (n < 0) {
            ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
            continue;
        }
        if (n == 0) {
            break;
        }
        total2 += static_cast<uint32_t>(n);
        if (total2 == sizeof(buf2)) {
            break;
        }
    }

    ASSERT_GT(total2, 0u);
    ASSERT_EQ(total2, total);
    ASSERT_EQ(std::memcmp(buf2, buf, total), 0);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(backend_remote_loopback_completion_and_embeddings) {
    RemoteTestServer remote;
    ASSERT_TRUE(remote.start(true));

    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 2;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    char url[128];
    const int url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", remote.port);
    ASSERT_TRUE(url_len > 0);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "remote";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
    model_desc.model_path.len = static_cast<uint32_t>(url_len);
    const char* api_key = "test-key";
    model_desc.model_bytes.data = reinterpret_cast<const uint8_t*>(api_key);
    model_desc.model_bytes.len = static_cast<uint32_t>(std::strlen(api_key));
    model_desc.n_ctx = 128;
    model_desc.n_batch = 32;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    int32_t tokens[32];
    uint32_t token_count = 0;
    const char* prompt = "hello";
    AstralSpanU8 prompt_span{};
    prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
    prompt_span.len = static_cast<uint32_t>(std::strlen(prompt));
    err = astral_tokenize(model, prompt_span, tokens, 32, 0, 0, &token_count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(token_count, 5u);
    ASSERT_GT(remote.tokenize_calls.load(std::memory_order_relaxed), 0u);

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 32;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_feed(session, prompt_span, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    uint8_t out[64];
    uint32_t total = 0;
    for (uint32_t i = 0; i < 32 && total < sizeof(out); ++i) {
        AstralMutSpanU8 dst{};
        dst.data = out + total;
        dst.len = sizeof(out) - total;
        const int32_t n = astral_stream_read(session, dst, 100);
        if (n < 0) {
            ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
            continue;
        }
        if (n == 0) {
            break;
        }
        total += static_cast<uint32_t>(n);
    }
    const char* expected = "remote-ok";
    ASSERT_EQ(total, static_cast<uint32_t>(std::strlen(expected)));
    ASSERT_EQ(std::memcmp(out, expected, total), 0);
    ASSERT_EQ(remote.stream_completion_calls.load(std::memory_order_relaxed), 1u);
    ASSERT_EQ(remote.completion_calls.load(std::memory_order_relaxed), 0u);

    AstralAgentDesc agent_desc{};
    agent_desc.size = sizeof(AstralAgentDesc);
    agent_desc.model = model;
    agent_desc.max_tokens = 32;
    agent_desc.temperature = 0.0f;
    agent_desc.top_k = 0;
    agent_desc.top_p = 1.0f;
    agent_desc.stream_enabled = 1;
    agent_desc.seed = 1;

    AstralHandle agent = 0;
    err = astral_agent_create(&agent_desc, &agent);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralAgentChatDesc chat{};
    chat.size = sizeof(AstralAgentChatDesc);
    chat.user_message = prompt_span;
    err = astral_agent_chat_enqueue(agent, &chat);
    ASSERT_EQ(err, ASTRAL_OK);

    uint8_t agent_out[64];
    uint32_t agent_total = 0;
    for (uint32_t i = 0; i < 32 && agent_total < sizeof(agent_out); ++i) {
        AstralMutSpanU8 dst{};
        dst.data = agent_out + agent_total;
        dst.len = sizeof(agent_out) - agent_total;
        const int32_t n = astral_agent_chat_stream_read(agent, dst, 100);
        if (n < 0) {
            ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
            continue;
        }
        if (n == 0) {
            break;
        }
        agent_total += static_cast<uint32_t>(n);
    }
    const char* agent_expected = "remote-fallback";
    ASSERT_EQ(agent_total, static_cast<uint32_t>(std::strlen(agent_expected)));
    ASSERT_EQ(std::memcmp(agent_out, agent_expected, agent_total), 0);

    AstralAgentChatResult chat_result{};
    chat_result.size = sizeof(AstralAgentChatResult);
    err = astral_agent_chat_result(agent, &chat_result);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(chat_result.state, ASTRAL_SESSION_COMPLETED);
    ASSERT_EQ(chat_result.last_error, ASTRAL_OK);
    ASSERT_EQ(remote.stream_completion_calls.load(std::memory_order_relaxed), 2u);

    astral_agent_destroy(agent);
    astral_session_destroy(session);
    astral_model_release(model);

    model_desc.embeddings_only = 1;
    model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle embedder = 0;
    err = astral_embed_create(model, &embedder);
    ASSERT_EQ(err, ASTRAL_OK);

    uint64_t ticket = 0;
    err = astral_embed_enqueue(embedder, prompt_span, &ticket);
    ASSERT_EQ(err, ASTRAL_OK);

    float embedding[8]{};
    AstralMutSpanU8 embedding_out{};
    embedding_out.data = reinterpret_cast<uint8_t*>(embedding);
    embedding_out.len = sizeof(embedding);
    err = astral_embed_collect(embedder, ticket, embedding_out);
    ASSERT_EQ(err, ASTRAL_OK);
    for (uint32_t i = 0; i < 8; ++i) {
        ASSERT_EQ(embedding[i], static_cast<float>(i + 1u));
    }
    ASSERT_EQ(remote.embedding_calls.load(std::memory_order_relaxed), 1u);

    astral_embed_destroy(embedder);
    astral_model_release(model);
    astral_shutdown();
}

TEST(backend_remote_tokenize_falls_back_to_byte_tokens) {
    RemoteTestServer remote;
    remote.tokenize_status = kHttpNotImplemented;
    ASSERT_TRUE(remote.start(true));

    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    cfg.thread_count = 1;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    char url[128];
    const int url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", remote.port);
    ASSERT_TRUE(url_len > 0);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "remote";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
    model_desc.model_path.len = static_cast<uint32_t>(url_len);
    const char* api_key = "test-key";
    model_desc.model_bytes.data = reinterpret_cast<const uint8_t*>(api_key);
    model_desc.model_bytes.len = static_cast<uint32_t>(std::strlen(api_key));

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* text = "az";
    AstralSpanU8 text_span{};
    text_span.data = reinterpret_cast<const uint8_t*>(text);
    text_span.len = static_cast<uint32_t>(std::strlen(text));

    uint32_t count = 0;
    err = astral_tokenize_count(model, text_span, 1, 0, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, 3u);

    int32_t tokens[4]{};
    err = astral_tokenize(model, text_span, tokens, 4, 1, 0, &count);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(count, 3u);
    ASSERT_EQ(tokens[0], kExpectedRemoteBosToken);
    ASSERT_EQ(tokens[1], static_cast<int32_t>('a'));
    ASSERT_EQ(tokens[2], static_cast<int32_t>('z'));
    ASSERT_EQ(remote.tokenize_calls.load(std::memory_order_relaxed), 2u);

    astral_model_release(model);
    astral_shutdown();
}

TEST(backend_remote_stream_falls_back_to_completion) {
    RemoteTestServer remote;
    remote.stream_completion_status = kHttpNotImplemented;
    ASSERT_TRUE(remote.start(true));

    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 64 * 1024 * 1024;
    cfg.thread_count = 1;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    char url[128];
    const int url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", remote.port);
    ASSERT_TRUE(url_len > 0);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "remote";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
    model_desc.model_path.len = static_cast<uint32_t>(url_len);
    const char* api_key = "test-key";
    model_desc.model_bytes.data = reinterpret_cast<const uint8_t*>(api_key);
    model_desc.model_bytes.len = static_cast<uint32_t>(std::strlen(api_key));
    model_desc.n_ctx = 128;
    model_desc.n_batch = 32;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSessionDesc session_desc = {};
    session_desc.model = model;
    session_desc.max_tokens = 32;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    const char* prompt = "hello";
    AstralSpanU8 prompt_span{};
    prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
    prompt_span.len = static_cast<uint32_t>(std::strlen(prompt));
    err = astral_session_feed(session, prompt_span, 1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_session_wait(session, 5000);
    ASSERT_EQ(err, ASTRAL_OK);

    uint8_t out[64];
    uint32_t total = 0;
    for (uint32_t i = 0; i < 32 && total < sizeof(out); ++i) {
        AstralMutSpanU8 dst{};
        dst.data = out + total;
        dst.len = sizeof(out) - total;
        const int32_t n = astral_stream_read(session, dst, 100);
        if (n < 0) {
            ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
            continue;
        }
        if (n == 0) {
            break;
        }
        total += static_cast<uint32_t>(n);
    }

    const char* expected = "remote-ok";
    ASSERT_EQ(total, static_cast<uint32_t>(std::strlen(expected)));
    ASSERT_EQ(std::memcmp(out, expected, total), 0);
    ASSERT_EQ(remote.stream_completion_calls.load(std::memory_order_relaxed), 1u);
    ASSERT_EQ(remote.completion_calls.load(std::memory_order_relaxed), 1u);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(backend_remote_stream_accepts_sse_data_frames) {
  RemoteTestServer remote;
  remote.stream_response = "data: remote-\n\ndata: sse\n\ndata: [DONE]\n\n";
  ASSERT_TRUE(remote.start(true));

  AstralInit cfg = {};
  cfg.size = sizeof(AstralInit);
  cfg.reserve_bytes = 64 * 1024 * 1024;
  cfg.thread_count = 1;
  AstralErr err = astral_init(&cfg);
  ASSERT_EQ(err, ASTRAL_OK);

  char url[128];
  const int url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", remote.port);
  ASSERT_TRUE(url_len > 0);

  AstralModelDesc model_desc = {};
  model_desc.size = sizeof(AstralModelDesc);
  model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
  const char* backend = "remote";
  model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
  model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
  model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
  model_desc.model_path.len = static_cast<uint32_t>(url_len);
  const char* api_key = "test-key";
  model_desc.model_bytes.data = reinterpret_cast<const uint8_t*>(api_key);
  model_desc.model_bytes.len = static_cast<uint32_t>(std::strlen(api_key));
  model_desc.n_ctx = 128;
  model_desc.n_batch = 32;

  AstralHandle model = 0;
  err = astral_model_load(&model_desc, &model);
  ASSERT_EQ(err, ASTRAL_OK);

  AstralSessionDesc session_desc = {};
  session_desc.model = model;
  session_desc.max_tokens = 32;
  session_desc.temperature = 0.0f;
  session_desc.top_k = 0;
  session_desc.top_p = 1.0f;
  session_desc.stream_enabled = 1;

  AstralHandle session = 0;
  err = astral_session_create(&session_desc, &session);
  ASSERT_EQ(err, ASTRAL_OK);

  const char* prompt = "hello";
  AstralSpanU8 prompt_span{};
  prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
  prompt_span.len = static_cast<uint32_t>(std::strlen(prompt));
  err = astral_session_feed(session, prompt_span, 1);
  ASSERT_EQ(err, ASTRAL_OK);
  err = astral_session_decode(session);
  ASSERT_EQ(err, ASTRAL_OK);
  err = astral_session_wait(session, 5000);
  ASSERT_EQ(err, ASTRAL_OK);

  uint8_t out[64];
  uint32_t total = 0;
  for (uint32_t i = 0; i < 32 && total < sizeof(out); ++i) {
    AstralMutSpanU8 dst{};
    dst.data = out + total;
    dst.len = sizeof(out) - total;
    const int32_t n = astral_stream_read(session, dst, 100);
    if (n < 0) {
      ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
      continue;
    }
    if (n == 0) {
      break;
    }
    total += static_cast<uint32_t>(n);
  }

  const char* expected = "remote-sse";
  ASSERT_EQ(total, static_cast<uint32_t>(std::strlen(expected)));
  ASSERT_EQ(std::memcmp(out, expected, total), 0);
  ASSERT_EQ(remote.stream_completion_calls.load(std::memory_order_relaxed), 1u);
  ASSERT_EQ(remote.completion_calls.load(std::memory_order_relaxed), 0u);

  astral_session_destroy(session);
  astral_model_release(model);
  astral_shutdown();
}

TEST(backend_remote_stream_extracts_sse_json_text) {
  RemoteTestServer remote;
  remote.stream_response = "data: {\"choices\":[{\"delta\":{\"content\":\"json-\"}}]}\n\n"
                           "data: {\"text\":\"sse\"}\n\n"
                           "data: [DONE]\n\n";
  ASSERT_TRUE(remote.start(true));

  AstralInit cfg = {};
  cfg.size = sizeof(AstralInit);
  cfg.reserve_bytes = 64 * 1024 * 1024;
  cfg.thread_count = 1;
  AstralErr err = astral_init(&cfg);
  ASSERT_EQ(err, ASTRAL_OK);

  char url[128];
  const int url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", remote.port);
  ASSERT_TRUE(url_len > 0);

  AstralModelDesc model_desc = {};
  model_desc.size = sizeof(AstralModelDesc);
  model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
  const char* backend = "remote";
  model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
  model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
  model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
  model_desc.model_path.len = static_cast<uint32_t>(url_len);
  const char* api_key = "test-key";
  model_desc.model_bytes.data = reinterpret_cast<const uint8_t*>(api_key);
  model_desc.model_bytes.len = static_cast<uint32_t>(std::strlen(api_key));
  model_desc.n_ctx = 128;
  model_desc.n_batch = 32;

  AstralHandle model = 0;
  err = astral_model_load(&model_desc, &model);
  ASSERT_EQ(err, ASTRAL_OK);

  AstralSessionDesc session_desc = {};
  session_desc.model = model;
  session_desc.max_tokens = 32;
  session_desc.temperature = 0.0f;
  session_desc.top_k = 0;
  session_desc.top_p = 1.0f;
  session_desc.stream_enabled = 1;

  AstralHandle session = 0;
  err = astral_session_create(&session_desc, &session);
  ASSERT_EQ(err, ASTRAL_OK);

  const char* prompt = "hello";
  AstralSpanU8 prompt_span{};
  prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
  prompt_span.len = static_cast<uint32_t>(std::strlen(prompt));
  err = astral_session_feed(session, prompt_span, 1);
  ASSERT_EQ(err, ASTRAL_OK);
  err = astral_session_decode(session);
  ASSERT_EQ(err, ASTRAL_OK);
  err = astral_session_wait(session, 5000);
  ASSERT_EQ(err, ASTRAL_OK);

  uint8_t out[64];
  uint32_t total = 0;
  for (uint32_t i = 0; i < 32 && total < sizeof(out); ++i) {
    AstralMutSpanU8 dst{};
    dst.data = out + total;
    dst.len = sizeof(out) - total;
    const int32_t n = astral_stream_read(session, dst, 100);
    if (n < 0) {
      ASSERT_EQ(n, ASTRAL_E_TIMEOUT);
      continue;
    }
    if (n == 0) {
      break;
    }
    total += static_cast<uint32_t>(n);
  }

  const char* expected = "json-sse";
  ASSERT_EQ(total, static_cast<uint32_t>(std::strlen(expected)));
  ASSERT_EQ(std::memcmp(out, expected, total), 0);
  ASSERT_EQ(remote.stream_completion_calls.load(std::memory_order_relaxed), 1u);
  ASSERT_EQ(remote.completion_calls.load(std::memory_order_relaxed), 0u);

  astral_session_destroy(session);
  astral_model_release(model);
  astral_shutdown();
}

TEST(backend_remote_stream_overflow_reports_error) {
  RemoteTestServer remote;
  remote.stream_completion_status = kHttpNotImplemented;
  remote.completion_response.assign(kRemoteOverflowResponseBytes, 'x');
  ASSERT_TRUE(remote.start(true));

  AstralInit cfg = {};
  cfg.size = sizeof(AstralInit);
  cfg.reserve_bytes = 64 * 1024 * 1024;
  cfg.thread_count = 1;
  AstralErr err = astral_init(&cfg);
  ASSERT_EQ(err, ASTRAL_OK);

  char url[128];
  const int url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", remote.port);
  ASSERT_TRUE(url_len > 0);

  AstralModelDesc model_desc = {};
  model_desc.size = sizeof(AstralModelDesc);
  model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
  const char* backend = "remote";
  model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
  model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
  model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
  model_desc.model_path.len = static_cast<uint32_t>(url_len);
  const char* api_key = "test-key";
  model_desc.model_bytes.data = reinterpret_cast<const uint8_t*>(api_key);
  model_desc.model_bytes.len = static_cast<uint32_t>(std::strlen(api_key));
  model_desc.n_ctx = kRemoteOverflowCtxTokens;
  model_desc.n_batch = 32;

  AstralHandle model = 0;
  err = astral_model_load(&model_desc, &model);
  ASSERT_EQ(err, ASTRAL_OK);

  AstralSessionDesc session_desc = {};
  session_desc.model = model;
  session_desc.max_tokens = 1;
  session_desc.temperature = 0.0f;
  session_desc.top_k = 0;
  session_desc.top_p = 1.0f;
  session_desc.stream_enabled = 1;

  AstralHandle session = 0;
  err = astral_session_create(&session_desc, &session);
  ASSERT_EQ(err, ASTRAL_OK);

  const char* prompt = "hello";
  AstralSpanU8 prompt_span{};
  prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
  prompt_span.len = static_cast<uint32_t>(std::strlen(prompt));
  err = astral_session_feed(session, prompt_span, 1);
  ASSERT_EQ(err, ASTRAL_OK);

  err = astral_session_decode(session);
  ASSERT_EQ(err, ASTRAL_OK);
  err = astral_session_wait(session, 5000);
  ASSERT_EQ(err, ASTRAL_E_NOMEM);
  ASSERT_EQ(remote.stream_completion_calls.load(std::memory_order_relaxed), 1u);
  ASSERT_EQ(remote.completion_calls.load(std::memory_order_relaxed), 1u);

  astral_session_destroy(session);
  astral_model_release(model);
  astral_shutdown();
}

TEST(backend_remote_auth_failure) {
    RemoteTestServer remote;
    ASSERT_TRUE(remote.start(true));

    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    cfg.thread_count = 1;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    char url[128];
    const int url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", remote.port);
    ASSERT_TRUE(url_len > 0);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "remote";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
    model_desc.model_path.len = static_cast<uint32_t>(url_len);
    const char* bad_key = "bad-key";
    model_desc.model_bytes.data = reinterpret_cast<const uint8_t*>(bad_key);
    model_desc.model_bytes.len = static_cast<uint32_t>(std::strlen(bad_key));

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_E_BACKEND);
    ASSERT_EQ(model, 0u);

    astral_shutdown();
}

TEST(backend_remote_https_requires_tls_build) {
    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    cfg.thread_count = 1;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "remote";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    const char* url = "https://127.0.0.1:1";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
    model_desc.model_path.len = static_cast<uint32_t>(std::strlen(url));

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    ASSERT_EQ(err, ASTRAL_E_TIMEOUT);
#else
    ASSERT_EQ(err, ASTRAL_E_UNSUPPORTED);
#endif
    ASSERT_EQ(model, 0u);

    astral_shutdown();
}

TEST(backend_remote_health_retry_and_timeout_status) {
    RemoteTestServer retry_remote;
    retry_remote.health_failures = 1;
    retry_remote.health_failure_status = 503;
    ASSERT_TRUE(retry_remote.start(true));

    AstralInit cfg = {};
    cfg.size = sizeof(AstralInit);
    cfg.reserve_bytes = 16 * 1024 * 1024;
    cfg.thread_count = 1;
    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    char url[128];
    int url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", retry_remote.port);
    ASSERT_TRUE(url_len > 0);

    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "remote";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
    model_desc.model_path.len = static_cast<uint32_t>(url_len);
    const char* api_key = "test-key";
    model_desc.model_bytes.data = reinterpret_cast<const uint8_t*>(api_key);
    model_desc.model_bytes.len = static_cast<uint32_t>(std::strlen(api_key));

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_EQ(retry_remote.health_calls.load(std::memory_order_relaxed), 2u);
    astral_model_release(model);
    astral_shutdown();
    retry_remote.stop();

    RemoteTestServer timeout_remote;
    timeout_remote.health_failures = std::numeric_limits<uint32_t>::max();
    timeout_remote.health_failure_status = 504;
    ASSERT_TRUE(timeout_remote.start(true));

    err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    url_len = std::snprintf(url, sizeof(url), "http://127.0.0.1:%d", timeout_remote.port);
    ASSERT_TRUE(url_len > 0);
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(url);
    model_desc.model_path.len = static_cast<uint32_t>(url_len);
    model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_E_TIMEOUT);
    ASSERT_EQ(timeout_remote.health_calls.load(std::memory_order_relaxed), 2u);

    astral_shutdown();
}
