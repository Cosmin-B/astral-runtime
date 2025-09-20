/**
 * test_integration.cpp - End-to-end integration tests with a real GGUF model
 *
 * Tests full inference pipeline with a small GGUF model (default downloader: GPT-2 Q2_K).
 * Validates:
 * - Runtime initialization/shutdown
 * - Model loading
 * - Session creation/destruction
 * - Prompt feeding
 * - Token streaming
 * - UTF-8 output validation
 * - Performance statistics
 * - Resource cleanup
 */

#include "test_framework.hpp"
#include "astral_rt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <unistd.h>

// ============================================================================
// Test Helpers
// ============================================================================

/**
 * Get path to test model.
 * Returns path to a GGUF model (downloaded by model_downloader.sh).
 * Returns nullptr if model not found.
 */
static const char* get_test_model_path() {
    const uint64_t min_bytes = []() -> uint64_t {
        const char* v = std::getenv("ASTRAL_MODEL_MIN_BYTES");
        if (v == nullptr || v[0] == '\0') {
            // Keep this in sync with tests/model_downloader.sh default.
            return 70000000ULL;
        }
        char* end = nullptr;
        unsigned long long x = std::strtoull(v, &end, 10);
        if (end == v) {
            return 70000000ULL;
        }
        return static_cast<uint64_t>(x);
    }();

    const char* env_path = std::getenv("ASTRAL_TEST_MODEL");
    if (env_path != nullptr && env_path[0] != '\0') {
        struct stat st;
        // If user explicitly provides a model, accept smaller files as long as they're not tiny/truncated.
        if (stat(env_path, &st) == 0 && st.st_size > (10 * 1024 * 1024)) {
            fprintf(stdout, "[INFO] Using model from ASTRAL_TEST_MODEL: %s\n", env_path);
            return env_path;
        }
        fprintf(stderr, "[SKIP] ASTRAL_TEST_MODEL set but file missing/too small: %s\n", env_path);
        return nullptr;
    }

    // Try multiple paths (different working directories)
    static const char* paths[] = {
        // Default download (small).
        "tests/models/gpt2.Q2_K.gguf",           // From project root
        "../tests/models/gpt2.Q2_K.gguf",        // From build/*
        "../../tests/models/gpt2.Q2_K.gguf",
        "../../../tests/models/gpt2.Q2_K.gguf",
        "../../../../tests/models/gpt2.Q2_K.gguf",

        // Legacy files (still accepted if present).
        "tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../../../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "../../../../tests/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",

        "tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../../../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../../../../tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    };

    struct stat st;
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i) {
        if (stat(paths[i], &st) == 0 && static_cast<uint64_t>(st.st_size) >= min_bytes) {
            fprintf(stdout, "[INFO] Found model at: %s\n", paths[i]);
            return paths[i];
        }
    }

    fprintf(stderr, "[SKIP] Model not found (tried %zu paths, run model_downloader.sh first)\n",
            sizeof(paths) / sizeof(paths[0]));
    return nullptr;
}

/**
 * Validate UTF-8 byte sequence.
 * Returns true if valid, false otherwise.
 *
 * UTF-8 encoding rules:
 * - 1-byte: 0xxxxxxx
 * - 2-byte: 110xxxxx 10xxxxxx
 * - 3-byte: 1110xxxx 10xxxxxx 10xxxxxx
 * - 4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
 */
static bool validate_utf8(const uint8_t* data, uint32_t len) {
    if (!data || len == 0) {
        return true; // Empty string is valid UTF-8
    }

    uint32_t i = 0;
    while (i < len) {
        uint8_t byte = data[i];

        // 1-byte sequence (ASCII)
        if ((byte & 0x80) == 0x00) {
            i++;
            continue;
        }

        // 2-byte sequence
        if ((byte & 0xE0) == 0xC0) {
            if (i + 1 >= len) return false;
            if ((data[i + 1] & 0xC0) != 0x80) return false;
            i += 2;
            continue;
        }

        // 3-byte sequence
        if ((byte & 0xF0) == 0xE0) {
            if (i + 2 >= len) return false;
            if ((data[i + 1] & 0xC0) != 0x80) return false;
            if ((data[i + 2] & 0xC0) != 0x80) return false;
            i += 3;
            continue;
        }

        // 4-byte sequence
        if ((byte & 0xF8) == 0xF0) {
            if (i + 3 >= len) return false;
            if ((data[i + 1] & 0xC0) != 0x80) return false;
            if ((data[i + 2] & 0xC0) != 0x80) return false;
            if ((data[i + 3] & 0xC0) != 0x80) return false;
            i += 4;
            continue;
        }

        // Invalid UTF-8
        return false;
    }

    return true;
}

/**
 * Print UTF-8 text for debugging.
 * Safely handles non-printable characters.
 */
static void print_output(const char* label, const uint8_t* data, uint32_t len) {
    fprintf(stdout, "[DEBUG] %s (%u bytes): \"", label, len);

    for (uint32_t i = 0; i < len; ++i) {
        uint8_t c = data[i];

        // Print printable ASCII
        if (c >= 32 && c <= 126) {
            fputc(c, stdout);
        }
        // Print newlines
        else if (c == '\n') {
            fprintf(stdout, "\\n");
        }
        // Print tabs
        else if (c == '\t') {
            fprintf(stdout, "\\t");
        }
        // Print other UTF-8 bytes as-is (they're part of multi-byte sequences)
        else if (c >= 0x80) {
            fputc(c, stdout);
        }
        // Print control characters as hex
        else {
            fprintf(stdout, "\\x%02x", c);
        }
    }

    fprintf(stdout, "\"\n");
}

/**
 * Monotonic clock for time measurement (milliseconds).
 * Uses CLOCK_MONOTONIC on Linux/macOS, fallback to time() on other platforms.
 */
static double get_monotonic_time_ms() {
#if defined(__linux__) || defined(__APPLE__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
#else
    return (double)time(nullptr) * 1000.0;
#endif
}

// ============================================================================
// Integration Tests
// ============================================================================

/**
 * Test: Initialize and shutdown Astral runtime.
 * Validates basic runtime lifecycle.
 */
TEST(runtime_init_shutdown) {
    AstralInit cfg = {0};
    cfg.reserve_bytes = 2ULL << 30; // 2GB
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFF; // Any NUMA node
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_shutdown();
}

/**
 * Test: Initialize runtime with null config (should fail).
 */
TEST(runtime_init_null_config) {
    AstralErr err = astral_init(nullptr);
    ASSERT_EQ(err, ASTRAL_E_INVALID);
}

/**
 * Test: Get error string for all error codes.
 */
TEST(error_strings) {
    ASSERT_NE(astral_error_string(ASTRAL_OK), nullptr);
    ASSERT_NE(astral_error_string(ASTRAL_E_INVALID), nullptr);
    ASSERT_NE(astral_error_string(ASTRAL_E_NOMEM), nullptr);
    ASSERT_NE(astral_error_string(ASTRAL_E_BUSY), nullptr);
    ASSERT_NE(astral_error_string(ASTRAL_E_TIMEOUT), nullptr);
    ASSERT_NE(astral_error_string(ASTRAL_E_STATE), nullptr);
    ASSERT_NE(astral_error_string(ASTRAL_E_BACKEND), nullptr);
}

/**
 * Test: Full end-to-end inference pipeline.
 *
 * Pipeline:
 * 1. Initialize runtime
 * 2. Load GGUF model
 * 3. Create session
 * 4. Feed prompt: "Once upon a time"
 * 5. Decode tokens (generate up to 50 tokens)
 * 6. Validate output is valid UTF-8
 * 7. Check statistics (tokens/sec, time to first token)
 * 8. Clean up all resources
 */
TEST(e2e_inference) {
    // Get model path
    const char* model_path = get_test_model_path();
    if (!model_path) {
        fprintf(stderr, "[SKIP] test_e2e_inference: Model not found\n");
        return; // Skip test if model not available
    }

    // Initialize Astral
    fprintf(stdout, "[INFO] Initializing Astral runtime...\n");
    AstralInit cfg = {0};
    cfg.reserve_bytes = 2ULL << 30; // 2GB
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFF; // Any NUMA node
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Load model
    fprintf(stdout, "[INFO] Loading model: %s\n", model_path);
    double t_load_start = get_monotonic_time_ms();

    AstralModelDesc model_desc = {0};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = (const uint8_t*)model_path;
    model_desc.model_path.len = (uint32_t)strlen(model_path);
    model_desc.n_ctx = 2048;
    model_desc.n_batch = 512;
    model_desc.n_threads = 4;
    model_desc.gpu_layers = 0; // CPU only
    model_desc.embeddings_only = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    if (err != ASTRAL_OK) {
        fprintf(stderr, "[FAIL] astral_model_load failed: %s (%s)\n",
                astral_error_string(err),
                astral_last_error());
    }
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_NE(model, 0);
    ASSERT_TRUE(astral_handle_valid(model));

    double t_load_end = get_monotonic_time_ms();
    fprintf(stdout, "[INFO] Model loaded in %.2f ms\n", t_load_end - t_load_start);

    // Create session
    fprintf(stdout, "[INFO] Creating inference session...\n");
    AstralSessionDesc session_desc = {0};
    session_desc.model = model;
    session_desc.max_tokens = 50;
    session_desc.temperature = 0.8f;
    session_desc.top_k = 40;
    session_desc.top_p = 0.95f;
    session_desc.stream_enabled = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_NE(session, 0);
    ASSERT_TRUE(astral_handle_valid(session));

    // Feed prompt
    const char* prompt = "Once upon a time";
    fprintf(stdout, "[INFO] Feeding prompt: \"%s\"\n", prompt);

    AstralSpanU8 prompt_span = {0};
    prompt_span.data = (const uint8_t*)prompt;
    prompt_span.len = (uint32_t)strlen(prompt);

    err = astral_session_feed(session, prompt_span, 1); // finalize=1
    ASSERT_EQ(err, ASTRAL_OK);

    // Start decoding
    fprintf(stdout, "[INFO] Starting decoding...\n");
    double t_decode_start = get_monotonic_time_ms();

    err = astral_session_decode(session);
    ASSERT_EQ(err, ASTRAL_OK);

    // Read tokens from stream
    uint8_t output_buf[4096];
    uint32_t total_bytes = 0;
    int tokens_read = 0;
    double t_first_token = 0.0;
    bool first_token_received = false;

    fprintf(stdout, "[INFO] Reading tokens...\n");

    for (int i = 0; i < 50; ++i) {
        AstralMutSpanU8 out_span = {0};
        out_span.data = output_buf + total_bytes;
        out_span.len = sizeof(output_buf) - total_bytes;

        // Read with 5 second timeout per token
        int32_t bytes_read = astral_stream_read(session, out_span, 5000);

        if (bytes_read < 0) {
            // End of stream or error
            if (bytes_read == ASTRAL_E_TIMEOUT) {
                fprintf(stderr, "[WARN] Stream read timeout after %d tokens\n", tokens_read);
            }
            break;
        }

        if (bytes_read > 0) {
            total_bytes += bytes_read;
            tokens_read++;

            // Record time to first token
            if (!first_token_received) {
                t_first_token = get_monotonic_time_ms();
                first_token_received = true;
                fprintf(stdout, "[INFO] First token received in %.2f ms\n",
                        t_first_token - t_decode_start);
            }
        }
    }

    double t_decode_end = get_monotonic_time_ms();
    double total_time_s = (t_decode_end - t_decode_start) / 1000.0;

    fprintf(stdout, "[INFO] Decoding completed: %d tokens, %u bytes in %.2f ms\n",
            tokens_read, total_bytes, t_decode_end - t_decode_start);

    // Ensure decode thread is finished before querying stats.
    fprintf(stdout, "[INFO] Waiting for session completion...\n");
    err = astral_session_wait(session, 60000);
    ASSERT_EQ(err, ASTRAL_OK);

    // Validate UTF-8
    fprintf(stdout, "[INFO] Validating UTF-8 output...\n");
    ASSERT_TRUE(validate_utf8(output_buf, total_bytes));

    // Print output for debugging
    print_output("Generated text", output_buf, total_bytes);

    // Check we generated some output
    ASSERT_GT(total_bytes, 0);
    ASSERT_GT(tokens_read, 0);

    // Get statistics
    fprintf(stdout, "[INFO] Checking statistics...\n");
    AstralStats stats = {0};
    err = astral_session_stats(session, &stats);
    ASSERT_EQ(err, ASTRAL_OK);

    fprintf(stdout, "[INFO] Statistics:\n");
    fprintf(stdout, "  - Init time: %.2f ms\n", stats.t_init_ms);
    fprintf(stdout, "  - Time to first token: %.2f ms\n", stats.t_first_token_ms);
    fprintf(stdout, "  - Tokens per second: %.2f tok/s\n", stats.tok_per_s);
    fprintf(stdout, "  - Committed memory: %llu bytes (%.2f MB)\n",
            (unsigned long long)stats.bytes_committed,
            stats.bytes_committed / (1024.0 * 1024.0));
    fprintf(stdout, "  - Reserved memory: %llu bytes (%.2f MB)\n",
            (unsigned long long)stats.bytes_reserved,
            stats.bytes_reserved / (1024.0 * 1024.0));

    // Validate statistics are reasonable
    ASSERT_GT(stats.tok_per_s, 0.0);
    ASSERT_GT(stats.t_first_token_ms, 0.0);
    ASSERT_LT(stats.t_first_token_ms, 10000.0); // <10 seconds

    // Performance validation (relaxed for CI/slow hardware)
    // Expected: >10 tok/s on CPU, <2s to first token
    // Actual: Allow >1 tok/s, <5s to first token for CI
    if (stats.tok_per_s < 1.0) {
        fprintf(stderr, "[WARN] Token throughput is low: %.2f tok/s (expected >10 tok/s)\n",
                stats.tok_per_s);
    }

    if (stats.t_first_token_ms > 5000.0) {
        fprintf(stderr, "[WARN] Time to first token is high: %.2f ms (expected <2000 ms)\n",
                stats.t_first_token_ms);
    }

    // Clean up
    fprintf(stdout, "[INFO] Cleaning up resources...\n");
    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();

    fprintf(stdout, "[INFO] Test completed successfully!\n");
}

/**
 * Test: Deterministic sampling via explicit seed + session reset.
 *
 * Runs the same prompt twice with the same seed and asserts identical output.
 * Uses a single backend thread to reduce nondeterminism.
 */
TEST(cpu_seed_reset_deterministic) {
    const char* model_path = get_test_model_path();
    if (!model_path) {
        fprintf(stderr, "[SKIP] test_cpu_seed_reset_deterministic: Model not found\n");
        return;
    }

    AstralInit cfg = {0};
    cfg.reserve_bytes = 2ULL << 30; // 2GB
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFF;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle model = 0;
    AstralHandle session = 0;

    uint8_t out1[2048];
    uint8_t out2[2048];
    uint32_t out1_len = 0;
    uint32_t out2_len = 0;

    auto run_once = [&](uint8_t* out, uint32_t cap, uint32_t* out_len) {
        const char* prompt = "Once upon a time";
        AstralSpanU8 chunk = {};
        chunk.data = reinterpret_cast<const uint8_t*>(prompt);
        chunk.len = static_cast<uint32_t>(strlen(prompt));

        AstralErr e = astral_session_feed(session, chunk, 1);
        ASSERT_EQ(e, ASTRAL_OK);

        e = astral_session_decode(session);
        ASSERT_EQ(e, ASTRAL_OK);

        uint32_t total = 0;
        for (uint32_t i = 0; i < 1024; ++i) {
            AstralMutSpanU8 span = {};
            span.data = out + total;
            span.len = cap - total;

            const int32_t n = astral_stream_read(session, span, 1000);
            if (n == ASTRAL_E_TIMEOUT) {
                continue;
            }
            ASSERT_GE(n, 0);
            if (n == 0) {
                break;
            }

            total += static_cast<uint32_t>(n);
            if (total == cap) {
                break;
            }
        }

        e = astral_session_wait(session, 60000);
        ASSERT_EQ(e, ASTRAL_OK);

        *out_len = total;
    };

    AstralModelDesc model_desc = {0};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(model_path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.n_threads = 1;
    model_desc.gpu_layers = 0;

    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralSessionDesc session_desc = {0};
    session_desc.model = model;
    session_desc.max_tokens = 8;
    session_desc.temperature = 1.0f;
    session_desc.top_k = 40;
    session_desc.top_p = 0.95f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1234;

    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    run_once(out1, sizeof(out1), &out1_len);

    err = astral_session_reset(session, &session_desc);
    ASSERT_EQ(err, ASTRAL_OK);

    run_once(out2, sizeof(out2), &out2_len);

    ASSERT_EQ(out2_len, out1_len);
    ASSERT_EQ(std::memcmp(out1, out2, out1_len), 0);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(cpu_json_schema_grammar_capability_and_set) {
    const char* model_path = get_test_model_path();
    if (!model_path) {
        fprintf(stderr, "[SKIP] test_cpu_json_schema_grammar_capability_and_set: Model not found\n");
        return;
    }

    AstralInit cfg = {0};
    cfg.reserve_bytes = 2ULL << 30; // 2GB
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFF;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralModelDesc model_desc = {0};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(model_path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.n_threads = 1;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    AstralCaps caps = 0;
    err = astral_model_caps(model, &caps);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE((caps & ASTRAL_CAP_GRAMMAR_JSON_SCHEMA) != 0);

    AstralSessionDesc sd = {0};
    sd.model = model;
    sd.max_tokens = 8;
    sd.temperature = 0.0f;
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 0;
    sd.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    // Minimal schema: any JSON object.
    const char* schema = "{}";
    AstralSpanU8 schema_span{};
    schema_span.data = reinterpret_cast<const uint8_t*>(schema);
    schema_span.len = static_cast<uint32_t>(strlen(schema));
    err = astral_session_set_grammar_json_schema(session, schema_span);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

TEST(cpu_slots_smoke) {
    const char* model_path = get_test_model_path();
    if (!model_path) {
        fprintf(stderr, "[SKIP] test_cpu_slots_smoke: Model not found\n");
        return;
    }

    // Ensure llama context is created with multiple sequences enabled.
    setenv("ASTRAL_LLAMA_MAX_SLOTS", "2", 1);

    AstralInit cfg = {0};
    cfg.reserve_bytes = 2ULL << 30; // 2GB
    cfg.thread_count = 4;
    cfg.numa_node = 0xFFFFFFFF;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralModelDesc model_desc = {0};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(model_path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.n_threads = 1;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    AstralSessionDesc sd = {0};
    sd.model = model;
    sd.max_tokens = 4;
    sd.temperature = 0.0f;
    sd.top_k = 0;
    sd.top_p = 1.0f;
    sd.stream_enabled = 0;
    sd.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&sd, &session);
    ASSERT_EQ(err, ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(session));

    err = astral_session_set_slot(session, 1);
    ASSERT_EQ(err, ASTRAL_OK);

    err = astral_session_set_slot(session, 0);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();

    unsetenv("ASTRAL_LLAMA_MAX_SLOTS");
}

/**
 * Test: Model loading with invalid path.
 */
TEST(model_load_invalid_path) {
    // Initialize runtime
    AstralInit cfg = {0};
    cfg.reserve_bytes = 2ULL << 30;
    cfg.thread_count = 4;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Try to load non-existent model
    const char* invalid_path = "/nonexistent/model.gguf";
    AstralModelDesc model_desc = {0};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = (const uint8_t*)invalid_path;
    model_desc.model_path.len = (uint32_t)strlen(invalid_path);
    model_desc.n_ctx = 2048;
    model_desc.n_batch = 512;
    model_desc.n_threads = 4;
    model_desc.gpu_layers = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);

    // Should fail (accept any error code except ASTRAL_OK for now)
    // Once backend is implemented, this should be ASTRAL_E_INVALID or similar
    ASSERT_NE(err, ASTRAL_OK);

    astral_shutdown();
}

/**
 * Test: Session creation with null model.
 */
TEST(session_create_null_model) {
    // Initialize runtime
    AstralInit cfg = {0};
    cfg.reserve_bytes = 2ULL << 30;
    cfg.thread_count = 4;
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Try to create session with null model
    AstralSessionDesc session_desc = {0};
    session_desc.model = 0; // Invalid
    session_desc.max_tokens = 50;
    session_desc.temperature = 0.8f;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);

    // Should fail with ASTRAL_E_INVALID (null model)
    // Stub returns ASTRAL_E_INVALID correctly
    ASSERT_EQ(err, ASTRAL_E_INVALID);

    astral_shutdown();
}

/**
 * Test: UTF-8 validation.
 */
TEST(utf8_validation) {
    // Valid UTF-8
    const uint8_t valid1[] = "Hello, world!";
    ASSERT_TRUE(validate_utf8(valid1, sizeof(valid1) - 1));

    // Valid UTF-8 with multi-byte characters (Unicode)
    const uint8_t valid2[] = u8"Hello, 世界!";
    ASSERT_TRUE(validate_utf8(valid2, sizeof(valid2) - 1));

    // Valid UTF-8 with emoji
    const uint8_t valid3[] = u8"Hello 🌍!";
    ASSERT_TRUE(validate_utf8(valid3, sizeof(valid3) - 1));

    // Empty string (valid)
    ASSERT_TRUE(validate_utf8(nullptr, 0));
    ASSERT_TRUE(validate_utf8((const uint8_t*)"", 0));

    // Invalid UTF-8 (truncated multi-byte sequence)
    const uint8_t invalid1[] = {0xC3}; // 2-byte sequence, missing continuation
    ASSERT_FALSE(validate_utf8(invalid1, sizeof(invalid1)));

    // Invalid UTF-8 (invalid continuation byte)
    const uint8_t invalid2[] = {0xC3, 0x00}; // 2-byte sequence, invalid continuation
    ASSERT_FALSE(validate_utf8(invalid2, sizeof(invalid2)));

    // Invalid UTF-8 (lone continuation byte)
    const uint8_t invalid3[] = {0x80};
    ASSERT_FALSE(validate_utf8(invalid3, sizeof(invalid3)));
}
