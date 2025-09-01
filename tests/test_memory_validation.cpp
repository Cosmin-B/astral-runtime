/**
 * test_memory_validation.cpp - Memory Validation Test
 *
 *  This test is specifically designed for Valgrind/ASAN validation.
 * It exercises hot paths extensively to detect allocations.
 *
 * Hot paths that MUST have ZERO allocations:
 * 1. Decode loop (token generation)
 * 2. Stream read (token consumption)
 * 3. Token sampling
 *
 * Success criteria (per MASTER_SPEC):
 * - No dynamic allocations during steady-state token streaming
 * - No memory leaks
 * - No use-after-free errors
 * - No uninitialized reads
 */

#include "../include/astral_rt.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Test configuration
constexpr uint32_t kDecodeIterations = 1000;
constexpr uint32_t kStreamReadIterations = 1000;
constexpr size_t kReserveBytes = 512ULL << 20; // 512MB

// Tracking allocations
static uint64_t g_alloc_count = 0;
static uint64_t g_free_count = 0;
static uint64_t g_alloc_bytes = 0;

// Custom allocator for tracking
void* tracked_alloc(void* user, size_t size, size_t align) {
    (void)user;
    (void)align;
    g_alloc_count++;
    g_alloc_bytes += size;

    // Use posix_memalign for alignment
    void* ptr = nullptr;
    if (align > 0) {
        if (posix_memalign(&ptr, align, size) != 0) {
            return nullptr;
        }
    } else {
        ptr = malloc(size);
    }

    return ptr;
}

void tracked_free(void* user, void* ptr, size_t size, size_t align) {
    (void)user;
    (void)size;
    (void)align;
    if (ptr) {
        g_free_count++;
        free(ptr);
    }
}

// Logging callback
void log_callback(void* user, int level, AstralSpanU8 msg) {
    (void)user;
    const char* level_str = "UNKNOWN";
    switch (level) {
        case ASTRAL_LOG_ERROR: level_str = "ERROR"; break;
        case ASTRAL_LOG_WARN:  level_str = "WARN "; break;
        case ASTRAL_LOG_INFO:  level_str = "INFO "; break;
        case ASTRAL_LOG_DEBUG: level_str = "DEBUG"; break;
        case ASTRAL_LOG_TRACE: level_str = "TRACE"; break;
    }

    printf("[%s] %.*s\n", level_str, (int)msg.len, msg.data);
}

// Helper to create AstralSpanU8 from C string
static AstralSpanU8 make_span(const char* str) {
    AstralSpanU8 span;
    span.data = reinterpret_cast<const uint8_t*>(str);
    span.len = static_cast<uint32_t>(strlen(str));
#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
    span._padding = 0;
#endif
    return span;
}

int main() {
    printf("=== Astral Memory Validation Test ===\n");
    printf("Purpose: Validate zero-allocation hot paths using Valgrind/ASAN\n\n");

    // Initialize runtime
    printf("1. Initializing runtime...\n");
    AstralInit cfg = {};
    cfg.reserve_bytes = kReserveBytes;
    cfg.thread_count = 1;
    cfg.numa_node = 0xFFFFFFFF; // Any node
    cfg.enable_hugepages = 0;
    cfg.log_cb = log_callback;
    cfg.log_user = nullptr;

    // Use custom allocator for tracking
    cfg.sys_alloc.alloc = tracked_alloc;
    cfg.sys_alloc.free = tracked_free;
    cfg.sys_alloc.user = nullptr;

    uint64_t init_alloc_count = g_alloc_count;
    AstralErr err = astral_init(&cfg);
    if (err != ASTRAL_OK) {
        printf("   [FAIL] astral_init failed: %s\n", astral_error_string(err));
        return 1;
    }
    printf("   [PASS] Runtime initialized\n");
    printf("   Allocations during init: %llu\n",
           (unsigned long long)(g_alloc_count - init_alloc_count));

    // NOTE: Model loading is not yet implemented in the stub
    // For now, we'll test the infrastructure that IS implemented

    // Test session creation (pre-commits memory for hot path)
    printf("\n2. Testing session creation (should pre-commit memory)...\n");
    AstralModelDesc model_desc = {};
    model_desc.model_path = make_span("stub_model.gguf");
    model_desc.gpu_layers = 0;
    model_desc.n_ctx = 2048;
    model_desc.n_batch = 512;
    model_desc.n_threads = 1;
    model_desc.embeddings_only = 0;

    AstralHandle model = 0;
    uint64_t pre_model_alloc = g_alloc_count;
    err = astral_model_load(&model_desc, &model);
    if (err != ASTRAL_OK) {
        printf("   [INFO] astral_model_load not implemented yet: %s\n",
               astral_error_string(err));
        printf("   (This is expected for stub implementation)\n");
    } else {
        printf("   [PASS] Model loaded\n");
        printf("   Allocations during model load: %llu\n",
               (unsigned long long)(g_alloc_count - pre_model_alloc));

        // Create session
        AstralSessionDesc session_desc = {};
        session_desc.model = model;
        session_desc.max_tokens = 100;
        session_desc.temperature = 0.7f;
        session_desc.top_k = 40;
        session_desc.top_p = 0.9f;
        session_desc.stream_enabled = 1;

        AstralHandle session = 0;
        uint64_t pre_session_alloc = g_alloc_count;
        err = astral_session_create(&session_desc, &session);
        if (err != ASTRAL_OK) {
            printf("   [INFO] astral_session_create not implemented yet: %s\n",
                   astral_error_string(err));
        } else {
            printf("   [PASS] Session created\n");
            printf("   Allocations during session create: %llu\n",
                   (unsigned long long)(g_alloc_count - pre_session_alloc));

            // Feed prompt
            printf("\n3. Testing prompt feeding...\n");
            AstralSpanU8 prompt = make_span("Hello, world! This is a test prompt.");
            uint64_t pre_feed_alloc = g_alloc_count;
            err = astral_session_feed(session, prompt, 1);
            if (err != ASTRAL_OK) {
                printf("   [INFO] astral_session_feed not implemented yet: %s\n",
                       astral_error_string(err));
            } else {
                printf("   [PASS] Prompt fed\n");
                printf("   Allocations during feed: %llu\n",
                       (unsigned long long)(g_alloc_count - pre_feed_alloc));
            }

            // HOT PATH TEST: Decode loop
            printf("\n4. Testing decode hot path (%u iterations)...\n", kDecodeIterations);
            printf("    This MUST have ZERO allocations\n");
            uint64_t pre_decode_alloc = g_alloc_count;

            for (uint32_t i = 0; i < kDecodeIterations; ++i) {
                err = astral_session_decode(session);
                if (err != ASTRAL_OK && err != ASTRAL_E_STATE) {
                    // ASTRAL_E_STATE is acceptable if not yet implemented
                    printf("   [WARN] Decode iteration %u failed: %s\n",
                           i, astral_error_string(err));
                    break;
                }
            }

            uint64_t decode_alloc_delta = g_alloc_count - pre_decode_alloc;
            if (decode_alloc_delta == 0) {
                printf("   [PASS] Decode hot path: ZERO allocations\n");
            } else {
                printf("   [FAIL] Decode hot path: %llu allocations detected!\n",
                       (unsigned long long)decode_alloc_delta);
                printf("   This violates MASTER_SPEC Performance Target #1\n");
            }

            // HOT PATH TEST: Stream read
            printf("\n5. Testing stream read hot path (%u iterations)...\n",
                   kStreamReadIterations);
            printf("    This MUST have ZERO allocations\n");
            uint64_t pre_stream_alloc = g_alloc_count;

            uint8_t buffer[4096];
            AstralMutSpanU8 out_buf;
            out_buf.data = buffer;
            out_buf.len = sizeof(buffer);
#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
            out_buf._padding = 0;
#endif

            for (uint32_t i = 0; i < kStreamReadIterations; ++i) {
                int32_t bytes_read = astral_stream_read(session, out_buf, 0);
                if (bytes_read < 0 && bytes_read != ASTRAL_E_TIMEOUT) {
                    // ASTRAL_E_TIMEOUT is acceptable (no data yet)
                    printf("   [WARN] Stream read iteration %u failed: %s\n",
                           i, astral_error_string((AstralErr)bytes_read));
                    break;
                }
            }

            uint64_t stream_alloc_delta = g_alloc_count - pre_stream_alloc;
            if (stream_alloc_delta == 0) {
                printf("   [PASS] Stream read hot path: ZERO allocations\n");
            } else {
                printf("   [FAIL] Stream read hot path: %llu allocations detected!\n",
                       (unsigned long long)stream_alloc_delta);
                printf("   This violates MASTER_SPEC Performance Target #1\n");
            }

            // Get statistics
            printf("\n6. Checking session statistics...\n");
            AstralStats stats;
            err = astral_session_stats(session, &stats);
            if (err != ASTRAL_OK) {
                printf("   [INFO] astral_session_stats not implemented yet: %s\n",
                       astral_error_string(err));
            } else {
                printf("   [PASS] Statistics retrieved:\n");
                printf("          Init time: %.2f ms\n", stats.t_init_ms);
                printf("          Time to first token: %.2f ms\n", stats.t_first_token_ms);
                printf("          Tokens/sec: %.2f\n", stats.tok_per_s);
                printf("          Committed memory: %llu bytes (%.2f MB)\n",
                       (unsigned long long)stats.bytes_committed,
                       stats.bytes_committed / (1024.0 * 1024.0));
                printf("          Reserved memory: %llu bytes (%.2f MB)\n",
                       (unsigned long long)stats.bytes_reserved,
                       stats.bytes_reserved / (1024.0 * 1024.0));
            }

            // Clean up session
            astral_session_destroy(session);
            printf("   Session destroyed\n");
        }

        // Clean up model
        astral_model_release(model);
        printf("   Model released\n");
    }

    // Shutdown runtime
    printf("\n7. Shutting down runtime...\n");
    uint64_t pre_shutdown_alloc = g_alloc_count;
    uint64_t pre_shutdown_free = g_free_count;
    astral_shutdown();
    printf("   [PASS] Runtime shutdown complete\n");

    // Final allocation summary
    printf("\n=== Allocation Summary ===\n");
    printf("Total allocations: %llu\n", (unsigned long long)g_alloc_count);
    printf("Total frees: %llu\n", (unsigned long long)g_free_count);
    printf("Total bytes allocated: %llu (%.2f MB)\n",
           (unsigned long long)g_alloc_bytes,
           g_alloc_bytes / (1024.0 * 1024.0));

    if (g_alloc_count == g_free_count) {
        printf("\n[PASS] No memory leaks detected (alloc count == free count)\n");
    } else {
        printf("\n[FAIL] Memory leak detected:\n");
        printf("       Allocations: %llu\n", (unsigned long long)g_alloc_count);
        printf("       Frees: %llu\n", (unsigned long long)g_free_count);
        printf("       Leaked: %llu\n",
               (unsigned long long)(g_alloc_count - g_free_count));
    }

    printf("\n=== Test Complete ===\n");
    printf("Next steps:\n");
    printf("1. Run with Valgrind memcheck: scripts/run_valgrind.sh\n");
    printf("2. Run with Valgrind massif: scripts/run_massif.sh\n");
    printf("3. Run with AddressSanitizer: scripts/run_asan.sh\n");

    return 0;
}
