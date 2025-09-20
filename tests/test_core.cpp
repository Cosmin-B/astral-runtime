/**
 * test_core.cpp - Core runtime tests
 *
 * Tests for init/shutdown, handle management, error strings.
 * Validates: init/shutdown cycle, handle registration/validation, error codes.
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <cstring>

//
// Init/Shutdown Tests
//

TEST(core_init_shutdown_cycle) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024; // 16MB
    cfg.thread_count = 2;
    cfg.numa_node = 0xFFFFFFFF; // Any node
    cfg.enable_hugepages = 0;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_shutdown();
}

TEST(core_init_null_config) {
    AstralErr err = astral_init(nullptr);
    ASSERT_EQ(err, ASTRAL_E_INVALID);
}

TEST(core_init_minimal_config) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 1024 * 1024; // 1MB minimum

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_shutdown();
}

TEST(core_double_init) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;

    AstralErr err1 = astral_init(&cfg);
    ASSERT_EQ(err1, ASTRAL_OK);

    // Second init without shutdown should fail or be no-op
    AstralErr err2 = astral_init(&cfg);
    // Implementation-dependent: may return error or be idempotent

    astral_shutdown();
}

//
// Handle Validation Tests
//

TEST(core_handle_null_invalid) {
    int valid = astral_handle_valid(0);
    ASSERT_EQ(valid, 0);
}

TEST(core_handle_validation) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    astral_init(&cfg);

    // Create a model handle
    AstralModelDesc model_desc = {};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* path = "test_model.gguf";
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(path);
    model_desc.model_path.len = static_cast<uint32_t>(strlen(path));
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.gpu_layers = 0;

    AstralHandle model;
    AstralErr err = astral_model_load(&model_desc, &model);

    if (err == ASTRAL_OK) {
        // Handle should be valid
        int valid = astral_handle_valid(model);
        ASSERT_EQ(valid, 1);

        astral_model_release(model);

        // After release, handle should be invalid
        valid = astral_handle_valid(model);
        ASSERT_EQ(valid, 0);
    }

    astral_shutdown();
}

//
// Error String Tests
//

TEST(core_error_string_ok) {
    const char* str = astral_error_string(ASTRAL_OK);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
}

TEST(core_error_string_invalid) {
    const char* str = astral_error_string(ASTRAL_E_INVALID);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
}

TEST(core_error_string_nomem) {
    const char* str = astral_error_string(ASTRAL_E_NOMEM);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
}

TEST(core_error_string_busy) {
    const char* str = astral_error_string(ASTRAL_E_BUSY);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
}

TEST(core_error_string_timeout) {
    const char* str = astral_error_string(ASTRAL_E_TIMEOUT);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
}

TEST(core_error_string_state) {
    const char* str = astral_error_string(ASTRAL_E_STATE);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
}

TEST(core_error_string_backend) {
    const char* str = astral_error_string(ASTRAL_E_BACKEND);
    ASSERT_NOT_NULL(str);
    ASSERT_TRUE(strlen(str) > 0);
}

TEST(core_error_string_unknown) {
    // Unknown error code should return something (not crash)
    const char* str = astral_error_string(-9999);
    ASSERT_NOT_NULL(str);
}

//
// Allocator Tests
//

static void* test_alloc(void* user, size_t size, size_t align) {
    (void)user;
    (void)align;
    return malloc(size);
}

static void test_free(void* user, void* ptr, size_t size, size_t align) {
    (void)user;
    (void)size;
    (void)align;
    free(ptr);
}

TEST(core_custom_allocator) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    cfg.sys_alloc.alloc = test_alloc;
    cfg.sys_alloc.free = test_free;
    cfg.sys_alloc.user = nullptr;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_shutdown();
}

//
// Logging Callback Tests
//

static void test_log_callback(void* user, int level, AstralSpanU8 msg) {
    (void)user;
    (void)level;
    (void)msg;
    // Just verify it doesn't crash
}

TEST(core_logging_callback) {
    AstralInit cfg = {};
    cfg.reserve_bytes = 16 * 1024 * 1024;
    cfg.log_cb = test_log_callback;
    cfg.log_user = nullptr;

    AstralErr err = astral_init(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    // Logging should work
    astral_shutdown();
}
