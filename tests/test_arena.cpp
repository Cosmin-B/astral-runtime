/**
 * test_arena.cpp - Arena-backed init/session tests
 *
 * Validates:
 * - astral_init2() supports borrowed and owned arena modes
 * - per-session scratch blocks are deterministic (fixed pool) and reusable
 */

#include "test_framework.hpp"
#include "../include/astral_rt.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

static AstralModelDesc make_mock_model_desc() {
    AstralModelDesc d{};
    d.size = sizeof(AstralModelDesc);
    d.source_kind = ASTRAL_MODEL_SOURCE_PATH;

    const char* backend = "mock";
    d.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    d.backend_name.len = static_cast<uint32_t>(std::strlen(backend));

    // Mock backend does not require a model path.
    d.model_path.data = nullptr;
    d.model_path.len = 0;

    d.gpu_layers = 0;
    d.n_ctx = 128;
    d.n_batch = 64;
    d.n_threads = 0;
    d.embeddings_only = 0;
    return d;
}

static AstralSessionDesc make_session_desc(AstralHandle model) {
    AstralSessionDesc s{};
    s.model = model;
    s.max_tokens = 8;
    s.temperature = 0.0f;
    s.top_k = 0;
    s.top_p = 1.0f;
    s.stream_enabled = 1;
    s.seed = 1;
    return s;
}

} // namespace

TEST(arena_init2_borrowed_and_reuse_blocks) {
    // 8MB arena: enough for freelist metadata + (at least) one 2MB session block.
    alignas(64) static uint8_t arena[8u * 1024u * 1024u];

    AstralInit2 cfg{};
    cfg.base.thread_count = 0;
    cfg.base.numa_node = 0xFFFFFFFFu;
    cfg.base.enable_hugepages = 0;

    cfg.memory_mode = ASTRAL_MEMMODE_ARENA_BORROWED;
    cfg.arena.base = arena;
    cfg.arena.size = sizeof(arena);
    cfg.arena.session_block_size = 2u * 1024u * 1024u;
    cfg.arena.session_block_count = 1; // deterministic: only one session at a time

    AstralErr err = astral_init2(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle model = 0;
    const AstralModelDesc model_desc = make_mock_model_desc();
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralSessionDesc session_desc = make_session_desc(model);

    AstralHandle s1 = 0;
    err = astral_session_create(&session_desc, &s1);
    ASSERT_EQ(err, ASTRAL_OK);

    // Second session should fail due to fixed block pool exhaustion.
    AstralHandle s2 = 0;
    err = astral_session_create(&session_desc, &s2);
    ASSERT_EQ(err, ASTRAL_E_NOMEM);

    astral_session_destroy(s1);

    // After freeing, we should be able to create another session.
    AstralHandle s3 = 0;
    err = astral_session_create(&session_desc, &s3);
    ASSERT_EQ(err, ASTRAL_OK);
    astral_session_destroy(s3);

    astral_model_release(model);
    astral_shutdown();
}

TEST(arena_init2_owned_smoke) {
    AstralInit2 cfg{};
    cfg.base.thread_count = 0;
    cfg.base.numa_node = 0xFFFFFFFFu;
    cfg.base.enable_hugepages = 0;

    cfg.memory_mode = ASTRAL_MEMMODE_ARENA_OWNED;
    cfg.arena.base = nullptr; // let Astral allocate
    cfg.arena.size = 8u * 1024u * 1024u;
    cfg.arena.session_block_size = 2u * 1024u * 1024u;
    cfg.arena.session_block_count = 0; // auto

    AstralErr err = astral_init2(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle model = 0;
    const AstralModelDesc model_desc = make_mock_model_desc();
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    const AstralSessionDesc session_desc = make_session_desc(model);

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}
