/**
 * test_arena.cpp - Arena-backed init/session tests
 *
 * Validates:
 * - astral_init2() supports borrowed and owned arena modes
 * - per-session scratch blocks are deterministic (fixed pool) and reusable
 */

#include "../include/astral_rt.h"
#include "../src/core/runtime_state.hpp"
#include "test_framework.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

struct RejectingAllocatorProbe {
  uint32_t alloc_calls;
  uint32_t free_calls;
};

void* rejecting_alloc(void* user, size_t size, size_t align) {
  (void)size;
  (void)align;
  auto* probe = static_cast<RejectingAllocatorProbe*>(user);
  ++probe->alloc_calls;
  return nullptr;
}

void rejecting_free(void* user, void* ptr, size_t size, size_t align) {
  (void)ptr;
  (void)size;
  (void)align;
  auto* probe = static_cast<RejectingAllocatorProbe*>(user);
  ++probe->free_calls;
}

AstralAllocator make_rejecting_allocator(RejectingAllocatorProbe* probe) {
  AstralAllocator alloc{};
  alloc.alloc = rejecting_alloc;
  alloc.free = rejecting_free;
  alloc.user = probe;
  return alloc;
}

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

static AstralInit2 make_borrowed_arena_init(void* base, uint64_t size) {
    AstralInit2 cfg{};
    cfg.base.thread_count = 0;
    cfg.base.numa_node = 0xFFFFFFFFu;
    cfg.base.enable_hugepages = 0;
    cfg.memory_mode = ASTRAL_MEMMODE_ARENA_BORROWED;
    cfg.arena.base = base;
    cfg.arena.size = size;
    cfg.arena.session_block_size = 2u * 1024u * 1024u;
    cfg.arena.session_block_count = 1;
    return cfg;
}

} // namespace

TEST(arena_init2_invalid_config_cleanup_allows_retry) {
    alignas(64) static uint8_t arena[8u * 1024u * 1024u];

    AstralInit2 invalid_mode = make_borrowed_arena_init(arena, sizeof(arena));
    invalid_mode.memory_mode = static_cast<AstralMemoryMode>(0xFFu);
    ASSERT_EQ(astral_init2(&invalid_mode), ASTRAL_E_INVALID);

    AstralInit2 missing_base = make_borrowed_arena_init(nullptr, sizeof(arena));
    ASSERT_EQ(astral_init2(&missing_base), ASTRAL_E_INVALID);

    AstralInit2 cfg = make_borrowed_arena_init(arena, sizeof(arena));
    AstralErr err = astral_init2(&cfg);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle model = 0;
    const AstralModelDesc model_desc = make_mock_model_desc();
    err = astral_model_load(&model_desc, &model);
    ASSERT_EQ(err, ASTRAL_OK);

    AstralHandle session = 0;
    const AstralSessionDesc session_desc = make_session_desc(model);
    err = astral_session_create(&session_desc, &session);
    ASSERT_EQ(err, ASTRAL_OK);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}

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

TEST(arena_runtime_alloc_size_class_boundaries) {
  alignas(64) static uint8_t arena[16u * 1024u * 1024u];

  AstralInit2 cfg = make_borrowed_arena_init(arena, sizeof(arena));
  cfg.arena._reserved[1] = 8u * 1024u * 1024u;
  ASSERT_EQ(astral_init2(&cfg), ASTRAL_OK);

  for (size_t base = 32; base <= 4096; base *= 2) {
    const size_t step = base / 8;
    for (size_t sub = 0; sub < 8; ++sub) {
      const size_t boundary = base + sub * step;
      for (size_t size : {boundary, boundary + 1}) {
        void* ptr = astral::core::runtime_alloc(size, 16);
        ASSERT_NOT_NULL(ptr);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) & 15u, 0u);
        std::memset(ptr, 0xA5, size);
        astral::core::runtime_free(ptr, size, 16);
      }
    }
  }

  astral_shutdown();
}

TEST(arena_runtime_alloc_local_cache_refill_flush) {
  constexpr uint32_t kAllocationCount = 96;
  constexpr size_t kAllocationSize = 64;
  alignas(64) static uint8_t arena[8u * 1024u * 1024u];

  AstralInit2 cfg = make_borrowed_arena_init(arena, sizeof(arena));
  ASSERT_EQ(astral_init2(&cfg), ASTRAL_OK);

  void* pointers[kAllocationCount]{};
  for (uint32_t pass = 0; pass < 2; ++pass) {
    for (uint32_t i = 0; i < kAllocationCount; ++i) {
      pointers[i] = astral::core::runtime_alloc(kAllocationSize, 16);
      ASSERT_NOT_NULL(pointers[i]);
      std::memset(pointers[i], static_cast<int>(i), kAllocationSize);
    }
    for (uint32_t i = 0; i < kAllocationCount; ++i) {
      astral::core::runtime_free(pointers[i], kAllocationSize, 16);
    }
  }

  astral_shutdown();
}

TEST(arena_runtime_alloc_never_spills_to_host_allocator) {
  constexpr uint32_t kMaxAllocations = 256;
  constexpr size_t kAllocationSize = 64u * 1024u;
  alignas(64) static uint8_t arena[8u * 1024u * 1024u];

  RejectingAllocatorProbe probe{};
  AstralInit2 cfg = make_borrowed_arena_init(arena, sizeof(arena));
  cfg.base.thread_count = 1;
  cfg.base.sys_alloc = make_rejecting_allocator(&probe);
  ASSERT_EQ(astral_init2(&cfg), ASTRAL_OK);
  ASSERT_EQ(probe.alloc_calls, 0u);

  void* pointers[kMaxAllocations]{};
  uint32_t allocated = 0;
  while (allocated < kMaxAllocations) {
    void* ptr = astral::core::runtime_alloc(kAllocationSize, 16);
    if (ptr == nullptr) {
      break;
    }
    pointers[allocated++] = ptr;
  }

  ASSERT_GT(allocated, 0u);
  ASSERT_LT(allocated, kMaxAllocations);
  ASSERT_NULL(astral::core::runtime_alloc(2u * 1024u * 1024u, 16));
  ASSERT_EQ(probe.alloc_calls, 0u);

  for (uint32_t i = 0; i < allocated; ++i) {
    astral::core::runtime_free(pointers[i], kAllocationSize, 16);
  }
  astral_shutdown();
  ASSERT_EQ(probe.free_calls, 0u);
}

#if ASTRAL_ENABLE_VIRTUAL_MEMORY
TEST(vm_runtime_alloc_uses_reserved_region_and_commits_on_demand) {
  constexpr uint32_t kAllocationCount = 128;
  constexpr size_t kAllocationSize = 64u * 1024u;

  RejectingAllocatorProbe probe{};
  AstralInit cfg{};
  cfg.reserve_bytes = 64u * 1024u * 1024u;
  cfg.thread_count = 1;
  cfg.numa_node = 0xFFFFFFFFu;
  cfg.sys_alloc = make_rejecting_allocator(&probe);
  ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);
  ASSERT_EQ(probe.alloc_calls, 0u);

  uint64_t committed_before = 0;
  astral::core::runtime_memory_stats(&committed_before, nullptr);

  void* pointers[kAllocationCount]{};
  for (uint32_t i = 0; i < kAllocationCount; ++i) {
    pointers[i] = astral::core::runtime_alloc(kAllocationSize, 16);
    ASSERT_NOT_NULL(pointers[i]);
  }

  uint64_t committed_after = 0;
  astral::core::runtime_memory_stats(&committed_after, nullptr);
  ASSERT_GT(committed_after, committed_before);
  ASSERT_EQ(probe.alloc_calls, 0u);

  for (uint32_t i = 0; i < kAllocationCount; ++i) {
    astral::core::runtime_free(pointers[i], kAllocationSize, 16);
  }
  astral_shutdown();
  ASSERT_EQ(probe.free_calls, 0u);
}
#endif

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
