/**
 * test_model_sources.cpp - Model source selection tests (PATH/MEMORY/IO)
 *
 * Validates:
 * - astral_model_load2 supports MEMORY and IO sources for backends that opt in
 * - surfaces are usable under arena-backed init2 (embedded-friendly)
 */

#include "../include/astral_rt.h"
#include "test_framework.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

static AstralSpanU8 span_from_cstr(const char* s) {
    AstralSpanU8 out{};
    out.data = reinterpret_cast<const uint8_t*>(s);
    out.len = s ? static_cast<uint32_t>(std::strlen(s)) : 0u;
    return out;
}

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return {};
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n <= 0) {
        return {};
    }
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(out.data()), n);
    if (!f) {
        return {};
    }
    return out;
}

static std::vector<uint8_t> read_cpu_test_model() {
  const char* path = std::getenv("ASTRAL_TEST_MODEL");
  if (path != nullptr && path[0] != '\0') {
    auto bytes = read_file(path);
    if (!bytes.empty()) {
      return bytes;
    }
  }
  return read_file(ASTRAL_TEST_SOURCE_DIR "/tests/models/gpt2.Q2_K.gguf");
}

struct IoString {
    const uint8_t* data;
    uint32_t len;
};

static uint64_t ASTRAL_CALL io_size_string(void* user) {
    const auto* s = static_cast<const IoString*>(user);
    return (s && s->data) ? static_cast<uint64_t>(s->len) : 0;
}

static uint32_t ASTRAL_CALL io_read_at_string(void* user, uint64_t offset, void* dst, uint32_t dst_len) {
    if (user == nullptr || dst == nullptr) {
        return 0;
    }
    const auto* s = static_cast<const IoString*>(user);
    if (s->data == nullptr || s->len == 0) {
        return 0;
    }
    if (offset >= s->len) {
        return 0;
    }
    const uint32_t avail = s->len - static_cast<uint32_t>(offset);
    const uint32_t n = (avail < dst_len) ? avail : dst_len;
    std::memcpy(dst, s->data + offset, n);
    return n;
}

struct IoMem {
    const uint8_t* data;
    uint64_t size;
};

static uint64_t ASTRAL_CALL io_size_mem(void* user) {
    const auto* m = static_cast<const IoMem*>(user);
    return (m && m->data) ? m->size : 0;
}

static uint32_t ASTRAL_CALL io_read_at_mem(void* user, uint64_t offset, void* dst, uint32_t dst_len) {
    const auto* m = static_cast<const IoMem*>(user);
    if (m == nullptr || m->data == nullptr || dst == nullptr) {
        return 0;
    }
    if (offset >= m->size) {
        return 0;
    }
    const uint64_t avail64 = m->size - offset;
    const uint32_t avail = avail64 > 0xFFFFFFFFu ? 0xFFFFFFFFu : static_cast<uint32_t>(avail64);
    const uint32_t n = (avail < dst_len) ? avail : dst_len;
    std::memcpy(dst, m->data + offset, n);
    return n;
}

static AstralInit2 make_small_borrowed_arena_cfg(void* base, uint64_t size) {
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

static AstralModelDesc make_desc_common_mock() {
    AstralModelDesc d{};
    d.size = sizeof(AstralModelDesc);
    d.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    d.backend_name = span_from_cstr("mock");
    d.gpu_layers = 0;
    d.n_ctx = 128;
    d.n_batch = 64;
    d.n_threads = 0;
    d.embeddings_only = 0;
    return d;
}

static AstralModelDesc make_desc_common_cpu() {
    AstralModelDesc d{};
    d.size = sizeof(AstralModelDesc);
    d.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    d.backend_name = span_from_cstr("cpu");
    d.gpu_layers = 0;
    d.n_ctx = 128;
    d.n_batch = 64;
    d.n_threads = 0;
    d.embeddings_only = 0;
    return d;
}

} // namespace

TEST(model_load2_memory_mock_smoke) {
    alignas(64) static uint8_t arena[8u * 1024u * 1024u];
    const AstralInit2 cfg = make_small_borrowed_arena_cfg(arena, sizeof(arena));
    ASSERT_EQ(astral_init2(&cfg), ASTRAL_OK);

    AstralModelDesc desc = make_desc_common_mock();
    desc.source_kind = ASTRAL_MODEL_SOURCE_MEMORY;
    desc.model_bytes = span_from_cstr("sampler");

    AstralHandle model = 0;
    ASSERT_EQ(astral_model_load2(&desc, &model), ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    AstralModelInfo info{};
    ASSERT_EQ(astral_model_info(model, &info), ASTRAL_OK);
    ASSERT_LT(info.token_eos, 0);

    astral_model_release(model);
    astral_shutdown();
}

TEST(model_load2_io_mock_smoke) {
    alignas(64) static uint8_t arena[8u * 1024u * 1024u];
    const AstralInit2 cfg = make_small_borrowed_arena_cfg(arena, sizeof(arena));
    ASSERT_EQ(astral_init2(&cfg), ASTRAL_OK);

    const char* tag = "infinite";
    IoString io_src{};
    io_src.data = reinterpret_cast<const uint8_t*>(tag);
    io_src.len = static_cast<uint32_t>(std::strlen(tag));

    AstralModelDesc desc = make_desc_common_mock();
    desc.source_kind = ASTRAL_MODEL_SOURCE_IO;
    desc.io.user = &io_src;
    desc.io.size = io_size_string;
    desc.io.read_at = io_read_at_string;

    AstralHandle model = 0;
    ASSERT_EQ(astral_model_load2(&desc, &model), ASTRAL_OK);
    ASSERT_TRUE(astral_handle_valid(model));

    AstralModelInfo info{};
    ASSERT_EQ(astral_model_info(model, &info), ASTRAL_OK);
    ASSERT_LT(info.token_eos, 0);

    astral_model_release(model);
    astral_shutdown();
}

TEST(model_load2_memory_cpu_smoke) {
#if !ASTRAL_ENABLE_VIRTUAL_MEMORY
  SKIP_TEST("CPU MEMORY source requires virtual memory support");
#endif
  const auto bytes = read_cpu_test_model();
  if (bytes.empty()) {
    SKIP_TEST("ASTRAL_TEST_MODEL or tests/models/gpt2.Q2_K.gguf is required for CPU MEMORY source "
              "coverage");
  }

  alignas(64) static uint8_t arena[16u * 1024u * 1024u];
  const AstralInit2 cfg = make_small_borrowed_arena_cfg(arena, sizeof(arena));
  ASSERT_EQ(astral_init2(&cfg), ASTRAL_OK);

  // Load a real GGUF model from the test corpus into memory, then load via MEMORY source.
  AstralModelDesc desc = make_desc_common_cpu();
  desc.source_kind = ASTRAL_MODEL_SOURCE_MEMORY;
  desc.model_bytes.data = bytes.data();
  desc.model_bytes.len = static_cast<uint32_t>(bytes.size());

  AstralHandle model = 0;
  ASSERT_EQ(astral_model_load2(&desc, &model), ASTRAL_OK);
  ASSERT_TRUE(astral_handle_valid(model));

  AstralModelInfo info{};
  ASSERT_EQ(astral_model_info(model, &info), ASTRAL_OK);
  ASSERT_GT(info.vocab_size, 0u);

  astral_model_release(model);
  astral_shutdown();
}

TEST(model_load2_io_cpu_smoke) {
#if !ASTRAL_ENABLE_VIRTUAL_MEMORY
  SKIP_TEST("CPU IO source requires virtual memory support");
#endif
  const auto bytes = read_cpu_test_model();
  if (bytes.empty()) {
    SKIP_TEST(
        "ASTRAL_TEST_MODEL or tests/models/gpt2.Q2_K.gguf is required for CPU IO source coverage");
  }

  alignas(64) static uint8_t arena[16u * 1024u * 1024u];
  const AstralInit2 cfg = make_small_borrowed_arena_cfg(arena, sizeof(arena));
  ASSERT_EQ(astral_init2(&cfg), ASTRAL_OK);

  IoMem io_src{};
  io_src.data = bytes.data();
  io_src.size = bytes.size();

  AstralModelDesc desc = make_desc_common_cpu();
  desc.source_kind = ASTRAL_MODEL_SOURCE_IO;
  desc.io.user = &io_src;
  desc.io.size = io_size_mem;
  desc.io.read_at = io_read_at_mem;

  AstralHandle model = 0;
  ASSERT_EQ(astral_model_load2(&desc, &model), ASTRAL_OK);
  ASSERT_TRUE(astral_handle_valid(model));

  AstralModelInfo info{};
  ASSERT_EQ(astral_model_info(model, &info), ASTRAL_OK);
  ASSERT_GT(info.vocab_size, 0u);

  astral_model_release(model);
  astral_shutdown();
}
