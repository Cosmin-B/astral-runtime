#include "../include/astral_rt.h"
#include "test_framework.hpp"

#include <cstdint>
#include <cstring>

namespace {

inline constexpr uint32_t kRuntimeReserveBytes = 16 * 1024 * 1024;
inline constexpr uint32_t kCacheEntries = 4;
inline constexpr uint32_t kCacheTokens = 16;
inline constexpr AstralHandle kModelHandle = 0x0100000100000001ull;
inline constexpr uint64_t kSystemKey = 1001;
inline constexpr uint64_t kUserKey = 2001;

void init_runtime() {
    AstralInit cfg{};
    cfg.reserve_bytes = kRuntimeReserveBytes;
    cfg.thread_count = 1;
    cfg.numa_node = 0xFFFFFFFFu;
    ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);
}

AstralPromptCacheDesc cache_desc(uint32_t entries = kCacheEntries, uint32_t tokens = kCacheTokens) {
    AstralPromptCacheDesc desc{};
    desc.size = sizeof(AstralPromptCacheDesc);
    desc.max_entries = entries;
    desc.max_tokens = tokens;
    desc.eviction_policy = ASTRAL_PROMPT_CACHE_EVICT_FIFO;
    desc.flags = ASTRAL_PROMPT_CACHE_FLAG_TRACK_STATS;
    return desc;
}

AstralPromptCacheKey cache_key(uint64_t key, uint32_t generation, uint32_t section = ASTRAL_PROMPT_SECTION_SYSTEM) {
    AstralPromptCacheKey out{};
    out.size = sizeof(AstralPromptCacheKey);
    out.section_kind = section;
    out.model = kModelHandle;
    out.key = key;
    out.generation = generation;
    return out;
}

AstralSpanU8 span_from_cstr(const char *text) {
    AstralSpanU8 span{};
    span.data = reinterpret_cast<const uint8_t *>(text);
    span.len = static_cast<uint32_t>(std::strlen(text));
    return span;
}

AstralHandle load_mock_model() {
    AstralModelDesc desc{};
    desc.size = sizeof(AstralModelDesc);
    desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char *backend = "mock";
    desc.backend_name.data = reinterpret_cast<const uint8_t *>(backend);
    desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend));
    desc.n_ctx = 128;
    desc.n_batch = 64;

    AstralHandle model = 0;
    ASSERT_EQ(astral_model_load(&desc, &model), ASTRAL_OK);
    return model;
}

} // namespace

TEST(prompt_cache_hit_miss_and_stats) {
    init_runtime();

    AstralHandle cache = 0;
    const AstralPromptCacheDesc desc = cache_desc();
    ASSERT_EQ(astral_prompt_cache_create(&desc, &cache), ASTRAL_OK);
    ASSERT_NE(cache, 0ull);

    const AstralPromptCacheKey key = cache_key(kSystemKey, 1);
    const int32_t tokens[] = {11, 22, 33};
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key, tokens, 3), ASTRAL_OK);

    int32_t out[4]{};
    uint32_t count = 0;
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &key, out, 4, &count), ASTRAL_OK);
    ASSERT_EQ(count, 3u);
    ASSERT_EQ(out[0], tokens[0]);
    ASSERT_EQ(out[2], tokens[2]);

    const int32_t *view = nullptr;
    ASSERT_EQ(astral_prompt_cache_get_token_view(cache, &key, &view, &count), ASTRAL_OK);
    ASSERT_NOT_NULL(view);
    ASSERT_EQ(count, 3u);
    ASSERT_EQ(view[1], tokens[1]);

    const AstralPromptCacheKey miss = cache_key(kUserKey, 1, ASTRAL_PROMPT_SECTION_USER);
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &miss, out, 4, &count), ASTRAL_E_NOT_FOUND);
    ASSERT_EQ(count, 0u);

    AstralPromptCacheStats stats{};
    stats.size = sizeof(AstralPromptCacheStats);
    ASSERT_EQ(astral_prompt_cache_stats(cache, &stats), ASTRAL_OK);
    ASSERT_EQ(stats.entries, 1u);
    ASSERT_EQ(stats.tokens, 3u);
    ASSERT_EQ(stats.hits, 2ull);
    ASSERT_EQ(stats.misses, 1ull);

    astral_prompt_cache_destroy(cache);
    astral_shutdown();
}

TEST(prompt_cache_eviction_keeps_lookup_cluster_valid) {
    init_runtime();

    AstralHandle cache = 0;
    const AstralPromptCacheDesc desc = cache_desc(2, kCacheTokens);
    ASSERT_EQ(astral_prompt_cache_create(&desc, &cache), ASTRAL_OK);

    const int32_t first[] = {1, 2};
    const int32_t second[] = {3, 4};
    const int32_t third[] = {5, 6};
    const AstralPromptCacheKey key0 = cache_key(10, 1);
    const AstralPromptCacheKey key1 = cache_key(20, 1);
    const AstralPromptCacheKey key2 = cache_key(30, 1);

    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key0, first, 2), ASTRAL_OK);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key1, second, 2), ASTRAL_OK);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key2, third, 2), ASTRAL_OK);

    int32_t out[2]{};
    uint32_t count = 0;
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &key0, out, 2, &count), ASTRAL_E_NOT_FOUND);
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &key1, out, 2, &count), ASTRAL_OK);
    ASSERT_EQ(out[0], second[0]);
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &key2, out, 2, &count), ASTRAL_OK);
    ASSERT_EQ(out[1], third[1]);

    AstralPromptCacheStats stats{};
    stats.size = sizeof(AstralPromptCacheStats);
    ASSERT_EQ(astral_prompt_cache_stats(cache, &stats), ASTRAL_OK);
    ASSERT_EQ(stats.entries, 2u);
    ASSERT_EQ(stats.evictions, 1ull);

  astral_prompt_cache_destroy(cache);
  astral_shutdown();
}

TEST(prompt_cache_save_and_load_roundtrip) {
  init_runtime();

  AstralHandle cache = 0;
  const AstralPromptCacheDesc desc = cache_desc();
  ASSERT_EQ(astral_prompt_cache_create(&desc, &cache), ASTRAL_OK);

  const int32_t first[] = {7, 8, 9};
  const int32_t second[] = {10, 11};
  const AstralPromptCacheKey key0 = cache_key(kSystemKey, 3);
  const AstralPromptCacheKey key1 = cache_key(kUserKey, 4, ASTRAL_PROMPT_SECTION_USER);
  ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key0, first, 3), ASTRAL_OK);
  ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key1, second, 2), ASTRAL_OK);

  uint32_t bytes = 0;
  ASSERT_EQ(astral_prompt_cache_save_size(cache, &bytes), ASTRAL_OK);
  ASSERT_TRUE(bytes > 0);

  uint8_t storage[512]{};
  AstralMutSpanU8 out{};
  out.data = storage;
  out.len = sizeof(storage);
  uint32_t written = 0;
  ASSERT_EQ(astral_prompt_cache_save(cache, out, &written), ASTRAL_OK);
  ASSERT_EQ(written, bytes);
  astral_prompt_cache_destroy(cache);

  AstralSpanU8 in{};
  in.data = storage;
  in.len = written;
  AstralHandle loaded = 0;
  ASSERT_EQ(astral_prompt_cache_load(&desc, in, &loaded), ASTRAL_OK);

  int32_t tokens[4]{};
  uint32_t count = 0;
  ASSERT_EQ(astral_prompt_cache_get_tokens(loaded, &key0, tokens, 4, &count), ASTRAL_OK);
  ASSERT_EQ(count, 3u);
  ASSERT_EQ(tokens[0], first[0]);
  ASSERT_EQ(tokens[2], first[2]);
  ASSERT_EQ(astral_prompt_cache_get_tokens(loaded, &key1, tokens, 4, &count), ASTRAL_OK);
  ASSERT_EQ(count, 2u);
  ASSERT_EQ(tokens[1], second[1]);

  astral_prompt_cache_destroy(loaded);
  astral_shutdown();
}

TEST(system_prompt_feeds_before_user_prompt) {
    init_runtime();
    AstralHandle model = load_mock_model();

    AstralSessionDesc desc{};
    desc.model = model;
    desc.max_tokens = 4;
    desc.temperature = 0.0f;

    AstralHandle session = 0;
    ASSERT_EQ(astral_session_create(&desc, &session), ASTRAL_OK);
    ASSERT_EQ(astral_session_set_system_prompt(session, span_from_cstr("sys")), ASTRAL_OK);
    ASSERT_EQ(astral_session_feed(session, span_from_cstr("user"), 1), ASTRAL_OK);
    ASSERT_EQ(astral_session_set_system_prompt(session, span_from_cstr("late")), ASTRAL_E_STATE);

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
}
