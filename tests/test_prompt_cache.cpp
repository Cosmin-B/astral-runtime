#include "../include/astral_rt.h"
#include "test_framework.hpp"

#include <cstdint>
#include <cstring>

namespace {

inline constexpr uint32_t kBytesPerKiB = 1024;
inline constexpr uint32_t kKiBPerMiB = 1024;
inline constexpr uint32_t kRuntimeReserveMiB = 16;
inline constexpr uint32_t kRuntimeReserveBytes = kRuntimeReserveMiB * kKiBPerMiB * kBytesPerKiB;
inline constexpr uint32_t kCacheEntries = 4;
inline constexpr uint32_t kCacheTokens = 16;
inline constexpr uint32_t kFifoEvictionEntries = 2;
inline constexpr uint32_t kSingleTokenCount = 1;
inline constexpr uint32_t kTokenReadCapacity = 1;
inline constexpr uint32_t kPromptCacheSnapshotBytes = 512;
inline constexpr AstralHandle kModelHandle = 0x0100000100000001ull;
inline constexpr uint64_t kSystemKey = 1001;
inline constexpr uint64_t kUserKey = 2001;
inline constexpr uint64_t kAssistantKey = 3001;
inline constexpr uint64_t kRawKey = 4001;
inline constexpr uint32_t kFifoGeneration = 7;
inline constexpr int32_t kFifoFirstToken = 101;
inline constexpr int32_t kFifoSecondToken = 202;
inline constexpr int32_t kFifoThirdToken = 303;

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

TEST(prompt_cache_key_from_bytes_scopes_sections_and_generations) {
    init_runtime();

    AstralPromptCacheKey system_v1{};
    AstralPromptCacheKey system_v1_again{};
    AstralPromptCacheKey system_v2{};
    AstralPromptCacheKey user_v1{};
    AstralErr err = astral_prompt_cache_key_from_bytes(
        kModelHandle, ASTRAL_PROMPT_SECTION_SYSTEM, 1, span_from_cstr("stable"), &system_v1);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_prompt_cache_key_from_bytes(
        kModelHandle, ASTRAL_PROMPT_SECTION_SYSTEM, 1, span_from_cstr("stable"), &system_v1_again);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_prompt_cache_key_from_bytes(
        kModelHandle, ASTRAL_PROMPT_SECTION_SYSTEM, 2, span_from_cstr("stable"), &system_v2);
    ASSERT_EQ(err, ASTRAL_OK);
    err = astral_prompt_cache_key_from_bytes(
        kModelHandle, ASTRAL_PROMPT_SECTION_USER, 1, span_from_cstr("stable"), &user_v1);
    ASSERT_EQ(err, ASTRAL_OK);

    ASSERT_EQ(system_v1.key, system_v1_again.key);
    ASSERT_EQ(system_v1.generation, 1u);
    ASSERT_EQ(system_v2.generation, 2u);
    ASSERT_EQ(system_v1.section_kind, ASTRAL_PROMPT_SECTION_SYSTEM);
    ASSERT_EQ(user_v1.section_kind, ASTRAL_PROMPT_SECTION_USER);

    AstralHandle cache = 0;
    const AstralPromptCacheDesc desc = cache_desc();
    ASSERT_EQ(astral_prompt_cache_create(&desc, &cache), ASTRAL_OK);

    const int32_t system_tokens[] = {10, 20};
    const int32_t user_tokens[] = {30, 40};
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &system_v1, system_tokens, 2), ASTRAL_OK);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &user_v1, user_tokens, 2), ASTRAL_OK);

    int32_t out[2]{};
    uint32_t count = 0;
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &system_v1_again, out, 2, &count), ASTRAL_OK);
    ASSERT_EQ(out[0], system_tokens[0]);
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &system_v2, out, 2, &count), ASTRAL_E_NOT_FOUND);
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &user_v1, out, 2, &count), ASTRAL_OK);
    ASSERT_EQ(out[0], user_tokens[0]);

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

TEST(prompt_cache_update_compacts_token_arena) {
    init_runtime();

    AstralHandle cache = 0;
    const AstralPromptCacheDesc desc = cache_desc(4, 8);
    ASSERT_EQ(astral_prompt_cache_create(&desc, &cache), ASTRAL_OK);

    const int32_t first[] = {1, 2};
    const int32_t second[] = {3, 4};
    const int32_t second_update[] = {9};
    const int32_t third[] = {5, 6};
    const int32_t fourth[] = {10, 11, 12, 13};
    const AstralPromptCacheKey key0 = cache_key(10, 1);
    const AstralPromptCacheKey key1 = cache_key(20, 1);
    const AstralPromptCacheKey key2 = cache_key(30, 1);
    const AstralPromptCacheKey key3 = cache_key(40, 1);

    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key0, first, 2), ASTRAL_OK);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key1, second, 2), ASTRAL_OK);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key2, third, 2), ASTRAL_OK);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key1, second_update, 1), ASTRAL_OK);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key3, fourth, 4), ASTRAL_OK);

    int32_t out[4]{};
    uint32_t count = 0;
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &key0, out, 4, &count), ASTRAL_E_NOT_FOUND);
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &key2, out, 4, &count), ASTRAL_E_NOT_FOUND);
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &key1, out, 4, &count), ASTRAL_OK);
    ASSERT_EQ(count, 1u);
    ASSERT_EQ(out[0], second_update[0]);
    ASSERT_EQ(astral_prompt_cache_get_tokens(cache, &key3, out, 4, &count), ASTRAL_OK);
    ASSERT_EQ(count, 4u);
    ASSERT_EQ(out[3], fourth[3]);

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

TEST(prompt_cache_fifo_order_survives_save_load) {
    init_runtime();

    AstralHandle cache = 0;
    const AstralPromptCacheDesc desc = cache_desc(kFifoEvictionEntries, kCacheTokens);
    ASSERT_EQ(astral_prompt_cache_create(&desc, &cache), ASTRAL_OK);

    const int32_t first[] = {kFifoFirstToken};
    const int32_t second[] = {kFifoSecondToken};
    const AstralPromptCacheKey key0 = cache_key(kAssistantKey, kFifoGeneration, ASTRAL_PROMPT_SECTION_HISTORY);
    const AstralPromptCacheKey key1 = cache_key(kRawKey, kFifoGeneration, ASTRAL_PROMPT_SECTION_RAW);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key0, first, kSingleTokenCount), ASTRAL_OK);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key1, second, kSingleTokenCount), ASTRAL_OK);

    uint8_t storage[kPromptCacheSnapshotBytes]{};
    AstralMutSpanU8 out{};
    out.data = storage;
    out.len = sizeof(storage);
    uint32_t written = 0;
    ASSERT_EQ(astral_prompt_cache_save(cache, out, &written), ASTRAL_OK);
    astral_prompt_cache_destroy(cache);

    AstralSpanU8 in{};
    in.data = storage;
    in.len = written;
    AstralHandle loaded = 0;
    ASSERT_EQ(astral_prompt_cache_load(&desc, in, &loaded), ASTRAL_OK);

    const int32_t third[] = {kFifoThirdToken};
    const AstralPromptCacheKey key2 = cache_key(kSystemKey, kFifoGeneration);
    ASSERT_EQ(astral_prompt_cache_put_tokens(loaded, &key2, third, kSingleTokenCount), ASTRAL_OK);

    int32_t token[kTokenReadCapacity]{};
    uint32_t count = 0;
    ASSERT_EQ(astral_prompt_cache_get_tokens(loaded, &key0, token, kTokenReadCapacity, &count), ASTRAL_E_NOT_FOUND);
    ASSERT_EQ(astral_prompt_cache_get_tokens(loaded, &key1, token, kTokenReadCapacity, &count), ASTRAL_OK);
    ASSERT_EQ(token[0], kFifoSecondToken);
    ASSERT_EQ(astral_prompt_cache_get_tokens(loaded, &key2, token, kTokenReadCapacity, &count), ASTRAL_OK);
    ASSERT_EQ(token[0], kFifoThirdToken);

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

TEST(prompt_cache_view_feeds_session_and_conversation_tokens) {
    constexpr uint32_t kTokenCount = 4;
    constexpr int32_t kTokens[kTokenCount] = {256, 's', 'y', 's'};

    init_runtime();
    AstralHandle model = load_mock_model();

    AstralHandle cache = 0;
    const AstralPromptCacheDesc cacheDesc = cache_desc();
    ASSERT_EQ(astral_prompt_cache_create(&cacheDesc, &cache), ASTRAL_OK);

    const AstralPromptCacheKey key = cache_key(kSystemKey, 1);
    ASSERT_EQ(astral_prompt_cache_put_tokens(cache, &key, kTokens, kTokenCount), ASTRAL_OK);

    const int32_t* view = nullptr;
    uint32_t count = 0;
    ASSERT_EQ(astral_prompt_cache_get_token_view(cache, &key, &view, &count), ASTRAL_OK);
    ASSERT_NOT_NULL(view);
    ASSERT_EQ(count, kTokenCount);

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = kCacheTokens;
    session_desc.temperature = 0.0f;

    AstralHandle session = 0;
    ASSERT_EQ(astral_session_create(&session_desc, &session), ASTRAL_OK);
    ASSERT_EQ(astral_session_feed_tokens(session, view, count, 0), ASTRAL_OK);
    ASSERT_EQ(astral_session_feed(session, span_from_cstr("user"), 1), ASTRAL_OK);
    astral_session_destroy(session);

    AstralConvDesc conv_desc{};
    conv_desc.size = sizeof(AstralConvDesc);
    conv_desc.model = model;
    conv_desc.max_tokens = kCacheTokens;
    conv_desc.temperature = 0.0f;
    conv_desc.top_p = 1.0f;

    AstralHandle conv = 0;
    ASSERT_EQ(astral_conv_create(&conv_desc, &conv), ASTRAL_OK);
    ASSERT_EQ(astral_conv_feed_tokens(conv, view, count, 0), ASTRAL_OK);
    ASSERT_EQ(astral_conv_feed(conv, span_from_cstr("user"), 1), ASTRAL_OK);
    astral_conv_destroy(conv);

    astral_prompt_cache_destroy(cache);
    astral_model_release(model);
    astral_shutdown();
}
