#include "../include/astral_rt.h"
#include "test_framework.hpp"

#include <cstring>

namespace {

AstralHandle load_mock_model() {
  AstralModelDesc desc{};
  desc.size = sizeof(AstralModelDesc);
  desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
  static constexpr char kBackend[] = "mock";
  desc.backend_name.data = reinterpret_cast<const uint8_t*>(kBackend);
  desc.backend_name.len = static_cast<uint32_t>(std::strlen(kBackend));
  desc.n_ctx = 128;

  AstralHandle model = 0;
  ASSERT_EQ(astral_model_load(&desc, &model), ASTRAL_OK);
  return model;
}

} // namespace

TEST(threads_disabled_rejects_conversation_executor) {
  AstralInit cfg{};
  cfg.reserve_bytes = 32 * 1024 * 1024;
  ASSERT_EQ(astral_init(&cfg), ASTRAL_OK);

  const AstralHandle model = load_mock_model();

  AstralExecutorDesc executor{};
  executor.size = sizeof(AstralExecutorDesc);
  executor.max_slots = 1;
  executor.max_batch_tokens = 8;
  ASSERT_EQ(astral_model_executor_configure(model, &executor), ASTRAL_E_UNSUPPORTED);

  astral_model_release(model);
  astral_shutdown();
}
