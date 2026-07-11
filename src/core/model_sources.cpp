#include "../concurrency/event_spin_lock.hpp"
#include "../utils/string_builder.hpp"
#include "model_sources.hpp"

#include <atomic>
#include <cstring>

namespace astral::core {

namespace {

constexpr uint64_t kInvalidId = 0;
constexpr uint32_t kMaxSources = 64;

struct Slot {
  uint64_t id = kInvalidId;
  ModelSource src{};
};

static std::atomic<uint64_t> g_next_id{1};
static concurrency::EventSpinLock g_lock;
static Slot g_slots[kMaxSources]{};

static uint32_t format_token(uint64_t id, char* out_token, uint32_t cap) {
  if (out_token == nullptr || cap == 0) {
    return 0;
  }

  utf8::StackStringBuilder<31> token;
  token.append_literal("astral-src:");
  token.append_u64(id);
  if (token.truncated() || token.length() >= cap) {
    out_token[0] = '\0';
    return 0;
  }

  std::memcpy(out_token, token.c_str(), static_cast<size_t>(token.length()) + 1u);
  return token.length();
}

} // namespace

AstralErr model_source_register(const ModelSource& source, uint64_t* out_id, char* out_token,
                                uint32_t token_cap) {
  if (out_id == nullptr || out_token == nullptr || token_cap == 0) {
    return ASTRAL_E_INVALID;
  }

  concurrency::EventSpinLockGuard lock(g_lock);

  uint32_t slot_idx = kMaxSources;
  for (uint32_t i = 0; i < kMaxSources; ++i) {
    if (g_slots[i].id == kInvalidId) {
      slot_idx = i;
      break;
    }
  }
  if (slot_idx == kMaxSources) {
    return ASTRAL_E_BUSY;
  }

  const uint64_t id = g_next_id.fetch_add(1, std::memory_order_relaxed);
  if (id == kInvalidId) {
    return ASTRAL_E_BUSY;
  }

  g_slots[slot_idx].id = id;
  g_slots[slot_idx].src = source;

  const uint32_t n = format_token(id, out_token, token_cap);
  if (n == 0) {
    g_slots[slot_idx].id = kInvalidId;
    g_slots[slot_idx].src = ModelSource{};
    return ASTRAL_E_INVALID;
  }

  *out_id = id;
  return ASTRAL_OK;
}

bool model_source_take(uint64_t id, ModelSource* out_source) {
  if (id == kInvalidId || out_source == nullptr) {
    return false;
  }

  concurrency::EventSpinLockGuard lock(g_lock);
  for (uint32_t i = 0; i < kMaxSources; ++i) {
    if (g_slots[i].id == id) {
      *out_source = g_slots[i].src;
      g_slots[i].id = kInvalidId;
      g_slots[i].src = ModelSource{};
      return true;
    }
  }
  return false;
}

void model_source_release(uint64_t id) {
  if (id == kInvalidId) {
    return;
  }
  concurrency::EventSpinLockGuard lock(g_lock);
  for (uint32_t i = 0; i < kMaxSources; ++i) {
    if (g_slots[i].id == id) {
      g_slots[i].id = kInvalidId;
      g_slots[i].src = ModelSource{};
      return;
    }
  }
}

bool model_source_present(uint64_t id) {
  if (id == kInvalidId) {
    return false;
  }
  concurrency::EventSpinLockGuard lock(g_lock);
  for (uint32_t i = 0; i < kMaxSources; ++i) {
    if (g_slots[i].id == id) {
      return true;
    }
  }
  return false;
}

} // namespace astral::core
