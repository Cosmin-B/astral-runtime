#include "model_sources.hpp"

#include <atomic>
#include <cinttypes>
#include <cstdio>

namespace astral::core {

namespace {

constexpr uint64_t kInvalidId = 0;
constexpr uint32_t kMaxSources = 64;

struct Slot {
    uint64_t id = kInvalidId;
    ModelSource src{};
};

static std::atomic<uint64_t> g_next_id{1};
static std::atomic_flag g_lock = ATOMIC_FLAG_INIT;
static Slot g_slots[kMaxSources]{};

struct ScopedSpinLock {
    ScopedSpinLock() { while (g_lock.test_and_set(std::memory_order_acquire)) {} }
    ~ScopedSpinLock() { g_lock.clear(std::memory_order_release); }
};

static uint32_t format_token(uint64_t id, char* out_token, uint32_t cap) {
    if (out_token == nullptr || cap == 0) {
        return 0;
    }
    const int n = std::snprintf(out_token, cap, "astral-src:%" PRIu64, id);
    if (n <= 0) {
        return 0;
    }
    if (static_cast<uint32_t>(n) >= cap) {
        out_token[0] = '\0';
        return 0;
    }
    return static_cast<uint32_t>(n);
}

} // namespace

AstralErr model_source_register(const ModelSource& source, uint64_t* out_id, char* out_token, uint32_t token_cap) {
    if (out_id == nullptr || out_token == nullptr || token_cap == 0) {
        return ASTRAL_E_INVALID;
    }

    ScopedSpinLock _l;

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

    ScopedSpinLock _l;
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
    ScopedSpinLock _l;
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
    ScopedSpinLock _l;
    for (uint32_t i = 0; i < kMaxSources; ++i) {
        if (g_slots[i].id == id) {
            return true;
        }
    }
    return false;
}

} // namespace astral::core

