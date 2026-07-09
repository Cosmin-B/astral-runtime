#pragma once

#include "../backend/backend.hpp"
#include "../concurrency/epoch.hpp"
#include "../platform/thread.h"
#include "../utils/trace.hpp"

#include "conversation.hpp"

#include <atomic>
#include <cstdint>

namespace astral::inference {

struct ModelExecutor {
    explicit ModelExecutor(Model* model_) noexcept;

    Model* model = nullptr;

    void* backend_session_ctx = nullptr;
    uint32_t max_slots = 1;
    uint32_t max_batch_tokens = 1;

    // Scheduling knobs (atomic so they can be tuned at runtime).
    std::atomic<uint32_t> max_prompt_tokens_per_slot_per_tick{8};
    uint32_t vocab_size = 0;
    uint32_t ctx_size = 0;

    // Full-vocab indices buffer (shared across conversations) for sampler fallback paths.
    uint32_t* indices_buffer = nullptr;

    // Slot registry protected by the model's slot-table lock.
    static constexpr uint32_t kMaxSlotsHard = 32;
    std::atomic<Conversation*> slots[kMaxSlotsHard];
    uint32_t active_slot_mask = 0;

    // One executor reader and one slot-lock-serialized retirement producer.
    using ConversationEpochManager = concurrency::EpochManager<2, 64>;
    ConversationEpochManager conversation_epochs;
    int32_t conversation_epoch_reader = -1;
    int32_t conversation_epoch_retire = -1;
    std::atomic<uint32_t> conversation_retire_count{0};

    // The model executor owns its long-lived provider thread.
    platform::Thread thread{};

    // Batch scratch.
    ::AstralBackendBatchToken* batch_tokens = nullptr;
    uint32_t* batch_output_slots = nullptr;

    std::atomic<bool> running{true};
};

AstralErr executor_start(ModelExecutor* ex);
void executor_stop_and_join(ModelExecutor* ex);

} // namespace astral::inference
