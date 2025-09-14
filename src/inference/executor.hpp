#pragma once

#include "../backend/backend.hpp"
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
    uint32_t worker_id = 0;

    // Scheduling knobs (atomic so they can be tuned at runtime).
    std::atomic<uint32_t> max_prompt_tokens_per_slot_per_tick{8};
    uint32_t vocab_size = 0;
    uint32_t ctx_size = 0;

    // Full-vocab indices buffer (shared across conversations) for sampler fallback paths.
    uint32_t* indices_buffer = nullptr;

    // Slot registry (atomic pointers; conv_destroy waits for in-flight refs).
    static constexpr uint32_t kMaxSlotsHard = 32;
    std::atomic<Conversation*> slots[kMaxSlotsHard];

    // Work item runs on the runtime worker pool (one model executor occupies one worker).
    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};

    // Batch scratch.
    ::AstralBackendBatchToken* batch_tokens = nullptr;
    uint32_t* batch_output_slots = nullptr;

    std::atomic<bool> running{true};
};

void executor_start(ModelExecutor* ex);
void executor_stop_and_join(ModelExecutor* ex);

} // namespace astral::inference
