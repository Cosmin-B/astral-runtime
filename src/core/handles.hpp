/**
 * handles.hpp - Generation-tagged handle table (v0.1 ABI hardening)
 *
 * Public ABI handles are 64-bit values encoding:
 * - kind (8 bits)
 * - slot index (24 bits)
 * - generation (32 bits)
 *
 * Goals:
 * - Detect stale handles after free (generation mismatch).
 * - Provider-agnostic (stores opaque pointers only).
 * - No dynamic allocation (fixed-size table).
 * - No atomic compare-and-swap usage.
 */

#pragma once

#include "../../include/astral_rt.h"
#include "../platform/atomics.h"

#include <atomic>
#include <cstdint>

namespace astral::core {

enum class HandleKind : uint32_t {
    Invalid = 0,
    Model = 1,
    Session = 2,
    Embedder = 3,
    Adapter = 4,
};

inline constexpr uint32_t kMaxHandles = 8192;

inline constexpr AstralHandle make_handle(HandleKind kind, uint32_t index, uint32_t generation) {
    return (static_cast<uint64_t>(kind) << 56) | (static_cast<uint64_t>(index & 0x00FFFFFFu) << 32) |
           static_cast<uint64_t>(generation);
}

inline constexpr HandleKind handle_kind(AstralHandle handle) {
    return static_cast<HandleKind>(static_cast<uint32_t>((handle >> 56) & 0xFFu));
}

inline constexpr uint32_t handle_index(AstralHandle handle) {
    return static_cast<uint32_t>((handle >> 32) & 0x00FFFFFFu);
}

inline constexpr uint32_t handle_generation(AstralHandle handle) {
    return static_cast<uint32_t>(handle & 0xFFFFFFFFu);
}

class HandleTable {
public:
    HandleTable() {
        for (uint32_t i = 0; i < kMaxHandles; ++i) {
            slots_[i].ptr.store(nullptr, std::memory_order_relaxed);
            slots_[i].generation.store(0, std::memory_order_relaxed);
            slots_[i].kind.store(static_cast<uint32_t>(HandleKind::Invalid), std::memory_order_relaxed);
        }
        next_index_.store(0, std::memory_order_relaxed);
    }

    AstralHandle register_ptr(HandleKind kind, void* ptr) {
        if (ptr == nullptr || kind == HandleKind::Invalid) {
            return 0;
        }

        lock_();

        const uint32_t start = next_index_.load(std::memory_order_relaxed);
        for (uint32_t i = 0; i < kMaxHandles; ++i) {
            const uint32_t index = (start + i) % kMaxHandles;
            Slot& slot = slots_[index];

            if (slot.ptr.load(std::memory_order_relaxed) != nullptr) {
                continue;
            }

            uint32_t gen = slot.generation.load(std::memory_order_relaxed) + 1u;
            if (gen == 0) {
                gen = 1;
            }

            slot.generation.store(gen, std::memory_order_relaxed);
            slot.kind.store(static_cast<uint32_t>(kind), std::memory_order_relaxed);
            slot.ptr.store(ptr, std::memory_order_release);

            next_index_.store((index + 1u) % kMaxHandles, std::memory_order_relaxed);

            unlock_();
            return make_handle(kind, index, gen);
        }

        unlock_();
        return 0;
    }

    void unregister_handle(AstralHandle handle, HandleKind expected_kind) {
        if (handle == 0 || expected_kind == HandleKind::Invalid) {
            return;
        }

        const uint32_t index = handle_index(handle);
        if (index >= kMaxHandles) {
            return;
        }

        const uint32_t gen = handle_generation(handle);
        if (gen == 0) {
            return;
        }

        lock_();

        Slot& slot = slots_[index];
        void* ptr = slot.ptr.load(std::memory_order_relaxed);
        if (ptr == nullptr) {
            unlock_();
            return;
        }

        const uint32_t slot_kind = slot.kind.load(std::memory_order_relaxed);
        const uint32_t slot_gen = slot.generation.load(std::memory_order_relaxed);
        if (slot_kind != static_cast<uint32_t>(expected_kind) || slot_gen != gen) {
            unlock_();
            return;
        }

        slot.ptr.store(nullptr, std::memory_order_release);
        slot.kind.store(static_cast<uint32_t>(HandleKind::Invalid), std::memory_order_relaxed);

        unlock_();
    }

    void* lookup(AstralHandle handle, HandleKind expected_kind) const {
        if (handle == 0 || expected_kind == HandleKind::Invalid) {
            return nullptr;
        }

        const HandleKind kind = handle_kind(handle);
        if (kind != expected_kind) {
            return nullptr;
        }

        const uint32_t index = handle_index(handle);
        if (index >= kMaxHandles) {
            return nullptr;
        }

        const uint32_t gen = handle_generation(handle);
        if (gen == 0) {
            return nullptr;
        }

        const Slot& slot = slots_[index];
        void* ptr = slot.ptr.load(std::memory_order_acquire);
        if (ptr == nullptr) {
            return nullptr;
        }

        const uint32_t slot_kind = slot.kind.load(std::memory_order_relaxed);
        const uint32_t slot_gen = slot.generation.load(std::memory_order_relaxed);
        if (slot_kind != static_cast<uint32_t>(expected_kind) || slot_gen != gen) {
            return nullptr;
        }

        return ptr;
    }

    bool is_valid(AstralHandle handle) const {
        const HandleKind kind = handle_kind(handle);
        if (kind == HandleKind::Invalid) {
            return false;
        }
        return lookup(handle, kind) != nullptr;
    }

private:
    struct Slot {
        std::atomic<void*> ptr;
        std::atomic<uint32_t> generation;
        std::atomic<uint32_t> kind;
    };

    Slot slots_[kMaxHandles];
    std::atomic<uint32_t> next_index_;

    mutable std::atomic_flag table_lock_ = ATOMIC_FLAG_INIT;

    void lock_() const {
        uint32_t spins = 0;
        while (table_lock_.test_and_set(std::memory_order_acquire)) {
            if (spins < 64) {
                astral::platform::cpu_pause();
            } else {
                astral::platform::cpu_wait_for_event();
            }
            if (spins < 1024) {
                ++spins;
            }
        }
    }

    void unlock_() const {
        table_lock_.clear(std::memory_order_release);
        astral::platform::cpu_signal_event();
    }
};

// Global handle table.
inline HandleTable& handle_table() {
    static HandleTable table;
    return table;
}

inline AstralHandle register_handle(HandleKind kind, void* ptr) {
    return handle_table().register_ptr(kind, ptr);
}

inline void unregister_handle(AstralHandle handle, HandleKind expected_kind) {
    handle_table().unregister_handle(handle, expected_kind);
}

inline void* lookup_handle(AstralHandle handle, HandleKind expected_kind) {
    return handle_table().lookup(handle, expected_kind);
}

inline bool handle_valid(AstralHandle handle) {
    return handle_table().is_valid(handle);
}

} // namespace astral::core
