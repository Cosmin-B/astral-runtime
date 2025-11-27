#pragma once

/// Lock-free concurrency primitives for Astral runtime.
///
/// This header includes all lock-free data structures and synchronization primitives
/// used throughout the Astral inference runtime.
///
/// Components:
/// - MpmcQueue: Multi-producer multi-consumer bounded queue for work scheduling
/// - MpscRing: Multi-producer single-consumer bounded ring for fan-in
/// - SpscFanIn: Per-producer SPSC lanes for fixed-owner fan-in
/// - SpscRing: Single-producer single-consumer ring for token streaming
/// - EpochManager: Epoch-based memory reclamation for safe deferred deletion
///
/// Design Philosophy:
/// - Zero allocations in hot paths (all structures are fixed-size)
/// - Explicit memory ordering (no seq_cst unless justified)
/// - Cache-line alignment to prevent false sharing (64 bytes)
/// - Power-of-2 sizes for fast modulo operations
/// - ARM weak memory model correctness (validated)
///
/// References:
/// - docs/architecture/CONCURRENCY_MODEL.md
/// - docs/rules/CODING_STANDARDS.md
/// - Custom MPMC design optimized for game engines
/// - Influenced by research on lock-free data structures:
///   * "Simple, Fast, and Practical Non-Blocking..." (Michael & Scott, 1996)
///   * Facebook Folly's ProducerConsumerQueue design patterns

#include "mpmc_queue.hpp"
#include "mpsc_ring.hpp"
#include "spsc_fan_in.hpp"
#include "spsc_ring.hpp"
#include "epoch.hpp"

namespace astral::concurrency {

/// Version information for concurrency primitives.
struct ConcurrencyVersion {
    static constexpr uint32_t kMajor = 0;
    static constexpr uint32_t kMinor = 1;
    static constexpr uint32_t kPatch = 0;
};

} // namespace astral::concurrency
