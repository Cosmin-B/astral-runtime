/**
 * test_backend_basic.cpp - Basic Backend Integration Test
 *
 * Validates:
 * - Backend registry initialization
 * - CPU backend registration
 * - Backend selection logic
 *
 * This is a compile-time test (no runtime execution required).
 */

#include "backend.hpp"

int main() {
    // Test backend registry singleton
    auto& registry = astral::backend::BackendRegistry::instance();
    
    // Test CPU backend selection (gpu_layers=0)
    const auto* cpu_backend = registry.select_backend(0);
    if (!cpu_backend || !cpu_backend->ops) return 1;
    
    // Test backend name
    if (cpu_backend->name == nullptr) return 2;
    
    // Test get_backend by name
    const auto* cpu_backend_by_name = registry.get_backend("cpu");
    if (cpu_backend_by_name != cpu_backend) return 3;
    
    // Success
    return 0;
}
