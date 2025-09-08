#pragma once

#include "../../include/astral_rt.h"
#include "model.hpp"

#include <atomic>
#include <cstdint>

namespace astral::inference {

struct Adapter {
    AstralHandle handle;
    Model* model; // strong ref held while adapter exists
    void* backend_adapter_ctx;
    std::atomic<uint32_t> refcount; // sessions using this adapter
};

AstralErr adapter_load(Model* model, const AstralAdapterDesc* desc, Adapter** out_adapter);
void adapter_retain(Adapter* a);
void adapter_release(Adapter* a);

} // namespace astral::inference

