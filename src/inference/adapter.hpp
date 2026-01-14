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
    uint8_t* path;
    uint32_t path_len;
    std::atomic<uint32_t> refcount; // sessions using this adapter
};

AstralErr adapter_load(Model* model, const AstralAdapterDesc* desc, Adapter** out_adapter);
AstralErr adapter_info(Adapter* a, AstralAdapterInfo* out_info);
AstralErr adapter_path_copy(Adapter* a, AstralMutSpanU8 out_path, uint32_t* out_len);
void adapter_retain(Adapter* a);
void adapter_release(Adapter* a);

} // namespace astral::inference
