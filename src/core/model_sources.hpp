#pragma once

#include "../../include/astral_rt.h"

#include <cstdint>

namespace astral::core {

struct ModelSource {
    AstralModelSourceKind kind;
    AstralSpanU8 bytes; // MEMORY
    AstralModelIO io;   // IO
};

// Registers a model source for consumption by a backend.
//
// Returns ASTRAL_OK and writes `out_token` as a NUL-terminated ASCII string of the form:
//   "astral-src:<id>"
//
// The backend is expected to "take" the source (consumes and unregisters it) during model load.
AstralErr model_source_register(const ModelSource& source, uint64_t* out_id, char* out_token, uint32_t token_cap);

// Takes (consumes) a previously-registered model source by id.
// Returns true if an entry was found and consumed.
bool model_source_take(uint64_t id, ModelSource* out_source);

// Cleanup path when a backend rejects a registered model source.
void model_source_release(uint64_t id);

// Debug/cleanup helper.
bool model_source_present(uint64_t id);

} // namespace astral::core
