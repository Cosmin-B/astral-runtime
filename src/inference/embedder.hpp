#pragma once

#include "../../include/astral_rt.h"

namespace astral::inference {

struct Model;
struct Embedder;

AstralErr embedder_create(Model* model, AstralHandle* out_embedder);
void embedder_destroy(Embedder* embedder);

AstralErr embedder_enqueue(Embedder* embedder, AstralSpanU8 text, uint64_t* out_ticket);
AstralErr embedder_collect(Embedder* embedder, uint64_t ticket, AstralMutSpanU8 out_vector);

} // namespace astral::inference

