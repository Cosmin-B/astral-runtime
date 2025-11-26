#pragma once

#include "../../include/astral_rt.h"

namespace astral::inference {

struct Model;
struct Embedder;

AstralErr embedder_create(Model* model, AstralHandle* out_embedder);
void embedder_destroy(Embedder* embedder);

AstralErr embedder_enqueue(Embedder* embedder, AstralSpanU8 text, uint64_t* out_ticket);
AstralErr embedder_enqueue_image(Embedder* embedder, const AstralImageDesc* image, uint64_t* out_ticket);
AstralErr embedder_enqueue_audio(Embedder* embedder, const AstralAudioDesc* audio, uint64_t* out_ticket);
AstralErr embedder_enqueue_multimodal(Embedder* embedder,
                                      AstralSpanU8 text,
                                      const AstralImageDesc* image,
                                      const AstralAudioDesc* audio,
                                      uint64_t* out_ticket);
AstralErr embedder_cancel(Embedder* embedder, uint64_t ticket);
AstralErr embedder_collect(Embedder* embedder, uint64_t ticket, AstralMutSpanU8 out_vector);

} // namespace astral::inference
