#pragma once

#include "../../include/astral_rt.h"

namespace astral::core {

class ModelLoadConfigScope {
public:
    explicit ModelLoadConfigScope(const AstralModelDesc* desc);
    ~ModelLoadConfigScope();

    ModelLoadConfigScope(const ModelLoadConfigScope&) = delete;
    ModelLoadConfigScope& operator=(const ModelLoadConfigScope&) = delete;

private:
    const AstralModelDesc* prev_;
};

const AstralModelDesc* model_load_desc();

} // namespace astral::core
