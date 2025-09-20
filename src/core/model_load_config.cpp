#include "model_load_config.hpp"

namespace {

thread_local const AstralModelDesc* g_model_desc = nullptr;

}

namespace astral::core {

ModelLoadConfigScope::ModelLoadConfigScope(const AstralModelDesc* desc)
    : prev_(g_model_desc) {
    g_model_desc = desc;
}

ModelLoadConfigScope::~ModelLoadConfigScope() {
    g_model_desc = prev_;
}

const AstralModelDesc* model_load_desc() {
    return g_model_desc;
}

} // namespace astral::core
