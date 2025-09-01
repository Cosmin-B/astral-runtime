/**
 * plugin_loader.cpp - Runtime-loadable backend providers (v0)
 *
 * This implements `astral_backend_load_plugin()` which loads a shared library,
 * queries its provider descriptor, and registers it with the backend registry.
 *
 * The plugin contract lives in `astral/include/astral_backend_plugin.h`.
 */

#include "../../include/astral_backend_plugin.h"

#include "backend.hpp"
#include "../core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#if defined(_WIN32) || defined(_WIN64)
  #define NOMINMAX
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

namespace astral::backend {

namespace {

constexpr size_t kMaxPluginLibraries = 8;

struct PluginLibrarySlot {
    void* dylib_handle;
};

static PluginLibrarySlot g_plugin_libs[kMaxPluginLibraries]{};
static size_t g_plugin_lib_count = 0;

static AstralErr load_plugin_library(const char* path,
                                     void** out_handle,
                                     AstralBackendPluginGetProviderV0Fn* out_fn) {
    if (out_handle == nullptr || out_fn == nullptr) {
        return ASTRAL_E_INVALID;
    }

#if defined(_WIN32) || defined(_WIN64)
    HMODULE mod = LoadLibraryA(path);
    if (mod == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    FARPROC sym = GetProcAddress(mod, "astral_backend_plugin_provider_v0");
    if (sym == nullptr) {
        FreeLibrary(mod);
        return ASTRAL_E_BACKEND;
    }

    *out_handle = reinterpret_cast<void*>(mod);
    *out_fn = reinterpret_cast<AstralBackendPluginGetProviderV0Fn>(sym);
    return ASTRAL_OK;
#else
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    void* sym = dlsym(handle, "astral_backend_plugin_provider_v0");
    if (sym == nullptr) {
        dlclose(handle);
        return ASTRAL_E_BACKEND;
    }

    *out_handle = handle;
    *out_fn = reinterpret_cast<AstralBackendPluginGetProviderV0Fn>(sym);
    return ASTRAL_OK;
#endif
}

static void unload_plugin_library(void* handle) {
#if defined(_WIN32) || defined(_WIN64)
    if (handle) {
        FreeLibrary(reinterpret_cast<HMODULE>(handle));
    }
#else
    if (handle) {
        dlclose(handle);
    }
#endif
}

static void* span_to_cstr(AstralSpanU8 s) {
    if (s.data == nullptr || s.len == 0) {
        return nullptr;
    }

    const size_t n = static_cast<size_t>(s.len);
    char* p = new (std::nothrow) char[n + 1];
    if (p == nullptr) {
        return nullptr;
    }
    std::memcpy(p, s.data, n);
    p[n] = '\0';
    return p;
}

static void set_err_invalid(const char* what) {
    astral::core::set_last_errorf("Invalid parameter: %s", what ? what : "");
}

static void set_err_backend(const char* what) {
    astral::core::set_last_errorf("Backend error: %s", what ? what : "");
}

static bool backend_ops_is_valid(const AstralBackendOps* ops) {
    if (ops == nullptr) {
        return false;
    }
    return ops->model_load != nullptr &&
           ops->model_unload != nullptr &&
           ops->tokenize != nullptr &&
           ops->detokenize != nullptr &&
           ops->model_info != nullptr &&
           ops->session_create != nullptr &&
           ops->session_destroy != nullptr &&
           ops->session_feed != nullptr &&
           ops->session_logits != nullptr &&
           ops->session_accept != nullptr;
}

static AstralErr register_plugin_provider(const AstralBackendProvider* plugin_provider, void* dylib_handle) {
    if (plugin_provider == nullptr || plugin_provider->name == nullptr || !backend_ops_is_valid(plugin_provider->ops)) {
        return ASTRAL_E_INVALID;
    }

    if (dylib_handle == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (g_plugin_lib_count >= kMaxPluginLibraries) {
        return ASTRAL_E_BUSY;
    }

    // Disallow name collisions with existing backends.
    if (BackendRegistry::instance().get_backend(plugin_provider->name) != nullptr) {
        return ASTRAL_E_BUSY;
    }

    const AstralErr reg_err = BackendRegistry::instance().register_backend(plugin_provider);
    if (reg_err != ASTRAL_OK) {
        return reg_err;
    }

    g_plugin_libs[g_plugin_lib_count].dylib_handle = dylib_handle;
    g_plugin_lib_count += 1;
    return ASTRAL_OK;
}

} // namespace

} // namespace astral::backend

extern "C" {

ASTRAL_API AstralErr ASTRAL_CALL astral_backend_load_plugin(AstralSpanU8 path) {
    if (path.data == nullptr || path.len == 0) {
        astral::backend::set_err_invalid("path");
        return ASTRAL_E_INVALID;
    }

    void* cstr = astral::backend::span_to_cstr(path);
    if (cstr == nullptr) {
        astral::core::set_last_error_from_code(ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }

    void* dylib_handle = nullptr;
    AstralBackendPluginGetProviderV0Fn get_provider = nullptr;
    const AstralErr load_err = astral::backend::load_plugin_library(static_cast<const char*>(cstr), &dylib_handle, &get_provider);
    delete[] static_cast<char*>(cstr);

    if (load_err != ASTRAL_OK || dylib_handle == nullptr || get_provider == nullptr) {
        astral::backend::set_err_backend("failed to load plugin or entry point");
        return load_err != ASTRAL_OK ? load_err : ASTRAL_E_BACKEND;
    }

    const AstralBackendProvider* provider = get_provider();
    const AstralErr reg_err = astral::backend::register_plugin_provider(provider, dylib_handle);
    if (reg_err != ASTRAL_OK) {
        astral::backend::unload_plugin_library(dylib_handle);
        if (reg_err == ASTRAL_E_BUSY) {
            astral::backend::set_err_backend("provider name collision or registry full");
        } else {
            astral::backend::set_err_backend("invalid provider");
        }
        return reg_err;
    }

    return ASTRAL_OK;
}

} // extern "C"
