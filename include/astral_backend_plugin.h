/**
 * astral_backend_plugin.h - Backend plugin interface (v0)
 *
 * This header defines the ABI contract for out-of-tree backend providers that
 * can be loaded at runtime via `astral_backend_load_plugin()`.
 *
 * Notes:
 * - Provider dispatch overhead remains a single indirect call (ops table lookup
 *   is done once at model load; no per-token provider selection).
 * - The plugin interface is C-compatible and uses the same core ABI types from
 *   `astral_rt.h`.
 */

#pragma once

#include "astral_backend.h"

#if defined(_WIN32) || defined(_WIN64)
  #define ASTRAL_BACKEND_PLUGIN_EXPORT __declspec(dllexport)
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define ASTRAL_BACKEND_PLUGIN_EXPORT __attribute__((visibility("default")))
  #else
    #define ASTRAL_BACKEND_PLUGIN_EXPORT
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Plugin entry point: returns a pointer to a static provider descriptor.
 *
 * Contract:
 * - The returned provider pointer must remain valid for the lifetime of the process.
 * - The plugin shared library must remain loaded while the provider may be used.
 *   Astral keeps the library loaded after successful registration.
 */
typedef const AstralBackendProvider* (ASTRAL_CALL * AstralBackendPluginGetProviderV0Fn)(void);

/**
 * Required exported symbol name for plugins.
 *
 * Example:
 *   ASTRAL_BACKEND_PLUGIN_EXPORT const AstralBackendProvider* ASTRAL_CALL astral_backend_plugin_provider_v0();
 */
ASTRAL_BACKEND_PLUGIN_EXPORT const AstralBackendProvider* ASTRAL_CALL astral_backend_plugin_provider_v0(void);

#ifdef __cplusplus
} // extern "C"
#endif
