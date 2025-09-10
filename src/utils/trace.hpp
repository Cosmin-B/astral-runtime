#pragma once

// Optional tracing/profiling integration.
//
// Design goals:
// - Zero cost when disabled (macros compile to nothing).
// - Coarse zones only (no per-token micro-instrumentation by default).
// - Safe defaults when enabled (on-demand, localhost, no broadcast).

#if defined(ASTRAL_ENABLE_TRACY) && ASTRAL_ENABLE_TRACY
  #include <tracy/Tracy.hpp>

  #define ASTRAL_TRACY_CONCAT2(a, b) a##b
  #define ASTRAL_TRACY_CONCAT(a, b) ASTRAL_TRACY_CONCAT2(a, b)

  // Use ZoneNamed* instead of ZoneScoped* so multiple zones can exist in the same scope
  // without redeclaring Tracy's default `___tracy_scoped_zone` variable.
  #define ASTRAL_ZONE() ZoneNamed(ASTRAL_TRACY_CONCAT(__astral_zone_, __LINE__), true)
  #define ASTRAL_ZONE_N(name) ZoneNamedN(ASTRAL_TRACY_CONCAT(__astral_zone_, __LINE__), name, true)
  #if defined(ASTRAL_ENABLE_TRACY_MICRO) && ASTRAL_ENABLE_TRACY_MICRO
    #define ASTRAL_ZONE_MICRO() ZoneNamed(ASTRAL_TRACY_CONCAT(__astral_zone_micro_, __LINE__), true)
    #define ASTRAL_ZONE_MICRO_N(name) ZoneNamedN(ASTRAL_TRACY_CONCAT(__astral_zone_micro_, __LINE__), name, true)
  #else
    #define ASTRAL_ZONE_MICRO() do { } while (0)
    #define ASTRAL_ZONE_MICRO_N(name) do { (void)sizeof(name); } while (0)
  #endif
  #define ASTRAL_THREAD_NAME(name) tracy::SetThreadName(name)
  #define ASTRAL_PLOT(name, value) TracyPlot(name, value)
  #define ASTRAL_FRAME_MARK(name) FrameMarkNamed(name)
#else
  #define ASTRAL_ZONE() do { } while (0)
  #define ASTRAL_ZONE_N(name) do { (void)sizeof(name); } while (0)
  #define ASTRAL_ZONE_MICRO() do { } while (0)
  #define ASTRAL_ZONE_MICRO_N(name) do { (void)sizeof(name); } while (0)
  #define ASTRAL_THREAD_NAME(name) do { (void)sizeof(name); } while (0)
  #define ASTRAL_PLOT(name, value) do { (void)sizeof(name); (void)(value); } while (0)
  #define ASTRAL_FRAME_MARK(name) do { (void)sizeof(name); } while (0)
#endif
