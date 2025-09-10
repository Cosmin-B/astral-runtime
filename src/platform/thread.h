#pragma once

#include <cstdint>

#if defined(_WIN32)
// Avoid pulling windows.h into every TU; use an opaque handle.
#else
#include <pthread.h>
#endif

namespace astral::platform {

using ThreadFn = void (*)(void* user);

struct Thread {
#if defined(_WIN32)
    void* handle = nullptr;
    uint32_t id = 0;
#else
    pthread_t handle{};
#endif
    ThreadFn fn = nullptr;
    void* user = nullptr;
    bool started = false;
};

uint32_t hardware_concurrency();

// Starts a thread that calls `fn(user)`. Returns false on failure.
bool thread_start(Thread* t, ThreadFn fn, void* user);

// Joins a started thread. Safe to call multiple times.
void thread_join(Thread* t);

} // namespace astral::platform
