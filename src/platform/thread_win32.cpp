#if defined(_WIN32)

#include "thread.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace astral::platform {

static DWORD WINAPI thread_entry(LPVOID param) {
    Thread* t = static_cast<Thread*>(param);
    if (t && t->fn) {
        t->fn(t->user);
    }
    return 0;
}

uint32_t hardware_concurrency() {
    const DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n > 0) {
        return static_cast<uint32_t>(n);
    }
    return 0;
}

bool thread_start(Thread* t, ThreadFn fn, void* user) {
    if (!t || !fn) {
        return false;
    }

    t->fn = fn;
    t->user = user;
    t->started = false;
    t->id = 0;

    HANDLE h = CreateThread(nullptr, 0, &thread_entry, t, 0, reinterpret_cast<DWORD*>(&t->id));
    if (!h) {
        return false;
    }

    t->handle = h;
    t->started = true;
    return true;
}

void thread_join(Thread* t) {
    if (!t || !t->started || !t->handle) {
        return;
    }

    (void)WaitForSingleObject(t->handle, INFINITE);
    (void)CloseHandle(t->handle);
    t->handle = nullptr;
    t->started = false;
}

} // namespace astral::platform

#endif

